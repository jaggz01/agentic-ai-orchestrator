"""
Agentic AI Orchestrator — Tool-Calling Agent via Local Ollama
=============================================================
Uses LangChain's `create_react_agent` (dev-agents style) with a local
Ollama model that supports function/tool calling (e.g. llama3.2, mistral-nemo,
qwen2.5, or any model with `tools` support listed by `ollama list`).

Prerequisites
-------------
pip install langchain langchain-ollama langchain-community

Ensure Ollama is running:
  ollama serve

Pull a tool-capable model (choose one):
  ollama pull llama3.2
  ollama pull qwen2.5
  ollama pull mistral-nemo

Run:
  python ollama_agent.py
"""

import math
import datetime
import json
from typing import Annotated

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "llama3.2"   # Change to any Ollama model with tool-calling support
OLLAMA_BASE_URL = "http://localhost:11434"

# ---------------------------------------------------------------------------
# Tool Definitions  (decorated with @tool so LangChain auto-generates schema)
# ---------------------------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Supports basic arithmetic (+, -, *, /), exponentiation (**),
    square root (sqrt), floor, ceil, pi, e.

    Args:
        expression: A math expression string, e.g. "sqrt(144) + 2 ** 8"

    Returns:
        The numeric result as a string.
    """
    allowed_names = {
        "sqrt": math.sqrt,
        "floor": math.floor,
        "ceil": math.ceil,
        "abs": abs,
        "round": round,
        "pi": math.pi,
        "e": math.e,
        "log": math.log,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


@tool
def get_current_datetime(timezone: str = "local") -> str:
    """
    Return the current date and time.

    Args:
        timezone: Pass "local" for the system local time, or "utc" for UTC.

    Returns:
        The current date/time as an ISO-8601 string.
    """
    now = datetime.datetime.utcnow() if timezone.lower() == "utc" else datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S") + (f" ({timezone})" if timezone else "")


@tool
def word_counter(text: str) -> str:
    """
    Count the number of words, characters, and sentences in a given text.

    Args:
        text: The text to analyse.

    Returns:
        A JSON string with word_count, char_count, and sentence_count.
    """
    words = text.split()
    sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    return json.dumps({
        "word_count": len(words),
        "char_count": len(text),
        "sentence_count": len(sentences),
    }, indent=2)


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert a value between common units.

    Supported conversions:
      - Temperature : celsius  ↔ fahrenheit ↔ kelvin
      - Length      : meters   ↔ feet ↔ inches ↔ kilometers ↔ miles
      - Weight      : kg       ↔ pounds ↔ grams ↔ ounces

    Args:
        value     : Numeric value to convert.
        from_unit : Source unit (e.g. "celsius", "meters", "kg").
        to_unit   : Target unit (e.g. "fahrenheit", "feet", "pounds").

    Returns:
        The converted value with units.
    """
    from_unit = from_unit.lower().strip()
    to_unit   = to_unit.lower().strip()

    # --- Temperature ---
    temp_to_celsius = {
        "celsius": lambda v: v,
        "fahrenheit": lambda v: (v - 32) * 5 / 9,
        "kelvin": lambda v: v - 273.15,
    }
    celsius_to_target = {
        "celsius": lambda c: c,
        "fahrenheit": lambda c: c * 9 / 5 + 32,
        "kelvin": lambda c: c + 273.15,
    }
    if from_unit in temp_to_celsius and to_unit in celsius_to_target:
        result = celsius_to_target[to_unit](temp_to_celsius[from_unit](value))
        return f"{round(result, 4)} {to_unit}"

    # --- Length (all convert via meters) ---
    to_meters = {
        "meters": 1, "feet": 0.3048, "inches": 0.0254,
        "kilometers": 1000, "miles": 1609.344,
    }
    if from_unit in to_meters and to_unit in to_meters:
        result = value * to_meters[from_unit] / to_meters[to_unit]
        return f"{round(result, 6)} {to_unit}"

    # --- Weight (all convert via kg) ---
    to_kg = {
        "kg": 1, "pounds": 0.453592, "grams": 0.001, "ounces": 0.0283495,
    }
    if from_unit in to_kg and to_unit in to_kg:
        result = value * to_kg[from_unit] / to_kg[to_unit]
        return f"{round(result, 6)} {to_unit}"

    return f"Unsupported conversion: {from_unit} → {to_unit}"


# ---------------------------------------------------------------------------
# All registered tools
# ---------------------------------------------------------------------------

TOOLS = [calculator, get_current_datetime, word_counter, unit_converter]


# ---------------------------------------------------------------------------
# Agent Factory
# ---------------------------------------------------------------------------

def build_agent() -> AgentExecutor:
    """
    Build a LangChain ReAct agent backed by a local Ollama model.

    The agent uses `bind_tools` so the model can emit structured tool-call
    requests, which `AgentExecutor` intercepts and routes to the right function.
    """
    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0,          # deterministic for tool-use tasks
    )

    # Bind tools so the LLM knows their signatures
    llm_with_tools = llm.bind_tools(TOOLS)

    # ReAct-style prompt expected by create_react_agent
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are a helpful AI assistant with access to the following tools:\n\n"
                "{tools}\n\n"
                "Use the following format:\n\n"
                "Question: the input question you must answer\n"
                "Thought: you should always think about what to do\n"
                "Action: the action to take, should be one of [{tool_names}]\n"
                "Action Input: the input to the action\n"
                "Observation: the result of the action\n"
                "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
                "Thought: I now know the final answer\n"
                "Final Answer: the final answer to the original input question"
            ),
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_react_agent(llm=llm_with_tools, tools=TOOLS, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,           # prints thought/action/observation loop
        handle_parsing_errors=True,
        max_iterations=8,
    )


# ---------------------------------------------------------------------------
# Interactive Chat Loop
# ---------------------------------------------------------------------------

def run_chat_loop(agent_executor: AgentExecutor) -> None:
    """Simple REPL that feeds user messages into the agent."""
    chat_history: list = []

    print("\n" + "=" * 60)
    print(f"  Ollama Tool-Calling Agent  (model: {MODEL_NAME})")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 60 + "\n")

    print("Available tools:")
    for t in TOOLS:
        print(f"  • {t.name}: {t.description.splitlines()[0]}")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        if not user_input:
            continue

        try:
            response = agent_executor.invoke(
                {
                    "input": user_input,
                    "chat_history": chat_history,
                }
            )
            answer = response.get("output", "")
            print(f"\nAgent: {answer}\n")

            # Maintain conversation history
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))

        except Exception as exc:
            print(f"\n[Error] {exc}\n")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent_executor = build_agent()
    run_chat_loop(agent_executor)
