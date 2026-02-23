import os


from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from couchbase.auth import PasswordAuthenticator


cluster = None

@tool
def fetch_key_value_from_couchbase(key : str) -> str:
    """Connects with the local couchbase installation and
    gets the document from the lookups bucket. """
    

    if cluster is None: 
        auth = PasswordAuthenticator("Administrator", "admin123")
        cluster = Cluster('couchbase://localhost', ClusterOptions(auth))
        bucket = cluster.bucket('lookups')
    
    try: 
        result = bucket.default_collection().get(key)
    except Exception as e: 
        return f"Error fetching key {key} from couchbase: {e}"
    
    if result:
        return result.content_as_dict()
    else: 
        return "Key not found"

TOOLS = [fetch_key_value_from_couchbase]

def create_ollama_agent(): 

    llm = ChatOllama(
        model="minimax-m2.5:cloud",
        base_url="http://localhost:11434"
    ).bind_tools(TOOLS)

    return create_react_agent(llm, prompt={"You are an helpful AI agent. You need to use tool calling to fetch documents from database if prompted"})


if __name__ == "__main__":
    agent = create_ollama_agent()
    user_input = input("Your query: ").split()
    message = HumanMessage(content=user_input)
    result = agent.invoke({"messages": [message]})
    print(result)