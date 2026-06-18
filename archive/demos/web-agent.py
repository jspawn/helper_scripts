from openai import OpenAI
import json
from duckduckgo_search import DDGS

# Connect to your local llama-server
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-no-key-needed")

# ---------------------------------------------------------------------------
# 1. Define the actual Python function that does the work
# ---------------------------------------------------------------------------
def search_web(query):
    print(f"\n[🔧 TOOL EXECUTION] Searching the web for: '{query}'...")
    try:
        # Fetch the top 3 results from DuckDuckGo
        results = DDGS().text(query, max_results=3)
        if not results:
            return "No results found."
        
        # Format the results into a string the LLM can easily read
        formatted_results = ""
        for i, r in enumerate(results, 1):
            formatted_results += f"Result {i}:\nTitle: {r['title']}\nSnippet: {r['body']}\nLink: {r['href']}\n\n"
        
        return formatted_results
    except Exception as e:
        return f"Search failed with error: {str(e)}"

# Map the tool name to the Python function
available_tools = {
    "search_web": search_web
}

# ---------------------------------------------------------------------------
# 2. Tell the LLM how to use the tool
# ---------------------------------------------------------------------------
tools_schema = [{
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the internet for current events, news, or factual information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on DuckDuckGo."
                }
            },
            "required": ["query"]
        }
    }
}]

# ---------------------------------------------------------------------------
# 3. The Agent Loop
# ---------------------------------------------------------------------------
# Give the model a personality and a task
messages = [
    {"role": "system", "content": "You are a helpful research assistant. Always use the search_web tool if you are asked about current events or need to verify facts."},
    {"role": "user", "content": "Who won the most recent Super Bowl and what was the final score?"}
]

print("[🤖 AGENT] Thinking about how to answer...")

# First API call: The model reads the prompt and decides to use a tool
response = client.chat.completions.create(
    model="local-model", 
    messages=messages, 
    tools=tools_schema, 
    temperature=0.2
)

response_message = response.choices[0].message
messages.append(response_message) # Save the tool request to chat history

# Check if the model asked to use a tool
if response_message.tool_calls:
    for tool_call in response_message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        # Look up the function in our dictionary and run it
        function_to_call = available_tools[function_name]
        search_results = function_to_call(**function_args)
        
        print("\n[🌍 SEARCH RESULTS GATHERED]")
        
        # Append the raw search results back into the chat history
        messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": search_results,
        })
    
    print("\n[🤖 AGENT] Reading search results and writing final answer...")
    
    # Second API call: The model reads the search results and formulates the final answer
    final_response = client.chat.completions.create(
        model="local-model", 
        messages=messages,
        temperature=0.2
    )
    
    print("\n================ FINAL ANSWER ================\n")
    print(final_response.choices[0].message.content)
    print("\n==============================================")

else:
    # If the model thought it already knew the answer and didn't use the tool
    print("\n================ FINAL ANSWER ================\n")
    print(response_message.content)
    print("\n==============================================")
