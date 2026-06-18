from openai import OpenAI
import json

# Point the client to your local llama-server
client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="sk-no-key-needed"
)

# 1. Define the tools you want the model to know about
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a specific location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g., San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# 2. Ask the model a question that requires the tool
messages = [{"role": "user", "content": "What is the weather like in Zurich right now?"}]

print("Thinking...")
response = client.chat.completions.create(
    model="local-model", # llama-server ignores this, but the library requires it
    messages=messages,
    tools=tools,
    temperature=0.1
)

# 3. Check if the model decided to use the tool
response_message = response.choices[0].message

if response_message.tool_calls:
    for tool_call in response_message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        print(f"\n[AGENT DECISION] The model wants to run a tool!")
        print(f"Function: {function_name}")
        print(f"Arguments: {function_args}")
        
        # Here is where your Python script would ACTUALLY fetch the weather,
        # append the result to the 'messages' array, and call the API again 
        # so the model can read the weather data and write a final answer.
else:
    print(f"\n[RESPONSE] {response_message.content}")
