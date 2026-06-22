from openai import OpenAI
import json

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-no-key-needed")

# 1. Actually define the Python function for your tool
def get_weather(location):
    # In a real app, you'd call an API like OpenWeatherMap here.
    # For now, we'll mock it.
    print(f"  [SYSTEM] Running 'get_weather' for {location}...")
    if "Zurich" in location:
        return "12°C and cloudy."
    return "Sunny and 25°C."

# Map the name the model outputs to your actual Python function
available_tools = {
    "get_weather": get_weather
}

tools_schema = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a specific location.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    }
}]

messages = [
    {"role": "system", "content": "You are a helpful assistant. Use tools if necessary."},
    {"role": "user", "content": "What should I wear in Zurich today based on the weather?"}
]

print("Thinking...")
response = client.chat.completions.create(
    model="local-model", messages=messages, tools=tools_schema, temperature=0.1
)
response_message = response.choices[0].message
messages.append(response_message) # Add the model's tool request to the history

# 2. THE LOOP: Check for tool calls, run them, and feed data back
if response_message.tool_calls:
    for tool_call in response_message.tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_tools[function_name]
        function_args = json.loads(tool_call.function.arguments)
        
        # Execute the actual Python function
        function_response = function_to_call(**function_args)
        
        # Feed the result back into the chat history
        messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": function_response,
        })
    
    print("Reading tool results and generating final answer...")
    # 3. Call the model a SECOND time with the new context
    second_response = client.chat.completions.create(
        model="local-model", messages=messages
    )
    print(f"\nFinal Answer: {second_response.choices[0].message.content}")
else:
    print(f"\nFinal Answer: {response_message.content}")
