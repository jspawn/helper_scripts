from smolagents import CodeAgent, DuckDuckGoSearchTool, OpenAIServerModel

# 1. Connect to your local llama-server
model = OpenAIServerModel(
    model_id="local-model",
    api_base="http://127.0.0.1:8080/v1",
    api_key="sk-not-needed"
)

# 2. Initialize the built-in web search tool
search_tool = DuckDuckGoSearchTool()

# 3. Create a CodeAgent
# Qwen-Coder excels at CodeAgents, which write small Python scripts 
# to execute tools dynamically instead of just outputting JSON.
agent = CodeAgent(tools=[search_tool], model=model)

# 4. Run the loop!
print("[🤖] Waking up the agent...\n")
agent.run("Who won the most recent Super Bowl and what was the final score?")
