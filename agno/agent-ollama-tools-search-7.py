from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.duckduckgo import DuckDuckGoTools

agent = Agent(
    model=Ollama(id="qwen2.5"),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True,
    tool_choice="auto",
)
agent.print_response("Whats happening in France?")