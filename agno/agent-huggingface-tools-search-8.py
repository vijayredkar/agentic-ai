from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.models.huggingface import HuggingFace
from agno.tools.duckduckgo import DuckDuckGoTools

agent = Agent(
    model=HuggingFace( id="Qwen/Qwen2.5-0.5B-Instruct"),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True,
    tool_choice="auto",
    debug_mode=True,
)
agent.print_response("Whats happening in France?")