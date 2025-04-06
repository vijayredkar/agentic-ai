from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.models.huggingface import HuggingFace
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

agent = Agent(
    model=HuggingFace( id="Qwen/Qwen2.5-0.5B-Instruct"),
    tools=[YFinanceTools()],
    show_tool_calls=True,
    markdown=True,
    tool_choice="auto",
)
agent.print_response("Provide TSLA stock status")