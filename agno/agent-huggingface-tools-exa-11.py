from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.models.huggingface import HuggingFace
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.hackernews import HackerNewsTools
from agno.tools.exa import ExaTools
import os

os.environ["EXA_API_KEY"]='DUMMY'
agent = Agent(
    model=HuggingFace( id="Qwen/Qwen2.5-0.5B-Instruct"),
    tools=[ExaTools()],
    #show_tool_calls=True,
    #markdown=True,
    tool_choice="auto",
    debug_mode=True,
)
agent.print_response("Provide ony string response. Recommend 2 sci-fi book titles")