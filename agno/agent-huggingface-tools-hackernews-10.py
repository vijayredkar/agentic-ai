from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.models.huggingface import HuggingFace
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.hackernews import HackerNewsTools

agent = Agent(
    model=HuggingFace( id="Qwen/Qwen2.5-0.5B-Instruct"),
    tools=[HackerNewsTools()],
    #show_tool_calls=True,
    #markdown=True,
    tool_choice="auto",
)
#agent.print_response("In 5 words only, provide key topics published")
agent.print_response("In 5 words only, provide key topics published other than Cats, Dogs, Toys, Video Games, Sports")
#agent.print_response("In 10 tokens only, summarize the latest post")
#agent.print_response("summarize the latest post")