from textwrap import dedent
from agno.agent import Agent, RunResponse
from agno.models.huggingface import HuggingFace
from agno.tools.yfinance import YFinanceTools
from agno.models.huggingface import HuggingFace
from agno.models.ollama import Ollama
from agno.tools.youtube import YouTubeTools

agent = Agent(
    model=HuggingFace(
        id="Qwen/Qwen2.5-0.5B-Instruct",
    ),

tools=[YouTubeTools()],


instructions=dedent("""\
        You are a seasoned YouTube content analyst
    """)
)

# Print the response on the terminal
agent.print_response("In 4 words only, analyze this video: https://www.youtube.com/watch?v=zjkBMFhNj_g", max_new_tokens=128)