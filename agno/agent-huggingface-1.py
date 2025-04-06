from agno.agent import Agent
from agno.models.huggingface import HuggingFace

agent = Agent(
    model=HuggingFace(
        id="microsoft/Phi-3-mini-4k-instruct", max_tokens=4096, temperature=0
    ),
)
agent.print_response(
    "What is meaning of life and then recommend 5 best books to read about it"
)