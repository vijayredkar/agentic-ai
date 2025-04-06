import asyncio
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.models.huggingface import HuggingFace
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.team.team import Team


agent1 = Agent(
    model=HuggingFace( id="Qwen/Qwen2.5-0.5B-Instruct"),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True,
    tool_choice="auto",
    debug_mode=True,
)

agent2 = Agent(
    model=HuggingFace( id="Qwen/Qwen2.5-0.5B-Instruct"),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True,
    tool_choice="auto",
    debug_mode=True,
)

agent_team = Team(
    name="Discussion Team",
    mode="collaborate",
    model=HuggingFace( id="Qwen/Qwen2.5-0.5B-Instruct"),
    members=[
        agent1,
        agent2,
    ],
    instructions=[
        "You are a discussion master.",
        "You have to stop the discussion when you think the team has reached a consensus.",
    ],
    success_criteria="The team has reached a consensus.",
    
)

#agent_team.print_response(message="Start the discussion on the topic: 'What is the best way to learn to code?'",)
#agent1.print_response(message="Start the discussion on the topic: 'What is the best way to learn to code?'",)
agent1.print_response("What is the best way to learn to code?")