import asyncio
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.models.huggingface import HuggingFace
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.team.team import Team
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.agent.sqlite import SqliteAgentStorage
from rich.pretty import pprint

agent = Agent(
    model=HuggingFace( id="Qwen/Qwen2.5-0.5B-Instruct"),
    description="You are an AI with a memory like an elephant!",
    storage=SqliteAgentStorage(table_name="agent_sessions", db_file="tmp/agent_storage.db"),
    add_history_to_messages=True,
    num_history_responses=3,
    session_id="my_chat_session",
    markdown=True,
	#debug_mode=True
)

agent.print_response("I love spicy Biryani. What is your favorite cuisine?")
agent.print_response("I love Thai more than Biryani.")
pprint([m.model_dump(include={"role", "content"}) for m in agent.memory.messages])

agent.print_response("What did I just say I love?")
agent.print_response("What did I just say I love more?")
pprint([m.model_dump(include={"role", "content"}) for m in agent.memory.messages])