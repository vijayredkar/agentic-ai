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

import typer
from rich.prompt import Prompt
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.lancedb import LanceDb
from agno.vectordb.search import SearchType
from agno.tools.duckduckgo import DuckDuckGoTools
from sentence_transformers import SentenceTransformer

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.embedder.openai import OpenAIEmbedder
from agno.storage.agent.sqlite import SqliteAgentStorage

# Import required libraries (assuming they are already installed)
from agno.agent import Agent
from agno.models.huggingface import HuggingFace
from agno.tools.duckduckgo import DuckDuckGoTools

# Define individual agents (Chef and WebResearcher)

# Chef Agent
chef = Agent(
    name="Chef",
    role="Recipe details provider",
    #model=HuggingFace(id="google/flan-t5-small"),  # Lightweight Hugging Face model
	model=Ollama(id="qwen2.5"),
    tools=[DuckDuckGoTools()],
    instructions=["Provide response in simple string format only. Provide detailed recipes for Thai dishes."],  # Chef's specific instructions
	debug_mode=True,
    show_tool_calls=True,
    markdown=True
)

# WebResearcher Agent
researcher = Agent(
    name="WebResearcher",
    role="Web info gatherer",
    #model=HuggingFace(id="google/flan-t5-small"),  # Same model as above for consistency
	tools=[DuckDuckGoTools()],
    instructions=["Provide response in simple string format only. Provide cultural significance of Thai dishes using web search."],  # Researcher's specific instructions
	debug_mode=True,
    show_tool_calls=True,
    markdown=True
)

# Team Leader Agent (coordinates Chef and WebResearcher)
team = Agent(
    team=[chef, researcher],  # Include Chef and Researcher in the team
    #model=HuggingFace(id="google/flan-t5-small"),  # Lightweight and instruction-tuned model
	model=Ollama(id="qwen2.5"),
    name="Discussion Team",
    role="Coordinator",
    description="You coordinate a chef and researcher for Thai food queries.",
    instructions=[
        "Chef provides detailed recipes.",
        "Researcher provides cultural information.",
        "Combine the responses into 10 words."
    ],
	debug_mode=True,
    tools=[DuckDuckGoTools()],
    markdown=True,
    show_tool_calls=True,
)

# Print response from the team
team.print_response("Provide response in simple string format only. In 10 words only, Tell me about Thai chicken soup and its cultural significance.", stream=True)
