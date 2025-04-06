import os
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


from agno.agent import Agent
from agno.team.team import Team

os.environ["HUGGINGFACEHUB_API_TOKEN"]='DUMMY'

english_agent = Agent(
    name="English Agent",
    role="You can only answer in English",
    ##model=HuggingFace(id="NousResearch/Hermes-3-Llama-3.1-8B", type="string"),
	model=Ollama(id="qwen2.5"),
    instructions=[
        "You must only respond in English",
    ],
	debug_mode=True,
)

japanese_agent = Agent(
    name="Japanese Agent",
    role="You can only answer in Japanese",
    #model=HuggingFace(id="NousResearch/Hermes-3-Llama-3.1-8B", type="string"),
	model=Ollama(id="qwen2.5"),
    instructions=[
        "You must only respond in Japanese",
    ],
	debug_mode=True,
)
chinese_agent = Agent(
    name="Chinese Agent",
    role="You can only answer in Chinese",
    #model=HuggingFace(id="NousResearch/Hermes-3-Llama-3.1-8B", type="string"),
	model=Ollama(id="qwen2.5"),
    instructions=[
        "You must only respond in Chinese",
    ],
	debug_mode=True,
)
spanish_agent = Agent(
    name="Spanish Agent",
    role="You can only answer in Spanish",
    #model=HuggingFace(id="NousResearch/Hermes-3-Llama-3.1-8B", type="string"),
	model=Ollama(id="qwen2.5"),
    instructions=[
        "You must only respond in Spanish",
    ],
	debug_mode=True,
)

french_agent = Agent(
    name="French Agent",
    role="You can only answer in French",
    #model=HuggingFace(id="NousResearch/Hermes-3-Llama-3.1-8B", type="string"),
	model=Ollama(id="qwen2.5"),
    instructions=[
        "You must only respond in French",
    ],
	debug_mode=True,
)

german_agent = Agent(
    name="German Agent",
    role="You can only answer in German",
    #model=HuggingFace(id="NousResearch/Hermes-3-Llama-3.1-8B", type="string"),
	model=Ollama(id="qwen2.5"),
    instructions=[
        "You must only respond in German",
    ],
	debug_mode=True,
)
multi_language_team = Team(
    name="Multi Language Team",
    mode="route",
    #model=HuggingFace(id="NousResearch/Hermes-3-Llama-3.1-8B", type="string"),
	model=Ollama(id="qwen2.5"),
    members=[
        english_agent,
        spanish_agent,
        japanese_agent,
        french_agent,
        german_agent,
        chinese_agent,
    ],
    show_tool_calls=True,
    markdown=True,
    instructions=[
        "You are a language router that directs questions to the appropriate language agent.",
        "If the user asks in a language whose agent is not a team member, respond in English with:",
        "'I can only answer in the following languages: English, Spanish, Japanese, French and German. Please ask your question in one of these languages.'",
        "Always check the language of the user's input before routing to an agent.",
        "For unsupported languages like Italian, respond in English with the above message.",
    ],
	debug_mode=True,
    show_members_responses=True,
	#debug_mode=true,
)


# Ask "How are you?" in all supported languages
# multi_language_team.print_response(
#     "How are you?", stream=True  # English
# )

# multi_language_team.print_response(
#     "你好吗？", stream=True  # Chinese
# )

# multi_language_team.print_response(
#     "お元気ですか?", stream=True  # Japanese
# )

#multi_language_team.print_response(
#    "Comment allez-vous?",
#)

multi_language_team.print_response(
    "Wie geht es Ihnen?",
)


# multi_language_team.print_response(
#     "Come stai?", stream=True  # Italian
# )