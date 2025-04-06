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

#from lancedb import LanceDB
#from lancedb.embedders import HuggingFaceEmbedder
# Use HuggingFaceEmbedder instead of the default OpenAIEmbedder
#embedder = HuggingFaceEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
#db = LanceDB("path/to/db", embedder=embedder)


from sentence_transformers import SentenceTransformer
#import lancedb

# Initialize Hugging Face embedder
#model = SentenceTransformer('all-MiniLM-L6-v2')



class HuggingFaceEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self._dimensions = self.model.get_sentence_embedding_dimension()

    def embed(self, texts):
        # Generate embeddings for a list of texts
        return self.model.encode(texts, convert_to_numpy=True)

    @property
    def dimensions(self):
        # Provide the dimensions property expected by the framework
        return self._dimensions

    def get_embedding(self, text):
        # Generate embeddings for a single text
        return self.model.encode([text], convert_to_numpy=True)[0]

    def get_embedding_and_usage(self, text):
        # Generate embeddings and track usage statistics
        embedding = self.get_embedding(text)
        usage_stats = {
            "token_count": len(text.split()),  # Count tokens in the input text
            "embedding_dimensions": self.dimensions
        }
        return embedding, usage_stats

  

# Create the custom Hugging Face embedder instance
embedder = HuggingFaceEmbedder()



# LanceDB Vector DB
vector_db = LanceDb(
    table_name="recipes",
    uri="tmp/lancedb",
    search_type=SearchType.hybrid,
	embedder=embedder
)

# Knowledge Base
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=vector_db
)

def lancedb_agent(user: str = "user"):
    agent = Agent(
        model=HuggingFace( id="Qwen/Qwen2.5-0.5B-Instruct"),
        description="You are a Thai cuisine expert with web backup!",
        user_id=user,
        knowledge=knowledge_base,
        tools=[DuckDuckGoTools()],
        instructions=[
            "Search the knowledge base for Thai recipes first.",
            "Use DuckDuckGo if more info is needed."
        ],
        show_tool_calls=True,
        #debug_mode=True,
        markdown=True
    )

    print(f"Session ID: {agent.session_id}\n")

    while True:
        message = Prompt.ask(f"[bold] :sunglasses: {user} [/bold]")
        if message in ("exit", "bye"):
            break
        agent.print_response(message)

if __name__ == "__main__":
    # Load once, comment out after first run
    knowledge_base.load(recreate=True)
    typer.run(lancedb_agent)