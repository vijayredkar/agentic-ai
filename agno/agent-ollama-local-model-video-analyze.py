from textwrap import dedent
from agno.agent import Agent, RunResponse
from agno.models.ollama import Ollama
from agno.tools.youtube import YouTubeTools

agent = Agent(
    model=Ollama(id="qwen2.5"),

tools=[YouTubeTools()],
debug_mode=True,

instructions=dedent("""\
        You are a seasoned YouTube content analyst
    """)
)


#test with generic question
#agent.print_response("Summarize this video: https://www.youtube.com/watch?v=ILRxOV4V4Cs", max_new_tokens=128)
#agent.print_response("What is the main topic discussed in this video: https://www.youtube.com/watch?v=ILRxOV4V4Cs", max_new_tokens=128)
#test with specific to the point question
agent.print_response("Name the vector databases being discussed in this video.Just provide the names only: https://www.youtube.com/watch?v=ILRxOV4V4Cs", max_new_tokens=128)