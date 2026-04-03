import os
from dotenv import load_dotenv
import flyte

load_dotenv()

base_env = flyte.TaskEnvironment(
    name="research-pipeline-env",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "langgraph>=1.0.7", "langchain-openai", "tavily-python",
        "markdown", "python-dotenv", "unionai-reuse",
    ),
    secrets=[
        flyte.Secret(key="SAGE_OPENAI_API_KEY", as_env_var="OPENAI_API_KEY"),
        flyte.Secret(key="TAVILY_API_KEY", as_env_var="TAVILY_API_KEY"),
    ],
    resources=flyte.Resources(cpu=2, memory="2Gi"),
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
