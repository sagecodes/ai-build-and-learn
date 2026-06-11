"""A minimal ReAct research agent: LLM + Tavily search, in a LangGraph loop.

Kept deliberately small so the Phoenix trace reads cleanly span by span:

    agent (LLM) -> tools (Tavily) -> agent (LLM) -> ... -> final answer

The OpenInference LangChain instrumentor (wired in workflow.py) traces every
node here automatically, since LangGraph runs on LangChain runnables. We do not
add any tracing code in this file; the agent stays a plain LangGraph app.

The LLM is provider-switchable:
  - openai  ChatOpenAI against the OpenAI API (default)
  - vllm    ChatOpenAI against the in-cluster gemma4 vLLM app (OpenAI-compatible)
"""

import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    VLLM_MODEL_ID,
    VLLM_URL,
)
from tools.search import create_search_tool

log = logging.getLogger(__name__)


def build_llm(provider: str = LLM_PROVIDER):
    """Build the chat model for the chosen provider."""
    if provider == "vllm":
        # The gemma4 vLLM app speaks the OpenAI API; point ChatOpenAI at it.
        # api_key is required by the client but unused by the in-cluster server.
        log.info(f"LLM provider: vllm ({VLLM_MODEL_ID} at {VLLM_URL})")
        return ChatOpenAI(
            model=VLLM_MODEL_ID,
            base_url=VLLM_URL.rstrip("/") + "/v1",
            api_key="not-used",
            temperature=0.0,
        )
    log.info(f"LLM provider: openai ({OPENAI_MODEL})")
    return ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0.0)


def build_agent(tavily_api_key: str, provider: str = LLM_PROVIDER, max_searches: int = 3):
    """Build a ReAct agent that researches a question with Tavily search."""
    web_search = create_search_tool(tavily_api_key)
    tools = [web_search]
    llm = build_llm(provider).bind_tools(tools)

    system_prompt = f"""\
You are a research agent. Answer the user's question by searching the web. \
Use the web_search tool up to {max_searches} times to gather information from \
different angles, then write a clear, well-sourced answer with key findings."""

    async def agent(state: MessagesState) -> MessagesState:
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)

        if getattr(response, "tool_calls", None):
            for tc in response.tool_calls:
                log.info(f"[agent] tool call: {tc['name']}({tc['args']})")
        elif response.content:
            log.info(f"[agent] response: {response.content[:200]}")

        return {"messages": [response]}

    def should_continue(state: MessagesState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "__end__"

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "__end__": "__end__",
    })
    graph.add_edge("tools", "agent")

    return graph.compile()
