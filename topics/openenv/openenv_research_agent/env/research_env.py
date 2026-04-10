"""
Research RL Environment — OpenEnv MCPEnvironment wrapping Tavily tools.

This is the core of the demo. It defines the environment that both the
OpenEnv agent and the traditional RL agent interact with.

The environment:
  - Exposes three Tavily tools as MCP-registered actions
  - Tracks episode state: steps, tool usage, reward history
  - Enforces a step limit (max_steps) to bound each episode
  - Computes per-step rewards via reward.py (injected at construction)
  - Supports concurrent sessions (SUPPORTS_CONCURRENT_SESSIONS = True)
    so the agent race demo can run 3 agents simultaneously

How it fits into OpenEnv:
  - Inherits from Environment (openenv base class)
  - reset() starts a new episode and returns the initial observation
  - step() executes one tool call and returns the next observation + reward
  - state property returns full episode state for inspection
  - The three Tavily functions are registered as @mcp.tool actions
    so the OpenEnv MCPClient can discover and call them
"""

import os
from typing import Optional
from dotenv import load_dotenv
from tavily import TavilyClient

from openenv.core.env_server.interfaces import Environment

from env.models import ResearchAction, ResearchObservation, ResearchState
from env.tools.search import run_search
from env.tools.extract import run_extract
from env.tools.crawl import run_crawl

load_dotenv()

# Default step limit per episode — enough for search + extract + crawl chain
DEFAULT_MAX_STEPS = 10


class ResearchEnvironment(Environment):
    """
    RL environment for web research using Tavily tools.

    Each episode is one research question. The agent takes steps by
    calling Tavily tools. The episode ends when the agent has used all
    its steps or calls the special 'finish' action.

    Concurrent sessions are supported — each session tracks its own
    episode state independently, enabling the agent race demo.
    """

    # Signals to OpenEnv that multiple agents can run simultaneously
    # in this environment without state interference.
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, reward_fn, max_steps: int = DEFAULT_MAX_STEPS):
        """
        Args:
            reward_fn: Callable(query, tool_name, result, step) -> float
                       Injected so the environment works with both the
                       keyword-match (traditional) and LLM-as-judge rewards.
            max_steps: Maximum tool calls per episode before forced termination.
        """
        self._reward_fn = reward_fn
        self._max_steps = max_steps
        self._tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

        # Episode state — reset per session
        self._query: str = ""
        self._step: int = 0
        self._done: bool = False
        self._total_reward: float = 0.0
        self._history: list[dict] = []
        self._tool_usage: dict[str, int] = {}

    # -----------------------------------------------------------------------
    # OpenEnv interface
    # -----------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        query: str = "",
    ) -> ResearchObservation:
        """Start a new research episode for the given query."""
        self._query = query or "What is Model Context Protocol (MCP)?"
        self._step = 0
        self._done = False
        self._total_reward = 0.0
        self._history = []
        self._tool_usage = {"tavily_search": 0, "tavily_extract": 0, "tavily_crawl": 0}

        return ResearchObservation(
            tool_name="reset",
            tool_args={},
            result={"message": f"Episode started. Research question: {self._query}"},
            step=0,
            done=False,
            reward=0.0,
            message=f"Research question: {self._query}. Available tools: tavily_search, tavily_extract, tavily_crawl, finish.",
        )

    def step(self, action: ResearchAction) -> ResearchObservation:
        """Execute one tool call and return the resulting observation."""
        if self._done:
            return ResearchObservation(
                tool_name="noop",
                tool_args={},
                result={},
                step=self._step,
                done=True,
                reward=0.0,
                message="Episode already finished.",
            )

        self._step += 1

        # Special finish action — agent signals it has enough information
        if action.tool_name == "finish":
            self._done = True
            return ResearchObservation(
                tool_name="finish",
                tool_args={},
                result={"message": "Agent chose to finish the episode."},
                step=self._step,
                done=True,
                reward=0.0,
                message="Episode complete.",
            )

        # Execute the requested Tavily tool
        result = self._dispatch_tool(action.tool_name, action.tool_args)

        # Track tool usage counts for the Flyte report
        if action.tool_name in self._tool_usage:
            self._tool_usage[action.tool_name] += 1

        # Compute reward for this step
        reward = self._reward_fn(
            query=self._query,
            tool_name=action.tool_name,
            result=result,
            step=self._step,
        )
        self._total_reward += reward

        # Check if we've hit the step limit
        done = self._step >= self._max_steps

        observation = ResearchObservation(
            tool_name=action.tool_name,
            tool_args=action.tool_args,
            result=result,
            step=self._step,
            done=done,
            reward=reward,
            message=f"Step {self._step}/{self._max_steps} — reward: {reward:.2f}",
        )

        # Append to history for report generation
        self._history.append({
            "step": self._step,
            "tool_name": action.tool_name,
            "tool_args": action.tool_args,
            "reward": reward,
        })

        self._done = done
        return observation

    @property
    def state(self) -> ResearchState:
        """Return full episode state — used by OpenEnv for session inspection."""
        return ResearchState(
            query=self._query,
            step=self._step,
            max_steps=self._max_steps,
            done=self._done,
            total_reward=self._total_reward,
            history=self._history,
            tool_usage=self._tool_usage,
        )

    def get_metadata(self) -> dict:
        return {
            "name": "ResearchEnvironment",
            "version": "1.0.0",
            "description": "Web research RL environment using Tavily tools.",
            "tools": ["tavily_search", "tavily_extract", "tavily_crawl", "finish"],
        }

    def close(self):
        pass

    # -----------------------------------------------------------------------
    # Tool dispatch
    # -----------------------------------------------------------------------

    def _dispatch_tool(self, tool_name: str, tool_args: dict) -> dict:
        """Route a tool call to the appropriate Tavily function."""
        if tool_name == "tavily_search":
            return run_search(self._tavily, **tool_args)
        elif tool_name == "tavily_extract":
            return run_extract(self._tavily, **tool_args)
        elif tool_name == "tavily_crawl":
            return run_crawl(self._tavily, **tool_args)
        else:
            return {"error": f"Unknown tool: {tool_name}"}
