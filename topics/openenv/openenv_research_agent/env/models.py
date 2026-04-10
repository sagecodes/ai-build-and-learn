"""
Pydantic models for the Research RL Environment.

Defines the Action, Observation, and State types that flow between
the agent and the OpenEnv environment. These types form the contract
between the two sides — the agent speaks in ResearchActions, the
environment responds with ResearchObservations.

OpenEnv expects:
  - Action subclass for agent inputs
  - Observation subclass for environment outputs (returned directly from reset/step)
  - State subclass for full environment state (used by env.state property)
"""

from typing import Any, Optional
from openenv.core.env_server.types import Action, Observation, State


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------
# The agent sends one of these to env.step() each turn.
# tool_name maps to one of the three registered Tavily tools:
#   "tavily_search", "tavily_extract", "tavily_crawl"
# tool_args is a dict of kwargs passed directly to the tool.

class ResearchAction(Action):
    tool_name: str          # which tool to call
    tool_args: dict[str, Any]  # arguments for that tool


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------
# Returned to the agent after each step.
# Contains the raw tool output plus metadata the agent can use to reason.

class ResearchObservation(Observation):
    tool_name: str              # which tool was called
    tool_args: dict[str, Any]   # what args were passed
    result: Any                 # raw tool output (dict from Tavily)
    step: int                   # current step number
    done: bool                  # whether the episode has ended
    reward: float               # reward earned this step (LLM-as-judge or keyword)
    message: Optional[str] = None  # optional human-readable status message


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
# Full environment state — returned by env.state property.
# Captures the entire episode history for inspection or replay.

class ResearchState(State):
    query: str                          # the original research question
    step: int                           # current step count
    max_steps: int                      # step limit for the episode
    done: bool                          # whether episode has ended
    total_reward: float                 # cumulative reward so far
    history: list[dict[str, Any]]       # list of {action, observation, reward} per step
    tool_usage: dict[str, int]          # counts per tool: {"tavily_search": 2, ...}
