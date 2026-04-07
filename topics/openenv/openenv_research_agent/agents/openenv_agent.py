"""
OpenEnv Research Agent — Claude via OpenEnv AnthropicClient.

This is the "after" side of the demo comparison. Unlike the traditional
agent, this agent:

  1. Discovers available tools dynamically at runtime (no fixed action space)
  2. Reasons about which tool to use based on prior observations
  3. Chains tools in sequence: search → extract → crawl as needed
  4. Earns high llm_judge_final_reward scores because it actually researches well

The ReAct loop is written manually using the Anthropic SDK — explicit and
inspectable for demo purposes.

Agents connect to the OpenEnv HTTP server via GenericEnvClient. The server
URL defaults to ENV_URL env var (http://localhost:8000 for local Docker),
or can be passed explicitly as env_url for Flyte task pods.

This agent can also be run in "race mode" — multiple instances connect to
the same server, each getting its own isolated session via
SUPPORTS_CONCURRENT_SESSIONS = True.

Usage (single agent — local Docker):
    agent = OpenEnvAgent(query="What is MCP?")
    for step_result in agent.run():
        print(step_result)

Usage (race — 3 agents, see app.py):
    agents = [OpenEnvAgent(query, agent_id=i) for i in range(3)]
    # each connects to the same Docker server, separate sessions

Usage (Flyte task — local server):
    agent = OpenEnvAgent(query="What is MCP?", env_url="http://127.0.0.1:PORT")
    for step_result in agent.run():
        print(step_result)
"""

import os
import json
from typing import Generator, Optional
from dotenv import load_dotenv
from anthropic import Anthropic
from openenv import GenericEnvClient

from env.models import ResearchAction
from reward import llm_judge_final_reward
from system_prompt import SYSTEM_PROMPT

load_dotenv()

ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

# Tool schemas passed to Claude so it knows how to call each Tavily action
_TOOL_SCHEMAS = [
    {
        "name": "tavily_search",
        "description": "Search the web using Tavily. Use first for any research task to discover relevant URLs and get an overview.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "max_results": {"type": "integer", "default": 5},
                "search_depth": {"type": "string", "enum": ["basic", "advanced"], "default": "basic"},
                "include_domains": {"type": "array", "items": {"type": "string"}},
                "exclude_domains": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query"],
        },
    },
    {
        "name": "tavily_extract",
        "description": "Extract full page content from specific URLs. Use after search when you need full content, not just snippets.",
        "input_schema": {
            "type": "object",
            "properties": {
                "urls": {"type": "array", "items": {"type": "string"}, "description": "URLs to extract"},
                "extract_depth": {"type": "string", "enum": ["basic", "advanced"], "default": "basic"},
            },
            "required": ["urls"],
        },
    },
    {
        "name": "tavily_crawl",
        "description": "Crawl a website from a root URL to gather site-wide content. Use for docs or when you need many pages.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Root URL to crawl from"},
                "max_depth": {"type": "integer", "default": 1},
                "max_breadth": {"type": "integer", "default": 10},
                "limit": {"type": "integer", "default": 10},
                "instructions": {"type": "string"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "finish",
        "description": "Signal that you have gathered enough information to answer the research question. Call this when done.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Brief summary of what you found"},
            },
            "required": ["summary"],
        },
    },
]


class OpenEnvAgent:
    """
    Claude-powered research agent running inside an OpenEnv environment.

    Uses the Anthropic SDK directly (matching the fastmcp_agent_tavily
    pattern) rather than OpenEnv's AnthropicClient wrapper, so the ReAct
    loop is explicit and inspectable for demo purposes.
    """

    def __init__(
        self,
        query: str,
        agent_id: int = 0,
        max_steps: int = 10,  # max tool calls (not LLM iterations)
        env_url: str = None,
    ):
        self.query = query
        self.agent_id = agent_id
        self.max_steps = max_steps
        self._env_url = env_url or os.getenv("ENV_URL", "http://localhost:8000")
        self._client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def run(self) -> Generator[dict, None, None]:
        """
        Run one research episode via Docker EnvClient, yielding a status dict after each step.

        Per-step: yields tool_name so the step log updates live.
        Final step: yields llm_final_score from the accumulated research judgment.

        max_steps limits actual tool calls (client.step() invocations), not LLM
        iterations. One LLM response can call multiple tools; the step counter is
        checked before each new LLM call so the limit is respected across batches.
        """
        messages = [{"role": "user", "content": self.query}]
        accumulated_results = []
        step_count = 0  # counts actual tool calls, not LLM iterations

        with GenericEnvClient(base_url=self._env_url).sync() as client:
            client.reset(query=self.query)

            # Loop bound is generous — actual limit is step_count vs max_steps.
            # One LLM call can execute multiple tools; we check before each new call.
            for _ in range(self.max_steps * 3):
                if step_count >= self.max_steps:
                    break

                response = self._client.messages.create(
                    model="claude-opus-4-6",
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=_TOOL_SCHEMAS,
                    messages=messages,
                )

                if response.stop_reason == "tool_use":
                    messages.append({"role": "assistant", "content": response.content})

                    # First pass: execute ALL tool calls and collect results.
                    # We must provide a tool_result for every tool_use block —
                    # breaking mid-loop causes Anthropic API 400 errors.
                    tool_results = []
                    step_yields = []
                    episode_done = False
                    finish_requested = False

                    for block in response.content:
                        if block.type != "tool_use":
                            continue

                        tool_name = block.name
                        tool_args = block.input

                        if tool_name == "finish":
                            finish_requested = True
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": "Episode finished.",
                            })
                            continue

                        action = ResearchAction(tool_name=tool_name, tool_args=tool_args)
                        step_result = client.step(action)
                        step_count += 1

                        obs = step_result.observation  # dict from ResearchObservation
                        tool_result = obs.get("result", {})
                        step_num = obs.get("step", 0)
                        done = step_result.done

                        accumulated_results.append(tool_result)

                        step_yields.append({
                            "step": step_num,
                            "tool_name": tool_name,
                            "tool_args": tool_args,
                            "llm_final_score": None,
                            "result_preview": _preview(tool_result),
                            "done": done,
                            "agent_id": self.agent_id,
                            "agent": "openenv",
                        })

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(tool_result)[:3000],
                        })

                        if done:
                            episode_done = True

                    # Yield step updates
                    for s in step_yields:
                        yield s

                    messages.append({"role": "user", "content": tool_results})

                    # Handle finish, episode done, or step limit after feeding all results back
                    if finish_requested or episode_done or step_count >= self.max_steps:
                        llm_final = llm_judge_final_reward(
                            query=self.query,
                            accumulated_results=accumulated_results,
                        )
                        yield {
                            "step": -1,
                            "tool_name": "final_judgment",
                            "tool_args": {},
                            "llm_final_score": llm_final,
                            "done": True,
                            "agent_id": self.agent_id,
                            "agent": "openenv",
                        }
                        return

                else:
                    # Claude finished without calling finish — compute final reward anyway
                    final_text = next(
                        (b.text for b in response.content if hasattr(b, "text")), ""
                    )
                    if final_text:
                        accumulated_results.append({"text": final_text})

                    llm_final = llm_judge_final_reward(
                        query=self.query,
                        accumulated_results=accumulated_results,
                    )
                    yield {
                        "step": -1,
                        "tool_name": "final_judgment",
                        "tool_args": {},
                        "llm_final_score": llm_final,
                        "result_preview": final_text[:500],
                        "done": True,
                        "agent_id": self.agent_id,
                        "agent": "openenv",
                    }
                    return

            # Hit max steps — compute final reward on what was gathered
            llm_final = llm_judge_final_reward(
                query=self.query,
                accumulated_results=accumulated_results,
            )
            yield {
                "step": -1,
                "tool_name": "final_judgment",
                "tool_args": {},
                "llm_final_score": llm_final,
                "done": True,
                "agent_id": self.agent_id,
                "agent": "openenv",
            }


def _preview(result: dict, max_chars: int = 200) -> str:
    """Return a short readable preview of a tool result for the step log."""
    try:
        text = json.dumps(result)
        return text[:max_chars] + ("..." if len(text) > max_chars else "")
    except Exception:
        return str(result)[:max_chars]
