"""
OpenEnv Research Agent — Claude via OpenEnv AnthropicClient.

This is the "after" side of the demo comparison. Unlike the traditional
agent, this agent:

  1. Discovers available tools dynamically at runtime (no fixed action space)
  2. Reasons about which tool to use based on prior observations
  3. Chains tools in sequence: search → extract → crawl as needed
  4. Earns high llm_judge_reward scores because it actually researches well

The ReAct loop is handled by OpenEnv's AnthropicClient.complete_with_tools()
rather than being written manually (unlike fastmcp_agent_tavily/agent.py).

This agent can also be run in "race mode" — multiple instances of this
class run concurrently against the same ResearchEnvironment, which supports
concurrent sessions via SUPPORTS_CONCURRENT_SESSIONS = True.

Usage (single agent):
    agent = OpenEnvAgent(query="What is MCP?")
    for step_result in agent.run(env):
        print(step_result)

Usage (race — 3 agents, see app.py):
    agents = [OpenEnvAgent(query, agent_id=i) for i in range(3)]
    # run concurrently via asyncio
"""

import os
import json
from typing import Generator, Optional
from dotenv import load_dotenv
from anthropic import Anthropic

from env.models import ResearchAction
from env.research_env import ResearchEnvironment
from reward import llm_judge_final_reward
from system_prompt import SYSTEM_PROMPT

load_dotenv()

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
        max_steps: int = 10,
    ):
        self.query = query
        self.agent_id = agent_id
        self.max_steps = max_steps
        self._client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def run(self, env: ResearchEnvironment) -> Generator[dict, None, None]:
        """
        Run one research episode, yielding a status dict after each step.

        Per-step: yields tool_name so the step log updates live.
        Final step: yields llm_final_score from the accumulated research judgment.
        """
        env.reset(query=self.query)

        messages = [{"role": "user", "content": self.query}]
        accumulated_results = []

        for _ in range(self.max_steps):
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
                    obs = env.step(action)
                    accumulated_results.append(obs.result)

                    step_yields.append({
                        "step": obs.step,
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "llm_final_score": None,
                        "result_preview": _preview(obs.result),
                        "done": obs.done,
                        "agent_id": self.agent_id,
                        "agent": "openenv",
                    })

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(obs.result)[:3000],
                    })

                    if obs.done:
                        episode_done = True

                # Yield step updates
                for s in step_yields:
                    yield s

                messages.append({"role": "user", "content": tool_results})

                # Handle finish or episode done after feeding all results back
                if finish_requested or episode_done:
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
