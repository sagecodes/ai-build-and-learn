"""
Traditional RL Agent — the "before" side of the demo comparison.

This agent deliberately mimics the limitations of classic RL approaches:

  1. Fixed discrete action space — it can only call tavily_search with
     pre-defined query templates. No ability to reason about which tool
     to use or adapt based on results.

  2. Keyword stuffing strategy — it constructs queries by appending as many
     query keywords as possible, explicitly to maximize the keyword_reward.
     This is reward hacking in action.

  3. No reasoning — actions are chosen by cycling through templates, not
     by reading or understanding the previous observation.

  4. No tool chaining — it never calls tavily_extract or tavily_crawl
     because its fixed policy doesn't include them.

The result: high per-step keyword_reward scores (0.8-1.0), low final
llm_judge_final_reward score (0.2-0.4). This gap is the central demo moment.

Usage:
    agent = TraditionalAgent(query="What is MCP?")
    for step_result in agent.run(env):
        print(step_result)
"""

import os
import re
from typing import Generator
from openenv import GenericEnvClient
from env.models import ResearchAction
from reward import keyword_reward, llm_judge_final_reward

ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

# Fixed action templates — the traditional agent's entire "policy"
# Each template stuffs the query keywords in different positions.
_QUERY_TEMPLATES = [
    "{query}",
    "{query} overview explanation",
    "{query} details facts sources",
    "{query} comprehensive information data",
    "{query} analysis research findings",
    "{keywords} {query} complete guide",
    "{keywords} {query} key points summary",
    "{keywords} {query} everything you need to know",
]


class TraditionalAgent:
    """
    A fixed-policy RL agent that keyword-stuffs to game the keyword reward.

    Demonstrates why naive reward functions fail for language tasks:
    the agent optimizes the metric, not the actual research goal.
    """

    def __init__(self, query: str, max_steps: int = 8, env_url: str = None):
        self.query = query
        self.max_steps = max_steps
        self._env_url = env_url or os.getenv("ENV_URL", "http://localhost:8000")
        self._keywords = self._extract_keywords(query)

    def _extract_keywords(self, query: str) -> str:
        """Extract content words from the query for stuffing."""
        stopwords = {"the", "a", "an", "is", "are", "what", "how", "why", "of", "in", "on", "vs", "for"}
        words = [w for w in re.findall(r'\w+', query.lower()) if w not in stopwords and len(w) > 2]
        # Repeat keywords to maximize match count — classic reward hacking
        return " ".join(words * 3)

    def _choose_action(self, step: int) -> ResearchAction:
        """
        Select the next action by cycling through templates.
        No reasoning, no adaptation — pure fixed policy.
        """
        template = _QUERY_TEMPLATES[step % len(_QUERY_TEMPLATES)]
        stuffed_query = template.format(
            query=self.query,
            keywords=self._keywords,
        )
        # Always uses tavily_search — never discovers extract or crawl
        return ResearchAction(
            tool_name="tavily_search",
            tool_args={"query": stuffed_query, "max_results": 5},
        )

    def run(self) -> Generator[dict, None, None]:
        """
        Run one episode via Docker EnvClient, yielding a status dict after each step.

        Per-step: yields keyword_score so the chart shows it climbing.
        Final step: yields llm_final_score showing the true quality gap.
        """
        accumulated_results = []
        kw_scores = []

        with GenericEnvClient(base_url=self._env_url).sync() as client:
            client.reset(query=self.query)

            for step in range(self.max_steps):
                action = self._choose_action(step)
                step_result = client.step(action)

                obs = step_result.observation  # dict from ResearchObservation
                tool_result = obs.get("result", {})
                tool_name = obs.get("tool_name", action.tool_name)
                step_num = obs.get("step", step + 1)
                done = step_result.done

                # Per-step keyword score — this is what the agent optimizes
                kw_score = keyword_reward(
                    query=self.query,
                    tool_name=tool_name,
                    result=tool_result,
                    step=step_num,
                )
                kw_scores.append(kw_score)
                accumulated_results.append(tool_result)

                yield {
                    "step": step_num,
                    "tool_name": tool_name,
                    "query_used": action.tool_args.get("query", ""),
                    "keyword_score": kw_score,
                    "llm_final_score": None,  # not computed yet
                    "done": done,
                    "agent": "traditional",
                }

                if done:
                    break

        # Final LLM judge — evaluates ALL accumulated research at once
        llm_final = llm_judge_final_reward(
            query=self.query,
            accumulated_results=accumulated_results,
        )
        avg_kw = sum(kw_scores) / max(len(kw_scores), 1)

        # Yield a final summary step with both scores for the chart
        yield {
            "step": -1,  # sentinel: this is the final summary entry
            "tool_name": "final_judgment",
            "query_used": "",
            "keyword_score": avg_kw,
            "llm_final_score": llm_final,
            "done": True,
            "agent": "traditional",
        }
