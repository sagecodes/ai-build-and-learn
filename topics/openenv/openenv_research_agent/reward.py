"""
Reward functions for the Research RL Environment.

Two reward strategies are provided and used side-by-side in the demo:

1. keyword_reward  — Traditional RL style. Counts keyword matches in each
                     tool result per step. Easy to game — an agent that stuffs
                     keywords scores high even with garbage content.

2. llm_judge_final_reward — OpenEnv style. Uses Claude as a judge to evaluate
                            the FULL accumulated research at episode end, not
                            each step in isolation. Rewards genuine depth and
                            breadth of research — much harder to game.

The demo's "reward hacking" moment:
  - Traditional agent: high keyword score per step, low final LLM score
  - OpenEnv agent: lower per-step scores, HIGH final LLM score
  - The gap proves why per-step keyword rewards fail for language tasks

Per-step keyword_reward signature:
    keyword_reward(query, tool_name, result, step) -> float (0.0 - 1.0)

Final LLM judge signature:
    llm_judge_final_reward(query, history) -> float (0.0 - 1.0)
    where history is the list of {tool_name, tool_args, reward} dicts
    from ResearchState.history
"""

import os
import re
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# 1. Keyword Match Reward (Traditional RL — per step)
# ---------------------------------------------------------------------------
# Reward = fraction of query keywords found in the tool result text.
# Simple, fast, and completely gameable.

def keyword_reward(
    query: str,
    tool_name: str,
    result: dict,
    step: int,
) -> float:
    """
    Traditional RL reward: count how many query keywords appear in the result.

    Returns a float between 0.0 and 1.0.
    Deliberately simple and gameable — this is the point of the demo.
    """
    score, _ = keyword_reward_with_detail(query, tool_name, result, step)
    return score


def keyword_reward_with_detail(
    query: str,
    tool_name: str,
    result: dict,
    step: int,
) -> tuple[float, list[str]]:
    """
    Same as keyword_reward but also returns the list of matched keywords.

    Returns (score, matched_keywords) where matched_keywords is the list
    of query keywords found in the result text — useful for surfacing
    exactly what the agent is gaming.
    """
    if not result or "error" in result:
        return 0.0, []

    result_text = _flatten_result(result).lower()

    stopwords = {"the", "a", "an", "is", "are", "what", "how", "why", "of", "in", "on", "vs", "for"}
    keywords = [
        w for w in re.findall(r'\w+', query.lower())
        if w not in stopwords and len(w) > 2
    ]

    if not keywords:
        return 0.0, []

    matched = [kw for kw in keywords if kw in result_text]
    score = len(matched) / len(keywords)

    # Small step penalty to discourage padding — but not enough to stop gaming
    step_penalty = max(0.0, (step - 1) * 0.02)
    score = max(0.0, round(score - step_penalty, 3))

    return score, matched


# ---------------------------------------------------------------------------
# 2. LLM-as-Judge Final Reward (OpenEnv style — end of episode)
# ---------------------------------------------------------------------------
# Judges the FULL accumulated research across all steps, not each step alone.
# This rewards agents that chain tools intelligently and gather real depth.

_anthropic_client = None

def _get_anthropic_client() -> Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _anthropic_client


def llm_judge_final_reward(
    query: str,
    accumulated_results: list[dict],
) -> float:
    """
    OpenEnv-style final reward: judge the full accumulated research at episode end.

    accumulated_results is a list of result dicts gathered across all steps.
    Claude evaluates the combined research for depth, breadth, and relevance
    — rewarding genuine understanding, not keyword frequency.

    Returns 0.0-1.0.
    """
    if not accumulated_results:
        return 0.0

    # Combine all results into one research summary for the judge.
    # Give each result more room so the OpenEnv agent's deeper content
    # isn't truncated to the same size as the traditional agent's snippets.
    combined = ""
    for i, result in enumerate(accumulated_results):
        text = _flatten_result(result)[:1500]  # 1500 chars per step
        if text.strip():
            combined += f"\n--- Result {i+1} ---\n{text}\n"

    if not combined.strip():
        return 0.0

    # Truncate total — generous limit so deep research isn't penalized
    combined = combined[:12000]

    prompt = f"""You are evaluating the complete research output of an AI agent.

Research question: {query}

The agent gathered the following information across multiple tool calls:
{combined}

Rate the OVERALL quality of this accumulated research for answering the question.

Scoring criteria — be strict and differentiate clearly:
- PENALIZE heavily if the agent repeated nearly identical searches multiple times
  without refining its approach. Repetition is a sign of poor reasoning.
- REWARD agents that used progressively refined queries based on prior results,
  extracted full page content from specific URLs, or followed specific leads.
- REWARD agents that found specific facts, figures, prices, or direct comparisons
  — not just pages that mention the topic keywords.
- PENALIZE if the research is shallow (only search snippets, no extraction or
  follow-up on specific sources).

Respond with ONLY a single integer from 1 to 10.
1 = repeated identical searches, no real information gathered
3 = some relevant snippets but repetitive or shallow
5 = decent coverage but missed depth or specifics
7 = good breadth and some depth, specific facts found
10 = comprehensive, specific, direct answers with evidence of deep research
"""

    try:
        client = _get_anthropic_client()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        score_text = response.content[0].text.strip()
        score = int(re.search(r'\d+', score_text).group())
        score = max(1, min(10, score))
        return round(score / 10.0, 2)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _flatten_result(result: dict) -> str:
    """Recursively flatten a result dict to a single string for text scanning."""
    parts = []
    for value in result.values():
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    parts.append(_flatten_result(item))
                elif isinstance(item, str):
                    parts.append(item)
        elif isinstance(value, dict):
            parts.append(_flatten_result(value))
    return " ".join(parts)
