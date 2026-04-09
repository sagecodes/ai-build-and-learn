"""
Shared utilities for Tavily tool wrappers.

Centralises rate-limit detection and retry logic so search.py, extract.py,
and crawl.py don't each maintain their own copy.
"""

import time


def is_rate_limit(error: Exception) -> bool:
    """Return True if the error looks like a Tavily rate-limit or quota error."""
    msg = str(error).lower()
    return "usage limit" in msg or "rate limit" in msg or "429" in msg


def tavily_call_with_retry(fn, on_error: dict, max_attempts: int = 3) -> dict:
    """
    Call fn() with exponential backoff on rate-limit errors.

    Args:
        fn:          Zero-argument callable that performs the Tavily API call
                     and returns a result dict on success.
        on_error:    Base dict merged with {"error": ...} on final failure.
                     Callers supply tool-specific fields (e.g. {"query": q}).
        max_attempts: Total attempts before giving up (default: 3).

    Returns:
        Result dict from fn() on success, or {**on_error, "error": message}
        on failure after all retries are exhausted.
    """
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            if is_rate_limit(e) and attempt < max_attempts - 1:
                time.sleep(2 ** attempt)   # 1 s, 2 s
                continue
            return {**on_error, "error": str(e)}
