"""Tavily Extract API examples.

Extract structured content from URLs — useful for pulling
clean text from pages your agent found via search.

Usage:
    python 02_extract.py
"""

import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()
client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# --- 1. Basic extraction ---

print("=" * 60)
print("1. Basic Extraction (single URL)")
print("=" * 60)

response = client.extract(
    "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
)

for result in response["results"]:
    print(f"\nURL: {result['url']}")
    print(f"Content (first 500 chars):\n{result['raw_content'][:500]}...")


# --- 2. Multiple URLs at once ---

print("\n" + "=" * 60)
print("2. Batch Extraction (multiple URLs)")
print("=" * 60)

response = client.extract(
    urls=[
        "https://en.wikipedia.org/wiki/Large_language_model",
        "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
    ]
)

for result in response["results"]:
    print(f"\nURL: {result['url']}")
    print(f"Content (first 300 chars):\n{result['raw_content'][:300]}...")


# --- 3. Extract with query for targeted chunks ---

print("\n" + "=" * 60)
print("3. Extract with Query (targeted chunks)")
print("=" * 60)

response = client.extract(
    urls="https://en.wikipedia.org/wiki/Large_language_model",
    query="What are the limitations of large language models?",
    chunks_per_source=2,
)

for result in response["results"]:
    print(f"\nURL: {result['url']}")
    print(f"Content (first 500 chars):\n{result['raw_content'][:500]}...")


# --- 4. Advanced extraction with text format ---

print("\n" + "=" * 60)
print("4. Advanced Extraction (text format)")
print("=" * 60)

response = client.extract(
    urls="https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    extract_depth="advanced",
    format="text",
)

for result in response["results"]:
    print(f"\nURL: {result['url']}")
    print(f"Content (first 500 chars):\n{result['raw_content'][:500]}...")