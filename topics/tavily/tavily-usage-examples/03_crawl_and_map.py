"""Tavily Crawl and Map API examples.

Crawl websites with instructions and map site structure.

Usage:
    python 03_crawl_and_map.py
"""

import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()
client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# --- 1. Basic crawl ---

print("=" * 60)
print("1. Basic Crawl")
print("=" * 60)

response = client.crawl(
    "https://docs.tavily.com",
    max_depth=1,
    max_breadth=5,
    limit=5,
)

print(f"Pages crawled: {len(response.get('results', []))}")
for result in response.get("results", []):
    print(f"\n- {result['url']}")
    print(f"  {result['raw_content'][:200]}...")


# --- 2. Crawl with instructions ---

print("\n" + "=" * 60)
print("2. Crawl with Instructions")
print("=" * 60)

response = client.crawl(
    "https://docs.tavily.com",
    instructions="Find pages about the Python SDK and API reference",
    max_depth=2,
    max_breadth=5,
    limit=5,
)

print(f"Pages found: {len(response.get('results', []))}")
for result in response.get("results", []):
    print(f"\n- {result['url']}")
    print(f"  {result['raw_content'][:200]}...")


# --- 3. Crawl with path filtering ---

print("\n" + "=" * 60)
print("3. Crawl with Path Filtering")
print("=" * 60)

response = client.crawl(
    "https://docs.tavily.com",
    select_paths=[r".*/api-reference/.*"],  # Only follow API reference pages
    exclude_paths=[r".*/integrations/.*"],  # Skip integration pages
    max_depth=3,
    limit=5,
)

print(f"Pages found: {len(response.get('results', []))}")
for result in response.get("results", []):
    print(f"\n- {result['url']}")


# --- 4. Map a website ---

print("\n" + "=" * 60)
print("4. Map Site Structure")
print("=" * 60)

response = client.map(
    "https://docs.tavily.com",
    max_depth=2,
    max_breadth=10,
    limit=20,
)

print(f"Pages mapped: {len(response.get('results', []))}")
for url in response.get("results", []):
    print(f"  - {url}")


# --- 5. Map with instructions ---

print("\n" + "=" * 60)
print("5. Map with Instructions")
print("=" * 60)

response = client.map(
    "https://docs.tavily.com",
    instructions="Find all API reference and SDK documentation pages",
    max_depth=2,
    limit=15,
)

print(f"Pages found: {len(response.get('results', []))}")
for url in response.get("results", []):
    print(f"  - {url}")