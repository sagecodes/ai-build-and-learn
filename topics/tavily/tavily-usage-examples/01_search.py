"""Tavily Search API examples.

Demonstrates basic search, advanced search with answers,
topic filters, time ranges, and domain filtering.

Usage:
    python 01_search.py
"""

import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()
client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# --- 1. Basic search ---

print("=" * 60)
print("1. Basic Search")
print("=" * 60)

response = client.search("What is retrieval augmented generation?")

for result in response["results"]:
    print(f"\n- {result['title']}")
    print(f"  {result['url']}")
    print(f"  {result['content'][:150]}...")


# --- 2. Advanced search with AI-generated answer ---

print("\n" + "=" * 60)
print("2. Advanced Search with Answer")
print("=" * 60)

response = client.search(
    "What are the main differences between RAG and fine-tuning?",
    search_depth="advanced",
    include_answer="advanced",
    max_results=5,
)

print(f"\nAnswer:\n{response['answer']}")
print(f"\nSources ({len(response['results'])}):")
for result in response["results"]:
    print(f"  - {result['title']}: {result['url']}")


# --- 3. News topic search with time range ---

print("\n" + "=" * 60)
print("3. News Search with Time Range")
print("=" * 60)

response = client.search(
    "AI agent frameworks",
    topic="news",
    time_range="week",
    max_results=5,
)

for result in response["results"]:
    print(f"\n- {result['title']}")
    print(f"  {result['url']}")


# --- 4. Domain-filtered search ---

print("\n" + "=" * 60)
print("4. Domain-Filtered Search")
print("=" * 60)

# Only search specific domains
response = client.search(
    "transformer architecture explained",
    include_domains=["arxiv.org", "github.com"],
    max_results=5,
)

print("Results from arxiv.org and github.com only:")
for result in response["results"]:
    print(f"\n- {result['title']}")
    print(f"  {result['url']}")

# Exclude specific domains
response = client.search(
    "python async tutorial",
    exclude_domains=["medium.com", "w3schools.com"],
    max_results=3,
)

print("\nResults excluding medium.com and w3schools.com:")
for result in response["results"]:
    print(f"\n- {result['title']}")
    print(f"  {result['url']}")


# --- 5. Search with images ---

print("\n" + "=" * 60)
print("5. Search with Images")
print("=" * 60)

response = client.search(
    "LangGraph agent architecture diagrams",
    include_images=True,
    include_image_descriptions=True,
    max_results=3,
)

for result in response["results"]:
    print(f"\n- {result['title']}: {result['url']}")

if response.get("images"):
    print(f"\nImages found: {len(response['images'])}")
    for img in response["images"][:3]:
        if isinstance(img, dict):
            print(f"  - {img.get('description', 'No description')}: {img['url']}")
        else:
            print(f"  - {img}")