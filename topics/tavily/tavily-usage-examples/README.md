# Tavily Usage Examples

Runnable examples demonstrating core Tavily API features.

## Setup

```bash
pip install tavily-python python-dotenv
```

Add your API key to a `.env` file in the project root:

```
TAVILY_API_KEY=your-key-here
```

## Examples

| File | What it covers |
|------|---------------|
| [01_search.py](01_search.py) | Basic search, advanced search, topic filters, time ranges, domain filtering |
| [02_extract.py](02_extract.py) | Extract content from URLs, markdown/text formats, image extraction |
| [03_crawl_and_map.py](03_crawl_and_map.py) | Crawl a website with instructions, map a site's structure |