import httpx
import trafilatura
from ddgs import DDGS

MAX_CHARS = 4000    # keep results small so they don't flood the model's context

def web_search(query: str) -> str:
    """Search the web and return the top results as plain text."""
    results = DDGS().text(query, max_results=5)
    if not results:
        return "No results found."
    return "\n\n".join(
        f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}" for r in results
    )

def fetch_page(url: str) -> str:
    """Download a web page and return its main text."""
    try:
        response = httpx.get(url, timeout=10, follow_redirects=True)
        response.raise_for_status()             # turn 404s/500s into exceptions
    except httpx.HTTPError as error:
        return f"Could not fetch page: {error}"
    text = trafilatura.extract(response.text)
    return (text or "Could not extract readable text.")[:MAX_CHARS]

# Map tool names (as the model will them) to the actual Python functions.
TOOL_FUNCTIONS = {"web_search": web_search, "fetch_page": fetch_page}

# Descriptions the model reads to decide when and how to use each tool.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information, news, or facts "
            "you are unsure about. Returns titles, URLs and short snippets.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search terms"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Read the full text of a web page. Use after web_search "
            "when a snippet isn't enough.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "Full URL"}},
                "required": ["url"],
            },
        },
    },
]




