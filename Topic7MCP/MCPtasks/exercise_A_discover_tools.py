"""
Exercise A:
To find papers about "transformer attention mechanisms", use
search_papers_by_relevance because it searches papers by keyword/relevance.

To find who else published in the same area as a specific author, use
search_authors_by_name to identify the author, then get_author_papers to inspect
that author's publications and related research areas.
"""

import json
import os

import requests


URL = "https://asta-tools.allen.ai/mcp/v1"


def parse_sse_json(resp):
    for line in resp.text.splitlines():
        if line.startswith("data:"):
            return json.loads(line.replace("data:", "").strip())
    raise ValueError("No JSON found in MCP response")


headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": os.environ["ASTA_API_KEY"],
}

payload = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {},
}

resp = requests.post(URL, headers=headers, json=payload, timeout=60)
resp.raise_for_status()
tools = parse_sse_json(resp)["result"]["tools"]

for tool in tools:
    description = " ".join(tool["description"].strip().split())
    print(f"\nTool: {tool['name']}")
    print(f"  Description: {description}")

    schema = tool["inputSchema"]
    props = schema.get("properties", {})
    required = schema.get("required", [])

    for name, meta in props.items():
        kind = meta.get("type", "unknown")
        label = "Required" if name in required else "Optional"
        print(f"  {label}: {name} ({kind})")
