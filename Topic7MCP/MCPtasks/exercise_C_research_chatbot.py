import json
import os

import requests
from openai import OpenAI


URL = "https://asta-tools.allen.ai/mcp/v1"
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": os.environ["ASTA_API_KEY"],
}


def parse_sse_json(resp):
    for line in resp.text.splitlines():
        if line.startswith("data:"):
            return json.loads(line.replace("data:", "").strip())
    raise ValueError("No JSON found in MCP response")


def get_asta_tools():
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {},
    }
    resp = requests.post(URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    tools = parse_sse_json(resp)["result"]["tools"]
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["inputSchema"],
            },
        }
        for tool in tools
    ]


def call_asta_tool(name, arguments):
    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    resp = requests.post(URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = parse_sse_json(resp)
    parts = [item["text"] for item in data["result"]["content"]]
    return "\n---\n".join(parts)


def chat(user_message, messages, tools):
    while True:
        messages.append({"role": "user", "content": user_message})
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
        )

        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content

        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            print(f"Calling: {name}")
            print(f"Args: {args}")

            result = call_asta_tool(name, args)
            if len(result) > 8000:
                result = result[:8000] + "\n... [truncated]"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result),
                }
            )
        user_message = ""


def main():
    print("=" * 60)
    print("Asta Research Chatbot")
    print("=" * 60)

    tools = get_asta_tools()
    print(f"Loaded {len(tools)} tools")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant with access to Semantic Scholar "
                "via Asta tools. Use only parameters defined in the schema, "
                "never invent parameter names, and omit optional fields when unsure."
            ),
        }
    ]

    test_queries = [
        "Find recent papers about large language model agents",
        "Who wrote Attention is All You Need and what else have they published?",
        "What papers cite the original BERT paper (ARXIV:1810.04805)?",
        "Summarize the references used in the ReAct paper (ARXIV:2210.03629)",
    ]

    for query in test_queries:
        print(f"\n{'-' * 60}")
        print(f"Query: {query}")
        print(f"{'-' * 60}\n")
        answer = chat(query, messages, tools)
        print(f"\nAnswer:\n{answer}\n")


if __name__ == "__main__":
    main()
