import requests
import os
import json

URL = "https://asta-tools.allen.ai/mcp/v1"

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": os.environ["ASTA_API_KEY"]
}


def parse_sse_json(resp):
    """
    Extract JSON from MCP text/event-stream response
    """
    for line in resp.text.splitlines():
        if line.startswith("data:"):
            return json.loads(line.replace("data:", "").strip())

    raise ValueError("No JSON found in MCP response")


def call_tool(name, arguments, req_id=1):

    payload = {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "tools/call",
        "params": {
            "name": name,
            "arguments": arguments
        }
    }

    resp = requests.post(URL, headers=headers, json=payload)
    data = parse_sse_json(resp)

    # Asta returns JSON embedded as text
    content = data["result"]["content"]

    return content

def drill1():

    papers = call_tool(
        "search_papers_by_relevance",
        {
            "keyword": "large language model agents",
            "fields": "title,abstract,year,authors",
            "limit": 5
        },
        2
    )

    print("\nTop LLM Agent Papers:")

    for i, _ in enumerate(papers):
        p = json.loads(papers[i]["text"])
        print(f"{i+1}. {p["title"]} ({p["year"]})")


def drill2():

    citations = call_tool(
        "get_citations",
        {
            "paper_id": "ARXIV:1810.04805",
            "fields": "title,year",
            "publication_date_range": "2023-01-01:",
            "limit": 10
        },
        3
    )

    print("\nNumber of citations since 2023:", len(citations))

    for i, _ in enumerate(citations[:5]):
        p = json.loads(citations[i]["text"])
        print(f"{i+1}. {p["citingPaper"]["title"]}")


def drill3():

    refs = call_tool(
        "get_paper",
        {
            "paper_id": "ARXIV:2210.03629",
            "fields": "references,references.title,references.year"
        },
        4
    )

    papers = json.loads(refs[0]["text"])
    references = sorted([p for p in papers["references"] if p["year"] is not None], key=lambda x: x["year"])
    print("\nReferences for ReAct:")

    for p in references:
        print(p["year"], "-", p["title"])


if __name__ == "__main__":
    drill1()
    drill2()
    drill3()