import json
import os
import sys

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


def call_tool(name, arguments, req_id=1):
    payload = {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    resp = requests.post(URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = parse_sse_json(resp)
    if "error" in data:
        raise RuntimeError(f"MCP error from {name}: {data['error']}")
    return data["result"]


def content_as_json_items(result):
    items = []
    for item in result.get("content", []):
        text = item.get("text", "").strip()
        if not text:
            continue
        if text.startswith("Error executing tool"):
            raise RuntimeError(text)
        try:
            items.append(json.loads(text))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Expected JSON tool content, got: {text[:300]}") from exc
    return items


def result_payload(result):
    structured = result.get("structuredContent", {}).get("result")
    if structured is not None:
        return structured

    items = content_as_json_items(result)
    if not items:
        raise ValueError(f"No parseable tool payload found in result: {result}")
    return items[0] if len(items) == 1 else items


def get_paper_payload(paper_id, fields):
    paper_ids = [paper_id]
    if paper_id == "ARXIV:2210.03629":
        paper_ids.append("URL:https://arxiv.org/abs/2210.03629")

    last_error = None
    for candidate_id in paper_ids:
        try:
            return result_payload(
                call_tool("get_paper", {"paper_id": candidate_id, "fields": fields})
            )
        except RuntimeError as exc:
            last_error = exc
    raise last_error


def generate_report(seed, references, citations, author_profiles):
    context = json.dumps(
        {
            "seed_paper": {
                "title": seed["title"],
                "abstract": seed["abstract"],
                "year": seed["year"],
                "authors": [author["name"] for author in seed["authors"]],
                "fields": seed["fieldsOfStudy"],
                "citations": seed["citationCount"],
            },
            "key_references": [
                {
                    "title": ref["title"],
                    "year": ref["year"],
                    "abstract": (ref["abstract"] or "")[:200],
                    "citations": ref["citationCount"],
                }
                for ref in references
            ],
            "recent_citations": [
                {
                    "title": paper["title"],
                    "year": paper["year"],
                    "abstract": (paper["abstract"] or "")[:200],
                    "citations": paper["citationCount"],
                }
                for paper in citations
            ],
            "author_profiles": [
                {
                    "name": profile["name"],
                    "notable_work": (
                        profile["top_paper"]["title"] if profile["top_paper"] else None
                    ),
                    "notable_citations": (
                        profile["top_paper"]["citationCount"]
                        if profile["top_paper"]
                        else None
                    ),
                }
                for profile in author_profiles
            ],
        },
        indent=2,
    )

    prompt = f"""Based on the given research data, generate a structured markdown report containing:

1. Summary - one paragraph about the seed paper
2. Foundational Works - the top 5 references with title, year, and significance
3. Recent Developments - the top 5 recent citing papers with title, year, and significance
4. Author Profiles - each author's name and their most notable other work

Research data:
{context}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an academic research analyst. Write clear, precise "
                    "markdown reports about scientific papers."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def run(seed_id):
    seed_paper = get_paper_payload(
        seed_id,
        "title,abstract,year,authors,fieldsOfStudy,citationCount",
    )

    paper = get_paper_payload(
        seed_id,
        (
            "references,references.title,references.year,"
            "references.abstract,references.citationCount"
        ),
    )
    references = sorted(
        [ref for ref in paper["references"] if ref["citationCount"] is not None],
        key=lambda ref: ref["citationCount"],
        reverse=True,
    )[:5]

    cites = call_tool(
        "get_citations",
        {
            "paper_id": seed_id,
            "fields": "title,year,abstract,citationCount",
            "publication_date_range": "2023-01-01:",
            "limit": 5,
        },
    )
    citation_results = result_payload(cites)
    citations = [item["citingPaper"] for item in citation_results[:5]]

    author_profiles = []
    for author in seed_paper["authors"]:
        papers = call_tool(
            "get_author_papers",
            {
                "author_id": author["authorId"],
                "paper_fields": "title,year,citationCount",
            },
        )
        author_papers = result_payload(papers)
        works = sorted(
            author_papers,
            key=lambda paper_item: paper_item["citationCount"],
            reverse=True,
        )
        if works and works[0]["title"] == seed_paper["title"] and len(works) > 1:
            top_paper = works[1]
        else:
            top_paper = works[0] if works else None
        author_profiles.append({"name": author["name"], "top_paper": top_paper})

    return generate_report(seed_paper, references, citations, author_profiles)


if __name__ == "__main__":
    seed_id = sys.argv[1] if len(sys.argv) > 1 else "ARXIV:2210.03629"
    print(f"Seed paper: {seed_id}")
    print("\n" + "=" * 60)
    print("AGENT REPORT")
    print("=" * 60 + "\n")
    print(run(seed_id))
