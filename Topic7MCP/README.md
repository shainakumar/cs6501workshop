# Topic 7 MCP

## Exercise A:
To find papers about "transformer attention mechanisms", we would use the `search_papers_by_relevance` tool (written as search_papers on the lesson plan) because it is designed for keyword or semantic paper search across the corpus.

Then, to find who else published in the same area as a specific author, there a multiple approaches we could take. The most intuitive approach would be to use `get_author_papers` to explore that specific author's publications and research areas. Subsequently, we could use `search_papers_by_relevance` on the recurring topic keywords or themes to find related authors.

## Exercise B

The structure of results across the three tools was different. `search_papers_by_relevance` and `get_citations` both returned list-style results through `structuredContent["result"]`, but each item had a different shape. `search_papers_by_relevance` returned paper objects directly, so fields could be read with paths like `result[i]["title"]`. `get_citations` wrapped each paper one level deeper under a `citingPaper` key, so the title had to be read with `result[i]["citingPaper"]["title"]`.

`get_paper` was different from the list-based tools. It did not use `structuredContent` in the same way. Instead, the paper object came back as a JSON string embedded inside `content[0]["text"]`, which required a second `json.loads()` call to deserialize:

```python
paper = json.loads(data["result"]["content"][0]["text"])
```

This double-parsing pattern was needed for the `get_paper` calls. Once parsed, `get_paper` returned a single paper object that could contain nested arrays such as `references` and `authors`. On the other hand, the list-based tools could be read directly from `structuredContent["result"]`: `search_papers_by_relevance` returned paper objects, while `get_citations` returned objects shaped like `{ "citingPaper": ... }`.

## Exercise C

Compared with Exercise B, the biggest change was that we no longer manually chose every tool from the ones available in Exercise A and wrote separate logic for each research question. In Exercise B, we hard-coded calls like `search_papers_by_relevance`, `get_citations`, and `get_paper`, along with the exact arguments for each drill. However, in Exercise C, the chatbot fetched the available tool schemas from Asta at startup with `tools/list`, converted those MCP schemas into OpenAI tool definitions, and let GPT-4o mini decide which tool to call. The chabot needed a reusable `get_asta_tools()` function, a reusable `call_asta_tool(name, arguments)` function, and a loop that passed tool results back to the model. Here, the schema came from the server rather than from hand-written tool definitions. As stated in the lesson, if Asta added new tools, the chatbot would not need a new manually written Python wrapper for that specific tool. 


## Closing Discussion

Writing tool schemas by hand helped us identify what MCP automates. As we learned, normally, a developer has to read API docs, decide what the model should see, write a compact tool description, define parameters, and keep that schema updated when the API changes. MCP gives the client those schemas dynamically. This level of automation buys  us flexibility and faster integration. It also makes agents more adaptable because a compliant client can inspect a server's tools at runtime. In contrast, the cost is that there are potential points of failure. The client now depends on the MCP server's schema and retrieval quality (might not be able to retrieve tools, outputs might need further parsing, etc.). 

In Exercise C, the tool results were limited to 8,000 characters with truncation before passing them back to the model. For the retrieved data in Exercise D, the abstracts were shortened to 200 characters, and only essential fields such as title, year, authors, citation counts, and selected relationships were kept. 

To let the LLM decide the tool-calling order, we would need to give it the same Asta tool schemas, a goal, and an agent loop that allows multiple tool calls until it decides it has enough evidence. That would make the agent more flexible, but it could also go wrong by calling tools in an inefficient order, getting stuck in unnecessary searches/infinite loops, missing required steps, overusing the API, or producing a report from incomplete evidence/producing inconsistent results across reports.

We would want a more mature MCP ecosystem to possibly include things like clear descriptions of what each tool is good for, what kind of output it returns, what its limitations are, and when it should or should not be used. We would also want MCP servers to expose useful task context, such as goals, constraints, permissions, rate limits, and examples of good tool calls.

## A2A Discussion Questions

### MCP vs A2A

Sending a task to another agent is different from calling an MCP tool because an agent can reason and choose its own approach rather than being deterministic. An agent can interpret ambiguity, use its own tools, break a task into smaller steps, and decide what kind of answer is most useful depending on the scenario. However, an agent is less predictable and presents its own set of limitations. 

### Discovery

We used a central registry in the tournament so the router could see which agents were available and what each one claimed to do. An alternative option would be to include a manually configured list of agents or peer-to-peer discovery. A centralized registry is easier to implement and monitor. However, decentralized discovery might be more flexible and resilient (avoids single point of failure).  

### System Prompts as Strategy

The system prompt mattered a lot for scoring because it shaped what each agent prioritized/its overall strategy. It might be possible to craft a prompt that does well across categories while still being funny on off-topic questions, but then there would need to be clarification of what is considered "off-topic." 

### Smart Routing

TF-IDF routing works when the question uses similar language to the agent description, but semantic embeddings would probably improve routing because they match meaning (patterns) rather than exact words. If agents could self-report confidence, routing could become more adaptive/choose the strongest candidate. The risk is the lack of self-reported confidence reliability. 

### Trust and Reliability

In a real multi-agent system, we would want validation steps such as checking sources, comparing answers from multiple agents, running tests, or using a separate verifier agent. With those additional evaluation steps, agents that repeatedly return bad data should be ranked lower or removed from routing, assigning the task to another appropriate agent. If an agent is slow or goes offline mid-task, the system should communicate that partial failure clearly, followed with retries and the use of backup agents. 

### Scaling

With 1,000 agents, the current routing and health checks would become inefficient. The router would spend too much time comparing every question against every agent, and the registry would struggle to ping every agent often enough. A better design might be to store searchable agent embeddings ahead of time and have agents send regular heartbeats to show they are still online.

