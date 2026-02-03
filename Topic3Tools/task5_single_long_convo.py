"""
Single Long Conversation Tool Agent using LangGraph
- Replaces manual for-loop in run_agent with LangGraph nodes + conditional edges
- Maintains chat history across turns
- Includes checkpointing + recovery (SQLite)
"""

import json
import math
import numexpr as ne
from typing import TypedDict, Annotated, List

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver


# ============================================
# PART 1: Define Tools (same as your code)
# ============================================

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location"""
    weather_data = {
        "San Francisco": "Sunny, 72°F",
        "New York": "Cloudy, 55°F",
        "London": "Rainy, 48°F",
        "Tokyo": "Clear, 65°F",
    }
    return weather_data.get(location, f"Weather data not available for {location}")


@tool
def geo_calculator(payload: str) -> str:
    """
    Evaluate an arithmetic / geometric expression.

    Preferred input (JSON string):
      {"expression": "sin(0.5) + 3*(4-1)"}

    Fallback: if payload is not valid JSON, treat it as raw expression text.

    Supports + - * / // % **, parentheses, sin(), cos(), tan(), sqrt(), pi.
    Returns JSON string: {"ok": true, "expression": "...", "result": <number>}
    """
    try:
        expr = None
        try:
            params = json.loads(payload)
            if isinstance(params, dict):
                expr = params.get("expression")
            elif isinstance(params, str):
                expr = params
        except Exception:
            expr = payload

        if not expr or not isinstance(expr, str):
            raise ValueError("Missing/invalid expression. Use JSON {'expression': '...'} or raw expression string.")

        result = float(ne.evaluate(expr))
        return json.dumps({"ok": True, "expression": expr, "result": result})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


@tool
def count_letter(text: str, letter: str) -> str:
    """Count occurrences of a letter in a piece of text. Returns JSON string."""
    if len(letter) != 1:
        return json.dumps({"ok": False, "error": "letter must be a single character"})
    count = text.lower().count(letter.lower())
    return json.dumps({"ok": True, "text": text, "letter": letter, "count": count})


@tool
def word_count(text: str) -> str:
    """Count words in text. Returns JSON string."""
    words = [w for w in text.strip().split() if w]
    return json.dumps({"ok": True, "text": text, "count": len(words)})


TOOLS = [get_weather, geo_calculator, count_letter, word_count]
TOOL_MAP = {t.name: t for t in TOOLS}


# ============================================
# PART 2: LLM w/ Tools
# ============================================

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM = SystemMessage(
    content=(
        "You are a helpful assistant. "
        "For ANY counting, you MUST use count_letter. "
        "For ANY math (including trig/geometry/arithmetic), you MUST use geo_calculator. "
        "Use word_count when asked about number of words."
    )
)


# ============================================
# PART 3: LangGraph State
# ============================================

class AgentState(TypedDict):
    # All conversation messages persist here via add_messages reducer
    messages: Annotated[List[AnyMessage], add_messages]


# ============================================
# PART 4: Nodes
# ============================================

def agent_node(state: AgentState) -> AgentState:
    """
    Calls the LLM with the accumulated messages.
    Returns an AI message, which may include tool_calls.
    """
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def tools_node(state: AgentState) -> AgentState:
    """
    Executes tool calls from the last AI message, returns ToolMessage(s).
    """
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []

    tool_messages: List[AnyMessage] = []

    for tc in tool_calls:
        name = tc["name"]
        args = tc["args"]
        call_id = tc["id"]

        if name in TOOL_MAP:
            result = TOOL_MAP[name].invoke(args)
        else:
            result = f"Error: Unknown tool {name}"

        tool_messages.append(ToolMessage(content=str(result), tool_call_id=call_id))

    return {"messages": tool_messages}


def route_after_agent(state: AgentState) -> str:
    """
    If the last AI message has tool_calls, go run tools, otherwise finish.
    """
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    return "tools" if tool_calls else END


# ============================================
# PART 5: Build Graph + Checkpointing
# ============================================

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

def build_app(db_path: str = "langgraph_chat_checkpoints.sqlite"):
    g = StateGraph(AgentState)

    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)

    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")

    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)  # <-- this is a real BaseCheckpointSaver
    return g.compile(checkpointer=saver)



# ============================================
# PART 6: Single Long Conversation CLI
# ============================================

def chat():
    """
    Runs a single long conversation.
    No per-turn run_agent reset. History is loaded/saved via checkpointing.
    """
    app = build_app()

    thread_id = input("Enter a thread_id (e.g., shaina-demo-1): ").strip() or "demo-thread"
    config = {"configurable": {"thread_id": thread_id}}

    print("\nType 'quit' to exit.")
    print("Type 'crash' to simulate a crash and test recovery.\n")

    # We seed the system message once at the start of the thread by including it
    # in the first invocation. It will be checkpointed and persist.
    seeded = False

    while True:
        user_text = input("You: ").strip()

        if user_text.lower() == "quit":
            print("Exiting.")
            break

        if user_text.lower() == "crash":
            raise SystemExit("Simulated crash. Restart and reuse same thread_id to recover.")

        # Build the per-turn input messages (only the new human message)
        new_msgs: List[AnyMessage] = [HumanMessage(content=user_text)]
        if not seeded:
            new_msgs = [SYSTEM] + new_msgs
            seeded = True

        # Invoke the graph for this turn; it will run agent->tools->agent... until END
        result = app.invoke({"messages": new_msgs}, config=config)

        # Last message should be the final AI response for this turn
        final = result["messages"][-1]
        print(f"Assistant: {final.content}\n")


if __name__ == "__main__":
    chat()
