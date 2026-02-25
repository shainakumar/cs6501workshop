"""
Persistent Conversation Agent with LangGraph
- SqliteSaver used correctly as a context manager (with-block)
- Checkpointing and recovery via thread_id
"""

import json
import math
import numexpr as ne
import uuid

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver   # used as context manager
from typing import TypedDict


# ============================================
# PART 1: Tools
# ============================================

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    weather_data = {
        "San Francisco": "Sunny, 72F",
        "New York": "Cloudy, 55F",
        "London": "Rainy, 48F",
        "Tokyo": "Clear, 65F",
    }
    result = weather_data.get(location, f"Weather data not available for {location}")
    print(f"  [Tool] get_weather({location!r}) -> {result}")
    return result


@tool
def geo_calculator(input_json: str) -> str:
    """
    Evaluate a mathematical expression.
    input_json must be a JSON string like {"expr": "sin(3 - 1)"}.
    Supports: + - * / ** sqrt sin cos tan pi e
    """
    try:
        params = json.loads(input_json)
        expr = params.get("expr") or params.get("expression")
        if not expr:
            raise ValueError("Missing 'expr' key.")
        result = float(
            ne.evaluate(
                expr,
                local_dict={
                    "pi": math.pi, "e": math.e,
                    "sin": math.sin, "cos": math.cos,
                    "tan": math.tan, "sqrt": math.sqrt,
                },
            )
        )
        out = json.dumps({"ok": True, "expression": expr, "result": result})
        print(f"  [Tool] geo_calculator({expr!r}) -> {result}")
        return out
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)})


@tool
def count_letter(text: str, letter: str) -> str:
    """Count occurrences of a single letter in text (case-insensitive)."""
    letter = letter.strip()
    if len(letter) != 1:
        return json.dumps({"ok": False, "error": "letter must be a single character"})
    count = text.lower().count(letter.lower())
    print(f"  [Tool] count_letter({text!r}, {letter!r}) -> {count}")
    return json.dumps({"ok": True, "text": text, "letter": letter, "count": count})


@tool
def word_count(text: str) -> str:
    """Count the number of words in text."""
    words = [w for w in text.strip().split() if w]
    print(f"  [Tool] word_count({text!r}) -> {len(words)}")
    return json.dumps({"ok": True, "text": text, "count": len(words)})


TOOLS = [get_weather, geo_calculator, count_letter, word_count]
TOOL_MAP = {t.name: t for t in TOOLS}

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a helpful assistant that remembers this entire conversation. "
    "For ANY letter counting use count_letter. "
    "For word totals use word_count. "
    "For ANY math use geo_calculator with input_json like {\"expr\":\"<expression>\"}. "
    "For weather use get_weather. "
    "Never calculate anything yourself -- always delegate to the tools."
))


# ============================================
# PART 2: State
# ============================================

class AgentState(TypedDict):
    user_input:   str
    should_exit:  bool
    chat_history: list   # full message history (SystemMessage + turns)
    llm_response: object # latest AIMessage from the LLM


# ============================================
# PART 3: Build Graph
# ============================================

def build_graph(llm):
    """
    Construct the StateGraph.  Returns the builder (not yet compiled)
    so the caller can inject a checkpointer at compile time.

    Node layout:

        __start__
            |
        get_user_input  <-----------------------------------------+
            |  (quit -> END)                                       |
            |  (empty/verbose/quiet -> self)                       |
            v                                                      |
        call_llm  --(tool_calls?)--> call_tools ---(loops back)---+
            |                                                      |
            | (no tool calls)                                      |
            v                                                      |
        print_response --------------------------------------------+
    """

    llm_with_tools = llm.bind_tools(TOOLS)

    # ------------------------------------------------------------------
    # Node: get_user_input
    # ------------------------------------------------------------------
    def get_user_input(state: AgentState) -> dict:
        """Read one line from the user and update state."""
        print("\n" + "=" * 50)
        print("You (or 'quit' to exit): ", end="")
        user_input = input().strip()

        # Quit
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            return {"user_input": user_input, "should_exit": True}

        # Empty or control commands -> loop back without calling LLM
        if user_input == "" or user_input.lower() in ("verbose", "quiet"):
            return {"user_input": user_input, "should_exit": False}

        # Initialise chat history with system prompt on first real message
        chat_history = state.get("chat_history") or [SYSTEM_PROMPT]

        # Append the new human turn
        chat_history = chat_history + [HumanMessage(content=user_input)]

        return {
            "user_input":   user_input,
            "should_exit":  False,
            "chat_history": chat_history,
        }

    # ------------------------------------------------------------------
    # Node: call_llm
    # ------------------------------------------------------------------
    def call_llm(state: AgentState) -> dict:
        """Invoke the LLM with the full chat history."""
        response = llm_with_tools.invoke(state["chat_history"])
        return {
            "llm_response": response,
            "chat_history": state["chat_history"] + [response],
        }

    # ------------------------------------------------------------------
    # Node: call_tools
    # ------------------------------------------------------------------
    def call_tools(state: AgentState) -> dict:
        """Execute every tool call in the last LLM response."""
        last_msg = state["llm_response"]
        tool_messages = []

        for tc in last_msg.tool_calls:
            name, args = tc["name"], tc["args"]
            result = TOOL_MAP[name].invoke(args) if name in TOOL_MAP \
                     else f"Error: unknown tool {name}"
            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )

        return {"chat_history": state["chat_history"] + tool_messages}

    # ------------------------------------------------------------------
    # Node: print_response
    # ------------------------------------------------------------------
    def print_response(state: AgentState) -> dict:
        """Print the assistant's final answer."""
        print(f"\nAssistant: {state['llm_response'].content}")
        return {}

    # ------------------------------------------------------------------
    # Routing functions
    # ------------------------------------------------------------------
    def route_user_input(state: AgentState) -> str:
        if state.get("should_exit"):
            return END
        inp = state.get("user_input", "")
        if inp == "" or inp.lower() in ("verbose", "quiet"):
            return "get_user_input"
        return "call_llm"

    def route_after_llm(state: AgentState) -> str:
        if state["llm_response"].tool_calls:
            return "call_tools"
        return "print_response"

    # ------------------------------------------------------------------
    # Assemble graph
    # ------------------------------------------------------------------
    gb = StateGraph(AgentState)

    gb.add_node("get_user_input", get_user_input)
    gb.add_node("call_llm",       call_llm)
    gb.add_node("call_tools",     call_tools)
    gb.add_node("print_response", print_response)

    gb.add_edge(START,            "get_user_input")
    gb.add_edge("call_tools",     "call_llm")          # tools always loop back to LLM
    gb.add_edge("print_response", "get_user_input")    # after reply, get next input

    gb.add_conditional_edges(
        "get_user_input",
        route_user_input,
        {"get_user_input": "get_user_input", "call_llm": "call_llm", END: END},
    )

    gb.add_conditional_edges(
        "call_llm",
        route_after_llm,
        {"call_tools": "call_tools", "print_response": "print_response"},
    )

    return gb


# ============================================
# PART 4: Graph image
# ============================================

def save_graph_image(graph, filename="lg_graph.png"):
    try:
        png_data = graph.get_graph(xray=True).draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_data)
        print(f"Graph image saved to {filename}")
    except Exception as exc:
        print(f"Could not save graph image: {exc}")
        print("Install grandalf if needed: pip install grandalf")


# ============================================
# PART 5: Main -- SqliteSaver as context manager
# ============================================

def main():
    print("=" * 50)
    print("Persistent Conversation Agent")
    print("=" * 50)

    llm = ChatOpenAI(model="gpt-4o-mini")
    graph_builder = build_graph(llm)
    
    with SqliteSaver.from_conn_string("conversation.db") as checkpointer:

        graph = graph_builder.compile(checkpointer=checkpointer)

        # Save diagram now that the graph is compiled
        save_graph_image(graph)

        # Change THREAD_ID to start a fresh conversation,
        # or keep it the same to continue a previous one.
        THREAD_ID = "conversation_1"
        config = {"configurable": {"thread_id": THREAD_ID}}

        # --------------------------------------------------------------
        # Recovery: check whether a prior checkpoint exists for this thread
        # --------------------------------------------------------------
        existing = graph.get_state(config)

        if existing.next:
            # Graph was interrupted mid-execution (e.g. crash inside a node)
            print(f"\n  Resuming interrupted conversation (thread: {THREAD_ID})")
            print(f"  Pending node(s): {existing.next}")
            graph.invoke(None, config=config)

        else:
            history = existing.values.get("chat_history", [])
            if history:
                # Prior completed turns exist -- continue the conversation
                print(f"\n  Continuing saved conversation (thread: {THREAD_ID})")
                print(f"  {len(history)} messages already in history.")
            else:
                print(f"\n  Starting new conversation (thread: {THREAD_ID})")

            initial_state: AgentState = {
                "user_input":   "",
                "should_exit":  False,
                "chat_history": [],
                "llm_response": None,
            }
            graph.invoke(initial_state, config=config)

    print("\n[Session ended -- history saved to conversation.db]")
    print(f"Restart the script to continue as thread '{THREAD_ID}'.")


if __name__ == "__main__":
    main()