# langgraph_vision_chat.py
# Exercise 1 — Vision-Language LangGraph Chat Agent (LLaVA via Ollama)

import argparse
import os
from typing import Annotated, Optional, Sequence, TypedDict

import ollama
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from PIL import Image

# ── optional Gradio ────────────────────────────────────────────────────────────
try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
MODEL = "llava:7b-v1.6-mistral-q4_0"
MAX_SIDE         = 216          # resize long edge to this (set None to skip)
CONTEXT_MESSAGES = 2           # rolling window: last N messages sent to LLaVA
CHECKPOINT_DB    = "vlm_checkpoints.db"
THREAD_ID        = "vlm_thread"

SYSTEM_PROMPT = (
    "You are a helpful vision-language assistant. "
    "Answer the user's questions about the provided image. "
    "Use the conversation history for context. Be concise but specific."
)


# ──────────────────────────────────────────────────────────────────────────────
# STATE
# ──────────────────────────────────────────────────────────────────────────────

class VLMState(TypedDict):
    """
    Full agent state — owned by LangGraph and persisted via SqliteSaver.

    messages    : Conversation history (LangChain message objects).
    user_input  : Latest text submitted by the user.
    image_path  : File path to the current image (set once, reused each turn).
    should_exit : CLI exit flag.
    verbose     : Tracing toggle.
    reprompt    : Skip model call and re-ask for input (CLI use).
    """
    messages:    Annotated[Sequence[AnyMessage], add_messages]
    user_input:  str
    image_path:  Optional[str]
    should_exit: bool
    verbose:     bool
    reprompt:    bool


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

_resize_cache: dict[str, str] = {}   # original_path → resized_path

def maybe_resize(image_path: str, max_side: int = MAX_SIDE) -> str:
    """
    Downscale image so its longest side <= max_side.
    Result is cached in memory so the file is written only once per session,
    even across many chat turns.
    Returns the original path unchanged if already small enough.
    """
    cache_key = f"{image_path}@{max_side}"
    if cache_key in _resize_cache:
        return _resize_cache[cache_key]

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) <= max_side:
        _resize_cache[cache_key] = image_path
        return image_path

    scale  = max_side / max(w, h)
    img    = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    out    = image_path.rsplit(".", 1)[0] + f"_resized_{max_side}.jpg"
    img.save(out, format="JPEG", quality=90)
    _resize_cache[cache_key] = out
    return out


def build_ollama_messages(messages: Sequence[AnyMessage],
                           image_path: str,
                           context_n: int = CONTEXT_MESSAGES) -> list:
    """
    Convert LangChain message history into Ollama's message format.
    - System prompt is always prepended.
    - Image is attached to the LATEST (most recent) user message only.
      This ensures LLaVA always sees the image alongside the current question
      without relying on KV-cache persistence across stateless Ollama calls.
    - Only the last `context_n` messages are included to cap token usage.
    """
    recent = list(messages)[-context_n:]

    # Index of the last HumanMessage — that's where the image goes
    last_human_idx = max(
        (i for i, m in enumerate(recent) if isinstance(m, HumanMessage)),
        default=None,
    )

    ollama_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]

    for i, m in enumerate(recent):
        if isinstance(m, HumanMessage):
            entry: dict = {"role": "user", "content": m.content}
            if image_path and i == last_human_idx:
                entry["images"] = [image_path]   # latest turn only
            ollama_msgs.append(entry)
        elif isinstance(m, AIMessage):
            ollama_msgs.append({"role": "assistant", "content": m.content})

    return ollama_msgs


# ──────────────────────────────────────────────────────────────────────────────
# NODE 1 — ingest_user_turn
# ──────────────────────────────────────────────────────────────────────────────

def ingest_user_turn(state: VLMState) -> dict:
    """
    Validates user input and appends a HumanMessage to history.
    Handles special commands (quit / verbose / quiet / empty).
    In Gradio mode, user_input is pre-populated before invoke().
    In CLI mode, input arrives via get_cli_input node.
    """
    verbose = state.get("verbose", False)
    text    = (state.get("user_input") or "").strip()

    if verbose:
        print(f"\n[TRACE] ingest_user_turn  user_input='{text}'")

    if text.lower() in ("quit", "exit", "q"):
        return {"should_exit": True, "reprompt": False}

    if text.lower() == "verbose":
        print("Verbose mode ON")
        return {"verbose": True, "reprompt": True}

    if text.lower() == "quiet":
        return {"verbose": False, "reprompt": True}

    if text == "":
        return {"reprompt": True}

    return {
        "reprompt": False,
        "messages": [HumanMessage(content=text)],
    }


# ──────────────────────────────────────────────────────────────────────────────
# NODE 2 — call_llava
# ──────────────────────────────────────────────────────────────────────────────

def call_llava(state: VLMState) -> dict:
    """
    Builds the Ollama message list from the rolling history and calls LLaVA.
    Resizes the image if needed before sending.
    """
    verbose  = state.get("verbose", False)
    img_path = state.get("image_path") or ""
    messages = state.get("messages", [])

    if verbose:
        print(f"\n[TRACE] call_llava  messages={len(messages)}  image={img_path}")

    if not img_path or not os.path.isfile(img_path):
        return {"messages": [AIMessage(content="Please upload an image first.")]}

    img_to_use  = maybe_resize(img_path, MAX_SIDE) if MAX_SIDE else img_path
    ollama_msgs = build_ollama_messages(messages, img_to_use)

    if verbose:
        print(f"[TRACE] Sending {len(ollama_msgs)} messages to Ollama ({MODEL})")

    resp   = ollama.chat(model=MODEL, messages=ollama_msgs)
    answer = resp["message"]["content"].strip()

    if verbose:
        print(f"[TRACE] LLaVA reply: {answer[:120]}")

    return {"messages": [AIMessage(content=answer)]}


# ──────────────────────────────────────────────────────────────────────────────
# NODE 3 — print_response  (CLI only)
# ──────────────────────────────────────────────────────────────────────────────

def print_response(state: VLMState) -> dict:
    """Prints the latest AI reply to stdout (CLI mode only)."""
    for m in reversed(list(state.get("messages", []))):
        if isinstance(m, AIMessage):
            print("\n" + "─" * 60)
            print("LLaVA:", m.content)
            print("─" * 60)
            break
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# NODE 4 — get_cli_input  (CLI only)
# ──────────────────────────────────────────────────────────────────────────────

def get_cli_input(state: VLMState) -> dict:
    """
    Interactive stdin node used only in CLI mode.
    When no image is set yet, treats input as a file path.
    """
    print("\n" + "═" * 60)
    if not state.get("image_path"):
        print("No image loaded. Enter a file path to an image:")
    else:
        print("Ask a question (or 'quit' / 'verbose' / 'quiet'):")
    print("═" * 60)
    print("> ", end="", flush=True)
    text = input().strip()

    # If no image yet, treat input as file path
    if not state.get("image_path"):
        if text.lower() in ("quit", "exit", "q"):
            return {"should_exit": True, "reprompt": False}
        if not os.path.isfile(text):
            print(f"  File not found: {text}")
            return {"reprompt": True}
        print(f"Image loaded: {text}")
        return {"image_path": text, "reprompt": True, "user_input": ""}

    return {"user_input": text, "reprompt": False}


# ──────────────────────────────────────────────────────────────────────────────
# ROUTING
# ──────────────────────────────────────────────────────────────────────────────

def route_after_ingest_cli(state: VLMState) -> str:
    if state.get("should_exit"):
        return END
    if state.get("reprompt"):
        return "get_cli_input"
    return "call_llava"


def route_after_ingest_gradio(state: VLMState) -> str:
    if state.get("should_exit") or state.get("reprompt"):
        return END
    return "call_llava"


def route_after_print(state: VLMState) -> str:
    if state.get("should_exit"):
        return END
    return "get_cli_input"


# ──────────────────────────────────────────────────────────────────────────────
# GRAPH BUILDERS
# ──────────────────────────────────────────────────────────────────────────────

def build_cli_graph(checkpointer):
    """
    CLI graph: loops via get_cli_input → ingest → llava → print → …
    """
    g = StateGraph(VLMState)
    g.add_node("get_cli_input",    get_cli_input)
    g.add_node("ingest_user_turn", ingest_user_turn)
    g.add_node("call_llava",       call_llava)
    g.add_node("print_response",   print_response)

    g.add_edge(START, "get_cli_input")
    g.add_edge("get_cli_input", "ingest_user_turn")
    g.add_conditional_edges(
        "ingest_user_turn",
        route_after_ingest_cli,
        {"get_cli_input": "get_cli_input", "call_llava": "call_llava", END: END},
    )
    g.add_edge("call_llava", "print_response")
    g.add_conditional_edges(
        "print_response",
        route_after_print,
        {"get_cli_input": "get_cli_input", END: END},
    )
    return g.compile(checkpointer=checkpointer)


def build_gradio_graph(checkpointer):
    """
    Gradio graph: single-turn per invoke — START → ingest → llava → END.
    Gradio is the outer loop; SqliteSaver carries history between HTTP calls.
    """
    g = StateGraph(VLMState)
    g.add_node("ingest_user_turn", ingest_user_turn)
    g.add_node("call_llava",       call_llava)

    g.add_edge(START, "ingest_user_turn")
    g.add_conditional_edges(
        "ingest_user_turn",
        route_after_ingest_gradio,
        {"call_llava": "call_llava", END: END},
    )
    g.add_edge("call_llava", END)
    return g.compile(checkpointer=checkpointer)


# ──────────────────────────────────────────────────────────────────────────────
# GRADIO UI
# ──────────────────────────────────────────────────────────────────────────────

def launch_gradio(checkpointer):
    import gradio as gr

    graph  = build_gradio_graph(checkpointer)
    config = {"configurable": {"thread_id": THREAD_ID}}

    def _last_ai(messages) -> str:
        for m in reversed(list(messages)):
            if isinstance(m, AIMessage):
                return m.content
        return "(No response)"

    def set_image(img_filepath):
        """Reset conversation and record new image path in state."""
        if not img_filepath:
            return "No image selected.", []
        init: VLMState = {
            "messages":    [],
            "user_input":  "",
            "image_path":  img_filepath,
            "should_exit": False,
            "verbose":     False,
            "reprompt":    False,
        }
        graph.invoke(init, config=config)
        return f"Image loaded: {os.path.basename(img_filepath)}", []

    def chat(user_text: str, chat_history: list):
        """One Gradio turn = one graph invoke."""
        text = user_text.strip()
        if not text:
            return chat_history, gr.update(value="")

        current = graph.get_state(config).values or {}

        # ── Handle verbose/quiet via a proper graph invoke ─────────────────
        # We set user_input to the command and let ingest_user_turn handle it
        # (it sets verbose + reprompt=True, routing to END with no LLaVA call).
        # This keeps the toggle going through the graph like any other input,
        # so the flag is persisted correctly by the checkpointer.
        if text.lower() in ("verbose", "quiet"):
            update: VLMState = {
                **current,
                "user_input":  text,
                "should_exit": False,
                "reprompt":    False,
            }
            graph.invoke(update, config=config)
            label = "🔍 Verbose mode ON — tracing will appear in server logs." \
                    if text.lower() == "verbose" else \
                    "🔇 Quiet mode ON — tracing suppressed."
            return chat_history + [[text, reply]], gr.update(value="")  # also clears msg_box

        img_path = current.get("image_path", "")
        verbose  = current.get("verbose", False)

        update: VLMState = {
            **current,
            "user_input":  text,
            "image_path":  img_path,
            "verbose":     verbose,
            "should_exit": False,
            "reprompt":    False,
        }

        out   = graph.invoke(update, config=config)
        reply = _last_ai(out.get("messages", []))
        return chat_history + [[text, reply]], gr.update(value="")  # also clears msg_box

    # ── layout ────────────────────────────────────────────────────────────────
    with gr.Blocks(title="LLaVA Vision Chat") as demo:
        gr.Markdown(
            "## Vision-Language Chat Agent\n"
            "Upload an image, click **Load Image**, then ask questions. "
            "Conversation history is preserved across turns.\n\n"
            "_Type_ `verbose` _or_ `quiet` _in the chat to toggle tracing._"
        )
        with gr.Row():
            with gr.Column(scale=1):
                img_input  = gr.Image(type="filepath", label="Upload Image")
                set_btn    = gr.Button("Load Image", variant="primary")
                img_status = gr.Textbox(label="Status", interactive=False, lines=1)
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation", height=500)
                with gr.Row():
                    msg_box  = gr.Textbox(
                        placeholder="Ask something about the image…",
                        show_label=False, scale=8,
                    )
                    send_btn = gr.Button("Send", scale=1, variant="primary")

        set_btn.click(set_image, inputs=img_input,          outputs=[img_status, chatbot])
        send_btn.click(chat, inputs=[msg_box, chatbot], outputs=[chatbot, msg_box])
        msg_box.submit(chat, inputs=[msg_box, chatbot], outputs=[chatbot, msg_box])

    demo.launch(share=True)


# ──────────────────────────────────────────────────────────────────────────────
# CLI MODE
# ──────────────────────────────────────────────────────────────────────────────

def launch_cli(checkpointer, preload_image: str = ""):
    graph  = build_cli_graph(checkpointer)
    config = {"configurable": {"thread_id": THREAD_ID}}

    print("=" * 60)
    print("  Vision-Language Chat Agent  (LLaVA via Ollama)")
    print("  Commands: verbose | quiet | quit")
    print("=" * 60)

    saved = graph.get_state(config)
    if saved.next:
        print("Resuming previous session...")
        graph.invoke(None, config=config)
        return

    init: VLMState = {
        "messages":    [],
        "user_input":  "",
        "image_path":  preload_image if os.path.isfile(preload_image) else "",
        "should_exit": False,
        "verbose":     False,
        "reprompt":    False,
    }
    if preload_image and os.path.isfile(preload_image):
        print(f"Pre-loaded image: {preload_image}")

    graph.invoke(init, config=config)
    print("\nGoodbye!")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLaVA Vision-Language LangGraph Chat")
    parser.add_argument(
        "--ui", choices=["gradio", "cli"],
        default="gradio" if HAS_GRADIO else "cli",
        help="Interface mode (default: gradio if installed, else cli)",
    )
    parser.add_argument(
        "--image", default="",
        help="Pre-load an image file path (CLI mode only)",
    )
    args = parser.parse_args()

    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        if args.ui == "gradio":
            if not HAS_GRADIO:
                print("Gradio not installed — pip install gradio")
                print("Falling back to CLI mode.")
                launch_cli(checkpointer, preload_image=args.image)
            else:
                launch_gradio(checkpointer)
        else:
            launch_cli(checkpointer, preload_image=args.image)


if __name__ == "__main__":
    main()