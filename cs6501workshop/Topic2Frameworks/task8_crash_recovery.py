# langgraph_multi_agent_chat.py
# Program demonstrates LangGraph with multi-agent conversation using chat history.
# It maintains conversation context with three participants: Human, Llama, and Qwen.
# User can switch between models by prefixing input with "Hey Qwen" or "Hey Llama".
# The chat history uses role attribution to distinguish between the three participants.

# Import necessary libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver


def get_device():
    """
    Detect and return the best available compute device.
    Returns 'cuda' for NVIDIA GPUs, 'mps' for Apple Silicon, or 'cpu' as fallback.
    """
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU) for inference")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) for inference")
        return "mps"
    else:
        print("Using CPU for inference")
        return "cpu"

# =============================================================================
# STATE DEFINITION WITH MULTI-AGENT SUPPORT
# =============================================================================

class AgentState(TypedDict):
    """
    State object for multi-agent conversation.

    Fields:
    - messages: List of messages representing the full conversation history
                Messages include speaker attribution (Human:, Llama:, Qwen:)
    - user_input: The text entered by the user
    - should_exit: Boolean flag indicating if user wants to quit
    - verbose: Boolean flag controlling tracing output
    - reprompt: Boolean flag indicating whether to re-prompt for input
    - active_model: String indicating which model to use ("llama" or "qwen")
    
    Message format in history:
    - Human messages: {role: "user", content: "Human: <text>"}
    - Llama messages: {role: "assistant", content: "Llama: <text>"} (when Llama responds)
                      {role: "user", content: "Llama: <text>"} (when Qwen is active)
    - Qwen messages: {role: "assistant", content: "Qwen: <text>"} (when Qwen responds)
                     {role: "user", content: "Qwen: <text>"} (when Llama is active)
    """
    messages: Annotated[Sequence[AnyMessage], add_messages]
    user_input: str
    should_exit: bool
    verbose: bool
    reprompt: bool
    active_model: str  # "llama" or "qwen"

def create_llm(model_id: str, model_name: str):
    """Create and configure an LLM from HuggingFace."""
    device = get_device()
    print(f"Loading {model_name} model: {model_id}")
    print("This may take a moment on first run as the model is downloaded...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
    )

    if device == "mps":
        model = model.to(device)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    print(f"{model_name} model loaded successfully!")
    return llm

def create_graph(llama_llm, qwen_llm, checkpointer):
    """
    Create the LangGraph state graph with multi-agent chat support.
    
    The graph maintains a shared conversation history where:
    - Human inputs are prefixed with "Human:"
    - Llama responses are prefixed with "Llama:"
    - Qwen responses are prefixed with "Qwen:"
    
    Each model sees the other's responses as "user" role messages,
    and sees its own previous responses as "assistant" role messages.
    """

    # =========================================================================
    # NODE 1: get_user_input
    # =========================================================================
    def get_user_input(state: AgentState) -> dict:
        """
        Prompts user for input and determines which model to activate.
        Adds human input to message history with "Human:" prefix.
        """
        verbose = state.get("verbose", False)
        
        if verbose:
            print("\n[TRACE] Entering node: get_user_input")
            print(f"[TRACE] Message history length: {len(state.get('messages', []))} messages")
        
        print("\n" + "=" * 70)
        print("Enter your text (or 'quit' to exit, 'verbose'/'quiet' to toggle tracing):")
        print("Tip: Start with 'Hey Qwen' or 'Hey Llama' to address a specific model")
        print("=" * 70)

        print("\n> ", end="")
        user_input = input()

        # Check for special commands
        if user_input.lower() in ['quit', 'exit', 'q']:
            if verbose:
                print("[TRACE] User requested exit")
            print("Goodbye!")
            return {
                "user_input": user_input,
                "should_exit": True,
                "verbose": verbose,
                "reprompt": False,
            }
        
        if user_input.lower() == 'verbose':
            print("Verbose mode enabled - tracing information will be displayed")
            return {
                "user_input": user_input,
                "should_exit": False,
                "verbose": True,
                "reprompt": True,
            }
        
        if user_input.lower() == 'quiet':
            if verbose:
                print("[TRACE] Disabling verbose mode")
            print("Quiet mode enabled - tracing information will be hidden")
            return {
                "user_input": user_input,
                "should_exit": False,
                "verbose": False,
                "reprompt": True,
            }

        if user_input.strip() == "":
            if verbose:
                print("[TRACE] Empty input received — reprompting for input")
            return {
                "user_input": "",
                "should_exit": False,
                "reprompt": True,
                "verbose": verbose,
            }
        
        # Determine which model to activate based on input
        user_input_lower = user_input.strip().lower()
        if user_input_lower.startswith("hey qwen"):
            active_model = "qwen"
            # Strip the "Hey Qwen" prefix for cleaner conversation
            clean_input = user_input.strip()[8:].lstrip(" ,:-")
        elif user_input_lower.startswith("hey llama"):
            active_model = "llama"
            # Strip the "Hey Llama" prefix for cleaner conversation
            clean_input = user_input.strip()[9:].lstrip(" ,:-")
        else:
            # Default to the currently active model, or llama if none set
            active_model = state.get("active_model", "llama")
            clean_input = user_input
        
        if verbose:
            print(f"[TRACE] Valid user input received: '{user_input}'")
            print(f"[TRACE] Active model: {active_model}")
            print(f"[TRACE] Adding HumanMessage with 'Human:' prefix")
        else:
            print(f"\n→ Addressing: {active_model.upper()}")
        
        # Add human message with "Human:" prefix
        # This is always a "user" role message
        return {
            "user_input": clean_input,
            "should_exit": False,
            "verbose": verbose,
            "reprompt": False,
            "active_model": active_model,
            "messages": [HumanMessage(content=f"Human: {clean_input}")],
        }

    # =========================================================================
    # NODE 2: call_llama
    # =========================================================================
    def call_llama(state: AgentState) -> dict:
        """
        Invokes Llama with conversation history.
        
        Converts messages so that:
        - Human messages appear as "user" role with "Human:" prefix
        - Llama's own messages appear as "assistant" role with "Llama:" prefix
        - Qwen's messages appear as "user" role with "Qwen:" prefix
        """
        verbose = state.get("verbose", False)
        messages = state.get("messages", [])
        user_input = state.get("user_input", "")
        
        if verbose:
            print("\n[TRACE] Entering node: call_llama")
            print(f"[TRACE] Processing {len(messages)} messages for Llama")
        
        # Create system message for Llama
        system_prompt = (
            "You are Llama. Participants are Human, Llama, and Qwen. "
            "The conversation so far is shown below with prefixes 'Human:', 'Llama:', 'Qwen:'. "
            "You MUST reply with exactly ONE line. "
            "That line MUST start with 'Llama: ' followed by your answer. "
            "Do NOT write any other speaker lines (no 'Human:' or 'Qwen:'). "
            "Do NOT continue the conversation beyond your one line."
        )

        
        # Build prompt for Llama
        prompt_parts = [f"System: {system_prompt}"]
        
        for msg in messages:
            content = msg.content
            
            # Determine the role based on who is speaking
            if content.startswith("Human:") or content.startswith("Qwen:"):
                # Other participants are "user" role for Llama
                prompt_parts.append(f"User: {content}")
            elif content.startswith("Llama:"):
                # Llama's own previous messages are "assistant" role
                prompt_parts.append(f"Assistant: {content}")
            else:
                # Fallback
                prompt_parts.append(f"User: {content}")
        
        prompt = "\n".join(prompt_parts) + "\nAssistant: Llama:"
        
        if verbose:
            print("\n[TRACE] ----- PROMPT SENT TO LLAMA -----")
            print(prompt)
            print("[TRACE] ----- END PROMPT -----\n")

        
        print("\nProcessing with Llama...")
        
        # Invoke Llama
        full_response = llama_llm.invoke(prompt)
        
        # Extract only the new content
        if full_response.startswith(prompt):
            response = full_response[len(prompt):].strip()
        else:
            parts = full_response.split("\nAssistant: Llama:")
            response = parts[-1].strip() if len(parts) > 1 else full_response.strip()
        
        if verbose:
            print(f"[TRACE] Llama response: '{response[:100]}...'")
            print("[TRACE] Adding HumanMessage with 'Llama:' prefix")
        
        # Add Llama's response as a HumanMessage (user role) with "Llama:" prefix
        # This ensures all conversation participants use the "user" role in storage
        return {
            "messages": [HumanMessage(content=f"Llama: {response}")]
        }

    # =========================================================================
    # NODE 3: call_qwen
    # =========================================================================
    def call_qwen(state: AgentState) -> dict:
        """
        Invokes Qwen with conversation history.
        
        Converts messages so that:
        - Human messages appear as "user" role with "Human:" prefix
        - Qwen's own messages appear as "assistant" role with "Qwen:" prefix
        - Llama's messages appear as "user" role with "Llama:" prefix
        """
        verbose = state.get("verbose", False)
        messages = state.get("messages", [])
        user_input = state.get("user_input", "")
        
        if verbose:
            print("\n[TRACE] Entering node: call_qwen")
            print(f"[TRACE] Processing {len(messages)} messages for Qwen")
        
        # Create system message for Qwen
        system_prompt = (
            "You are Qwen. Participants are Human, Llama, and Qwen. "
            "The conversation so far is shown below with prefixes 'Human:', 'Llama:', 'Qwen:'. "
            "You MUST reply with exactly ONE line. "
            "That line MUST start with 'Qwen: ' followed by your answer. "
            "Do NOT write any other speaker lines (no 'Human:' or 'Llama:'). "
            "Do NOT continue the conversation beyond your one line."
        )

        
        # Build prompt for Qwen
        prompt_parts = [f"System: {system_prompt}"]
        
        for msg in messages:
            content = msg.content
            
            # Determine the role based on who is speaking
            if content.startswith("Human:") or content.startswith("Llama:"):
                # Other participants are "user" role for Qwen
                prompt_parts.append(f"User: {content}")
            elif content.startswith("Qwen:"):
                # Qwen's own previous messages are "assistant" role
                prompt_parts.append(f"Assistant: {content}")
            else:
                # Fallback
                prompt_parts.append(f"User: {content}")
        
        prompt = "\n".join(prompt_parts) + "\nAssistant: Qwen:"
        
        if verbose:
          print("\n[TRACE] ----- PROMPT SENT TO QWEN -----")
          print(prompt)
          print("[TRACE] ----- END PROMPT -----\n")

        print("\nProcessing with Qwen...")
        
        # Invoke Qwen
        full_response = qwen_llm.invoke(prompt)
        
        # Extract only the new content
        if full_response.startswith(prompt):
            response = full_response[len(prompt):].strip()
        else:
            parts = full_response.split("\nAssistant: Qwen:")
            response = parts[-1].strip() if len(parts) > 1 else full_response.strip()
        
        if verbose:
            print(f"[TRACE] Qwen response: '{response[:100]}...'")
            print("[TRACE] Adding HumanMessage with 'Qwen:' prefix")
        
        # KEEP ONLY FIRST LINE (one turn)
        response = response.splitlines()[0].strip()

        # REMOVE repeated Qwen prefixes
        while response.lower().startswith("qwen:"):
            response = response.split(":", 1)[1].strip()

        # Add Qwen's response as a HumanMessage (user role) with "Qwen:" prefix
        # This ensures all conversation participants use the "user" role in storage
        return {
            "messages": [HumanMessage(content=f"Qwen: {response}")]
        }

    # =========================================================================
    # NODE 4: print_response
    # =========================================================================
    def print_response(state: AgentState) -> dict:
        """Prints the most recent AI response."""
        verbose = state.get("verbose", False)
        messages = state.get("messages", [])
        
        # Find the most recent message that's not from Human
        last_response = None
        for msg in reversed(messages):
            content = msg.content
            if content.startswith("Llama:") or content.startswith("Qwen:"):
                last_response = content
                break
        
        if verbose:
            print("\n[TRACE] Entering node: print_response")
            print(f"[TRACE] Total messages in history: {len(messages)}")
        
        print("\n" + "=" * 70)
        print("RESPONSE:")
        print("=" * 70)
        if last_response:
            print(last_response)
        else:
            print("(No response found)")

        if verbose:
            print("\n[TRACE] Response printed to stdout")
            print("[TRACE] Looping back to get_user_input")

        return {}

    # =========================================================================
    # ROUTING FUNCTION
    # =========================================================================
    def route_after_input(state: AgentState) -> str:
        """Routes to the appropriate model based on active_model flag."""
        verbose = state.get("verbose", False)
        
        if verbose:
            print("\n[TRACE] Routing decision after get_user_input")
        
        if state.get("should_exit", False):
            if verbose:
                print("[TRACE] Routing to: END")
            return END

        if state.get("reprompt", False):
            if verbose:
                print("[TRACE] Reprompt flag set, routing to: get_user_input")
            return "get_user_input"

        # Route based on active_model
        active_model = state.get("active_model", "llama")
        if active_model == "qwen":
            if verbose:
                print("[TRACE] Routing to: call_qwen")
            return "call_qwen"
        else:
            if verbose:
                print("[TRACE] Routing to: call_llama")
            return "call_llama"

    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llama", call_llama)
    graph_builder.add_node("call_qwen", call_qwen)
    graph_builder.add_node("print_response", print_response)

    graph_builder.add_edge(START, "get_user_input")

    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "call_llama": "call_llama",
            "call_qwen": "call_qwen",
            "get_user_input": "get_user_input",
            END: END
        }
    )

    graph_builder.add_edge("call_llama", "print_response")
    graph_builder.add_edge("call_qwen", "print_response")
    graph_builder.add_edge("print_response", "get_user_input")

    graph = graph_builder.compile(checkpointer=checkpointer)

    return graph

def save_graph_image(graph, filename="lg_graph.png"):
    """Generate a Mermaid diagram of the graph and save it as a PNG image."""
    try:
        png_data = graph.get_graph(xray=True).draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_data)
        print(f"Graph image saved to {filename}")
    except Exception as e:
        print(f"Could not save graph image: {e}")
        print("You may need to install additional dependencies: pip install grandalf")

def main():
    """
    Main function for multi-agent conversation system.
    
    Maintains a shared conversation history between Human, Llama, and Qwen.
    Users can switch between models mid-conversation.
    """
    THREAD_ID = "default"
    CHECKPOINT_DB = "checkpoints.db"

    print("=" * 70)
    print("LangGraph Multi-Agent Chat: Human + Llama + Qwen")
    print("=" * 70)
    print()

    print("Loading Llama model...")
    llama_llm = create_llm("meta-llama/Llama-3.2-1B-Instruct", "Llama")
    
    print("\nLoading Qwen model...")
    qwen_llm = create_llm("Qwen/Qwen2.5-0.5B-Instruct", "Qwen")

    config = {"configurable": {"thread_id": THREAD_ID}}

    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        print("\nCreating LangGraph with multi-agent support...")
        graph = create_graph(llama_llm, qwen_llm, checkpointer)
        print("Graph created successfully!")

        print("\nSaving graph visualization...")
        save_graph_image(graph)

        initial_state: AgentState = {
            "messages": [],
            "user_input": "",
            "should_exit": False,
            "verbose": False,
            "reprompt": False,
            "active_model": "llama",
        }

        # Resume or start
        state = graph.get_state(config)
        print(f"[DEBUG] next={state.next} history_len={len(state.values.get('messages', []))}")

        if state.next:
            print("🔄 Resuming after restart...")
            graph.invoke(None, config=config)
        else:
            print("▶️ Starting fresh...")
            graph.invoke(initial_state, config=config)


if __name__ == "__main__":
    main()