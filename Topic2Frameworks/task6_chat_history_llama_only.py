# langgraph_with_history.py
# Program demonstrates use of LangGraph with chat history using the Message API.
# It maintains conversation context across multiple turns.
# The LLM should use Cuda if available, if not then if mps is available then use that,
# otherwise use cpu.
# After the LangGraph graph is created but before it executes, the program
# uses the Mermaid library to write a image of the graph to the file lg_graph.png
# The program gets the LLM from Hugging Face and wraps it for LangChain using HuggingFacePipeline.
# The code is commented in detail so a reader can understand each step.
# Supports verbose mode: type "verbose" to enable tracing, "quiet" to disable.

# Import necessary libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage
from langgraph.graph.message import add_messages


# Determine the best available device for inference
# Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
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
# STATE DEFINITION WITH MESSAGE HISTORY
# =============================================================================
# The state now uses the Message API to maintain chat history.
# The 'messages' field uses the add_messages reducer to automatically append new messages.

class AgentState(TypedDict):
    """
    State object that flows through the LangGraph nodes.

    Fields:
    - messages: List of messages (SystemMessage, HumanMessage, AIMessage) representing chat history
                Uses add_messages reducer to automatically append messages to the list
    - user_input: The text entered by the user (set by get_user_input node)
    - should_exit: Boolean flag indicating if user wants to quit (set by get_user_input node)
    - verbose: Boolean flag controlling tracing output (set by get_user_input node)
    - reprompt: Boolean flag indicating whether to re-prompt for input

    State Flow:
    1. Initial state: messages contains only system message, other fields empty/default
    2. After get_user_input: user_input, should_exit, verbose, and reprompt are populated
                             HumanMessage is added to messages list
    3. After call_llm: AIMessage with response is added to messages list
    4. After print_response: state unchanged (node only reads, doesn't write)

    The messages list grows over time, maintaining full conversation history:
        [SystemMessage, HumanMessage, AIMessage, HumanMessage, AIMessage, ...]
    """
    messages: Annotated[Sequence[AnyMessage], add_messages]
    user_input: str
    should_exit: bool
    verbose: bool
    reprompt: bool

def create_llm(model_id: str, model_name: str):
    """
    Create and configure an LLM using HuggingFace's transformers library.
    Downloads the specified model from HuggingFace Hub and wraps it
    for use with LangChain via HuggingFacePipeline.
    
    Args:
        model_id: The HuggingFace model identifier (e.g., "meta-llama/Llama-3.2-1B-Instruct")
        model_name: A friendly name for logging (e.g., "Llama")
    """
    # Get the optimal device for inference
    device = get_device()

    print(f"Loading {model_name} model: {model_id}")
    print("This may take a moment on first run as the model is downloaded...")

    # Load the tokenizer - converts text to tokens the model understands
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load the model itself with appropriate settings for the device
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
    )

    # Move model to MPS device if using Apple Silicon
    if device == "mps":
        model = model.to(device)

    # Create a text generation pipeline that combines model and tokenizer
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,  # Maximum tokens to generate in response
        do_sample=True,      # Enable sampling for varied responses
        temperature=0.7,     # Controls randomness (lower = more deterministic)
        top_p=0.95,          # Nucleus sampling parameter
        pad_token_id=tokenizer.eos_token_id,  # Suppress pad_token_id warning
    )

    # Wrap the HuggingFace pipeline for use with LangChain
    llm = HuggingFacePipeline(pipeline=pipe)

    print(f"{model_name} model loaded successfully!")
    return llm

def create_graph(llm):
    """
    Create the LangGraph state graph with chat history support:
    1. get_user_input: Reads input from stdin and adds HumanMessage to history
    2. call_llm: Sends full conversation history to LLM and adds AIMessage to history
    3. print_response: Prints the LLM's response to stdout

    Graph structure:
        START -> get_user_input -> [conditional] -> call_llm -> print_response -+
                       ^                 |                                       |
                       |                 +-> END (quit)                          |
                       |                 +-> get_user_input (reprompt)           |
                       |                                                         |
                       +---------------------------------------------------------+

    The graph maintains conversation history in the messages list, allowing the LLM
    to reference previous turns in the conversation.
    """

    # =========================================================================
    # NODE 1: get_user_input
    # =========================================================================
    def get_user_input(state: AgentState) -> dict:
        """
        Node that prompts the user for input via stdin and adds it to message history.

        Reads state: 
            - verbose: Current verbose mode setting
            - messages: Current message history
        Updates state:
            - user_input: The raw text entered by the user
            - should_exit: True if user wants to quit, False otherwise
            - verbose: Updated if user entered "verbose" or "quiet"
            - reprompt: True if empty input or mode toggle
            - messages: Appends HumanMessage with user input (if valid input)
        """
        # Get current verbose setting from state
        verbose = state.get("verbose", False)
        
        # Print tracing information if verbose mode is enabled
        if verbose:
            print("\n[TRACE] Entering node: get_user_input")
            print(f"[TRACE] Current state - verbose: {verbose}")
            print(f"[TRACE] Message history length: {len(state.get('messages', []))} messages")
        
        # Display banner before each prompt
        print("\n" + "=" * 50)
        print("Enter your text (or 'quit' to exit, 'verbose'/'quiet' to toggle tracing):")
        print("=" * 50)

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
        
        # Handle verbose mode toggle
        if user_input.lower() == 'verbose':
            print("Verbose mode enabled - tracing information will be displayed")
            if verbose:
                print("[TRACE] Verbose mode was already enabled")
            return {
                "user_input": user_input,
                "should_exit": False,
                "verbose": True,
                "reprompt": True,
            }
        
        # Handle quiet mode toggle
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

        # If the user submitted an empty line, don't call the LLM — reprompt.
        if user_input.strip() == "":
            if verbose:
                print("[TRACE] Empty input received — reprompting for input")
            return {
                "user_input": "",
                "should_exit": False,
                "reprompt": True,
                "verbose": verbose,
            }
        
        # Valid input - add HumanMessage to conversation history
        if verbose:
            print(f"[TRACE] Valid user input received: '{user_input}'")
            print(f"[TRACE] Adding HumanMessage to conversation history")
            print("[TRACE] Exiting node: get_user_input")
        
        # Return the user input and add it to messages as a HumanMessage
        # The add_messages reducer will automatically append this to the messages list
        return {
            "user_input": user_input,
            "should_exit": False,
            "verbose": verbose,
            "reprompt": False,
            "messages": [HumanMessage(content=user_input)],
        }

    # =========================================================================
    # NODE 2: call_llm
    # =========================================================================
    def call_llm(state: AgentState) -> dict:
        """
        Node that invokes the LLM with the full conversation history.

        Reads state:
            - messages: The full conversation history to send to the LLM
            - verbose: Whether to print tracing information
        Updates state:
            - messages: Appends AIMessage with the LLM's response
        """
        verbose = state.get("verbose", False)
        messages = state.get("messages", [])
        
        if verbose:
            print("\n[TRACE] Entering node: call_llm")
            print(f"[TRACE] Sending {len(messages)} messages to LLM")
            for i, msg in enumerate(messages):
                msg_type = type(msg).__name__
                content_preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
                print(f"[TRACE]   Message {i}: {msg_type} - '{content_preview}'")

        # Convert messages to the format expected by the LLM
        # Format: "System: ...\nUser: ...\nAssistant: ...\nUser: ...\nAssistant:"
        prompt_parts = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                prompt_parts.append(f"System: {msg.content}")
            elif isinstance(msg, HumanMessage):
                prompt_parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                prompt_parts.append(f"Assistant: {msg.content}")
        
        # Add the Assistant prompt at the end to get the response
        prompt = "\n".join(prompt_parts) + "\nAssistant:"
        
        if verbose:
            print(f"[TRACE] Formatted prompt ({len(prompt)} chars):")
            print(f"[TRACE] {prompt[:200]}..." if len(prompt) > 200 else f"[TRACE] {prompt}")
        
        print("\nProcessing with Llama (with conversation history)...")

        # Invoke the LLM and get the response
        # The model returns the full text including the prompt, so we need to extract only the new content
        full_response = llm.invoke(prompt)
        
        # Extract only the assistant's response by removing the prompt
        # The response starts after the last "Assistant:" in the prompt
        if full_response.startswith(prompt):
            # Remove the prompt to get only the new generation
            response = full_response[len(prompt):].strip()
        else:
            # Fallback: try to find where the new content starts
            # Look for the content after the final "Assistant:" 
            assistant_marker = "\nAssistant:"
            if assistant_marker in full_response:
                # Split and take everything after the last "Assistant:"
                parts = full_response.split(assistant_marker)
                response = parts[-1].strip()
            else:
                # If we can't parse it, use the full response
                response = full_response.strip()

        if verbose:
            print(f"[TRACE] LLM response received (length: {len(response)} chars)")
            print("[TRACE] Adding AIMessage to conversation history")
            print("[TRACE] Exiting node: call_llm")

        # Return the response as an AIMessage
        # The add_messages reducer will automatically append this to the messages list
        return {
            "messages": [AIMessage(content=response)]
        }

    # =========================================================================
    # NODE 3: print_response
    # =========================================================================
    def print_response(state: AgentState) -> dict:
        """
        Node that prints the most recent LLM response to stdout.

        Reads state:
            - messages: The conversation history (we'll print the last AIMessage)
            - verbose: Whether to print tracing information
        Updates state:
            - Nothing (returns empty dict, state unchanged)
        """
        verbose = state.get("verbose", False)
        messages = state.get("messages", [])
        
        # Find the most recent AIMessage
        last_ai_message = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                last_ai_message = msg
                break
        
        if verbose:
            print("\n[TRACE] Entering node: print_response")
            print(f"[TRACE] Total messages in history: {len(messages)}")
            if last_ai_message:
                print(f"[TRACE] Response length: {len(last_ai_message.content)} characters")
        
        print("\n" + "=" * 70)
        print("LLAMA 3.2 1B RESPONSE:")
        print("=" * 70)
        if last_ai_message:
            print(last_ai_message.content)
        else:
            print("(No response found)")

        if verbose:
            print("\n[TRACE] Response printed to stdout")
            print("[TRACE] Exiting node: print_response")
            print("[TRACE] Looping back to get_user_input")

        return {}

    # =========================================================================
    # ROUTING FUNCTION
    # =========================================================================
    def route_after_input(state: AgentState) -> str:
        """
        Routing function that determines the next node based on state.

        Examines state:
            - should_exit: If True, terminate the graph
            - reprompt: If True, loop back to get_user_input
            - verbose: For tracing output

        Returns:
            - "__end__": If user wants to quit
            - "get_user_input": If reprompt (empty input or mode toggle)
            - "call_llm": If valid input to process
        """
        verbose = state.get("verbose", False)
        
        if verbose:
            print("\n[TRACE] Routing decision after get_user_input")
        
        # Check if user wants to exit
        if state.get("should_exit", False):
            if verbose:
                print("[TRACE] Routing to: END")
            return END

        # If get_user_input indicated reprompt (empty input or mode toggle), route back
        if state.get("reprompt", False):
            if verbose:
                print("[TRACE] Reprompt flag set, routing to: get_user_input")
            return "get_user_input"

        # Valid input: Proceed to LLM
        if verbose:
            print("[TRACE] Valid input, routing to: call_llm")
        return "call_llm"

    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================
    # Create a StateGraph with our defined state structure
    graph_builder = StateGraph(AgentState)

    # Add all nodes to the graph
    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llm", call_llm)
    graph_builder.add_node("print_response", print_response)

    # Define edges:
    # 1. START -> get_user_input (always start by getting user input)
    graph_builder.add_edge(START, "get_user_input")

    # 2. get_user_input -> [conditional] -> call_llm OR get_user_input OR END
    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "call_llm": "call_llm",              # Valid input -> process with LLM
            "get_user_input": "get_user_input",  # Reprompt -> loop back
            END: END                              # Quit -> terminate
        }
    )

    # 3. call_llm -> print_response
    graph_builder.add_edge("call_llm", "print_response")

    # 4. print_response -> get_user_input (loop back for next input)
    graph_builder.add_edge("print_response", "get_user_input")

    # Compile the graph into an executable form
    graph = graph_builder.compile()

    return graph

def save_graph_image(graph, filename="lg_graph.png"):
    """
    Generate a Mermaid diagram of the graph and save it as a PNG image.
    Uses the graph's built-in Mermaid export functionality.
    """
    try:
        # Get the Mermaid PNG representation of the graph
        # This requires the 'grandalf' package for rendering
        png_data = graph.get_graph(xray=True).draw_mermaid_png()

        # Write the PNG data to file
        with open(filename, "wb") as f:
            f.write(png_data)

        print(f"Graph image saved to {filename}")
    except Exception as e:
        print(f"Could not save graph image: {e}")
        print("You may need to install additional dependencies: pip install grandalf")

def main():
    """
    Main function that orchestrates the chat agent with history:
    1. Initialize the LLM
    2. Create the LangGraph with message history support
    3. Save the graph visualization
    4. Run the graph (it loops internally until user quits)

    The graph maintains conversation history using the Message API:
    - get_user_input: Prompts, reads from stdin, adds HumanMessage to history
    - call_llm: Processes full message history, adds AIMessage to history
    - print_response: Displays the response
    - Loop back to get_user_input

    The conversation history allows the LLM to reference previous exchanges.
    
    Special commands:
    - Type "verbose" to enable tracing information in all nodes
    - Type "quiet" to disable tracing information
    """
    print("=" * 70)
    print("LangGraph Chat Agent with Conversation History")
    print("Using Llama 3.2 1B with Message API")
    print("=" * 70)
    print()

    # Step 1: Create and configure the LLM
    print("Loading Llama model...")
    llm = create_llm("meta-llama/Llama-3.2-1B-Instruct", "Llama")

    # Step 2: Build the LangGraph with message history
    print("\nCreating LangGraph with conversation history support...")
    graph = create_graph(llm)
    print("Graph created successfully!")

    # Step 3: Save a visual representation of the graph before execution
    print("\nSaving graph visualization...")
    save_graph_image(graph)

    # Step 4: Run the graph - it will loop internally until user quits
    # Initialize with a system message that sets the assistant's behavior
    initial_state: AgentState = {
        "messages": [
            SystemMessage(content="You are a helpful AI assistant. You maintain context from previous messages in the conversation.")
        ],
        "user_input": "",
        "should_exit": False,
        "verbose": False,
        "reprompt": False,
    }

    print("\n" + "=" * 70)
    print("Chat session started. The assistant will remember your conversation.")
    print("Try asking follow-up questions to test the memory!")
    print("=" * 70)

    # Single invocation - the graph loops internally
    # Message history is maintained across all turns
    graph.invoke(initial_state)

# Entry point - only run main() if this script is executed directly
if __name__ == "__main__":
    main()