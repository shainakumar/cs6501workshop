"""
Bare-Bones Chat Agent for Llama 3.2-1B-Instruct

This is a minimal chat interface that demonstrates:
1. How to load a model without quantization
2. How chat history is maintained and fed back to the model
3. The difference between plain text history and tokenized input

Modifications from original:
- Added --no-history flag to disable conversation history
- Added fixed window context management to prevent context overflow

No classes, no fancy features - just the essentials.

Usage:
  python simple_chat_agent.py                # normal chat with history
  python simple_chat_agent.py --no-history   # each turn is independent
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================================
# CONFIGURATION - Change these settings as needed
# ============================================================================

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# System prompt - This sets the chatbot's behavior and personality
# Change this to customize how the chatbot responds
SYSTEM_PROMPT = "You are a helpful AI assistant. Be concise and friendly."

# Maximum number of messages to keep in history (not counting system prompt).
# Must be even so user/assistant pairs are never split (e.g. 10 = 5 pairs).
MAX_HISTORY_MESSAGES = 10

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

# Check for --no-history flag
USE_HISTORY = "--no-history" not in sys.argv

# ============================================================================
# LOAD MODEL (NO QUANTIZATION)
# ============================================================================

print("Loading model (this takes 1-2 minutes)...")

# Load tokenizer (converts text to numbers and vice versa)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model in half precision (float16) for efficiency
# Use float16 on GPU, or float32 on CPU if needed
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,                        # Use FP16 for efficiency
    device_map="auto",                          # Automatically choose GPU/CPU
    low_cpu_mem_usage=True
)

model.eval()  # Set to evaluation mode (no training)
print(f"✓ Model loaded! Using device: {model.device}")
print(f"✓ Memory usage: ~2.5 GB (FP16)\n")

# ============================================================================
# CHAT HISTORY - This is stored as PLAIN TEXT (list of dictionaries)
# ============================================================================
# The chat history is a list of messages in this format:
# [
#   {"role": "system", "content": "You are a helpful assistant"},
#   {"role": "user", "content": "Hello!"},
#   {"role": "assistant", "content": "Hi! How can I help?"},
#   {"role": "user", "content": "What's 2+2?"},
#   {"role": "assistant", "content": "2+2 equals 4."}
# ]
#
# This is PLAIN TEXT - humans can read it
# The model CANNOT use this directly - it needs to be tokenized first

chat_history = []

# Add system prompt to history (this persists across the entire conversation)
chat_history.append({
    "role": "system",
    "content": SYSTEM_PROMPT
})

# ============================================================================
# CHAT LOOP
# ============================================================================

print("="*70)
print(f"Chat started! History: {'ON (fixed window, last ' + str(MAX_HISTORY_MESSAGES) + ' messages)' if USE_HISTORY else 'OFF (stateless)'}")
print("Type 'quit' or 'exit' to end the conversation.")
print("="*70 + "\n")

while True:
    # ========================================================================
    # STEP 1: Get user input (PLAIN TEXT)
    # ========================================================================
    user_input = input("You: ").strip()
    
    # Check for exit commands
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("\nGoodbye!")
        break
    
    # Skip empty inputs
    if not user_input:
        continue

    # ========================================================================
    # STEP 2: Add user message to chat history (PLAIN TEXT)
    # ========================================================================
    # The chat history grows with each exchange
    # We append the new user message to the existing history
    chat_history.append({
        "role": "user",
        "content": user_input
    })

    # At this point, chat_history looks like:
    # [
    #   {"role": "system", "content": "You are helpful..."},
    #   {"role": "user", "content": "Hello!"},
    #   {"role": "assistant", "content": "Hi!"},
    #   {"role": "user", "content": "What's 2+2?"},      <- Just added
    # ]
    # This is still PLAIN TEXT

    # ========================================================================
    # STEP 2b: Apply fixed window (only when history is enabled)
    # ========================================================================
    # If history is DISABLED, we build a fresh single-turn context each time:
    #   [system, current user message]
    # This means the model has NO memory of previous turns.
    #
    # If history is ENABLED, we keep only the last MAX_HISTORY_MESSAGES
    # messages (not counting the system prompt), dropping the oldest pairs.

    if USE_HISTORY:
        # Keep system prompt + last MAX_HISTORY_MESSAGES messages.
        # Since MAX_HISTORY_MESSAGES is even, pairs are never split.
        if len(chat_history) - 1 > MAX_HISTORY_MESSAGES:
            print("[Note: earliest messages removed to fit fixed window]")
            chat_history = [chat_history[0]] + chat_history[-MAX_HISTORY_MESSAGES:]
        context_to_send = chat_history
    else:
        # Stateless: only send system prompt + current message
        context_to_send = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_input}
        ]

    # ========================================================================
    # STEP 3: Convert chat history to model input (TOKENIZATION)
    # ========================================================================
    # The model needs numbers (tokens), not text
    # apply_chat_template() does two things:
    #   1. Formats the chat history with special tokens (like <|start|>, <|end|>)
    #   2. Converts the formatted text into token IDs (numbers)
    
    # First, apply_chat_template formats the history and converts to tokens
    input_ids = tokenizer.apply_chat_template(
    context_to_send,
    add_generation_prompt=True,
    return_tensors="pt"
    )

    # apply_chat_template may return a BatchEncoding dict or a plain tensor
    # extract the input_ids tensor either way
    if hasattr(input_ids, 'input_ids'):
        attention_mask = input_ids.attention_mask.to(model.device)
        input_ids = input_ids.input_ids.to(model.device)
    else:
        input_ids = input_ids.to(model.device)
        attention_mask = torch.ones_like(input_ids)

    # Now input_ids is TOKENIZED - it's a tensor of numbers like:
    # tensor([[128000, 128006, 9125, 128007, 271, 2675, 527, 264, ...]])
    # These numbers represent our entire conversation history

    # ========================================================================
    # STEP 4: Generate assistant response (MODEL INFERENCE)
    # ========================================================================
    # The model looks at the ENTIRE chat history (in tokenized form)
    # and generates a response

    print("Assistant: ", end="", flush=True)

    with torch.no_grad():  # Don't calculate gradients (we're not training)
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,   # Explicitly pass attention mask
            max_new_tokens=512,              # Maximum length of response
            do_sample=True,                  # Use sampling for variety
            temperature=0.7,                 # Lower = more focused, higher = more random
            top_p=0.9,                       # Nucleus sampling
            pad_token_id=tokenizer.eos_token_id
        )
    
    # outputs contains: [original input tokens + new generated tokens]
    # We only want the NEW tokens (the assistant's response)
    
    # ========================================================================
    # STEP 5: Decode the response (DETOKENIZATION)
    # ========================================================================
    # Extract only the newly generated tokens
    new_tokens = outputs[0][input_ids.shape[1]:]
    
    # Convert tokens (numbers) back to text (PLAIN TEXT)
    assistant_response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True  # Remove special tokens like <|end|>
    )
    
    print(assistant_response)  # Display the response
    
    # ========================================================================
    # STEP 6: Add assistant response to chat history (PLAIN TEXT)
    # ========================================================================
    # This is crucial! We add the assistant's response to the history
    # so the model remembers what it said in future turns.
    # When history is OFF we skip this - there's nothing to remember.

    if USE_HISTORY:
        chat_history.append({
            "role": "assistant",
            "content": assistant_response
        })
    
    # Now chat_history has grown again:
    # [
    #   {"role": "system", "content": "You are helpful..."},
    #   {"role": "user", "content": "Hello!"},
    #   {"role": "assistant", "content": "Hi!"},
    #   {"role": "user", "content": "What's 2+2?"},
    #   {"role": "assistant", "content": "4"}              <- Just added
    # ]
    
    # When the loop repeats:
    # - User enters new message
    # - We add it to chat_history
    # - We tokenize the ENTIRE history (including all previous exchanges)
    # - Model sees everything and generates response
    # - We add response to history
    # - Repeat...
    
    # This is how the chatbot "remembers" the conversation!
    # Each turn, we feed it the ENTIRE conversation history
    
    print()  # Blank line for readability

# ============================================================================
# SUMMARY OF HOW CHAT HISTORY WORKS
# ============================================================================
"""
PLAIN TEXT vs TOKENIZED:

1. PLAIN TEXT (chat_history):
   - Human-readable format
   - List of dictionaries: [{"role": "user", "content": "Hi"}, ...]
   - Stored in memory between turns
   - Gets longer with each message

2. TOKENIZED (input_ids):
   - Numbers (token IDs)
   - Created fresh each turn from chat_history
   - This is what the model actually "reads"
   - Example: [128000, 128006, 9125, 128007, ...]

PROCESS EACH TURN:
   User input (text)
   ↓
   Add to chat_history (text)
   ↓
   [If history enabled] Drop oldest pairs if over MAX_HISTORY_MESSAGES
   [If history disabled] Use only [system + current message]
   ↓
   Tokenize context (text → numbers)
   ↓
   Model generates response (numbers)
   ↓
   Decode response (numbers → text)
   ↓
   [If history enabled] Add response to chat_history (text)
   ↓
   Loop back to start

HISTORY ON vs OFF:
   ON  → model sees last MAX_HISTORY_MESSAGES messages; follow-ups work
   OFF → model sees only the current message; every turn starts fresh

FIXED WINDOW (when history is ON):
   - After adding each message, total count is checked
   - If over MAX_HISTORY_MESSAGES, oldest messages are sliced off
   - System message is always kept at index 0
   - MAX_HISTORY_MESSAGES must be even to avoid splitting user/assistant pairs

WHAT HAPPENS AS CONVERSATION GROWS?
- chat_history gets longer (more messages)
- Once it exceeds MAX_HISTORY_MESSAGES, oldest pairs are dropped
- History stays a fixed size from that point on
"""