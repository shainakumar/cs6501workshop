"""
Tool Calling with LangChain
Shows how LangChain abstracts tool calling.
"""

import json
import math
import numexpr as ne

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# ============================================
# PART 1: Define Your Tools
# ============================================

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location"""
    # Simulated weather data
    weather_data = {
        "San Francisco": "Sunny, 72°F",
        "New York": "Cloudy, 55°F",
        "London": "Rainy, 48°F",
        "Tokyo": "Clear, 65°F"
    }
    return weather_data.get(location, f"Weather data not available for {location}")

@tool
def geo_calculator(input_json: str) -> str:
    """
    Calculator tool. input_json MUST be a JSON string with ONE key:
      - {"expr": "<expression>"}  OR {"expression": "<expression>"}

    The expression may use + - * / ** parentheses and functions sin, cos, tan, sqrt,
    and constants pi, e.

    Example input_json:
      {"expr":"sin(5-5)"}
      {"expr":"sin(48 + 2 - 4)"}
    Returns a JSON string: {"ok": true, "expression": "...", "result": <float>}
    """
    try:
        params = json.loads(input_json)

        if not isinstance(params, dict):
            raise ValueError("Provide JSON like {'expr': '2*(3+4)'}")

        expr = params.get("expr") or params.get("expression")

        if not expr or not isinstance(expr, str):
            raise ValueError("Missing or invalid expression.")

        result = float(
            ne.evaluate(
                expr,
                local_dict={
                    "pi": math.pi,
                    "e": math.e,
                    "sin": math.sin,
                    "cos": math.cos,
                    "tan": math.tan,
                    "sqrt": math.sqrt,
                }
            )
        )

        return json.dumps({
            "ok": True,
            "expression": expr,
            "result": result
        })

    except Exception as e:
        return json.dumps({
            "ok": False,
            "error": str(e)
        })

@tool
def count_letter(text: str, letter: str) -> str:
    """
    Count occurrences of a letter in a piece of text.
    Returns JSON string.
    """
    letter = letter.strip()
    if len(letter) != 1:
        return json.dumps({"ok": False, "error": "letter must be a single character"})

    count = text.lower().count(letter.lower())
    return json.dumps({"ok": True, "text": text, "letter": letter, "count": count})

@tool
def word_count(text: str) -> str:
    """
    Count words in text. Returns JSON string.
    """
    words = [w for w in text.strip().split() if w]
    return json.dumps({"ok": True, "text": text, "count": len(words)})


# ============================================
# PART 2: Create LLM with Tools
# ============================================

# Create LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Bind tools to LLM
tools = [get_weather, geo_calculator, count_letter, word_count]
tool_map = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)


# ============================================
# PART 3: The Agent Loop
# ============================================

def run_agent(user_query: str):
    """
    Simple agent that can use tools.
    Shows the manual loop that LangGraph automates.
    """
    
    # Start conversation with user query
    messages = [
        SystemMessage(content="You are a helpful assistant. "
    "For ANY counting, you MUST use count_letter. "
    "For word totals, you MUST use word_count. "
    "For ANY math, you MUST use geo_calculator. "
    "IMPORTANT: geo_calculator must be called with input_json formatted exactly like {\"expr\":\"<expression>\"}."),
        HumanMessage(content=user_query)
    ]
    
    print(f"User: {user_query}\n")
    
    # Agent loop - can iterate up to 5 times
    for iteration in range(5):
        print(f"--- Iteration {iteration + 1} ---")
        
        # Call the LLM
        response = llm_with_tools.invoke(messages)
        
        # Check if the LLM wants to call a tool
        if response.tool_calls:
            print(f"LLM wants to call {len(response.tool_calls)} tool(s)")
            
            # Add the assistant's response to messages
            messages.append(response)
            
            # Execute each tool call
            for tool_call in response.tool_calls:
                function_name = tool_call["name"]
                function_args = tool_call["args"]
                
                print(f"  Tool: {function_name}")
                print(f"  Args: {function_args}")
                
                # Execute the tool
                if function_name in tool_map:
                    result = tool_map[function_name].invoke(function_args)
                else:
                    result = f"Error: Unknown function {function_name}"

                print(f"  Result: {result}")
                
                # Add the tool result back to the conversation
                messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"]
                ))
            
            print()
            # Loop continues - LLM will see the tool results
            
        else:
            # No tool calls - LLM provided a final answer
            print(f"Assistant: {response.content}\n")
            return response.content
    
    return "Max iterations reached"


# ============================================
# PART 4: Test It
# ============================================

if __name__ == "__main__":
    # Test query that requires tool use
    print("="*60)
    print("TEST 1: Query requiring tool")
    print("="*60)
    run_agent("What's the weather like in San Francisco?")
    
    print("\n" + "="*60)
    print("TEST 2: Query not requiring tool")
    print("="*60)
    run_agent("Say hello!")
    
    print("\n" + "="*60)
    print("TEST 3: Multiple tool calls")
    print("="*60)
    run_agent("What's the weather in New York and London?")

    print("\n" + "="*60)
    print("TEST: word_count (basic)")
    print("="*60)
    run_agent('How many words are in "I love my cat"? Use tools.')

    print("\n" + "="*60)
    print("TEST: word_count (punctuation + spacing)")
    print("="*60)
    run_agent('How many words are in "Hello,   world!  I  love AI."? Use tools.')

    print("\n" + "="*60)
    print("TEST: Two tool calls in one turn (count i and s)")
    print("="*60)
    run_agent("Are there more i's than s's in 'Mississippi riverboats'? Use tools.")

    print("\n" + "="*60)
    print("TEST: Chaining (counts -> sin(diff))")
    print("="*60)
    run_agent("What is the sin of (#i - #s) in 'Mississippi riverboats'? Use tools.")

    print("\n" + "="*60)
    print("TEST: Use ALL tools")
    print("="*60)
    run_agent(
        "In 'Mississippi riverboats', count i's and s's, compute sin(i_count - s_count), "
        "tell me how many words are in the text, and also what's the weather in London. "
        "Use tools.")

    print("\n" + "="*60)
    print("TEST: Try to hit 5-iteration limit")
    print("="*60)
    run_agent(
        "Count i's and s's in 'Mississippi riverboats'. "
        "Compute d = i_count - s_count. Then compute sin(d), sin(sin(d)), sin(sin(sin(d))), sin(sin(sin(sin(d)))), sin(sin(sin(sin(sin(d)))))) "
        "and show each intermediate value. Use tools for every step."
    )

    print("\n" + "="*60)
    print("TEST: Use ALL tools (no weather JSON needed)")
    print("="*60)
    run_agent(
        'Use tools only. Get the weather in London, and use the Fahrenheit temperature number from the tool output. '
        'Count the number of "c" letters in "zucchini". '
        'Count the number of words in "I love my cat". '
        'Compute sin(temp_f + c_count - word_count). '
        'Show temp_f, c_count, word_count, and the final result.'
    )


