"""
Manual Tool Calling Exercise
Students will see how tool calling works under the hood.
"""

import json
from openai import OpenAI
import math
import numexpr as ne

# ============================================
# PART 1: Define Your Tools
# ============================================

def get_weather(location: str) -> str:
    """Get the current weather for a location"""
    # Simulated weather data
    weather_data = {
        "San Francisco": "Sunny, 72°F",
        "New York": "Cloudy, 55°F",
        "London": "Rainy, 48°F",
        "Tokyo": "Clear, 65°F"
    }
    return weather_data.get(location, f"Weather data not available for {location}")

def geo_calculator(input_json: str) -> str:
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

# ============================================
# PART 2: Describe Tools to the LLM
# ============================================

# This is the JSON schema that tells the LLM what tools exist
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g. San Francisco"
                    }
                },
                "required": ["location"]
            }
        }
    },
    # NEW TOOL 
    {
  "type": "function",
  "function": {
    "name": "geo_calculator",
    "description": "Use this tool for ALL calculations (arithmetic or trig). Provide input_json as a JSON string.",
    "parameters": {
      "type": "object",
      "properties": {
        "input_json": {
          "type": "string",
          "description": "JSON string like {\"expr\":\"19.5*(7+2)\"} or {\"expr\":\"sin(pi/2)+cos(0)\"}"
        }
      },
      "required": ["input_json"]
    }
  }
}
]


# ============================================
# PART 3: The Agent Loop
# ============================================

def run_agent(user_query: str, model: str = "gpt-4.1-mini", force_tool_use: bool = False):
    """
    Simple agent that can use tools.
    Shows the manual loop that LangGraph automates.
    """
    
    # Initialize OpenAI client
    client = OpenAI()

    system_prompt = (
        "You are a helpful assistant. "
        "If the user asks for ANY calculation (arithmetic or trig), you MUST call geo_calculator. "
        "Do NOT compute yourself."
    )
    
    # Start conversation with user query
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    tool_choice = "required" if force_tool_use else "auto"
    
    print(f"User: {user_query}\n")
    
    # Agent loop - can iterate up to 5 times
    for iteration in range(5):
        print(f"--- Iteration {iteration + 1} ---")
        
        # Call the LLM
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,  # ← This tells the LLM what tools are available
            tool_choice=tool_choice  # Let the model decide whether to use tools
        )
        
        assistant_message = response.choices[0].message
        
        # Check if the LLM wants to call a tool
        if assistant_message.tool_calls:
            print(f"LLM wants to call {len(assistant_message.tool_calls)} tool(s)")
            
            # Add the assistant's response to messages
            messages.append(assistant_message)
            
            # Execute each tool call
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"  Tool: {function_name}")
                print(f"  Args: {function_args}")
                
                # THIS IS THE MANUAL DISPATCH
                # In a real system, you'd use a dictionary lookup
                tool_dispatch = {
                    "get_weather": get_weather,
                    "geo_calculator": geo_calculator,  # takes input_json (string)
                }

                if function_name in tool_dispatch:
                    result = tool_dispatch[function_name](**function_args)
                else:
                    result = json.dumps({"ok": False, "error": f"Unknown function {function_name}"})

                print(f"  Result: {result}")
                
                # Add the tool result back to the conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": result
                })
            
            print()
            tool_choice = "auto"
        else:
            # No tool calls - LLM provided a final answer
            print(f"Assistant: {assistant_message.content}\n")
            return assistant_message.content
    
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
    print("TEST 4: Calculator expr")
    print("="*60)
    run_agent("Compute 19.5 * (7 + 2).")

    print("\n" + "="*60)
    print("TEST 5: Trig expression")
    print("="*60)
    run_agent("Compute sin(pi/2) + cos(0).")

    print("\n" + "="*60)
    print("TEST 6: Force tool use")
    print("="*60)
    run_agent("Compute sqrt(2) * sqrt(8). Return a decimal.", force_tool_use=True)