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
    """
    Custom calculator tool (from scratch).

    Requirements met:
    - Parse tool input with json.loads
    - Format output with json.dumps
    - Use numexpr for arithmetic expression evaluation
    - Include geometry functions

    Input JSON string supports either:
      1) {"expr": "2*(3+4)"}
      2) {"op": "area_circle", "r": 3}
         {"op": "hypotenuse", "a": 3, "b": 4}
         {"op": "distance_2d", "x1":0, "y1":0, "x2":3, "y2":4}
         {"op": "area_rectangle", "w": 5, "h": 2}
         {"op": "area_triangle", "b": 10, "h": 4}
         {"op": "circumference_circle", "r": 3}
    """
    try:
        payload = json.loads(input_json)
    except json.JSONDecodeError as e:
        return json.dumps({"ok": False, "error": f"Invalid JSON input: {e}"})

    # Expression mode (arithmetic)
    if "expr" in payload:
        expr = payload["expr"]
        try:
            val = float(ne.evaluate(expr))
            return json.dumps({"ok": True, "mode": "expr", "expr": expr, "value": val})
        except Exception as e:
            return json.dumps({"ok": False, "mode": "expr", "expr": expr, "error": str(e)})

    # Geometry mode
    op = payload.get("op")
    try:
        if op == "area_circle":
            r = float(payload["r"])
            return json.dumps({"ok": True, "op": op, "value": math.pi * r * r})

        if op == "circumference_circle":
            r = float(payload["r"])
            return json.dumps({"ok": True, "op": op, "value": 2 * math.pi * r})

        if op == "area_rectangle":
            w = float(payload["w"])
            h = float(payload["h"])
            return json.dumps({"ok": True, "op": op, "value": w * h})

        if op == "area_triangle":
            b = float(payload["b"])
            h = float(payload["h"])
            return json.dumps({"ok": True, "op": op, "value": 0.5 * b * h})

        if op == "hypotenuse":
            a = float(payload["a"])
            b = float(payload["b"])
            return json.dumps({"ok": True, "op": op, "value": math.hypot(a, b)})

        if op == "distance_2d":
            x1 = float(payload["x1"]); y1 = float(payload["y1"])
            x2 = float(payload["x2"]); y2 = float(payload["y2"])
            return json.dumps({"ok": True, "op": op, "value": math.hypot(x2 - x1, y2 - y1)})

        return json.dumps({"ok": False, "error": f"Unknown op '{op}'. Provide 'expr' or a supported 'op'."})

    except KeyError as e:
        return json.dumps({"ok": False, "op": op, "error": f"Missing parameter: {e}"})
    except Exception as e:
        return json.dumps({"ok": False, "op": op, "error": str(e)})


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
            "description": (
                "Use this tool for ALL calculations (arithmetic or geometry). "
                "Do not do math in your head. Provide input_json as a JSON string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "input_json": {
                        "type": "string",
                        "description": (
                            "A JSON string. Either expression mode: "
                            "{\"expr\":\"2*(3+4)\"} "
                            "or geometry mode examples: "
                            "{\"op\":\"area_circle\",\"r\":3}, "
                            "{\"op\":\"hypotenuse\",\"a\":3,\"b\":4}, "
                            "{\"op\":\"distance_2d\",\"x1\":0,\"y1\":0,\"x2\":3,\"y2\":4}"
                        )
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
        "If the user asks for ANY calculation (arithmetic or geometry), you MUST call geo_calculator. "
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
            tool_choice="auto"  # Let the model decide whether to use tools
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
            # Loop continues - LLM will see the tool results
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
    print("TEST 2: Calculator expr")
    print("="*60)
    run_agent("Compute 19.5 * (7 + 2).")

    print("\n" + "="*60)
    print("TEST 3: Geometry")
    print("="*60)
    run_agent("What is the area of a circle with radius 3?")

    print("\n" + "="*60)
    print("TEST 4: If model resists tool use, force it")
    print("="*60)
    run_agent("What is the hypotenuse when a=3 and b=4?", force_tool_use=True)
