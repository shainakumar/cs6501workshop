## TEAM  
- Ariful Islam
- Prithvi Raj 
- Shaina	Kumar

# Topic 4: Exploring Tools

This directory contains our exploration of ToolNode-based orchestration, LangChain ReAct agents, and a completed 2-Hour Agent Project.

---

## Directory Structure 

    Topic4Exploring/
    ├── toolnode_example.py
    │   Example using ToolNode-based orchestration.
    │   Demonstrates explicit routing and parallel tool dispatch.
    │
    ├── react_agent_example.py
    │   Example using LangChain's ReAct agent.
    │   Demonstrates the Thought → Action → Observation loop.
    │
    ├── Topic4Exploring.pdf
    │   Written answers to all required portfolio questions (Task 3).
    │
    ├── README.md
    │
    ├── outputs/
    │   Saved terminal runs and generated graph images.
    │
    │   ├── langchain_manual_tool_graph.png
    │   │   Mermaid graph for ToolNode example.
    │   │
    │   ├── langchain_react_agent.png
    │   │   Mermaid graph for ReAct example.
    │   │
    │   ├── langchain_conversation_graph.png
    │   │   Conversation wrapper graph.
    │   │
    │   ├── task3_toolnode_verbose_run.txt
    │   ├── task3_toolnode_exit_run.txt
    │   │   ToolNode command-handling outputs.
    │   │
    │   ├── task3_react_agent_verbose_run.txt
    │   └── task3_react_agent_exit_run.txt
    │       ReAct command-handling outputs.
    │
    └── 2HourProject/
        ├── weather_travel_planner.py
        │   Smart Travel Planner implementation.
        │   Includes:
        │     - OpenWeatherMap tool
        │     - ReAct agent integration
        │     - Structured output enforcement
        │     - Command handling
        │     - Graph visualization
        │
        ├── task5_sample_output.txt
        │   Example agent interaction.
        │
        └── task5_test_output.txt
            Tool-only test output (API validation + error handling).

---

## 2-Hour Agent Project

The Smart Travel Planner:

- Accepts destination + travel dates
- Calls OpenWeatherMap API
- Generates:
  - Weather snapshot
  - Packing list (with reasons)
  - Activity recommendations
- Handles:
  - Invalid city
  - Invalid date ranges
  - API errors
  - `verbose`, `quiet`, `exit` commands
 
The Smart Travel Planner uses a LangGraph conversation wrapper around a LangChain ReAct agent. It defines a get_weather_forecast tool that calls the OpenWeatherMap API for a specific city and date range, then formats the results. The graph controls the flow: an input node handles user commands (verbose, quiet, exit), the agent node generates responses using the weather tool, an output node prints the result, and a trim node limits conversation history. A system prompt enforces a structured response format (weather snapshot, packing list, activities). Because the free OpenWeatherMap API tier only provides a 5-day forecast, the planner can only return weather data within that window. If a user provides a longer trip, the tool warns that only the first five days are available and returns data for the supported range. The --test-tool mode runs the weather tool independently without the agent. It executes predefined test cases to verify API connectivity, unit handling, date validation, and error handling before full agent integration.

---

Run full agent:

    python weather_travel_planner.py

Run tool test mode:

    python weather_travel_planner.py --test-tool

Link to Google Colab Notebook: https://colab.research.google.com/drive/1wn5cdGza_iPSRci_FpkEWOaOcooBOvN0?usp=sharing
