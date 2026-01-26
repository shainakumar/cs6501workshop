Link to Google Colab Notebook: 
https://colab.research.google.com/drive/1g2zNiT4VDubuiZYH4b4r4AbFRur_-mvz?usp=sharing

# Topic2Frameworks — LangGraph Agents

This directory contains my completed implementations for Topic 2 (Frameworks), demonstrating LangGraph concepts including routing, conditional edges, multi-agent orchestration, chat history with the Message API, and crash recovery. Each task is implemented as a separate Python file and saved with a descriptive name.

---

## Table of Contents

1. [Base Setup](#base-setup)
2. [Task 2 – Verbose / Quiet Tracing](#task-2--verbose--quiet-tracing)
3. [Task 3 – Handling Empty Input](#task-3--handling-empty-input)
4. [Task 4 – Parallel Llama and Qwen](#task-4--parallel-llama-and-qwen)
5. [Task 5 – Routing Between Llama and Qwen](#task-5--routing-between-llama-and-qwen)
6. [Task 6 – Chat History (Llama Only)](#task-6--chat-history-llama-only)
7. [Task 7 – Multi-Agent Shared Chat History](#task-7--multiagent-shared-chat-history)
8. [Task 8 – Crash Recovery](#task-8--crash-recovery)
9. [Artifacts](#artifacts)

---

## Base Setup

**`task2_base_llama_agent.py`**  
Initial LangGraph agent using a Hugging Face Llama model. Demonstrates basic graph structure with nodes, routing, and conditional edges. Generates a Mermaid visualization of the graph.

---

## Task 2 – Verbose / Quiet Tracing

**`task2_verbose_quiet.py`**  
Adds runtime tracing control. If the user inputs `verbose`, each node prints tracing information to stdout; if `quiet`, tracing is disabled. Tracing is handled inside nodes without changing graph structure.

---

## Task 3 – Handling Empty Input

**`task3_no_empty_input_branch.py`**  
Prevents empty input from being passed to the LLM using LangGraph control flow. Implements a three-way conditional branch from the input node, with empty input looping back to itself. Demonstrates why small LLMs behave inconsistently on empty prompts.

---

## Task 4 – Parallel Llama and Qwen

**`task4_parallel_llama_qwen25_05b.py`**  
Demonstrates graph fan-out and fan-in. User input is passed in parallel to a Llama node and a Qwen node, and a downstream join node prints both model outputs.

---

## Task 5 – Routing Between Llama and Qwen

**`task5_route_llama_vs_qwen.py`**  
Replaces parallel execution with routing logic. If the user input begins with `"Hey Qwen"`, the request is routed to Qwen; otherwise it is routed to Llama. Only one model runs per turn.

---

## Task 6 – Chat History (Llama Only)

**`task6_chat_history_llama_only.py`**  
Adds multi-turn conversation memory using the LangChain Message API. Demonstrates how chat history is maintained across turns using supported roles. Qwen support is disabled for this task.

---

## Task 7 – Multi-Agent Shared Chat History

**`task7_multi_agent_shared_history.py`**  
Extends chat history to support both Llama and Qwen despite role limitations. Speaker identity is encoded in message content (e.g., `"Human:"`, `"Llama:"`, `"Qwen:"`). Model-specific system prompts describe participants and context.

---

## Task 8 – Crash Recovery

**`task8_crash_recovery.py`**  
Adds LangGraph checkpointing to support crash recovery. The program can be terminated mid-conversation and restarted with the full graph state and chat history preserved.

---

## Artifacts

- `requirements.txt` — Python dependencies
- `lg_graph.png` — Graph visualization for the base agent
- `lg_graph_task6.png` — Graph visualization with chat history
- `outputs/` — Saved terminal transcripts demonstrating behavior for each task
