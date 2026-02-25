## (1) Sequential vs Parallel MMLU Evaluation 

Llama 3.2-1B was evaluated on two separate MMLU subjects (`astronomy` and `abstract_algebra`) using two different execution setups: the original Hugging Face Transformers implementation and a modified version that sends requests to an Ollama server. Per the instructions, two separate scripts (one per subject) were created and measured real wall-clock time using the `time` shell command. 

---
Abstract Algebra

real    0m21.119s  
user    0m16.767s  
sys     0m2.062s  

Astronomy

real    0m23.165s  
user    0m18.346s  
sys     0m2.212s  


For both subjects, wall-clock time (`real`) was approximately 21–23 seconds. Most of this time was spent in user space (16–18 seconds): CPU was actively executing Python and model computation. The system time was small (~2 seconds), indicating minimal OS overhead. The internal script-reported duration (~0.1 minutes) measured only the evaluation loop, not model initialization. Therefore, roughly 15 seconds of the 21–23 seconds total runtime is attributable to model loading and setup.

--- 
Ollama Sequential Execution

real    2m9.546s  
user    0m14.226s  
sys     0m1.700s  

Ollama Parallel Execution

real    1m57.275s  
user    0m17.900s  
sys     0m1.989s 

--- 
In the original versions, each script loads the model directly through Hugging Face Transformers on a Tesla T4 GPU in Colab. This means that each script independently initializes the model before running inference. Once loaded, evaluation runs extremely quickly on the GPU. 
When modifying the scripts to use Ollama, the model is loaded once by the server and reused across requests. Each MMLU question triggered a separate API call to the running model. In this configuration, inference was noticeably slower than the direct Transformers approach. 
In both Ollama runs, wall-clock time is nearly 2 minutes, while user time remains around 14–18 seconds. This indicates that the Python process is spending most of its time waiting rather than actively computing.
Each question required a network call, so there was additional overhead per request. Throughput dropped significantly compared to the 30–36 questions per second observed in the unmodified GPU-based implementation.
When running sequentially, the second script began only after the first completed. Since both scripts shared the same Ollama server instance, the model did not need to reload between runs, but each question was still processed one at a time. Comparatively, when running in parallel, both scripts issued requests simultaneously to the same Ollama server. However, because both processes shared a single model backend, true parallel speedup was limited. As a result, parallel execution did not produce a 2× speedup that might have seemed intuitive. Instead, there was approximately a 12-second improvement.  
These runs provide evidence that direct Transformers inference on a GPU is significantly faster per question than serving through Ollama over HTTP and that Olama avoids repeated model loading but introduces per-request overhead. Thus, parallel execution does not guarantee speedup when both processes compete for the same GPU or model server, which also indicates that performance relies on several factors beyond model size (architectural decisions such as model loading strategy, API overhead, resource contention, etc.). 


## (3) Manual Tool Handling with Calculator  

For this task, I added a new tool, `geo_calculator`, to the provided manual tool-handling starter code. This tool is a plain Python function that takes a single argument `input_json` (a JSON string), parses it with `json.loads`, evaluates the requested expression, and returns a structured JSON result using `json.dumps`. I used `numexpr` for evaluation and explicitly provided math constants/functions (`pi`, `e`, `sin`, `cos`, `tan`, `sqrt`) via `local_dict` so the tool can handle both arithmetic and trig expressions. This ensures all computations are performed by the tool rather than the LLM. Additionally, I updated the tool schema passed to the model so it knows the tool name, what it does, and how to call it. In particular, the tool schema requires `input_json` and documents the expected format (e.g., `{"expr":"19.5*(7+2)"}`, `{"expr":"sin(pi/2)+cos(0)"}`). To test the implementation, I ran six test prompts. Tests 1 and 3 (directly from the starter code) confirm the model correctly calls the weather tool once (or twice) and then produces a final natural-language answer. Tests 4 and 5 confirm the model calls `geo_calculator` for arithmetic and trig (e.g., `19.5*(7+2)` and `sin(pi/2)+cos(0)`), receives the JSON tool result, and then outputs the final answer using the tool’s computed value. For Test 6, I explored “forcing” tool usage by setting `tool_choice="required"` initially; I observed that keeping `required` for every iteration can cause repeated tool calls. The fix was to force the tool call only on the first step and then switch back to `auto` so the model can stop calling the tool and produce a final answer. After this change, the forced-tool-use test successfully called `geo_calculator` once and then returned a final response (approximately 4.0).

## (4) Tool Calling with LangChain 

For this task, we added three additional tools to the starter code. First, I integrated my custom `geo_calculator` tool from Task 3. Second, I created a `count_letter` tool that counts how many times a specified letter appears in a given text string and returns a JSON payload containing the count. This enables questions such as the recommended “How many s’s are in Mississippi riverboats?” and supports multiple tool calls within a single model turn. Third, I created a `word_count` tool of my own, which counts whitespace-delimited words in a string and returns the total as JSON (not the most creative, but useful). I included standalone tests to confirm that this tool works both for simple inputs and for phrases containing punctuation and irregular spacing. The tool execution dispatch code logic was modified as instructed. Moreover, I created individual tests to demonstrate each tool in isolation before testing multi-tool interactions. For example, when asking “How many words are in ‘I love my cat’?”, the model invoked `word_count` once, the tool returned a count of four, and the model produced a final answer. A similar test using punctuation and extra spaces confirmed that the tool correctly returned five words for “Hello,   world!  I  love AI.” The intention was to demonstrate that the tool works independently and reliably before being combined with others.

To demonstrate multiple tool calls within a single iteration, I asked whether there are more i’s than s’s in “Mississippi riverboats.” In the first iteration, the model invoked `count_letter` twice—once for `i` and once for `s`. Both counts returned five, and in the next iteration the model concluded that the counts were equal. I also constructed prompts that required sequential chaining across outer-loop iterations. For the question “What is the sin of (#i - #s) in ‘Mississippi riverboats’?”, the model first invoked `count_letter` twice to compute the counts. In the next iteration, it invoked `geo_calculator` to compute the difference (5 - 5), and in a subsequent iteration invoked `geo_calculator` again to compute `sin(0)`. Finally, it produced a natural-language answer. To show all tools working together, I created a combined query requiring letter counts, word counts, weather lookup, and trigonometric computation in one request (In 'Mississippi riverboats', count i's and s's, compute sin(i_count - s_count), tell me how many words are in the text, and also what's the weather in London. Use tools). In the first iteration, the model invoked four tools in the same turn: two `count_letter` calls, one `word_count`, and one `get_weather`. In the second iteration, it invoked `geo_calculator` to compute `sin(i_count - s_count)`. In the third iteration, it produced a final answer summarizing all results. I tried to get seqential chaining to hit the 5 turn limit in the outer loop by requesting nested trigonometric computations such as `sin(d)`, `sin(sin(d))`, `sin(sin(sin(d)))`, and so on. In this case, the model continued invoking the calculator tool across iterations and reached the five-iteration limit before producing a final summary. Lastly, I also constructed a query that required the model to combine structured outputs from different tool types (string parsing, counting, and math) into a final trigonometric computation: “Use tools only. Get the weather in London, and use the Fahrenheit temperature number from the tool output. Count the number of "c" letters in "zucchini". Count the number of words in "I love my cat". Compute sin(temp_f + c_count - word_count). Show temp_f, c_count, word_count, and the final result.” This query required extracting a numeric value from the weather tool output, counting character occurrences using `count_letter`, counting words using `word_count`, and computing a trigonometric expression using `geo_calculator`. The model first invoked `get_weather`, then in the next iteration invoked both `count_letter` and `word_count` in parallel, and finally invoked `geo_calculator` to compute the arithmetic expression before applying sine. 


## (5) Persistent Conversation Agent with LangGraph, Checkpointing, and Recovery

This implementation rewrites the earlier manual tool-handling loop into a fully persistent, stateful LangGraph agent. Instead of restarting the conversation on each invocation, the system maintains a single long-running conversation using a SQLite-backed checkpointer (`SqliteSaver`) and a fixed `thread_id`. All conversation state, including the complete message history, is stored durably in SQLite using `SqliteSaver.from_conn_string("conversation.db")` inside a context manager. Each conversation is keyed by a `thread_id`, which allows recovery and continuation of the same conversation across program restarts. I found that the system does successfully demonstrate multi-tool orchestration, cross-turn reasoning, and persistent context. In the conversation, I requested that the assistant use tools only to retrieve the weather in London, count the number of "c" letters in "zucchini," and count the number of words in "I love my cat." The agent correctly executed three tool calls in a single turn: `get_weather('London')` returned "Rainy, 48F," `count_letter('zucchini','c')` returned 2, and `word_count('I love my cat')` returned 4. Then, the assistant computed `sin(48 + 2 - 4)` using the `geo_calculator` tool, which returned approximately 0.9018. Later, when it was asked to recall previous results without re-running tools, the assistant correctly remembered that `c_count = 2`, `word_count = 4`, the favorite word was "zucchini," and the sine result was approximately 0.9018. Checkpoint recovery was demonstrated by interrupting the program using `Ctrl+C` and then restarting it with the same `THREAD_ID`. Upon restart, the system printed “Resuming interrupted conversation (thread: conversation_1)” and detected pending nodes via the checkpointer. After recovery, the user asked “What did we compute earlier?” and the assistant correctly recalled the previously computed weather result, letter count, word count, and sine calculation without invoking any new tools. 

## (6) Opportunity for Parallelization

Currently, tool execution is handled inside the `call_tools` node by iterating through `last_msg.tool_calls` and invoking each tool one at a time. Thus, even when the LLM requests multiple independent tool calls in the same turn (e.g., getting the weather in London while also counting letters in “zucchini” and counting words in “I love my cat”), those tools are executed sequentially. We know that these tool calls do not depend on each other’s outputs, so the agent could run independent tool invocations in parallel and then combine all resulting `ToolMessage`s once they complete. 

Link to Colab Notebook: https://colab.research.google.com/drive/1clURnZmY7Llq5bXzVnlz9eyemZMSqHZO?usp=sharing 

## Table of Contents

### Core Scripts
- `manual-tool-handling.py` – Baseline manual tool-calling loop  
- `task1_*` – MMLU evaluation scripts (base + Ollama + subject runs)  
- `task3_manual_tool_handling_geo_calc.py` – Custom calculator tool  
- `task4_multiple_tool_use.py` – Multi-tool orchestration with LangGraph  
- `task5_single_long_convo.py` – Persistent LangGraph agent with SQLite recovery  

### Directories
- `outputs/` – Execution traces, timing logs, graph image, recovery logs  
- `raw_results/` – Raw MMLU JSON results and Ollama logs  
