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

