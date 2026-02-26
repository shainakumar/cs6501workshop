## (4) Local Timing Experiments (MacBook Air M1)

The assignment required comparing:
- GPU + no quantization  
- GPU + 4-bit quantization  
- GPU + 8-bit quantization  
- CPU + no quantization  
- CPU + 4-bit quantization

Only two of the assignment configurations were feasible for me. My laptop uses Apple Silicon (M1) and does **not** support CUDA.
The 4-bit and 8-bit quantization workflow in Transformers relies on the `bitsandbytes` library, which requires NVIDIA GPUs and CUDA support. 

Given that Apple Silicon uses **Metal Performance Shaders (MPS)** instead of CUDA:
- GPU 4-bit quantization is not supported
- GPU 8-bit quantization is not supported
- CPU 4-bit quantization is also not supported in this setup

Therefore, only these configurations were possible:

1. **MPS (Apple GPU) + no quantization**
2. **CPU + no quantization**

---

## Timing Results (2 MMLU Subjects)

| Setup | Device | Quantization | Real Time | User CPU Time | Sys CPU Time |
|-------|--------|--------------|-----------|---------------|--------------|
| MPS (Apple GPU) | `mps` | None | **170.45 s** (~2.84 min) | 72.62 s | 33.63 s |
| CPU | `cpu` | None | **890.78 s** (~14.85 min) | 1888.90 s | 474.24 s |

---

## Observations

- The MPS run was approximately **5.2× faster** than the CPU run in wall-clock time.
- CPU `user` time exceeds `real` time because PyTorch uses multiple threads and `time` aggregates CPU usage across cores.
- Even without quantization, GPU acceleration via MPS provides a substantial performance improvement over CPU execution.

--- 
## (5) Code Modifications  

The original `llama_mmlu_eval.py` script was designed to evaluate a single model (Llama 3.2-1B) on a small subset of MMLU subjects and print a summary of overall accuracy. I modified this script to support multi-model evaluation, detailed timing instrumentation, and optional verbose per-question output.

First, the evaluation was extended to run on a fixed selection of 10 MMLU subjects instead of the original small test subset. In addition to `meta-llama/Llama-3.2-1B-Instruct`, I added two additional tiny/small instruction-tuned models: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` and `Qwen/Qwen2.5-0.5B-Instruct`. The modified script loops over a `MODELS` list, loading each model sequentially and evaluating it across all 10 subjects. This allows direct comparison of accuracy and runtime performance across the multiple lightweight models within a single execution.

Second, detailed timing instrumentation was added to measure computational cost for each model. For every generation call, the script now records wall-clock time (`real time`) using `time.time()`, CPU time using `time.process_time()`, and GPU time (when running on CUDA) using `torch.cuda.Event` and synchronization. A `TimingInfo` structure accumulates these values across questions and subjects. The final evaluation summary now reports total real time, total CPU time, and total GPU time for each model, allowing performance comparisons and accuracy comparisons. As mentioned, on my local MacBook Air (M1), GPU timing via CUDA events is not available (since it uses MPS rather than CUDA), so GPU time remains zero in local runs, but the instrumentation works correctly on CUDA-enabled systems such as Google Colab.

Lastly, a `--verbose` command-line flag was included using `argparse`. When this flag is enabled, the script prints each question, the answer choices, the correct answer, the model’s predicted answer (or INVALID if parsing fails), and whether the prediction was correct or incorrect. When the flag is not provided, the script behaves in summary mode and prints only aggregated subject-level and model-level results. This makes it possible to inspect model reasoning patterns and error types without modifying the core evaluation logic.

--- 
## (6) Local Analysis 

In the local run, the overall accuracies were Llama-3.2-1B-Instruct: 0.432, Qwen-2.5-0.5B-Instruct: 0.400, and TinyLlama-1.1B-Chat: 0.240. The overall accuracy was computed as the mean of the Boolean is_correct values across all evaluated questions. The performance hierarchy appears to be structured as Llama, then Qwen (only 3.2 percentage points lower), and then TinyLlama (19.2 percentage points below Llama). At the subject level, model performance varied considerably by domain. For Llama-3.2-1B-Instruct, the best subject was clinical knowledge and the worst were abstract algebra and college mathematics. For Qwen-2.5-0.5B-Instruct, the best subject was business ethics and the worst was college chemistry. For TinyLlama-1.1B-Chat, the best subject was college chemistry and the worst was abstract algebra. The consistent drop in math-related domains suggests that the models might have shared limitations in symbolic reasoning. To evaluate whether mistakes were random, the wrong-question Jaccard overlap metric (measures how similar two models’ mistakes are) and correctness confusion matricies were computed. The overall wrong-question overlaps were: Llama vs TinyLlama: 0.511, Llama vs Qwen: 0.534, and TinyLlama vs Qwen: 0.549. These values indicate that approximately 51–55% of the questions missed by one model were also missed by the other, so the mistakes the models make do not appear random and independent. The correctness confusion matrices further support this conclusion. Each matrix counts how often two models were both wrong, both right, or disagreed. For Llama vs TinyLlama, there were 615 cases where both were wrong compared to only 165 where both were right. For Llama vs Qwen, there were 557 both-wrong cases and 326 both-right cases. For TinyLlama vs Qwen, there were 662 both-wrong cases and only 164 both-right cases. Across all three comparisons, the “both wrong” counts are substantially larger than the “both right” counts, which reinforces that there seems to be a shared subset of particularly/systematically difficult questions that multiple models fail on. 

--- 
## (6) Google Colab Analysis 

In the Google Colab run, the overall accuracies were Llama-3.2-1B-Instruct: 0.433, Qwen-2.5-0.5B-Instruct: 0.398, and TinyLlama-1.1B-Chat: 0.238. Parallel to the local run, the performance hierarchy was Llama performing best, followed closely by Qwen (3.5 percentage points lower), and then TinyLlama (19.5 percentage points below Llama). These near-identical values compared to the local run indicate that changing the execution environment did not meaningfully affect the correctness of the models. At the subject level, the same domain-dependent patterns appeared. For Llama-3.2-1B-Instruct, the highest accuracy was again in clinical knowledge (0.543), while abstract algebra (0.240) and college mathematics (0.240) remained among the lowest. For Qwen-2.5-0.5B-Instruct, business ethics (0.500) was the strongest subject, and college chemistry (0.280) was the weakest. For TinyLlama-1.1B-Chat, college chemistry (0.320) was the strongest subject, and abstract algebra (0.150) was the weakest. As in the local run, performance consistently dropped in more math/reasoning-heavy domains across all three models, again suggesting shared limitations in symbolic or multi-step reasoning. Once again, to evaluate whether mistakes were random in the Colab run, the same wrong-question Jaccard overlap metric and correctness confusion matrices were analyzed. The overall wrong-question overlaps were again 0.511 (Llama vs TinyLlama), 0.534 (Llama vs Qwen), and 0.549 (TinyLlama vs Qwen). These values are identical to the local run and indicate that approximately 51–55% of missed questions were shared between model pairs, reinforcing that the errors are structured/have patterns. The correctness confusion matrices show the same trend as in the local evaluation. For Llama vs TinyLlama, there were 615 both-wrong cases compared to 165 both-right cases. For Llama vs Qwen, there were 557 both-wrong cases and 326 both-right cases. For TinyLlama vs Qwen, there were 662 both-wrong cases and 164 both-right cases. In all three pairwise comparisons, the number of shared incorrect answers substantially exceeds the number of shared correct answers; the Google Colab results mirror the local findings. 

--- 
## (8) Chat Agent Implementation 

The provided starter code allowed conversation history to grow without limit, which would eventually exceed the model’s context window; thus, I implemented a **fixed window strategy** (strategy 3 in the Llama Chat Context Management Guide) with system prompt preservation. The agent keeps the system message and only the most recent N user/assistant turns. If the history exceeds this limit, the oldest user/assistant pair is removed while preserving the system message. This ensures that:

- The system prompt defining assistant behavior is always retained.
- Memory usage remains bounded.
- The model does not exceed its maximum context length.
- Long conversations do not crash due to token overflow.

This approach is simple, predictable, and computationally inexpensive compared to summarization-based strategies. The use case here is short/Q&A style conversations. 
I added a command-line flag that allows conversation history to be disabled. When the flag is turned off, the model receives only the current user message (plus the system prompt), and previous turns are not included in the prompt. When history is enabled, the full managed conversation window is included.

This makes it easy to directly compare multi-turn performance under the two conditions (enabled or disabled history). In multi-turn conversations, the difference was immediately visible. When history was enabled, the model maintained context across turns, and could correctly reference earlier information (my name) and maintain topic continuity about me and my cat. When history was disabled, the model lost context, could not recall prior details, and responded saying it did not have any information related to my questions. It also thought each turn was our first conversation. As expected, this demonstrates that conversation memory is essential for coherent multi-turn dialogue. Without context management, the agent either fails due to overflow (if history grows unbounded) or loses conversational continuity (if history is disabled). Ultimately, strong context window management strategies provide a balance between memory retention and computational constraints.


