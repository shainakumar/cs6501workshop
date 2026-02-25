## (4) Local Timing Experiments (MacBook Air M1)

All experiments were timed using the shell `time` command:

```bash
time python llama_mmlu_eval.py
```

The `time` command reports:

- **real** → Wall-clock time (total elapsed time)
- **user** → CPU time spent in user space
- **sys** → CPU time spent in kernel space

Wall-clock time (`real`) is used as the primary performance metric.

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
## (8) Chat Agent Implementation 

The provided starter code allowed conversation history to grow without limit, which would eventually exceed the model’s context window; thus, I implemented a **fixed window strategy** (strategy 3 in the Llama Chat Context Management Guide) with system prompt preservation. The agent keeps the system message and only the most recent N user/assistant turns. If the history exceeds this limit, the oldest user/assistant pair is removed while preserving the system message. This ensures that:

- The system prompt defining assistant behavior is always retained.
- Memory usage remains bounded.
- The model does not exceed its maximum context length.
- Long conversations do not crash due to token overflow.

This approach is simple, predictable, and computationally inexpensive compared to summarization-based strategies. The use case here is short/Q&A style conversations. 
I added a command-line flag that allows conversation history to be disabled. When the flag is turned off, the model receives only the current user message (plus the system prompt), and previous turns are not included in the prompt. When history is enabled, the full managed conversation window is included.

This makes it easy to directly compare multi-turn performance under the two conditions (enabled or disabled history). In multi-turn conversations, the difference was immediately visible. When history was enabled, the model maintained context across turns, and could correctly reference earlier information (my name) and maintain topic continuity about me and my cat. When history was disabled, the model lost context, could not recall prior details, and responded saying it did not have any information related to my questions. It also thought each turn was our first conversation. As expected, this demonstrates that conversation memory is essential for coherent multi-turn dialogue. Without context management, the agent either fails due to overflow (if history grows unbounded) or loses conversational continuity (if history is disabled). Ultimately, strong context window management strategies provide a balance between memory retention and computational constraints.
