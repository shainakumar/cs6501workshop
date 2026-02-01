"""
Multi-Model MMLU Evaluation Script (Laptop Optimized with Quantization)

This script evaluates multiple small language models on the MMLU benchmark.
Optimized for laptops with 4-bit or 8-bit quantization to reduce memory usage.

Features:
- Evaluates 6 models in 2 categories:
  * Tiny/Small (1-1.5B): Llama 3.2-1B, TinyLlama-1.1B, Qwen2.5-0.5B
  * Small/Medium (7-8B): Qwen2.5-7B, Mistral 7B, OLMo 2 7B
- 10 MMLU subjects
- Detailed timing information (real time, CPU time, GPU time)
- Optional verbose mode to print each question and answer immediately

Quantization options:
- 4-bit: ~1.5-4 GB VRAM/RAM per model
- 8-bit: ~2.5-8 GB VRAM/RAM per model
- No quantization: ~5-28 GB VRAM/RAM per model

Usage:
1. Install: pip install transformers torch datasets accelerate tqdm bitsandbytes
2. Login: huggingface-cli login
3. Run: python llama_mmlu_eval_modified.py [--verbose] [--models {all,tiny,medium}]

Options:
  --verbose          Print each question and answer as evaluation proceeds
  --models all       Evaluate all 6 models (default)
  --models tiny      Evaluate only tiny/small models (1-1.5B)
  --models medium    Evaluate only small/medium models (7-8B)

Set QUANTIZATION_BITS below to choose quantization level.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import json
from tqdm.auto import tqdm
import os
from datetime import datetime
import sys
import platform
import time
import argparse

# ============================================================================
# CONFIGURATION - Modify these settings
# ============================================================================

# Models to evaluate
# Tiny/Small models (1-1.5B parameters)
TINY_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2.5-0.5B-Instruct"
]

# Small/Medium models (7-8B parameters)
MEDIUM_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "allenai/OLMo-2-1124-7B-Instruct"
]


# Combine all models for evaluation
MODELS = TINY_MODELS + MEDIUM_MODELS

# GPU settings
# If True, will attempt to use the best available GPU (CUDA for NVIDIA, MPS for Apple Silicon)
# If False, will always use CPU regardless of available hardware
USE_GPU = True  # Set to False to force CPU-only execution

MAX_NEW_TOKENS = 1

# Quantization settings
# Options: 4, 8, or None (default is None for full precision)
#
# To enable quantization, change QUANTIZATION_BITS to one of the following:
#   QUANTIZATION_BITS = 4   # 4-bit quantization: ~1.5 GB memory for tiny, ~3-4 GB for 7-8B
#   QUANTIZATION_BITS = 8   # 8-bit quantization: ~2.5 GB memory for tiny, ~7-8 GB for 7-8B
#   QUANTIZATION_BITS = None  # No quantization: ~5 GB for tiny, ~28 GB for 7-8B (full precision, best quality)
#
# Notes:
# - Quantization requires the 'bitsandbytes' package: pip install bitsandbytes
# - Quantization only works with CUDA (NVIDIA GPUs), not with Apple Metal (MPS)
# - If using Apple Silicon, quantization will be automatically disabled
# - For 7-8B models, you may want to use quantization if VRAM is limited

QUANTIZATION_BITS = 4 # Change to 4 or 8 to enable quantization

# 10 MMLU subjects for evaluation
MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy", 
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine"
]


class TimingInfo:
    """Class to track timing information for model evaluation"""
    def __init__(self):
        self.real_time = 0
        self.cpu_time = 0
        self.gpu_time = 0
        
    def __str__(self):
        return f"Real: {self.real_time:.2f}s, CPU: {self.cpu_time:.2f}s, GPU: {self.gpu_time:.2f}s"


def detect_device():
    """Detect the best available device (CUDA, MPS, or CPU)"""

    # If GPU is disabled, always use CPU
    if not USE_GPU:
        return "cpu"

    # Check for CUDA
    if torch.cuda.is_available():
        return "cuda"

    # Check for Apple Silicon with Metal
    if torch.backends.mps.is_available():
        # Check if we're actually on Apple ARM
        is_apple_arm = platform.system() == "Darwin" and platform.processor() == "arm"

        if is_apple_arm:
            # Metal is available but incompatible with quantization
            if QUANTIZATION_BITS is not None:
                print("\n" + "="*70)
                print("ERROR: Metal and Quantization Conflict")
                print("="*70)
                print("Metal Performance Shaders (MPS) is incompatible with quantization.")
                print(f"You have USE_GPU = True and QUANTIZATION_BITS = {QUANTIZATION_BITS}")
                print("")
                print("Please choose one of the following options:")
                print("  1. Set USE_GPU = False to use CPU with quantization")
                print("  2. Set QUANTIZATION_BITS = None to use Metal without quantization")
                print("="*70 + "\n")
                sys.exit(1)
            return "mps"

    # Default to CPU
    return "cpu"


def check_environment():
    global QUANTIZATION_BITS
    """Check environment and dependencies"""
    print("="*70)
    print("Environment Check")
    print("="*70)

    # Check if in Colab
    try:
        import google.colab
        print("✓ Running in Google Colab")
        in_colab = True
    except:
        print("✓ Running locally (not in Colab)")
        in_colab = False

    # Check system info
    print(f"✓ Platform: {platform.system()} ({platform.machine()})")
    if platform.system() == "Darwin":
        print(f"✓ Processor: {platform.processor()}")

    # Detect and set device
    device = detect_device()

    # Check device
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU Available: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.2f} GB")
    elif device == "mps":
        print("✓ Apple Metal (MPS) Available")
        print("✓ Using Metal Performance Shaders for GPU acceleration")
    else:
        print("⚠️  No GPU detected - running on CPU")
       
    # Check quantization support

    if QUANTIZATION_BITS is not None:
        try:
            import bitsandbytes
            print(f"✓ bitsandbytes installed - {QUANTIZATION_BITS}-bit quantization available")
        except ImportError:
            print(f"❌ bitsandbytes NOT installed - cannot use quantization")
            sys.exit(1)
        if device == 'mps':
            print(f"❌ Apple METAL is incompatible with quantization")
            print("✓ Quantization disabled - loading full precision model")
            QUANTIZATION_BITS = None
            sys.exit(1)
    else:
        print("✓ Quantization disabled - loading full precision model")
    
    # Check HF authentication
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            print("✓ Hugging Face authenticated")
        else:
            print("⚠️  No Hugging Face token found")
            print("Run: huggingface-cli login")
    except:
        print("⚠️  Could not check Hugging Face authentication")
    
    # Print configuration
    print("\n" + "="*70)
    print("Configuration")
    print("="*70)
    print(f"Models: {len(MODELS)}")
    for model in MODELS:
        print(f"  - {model}")
    print(f"Device: {device}")
    if QUANTIZATION_BITS is not None:
        print(f"Quantization: {QUANTIZATION_BITS}-bit")
        if QUANTIZATION_BITS == 4:
            print(f"Expected memory per tiny model (1-1.5B): ~1.5 GB")
            print(f"Expected memory per medium model (7-8B): ~3-4 GB")
        elif QUANTIZATION_BITS == 8:
            print(f"Expected memory per tiny model (1-1.5B): ~2.5 GB")
            print(f"Expected memory per medium model (7-8B): ~7-8 GB")
    else:
        print(f"Quantization: None (full precision)")
        if device == "cuda":
            print(f"Expected memory per tiny model (1-1.5B): ~2.5 GB (FP16)")
            print(f"Expected memory per medium model (7-8B): ~14 GB (FP16)")
        elif device == "mps":
            print(f"Expected memory per tiny model (1-1.5B): ~2.5 GB (FP16)")
            print(f"Expected memory per medium model (7-8B): ~14 GB (FP16)")
        else:
            print(f"Expected memory per tiny model (1-1.5B): ~5 GB (FP32)")
            print(f"Expected memory per medium model (7-8B): ~28 GB (FP32)")
    print(f"Number of subjects: {len(MMLU_SUBJECTS)}")

    print("="*70 + "\n")
    return in_colab, device


def get_quantization_config():
    """Create quantization config based on settings"""
    if QUANTIZATION_BITS is None:
        return None
    
    if QUANTIZATION_BITS == 4:
        # 4-bit quantization (most memory efficient)
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # Double quantization for extra compression
            bnb_4bit_quant_type="nf4"  # NormalFloat4 - better for LLMs
        )
        print("Using 4-bit quantization (NF4 + double quant)")
        print("Memory usage: ~1.5 GB")
    elif QUANTIZATION_BITS == 8:
        # 8-bit quantization (balanced)
        config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
        )
        print("Using 8-bit quantization")
        print("Memory usage: ~2.5 GB")
    else:
        print(f"Invalid QUANTIZATION_BITS value: {QUANTIZATION_BITS}")
        print("Must be 4, 8, or None")
        sys.exit(1)
    
    return config


def load_model_and_tokenizer(model_name, device):
    """Load model and tokenizer with optional quantization"""
    print(f"\n{'='*70}")
    print(f"Loading model: {model_name}")
    print(f"{'='*70}")
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✓ Tokenizer loaded")
        
        # Get quantization config
        quant_config = get_quantization_config()
        
        # Load model
        print(f"Loading model on {device}...")
        if quant_config is not None:
            # Quantized loading (only works on CUDA)
            if device != "cuda":
                print(f"❌ Quantization only works with CUDA, but device is {device}")
                sys.exit(1)
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            # Non-quantized loading
            if device == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                model = model.to(device)
            elif device == "mps":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                model = model.to(device)
            else:  # CPU
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                model = model.to(device)

        model.eval()

        # Print model info
        print("✓ Model loaded successfully!")
        print(f"  Model device: {next(model.parameters()).device}")
        print(f"  Model dtype: {next(model.parameters()).dtype}")

        # Check memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            memory_reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"  GPU Memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")

            # Check if using quantization
            if quant_config is not None:
                print(f"  Quantization: {QUANTIZATION_BITS}-bit active")
        elif device == "mps":
            print(f"  Running on Apple Metal (MPS)")

        return model, tokenizer
        
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("\nPossible causes:")
        print("1. No Hugging Face token - Run: huggingface-cli login")
        print("2. Model license not accepted - Visit model page on Hugging Face")
        print("3. bitsandbytes not installed - Run: pip install bitsandbytes")
        print("4. Out of memory - Try 4-bit quantization or smaller model")
        raise


def format_mmlu_prompt(question, choices):
    """Format MMLU question as multiple choice"""
    choice_labels = ["A", "B", "C", "D"]
    prompt = f"{question}\n\n"
    for label, choice in zip(choice_labels, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt


def get_model_prediction(model, tokenizer, prompt, timing_info, device):
    """Get model's prediction for multiple-choice question with timing
    
    Returns:
        tuple: (answer, raw_output) where answer is one of A/B/C/D or None if invalid
    """
    # Start timing
    start_real = time.time()
    start_cpu = time.process_time()
    
    # GPU timing (if available)
    if device == "cuda":
        torch.cuda.synchronize()
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)
        gpu_start.record()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=1.0
        )
    
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    
    # End timing
    if device == "cuda":
        gpu_end.record()
        torch.cuda.synchronize()
        gpu_time = gpu_start.elapsed_time(gpu_end) / 1000.0  # Convert to seconds
        timing_info.gpu_time += gpu_time
    
    end_cpu = time.process_time()
    end_real = time.time()
    
    timing_info.real_time += (end_real - start_real)
    timing_info.cpu_time += (end_cpu - start_cpu)
    
    # Try to extract a valid answer (A, B, C, or D)
    answer = generated_text.strip()[:1].upper()
    
    if answer not in ["A", "B", "C", "D"]:
        # Try to find A/B/C/D anywhere in the response
        for char in generated_text.upper():
            if char in ["A", "B", "C", "D"]:
                answer = char
                break
        else:
            # No valid answer found - return None to indicate invalid response
            answer = None
    
    return answer, generated_text


def evaluate_subject(model, tokenizer, subject, device, verbose=False):
    """Evaluate model on a specific MMLU subject"""
    print(f"\n{'='*70}")
    print(f"Evaluating subject: {subject}")
    print(f"{'='*70}")
    
    try:
        dataset = load_dataset("cais/mmlu", subject, split="test")
    except Exception as e:
        print(f"❌ Error loading subject {subject}: {e}")
        return None
    
    correct = 0
    total = 0
    invalid_responses = 0
    timing_info = TimingInfo()
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Detailed Results for {subject}")
        print(f"{'='*70}")
    
    for i, example in enumerate(tqdm(dataset, desc=f"Testing {subject}", leave=True, disable=verbose), 1):
        question = example["question"]
        choices = example["choices"]
        correct_answer_idx = example["answer"]
        correct_answer = ["A", "B", "C", "D"][correct_answer_idx]
        
        prompt = format_mmlu_prompt(question, choices)
        predicted_answer, raw_output = get_model_prediction(model, tokenizer, prompt, timing_info, device)
        
        # Check if response was valid
        if predicted_answer is None:
            invalid_responses += 1
            is_correct = False
        else:
            is_correct = predicted_answer == correct_answer
            if is_correct:
                correct += 1
        
        total += 1
        
        # Print immediately in verbose mode
        if verbose:
            print(f"\nQuestion {i}/{len(dataset)}:")
            print(f"Q: {question}")
            for j, choice in enumerate(choices):
                print(f"  {['A', 'B', 'C', 'D'][j]}. {choice}")
            print(f"Correct Answer: {correct_answer}")
            if predicted_answer is None:
                print(f"Model Answer: INVALID (raw output: '{raw_output}')")
                print(f"Result: ✗ INVALID RESPONSE")
            else:
                print(f"Model Answer: {predicted_answer}")
                status = "✓ CORRECT" if is_correct else "✗ WRONG"
                print(f"Result: {status}")
    
    accuracy = (correct / total * 100) if total > 0 else 0
    valid_responses = total - invalid_responses
    valid_accuracy = (correct / valid_responses * 100) if valid_responses > 0 else 0
    
    print(f"\n✓ Result: {correct}/{total} correct = {accuracy:.2f}%")
    if invalid_responses > 0:
        print(f"⚠️  Invalid responses: {invalid_responses}/{total} ({invalid_responses/total*100:.1f}%)")
        print(f"✓ Accuracy on valid responses: {correct}/{valid_responses} = {valid_accuracy:.2f}%")
    print(f"✓ Timing: {timing_info}")
    
    return {
        "subject": subject,
        "correct": correct,
        "total": total,
        "invalid_responses": invalid_responses,
        "accuracy": accuracy,
        "valid_accuracy": valid_accuracy,
        "timing": {
            "real_time": timing_info.real_time,
            "cpu_time": timing_info.cpu_time,
            "gpu_time": timing_info.gpu_time
        }
    }


def evaluate_model(model_name, device, verbose=False):
    """Evaluate a single model on all subjects"""
    print(f"\n{'='*70}")
    print(f"EVALUATING MODEL: {model_name}")
    print(f"{'='*70}\n")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    
    # Evaluate
    results = []
    total_correct = 0
    total_questions = 0
    total_invalid = 0
    total_timing = TimingInfo()
    
    print(f"\n{'='*70}")
    print(f"Starting evaluation on {len(MMLU_SUBJECTS)} subjects")
    print(f"{'='*70}\n")
    
    for i, subject in enumerate(MMLU_SUBJECTS, 1):
        print(f"\nProgress: {i}/{len(MMLU_SUBJECTS)} subjects")
        result = evaluate_subject(model, tokenizer, subject, device, verbose)
        if result:
            results.append(result)
            total_correct += result["correct"]
            total_questions += result["total"]
            total_invalid += result["invalid_responses"]
            total_timing.real_time += result["timing"]["real_time"]
            total_timing.cpu_time += result["timing"]["cpu_time"]
            total_timing.gpu_time += result["timing"]["gpu_time"]
    
    # Calculate overall accuracy
    overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
    total_valid = total_questions - total_invalid
    valid_accuracy = (total_correct / total_valid * 100) if total_valid > 0 else 0
    
    # Clean up model to free memory
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "model_name": model_name,
        "overall_accuracy": overall_accuracy,
        "valid_accuracy": valid_accuracy,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "total_invalid": total_invalid,
        "subject_results": results,
        "total_timing": {
            "real_time": total_timing.real_time,
            "cpu_time": total_timing.cpu_time,
            "gpu_time": total_timing.gpu_time
        }
    }


def main():
    """Main evaluation function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Multi-Model MMLU Evaluation')
    parser.add_argument('--verbose', action='store_true', 
                       help='Print each question, answer, and correctness')
    parser.add_argument('--models', choices=['all', 'tiny', 'medium'], default='all',
                       help='Which models to evaluate: all (default), tiny (1-1.5B), or medium (7-8B)')
    args = parser.parse_args()
    
    # Select models based on argument
    if args.models == 'tiny':
        models_to_eval = TINY_MODELS
        print("Evaluating TINY/SMALL models only (1-1.5B)")
    elif args.models == 'medium':
        models_to_eval = MEDIUM_MODELS
        print("Evaluating SMALL/MEDIUM models only (7-8B)")
    else:
        models_to_eval = MODELS
        print("Evaluating ALL models (tiny + medium)")
    
    print("\n" + "="*70)
    print("Multi-Model MMLU Evaluation")
    print("="*70 + "\n")

    # Check environment
    in_colab, device = check_environment()
    
    # Evaluate all models
    all_results = []
    
    start_time = datetime.now()
    
    for model_name in models_to_eval:
        model_result = evaluate_model(model_name, device, args.verbose)
        all_results.append(model_result)
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    # Print comparison summary
    print("\n" + "="*70)
    print("MULTI-MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"Device: {device}")
    print(f"Quantization: {QUANTIZATION_BITS}-bit" if QUANTIZATION_BITS else "None (full precision)")
    print(f"Total Subjects: {len(MMLU_SUBJECTS)}")
    print(f"Total Duration: {total_duration/60:.1f} minutes")
    print("="*70)
    
    # Print results for each model
    if args.models == 'all':
        # Show both categories separately
        print("\nTINY/SMALL MODELS (1-1.5B parameters):")
        print("-" * 70)
        for result in all_results[:len(TINY_MODELS)]:
            print(f"\n{result['model_name']}:")
            print(f"  Overall Accuracy: {result['overall_accuracy']:.2f}% ({result['total_correct']}/{result['total_questions']})")
            if result['total_invalid'] > 0:
                print(f"  Invalid Responses: {result['total_invalid']}/{result['total_questions']} ({result['total_invalid']/result['total_questions']*100:.1f}%)")
                print(f"  Accuracy on Valid: {result['valid_accuracy']:.2f}% ({result['total_correct']}/{result['total_questions']-result['total_invalid']})")
            print(f"  Timing:")
            print(f"    Real Time: {result['total_timing']['real_time']:.2f}s ({result['total_timing']['real_time']/60:.1f} min)")
            print(f"    CPU Time:  {result['total_timing']['cpu_time']:.2f}s ({result['total_timing']['cpu_time']/60:.1f} min)")
            print(f"    GPU Time:  {result['total_timing']['gpu_time']:.2f}s ({result['total_timing']['gpu_time']/60:.1f} min)")
        
        print("\n" + "="*70)
        print("\nSMALL/MEDIUM MODELS (7-8B parameters):")
        print("-" * 70)
        for result in all_results[len(TINY_MODELS):]:
            print(f"\n{result['model_name']}:")
            print(f"  Overall Accuracy: {result['overall_accuracy']:.2f}% ({result['total_correct']}/{result['total_questions']})")
            if result['total_invalid'] > 0:
                print(f"  Invalid Responses: {result['total_invalid']}/{result['total_questions']} ({result['total_invalid']/result['total_questions']*100:.1f}%)")
                print(f"  Accuracy on Valid: {result['valid_accuracy']:.2f}% ({result['total_correct']}/{result['total_questions']-result['total_invalid']})")
            print(f"  Timing:")
            print(f"    Real Time: {result['total_timing']['real_time']:.2f}s ({result['total_timing']['real_time']/60:.1f} min)")
            print(f"    CPU Time:  {result['total_timing']['cpu_time']:.2f}s ({result['total_timing']['cpu_time']/60:.1f} min)")
            print(f"    GPU Time:  {result['total_timing']['gpu_time']:.2f}s ({result['total_timing']['gpu_time']/60:.1f} min)")
    else:
        # Show single category
        category = "TINY/SMALL MODELS (1-1.5B parameters)" if args.models == 'tiny' else "SMALL/MEDIUM MODELS (7-8B parameters)"
        print(f"\n{category}:")
        print("-" * 70)
        for result in all_results:
            print(f"\n{result['model_name']}:")
            print(f"  Overall Accuracy: {result['overall_accuracy']:.2f}% ({result['total_correct']}/{result['total_questions']})")
            if result['total_invalid'] > 0:
                print(f"  Invalid Responses: {result['total_invalid']}/{result['total_questions']} ({result['total_invalid']/result['total_questions']*100:.1f}%)")
                print(f"  Accuracy on Valid: {result['valid_accuracy']:.2f}% ({result['total_correct']}/{result['total_questions']-result['total_invalid']})")
            print(f"  Timing:")
            print(f"    Real Time: {result['total_timing']['real_time']:.2f}s ({result['total_timing']['real_time']/60:.1f} min)")
            print(f"    CPU Time:  {result['total_timing']['cpu_time']:.2f}s ({result['total_timing']['cpu_time']/60:.1f} min)")
            print(f"    GPU Time:  {result['total_timing']['gpu_time']:.2f}s ({result['total_timing']['gpu_time']/60:.1f} min)")
    
    print("="*70)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quant_suffix = f"_{QUANTIZATION_BITS}bit" if QUANTIZATION_BITS else "_full"
    model_suffix = f"_{args.models}" if args.models != 'all' else ""
    output_file = f"multi_model_mmlu_results{quant_suffix}{model_suffix}_{timestamp}.json"
    
    output_data = {
        "quantization_bits": QUANTIZATION_BITS,
        "timestamp": timestamp,
        "device": str(device),
        "total_duration_seconds": total_duration,
        "subjects": MMLU_SUBJECTS,
        "model_selection": args.models,
        "tiny_models": TINY_MODELS,
        "medium_models": MEDIUM_MODELS,
        "evaluated_models": models_to_eval,
        "model_results": all_results
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Print best/worst subjects for each model
    for result in all_results:
        if len(result['subject_results']) > 0:
            sorted_results = sorted(result['subject_results'], key=lambda x: x["accuracy"], reverse=True)
            
            print(f"\n📊 Top 3 Subjects for {result['model_name'].split('/')[-1]}:")
            for i, subj_result in enumerate(sorted_results[:3], 1):
                print(f"  {i}. {subj_result['subject']}: {subj_result['accuracy']:.2f}%")
            
            print(f"\n📉 Bottom 3 Subjects for {result['model_name'].split('/')[-1]}:")
            for i, subj_result in enumerate(sorted_results[-3:], 1):
                print(f"  {i}. {subj_result['subject']}: {subj_result['accuracy']:.2f}%")
    
    # Colab-specific instructions
    if in_colab:
        print("\n" + "="*70)
        print("💾 To download results in Colab:")
        print("="*70)
        print(f"from google.colab import files")
        print(f"files.download('{output_file}')")
    
    print("\n✅ Evaluation complete!")
    return output_file


if __name__ == "__main__":
    try:
        output_file = main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()