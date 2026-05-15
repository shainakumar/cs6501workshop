import json
import os
import random
from pathlib import Path

import numpy as np
from sql_matches import sql_matches
from tinker import types


BASE_MODEL = "meta-llama/Llama-3.2-1B"
NUM_TEST_EXAMPLES = 200


def find_dataset_path() -> Path:
    candidates = []
    if os.getenv("SQL_DATA_PATH"):
        candidates.append(Path(os.environ["SQL_DATA_PATH"]))
    root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            Path.cwd() / "sql_create_context_v4.json",
            Path.cwd() / "sql_create_context_4v.json",
            root / "sql_create_context_v4.json",
            root / "sql_create_context_4v.json",
            root / "Topic8FineTuning" / "sql_create_context_v4.json",
            root / "Topic8FineTuning" / "sql_create_context_4v.json",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find sql_create_context_v4.json. Put it in the repo root, "
        "Topic8FineTuning, or set SQL_DATA_PATH=/path/to/file."
    )


def load_split_data(seed: int = 0):
    with open(find_dataset_path()) as f:
        data = json.load(f)
    rng = random.Random(seed)
    rng.shuffle(data)
    test_data = data[:NUM_TEST_EXAMPLES]
    train_data = data[NUM_TEST_EXAMPLES:]
    return data, train_data, test_data


def sample_from_model(sampling_client, tokenizer, context: str, question: str) -> str:
    prompt = f"""Table schema:
{context}
Question: {question}
SQL: """
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    model_input = types.ModelInput.from_ints(tokens=prompt_tokens)
    params = types.SamplingParams(
        max_tokens=150, temperature=0.0, stop=["\n\n", "Question:"]
    )
    result = sampling_client.sample(
        prompt=model_input, sampling_params=params, num_samples=1
    ).result()
    if result.sequences:
        return tokenizer.decode(result.sequences[0].tokens).strip()
    return ""


def eval_one(sampling_client, tokenizer, ex: dict) -> bool:
    sql = sample_from_model(sampling_client, tokenizer, ex["context"], ex["question"])
    return sql_matches(sql, ex["answer"], schema=ex["context"])


def evaluate_test_set(sampling_client, tokenizer, test_data: list) -> float:
    correct = sum(1 for ex in test_data if eval_one(sampling_client, tokenizer, ex))
    return correct / len(test_data)


def format_prompt(example: dict) -> tuple[str, str]:
    prompt = f"""Table schema:
{example['context']}
Question: {example['question']}
SQL: """
    return prompt, example["answer"]


def process_example(example: dict, tokenizer) -> types.Datum:
    prompt, completion = format_prompt(example)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0.0] * len(prompt_tokens)

    completion_tokens = tokenizer.encode(f" {completion}\n\n", add_special_tokens=False)
    completion_weights = [1.0] * len(completion_tokens)

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "target_tokens": np.array(target_tokens, dtype=np.int64),
            "weights": np.array(weights, dtype=np.float32),
        },
    )


def process_train_set(train_data: list, tokenizer) -> list[types.Datum]:
    processed_train = [process_example(ex, tokenizer) for ex in train_data]
    random.shuffle(processed_train)
    return processed_train
