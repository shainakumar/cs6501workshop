import random

import numpy as np
import tinker
from tinker import types

from finetuning_common import (
    BASE_MODEL,
    evaluate_test_set,
    load_split_data,
    process_train_set,
)


NUM_EPOCHS = 1
BATCH_SIZE = 256
LEARNING_RATE = 5e-4


_, train_data, test_data = load_split_data()

service_client = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(base_model=BASE_MODEL)
tokenizer = training_client.get_tokenizer()
processed_train = process_train_set(train_data, tokenizer)

print("--- Training Fine-Tuned Model ---")
print(f"Training examples: {len(processed_train)}")
print(f"Batch size: {BATCH_SIZE}")

step = 0
for epoch in range(NUM_EPOCHS):
    random.shuffle(processed_train)
    for batch_idx in range(0, len(processed_train), BATCH_SIZE):
        batch = processed_train[batch_idx : batch_idx + BATCH_SIZE]
        if not batch:
            break

        fwdbwd_future = training_client.forward_backward(batch, "cross_entropy")
        optim_future = training_client.optim_step(
            types.AdamParams(learning_rate=LEARNING_RATE)
        )

        fwdbwd_result = fwdbwd_future.result()
        optim_future.result()

        to_arr = lambda x: x.to_numpy() if hasattr(x, "to_numpy") else np.array(x.tolist())
        logprobs = np.concatenate(
            [to_arr(output["logprobs"]) for output in fwdbwd_result.loss_fn_outputs]
        )
        weights = np.concatenate(
            [to_arr(datum.loss_fn_inputs["weights"]) for datum in batch]
        )
        loss = float(-np.dot(logprobs, weights) / (weights.sum() + 1e-8))

        step += 1
        if step % 100 == 0 or batch_idx + BATCH_SIZE >= len(processed_train):
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, update {step}, loss: {loss:.4f}")

sampling_path = training_client.save_weights_for_sampler(name="checkpoint").result().path
print(f"Model saved to: {sampling_path}")

sampling_client = service_client.create_sampling_client(model_path=sampling_path)
accuracy = evaluate_test_set(sampling_client, tokenizer, test_data)
print(f"Finetuned model accuracy: {accuracy:.2%} ({int(accuracy * 200)}/200)")
print(f"Use this checkpoint for Step 7: {sampling_path}")
