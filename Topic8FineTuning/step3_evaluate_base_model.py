import tinker

from finetuning_common import BASE_MODEL, evaluate_test_set, load_split_data


_, _, test_data = load_split_data()

service_client = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(base_model=BASE_MODEL)
tokenizer = training_client.get_tokenizer()

print("\n--- Evaluating Base Model on 200 Test Questions ---")
base_sampling_client = training_client.save_weights_and_get_sampling_client(
    name="base-model"
)
base_accuracy = evaluate_test_set(base_sampling_client, tokenizer, test_data)
print(f"Base model accuracy: {base_accuracy:.2%} ({int(base_accuracy * 200)}/200)")
