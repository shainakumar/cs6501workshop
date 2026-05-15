from finetuning_common import load_split_data


data, train_data, test_data = load_split_data()

print(f"Total examples: {len(data)}")
print(f"Training examples: {len(train_data)}")
print(f"Test examples: {len(test_data)}")

print("\nSample example:")
example = data[0]
print(f"  Question: {example['question']}")
print(f"  Context:  {example['context'][:120]}...")
print(f"  Answer:   {example['answer']}")
