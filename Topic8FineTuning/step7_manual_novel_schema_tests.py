import os
import sys

import tinker

from finetuning_common import sample_from_model


DEFAULT_CHECKPOINT = "tinker://152d5797-a881-5247-902b-1dd6875e363e:train:0/sampler_weights/checkpoint"

model_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("TINKER_MODEL_PATH", DEFAULT_CHECKPOINT)

service_client = tinker.ServiceClient()
sampling_client = service_client.create_sampling_client(model_path=model_path)
tokenizer = sampling_client.get_tokenizer()

novel_questions = [
    {
        "question": "What are the names of employees in the engineering department?",
        "context": "CREATE TABLE employees (id INTEGER, name VARCHAR, salary REAL, department VARCHAR)",
    },
    {
        "question": "How many products cost more than 50 dollars?",
        "context": "CREATE TABLE products (id INTEGER, name VARCHAR, price REAL, category VARCHAR)",
    },
    {
        "question": "What is the highest score in the science class?",
        "context": "CREATE TABLE students (id INTEGER, name VARCHAR, score INTEGER, class VARCHAR)",
    },
    {
        "question": "List the top 3 customers by total order amount.",
        "context": "CREATE TABLE orders (id INTEGER, customer VARCHAR, amount REAL, date VARCHAR)",
    },
    {
        "question": "How many students are enrolled in each department?",
        "context": (
            "CREATE TABLE courses (id INTEGER, name VARCHAR, department VARCHAR); "
            "CREATE TABLE enrollments (student_id INTEGER, course_id INTEGER, grade VARCHAR)"
        ),
    },
]

print(f"Using checkpoint: {model_path}")
for ex in novel_questions:
    model_response = sample_from_model(
        sampling_client, tokenizer, ex["context"], ex["question"]
    )
    print("Input:", ex)
    print("Model Output:", model_response)
    print("==" * 60)
