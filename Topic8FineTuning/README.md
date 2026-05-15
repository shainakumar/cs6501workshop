# Topic 8 Fine Tuning

## Before vs. After

Before fine-tuning, the base `meta-llama/Llama-3.2-1B` model got 82 out of 200 held-out text-to-SQL questions correct, which was 41.00% accuracy. After LoRA fine-tuning with Tinker, the model got 173 out of 200 correct, which was 86.50% accuracy. Thus, we observed a 45.5 percentage point improvement. The model seemed to improve at both SQL syntax and schema grounding. After fine-tuning, it more consistently produced SQL-shaped answers instead of extra explanation or unrelated text. It also became better at mapping natural language questions onto common SQL patterns like `COUNT`, `MAX`, `WHERE`, `GROUP BY`, `ORDER BY`, and `LIMIT`. However, the Step 7 manual questions showed that schema grounding was still imperfect. The model did well on some novel schemas (like finding the maximum science score and listing the top 3 customers by total order amount), but it also made clear mistakes 2/5 times (for the product-count question, it used `SUM(id)` and added unsupported filters, and for the enrollment question, it hallucinated a `students` table that was not in the schema). The model was less reliable on the additional, novel test questions.

## RAG Comparison

A RAG system with 1,000 question-SQL examples in a vector database would work well when a new question closely matches a example stored in the database. For example, questions like "How many products cost more than 50 dollars?" or "What is the highest score in the science class?" would likely retrieve similar examples using `COUNT(*)`, `WHERE`, or `MAX`. It would struggle more on questions that require adapting to a new/unfamiliar schema or composing multiple operations. 

## Error Analysis

When the fine-tuned model was wrong, the errors were mostly logic and schema-grounding errors rather than basic SQL syntax errors. The product-count example should have used `COUNT(*)`, but the model generated `SUM(id)` and added unnecessary filters like `category = 'food'` and `name = 'food'`. That suggests the model learned common SQL surface patterns but did not fully understand the intent of "how many."

The enrollment example showed a schema hallucination error. The provided schema had `courses` and `enrollments`, but the model referenced a nonexistent `students` table. This means the model was not always constrained by the schema in the prompt.

## Directory Contents

| Path | Purpose |
| --- | --- |
| `Topic8FineTuning.ipynb` | Topic 8 notebook |
| `step1_load_explore_data.py` | Load and inspect dataset |
| `step3_evaluate_base_model.py` | Evaluate base model |
| `step5_train_step6_evaluate_finetuned.py` | Train and evaluate fine-tuned model |
| `step7_manual_novel_schema_tests.py` | Manual novel-schema tests |
| `sql_matches.py`, `finetuning_common.py` | Shared helpers |
| `run_finetuning_outputs.sh` | Runs scripts and saves outputs |
| `outputs/` | Saved terminal outputs |
