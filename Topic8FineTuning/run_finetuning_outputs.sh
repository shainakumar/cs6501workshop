#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p outputs

if [[ -x ../.venv/bin/python ]]; then
  PYTHON=../.venv/bin/python
elif [[ -x "$HOME/.venv/bin/python" ]]; then
  PYTHON="$HOME/.venv/bin/python"
else
  PYTHON=python3
fi

"$PYTHON" step1_load_explore_data.py | tee outputs/step1_load_explore_data_output.txt
"$PYTHON" step3_evaluate_base_model.py | tee outputs/step3_base_model_eval_output.txt
"$PYTHON" step5_train_step6_evaluate_finetuned.py | tee outputs/step5_train_step6_finetuned_eval_output.txt
"$PYTHON" step7_manual_novel_schema_tests.py | tee outputs/step7_manual_novel_schema_output.txt
