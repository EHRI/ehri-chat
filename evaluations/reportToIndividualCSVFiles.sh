#!/bin/bash

modes=("vanilla" "rag" "graphrag" "mcp")
models=("mistral" "gemini")

for mode in "${modes[@]}"
do
  for model in "${models[@]}"
  do
    python report.py --format csv --mode "${mode}" --model "${model}" --output "report_${mode}_${model}.csv"
  done
done