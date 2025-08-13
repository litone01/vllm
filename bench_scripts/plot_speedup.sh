python plot_speedup.py \
  --dataset sharegpt \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --org-file /home/eecs/xiaoxuanliu/jerry/vllm/bench_scripts/results/without_spec/latency_filtering_False_threshold_1_logging_False_sharegpt_none_all_meta-llama_Llama-3.1-8B-Instruct.json \
  --eagle3 0 /home/eecs/xiaoxuanliu/jerry/vllm/bench_scripts/results/v0.1/latency_filtering_False_logging_False_sharegpt_eagle3_all_meta-llama_Llama-3.1-8B-Instruct.json \
  --eagle3 0.1 results/v0.1/latency_filtering_True_threshold_0.1_logging_False_sharegpt_eagle3_all_meta-llama_Llama-3.1-8B-Instruct.json \
  --eagle3 0.2 results/v0.1/latency_filtering_True_threshold_0.2_logging_False_sharegpt_eagle3_all_meta-llama_Llama-3.1-8B-Instruct.json \
  --eagle3 0.3 results/v0.1/latency_filtering_True_threshold_0.3_logging_False_sharegpt_eagle3_all_meta-llama_Llama-3.1-8B-Instruct.json \
  --eagle3 0.5 results/v0.1/latency_filtering_True_threshold_0.5_logging_False_sharegpt_eagle3_all_meta-llama_Llama-3.1-8B-Instruct.json \
  --eagle3 0.7 results/v0.1/latency_filtering_True_threshold_0.7_logging_False_sharegpt_eagle3_all_meta-llama_Llama-3.1-8B-Instruct.json