#!/bin/bash

export CUDA_VISIBLE_DEVICES=6
export VLLM_USE_V1=1
model=meta-llama/Llama-3.1-8B-Instruct
# model=meta-llama/Meta-Llama-3-8B-Instruct

# for perc in $(seq 1 0.2 1); do
#     for threshold in $(seq 0.2 0.3 0.5); do
#         # "Warmup run", mainly for logging purpose
#         {
#             echo "===== Warmup Run Parameters ====="
#             echo "Threshold: $threshold"
#             echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
#             echo "VLLM_USE_V1=$VLLM_USE_V1"
#             echo "Target model: $model"
#             echo "Method: eagle3"
#             echo "Num spec tokens: 5"
#             echo "Enable filtering: True"
#             echo "Log filtering info: True"
#             echo "Threshold: $threshold"
#             echo "Percentage: $perc"
#             echo "================================="
#             python -m spec_bench \
#                 --target_model $model \
#                 --method "eagle3" \
#                 --num_spec_tokens 5 \
#                 --log_filtering_info True \
#                 --enable_draft_token_filtering True \
#                 --draft_token_filtering_threshold $threshold \
#                 --draft_token_filtering_percentage $perc
#             echo "Logging Round Done"
#         } > logs/with-filtering-8B-warmup-thresh-${threshold}-perc-${perc}.log 2>&1

#         # Main run
#         {
#             echo "===== Main Run Parameters ====="
#             echo "Threshold: $threshold"
#             echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
#             echo "VLLM_USE_V1=$VLLM_USE_V1"
#             echo "Target model: $model"
#             echo "Method: eagle3"
#             echo "Num spec tokens: 5"
#             echo "Enable filtering: True"
#             echo "Log filtering info: False"
#             echo "Threshold: $threshold"
#             echo "Percentage: $perc"
#             echo "================================"
#             python -m spec_bench \
#                 --target_model $model \
#                 --method "eagle3" \
#                 --num_spec_tokens 5 \
#                 --enable_draft_token_filtering True \
#                 --draft_token_filtering_threshold $threshold \
#                 --draft_token_filtering_percentage $perc
#             echo "Main Run Done"
#         } > logs/with-filtering-8B-thresh-${threshold}-perc-${perc}.log 2>&1

#     done
# done

for threshold in $(seq 0.2 0.3 0.2); do
    # # "Warmup run", mainly for logging purpose
    # {
    #     echo "===== Warmup Run Parameters ====="
    #     echo "Threshold: $threshold"
    #     echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    #     echo "VLLM_USE_V1=$VLLM_USE_V1"
    #     echo "Target model: $model"
    #     echo "Method: eagle3"
    #     echo "Num spec tokens: 5"
    #     echo "Enable filtering: True"
    #     echo "Log filtering info: True"
    #     echo "Threshold: $threshold"
    #     echo "================================="
    #     python -m spec_bench \
    #         --target_model $model \
    #         --method "eagle3" \
    #         --num_spec_tokens 5 \
    #         --log_filtering_info True \
    #         --enable_draft_token_filtering True \
    #         --draft_token_filtering_threshold $threshold
    #     echo "Logging Round Done"
    # } > logs/with-filtering-8B-warmup-thresh-${threshold}-perc-${perc}.log 2>&1

    # Main run
    {
        echo "===== Main Run Parameters ====="
        echo "Threshold: $threshold"
        echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        echo "VLLM_USE_V1=$VLLM_USE_V1"
        echo "Target model: $model"
        echo "Method: eagle3"
        echo "Num spec tokens: 5"
        echo "Enable filtering: True"
        echo "Log filtering info: False"
        echo "Threshold: $threshold"
        echo "================================"
        python -m spec_bench \
            --target_model $model \
            --method "eagle3" \
            --num_spec_tokens 5 \
            --enable_draft_token_filtering True \
            --draft_token_filtering_threshold $threshold
        echo "Main Run Done"
    } > logs/with-filtering-8B-thresh-${threshold}-perc-${perc}.log 2>&1

done