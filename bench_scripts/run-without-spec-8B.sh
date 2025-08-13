export CUDA_VISIBLE_DEVICES=7
export VLLM_USE_V1=1

model=meta-llama/Llama-3.1-8B-Instruct
# model=meta-llama/Meta-Llama-3-8B-Instruct

# Main run
{
    echo "===== Main Run Parameters ====="
    echo "Threshold: $threshold"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "VLLM_USE_V1=$VLLM_USE_V1"
    echo "Target model: $model"
    echo "Method: None"
    echo "Enable filtering: False"
    echo "Log filtering info: False"
    echo "================================"
    python -m spec_bench \
        --target_model "$model" \
        --method "none" 
    echo "Main Run Done"
} > logs/without-spec-8B.log 2>&1
