# SPDX-License-Identifier: Apache-2.0
import json
import os
import time
import argparse
import sys
sys.path.append("../benchmarks")

from benchmark_dataset import (ShareGPTDataset)
from common import (get_eagle_model,
                    get_output_filename)

from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer

# Common setups
SEED = 42
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V1"] = "1"

# Sampling setup
sampling_params = SamplingParams(temperature=0.0,
                                 max_tokens=1*1024,
                                 ignore_eos=False)
# Control expt size
REPEAT = 10
batch_sizes = [1, 16, 64, 128]
# REPEAT = 1
# batch_sizes = [2]


def get_dataset(dataset_name):
    if dataset_name == "sharegpt":
        dataset_path = "/data/lily/ShareGPT_V3_unfiltered_cleaned_split.json"
        dataset = ShareGPTDataset(
            dataset_path=dataset_path,
            random_seed=SEED,
        )
    else:
        raise Exception("Unable to find datasets.")
    return dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    # parser.add_argument("--draft_model", type=str, default="yuhuili/EAGLE-LLaMA3-Instruct-8B")
    parser.add_argument("--method", type=str, default="eagle3")
    parser.add_argument("--num_spec_tokens", type=int, default=5)
    parser.add_argument("--enable_draft_token_filtering", type=bool, default=False)
    parser.add_argument("--log_filtering_info", type=bool, default=False)
    parser.add_argument("--draft_token_filtering_threshold", type=float, default=-1)
    parser.add_argument("--draft_token_filtering_percentage", type=float, default=-1)

    return parser.parse_args()


def get_llm(args):
    if args.method == "none":
        speculative_config = None
    # elif args.method == "ngram":
    #     speculative_config={
    #         "method": "ngram",
    #         "num_speculative_tokens": 5,
    #         "prompt_lookup_max": 7,
    #         "prompt_lookup_min": 3,
    #     }
    elif args.method in ["eagle", "eagle3"]:
        speculative_config={
            "method": args.method,
            "model": get_eagle_model(args.target_model, args.method == "eagle3"),
            "num_speculative_tokens": args.num_spec_tokens,
            "enable_draft_token_filtering": args.enable_draft_token_filtering,
            "draft_token_filtering_threshold": args.draft_token_filtering_threshold,
            "log_filtering_info": args.log_filtering_info,
            "draft_token_filtering_percentage": args.draft_token_filtering_percentage,
        }

    llm = LLM(
            model=args.target_model,
            tensor_parallel_size=4 if "70" in args.target_model else 1,
            speculative_config=speculative_config,
            disable_log_stats=False,
            enable_prefix_caching=False,
            seed=SEED,
            dtype="bfloat16",
            trust_remote_code=True,
            enforce_eager=True, # NOTE: not in Lily's script
        )
    return llm

def main():
    args = parse_args()

    llm = get_llm(args)
    tokenizer = get_tokenizer(args.target_model,
                        tokenizer_mode="auto",
                        trust_remote_code=False)
    dataset = get_dataset('sharegpt')
    latencies = []
    all_requests = dataset.sample(
        num_requests=REPEAT,
        tokenizer=tokenizer,
        output_len=None,
    )

    for batch_size in batch_sizes:
        latencies = []
        for i in range(REPEAT):
            input_request = all_requests[i]
            prompts = [input_request.prompt] * batch_size
            start_time = time.time()
            outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
            end_time = time.time()
            duration = end_time - start_time
            result = {
                "duration": duration,
                "batch_size": batch_size,
                "model": args.target_model,
                "method": args.method,
                "dataset": "sharegpt",
                "prompt": input_request.prompt,
                "prompt_len": input_request.prompt_len,
                "output_len": len(outputs[0].outputs[0].token_ids),
                "generated_output": outputs[0].outputs[0].text,
                "finished_reason": outputs[0].outputs[0].finish_reason,
            }
            # If it's the logging round, don't create a result.json,
            # because the duration is usually inaccurate
            if not args.log_filtering_info:
                with open(get_output_filename(args), "a") as f:
                    f.write(json.dumps(result))
                    f.write("\n")
            latencies.append(duration)
        print(f"Average latency: {sum(latencies) / len(latencies)} seconds")

    # # Add a buffer to wait for profiler in the background process
    # # (in case MP is on) to finish writing profiling output.
    # time.sleep(10)

if __name__ == "__main__":
    main()