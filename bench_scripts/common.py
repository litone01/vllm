import argparse

THRESH_COLORS = {
    0: "#9467bd",
    0.1: "#1f77b4",  # blue
    0.2: "#ff7f0e",  # orange
    0.3: "#2ca02c",  # green
    0.4: "#17becf",  # red
    0.5: "darkgreen",
    0.6: "#8c564b",  # pink
    0.7: "#e377c2",  # pinke
}
BASELINE_COLOR = "#d62728"  # red

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--model",
#         type=str,
#         default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
#         help="Model name or path.",
#     )
#     parser.add_argument(
#         "--dataset",
#         choices=["aime", "sonnet", "mtbench", "instructcode", "sharegpt", "cnndailymail"],
#         default="sharegpt",
#         help="Dataset to use for generating requests.",
#     )
#     parser.add_argument(
#         "--method",
#         choices=["none", "ngram", "eagle", "eagle3"],
#         default="ngram",
#         help="Speculative decoding method to use.",
#     )
#     args = parser.parse_args()
#     return args

def get_eagle_model(model, use_eagle3=False):
    if model == "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":
        return "yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B"
    elif model == "meta-llama/Llama-3.1-8B-Instruct":
        if use_eagle3:
            return "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
        else:
            return "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
    elif model == "meta-llama/Meta-Llama-3-8B-Instruct":
        if use_eagle3:
            raise ValueError("EAGLE3 is not supported for Meta-Llama-3-8B-Instruct.")
        else:
            return "yuhuili/EAGLE-LLaMA3-Instruct-8B"
    elif model == "meta-llama/Meta-Llama-3-70B-Instruct":
        if use_eagle3:
            raise ValueError("EAGLE3 is not supported for meta-llama/Meta-Llama-3-70B-Instruct model.")
        else:
            return "yuhuili/EAGLE-LLaMA3-Instruct-70B"

    else:
        raise ValueError(f"Unsupported model for EAGLE: {model}.")

def get_output_filename(args):
    # TODO: rewrite this! We are getting more and more params
    return f"results/latency_filtering_{args.enable_draft_token_filtering}_threshold_{args.draft_token_filtering_threshold}_perc_{args.draft_token_filtering_percentage}_{"sharegpt"}_{args.method}_all_{args.target_model.replace('/', '_')}.json"

def get_output_filename_for_plotting(args):
    return f"results/latency_filtering_{args.enable_draft_token_filtering}_threshold_{args.draft_token_filtering_threshold}_{"sharegpt"}_{args.method}_all_{args.target_model.replace('/', '_')}.json"