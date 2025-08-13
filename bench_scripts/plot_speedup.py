#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict

from common import THRESH_COLORS

import matplotlib.pyplot as plt

FIELD_TO_LABELS = {
    "duration": "Request Latency (s)",
    "output_len": "Output Length (tokens)",
}

# Threshold -> Percentage data
EAGLE3_DATA = {
    0.2: {
        0.3: [4.072782969474792, 4.661778545379638, 7.543803286552429, 11.207693219184875],
        0.5: [4.053783559799195, 4.6393190145492555, 7.532944655418396, 11.274755573272705],
        0.7: [4.143309164047241, 4.726174521446228, 7.663419628143311, 11.409644603729248],
        1: [4.091158223152161, 4.789428019523621, 7.714113664627075, 11.360840678215027],
        -1: [4.1352471828460695, 4.47955539226532, 5.5530320088068645, 7.023788952827454],
        -1: [4.168939065933228, 4.868376135826111, 7.766388320922852, 11.498052334785461]
    },
    0.5: {
        0.3: [4.722144865989685, 5.349060702323913, 7.614322805404663, 10.972771263122558],
        0.5: [4.72994704246521, 5.32381911277771, 7.675563097000122, 10.983605217933654],
        0.7: [4.739162445068359, 5.348842692375183, 7.615800523757935, 11.08501365184784],
        1: [4.788789010047912, 5.437917995452881, 7.665308666229248, 11.166690707206726],
        -1: [5.091900491714478, 5.7633391380310055, 7.928857040405274, 11.462403798103333],
    }
}


def load_data(filename):
    """
    Loads line‐delimited JSON records from `filename` into a nested dict
    keyed by batch_size then request_id.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    data = defaultdict(dict)
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            batch = record["batch_size"]
            req_id = hash(record["prompt"])
            data[batch][req_id] = record
    return data


def plot_speedups(dataset, model, field,
                  org_data,
                  ngram_data=None,
                  eagle_data=None,
                  eagle3_data_dict=None):
    """
    Plots speedups vs batch size for org, ngram, eagle and multiple eagle3 thresholds.
    - org_data:     baseline data
    - ngram_data:   optional dict
    - eagle_data:   optional dict
    - eagle3_data_dict: dict mapping threshold→data dict
    """
    plt.figure(figsize=(4, 3.5))
    batches = sorted(org_data.keys())
    org_lat = [
        sum(rec[field] for rec in org_data[b].values()) / len(org_data[b])
        for b in batches
    ]
    print(f"Baseline: {org_lat}")

    # optional: plot ngram
    if ngram_data:
        ngram_lat = [
            sum(rec[field] for rec in ngram_data[b].values()) / len(ngram_data[b])
            for b in batches
        ]
        speed_n = [o / n if n else None for o, n in zip(org_lat, ngram_lat)]
        plt.plot(batches, speed_n, marker='s', linewidth=2, label="N-gram")

    # optional: plot eagle
    if eagle_data:
        eagle_lat = [
            sum(rec[field] for rec in eagle_data[b].values()) / len(eagle_data[b])
            for b in batches
        ]
        speed_e = [o / e if e else None for o, e in zip(org_lat, eagle_lat)]

        plt.plot(batches, speed_e, marker='o', linewidth=2, label="Eagle")

    # plot each eagle3 threshold
    for thr, data in eagle3_data_dict.items():
        eagle3_lat = [
            sum(rec[field] for rec in data[b].values()) / len(data[b])
            for b in batches
        ]
        speed_e3 = [o / e3 if e3 else None for o, e3 in zip(org_lat, eagle3_lat)]

        print(f"Eagle3-thr-{thr}: {eagle3_lat}")

        if thr == 0:
            plt.plot(batches, speed_e3, marker='*', linestyle="dotted", linewidth=2, color=THRESH_COLORS.get(thr),
                 label=f"Eagle3")
        else:
            plt.plot(batches, speed_e3, marker='*', linewidth=2, color=THRESH_COLORS.get(thr),
                 label=f"Eagle3 (thr={thr})")


    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Speedup", fontsize=12)
    # plt.title(f"{dataset} | {model}", fontsize=14)
    plt.ylim((1.0, 2.3))
    plt.xticks([1, 16, 64, 128])
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    out_dir = "figures/paper"
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{dataset}_{model.replace('/', '_')}_{field}_speedup_varying_threshold_v0.1_corrected.png"
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Plot speedup of Eagle3 at various thresholds"
    )
    p.add_argument("--dataset",     required=True,
                   help="Dataset name (e.g. sharegpt)")
    p.add_argument("--model",       required=True,
                   help="Model identifier (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    p.add_argument("--field",       default="duration", choices=FIELD_TO_LABELS,
                   help="Which metric to plot")
    p.add_argument("--org-file",    required=True,
                   help="Path to baseline JSONL results")
    p.add_argument("--ngram-file",
                   help="Optional N-gram JSONL")
    p.add_argument("--eagle-file",
                   help="Optional Eagle JSONL")
    p.add_argument("--eagle3", nargs=2, action="append", metavar=("THR", "FILE"),
                   required=True,
                   help="Provide pairs: threshold and path to Eagle3 JSONL; "
                        "repeat for multiple thresholds")
    args = p.parse_args()

    # load baseline
    org = load_data(args.org_file)

    # optional variants
    ngram = load_data(args.ngram_file) if args.ngram_file else None
    eagle = load_data(args.eagle_file) if args.eagle_file else None

    # eagle3 thresholds
    e3_dict = {}
    for thr_str, fn in args.eagle3:
        try:
            thr = float(thr_str)
        except ValueError:
            p.error(f"Threshold must be a number, got '{thr_str}'")
        e3_dict[thr] = load_data(fn)

    plot_speedups(args.dataset,
                  args.model,
                  args.field,
                  org,
                  ngram_data=ngram,
                  eagle_data=eagle,
                  eagle3_data_dict=e3_dict)
