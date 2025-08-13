#!/usr/bin/env python3
import argparse
import os
import sys
import re

def print_avg_latency(log_path):
    """
    Read a log, split on 'Average latency:', and count 'Draft: ' in
    each of the first max_sections chunks.
    """
    if not os.path.exists(log_path):
        print(f"Error: file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex pattern to match "Average latency: <number> seconds"
    pattern = r"Average latency:\s*([\d\.]+)\s*seconds"

    # Find all matches
    latencies = re.findall(pattern, content)

    # Convert to floats (optional)
    latencies = [float(lat) for lat in latencies]

    print(latencies)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count 'Draft: ' per chunk and plot differences "
                    "between baseline and Eagle3 thresholds."
    )
    parser.add_argument(
        "--log", "-l",
        required=True,
        help="Path to baseline log file"
    )

    args = parser.parse_args()

    print_avg_latency(args.log)
