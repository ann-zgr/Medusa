
import argparse
import numpy as np
import json

def read_json_lines(path):
    with open(path, "r") as f:
        data = [json.loads(line) for line in f]
    return data

def parse_result(data):
    tokens = 0
    steps = 0
    times = 0
    for line in data:
        choices = line["choices"]
        for choice in choices:
            tokens +=np.sum(choice["new_tokens"])
            steps += np.sum(choice["idxs"])
            times += np.sum(choice["wall_time"])
    return tokens, steps, times

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--medusa", type=str, required=True, help="Path to the Medusa result file"
    )
    parser.add_argument(
        "--base", type=str, required=True, help="Path to the baseline result file"
    )
    args = parser.parse_args()
    medusa_json = read_json_lines(args.medusa)
    base_json = read_json_lines(args.base)
    medusa_tokens, medusa_steps, medusa_times = parse_result(medusa_json)
    base_tokens, base_steps, base_times = parse_result(base_json)
    print(f"Medusa: tokens={medusa_tokens}, steps={medusa_steps}, time={medusa_times:.2f}s, speed={medusa_tokens/medusa_times:.2f} tokens/s")
    print(f"Baseline: tokens={base_tokens}, steps={base_steps}, time={base_times:.2f}s, speed={base_tokens/base_times:.2f} tokens/s")
    print(f"Acc. Rate: { (medusa_tokens / medusa_steps) / (base_tokens / base_steps) }")
    print(f"Overhead: { (medusa_times / medusa_steps) / (base_times / base_steps) }")
    print(f"Speedup: { (medusa_tokens / medusa_times) / (base_tokens / base_times) }")

if __name__ == "__main__":
    main()