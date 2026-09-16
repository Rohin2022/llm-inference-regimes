import argparse
import json
import os

import pandas as pd
import requests
from openai import OpenAI, BadRequestError

from prompts import retrieve_full_prompt


def get_metric_sum(metrics_url, metric_name):
    """Get the cumulative sum of a vLLM histogram/counter."""
    try:
        metrics = requests.get(metrics_url, timeout=5).text
    except requests.RequestException:
        return 0.0

    total = 0.0
    found = False

    for line in metrics.splitlines():
        if line.startswith(f"{metric_name}_sum"):
            total += float(line.split()[-1])
            found = True

    return total if found else 0.0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--llm_output_path", required=True)

    parser.add_argument(
        "--task",
        required=True,
        choices=["SUMMARIZATION", "CREATION"],
    )

    parser.add_argument("--max_tokens", type=int, required=True)

    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--metrics_url", default="http://localhost:8000/metrics")

    args = parser.parse_args()

    client = OpenAI(
        api_key="dummy",
        base_url=args.base_url,
    )

    model = client.models.list().data[0].id

    # Load input data
    df = pd.read_csv(args.csv_path)

    # Determine which column contains the input text
    text_column = "report" if args.task == "SUMMARIZATION" else "summary"

    if text_column not in df.columns:
        raise ValueError(
            f"Expected column '{text_column}' for task {args.task}. "
            f"Available columns: {list(df.columns)}"
        )

    # Load already-processed rows so the script can resume
    processed = set()

    if os.path.exists(args.output_path):
        existing = pd.read_csv(args.output_path)
        if "_row_idx" in existing.columns:
            processed = set(existing["_row_idx"].astype(int))

    print(f"Total rows: {len(df)}")
    print(f"Already processed: {len(processed)}")
    print(f"Remaining: {len(df) - len(processed)}")
    print(f"Task: {args.task}")
    print(f"Model: {model}")

    # Make sure output directories exist
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.llm_output_path) or ".", exist_ok=True)

    for row_idx, row in df.iterrows():

        if row_idx in processed:
            continue

        text = row[text_column]

        if pd.isna(text):
            print(f"Skipping row {row_idx}: empty {text_column}")
            continue

        prompt = retrieve_full_prompt(args.task, text)

        if prompt is None:
            raise ValueError(f"Invalid task: {args.task}")

        print(f"\nProcessing row {row_idx}...")

        # Metrics before request
        prefill_before = get_metric_sum(
            args.metrics_url,
            "vllm:request_prefill_time_seconds",
        )

        decode_before = get_metric_sum(
            args.metrics_url,
            "vllm:request_decode_time_seconds",
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=args.max_tokens,
            )
        except BadRequestError as e:
            print(f"Skipping row {row_idx}: {e}")
            continue

        # Metrics after request
        prefill_after = get_metric_sum(
            args.metrics_url,
            "vllm:request_prefill_time_seconds",
        )

        decode_after = get_metric_sum(
            args.metrics_url,
            "vllm:request_decode_time_seconds",
        )

        prefill_time = prefill_after - prefill_before
        decode_time = decode_after - decode_before
        total_time = prefill_time + decode_time

        output = response.choices[0].message.content

        # Token usage reported by OpenAI-compatible API
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None

        if response.usage is not None:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

        # Save metrics immediately
        result = {
            "_row_idx": row_idx,
            "task": args.task,
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": total_time,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        result_df = pd.DataFrame([result])

        result_df.to_csv(
            args.output_path,
            mode="a",
            header=not os.path.exists(args.output_path),
            index=False,
        )

        # Save generated output immediately as JSONL
        output_record = {
            "row_idx": row_idx,
            "task": args.task,
            "output": output,
        }

        with open(args.llm_output_path, "a") as f:
            f.write(json.dumps(output_record) + "\n")

        print(f"Prefill: {prefill_time:.6f} s")
        print(f"Decode:  {decode_time:.6f} s")
        print(f"Total:   {total_time:.6f} s")

    print("\nFinished.")


if __name__ == "__main__":
    main()