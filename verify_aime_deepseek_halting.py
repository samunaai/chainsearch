#!/usr/bin/env python3
"""
Validate the AIME non-halting claim for
DeepSeek-R1-Distill-Qwen-7B.

Defaults reproduce the released Zhang et al. generation settings:

    temperature = 0.6
    top_p = 0.95
    max_tokens = 32768

Use --max-tokens 30000 to test the paper's "30K tokens" wording
directly.

The JSONL output is append-only and resumable.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

SERVED_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

BASE_URL = "http://127.0.0.1:8000/v1"

# Local Hugging Face dataset repository containing all 30
# AIME I and AIME II 2025 problems.
DATASET = "/workspace/user/chainsearch/quickAIMETest/aime_2025"

TEMPERATURE = 0.6
TOP_P = 0.95

PROMPT_SUFFIX = (
    "\nPlease reason step by step, "
    "and put your final answer within \\boxed{}."
)

# AIME answers are integers from 000 to 999. Require a complete
# \boxed{...} expression rather than merely the opening "\boxed{".
BOXED_PATTERN = re.compile(
    r"\\boxed\s*\{\s*(\d{1,3})\s*\}"
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=SERVED_MODEL_NAME,
        help="Model name exposed by the local vLLM server.",
    )

    parser.add_argument(
        "--base-url",
        default=BASE_URL,
        help="OpenAI-compatible base URL of the local vLLM server.",
    )

    parser.add_argument(
        "--dataset",
        default=DATASET,
        help=(
            "Local AIME 2025 dataset directory or Hugging Face "
            "dataset ID."
        ),
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help=(
            "Use 32768 to match the released code, or 30000 "
            "to test the paper's 30K wording."
        ),
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Independent rollouts per AIME question.",
    )

    parser.add_argument(
        "--base-seed",
        type=int,
        default=12345,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional question limit for a cheap smoke test.",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL path.",
    )

    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    """Read valid rows while tolerating an interrupted final line."""

    if not path.exists():
        return []

    rows: list[dict] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(
                    f"Warning: skipping malformed line "
                    f"{line_number} in {path}"
                )

    return rows

def load_aime_dataset(source: str):
    """
    Load AIME 2025 from either a local Hugging Face repository
    or a Hub dataset ID.

    The local repository contains full, part1, and part2 Parquet
    files. We intentionally load only data/*.parquet, which is the
    combined 30-question dataset.
    """

    source_path = Path(source)

    if source_path.exists():
        parquet_files = sorted(
            (source_path / "data").glob("*.parquet")
        )

        if not parquet_files:
            raise FileNotFoundError(
                "No Parquet files found under "
                f"{source_path / 'data'}."
            )

        return load_dataset(
            "parquet",
            data_files={
                "train": [str(path) for path in parquet_files]
            },
            split="train",
        )

    return load_dataset(source, split="train")



def summarize(
    rows: list[dict],
    runs: int,
    requested_max_tokens: int,
) -> None:
    """
    Summarize strict API truncation.

    We define a truncated/non-halting generation as one whose
    finish_reason is exactly "length".
    """

    # If a failed request was rerun successfully, retain the newest
    # successful result for that question/run pair.
    latest: dict[tuple[int, int], dict] = {}

    for row in rows:
        if row.get("status") == "ok":
            key = (
                int(row["question_index"]),
                int(row["run"]),
            )
            latest[key] = row

    rows = list(latest.values())

    if not rows:
        print("No successful generations were found.")
        return

    print()
    print("=" * 72)
    print('TRUNCATION RATE: finish_reason == "length"')
    print("=" * 72)

    for run in range(runs):
        run_rows = [
            row for row in rows
            if int(row["run"]) == run
        ]

        if not run_rows:
            continue

        truncated = sum(
            bool(row["hit_limit"])
            for row in run_rows
        )

        rate = truncated / len(run_rows)

        print(
            f"run {run}: "
            f"{truncated}/{len(run_rows)} "
            f"= {100.0 * rate:.1f}%"
        )

    total_truncated = sum(
        bool(row["hit_limit"])
        for row in rows
    )

    total_rate = total_truncated / len(rows)

    print(
        f"overall generation-level rate: "
        f"{total_truncated}/{len(rows)} "
        f"= {100.0 * total_rate:.1f}%"
    )

    # Question-level persistence across repeated runs.
    by_question: dict[int, list[dict]] = {}

    for row in rows:
        by_question.setdefault(
            int(row["question_index"]),
            [],
        ).append(row)

    truncated_at_least_once = sum(
        any(result["hit_limit"] for result in results)
        for results in by_question.values()
    )

    truncated_every_time = sum(
        all(result["hit_limit"] for result in results)
        for results in by_question.values()
    )

    print(
        "questions truncated at least once: "
        f"{truncated_at_least_once}/{len(by_question)}"
    )

    print(
        "questions truncated in every completed run: "
        f"{truncated_every_time}/{len(by_question)}"
    )

    # Particularly relevant to the "does not know when to stop"
    # argument. These traces produced a boxed candidate but still
    # continued until the API cut them off.
    boxed_before_truncation = sum(
        bool(row["hit_limit"]) and bool(row["has_boxed"])
        for row in rows
    )

    print(
        "truncated outputs already containing "
        f"\\boxed{{...}}: "
        f"{boxed_before_truncation}/{total_truncated}"
    )

    correct_boxed_before_truncation = sum(
        bool(row["hit_limit"])
        and bool(row.get("has_correct_boxed", False))
        for row in rows
    )

    print(
        "truncated outputs already containing the correct "
        f"\\boxed{{...}} answer: "
        f"{correct_boxed_before_truncation}/{total_truncated}"
    )

    # Detect an unexpected serving or context-length cap.
    truncated_token_counts = Counter(
        row["completion_tokens"]
        for row in rows
        if row["hit_limit"]
        and row["completion_tokens"] is not None
    )

    if truncated_token_counts:
        print(
            "completion-token counts among truncated outputs: "
            f"{dict(truncated_token_counts)}"
        )

        most_common_tokens, most_common_count = (
            truncated_token_counts.most_common(1)[0]
        )

        threshold = max(
            2,
            math.ceil(0.8 * total_truncated),
        )

        if (
            most_common_tokens < requested_max_tokens
            and most_common_count >= threshold
        ):
            print()
            print(
                "WARNING: most truncated outputs stopped at "
                f"{most_common_tokens} tokens, below the requested "
                f"max_tokens={requested_max_tokens}."
            )
            print(
                "This likely indicates a serving or context-length cap. "
                "Do not use this run to support a 30K-token claim."
            )


def main() -> None:
    args = parse_args()

    if args.runs < 1:
        raise SystemExit("--runs must be at least 1.")

    if args.max_tokens < 1:
        raise SystemExit("--max-tokens must be positive.")

    if args.out is None:
        model_slug = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            args.model,
        )

        args.out = Path(
            f"aime25_{model_slug}_"
            f"max{args.max_tokens}_"
            f"runs{args.runs}.jsonl"
        )

    args.out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = load_aime_dataset(args.dataset)

    if args.limit is not None:
        dataset = dataset.select(
            range(min(args.limit, len(dataset)))
        )

    if args.limit is None and len(dataset) != 30:
        raise SystemExit(
            "Expected 30 AIME 2025 problems, "
            f"but found {len(dataset)}."
        )

    if "problem" in dataset.column_names:
        question_column = "problem"
    elif "question" in dataset.column_names:
        question_column = "question"
    else:
        raise SystemExit(
            "Could not find a problem or question column. "
            f"Columns were {dataset.column_names}."
        )

    print(f"dataset: {args.dataset}")
    print(f"questions: {len(dataset)}")
    print(f"model: {args.model}")
    print(f"base_url: {args.base_url}")
    print(f"temperature: {TEMPERATURE}")
    print(f"top_p: {TOP_P}")
    print(f"max_tokens: {args.max_tokens}")
    print(f"runs per question: {args.runs}")
    print(f"output: {args.out}")

    client = OpenAI(
        # vLLM does not require an API key unless one is configured
        # when launching the server.
        api_key="EMPTY",
        base_url=args.base_url,
        timeout=7200.0,
        # The JSONL file is resumable, so fail rather than silently
        # repeating an expensive generation request.
        max_retries=0,
    )

    previous_rows = load_jsonl(args.out)

    # Resume only results generated with the same important settings.
    completed = {
        (
            int(row["question_index"]),
            int(row["run"]),
        )
        for row in previous_rows
        if row.get("status") == "ok"
        and row.get("model") == args.model
        and row.get("dataset") == args.dataset
        and row.get("base_url") == args.base_url
        and row.get("requested_max_tokens") == args.max_tokens
        and row.get("temperature") == TEMPERATURE
        and row.get("top_p") == TOP_P
        and int(row["question_index"]) < len(dataset)
        and int(row["run"]) < args.runs
    }

    total_jobs = len(dataset) * args.runs

    progress = tqdm(
        total=total_jobs,
        initial=len(completed),
        desc="AIME rollouts",
    )

    with args.out.open("a", encoding="utf-8") as output_file:
        for run in range(args.runs):
            for question_index, example in enumerate(dataset):
                key = (question_index, run)

                if key in completed:
                    continue

                problem = str(
                    example[question_column]
                )

                reference_answer = str(
                    example.get("answer", "")
                )

                try:
                    reference_value = int(
                        reference_answer.strip()
                    )
                except ValueError:
                    reference_value = None

                seed = (
                    args.base_seed
                    + run * len(dataset)
                    + question_index
                )

                try:
                    start_time = time.perf_counter()

                    response = client.chat.completions.create(
                        model=args.model,
                        messages=[
                            {
                                "role": "user",
                                "content": problem + PROMPT_SUFFIX,
                            }
                        ],
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        max_tokens=args.max_tokens,
                        seed=seed,
                    )
                    elapsed_seconds = time.perf_counter() - start_time

                    raw_response = response.model_dump()
                    choice = raw_response["choices"][0]
                    message = choice.get("message") or {}

                    content = message.get("content")

                    if not isinstance(content, str):
                        content = ""

                    reasoning = message.get(
                        "reasoning_content"
                    )

                    if not isinstance(reasoning, str):
                        reasoning = message.get("reasoning")

                    if not isinstance(reasoning, str):
                        reasoning = ""

                    if reasoning and content:
                        completion = reasoning + "\n" + content
                    else:
                        completion = reasoning or content

                    boxed_answers = [
                        int(answer)
                        for answer in BOXED_PATTERN.findall(
                            completion
                        )
                    ]

                    usage = raw_response.get("usage") or {}
                    finish_reason = choice.get(
                        "finish_reason"
                    )

                    row = {
                        "status": "ok",
                        "dataset": args.dataset,
                        "base_url": args.base_url,
                        "question_index": question_index,
                        "run": run,
                        "seed": seed,
                        "model": args.model,
                        "provider_model": raw_response.get(
                            "model"
                        ),
                        "requested_max_tokens": args.max_tokens,
                        "temperature": TEMPERATURE,
                        "top_p": TOP_P,
                        "problem": problem,
                        "reference_answer": reference_answer,
                        "finish_reason": finish_reason,
                        "hit_limit": (
                            finish_reason == "length"
                        ),
                        "prompt_tokens": usage.get(
                            "prompt_tokens"
                        ),
                        "completion_tokens": usage.get(
                            "completion_tokens"
                        ),
                        "total_tokens": usage.get(
                            "total_tokens"
                        ), 
                        "elapsed_seconds": elapsed_seconds,
                        "boxed_answers": boxed_answers,
                        "has_boxed": bool(boxed_answers),
                        "has_correct_boxed": (
                            reference_value in boxed_answers
                            if reference_value is not None
                            else False
                        ),
                        "completion": completion,
                    }

                except Exception as error:
                    error_row = {
                        "status": "error",
                        "dataset": args.dataset,
                        "base_url": args.base_url,
                        "question_index": question_index,
                        "run": run,
                        "seed": seed,
                        "model": args.model,
                        "requested_max_tokens": args.max_tokens,
                        "temperature": TEMPERATURE,
                        "top_p": TOP_P,
                        "problem": problem,
                        "reference_answer": reference_answer,
                        "error": repr(error),
                    }

                    output_file.write(
                        json.dumps(
                            error_row,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    output_file.flush()
                    progress.close()
                    raise

                output_file.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                output_file.flush()
                completed.add(key)
                progress.update(1)

                progress.set_postfix(
                    q=question_index,
                    run=run,
                    tokens=row["completion_tokens"],
                    secs=f"{elapsed_seconds:.1f}",
                    finish=finish_reason,
                    boxed=row["has_boxed"],
                )

    progress.close()

    all_rows = load_jsonl(args.out)

    current_rows = [
        row
        for row in all_rows
        if row.get("status") == "ok"
        and row.get("model") == args.model
        and row.get("dataset") == args.dataset
        and row.get("base_url") == args.base_url
        and row.get("requested_max_tokens") == args.max_tokens
        and row.get("temperature") == TEMPERATURE
        and row.get("top_p") == TOP_P
        and int(row["question_index"]) < len(dataset)
        and int(row["run"]) < args.runs
    ]

    summarize(
        current_rows,
        runs=args.runs,
        requested_max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()