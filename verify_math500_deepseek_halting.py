#!/usr/bin/env python3
"""
Validate the MATH non-halting claim for
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

# Local Hugging Face dataset repository containing all MATH-500 problems.
DATASET = "/workspace/user/chainsearch/quickAIMETest/MATH-500"

TEMPERATURE = 0.6
TOP_P = 0.95

PROMPT_SUFFIX = (
    "\nPlease reason step by step, "
    "and put your final answer within \\boxed{}."
)

BOXED_PATTERN = re.compile(r"\\boxed\s*\{")

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
            "Local MATH dataset directory or Hugging Face "
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
        help="Independent rollouts per MATH question.",
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

    parser.add_argument(
        "--grade-jsonl",
        type=Path,
        default=None,
        help=(
            "Grade an existing MATH-500 JSONL and analyze whether "
            "token-limit truncation is associated with incorrect answers. "
            "No model inference is performed."
        ),
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

 
def load_math500_dataset(source: str):
    """
    Load HuggingFaceH4/MATH-500 from either the local snapshot
    or directly from the Hugging Face Hub.
    """

    source_path = Path(source)

    if source_path.exists():
        jsonl_file = source_path / "test.jsonl"

        if not jsonl_file.exists():
            raise FileNotFoundError(
                f"Could not find {jsonl_file}."
            )

        return load_dataset(
            "json",
            data_files={"test": str(jsonl_file)},
            split="test",
        )

    return load_dataset(source, split="test")


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

def analyze_selection_bias(path: Path) -> None:
    """
    Grade an existing MATH-500 generation JSONL and compare
    correctness between naturally terminated and truncated traces.
    """

    try:
        from math_verify import parse, verify
    except ImportError as error:
        raise SystemExit(
            "math_verify is not installed. Install it with:\n"
            "pip install math-verify"
        ) from error

    rows = [
        row
        for row in load_jsonl(path)
        if row.get("status") == "ok"
    ]

    if not rows:
        raise SystemExit(
            f"No successful generations found in {path}."
        )

    results = []
    ungraded = []

    for row in tqdm(rows, desc="Grading MATH-500"):
        gold = str(
            row.get("reference_answer", "")
        ).strip()

        prediction = str(
            row.get("completion", "")
        )

        try:
            # Put the reference answer into an explicit boxed form so
            # math_verify knows exactly which expression is the gold.
            gold_parsed = parse(
                f"\\boxed{{{gold}}}"
            )

            prediction_parsed = parse(
                prediction
            )

            if not gold_parsed:
                raise ValueError(
                    "math_verify could not parse the gold answer"
                )

            # No extractable model answer means an incorrect
            # task prediction, not an ungradable example.
            if not prediction_parsed:
                correct = False
            else:
                correct = bool(
                    verify(
                        gold_parsed,
                        prediction_parsed,
                    )
                )

        except Exception as error:
            ungraded.append(
                {
                    "q": int(row["question_index"]),
                    "hit_limit": bool(row["hit_limit"]),
                    "tokens": row.get("completion_tokens"),
                    "error": repr(error),
                }
            )
            continue

        results.append(
            {
                "q": int(row["question_index"]),
                "hit_limit": bool(row["hit_limit"]),
                "tokens": int(row["completion_tokens"]),
                "correct": correct,
            }
        )

    truncated = [
        row
        for row in results
        if row["hit_limit"]
    ]

    natural = [
        row
        for row in results
        if not row["hit_limit"]
    ]

    def show(name: str, subset: list[dict]) -> None:
        if not subset:
            print(f"{name}: no graded examples")
            return

        correct = sum(
            row["correct"]
            for row in subset
        )

        wrong = len(subset) - correct

        print(
            f"{name}: "
            f"{correct}/{len(subset)} correct "
            f"= {100.0 * correct / len(subset):.1f}%"
        )

        print(
            f"  wrong: "
            f"{wrong}/{len(subset)} "
            f"= {100.0 * wrong / len(subset):.1f}%"
        )

    print()
    print("=" * 72)
    print("SELECTION-BIAS CHECK")
    print("=" * 72)

    print(f"JSONL: {path}")
    print(f"Successful generations: {len(rows)}")
    print(f"Successfully graded: {len(results)}")
    print(f"Could not grade: {len(ungraded)}")
    print()

    show(
        "Stopped naturally",
        natural,
    )

    show(
        "Hit token limit",
        truncated,
    )

    total_wrong = sum(
        not row["correct"]
        for row in results
    )

    truncated_wrong = sum(
        row["hit_limit"]
        and not row["correct"]
        for row in results
    )

    print()
    print(f"Total graded errors: {total_wrong}/{len(results)}")

    if total_wrong:
        print(
            "Errors coming from token-limit examples: "
            f"{truncated_wrong}/{total_wrong} "
            f"= {100.0 * truncated_wrong / total_wrong:.1f}%"
        )

    print()
    print("2x2 TABLE")
    print("                    correct   wrong")

    print(
        f"stopped naturally   "
        f"{sum(r['correct'] for r in natural):7d}   "
        f"{sum(not r['correct'] for r in natural):5d}"
    )

    print(
        f"hit token limit     "
        f"{sum(r['correct'] for r in truncated):7d}   "
        f"{sum(not r['correct'] for r in truncated):5d}"
    )

    if ungraded:
        print()
        print("UNGRADABLE EXAMPLES")

        for row in ungraded:
            print(
                f"Q{row['q'] + 1:03d}  "
                f"hit_limit={row['hit_limit']}  "
                f"tokens={row['tokens']}  "
                f"{row['error']}"
            )

def main() -> None:
    
    args = parse_args()

    if args.grade_jsonl is not None:
        analyze_selection_bias(
            args.grade_jsonl
        )
        return
        
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
            f"math500_{model_slug}_"
            f"max{args.max_tokens}_"
            f"runs{args.runs}.jsonl"
        )

    args.out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = load_math500_dataset(args.dataset)

    if args.limit is not None:
        dataset = dataset.select(
            range(min(args.limit, len(dataset)))
        )

    if args.limit is None and len(dataset) != 500:
        raise SystemExit(
            "Expected 500 MATH-500 problems, "
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
        desc="MATH-500 rollouts",
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

                    has_boxed = bool(
                        BOXED_PATTERN.search(completion)
                    )

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
                        "has_boxed": has_boxed,
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