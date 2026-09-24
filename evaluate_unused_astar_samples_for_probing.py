#!/usr/bin/env python3
"""
evaluate_unused_astar_samples_for_probing.py

Run the fine-tuned A* model on unused, target-held-out A* cache rows. Generate
one deterministic reasoning trace per prompt, evaluate each generated trace,
and save packed generated token IDs plus per-sample correctness labels.

This script deliberately does not cache hidden activations. The next script
selects the final balanced probing samples and replays these exact generated
token sequences to cache only the activations that will actually be used.
"""

import math
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

# Select the A* stage before importing the shared model module.
os.environ["COUNTDOWN_STAGE"] = "astar"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from tqdm.auto import tqdm

import pretrain_and_sft_model as pm


# =========================
# DEFAULT CONFIG
# =========================

RUN_TAG = "exp34"
RANDOM_SEED = 42

ASTAR_DIR = "/workspace/user/chainsearch_archives/data_pts/astar"
MODEL_DIR = "/workspace/user/chainsearch_archives/model_pts/astar"
PROBING_DIR = os.path.join(ASTAR_DIR, "probing")

TOKEN_CACHE_PATH = os.path.join(
    ASTAR_DIR,
    "tokenisation_cache",
    f"{RUN_TAG}_astar_max2048_drop.pt",
)

SPLIT_PATH = os.path.join(
    ASTAR_DIR,
    f"{RUN_TAG}_astar_target_random_"
    "train_val_test_probe_split.pt",
)

CHECKPOINT_PATH = os.path.join(
    MODEL_DIR,
    f"{RUN_TAG}_astar_best.pt",
)

OUTPUT_PATH = os.path.join(
    PROBING_DIR,
    f"{RUN_TAG}_astar_unused_samples_evaluated_for_probing.pt",
)

PARTIAL_PATH = os.path.join(
    PROBING_DIR,
    f"{RUN_TAG}_astar_unused_samples_evaluated_for_probing.partial.pt",
)

# Evaluate every dedicated probe sample in a deterministic shuffled order.
MAX_CANDIDATES = -1
MIN_EVALUATED = 25_000
MIN_PRIMARY_CORRECT = 15_000
MIN_PRIMARY_INCORRECT = 10_000

GENERATION_BATCH_SIZE = 512
SAVE_EVERY_SAMPLES = 2_000
NUM_EXAMPLES_PER_CLASS_TO_PRINT = 3
MAX_PRINT_CHARS = 3_000

# Keep generation greedy so outputs are reproducible and match the existing
# evaluation setup. Activations are not collected in this script.
STOCHASTIC_STEPS = 0
TEMPERATURE = 0.7
TOP_P = 1.0
TOP_K = 0

RESUME_PARTIAL = False
REPORT_ONLY_FROM_SAVED = False
OVERWRITE_FINAL = True
COMPILE_MODEL = False

# Primary probe label. A sample is positive only when every parsed arithmetic
# step is legal and state-consistent, SUCCESS appears, and some valid step
# reaches the target.
PRIMARY_LABEL_NAME = "solved_valid_trace"

FAILURE_CATEGORY_NAMES = {
    0: "solved_valid_trace",
    1: "valid_trace_wrong_target",
    2: "missing_success",
    3: "no_parsed_operation",
    4: "invalid_or_state_inconsistent_trace",
}

BOOL_FIELDS = [
    "program_exact_match",
    "any_step_target_hit",
    "valid_trace",
    "solved_valid_trace",
    "best_step_coverage",
    "expr_exact_match",
    "expr_target_correct",
    "expr_parseable",
    "has_success",
    "primary_correct",
]

INT_FIELDS = [
    "cache_idx",
    "target",
    "gold_back_count",
    "prompt_length",
    "generated_length",
    "generated_back_count",
    "op_valid_count",
    "op_total",
    "state_consistent_count",
    "state_total",
    "failure_category",
]

FLOAT_FIELDS = [
    "expr_target_abs_error",
    "best_step_target_error",
]


# =========================
# CACHE / PACKING HELPERS
# =========================


def safe_torch_save(obj: Any, path: str) -> None:
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, final_path)


def cache_row_ids(blob: dict, idx: int) -> list[int]:
    start = int(blob["offsets"][idx].item())
    length = int(blob["lengths"][idx].item())
    return blob["flat_ids"][start:start + length].tolist()


def cache_row_parts(
    blob: dict,
    idx: int,
    tok: pm.Tokenizer,
) -> tuple[list[int], list[int]]:
    ids = cache_row_ids(blob, idx)
    sep_positions = [i for i, token_id in enumerate(ids) if token_id == tok.sep_id]
    if len(sep_positions) != 1:
        raise ValueError(
            f"Cache row {idx} has {len(sep_positions)} SEP tokens; expected exactly 1."
        )
    sep = sep_positions[0]
    prompt_ids = ids[:sep + 1]
    gold_trace_ids = ids[sep + 1:]
    return prompt_ids, gold_trace_ids


def pack_sequences(sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.int32)
    offsets = torch.zeros(len(sequences), dtype=torch.int64)
    if len(sequences) > 1:
        offsets[1:] = torch.cumsum(lengths[:-1].to(torch.int64), dim=0)

    total = int(lengths.sum().item()) if lengths.numel() else 0
    flat = torch.empty(total, dtype=torch.int16)
    cursor = 0
    for seq in sequences:
        n = len(seq)
        if n:
            flat[cursor:cursor + n] = torch.tensor(seq, dtype=torch.int16)
        cursor += n
    return flat, offsets, lengths


def unpack_sequences(
    flat: torch.Tensor,
    offsets: torch.Tensor,
    lengths: torch.Tensor,
) -> list[list[int]]:
    out = []
    for i in range(int(lengths.numel())):
        start = int(offsets[i].item())
        length = int(lengths[i].item())
        out.append(flat[start:start + length].to(torch.int64).tolist())
    return out

def saved_generated_ids(payload: dict, position: int) -> list[int]:
    start = int(payload["generated_offsets"][position].item())
    length = int(payload["generated_lengths"][position].item())
    return (
        payload["generated_flat_ids"][start:start + length]
        .to(torch.int64)
        .tolist()
    )



def new_accumulator() -> dict[str, list]:
    acc = {name: [] for name in BOOL_FIELDS + INT_FIELDS + FLOAT_FIELDS}
    acc["input_numbers"] = []
    acc["generated_ids"] = []
    return acc


def accumulator_size(acc: dict[str, list]) -> int:
    return len(acc["cache_idx"])


def append_record(
    acc: dict[str, list],
    *,
    cache_idx: int,
    target: int,
    gold_back_count: int,
    prompt_length: int,
    input_numbers: list[int],
    generated_ids: list[int],
    labels: dict[str, Any],
) -> None:
    acc["cache_idx"].append(int(cache_idx))
    acc["target"].append(int(target))
    acc["gold_back_count"].append(int(gold_back_count))
    acc["prompt_length"].append(int(prompt_length))
    acc["input_numbers"].append([int(x) for x in input_numbers])
    acc["generated_ids"].append([int(x) for x in generated_ids])

    for name in BOOL_FIELDS:
        acc[name].append(bool(labels[name]))
    for name in INT_FIELDS:
        if name in {"cache_idx", "target", "gold_back_count", "prompt_length"}:
            continue
        acc[name].append(int(labels[name]))
    for name in FLOAT_FIELDS:
        acc[name].append(float(labels[name]))


def mean_bool(values: list[bool]) -> float:
    return (sum(bool(v) for v in values) / len(values)) if values else float("nan")


def finite_mean(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return (sum(finite) / len(finite)) if finite else float("nan")


def build_summary(acc: dict[str, list]) -> dict[str, Any]:
    n = accumulator_size(acc)
    if n == 0:
        return {"samples": 0}

    correct = sum(bool(x) for x in acc["primary_correct"])
    incorrect = n - correct

    op_total = sum(int(x) for x in acc["op_total"])
    op_valid = sum(int(x) for x in acc["op_valid_count"])
    state_total = sum(int(x) for x in acc["state_total"])
    state_valid = sum(int(x) for x in acc["state_consistent_count"])

    back_diffs = [
        int(pred) - int(gold)
        for pred, gold in zip(acc["generated_back_count"], acc["gold_back_count"])
    ]

    failure_counts = Counter(int(x) for x in acc["failure_category"])

    return {
        "samples": n,
        "primary_label_name": PRIMARY_LABEL_NAME,
        "primary_correct_count": correct,
        "primary_incorrect_count": incorrect,
        "program_exact_match": mean_bool(acc["program_exact_match"]),
        "any_step_target_hit_rate": mean_bool(acc["any_step_target_hit"]),
        "valid_trace_rate": mean_bool(acc["valid_trace"]),
        "solved_valid_trace_rate": mean_bool(acc["solved_valid_trace"]),
        "best_step_coverage": mean_bool(acc["best_step_coverage"]),
        "expr_exact_match": mean_bool(acc["expr_exact_match"]),
        "expr_target_accuracy": mean_bool(acc["expr_target_correct"]),
        "expr_parse_rate": mean_bool(acc["expr_parseable"]),
        "expr_target_mae": finite_mean(acc["expr_target_abs_error"]),
        "best_step_target_mae": finite_mean(acc["best_step_target_error"]),
        "back_count_mae": (
            sum(abs(x) for x in back_diffs) / n
        ),
        "back_count_bias": sum(back_diffs) / n,
        "operation_validity_rate": (op_valid / op_total) if op_total else 0.0,
        "operation_state_consistency_rate": (
            state_valid / state_total
        ) if state_total else 0.0,
        "failure_category_counts": {
            FAILURE_CATEGORY_NAMES[k]: int(v)
            for k, v in sorted(failure_counts.items())
        },
    }


def build_payload(
    *,
    acc: dict[str, list],
    candidate_cache_idx: list[int],
    source_unused: dict,
    model_config: dict,
    model_max_len: int,
    max_output_tokens: int,
    complete: bool,
) -> dict[str, Any]:
    generated_flat, generated_offsets, generated_lengths = pack_sequences(
        acc["generated_ids"]
    )

    n = accumulator_size(acc)
    input_numbers = (
        torch.tensor(acc["input_numbers"], dtype=torch.int16)
        if n
        else torch.empty((0, 6), dtype=torch.int16)
    )

    payload: dict[str, Any] = {
        "format_version": 1,
        "complete": bool(complete),
        "n_completed": n,
        "candidate_cache_idx": torch.tensor(candidate_cache_idx, dtype=torch.int64),
        "input_numbers": input_numbers,
        "generated_flat_ids": generated_flat,
        "generated_offsets": generated_offsets,
        "generated_lengths": generated_lengths,
        "summary": build_summary(acc),
        "failure_category_names": dict(FAILURE_CATEGORY_NAMES),
        "primary_label_name": PRIMARY_LABEL_NAME,
        "primary_label_definition": (
            "all parsed arithmetic steps are legal and state-consistent, "
            "SUCCESS appears, and at least one valid step reaches TGT"
        ),
        "source_token_cache_path": TOKEN_CACHE_PATH,
        "source_checkpoint_path": CHECKPOINT_PATH,
        "source_split_path": SPLIT_PATH,
        "index_space": "astar_tokenisation_cache_rows",
        "run_tag": RUN_TAG,
        "random_seed": RANDOM_SEED,
        "max_candidates": MAX_CANDIDATES,
        "min_evaluated": MIN_EVALUATED,
        "min_primary_correct": MIN_PRIMARY_CORRECT,
        "min_primary_incorrect": MIN_PRIMARY_INCORRECT,
        "generation_batch_size": GENERATION_BATCH_SIZE,
        "max_output_tokens": int(max_output_tokens),
        "generation_is_greedy": STOCHASTIC_STEPS == 0,
        "model_max_len": int(model_max_len),
        "model_config": model_config,
    }

    for name in BOOL_FIELDS:
        payload[name] = torch.tensor(acc[name], dtype=torch.bool)

    for name in INT_FIELDS:
        dtype = torch.int64 if name == "cache_idx" else torch.int32
        payload[name] = torch.tensor(acc[name], dtype=dtype)

    for name in FLOAT_FIELDS:
        payload[name] = torch.tensor(acc[name], dtype=torch.float32)

    return payload


def restore_accumulator(payload: dict[str, Any]) -> dict[str, list]:
    acc = new_accumulator()
    for name in BOOL_FIELDS + INT_FIELDS + FLOAT_FIELDS:
        if name not in payload:
            raise KeyError(f"Partial file is missing field {name!r}.")
        acc[name] = payload[name].tolist()
    acc["input_numbers"] = payload["input_numbers"].to(torch.int64).tolist()
    acc["generated_ids"] = unpack_sequences(
        payload["generated_flat_ids"],
        payload["generated_offsets"],
        payload["generated_lengths"],
    )
    return acc


# =========================
# MODEL / LABELING HELPERS
# =========================


def load_model(
    tok: pm.Tokenizer,
    device: torch.device,
) -> tuple[torch.nn.Module, dict, int]:
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint).__name__}.")

    state = checkpoint.get("model_state") or checkpoint.get("state")
    if state is None:
        raise KeyError("Checkpoint has neither 'model_state' nor 'state'.")

    config = dict(pm.HPARAMS)
    config.update(checkpoint.get("config") or {})
    model_max_len = pm.infer_model_max_len_from_checkpoint(CHECKPOINT_PATH)

    model = pm._build_model(
        config,
        tok,
        model_max_len,
        use_grad_ckpt=False,
        dropout=0.0,
    )
    pm._load_model_state(model, state)
    model.to(device)
    model.eval()

    if COMPILE_MODEL:
        model = torch.compile(model)

    return model, config, model_max_len


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def label_generation(
    *,
    prompt_text: str,
    gold_text: str,
    generated_text: str,
    target: int,
    gold_back_count: int,
) -> dict[str, Any]:
    initial_numbers = pm.parse_initial_numbers(prompt_text)
    if not initial_numbers:
        raise ValueError(f"Could not parse initial numbers from prompt: {prompt_text!r}")

    has_success = "SUCCESS" in generated_text
    program_exact_match = normalize_text(generated_text) == normalize_text(gold_text)

    gold_expr = pm.extract_expr(gold_text)
    pred_expr = pm.extract_expr(generated_text)
    expr_exact_match = (
        gold_expr is not None
        and pred_expr is not None
        and normalize_text(pred_expr) == normalize_text(gold_expr)
    )

    expr_parseable = False
    expr_target_correct = False
    expr_target_abs_error = float("nan")
    if pred_expr is not None:
        try:
            expr_value = pm.eval_expr_text(pred_expr)
            expr_parseable = True
            expr_target_abs_error = float(abs(expr_value - target))
            expr_target_correct = expr_value == target
        except Exception:
            pass

    generated_back_count = generated_text.count("BACK")

    current_state = Counter(int(v) for v in initial_numbers)
    any_step = False
    any_step_target_hit = False
    best_step_target_error = None

    op_valid_count = 0
    op_total = 0
    state_consistent_count = 0
    state_total = 0

    for match in pm.RE_EVENT.finditer(generated_text):
        back_state_text = match.group(1)
        if back_state_text is not None:
            current_state = Counter(pm._ints_in(back_state_text))
            continue

        before_text = match.group(2)
        n1_text, op, n2_text, result_text, after_text = (
            match.group(3),
            match.group(4),
            match.group(5),
            match.group(6),
            match.group(7),
        )

        any_step = True
        n1 = int(n1_text)
        n2 = int(n2_text)
        result = int(result_text)

        arithmetic_ok = (
            (op == "+" and n1 + n2 == result)
            or (op == "-" and n1 > n2 and n1 - n2 == result)
            or (op == "*" and n1 > 1 and n2 > 1 and n1 * n2 == result)
            or (
                op == "/"
                and n2 > 0
                and n1 % n2 == 0
                and n1 // n2 == result
            )
        )

        op_total += 1
        if arithmetic_ok:
            op_valid_count += 1

        state_total += 1

        if before_text:
            stated_before = Counter(pm._ints_in(before_text))
        else:
            stated_before = current_state.copy()
        stated_after = Counter(pm._ints_in(after_text))

        if stated_before != current_state or not arithmetic_ok:
            current_state = stated_after
            continue

        if n1 == n2:
            operands_available = current_state.get(n1, 0) >= 2
        else:
            operands_available = (
                current_state.get(n1, 0) >= 1
                and current_state.get(n2, 0) >= 1
            )

        if not operands_available:
            current_state = stated_after
            continue

        expected_values = list(current_state.elements())
        expected_values.remove(n1)
        expected_values.remove(n2)
        expected_values.append(result)
        expected_after = Counter(expected_values)

        if expected_after == stated_after:
            state_consistent_count += 1
            error = abs(result - target)
            if best_step_target_error is None:
                best_step_target_error = error
            else:
                best_step_target_error = min(best_step_target_error, error)
            if result == target:
                any_step_target_hit = True

        current_state = stated_after

    valid_trace = (
        any_step
        and op_valid_count == op_total
        and state_consistent_count == state_total
        and has_success
    )
    solved_valid_trace = valid_trace and any_step_target_hit

    if solved_valid_trace:
        failure_category = 0
    elif valid_trace:
        failure_category = 1
    elif not has_success:
        failure_category = 2
    elif not any_step:
        failure_category = 3
    else:
        failure_category = 4

    return {
        "program_exact_match": program_exact_match,
        "any_step_target_hit": any_step_target_hit,
        "valid_trace": valid_trace,
        "solved_valid_trace": solved_valid_trace,
        "best_step_coverage": best_step_target_error is not None,
        "expr_exact_match": expr_exact_match,
        "expr_target_correct": expr_target_correct,
        "expr_parseable": expr_parseable,
        "has_success": has_success,
        "primary_correct": solved_valid_trace,
        "generated_length": 0,  # filled by caller
        "generated_back_count": generated_back_count,
        "op_valid_count": op_valid_count,
        "op_total": op_total,
        "state_consistent_count": state_consistent_count,
        "state_total": state_total,
        "failure_category": failure_category,
        "expr_target_abs_error": expr_target_abs_error,
        "best_step_target_error": (
            float(best_step_target_error)
            if best_step_target_error is not None
            else float("nan")
        ),
        "gold_back_count": int(gold_back_count),
    }


def pretty_trace(text: str) -> str:
    normalized = normalize_text(text)
    parts = [part.strip() for part in re.split(r"\s*;\s*", normalized) if part.strip()]
    lines = []
    for part in parts:
        lines.append(part if part.startswith("EXPR") else f"{part} ;")
    out = "\n".join(lines)
    if len(out) > MAX_PRINT_CHARS:
        out = out[:MAX_PRINT_CHARS] + "\n... <truncated>"
    return out


def print_summary(summary: dict[str, Any]) -> None:
    n = int(summary["samples"])
    print()
    print("=" * 88)
    print("A* PROBE SAMPLES - GENERATION SUMMARY".center(88))
    print("=" * 88)
    print(f"Samples evaluated:                                      {n:>10,}")
    print()
    print(f"Program exact match:                         {100*summary['program_exact_match']:>9.2f}%")
    print(f"Any-step target hit:                         {100*summary['any_step_target_hit_rate']:>9.2f}%")
    print(f"Valid trace rate:                            {100*summary['valid_trace_rate']:>9.2f}%")
    print(f"Solved valid trace rate:                     {100*summary['solved_valid_trace_rate']:>9.2f}%")
    print(f"Best-step coverage:                          {100*summary['best_step_coverage']:>9.2f}%")
    print(f"EXPR exact match:                            {100*summary['expr_exact_match']:>9.2f}%")
    print(f"EXPR target accuracy:                        {100*summary['expr_target_accuracy']:>9.2f}%")
    print(f"EXPR parse rate:                             {100*summary['expr_parse_rate']:>9.2f}%")
    print(f"EXPR target MAE:                             {summary['expr_target_mae']:>12.2f}")
    print(f"Best-step target MAE:                        {summary['best_step_target_mae']:>12.2f}")
    print(f"BACK count MAE:                              {summary['back_count_mae']:>12.2f}")
    print(f"BACK count bias:                             {summary['back_count_bias']:>+12.2f}")
    print(f"Operation validity rate:                     {100*summary['operation_validity_rate']:>9.2f}%")
    print(f"Operation state-consistency rate:            {100*summary['operation_state_consistency_rate']:>9.2f}%")
    print()
    print(f"Primary positive ({PRIMARY_LABEL_NAME}):      {summary['primary_correct_count']:>10,}")
    print(f"Primary negative:                            {summary['primary_incorrect_count']:>10,}")
    print()
    print("Primary failure categories")
    for name, count in summary["failure_category_counts"].items():
        print(f"  {name:<42} {count:>10,}")
    print("=" * 88)


# =========================
# MAIN
# =========================


def main() -> None:
    width = 88
    print("=" * width)
    print("EVALUATE UNUSED A* SAMPLES FOR PROBING".center(width))
    print("=" * width)
    print(f"Checkpoint:       {CHECKPOINT_PATH}")
    print(f"Token cache:      {TOKEN_CACHE_PATH}")
    print(f"Probe split:      {SPLIT_PATH}")
    print(f"Output:           {OUTPUT_PATH}")
    print(f"Partial progress: {PARTIAL_PATH}")
    if MAX_CANDIDATES < 0:
        print("Candidates:        all probe samples")
    else:
        print(f"Maximum candidates: {MAX_CANDIDATES:,}")
    print("=" * width)
    print()

    if REPORT_ONLY_FROM_SAVED:
        for required_path in (TOKEN_CACHE_PATH, OUTPUT_PATH):
            if not os.path.isfile(required_path):
                raise FileNotFoundError(required_path)

        tok = pm.Tokenizer()

        print("Loading token cache...", flush=True)
        blob, _, _, _, _, _ = pm._load_enc_blob(
            TOKEN_CACHE_PATH,
            tok,
            safe_max_len=2048,
            long_seq_policy="drop",
        )

        print("Loading saved evaluation artifact...", flush=True)
        saved = torch.load(
            OUTPUT_PATH,
            map_location="cpu",
            weights_only=False,
        )

        if not saved.get("complete", False):
            raise RuntimeError(
                "Saved evaluation artifact is not marked complete."
            )

        print_summary(saved["summary"])

        for desired_label, heading in (
            (True, "POSITIVE EXAMPLES"),
            (False, "NEGATIVE EXAMPLES"),
        ):
            print()
            print(heading)
            shown = 0

            for i, is_correct in enumerate(
                saved["primary_correct"].tolist()
            ):
                if bool(is_correct) != desired_label:
                    continue

                cache_idx = int(saved["cache_idx"][i].item())
                target = int(saved["target"][i].item())

                prompt_ids, gold_trace_ids = cache_row_parts(
                    blob,
                    cache_idx,
                    tok,
                )
                generated_ids = saved_generated_ids(saved, i)

                prompt_text = tok.decode(prompt_ids[:-1]).strip()
                gold_text = tok.decode(gold_trace_ids).strip()
                generated_text = tok.decode(generated_ids).strip()

                print("-" * width)
                print(f"Cache index: {cache_idx}")
                print(f"Target:      {target}")
                print(f"Label:       {PRIMARY_LABEL_NAME}={desired_label}")
                print(f"Problem:     {prompt_text}")
                print()
                print("GROUND-TRUTH TRACE")
                print(pretty_trace(gold_text))
                print()
                print("GENERATED TRACE")
                print(pretty_trace(generated_text))

                shown += 1
                if shown >= NUM_EXAMPLES_PER_CLASS_TO_PRINT:
                    break

        print()
        print("=" * width)
        print("Report-only mode completed.")
        print("No model was loaded and no inference was rerun.")
        print(f"Existing artifact was not modified: {OUTPUT_PATH}")
        print("=" * width)
        return

    for required_path in (CHECKPOINT_PATH, TOKEN_CACHE_PATH, SPLIT_PATH):
        if not os.path.isfile(required_path):
            raise FileNotFoundError(required_path)

    if os.path.exists(OUTPUT_PATH) and not OVERWRITE_FINAL:
        raise FileExistsError(
            f"Final output exists and OVERWRITE_FINAL=False: {OUTPUT_PATH}"
        )

    os.makedirs(PROBING_DIR, exist_ok=True)

    tok = pm.Tokenizer()
    print("Loading token cache...", flush=True)
    blob, _, ml, mi, mo, _ = pm._load_enc_blob(
        TOKEN_CACHE_PATH,
        tok,
        safe_max_len=2048,
        long_seq_policy="drop",
    )

    print("Loading probe split...", flush=True)
    split = torch.load(
        SPLIT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if "probe_idx" not in split:
        raise KeyError(
            f"Split file has no probe_idx: {SPLIT_PATH}"
        )

    split_cache_path = split.get("enc_cache_path")
    if (
        split_cache_path is not None
        and os.path.realpath(str(split_cache_path))
        != os.path.realpath(TOKEN_CACHE_PATH)
    ):
        raise RuntimeError(
            "Split file points to a different token cache."
        )

    probe_cache_idx = [
        int(idx)
        for idx in split["probe_idx"]
    ]

    if len(probe_cache_idx) != len(set(probe_cache_idx)):
        raise RuntimeError("probe_idx contains duplicate indices.")

    dataset_size = int(blob["lengths"].numel())
    if any(
        idx < 0 or idx >= dataset_size
        for idx in probe_cache_idx
    ):
        raise RuntimeError("probe_idx contains out-of-range indices.")

    probe_targets = []
    probe_gold_backs = []

    for cache_idx in tqdm(
        probe_cache_idx,
        desc="Reading probe metadata",
        dynamic_ncols=True,
    ):
        prompt_ids, _ = cache_row_parts(
            blob,
            cache_idx,
            tok,
        )
        prompt_text = tok.decode(
            prompt_ids[:-1]
        ).strip()

        target_match = re.search(
            r"TGT:\s*(\d+)",
            prompt_text,
        )
        if target_match is None:
            raise RuntimeError(
                f"Could not parse target at cache row {cache_idx}."
            )

        probe_targets.append(
            int(target_match.group(1))
        )
        probe_gold_backs.append(
            int(blob["bk_counts"][cache_idx].item())
        )

    order = list(range(len(probe_cache_idx)))
    random.Random(RANDOM_SEED).shuffle(order)

    if MAX_CANDIDATES >= 0:
        order = order[
            :min(MAX_CANDIDATES, len(order))
        ]

    candidate_cache_idx = [
        probe_cache_idx[i]
        for i in order
    ]
    candidate_targets = [
        probe_targets[i]
        for i in order
    ]
    candidate_gold_backs = [
        probe_gold_backs[i]
        for i in order
    ]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)

    print(f"Loading model on {device}...", flush=True)
    model, model_config, model_max_len = load_model(tok, device)
    use_amp = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    max_output_tokens = min(
        int(model_config.get("max_output_len", mo)),
        int(model_max_len) - int(mi),
    )
    if max_output_tokens <= 0:
        raise RuntimeError(
            f"No generation budget remains: model_max_len={model_max_len}, max_prompt_len={mi}."
        )

    generation_config = dict(model_config)
    generation_config.update(
        stochastic_steps=STOCHASTIC_STEPS,
        temperature_ops=TEMPERATURE,
        top_p_ops=TOP_P,
        top_k_ops=TOP_K,
    )

    acc = new_accumulator()
    completed = 0

    if os.path.exists(PARTIAL_PATH):
        if RESUME_PARTIAL:
            print(f"Resuming from partial artifact: {PARTIAL_PATH}", flush=True)
            partial = torch.load(
                PARTIAL_PATH,
                map_location="cpu",
                weights_only=False,
            )
            saved_candidates = partial["candidate_cache_idx"].to(torch.int64).tolist()
            if saved_candidates != candidate_cache_idx:
                raise RuntimeError(
                    "Partial artifact candidate order does not match the current configuration."
                )
            if int(partial.get("max_output_tokens", -1)) != max_output_tokens:
                raise RuntimeError("Partial artifact max_output_tokens mismatch.")
            if os.path.realpath(str(partial.get("source_checkpoint_path"))) != os.path.realpath(CHECKPOINT_PATH):
                raise RuntimeError("Partial artifact checkpoint mismatch.")
            if os.path.realpath(str(partial.get("source_token_cache_path"))) != os.path.realpath(TOKEN_CACHE_PATH):
                raise RuntimeError("Partial artifact token-cache mismatch.")
            acc = restore_accumulator(partial)
            completed = accumulator_size(acc)
            if completed > len(candidate_cache_idx):
                raise RuntimeError("Partial artifact contains too many completed samples.")
        else:
            os.remove(PARTIAL_PATH)

    next_save_at = (
        ((completed // SAVE_EVERY_SAMPLES) + 1) * SAVE_EVERY_SAMPLES
    )

    progress = tqdm(
        total=len(candidate_cache_idx),
        initial=completed,
        desc="Generating and labeling",
        dynamic_ncols=True,
    )

    while completed < len(candidate_cache_idx):
        end = min(completed + GENERATION_BATCH_SIZE, len(candidate_cache_idx))
        batch_cache_idx = candidate_cache_idx[completed:end]
        batch_targets = candidate_targets[completed:end]
        batch_gold_backs = candidate_gold_backs[completed:end]

        prompt_batch = []
        gold_text_batch = []
        prompt_text_batch = []
        input_numbers_batch = []

        for cache_idx, expected_target, expected_gold_back in zip(
            batch_cache_idx,
            batch_targets,
            batch_gold_backs,
        ):
            prompt_ids, gold_trace_ids = cache_row_parts(blob, cache_idx, tok)
            prompt_text = tok.decode(prompt_ids[:-1]).strip()
            gold_text = tok.decode(gold_trace_ids).strip()
            input_numbers = sorted(pm.parse_initial_numbers(prompt_text))
            target_match = re.search(r"TGT:\s*(\d+)", prompt_text)
            if target_match is None or int(target_match.group(1)) != expected_target:
                raise RuntimeError(
                    f"Target mismatch at cache row {cache_idx}: artifact={expected_target}, prompt={prompt_text!r}"
                )
            if int(blob["bk_counts"][cache_idx].item()) != expected_gold_back:
                raise RuntimeError(
                    f"Gold BACK mismatch at cache row {cache_idx}."
                )
            if len(input_numbers) != 6:
                raise RuntimeError(
                    f"Expected six input numbers at cache row {cache_idx}, got {input_numbers}."
                )

            prompt_batch.append(prompt_ids)
            gold_text_batch.append(gold_text)
            prompt_text_batch.append(prompt_text)
            input_numbers_batch.append(input_numbers)

        generated_batch = pm.sample_batch(
            model,
            tok,
            prompt_batch,
            max_output_tokens,
            device,
            cfg=generation_config,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
        )

        for cache_idx, target, gold_back, prompt_ids, prompt_text, gold_text, input_numbers, generated_ids in zip(
            batch_cache_idx,
            batch_targets,
            batch_gold_backs,
            prompt_batch,
            prompt_text_batch,
            gold_text_batch,
            input_numbers_batch,
            generated_batch,
        ):
            generated_text = tok.decode(generated_ids).strip() if generated_ids else ""
            labels = label_generation(
                prompt_text=prompt_text,
                gold_text=gold_text,
                generated_text=generated_text,
                target=target,
                gold_back_count=gold_back,
            )
            labels["generated_length"] = len(generated_ids)

            append_record(
                acc,
                cache_idx=cache_idx,
                target=target,
                gold_back_count=gold_back,
                prompt_length=len(prompt_ids),
                input_numbers=input_numbers,
                generated_ids=generated_ids,
                labels=labels,
            )

        previous = completed
        completed = accumulator_size(acc)
        progress.update(completed - previous)

        if completed >= next_save_at:
            partial_payload = build_payload(
                acc=acc,
                candidate_cache_idx=candidate_cache_idx,
                source_unused=split,
                model_config=model_config,
                model_max_len=model_max_len,
                max_output_tokens=max_output_tokens,
                complete=False,
            )
            safe_torch_save(partial_payload, PARTIAL_PATH)
            next_save_at += SAVE_EVERY_SAMPLES

    progress.close()

    summary = build_summary(acc)
    final_payload = build_payload(
        acc=acc,
        candidate_cache_idx=candidate_cache_idx,
        source_unused=split,
        model_config=model_config,
        model_max_len=model_max_len,
        max_output_tokens=max_output_tokens,
        complete=True,
    )
    safe_torch_save(final_payload, OUTPUT_PATH)
    if os.path.exists(PARTIAL_PATH):
        os.remove(PARTIAL_PATH)

    print_summary(summary)

    for desired_label, heading in ((True, "POSITIVE EXAMPLES"), (False, "NEGATIVE EXAMPLES")):
        print()
        print(heading)
        shown = 0
        for i, is_correct in enumerate(acc["primary_correct"]):
            if bool(is_correct) != desired_label:
                continue
            cache_idx = int(acc["cache_idx"][i])
            prompt_ids, _ = cache_row_parts(blob, cache_idx, tok)
            prompt_text = tok.decode(prompt_ids[:-1]).strip()
            generated_text = tok.decode(acc["generated_ids"][i]).strip()
            print("-" * width)
            print(f"Cache index: {cache_idx}")
            print(f"Target:      {acc['target'][i]}")
            print(f"Label:       {PRIMARY_LABEL_NAME}={desired_label}")
            print(f"Problem:     {prompt_text}")
            print("Generated trace")
            print(pretty_trace(generated_text))
            shown += 1
            if shown >= NUM_EXAMPLES_PER_CLASS_TO_PRINT:
                break

    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 ** 2)
    print()
    print("=" * width)
    print(f"Saved evaluated probing candidates to: {OUTPUT_PATH}")
    print(f"Evaluated samples: {summary['samples']:,}")
    print(f"Output size: {size_mb:.2f} MiB")
    if (
        int(summary["primary_correct_count"]) < MIN_PRIMARY_CORRECT
        or int(summary["primary_incorrect_count"]) < MIN_PRIMARY_INCORRECT
    ):
        print(
            "WARNING: requested class-count targets were not both reached. "
            "Increase MAX_CANDIDATES before running the confirmation script."
        )
    print("=" * width)


if __name__ == "__main__":
    main()
