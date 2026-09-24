#!/usr/bin/env python3
"""
generate_unused_astar_samples_for_probing.py

Find A* token-cache rows that were not included in the saved train/val/test
split, verify that their cached gold traces solve their targets exactly, and
save their indices for later probing experiments.

The saved split indices live in tokenisation-cache row space, so this script
uses that same cache rather than applying them to the raw A* dataset order.
The generated file does not duplicate prompt or trace strings.

This script checks that every unused gold trace:

  1. Has a parseable TGT value in its prompt
  2. Contains exactly one SUCCESS token
  3. Has a final pre-SUCCESS state containing TGT
  4. Has a parseable EXPR
  5. Has an EXPR that evaluates exactly to TGT

This audits the gold A* traces. It does not yet label the fine-tuned model's
generated outputs as correct or incorrect. That will happen in the later
activation-collection/probing script.
"""

import ast
import os
import re
from collections import Counter
from pathlib import Path

import torch
from tqdm.auto import tqdm

# Use the same A* tokenizer contract as the split generator and trainer.
os.environ["COUNTDOWN_STAGE"] = "astar"
import pretrain_and_sft_model as pm

# =========================
# DEFAULT CONFIG
# =========================

RUN_TAG = "exp31"
SPLIT_KIND = "target_random"

ASTAR_DIR = "/workspace/user/chainsearch_archives/data_pts/astar"

ASTAR_PATH = os.path.join(
    ASTAR_DIR,
    "astar_data_NALL_S6-6_T{6:100-999}_exp2500_trace10000_sw8_stick0_"
    "compact0_useh20_h2all0_h2k1.0_seed42_fmtv2.pt",
)

SAFE_MAX_LEN = 2048
LONG_SEQ_POLICY = "drop"

TOKEN_CACHE_PATH = os.path.join(
    ASTAR_DIR,
    "tokenisation_cache",
    f"{RUN_TAG}_astar_max{SAFE_MAX_LEN}_{LONG_SEQ_POLICY}.pt",
)

SPLIT_PATH = os.path.join(
    ASTAR_DIR,
    f"{RUN_TAG}_astar_{SPLIT_KIND}_train_val_test_split.pt",
)

OUT_PATH = os.path.join(
    ASTAR_DIR,
    f"{RUN_TAG}_astar_{SPLIT_KIND}_unused_probe_indices.pt",
)

OVERWRITE = True
NUM_EXAMPLES_TO_PRINT = 10
MAX_BAD_EXAMPLES_TO_PRINT = 10


# =========================
# PARSING
# =========================

TGT_RE = re.compile(r"\bTGT:\s*(\d+)")
EXPR_RE = re.compile(r"\bEXPR\s+(.+?)\s*$", re.DOTALL)
SUCCESS_RE = re.compile(r"\bSUCCESS\b")

# Captures the contents of every state written after an arrow:
#
#     -> [ 1, 2, 100 ]
#
ARROW_STATE_RE = re.compile(r"->\s*\[\s*([^\]]*?)\s*\]")


def parse_target(prompt: str) -> int:
    """Parse the integer TGT value from an A* prompt."""
    match = TGT_RE.search(prompt)
    if not match:
        raise ValueError(f"Could not parse TGT from prompt: {prompt!r}")
    return int(match.group(1))


def extract_expr(trace: str) -> str:
    """Extract the final EXPR string from a gold trace."""
    match = EXPR_RE.search(trace.strip())
    if not match:
        raise ValueError("Trace has no parseable EXPR.")
    return match.group(1).strip()


def parse_state_values(state_text: str) -> list[int]:
    """Parse the integer values from a textual state."""
    return [int(x) for x in re.findall(r"-?\d+", state_text)]


def evaluate_expr(expr_text: str) -> int:
    """
    Safely evaluate a Countdown EXPR using integer arithmetic only.

    Allowed operations:

      +
      -
      *
      /

    Division must be exact integer division.
    """

    def evaluate_node(node) -> int:
        if isinstance(node, ast.Expression):
            return evaluate_node(node.body)

        if isinstance(node, ast.Constant):
            if type(node.value) is not int:
                raise ValueError(
                    f"Non-integer constant in EXPR: {node.value!r}"
                )
            return int(node.value)

        # Compatibility with older Python AST representations.
        if hasattr(ast, "Num") and isinstance(node, ast.Num):
            if type(node.n) is not int:
                raise ValueError(
                    f"Non-integer number in EXPR: {node.n!r}"
                )
            return int(node.n)

        if isinstance(node, ast.UnaryOp):
            value = evaluate_node(node.operand)

            if isinstance(node.op, ast.USub):
                return -value

            if isinstance(node.op, ast.UAdd):
                return value

            raise ValueError(
                f"Unsupported unary operator in EXPR: {ast.dump(node.op)}"
            )

        if isinstance(node, ast.BinOp):
            left = evaluate_node(node.left)
            right = evaluate_node(node.right)

            if isinstance(node.op, ast.Add):
                return left + right

            if isinstance(node.op, ast.Sub):
                return left - right

            if isinstance(node.op, ast.Mult):
                return left * right

            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ValueError("Division by zero in EXPR.")

                if left % right != 0:
                    raise ValueError(
                        f"Non-integer division in EXPR: {left} / {right}"
                    )

                return left // right

            raise ValueError(
                f"Unsupported binary operator in EXPR: {ast.dump(node.op)}"
            )

        raise ValueError(
            f"Unsupported AST node in EXPR: {ast.dump(node)}"
        )

    tree = ast.parse(expr_text, mode="eval")
    return int(evaluate_node(tree))


# =========================
# INDEX / IO HELPERS
# =========================

def normalize_indices(values, name: str, dataset_size: int):
    """
    Convert a saved index sequence into both a list and a set.

    Also checks for duplicate or out-of-range indices.
    """
    indices = [int(i) for i in values]
    unique_indices = set(indices)

    if len(indices) != len(unique_indices):
        raise RuntimeError(
            f"{name} contains "
            f"{len(indices) - len(unique_indices):,} duplicate indices."
        )

    bad = [i for i in indices if i < 0 or i >= dataset_size]
    if bad:
        raise RuntimeError(
            f"{name} contains out-of-range indices. "
            f"Dataset size={dataset_size:,}; examples={bad[:10]}"
        )

    return indices, unique_indices

def cache_row_ids(blob: dict, idx: int) -> list[int]:
    """
    Return one complete encoded example from token-cache row space.
    """
    start = int(blob["offsets"][idx].item())
    length = int(blob["lengths"][idx].item())
    return blob["flat_ids"][start:start + length].tolist()


def decode_cache_example(
    blob: dict,
    idx: int,
    tok: pm.Tokenizer,
) -> tuple[str, str]:
    """
    Decode one token-cache row into its prompt and gold trace.
    """
    ids = cache_row_ids(blob, idx)
    sep_positions = [
        i
        for i, token_id in enumerate(ids)
        if token_id == tok.sep_id
    ]

    if len(sep_positions) != 1:
        raise ValueError(
            f"Cache row {idx} has {len(sep_positions)} SEP tokens; "
            "expected exactly 1."
        )

    sep = sep_positions[0]
    prompt = tok.decode(ids[:sep]).strip()
    trace = tok.decode(ids[sep + 1:]).strip()

    if not prompt:
        raise ValueError(f"Cache row {idx} decoded to an empty prompt.")
    if not trace:
        raise ValueError(f"Cache row {idx} decoded to an empty trace.")

    return prompt, trace

def derive_target_set(
    blob: dict,
    indices: list[int],
    tok: pm.Tokenizer,
    description: str,
) -> set[int]:
    """Derive a target set from token-cache rows."""
    targets = set()

    for idx in tqdm(
        indices,
        desc=description,
        dynamic_ncols=True,
    ):
        prompt, _ = decode_cache_example(blob, idx, tok)
        targets.add(parse_target(prompt))

    return targets


def safe_torch_save(obj, path: str) -> None:
    """Atomically save a torch artifact."""
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")

    torch.save(obj, tmp_path)
    os.replace(tmp_path, final_path)


# =========================
# MAIN
# =========================

def main() -> None:
    width = 88

    print("=" * width)
    print("UNUSED A* PROBE-INDEX GENERATOR AND EXACT-TARGET AUDIT".center(width))
    print("=" * width)
    print(f"Run tag:       {RUN_TAG}")
    print(f"Split kind:    {SPLIT_KIND}")
    print(f"A* source:     {ASTAR_PATH}")
    print(f"Token cache:   {TOKEN_CACHE_PATH}")
    print(f"Split file:    {SPLIT_PATH}")
    print(f"Output file:   {OUT_PATH}")
    print("=" * width)
    print()

    if not os.path.isfile(TOKEN_CACHE_PATH):
        raise FileNotFoundError(
            f"A* tokenisation cache not found: {TOKEN_CACHE_PATH}"
        )

    if not os.path.isfile(SPLIT_PATH):
        raise FileNotFoundError(f"Split file not found: {SPLIT_PATH}")

    if os.path.exists(OUT_PATH) and not OVERWRITE:
        raise FileExistsError(
            f"Output already exists and OVERWRITE=False: {OUT_PATH}"
        )

    tok = pm.Tokenizer()

    print("Loading A* tokenisation cache...", flush=True)
    blob, _, _, _, _, _ = pm._load_enc_blob(
        TOKEN_CACHE_PATH,
        tok,
        safe_max_len=SAFE_MAX_LEN,
        long_seq_policy=LONG_SEQ_POLICY,
    )

    print("Loading saved split...", flush=True)
    split = torch.load(
        SPLIT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(split, dict):
        raise TypeError(
            f"Expected split dict, got {type(split).__name__}"
        )

    required_cache_keys = {
        "flat_ids",
        "offsets",
        "lengths",
        "bk_counts",
    }
    missing_cache = required_cache_keys - set(blob)

    if missing_cache:
        raise KeyError(
            f"Token cache is missing keys: {sorted(missing_cache)}"
        )

    required_split_keys = {"train_idx", "val_idx", "test_idx"}
    missing_split = required_split_keys - set(split)

    if missing_split:
        raise KeyError(
            f"Split file is missing keys: {sorted(missing_split)}"
        )

    # The split file should identify the exact token cache from which its
    # indices were generated. Refuse to proceed if it points somewhere else.
    split_cache_path = split.get("enc_cache_path")
    if split_cache_path is not None:
        if (
            os.path.realpath(str(split_cache_path))
            != os.path.realpath(TOKEN_CACHE_PATH)
        ):
            raise RuntimeError(
                "Split/cache mismatch:\n"
                f"  split was built from: {split_cache_path}\n"
                f"  script is loading:    {TOKEN_CACHE_PATH}"
            )

    split_safe_max_len = split.get("safe_max_len")
    if (
        split_safe_max_len is not None
        and int(split_safe_max_len) != SAFE_MAX_LEN
    ):
        raise RuntimeError(
            f"Split safe_max_len={split_safe_max_len} does not match "
            f"script SAFE_MAX_LEN={SAFE_MAX_LEN}."
        )

    split_policy = split.get("long_seq_policy")
    if (
        split_policy is not None
        and str(split_policy).lower() != LONG_SEQ_POLICY.lower()
    ):
        raise RuntimeError(
            f"Split long_seq_policy={split_policy!r} does not match "
            f"script LONG_SEQ_POLICY={LONG_SEQ_POLICY!r}."
        )

    dataset_size = int(blob["lengths"].numel())

    train_idx, train_set = normalize_indices(
        split["train_idx"],
        "train_idx",
        dataset_size,
    )

    val_idx, val_set = normalize_indices(
        split["val_idx"],
        "val_idx",
        dataset_size,
    )

    test_idx, test_set = normalize_indices(
        split["test_idx"],
        "test_idx",
        dataset_size,
    )

    # -------------------------
    # Check index disjointness
    # -------------------------

    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    print()
    print("INDEX-LEVEL SPLIT CHECK")
    print(f"Total cached examples:   {dataset_size:>12,}")
    print(f"Train indices:           {len(train_set):>12,}")
    print(f"Validation indices:      {len(val_set):>12,}")
    print(f"Test indices:            {len(test_set):>12,}")
    print(f"Train/val overlap:       {len(train_val_overlap):>12,}")
    print(f"Train/test overlap:      {len(train_test_overlap):>12,}")
    print(f"Val/test overlap:        {len(val_test_overlap):>12,}")

    if train_val_overlap:
        raise RuntimeError("Train/validation index overlap detected.")

    if train_test_overlap:
        raise RuntimeError("Train/test index overlap detected.")

    if val_test_overlap:
        raise RuntimeError("Validation/test index overlap detected.")

    used_set = train_set | val_set | test_set

    unused_idx = [
        idx
        for idx in range(dataset_size)
        if idx not in used_set
    ]

    expected_unused = (
        dataset_size
        - len(train_set)
        - len(val_set)
        - len(test_set)
    )

    if len(unused_idx) != expected_unused:
        raise RuntimeError(
            f"Unused-index bookkeeping mismatch: "
            f"computed={len(unused_idx):,}, "
            f"expected={expected_unused:,}"
        )

    print(f"Used total:              {len(used_set):>12,}")
    print(f"Unused examples:         {len(unused_idx):>12,}")
    print()

    # -------------------------
    # Obtain split target sets
    # -------------------------

    saved_train_targets = split.get("train_targets")
    saved_val_targets = split.get("val_targets")
    saved_test_targets = split.get("test_targets")

    if saved_train_targets:
        train_targets = {int(t) for t in saved_train_targets}
    else:
        print(
            "No train_targets metadata found. "
            "Deriving train targets from prompts..."
        )
        train_targets = derive_target_set(
            blob,
            train_idx,
            tok,
            "Deriving train targets",
        )

    if saved_val_targets:
        val_targets = {int(t) for t in saved_val_targets}
    else:
        print(
            "No val_targets metadata found. "
            "Deriving validation targets from prompts..."
        )
        val_targets = derive_target_set(
            blob,
            val_idx,
            tok,
            "Deriving validation targets",
        )

    if saved_test_targets:
        test_targets = {int(t) for t in saved_test_targets}
    else:
        print(
            "No test_targets metadata found. "
            "Deriving test targets from prompts..."
        )
        test_targets = derive_target_set(
            blob,
            test_idx,
            tok,
            "Deriving test targets",
        )

    # -------------------------
    # Audit unused gold traces
    # -------------------------

    failure_counts = Counter()
    bad_examples = []

    unused_targets_aligned = []
    unused_back_counts_aligned = []

    unused_target_counts = Counter()
    unused_back_hist = Counter()

    n_exact = 0
    n_failed_records = 0

    print("Auditing unused A* gold traces...", flush=True)

    for idx in tqdm(
        unused_idx,
        desc="Exact-target audit",
        dynamic_ncols=True,
    ):
        prompt, trace = decode_cache_example(blob, idx, tok)

        reasons = []

        # Parse target.
        target = None

        try:
            target = parse_target(prompt)
        except Exception as error:
            reasons.append("missing_or_bad_target")
            target_error = repr(error)
        else:
            target_error = None

        # Require exactly one SUCCESS token.
        success_count = len(SUCCESS_RE.findall(trace))

        if success_count == 0:
            reasons.append("missing_success")
        elif success_count > 1:
            reasons.append("multiple_success")

        # Extract and evaluate EXPR.
        expr = None
        expr_value = None
        expr_error = None

        try:
            expr = extract_expr(trace)
            expr_value = evaluate_expr(expr)
        except Exception as error:
            reasons.append("missing_or_bad_expr")
            expr_error = repr(error)

        if (
            target is not None
            and expr_value is not None
            and expr_value != target
        ):
            reasons.append("expr_not_equal_target")

        # Check that the last state immediately before SUCCESS contains TGT.
        final_state = None

        if target is not None and success_count >= 1:
            before_success = trace.split("SUCCESS", 1)[0]
            states = ARROW_STATE_RE.findall(before_success)

            if not states:
                reasons.append("missing_pre_success_state")
            else:
                final_state = parse_state_values(states[-1])

                if target not in final_state:
                    reasons.append(
                        "final_pre_success_state_missing_target"
                    )

        # Count all failure reasons for diagnostics.
        if reasons:
            n_failed_records += 1
            failure_counts.update(reasons)

            if len(bad_examples) < MAX_BAD_EXAMPLES_TO_PRINT:
                bad_examples.append(
                    {
                        "idx": idx,
                        "reasons": list(reasons),
                        "prompt": prompt,
                        "target": target,
                        "target_error": target_error,
                        "success_count": success_count,
                        "final_state": final_state,
                        "expr": expr,
                        "expr_value": expr_value,
                        "expr_error": expr_error,
                        "trace_prefix": trace[:1000],
                    }
                )

            continue

        n_exact += 1

        back_count = int(blob["bk_counts"][idx].item())
        decoded_back_count = trace.count("BACK")

        if decoded_back_count != back_count:
            raise RuntimeError(
                f"Cache row {idx} BACK-count mismatch: "
                f"metadata={back_count}, "
                f"decoded={decoded_back_count}."
            )

        unused_targets_aligned.append(target)
        unused_back_counts_aligned.append(back_count)

        unused_target_counts[target] += 1
        unused_back_hist[back_count] += 1

    # -------------------------
    # Report exactness
    # -------------------------

    print()
    print("EXACT-TARGET AUDIT")
    print(f"Unused traces checked:   {len(unused_idx):>12,}")
    print(f"Exact gold traces:       {n_exact:>12,}")
    print(f"Failed gold traces:      {n_failed_records:>12,}")
    print()

    if failure_counts:
        print("Failure-reason counts:")

        for reason, count in sorted(failure_counts.items()):
            print(f"  {reason:<44} {count:>12,}")

        print()

    if bad_examples:
        print("A few failed examples:")

        for example in bad_examples:
            print("-" * width)

            for key, value in example.items():
                print(f"{key}: {value}")

        print("-" * width)
        print()

    if n_failed_records:
        raise SystemExit(
            "FAILED: some unused A* gold traces did not pass the "
            "exact-target audit. No probe-index file was saved."
        )

    if len(unused_targets_aligned) != len(unused_idx):
        raise RuntimeError(
            "Aligned target metadata does not match unused-index length."
        )

    if len(unused_back_counts_aligned) != len(unused_idx):
        raise RuntimeError(
            "Aligned BACK-count metadata does not match unused-index length."
        )

    # -------------------------
    # Check target overlap
    # -------------------------

    unused_targets = set(unused_targets_aligned)

    unused_train_overlap = unused_targets & train_targets
    unused_val_overlap = unused_targets & val_targets
    unused_test_overlap = unused_targets & test_targets

    print("TARGET-LEVEL CHECK")
    print(f"Train target values:     {len(train_targets):>12,}")
    print(f"Validation targets:      {len(val_targets):>12,}")
    print(f"Test target values:      {len(test_targets):>12,}")
    print(f"Unused target values:    {len(unused_targets):>12,}")
    print()
    print(
        f"Unused/train overlap:    "
        f"{len(unused_train_overlap):>12,}"
    )
    print(
        f"Unused/validation overlap:"
        f"{len(unused_val_overlap):>12,}"
    )
    print(
        f"Unused/test overlap:     "
        f"{len(unused_test_overlap):>12,}"
    )
    print()

    # For target_random, unused samples came from held-out target pools.
    # Therefore they must not overlap training targets. They may overlap the
    # selected validation/test target values because val/test sampled only a
    # capped number of examples from those held-out target pools.
    if unused_train_overlap:
        raise SystemExit(
            "FAILED: unused samples contain target values that appeared "
            "in the A* fine-tuning training set."
        )

    print("UNUSED BACK HISTOGRAM")

    for back_count in sorted(unused_back_hist):
        print(
            f"  {back_count:>2} BACK: "
            f"{unused_back_hist[back_count]:>12,}"
        )

    print()
    print("UNUSED TARGET COUNTS")

    for target in sorted(unused_target_counts):
        print(
            f"  target {target:>3}: "
            f"{unused_target_counts[target]:>8,}"
        )

    # -------------------------
    # Print a few exact examples
    # -------------------------

    print()
    print("A FEW VERIFIED UNUSED EXAMPLES")

    for position, idx in enumerate(
        unused_idx[:NUM_EXAMPLES_TO_PRINT]
    ):
        target = unused_targets_aligned[position]
        back_count = unused_back_counts_aligned[position]
        prompt, trace = decode_cache_example(blob, idx, tok)
        expr = extract_expr(trace)

        print("-" * width)
        print(f"Cache index: {idx}")
        print(f"Target:      {target}")
        print(f"BACK count:  {back_count}")
        print(f"Prompt:      {prompt}")
        print(f"Gold EXPR:   {expr}")

    print("-" * width)
    print()

    # -------------------------
    # Save artifact
    # -------------------------

    payload = {
        # These indices refer to rows in TOKEN_CACHE_PATH, not raw A* rows.
        "unused_cache_idx": unused_idx,
        "index_space": "astar_tokenisation_cache_rows",

        # These two lists are aligned position-by-position with unused_idx.
        "unused_targets": unused_targets_aligned,
        "unused_back_counts": unused_back_counts_aligned,

        "source_astar_path": ASTAR_PATH,
        "source_token_cache_path": TOKEN_CACHE_PATH,
        "source_split_path": SPLIT_PATH,
        "safe_max_len": SAFE_MAX_LEN,
        "long_seq_policy": LONG_SEQ_POLICY,

        "run_tag": split.get("run_tag", RUN_TAG),
        "split_kind": split.get("split_kind", SPLIT_KIND),
        "split_type": split.get("split_type"),

        "n_total": dataset_size,
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
        "n_used": len(used_set),
        "n_unused": len(unused_idx),

        "train_targets": sorted(train_targets),
        "val_targets": sorted(val_targets),
        "test_targets": sorted(test_targets),
        "unused_target_values": sorted(unused_targets),

        "unused_target_counts": dict(
            sorted(unused_target_counts.items())
        ),
        "unused_back_hist": dict(
            sorted(unused_back_hist.items())
        ),

        "exactness_audit": {
            "n_checked": len(unused_idx),
            "n_exact": n_exact,
            "n_failed": n_failed_records,
            "required_exactly_one_success": True,
            "required_target_in_final_pre_success_state": True,
            "required_expr_equal_target": True,
        },
    }

    if os.path.exists(OUT_PATH) and OVERWRITE:
        print(f"Overwriting existing output: {OUT_PATH}")

    safe_torch_save(payload, OUT_PATH)

    output_size_mb = os.path.getsize(OUT_PATH) / (1024 ** 2)

    print("=" * width)
    print(
        f"PASS: all {len(unused_idx):,} unused A* gold traces "
        f"solve their targets exactly."
    )
    print(
        "PASS: unused target values are disjoint from "
        "fine-tuning train targets."
    )
    print(f"Saved token-cache probe indices to: {OUT_PATH}")
    print(f"Output size: {output_size_mb:.2f} MiB")
    print("=" * width)


if __name__ == "__main__":
    main()