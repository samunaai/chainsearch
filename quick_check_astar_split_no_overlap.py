#!/usr/bin/env python3
"""
quick_check_astar_split_no_overlap.py

Independently audit a saved A* train/val/test/probe split.

Checks:
  - all indices are valid and unique
  - train/val/test/probe/unassigned rows are pairwise disjoint
  - those five sets cover the complete token cache exactly once
  - train/val/test/probe target values are pairwise disjoint
  - saved target metadata matches targets recomputed from the token cache
  - train and probe contain every row belonging to their target values
  - unassigned rows are exactly the unused validation/test-pool rows
  - saved BACK histograms match the token cache
  - validation and test BACK quotas are exact
"""

from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import median

import torch


RUN_TAG = "exp35"

SPLIT_PATH = Path(
    "/workspace/user/chainsearch_archives/data_pts/astar/"
    f"{RUN_TAG}_astar_target_random_train_val_test_probe_split.pt"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAILED: {message}")


def target_from_cache_row(
    blob: dict,
    idx: int,
    tgt_id: int,
    sep_id: int,
    digit_by_id: dict[int, str],
) -> int:
    start = int(blob["offsets"][idx].item())
    length = int(blob["lengths"][idx].item())

    # The prompt and SEP occur well before this limit.
    ids = blob["flat_ids"][
        start:start + min(length, 128)
    ].tolist()

    try:
        tgt_pos = ids.index(tgt_id)
        sep_pos = ids.index(sep_id, tgt_pos + 1)
    except ValueError as exc:
        raise SystemExit(
            f"FAILED: could not parse target from cache row {idx}."
        ) from exc

    digits: list[str] = []

    for token_id in ids[tgt_pos + 1:sep_pos]:
        digit = digit_by_id.get(int(token_id))

        if digit is None:
            if digits:
                break
            continue

        digits.append(digit)

    require(bool(digits), f"cache row {idx} has no target digits.")
    return int("".join(digits))


def back_hist(indices: list[int], bk_counts: list[int]) -> dict[int, int]:
    return dict(
        sorted(
            Counter(
                int(bk_counts[idx])
                for idx in indices
            ).items()
        )
    )


def distribution_text(counts: Counter) -> str:
    values = sorted(counts.values())

    return (
        f"mean={sum(values) / len(values):.2f}, "
        f"median={median(values):.1f}, "
        f"min={min(values):,}, "
        f"max={max(values):,}"
    )


def main() -> None:
    require(SPLIT_PATH.is_file(), f"split file not found: {SPLIT_PATH}")

    split = torch.load(
        SPLIT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    required_keys = {
        "train_idx",
        "val_idx",
        "test_idx",
        "probe_idx",
        "unassigned_idx",
        "train_targets",
        "val_targets",
        "test_targets",
        "probe_targets",
        "val_target_pool",
        "test_target_pool",
        "probe_target_pool",
        "enc_cache_path",
        "all_back_buckets",
        "train_back_hist",
        "val_back_hist",
        "test_back_hist",
        "probe_back_hist",
        "val_per_back",
        "test_per_back",
        "val_targets_requested",
        "test_targets_requested",
    }

    missing = required_keys - set(split)
    require(
        not missing,
        f"split file is missing keys: {sorted(missing)}",
    )

    require(
        split.get("run_tag") == RUN_TAG,
        f"expected run_tag={RUN_TAG!r}, "
        f"got {split.get('run_tag')!r}",
    )
    require(
        split.get("split_kind") == "target_random",
        f"expected target_random, got {split.get('split_kind')!r}",
    )
    require(
        "train_val_test_probe" in str(split.get("split_type")),
        f"unexpected split_type: {split.get('split_type')!r}",
    )

    cache_path = Path(split["enc_cache_path"])
    require(cache_path.is_file(), f"token cache not found: {cache_path}")

    blob = torch.load(
        cache_path,
        map_location="cpu",
        weights_only=False,
    )

    n_examples = int(blob["lengths"].numel())
    bk_counts = blob["bk_counts"].tolist()

    index_lists = {
        "train": list(split["train_idx"]),
        "val": list(split["val_idx"]),
        "test": list(split["test_idx"]),
        "probe": list(split["probe_idx"]),
        "unassigned": list(split["unassigned_idx"]),
    }
    index_sets = {
        name: set(indices)
        for name, indices in index_lists.items()
    }

    for name, indices in index_lists.items():
        require(
            len(indices) == len(index_sets[name]),
            f"{name} contains duplicate indices.",
        )
        require(bool(indices), f"{name} is empty.")
        require(
            min(indices) >= 0 and max(indices) < n_examples,
            f"{name} contains an out-of-range index.",
        )

    for left, right in combinations(index_sets, 2):
        require(
            index_sets[left].isdisjoint(index_sets[right]),
            f"{left}/{right} index overlap detected.",
        )

    all_rows = set().union(*index_sets.values())

    require(
        len(all_rows) == n_examples,
        "the five index sets do not cover the token cache.",
    )
    require(
        sum(len(values) for values in index_lists.values())
        == n_examples,
        "rows are not covered exactly once.",
    )

    vocab = list(blob["vocab"])
    tok2id = {
        token: token_id
        for token_id, token in enumerate(vocab)
    }

    require("TGT:" in tok2id, "token cache has no TGT token.")
    require("SEP" in tok2id, "token cache has no SEP token.")

    digit_by_id = {
        tok2id[str(digit)]: str(digit)
        for digit in range(10)
    }

    print(f"Recomputing targets from {n_examples:,} cache rows...")

    row_targets = [
        target_from_cache_row(
            blob,
            idx,
            tok2id["TGT:"],
            tok2id["SEP"],
            digit_by_id,
        )
        for idx in range(n_examples)
    ]

    target_counts = {
        name: Counter(
            row_targets[idx]
            for idx in indices
        )
        for name, indices in index_lists.items()
    }
    target_sets = {
        name: set(counts)
        for name, counts in target_counts.items()
    }

    active_names = ("train", "val", "test", "probe")

    for left, right in combinations(active_names, 2):
        require(
            target_sets[left].isdisjoint(target_sets[right]),
            f"{left}/{right} target overlap detected.",
        )

    all_targets = set(row_targets)
    assigned_targets = set().union(
        *(target_sets[name] for name in active_names)
    )

    require(
        assigned_targets == all_targets,
        "train/val/test/probe do not cover every target value.",
    )

    for name in active_names:
        saved_targets = set(split[f"{name}_targets"])

        require(
            target_sets[name] == saved_targets,
            f"{name} target metadata disagrees with the token cache.",
        )
        require(
            int(split[f"{name}_target_count"])
            == len(target_sets[name]),
            f"{name}_target_count metadata is incorrect.",
        )

    require(
        set(split["val_target_pool"]) == target_sets["val"],
        "validation target-pool metadata is incorrect.",
    )
    require(
        set(split["test_target_pool"]) == target_sets["test"],
        "test target-pool metadata is incorrect.",
    )
    require(
        set(split["probe_target_pool"]) == target_sets["probe"],
        "probe target-pool metadata is incorrect.",
    )

    require(
        len(target_sets["val"])
        == int(split["val_targets_requested"]),
        "validation target count differs from the request.",
    )
    require(
        len(target_sets["test"])
        == int(split["test_targets_requested"]),
        "test target count differs from the request.",
    )

    expected_train = {
        idx
        for idx, target in enumerate(row_targets)
        if target in target_sets["train"]
    }
    expected_probe = {
        idx
        for idx, target in enumerate(row_targets)
        if target in target_sets["probe"]
    }

    require(
        index_sets["train"] == expected_train,
        "train does not contain every row from its targets.",
    )
    require(
        index_sets["probe"] == expected_probe,
        "probe does not contain every row from its targets.",
    )

    evaluation_targets = target_sets["val"] | target_sets["test"]
    evaluation_pool_rows = {
        idx
        for idx, target in enumerate(row_targets)
        if target in evaluation_targets
    }
    expected_unassigned = (
        evaluation_pool_rows
        - index_sets["val"]
        - index_sets["test"]
    )

    require(
        index_sets["unassigned"] == expected_unassigned,
        "unassigned rows are not exactly the unused val/test-pool rows.",
    )
    require(
        target_sets["unassigned"].issubset(evaluation_targets),
        "unassigned rows contain a train or probe target.",
    )

    computed_histograms = {
        name: back_hist(indices, bk_counts)
        for name, indices in index_lists.items()
        if name != "unassigned"
    }

    for name in active_names:
        require(
            computed_histograms[name]
            == split[f"{name}_back_hist"],
            f"{name} BACK histogram metadata is incorrect.",
        )

    back_buckets = [
        int(bucket)
        for bucket in split["all_back_buckets"]
    ]

    expected_val_hist = {
        bucket: int(split["val_per_back"])
        for bucket in back_buckets
    }
    expected_test_hist = {
        bucket: int(split["test_per_back"])
        for bucket in back_buckets
    }

    require(
        computed_histograms["val"] == expected_val_hist,
        "validation BACK quotas are incorrect.",
    )
    require(
        computed_histograms["test"] == expected_test_hist,
        "test BACK quotas are incorrect.",
    )

    print()
    print("=" * 88)
    print("A* TRAIN/VAL/TEST/PROBE SPLIT AUDIT".center(88))
    print("=" * 88)
    print(f"Split path:       {SPLIT_PATH}")
    print(f"Token cache:      {cache_path}")
    print(f"Run tag:          {split.get('run_tag')}")
    print(f"Split type:       {split.get('split_type')}")
    print()

    print("ROWS AND TARGETS")
    for name in index_lists:
        print(
            f"  {name:<10} "
            f"rows={len(index_lists[name]):>10,} "
            f"targets={len(target_sets[name]):>4,}"
        )

    print()
    print("TARGET ROWS PER TARGET")
    for name in active_names:
        print(
            f"  {name:<10} "
            f"{distribution_text(target_counts[name])}"
        )

    print()
    print("BACK HISTOGRAMS")
    for name in active_names:
        print(
            f"  {name:<10} "
            f"{computed_histograms[name]}"
        )

    print()
    print("IN-FAMILY OVERLAPS — INFORMATIONAL AND ALLOWED")
    print(
        f"  train/val:   "
        f"{split.get('train_val_family_overlap')}"
    )
    print(
        f"  train/test:  "
        f"{split.get('train_test_family_overlap')}"
    )
    print(
        f"  val/test:    "
        f"{split.get('val_test_family_overlap')}"
    )
    print(
        f"  train/probe: "
        f"{split.get('train_probe_family_overlap')}"
    )
    print(
        f"  val/probe:   "
        f"{split.get('val_probe_family_overlap')}"
    )
    print(
        f"  test/probe:  "
        f"{split.get('test_probe_family_overlap')}"
    )

    print()
    print("CHECKS")
    print("  Pairwise index overlap:                   0")
    print("  Pairwise train/val/test/probe targets:    0")
    print(f"  Rows covered exactly once:       {n_examples:>10,}")
    print(f"  Target values covered:           {len(all_targets):>10,}")
    print("  Train and probe full-target coverage:     yes")
    print("  Unassigned-row policy:                    valid")
    print("  Saved target metadata:                    verified")
    print("  Saved BACK histograms:                    verified")
    print("  Validation/test BACK quotas:              exact")
    print()
    print(
        f"PASS: {RUN_TAG} split is complete, index-disjoint, "
        "target-disjoint, and quota-consistent."
    )
    print("=" * 88)


if __name__ == "__main__":
    main()