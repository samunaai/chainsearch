#!/usr/bin/env python3
"""
generate_custom_astar_splits.py
================================
Generate custom train/val/test/probe split indices for the A* Countdown dataset.

Supported split kinds:
  - in_family: old exp30 behavior. Split by IN-number family, so all examples
    with the same IN: [ ... ] multiset stay together and train/val/test IN
    families are mutually disjoint.
  - target_random: split by exact target value. Target values are disjoint
    across train/val/test/probe, but IN families are allowed to overlap.
    Validation and test examples are sampled per BACK bucket from dedicated
    target pools, while all rows from probe-only targets form the probe pool.
    This is the default experiment for the target-heldout fine-tuning run.
  - target_mod: deterministic near-target split. The default preset kept in
    this file is the fuller 5-off setting:
        train: T % 10 in {0,1,2,3,4,7,8,9}
        val:   T % 10 == 6
        test:  T % 10 == 5
    A sparser setting, train on T % 10 == 0 only, is also kept below.

The output .pt is consumed by pretrain_and_sft_model.py via:
    split_mode="train_val_test"
    split_path="..."
The saved probe_idx is consumed by the later probing pipeline.
"""

import os

# Force astar config before importing pretrain_and_sft_model.
os.environ["COUNTDOWN_STAGE"] = "astar"

import argparse
import logging
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import torch
from tqdm import tqdm

import pretrain_and_sft_model as pm


DEFAULT_DATASET_PATH = (
    "/workspace/user/chainsearch_archives/data_pts/astar/"
    "astar_data_NALL_S6-6_T{6:100-999}_exp2500_trace10000_sw8_stick0_"
    "compact0_useh20_h2all0_h2k1.0_seed42_fmtv2.pt"
)

DEFAULT_TOKEN_CACHE_DIR = (
    "/workspace/user/chainsearch_archives/data_pts/astar/tokenisation_cache"
)

DEFAULT_SPLIT_DIR = (
    "/workspace/user/chainsearch_archives/data_pts/astar"
)

# =========================
# DEFAULT EXPERIMENT CONFIG
# =========================
#
# The intended no-flag command is:
#
#     python generate_custom_astar_splits.py &> out_custom_astar_splits.txt
#
# For now, that should run the target-random split experiment.
# To switch the no-flag behavior later, change DEFAULT_SPLIT_KIND here.
DEFAULT_SPLIT_KIND = "target_random"

DEFAULT_RUN_TAG = "exp35"

# Mode-specific train fraction. For target_random this is an expected final
# fraction used for auditing. Targets left after val/test/probe reservation
# are assigned to training. For target_mod it remains metadata.
DEFAULT_TRAIN_FRAC_BY_SPLIT_KIND = {
    "in_family": 0.20,
    "target_random": 0.80,
    "target_mod": 0.80,
}

DEFAULT_VAL_PER_BACK = 100
DEFAULT_TEST_PER_BACK = 1000
DEFAULT_VAL_TARGETS = 10
DEFAULT_TEST_TARGETS = 60
DEFAULT_PROBE_EXAMPLES = 101000
DEFAULT_VAL_HELDOUT_FRAC = 0.50
DEFAULT_OVERWRITE = True

TARGET_MOD_PRESETS = {
    # Clean but small: train only on every 10th target, test on 5-off targets.
    "sparse_5off": {
        "base": 10,
        "train": "0",
        "val": "6",
        "test": "5",
    },
    # Recommended later: train on most residues, hold out 5-off targets.
    "full_5off": {
        "base": 10,
        "train": "0-4,7-9",
        "val": "6",
        "test": "5",
    },
}

DEFAULT_TARGET_MOD_PRESET = "full_5off"

 
def parse_args():
    p = argparse.ArgumentParser()

    target_mod_defaults = TARGET_MOD_PRESETS[DEFAULT_TARGET_MOD_PRESET]

    p.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    p.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    p.add_argument("--token-cache-dir", default=DEFAULT_TOKEN_CACHE_DIR)
    p.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR)
    p.add_argument("--split-path", default=None)

    p.add_argument("--safe-max-len", type=int, default=2048)
    p.add_argument("--long-seq-policy", default="drop")
    p.add_argument("--encode-workers", type=int, default=24)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=None)

    p.add_argument(
        "--split-kind",
        choices=("in_family", "target_random", "target_mod"),
        default=DEFAULT_SPLIT_KIND,
        help=(
            "How to create train/val/test splits. "
            "in_family preserves the old exp30 behavior; target_random groups "
            "by exact TGT value; target_mod groups by TGT modulo --target-mod-base."
        ),
    )

    # Small validation set for checkpointing / early stopping.
    p.add_argument("--val-per-back", type=int, default=DEFAULT_VAL_PER_BACK)
    p.add_argument(
        "--val-targets",
        type=int,
        default=DEFAULT_VAL_TARGETS,
    )

    # Deeper final test set. This is only used at final evaluation by the trainer.
    p.add_argument("--test-per-back", type=int, default=DEFAULT_TEST_PER_BACK)
    p.add_argument(
        "--test-targets",
        type=int,
        default=DEFAULT_TEST_TARGETS,
    )

    # Approximate number of rows reserved from probe-only target values.
    p.add_argument(
        "--probe-examples",
        type=int,
        default=DEFAULT_PROBE_EXAMPLES,
    )

    # For in_family: heldout IN families are split into validation/test pools.
    # For target_random: heldout target values are split into validation/test pools.
    # target_mod does not use this; it uses explicit target-mod buckets below.
    #
    # Keep --val-family-frac as a backwards-compatible alias, but internally
    # treat it as a generic heldout-group fraction.
    p.add_argument(
        "--val-heldout-frac",
        "--val-family-frac",
        dest="val_heldout_frac",
        type=float,
        default=DEFAULT_VAL_HELDOUT_FRAC,
    )

    # Used only by --split-kind target_mod. These are comma-separated residue
    # sets, with optional inclusive ranges such as "0-4,7,8,9".
    p.add_argument("--target-mod-base", type=int, default=int(target_mod_defaults["base"]))
    p.add_argument("--train-target-mods", default=str(target_mod_defaults["train"]))
    p.add_argument("--val-target-mods", default=str(target_mod_defaults["val"]))
    p.add_argument("--test-target-mods", default=str(target_mod_defaults["test"]))

    p.add_argument("--overwrite", dest="overwrite", action="store_true", default=DEFAULT_OVERWRITE)
    p.add_argument("--no-overwrite", dest="overwrite", action="store_false")

    args = p.parse_args()
    if args.train_frac is None:
        args.train_frac = DEFAULT_TRAIN_FRAC_BY_SPLIT_KIND[args.split_kind]
    return args


def prefix_ids_for_idx(blob, idx: int, max_prefix: int = 128) -> list[int]:
    start = int(blob["offsets"][idx].item())
    length = int(blob["lengths"][idx].item())
    n = min(length, max_prefix)
    return blob["flat_ids"][start:start + n].tolist()

def require_valid_indices(
    name: str,
    indices: list[int],
    n_examples: int,
) -> None:
    if len(indices) != len(set(indices)):
        raise RuntimeError(
            f"{name} contains duplicate indices."
        )

    if indices and (
        min(indices) < 0
        or max(indices) >= n_examples
    ):
        raise RuntimeError(
            f"{name} contains an out-of-range index."
        )

def in_family_from_prefix_ids(prefix_ids: list[int], tok: pm.Tokenizer) -> tuple[int, ...]:
    """
    Parse the IN-number family from token ids.

    Example:
        IN: [ 1, 2, 3, 4, 5, 6 ] TGT: 248

    returns:
        (1, 2, 3, 4, 5, 6)

    Sorting makes this robust to accidental input-order differences.
    Duplicate numbers are preserved.
    """
    lbr = tok.tok2id["["]
    rbr = tok.tok2id["]"]
    digit_id_to_char = {tok.tok2id[str(d)]: str(d) for d in range(10)}

    try:
        i = prefix_ids.index(lbr)
        j = prefix_ids.index(rbr, i + 1)
    except ValueError as e:
        raise ValueError(f"Could not parse IN family from token prefix: {prefix_ids}") from e

    nums = []
    cur_digits = []

    def flush_number():
        nonlocal cur_digits
        if cur_digits:
            nums.append(int("".join(cur_digits)))
            cur_digits = []

    for tid in prefix_ids[i + 1:j]:
        ch = digit_id_to_char.get(int(tid))
        if ch is None:
            flush_number()
        else:
            cur_digits.append(ch)

    flush_number()

    if not nums:
        raise ValueError(f"Parsed empty IN family from token prefix: {prefix_ids}")

    return tuple(sorted(nums))

def target_from_prefix_ids(prefix_ids: list[int], tok: pm.Tokenizer) -> int:
    """
    Parse the target value from token ids.

    Example:
        IN: [ 1, 2, 3, 4, 5, 6 ] TGT: 248 SEP ...

    returns:
        248
    """
    tgt_id = tok.tok2id["TGT:"]
    sep_id = tok.tok2id["SEP"]
    digit_id_to_char = {tok.tok2id[str(d)]: str(d) for d in range(10)}

    try:
        i = prefix_ids.index(tgt_id)
    except ValueError as e:
        raise ValueError(f"Could not parse target from token prefix: {prefix_ids}") from e

    try:
        j = prefix_ids.index(sep_id, i + 1)
    except ValueError:
        j = len(prefix_ids)

    digits = []
    for tid in prefix_ids[i + 1:j]:
        ch = digit_id_to_char.get(int(tid))
        if ch is None:
            if digits:
                break
            continue
        digits.append(ch)

    if not digits:
        raise ValueError(f"Parsed empty target from token prefix: {prefix_ids}")

    return int("".join(digits))


def parse_int_set(spec: str, *, name: str) -> set[int]:
    """
    Parse comma-separated integers/ranges.

    Examples:
        "0"          -> {0}
        "0,1,2"      -> {0, 1, 2}
        "0-4,7,8,9"  -> {0, 1, 2, 3, 4, 7, 8, 9}
    """
    out = set()
    for raw_part in str(spec).split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            pieces = part.split("-", 1)
            if len(pieces) != 2 or not pieces[0].strip() or not pieces[1].strip():
                raise ValueError(f"Bad integer range in --{name}: {part!r}")
            lo, hi = int(pieces[0]), int(pieces[1])
            if lo > hi:
                raise ValueError(f"Bad integer range in --{name}: {part!r}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    if not out:
        raise ValueError(f"--{name} must specify at least one integer")
    return out



def hist_by_back(indices, bk_counts) -> dict[int, int]:
    return dict(sorted(Counter(int(bk_counts[i]) for i in indices).items()))


def sample_per_back_from_groups(
    *,
    groups,
    group_to_indices,
    bk_counts,
    all_back_buckets,
    per_back: int,
    rng: random.Random,
):
    by_back = defaultdict(list)

    for group in sorted(groups):
        for idx in group_to_indices[group]:
            by_back[int(bk_counts[idx])].append(idx)

    selected_idx = []
    short = {}

    for b in all_back_buckets:
        idxs = by_back.get(b, [])
        rng.shuffle(idxs)

        take = min(per_back, len(idxs))
        selected_idx.extend(idxs[:take])

        if take < per_back:
            short[b] = {"available": take, "requested": per_back}

    rng.shuffle(selected_idx)
    return selected_idx, short


def indices_from_groups(groups, group_to_indices) -> list[int]:
    return [
        idx
        for group in sorted(groups)
        for idx in group_to_indices[group]
    ]


def reserve_groups_for_back_quota(
    *,
    groups,
    group_to_indices,
    bk_counts,
    all_back_buckets,
    per_back: int,
    group_count: int,
    rng: random.Random,
    leave_per_back: int = 0,
):
    """Reserve exactly group_count groups while satisfying BACK quotas."""
    remaining = list(groups)
    rng.shuffle(remaining)
    if group_count <= 0 or group_count >= len(remaining):
        raise ValueError(
            f"Invalid target-group count: {group_count}"
        )

    group_hists = {
        group: Counter(
            int(bk_counts[idx])
            for idx in group_to_indices[group]
        )
        for group in remaining
    }
    
    remaining_totals = Counter()
    for histogram in group_hists.values():
        remaining_totals.update(histogram)

    selected = set()
    totals = Counter()

    while True:
        deficits = {
            bucket: max(0, per_back - totals[bucket])
            for bucket in all_back_buckets
        }

        quota_met = not any(deficits.values())

        if quota_met and len(selected) == group_count:
            return selected, remaining

        if len(selected) >= group_count:
            missing = {
                bucket: deficit
                for bucket, deficit in deficits.items()
                if deficit > 0
            }
            raise RuntimeError(
                "Could not satisfy BACK quotas using exactly "
                f"{group_count} target groups. "
                f"Remaining deficits: {missing}"
            )

        def selection_key(group):
            coverage = sum(
                min(
                    deficits[bucket],
                    group_hists[group][bucket],
                )
                for bucket in all_back_buckets
            )
            return coverage, -len(group_to_indices[group])

        def can_select(group):
            histogram = group_hists[group]

            for bucket in all_back_buckets:
                future_deficit = max(
                    0,
                    deficits[bucket] - histogram[bucket],
                )
                required_remaining = (
                    leave_per_back + future_deficit
                )
                available_remaining = (
                    remaining_totals[bucket]
                    - histogram[bucket]
                )

                if available_remaining < required_remaining:
                    return False

            return True

        candidates = [
            group
            for group in remaining
            if can_select(group)
        ]


        if quota_met:
            # Once quotas are met, fill the remaining target slots
            # randomly rather than preferring unusually small groups.
            best_group = (
                rng.choice(candidates)
                if candidates
                else None
            )
        else:
            best_group = max(
                candidates,
                key=selection_key,
                default=None,
            )

        if best_group is None:
            missing = {
                bucket: deficit
                for bucket, deficit in deficits.items()
                if deficit > 0
            }
            raise RuntimeError(
                "Could not reserve the requested number of target "
                "groups while preserving later BACK quotas. "
                f"Remaining deficits: {missing}"
            )

        if (
            not quota_met
            and selection_key(best_group)[0] == 0
        ):
            missing = {
                bucket: deficit
                for bucket, deficit in deficits.items()
                if deficit > 0
            }
            raise RuntimeError(
                "Remaining target groups cannot satisfy the "
                f"BACK quotas: {missing}"
            )

        remaining.remove(best_group)
        selected.add(best_group)
        totals.update(group_hists[best_group])
        remaining_totals.subtract(group_hists[best_group])


def reserve_groups_near_example_count(
    *,
    groups,
    group_to_indices,
    wanted_examples: int,
    rng: random.Random,
):
    """Reserve whole groups with a row count near wanted_examples."""
    ordered = list(groups)
    rng.shuffle(ordered)

    selected = set()
    total = 0

    for group in ordered:
        with_group = total + len(group_to_indices[group])

        if abs(with_group - wanted_examples) < abs(
            total - wanted_examples
        ):
            selected.add(group)
            total = with_group

            if total >= wanted_examples:
                break

    remaining = [
        group
        for group in ordered
        if group not in selected
    ]

    if not selected or not remaining:
        raise RuntimeError(
            "Could not create nonempty probe and "
            "training target pools."
        )

    return selected, remaining


def require_pairwise_disjoint(named_sets, *, label: str) -> None:
    for (left_name, left), (right_name, right) in combinations(
        named_sets.items(),
        2,
    ):
        overlap = left & right
        if overlap:
            raise RuntimeError(
                f"{left_name}/{right_name} {label} overlap "
                f"detected: {len(overlap):,}"
            )




def main():
    args = parse_args()

    if not (0.0 < args.train_frac < 1.0):
        raise ValueError("--train-frac must be between 0 and 1")
    if not (0.0 < args.val_heldout_frac < 1.0):
        raise ValueError("--val-heldout-frac must be between 0 and 1")
    if args.val_per_back <= 0:
        raise ValueError("--val-per-back must be positive")
    if args.test_per_back <= 0:
        raise ValueError("--test-per-back must be positive")
    if args.val_targets <= 0:
        raise ValueError("--val-targets must be positive")
    if args.test_targets <= 0:
        raise ValueError("--test-targets must be positive")
    if args.probe_examples <= 0:
        raise ValueError("--probe-examples must be positive")

    if args.target_mod_base <= 1:
        raise ValueError("--target-mod-base must be greater than 1")

    train_mods = val_mods = test_mods = None
    if args.split_kind == "target_mod":
        train_mods = parse_int_set(args.train_target_mods, name="train-target-mods")
        val_mods = parse_int_set(args.val_target_mods, name="val-target-mods")
        test_mods = parse_int_set(args.test_target_mods, name="test-target-mods")
        all_mods = train_mods | val_mods | test_mods
        bad_mods = sorted(m for m in all_mods if m < 0 or m >= args.target_mod_base)
        if bad_mods:
            raise ValueError(
                f"Target-mod residues must be in [0, {args.target_mod_base - 1}], got {bad_mods}"
            )
        if not train_mods.isdisjoint(val_mods):
            raise ValueError("--train-target-mods and --val-target-mods overlap")
        if not train_mods.isdisjoint(test_mods):
            raise ValueError("--train-target-mods and --test-target-mods overlap")
        if not val_mods.isdisjoint(test_mods):
            raise ValueError("--val-target-mods and --test-target-mods overlap")

    split_path = args.split_path
    if split_path is None:
        split_suffix = "group" if args.split_kind == "in_family" else args.split_kind
        split_tail = (
            "train_val_test_probe_split.pt"
            if args.split_kind == "target_random"
            else "train_val_test_split.pt"
        )
        split_path = os.path.join(
            args.split_dir,
            f"{args.run_tag}_astar_{split_suffix}_{split_tail}",
        )

    if os.path.exists(split_path) and not args.overwrite:
        raise FileExistsError(
            f"Split already exists: {split_path}\n"
            f"Pass --overwrite to replace it."
        )

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    tok = pm.Tokenizer()

    enc_cache_path = pm.build_enc_cache_path(
        args.token_cache_dir,
        args.safe_max_len,
        args.long_seq_policy,
        run_tag=f"{args.run_tag}_astar",
    )

    if os.path.exists(enc_cache_path):
        logging.info(f"Loading token cache: {enc_cache_path}")
        blob, _, ml, mi, mo, bk_counts = pm._load_enc_blob(
            enc_cache_path,
            tok,
            safe_max_len=args.safe_max_len,
            long_seq_policy=args.long_seq_policy,
        )
    else:
        logging.info(f"Building token cache: {enc_cache_path}")
        seqs = (
            f"{prompt} SEP {trace}"
            for prompt, trace in pm.iter_trace_bundle(args.dataset_path)
        )
        blob, _, ml, mi, mo, bk_counts = pm.get_or_build_enc_blob(
            seqs,
            tok,
            args.safe_max_len,
            args.long_seq_policy,
            enc_cache_path,
            rank=0,
            cfg={"encode_workers": args.encode_workers},
        )

    n_examples = int(blob["lengths"].numel())
    bk_counts = blob["bk_counts"].tolist()
    all_back_buckets = sorted(set(int(b) for b in bk_counts))

    logging.info(f"Encoded examples: {n_examples:,}")
    logging.info(f"BACK buckets: {all_back_buckets}")
    logging.info("Parsing IN-number families and target values...")

    family_to_indices = defaultdict(list)
    target_to_indices = defaultdict(list)
    idx_to_family = [None] * n_examples
    idx_to_target = [None] * n_examples

    for idx in tqdm(range(n_examples), desc="Grouping", dynamic_ncols=True):
        prefix_ids = prefix_ids_for_idx(blob, idx)
        fam = in_family_from_prefix_ids(prefix_ids, tok)
        tgt = target_from_prefix_ids(prefix_ids, tok)
        family_to_indices[fam].append(idx)
        target_to_indices[tgt].append(idx)
        idx_to_family[idx] = fam
        idx_to_target[idx] = tgt

    rng = random.Random(args.seed) 

    val_short = {}
    test_short = {}
    train_idx = []
    val_idx = []
    test_idx = []
    probe_idx = []
    val_pool_targets = set()
    test_pool_targets = set()
    probe_pool_targets = set()
    split_type = None

    if args.split_kind == "in_family":
        family_items = list(family_to_indices.items())
        rng.shuffle(family_items)

        target_train_examples = int(round(args.train_frac * n_examples))
        train_families = set()
        heldout_families = []

        for fam, idxs in family_items:
            if len(train_idx) < target_train_examples:
                train_families.add(fam)
                train_idx.extend(idxs)
            else:
                heldout_families.append(fam)

        rng.shuffle(train_idx)
        rng.shuffle(heldout_families)

        if len(heldout_families) < 2:
            raise RuntimeError("Not enough held-out IN families left for validation/test.")

        n_val_pool = int(round(args.val_heldout_frac * len(heldout_families)))
        n_val_pool = min(max(1, n_val_pool), len(heldout_families) - 1)

        val_pool_families = set(heldout_families[:n_val_pool])
        test_pool_families = set(heldout_families[n_val_pool:])

        val_idx, val_short = sample_per_back_from_groups(
            groups=val_pool_families,
            group_to_indices=family_to_indices,
            bk_counts=bk_counts,
            all_back_buckets=all_back_buckets,
            per_back=args.val_per_back,
            rng=rng,
        )

        test_idx, test_short = sample_per_back_from_groups(
            groups=test_pool_families,
            group_to_indices=family_to_indices,
            bk_counts=bk_counts,
            all_back_buckets=all_back_buckets,
            per_back=args.test_per_back,
            rng=rng,
        )

        split_type = "in_family_group_train_val_test_split"


    elif args.split_kind == "target_random":
        remaining_targets = list(target_to_indices)

        # Reserve exactly 10 validation targets while preserving enough
        # BACK-bucket coverage for the later test pool.
        val_pool_targets, remaining_targets = (
            reserve_groups_for_back_quota(
                groups=remaining_targets,
                group_to_indices=target_to_indices,
                bk_counts=bk_counts,
                all_back_buckets=all_back_buckets,
                per_back=args.val_per_back,
                group_count=args.val_targets,
                rng=rng,
                leave_per_back=args.test_per_back,
            )
        )

        # Reserve exactly 90 test targets.
        test_pool_targets, remaining_targets = (
            reserve_groups_for_back_quota(
                groups=remaining_targets,
                group_to_indices=target_to_indices,
                bk_counts=bk_counts,
                all_back_buckets=all_back_buckets,
                per_back=args.test_per_back,
                group_count=args.test_targets,
                rng=rng,
                leave_per_back=0,
            )
        )

        probe_pool_targets, remaining_targets = (
            reserve_groups_near_example_count(
                groups=remaining_targets,
                group_to_indices=target_to_indices,
                wanted_examples=args.probe_examples,
                rng=rng,
            )
        )

        train_targets = set(remaining_targets)

        train_idx = indices_from_groups(
            train_targets,
            target_to_indices,
        )
        probe_idx = indices_from_groups(
            probe_pool_targets,
            target_to_indices,
        )

        rng.shuffle(train_idx)
        rng.shuffle(probe_idx)

        val_idx, val_short = sample_per_back_from_groups(
            groups=val_pool_targets,
            group_to_indices=target_to_indices,
            bk_counts=bk_counts,
            all_back_buckets=all_back_buckets,
            per_back=args.val_per_back,
            rng=rng,
        )

        test_idx, test_short = sample_per_back_from_groups(
            groups=test_pool_targets,
            group_to_indices=target_to_indices,
            bk_counts=bk_counts,
            all_back_buckets=all_back_buckets,
            per_back=args.test_per_back,
            rng=rng,
        )

        expected_val = (
            args.val_per_back * len(all_back_buckets)
        )
        expected_test = (
            args.test_per_back * len(all_back_buckets)
        )

        if (
            val_short
            or test_short
            or len(val_idx) != expected_val
            or len(test_idx) != expected_test
        ):
            raise RuntimeError(
                "Reserved target pools did not produce "
                "the requested validation/test counts."
            )

        split_type = (
            "target_random_group_"
            "train_val_test_probe_split"
        )

    elif args.split_kind == "target_mod":
        train_targets = {t for t in target_to_indices if (t % args.target_mod_base) in train_mods}
        val_targets_pool = {t for t in target_to_indices if (t % args.target_mod_base) in val_mods}
        test_targets_pool = {t for t in target_to_indices if (t % args.target_mod_base) in test_mods}

        if not train_targets:
            raise RuntimeError("No train targets matched --train-target-mods.")
        if not val_targets_pool:
            raise RuntimeError("No validation targets matched --val-target-mods.")
        if not test_targets_pool:
            raise RuntimeError("No test targets matched --test-target-mods.")

        for tgt in train_targets:
            train_idx.extend(target_to_indices[tgt])
        rng.shuffle(train_idx)

        val_idx, val_short = sample_per_back_from_groups(
            groups=val_targets_pool,
            group_to_indices=target_to_indices,
            bk_counts=bk_counts,
            all_back_buckets=all_back_buckets,
            per_back=args.val_per_back,
            rng=rng,
        )

        test_idx, test_short = sample_per_back_from_groups(
            groups=test_targets_pool,
            group_to_indices=target_to_indices,
            bk_counts=bk_counts,
            all_back_buckets=all_back_buckets,
            per_back=args.test_per_back,
            rng=rng,
        )

        split_type = "target_mod_group_train_val_test_split"

    else:
        raise AssertionError(f"Unhandled split kind: {args.split_kind}")


    index_lists = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    if args.split_kind == "target_random":
        index_lists["probe"] = probe_idx

    for name, indices in index_lists.items():
        require_valid_indices(
            name,
            indices,
            n_examples,
        )


    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)
    probe_set = set(probe_idx)

    active_sets = {
        "train": train_set,
        "val": val_set,
        "test": test_set,
    }

    if args.split_kind == "target_random":
        active_sets["probe"] = probe_set

    for name, values in active_sets.items():
        if not values:
            raise RuntimeError(f"{name} split is empty.")

    require_pairwise_disjoint(
        active_sets,
        label="index",
    )

    active_union = set().union(*active_sets.values())
    unassigned_idx = sorted(
        set(range(n_examples)) - active_union
    )

    if len(active_union) + len(unassigned_idx) != n_examples:
        raise RuntimeError(
            "Split accounting does not cover the dataset."
        )

    train_families = {
        idx_to_family[i]
        for i in train_idx
    }
    val_families = {
        idx_to_family[i]
        for i in val_idx
    }
    test_families = {
        idx_to_family[i]
        for i in test_idx
    }
    probe_families = {
        idx_to_family[i]
        for i in probe_idx
    }

    train_targets = {
        idx_to_target[i]
        for i in train_idx
    }
    val_targets = {
        idx_to_target[i]
        for i in val_idx
    }
    test_targets = {
        idx_to_target[i]
        for i in test_idx
    }
    probe_targets = {
        idx_to_target[i]
        for i in probe_idx
    }

    if args.split_kind == "in_family":
        require_pairwise_disjoint(
            {
                "train": train_families,
                "val": val_families,
                "test": test_families,
            },
            label="IN-family",
        )

    elif args.split_kind == "target_random":
        require_pairwise_disjoint(
            {
                "train": train_targets,
                "val": val_pool_targets,
                "test": test_pool_targets,
                "probe": probe_pool_targets,
            },
            label="target",
        )

        all_targets = set(target_to_indices)
        assigned_targets = (
            train_targets
            | val_pool_targets
            | test_pool_targets
            | probe_pool_targets
        )

        if assigned_targets != all_targets:
            missing = all_targets - assigned_targets
            extra = assigned_targets - all_targets
            raise RuntimeError(
                "Target-pool coverage is incomplete. "
                f"Missing={len(missing)}, "
                f"extra={len(extra)}"
            )


        if val_targets != val_pool_targets:
            raise RuntimeError(
                "Validation rows do not cover every reserved "
                "validation target."
            )

        if test_targets != test_pool_targets:
            raise RuntimeError(
                "Test rows do not cover every reserved test target."
            )

        if probe_targets != probe_pool_targets:
            raise RuntimeError(
                "Probe rows do not cover exactly the "
                "reserved probe targets."
            )

        evaluation_pool_indices = set(
            indices_from_groups(
                val_pool_targets | test_pool_targets,
                target_to_indices,
            )
        )

        expected_unassigned = (
            evaluation_pool_indices
            - val_set
            - test_set
        )

        if set(unassigned_idx) != expected_unassigned:
            raise RuntimeError(
                "Unassigned rows do not exactly match the "
                "unused validation/test target-pool rows."
            )

    else:
        require_pairwise_disjoint(
            {
                "train": train_targets,
                "val": val_targets,
                "test": test_targets,
            },
            label="target",
        )
    train_val_family_overlap = len(train_families & val_families)
    train_test_family_overlap = len(train_families & test_families)
    val_test_family_overlap = len(val_families & test_families)

    train_probe_family_overlap = len(
        train_families & probe_families
    )
    val_probe_family_overlap = len(
        val_families & probe_families
    )
    test_probe_family_overlap = len(
        test_families & probe_families
    )

    train_frac_actual = len(train_idx) / n_examples

    if (
        args.split_kind == "target_random"
        and abs(train_frac_actual - args.train_frac) > 0.02
    ):
        logging.warning(
            "Actual train fraction differs from the requested "
            "fraction by more than 2 percentage points. "
            f"Requested={args.train_frac:.2%}, "
            f"actual={train_frac_actual:.2%}."
        )

    logging.info("")
    logging.info("=" * 80)
    logging.info(
        "CUSTOM A* TRAIN/VAL/TEST/PROBE SPLIT"
        if args.split_kind == "target_random"
        else "CUSTOM A* TRAIN/VAL/TEST SPLIT"
    )
    logging.info("=" * 80)
    logging.info(f"Split path:             {split_path}")
    logging.info(f"Split kind:             {args.split_kind}")
    logging.info(f"Token cache path:       {enc_cache_path}")
    logging.info(f"Total examples:         {n_examples:,}")
    logging.info(f"Total IN families:      {len(family_to_indices):,}")
    logging.info(f"Total target values:    {len(target_to_indices):,}")
    logging.info(
        f"Train examples:         "
        f"{len(train_idx):,} ({train_frac_actual:.2%})"
    )
    logging.info(f"Val examples:           {len(val_idx):,}")
    logging.info(f"Test examples:          {len(test_idx):,}")
    if args.split_kind == "target_random":
        logging.info(
            f"Probe examples:         {len(probe_idx):,} "
            f"(requested about {args.probe_examples:,})"
        )
        logging.info(
            f"Unassigned examples:    {len(unassigned_idx):,}"
        )
    logging.info(f"Train targets:          {len(train_targets):,}")
    logging.info(f"Val selected targets:   {len(val_targets):,}")
    logging.info(f"Test selected targets:  {len(test_targets):,}")
    if args.split_kind == "target_random":
        logging.info(
            f"Val target pool:        {len(val_pool_targets):,}"
        )
        logging.info(
            f"Test target pool:       {len(test_pool_targets):,}"
        )
        logging.info(
            f"Probe targets:          {len(probe_targets):,}"
        )
    logging.info(f"Train families:         {len(train_families):,}")
    logging.info(f"Val selected families:  {len(val_families):,}")
    logging.info(f"Test selected families: {len(test_families):,}")
    if args.split_kind == "target_random":
        logging.info(
            f"Probe families:         {len(probe_families):,}"
        )
    if args.split_kind == "in_family":
        logging.info("IN-family overlap policy: disallowed and checked")
    else:
        logging.info("Target overlap policy:    disallowed and checked")
        logging.info("IN-family overlap policy: allowed")
        logging.info(f"Train/val IN overlap:   {train_val_family_overlap:,}")
        logging.info(f"Train/test IN overlap:  {train_test_family_overlap:,}")
        logging.info(f"Val/test IN overlap:    {val_test_family_overlap:,}")
        if args.split_kind == "target_random":
            logging.info(
                f"Train/probe IN overlap: {train_probe_family_overlap:,}"
            )
            logging.info(
                f"Val/probe IN overlap:   {val_probe_family_overlap:,}"
            )
            logging.info(
                f"Test/probe IN overlap:  {test_probe_family_overlap:,}"
            )
    if args.split_kind == "target_mod":
        logging.info(f"Target mod base:        {args.target_mod_base}")
        logging.info(f"Train target mods:      {sorted(train_mods)}")
        logging.info(f"Val target mods:        {sorted(val_mods)}")
        logging.info(f"Test target mods:       {sorted(test_mods)}")
    logging.info("")
    logging.info(f"Train BACK histogram: {hist_by_back(train_idx, bk_counts)}")
    logging.info(f"Val BACK histogram:   {hist_by_back(val_idx, bk_counts)}")
    logging.info(f"Test BACK histogram:  {hist_by_back(test_idx, bk_counts)}")
    if args.split_kind == "target_random":
        logging.info(
            f"Probe BACK histogram: {hist_by_back(probe_idx, bk_counts)}"
        )

    if val_short:
        logging.warning(f"Validation underfilled BACK buckets: {val_short}")
    if test_short:
        logging.warning(f"Test underfilled BACK buckets: {test_short}")

    Path(split_path).parent.mkdir(parents=True, exist_ok=True)

    pm.safe_torch_save(
        {
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "probe_idx": probe_idx,
            "unassigned_idx": unassigned_idx,
            "split_type": split_type,
            "split_kind": args.split_kind,
            "split_seed": args.seed,
            "run_tag": args.run_tag,
            "dataset_path": args.dataset_path,
            "enc_cache_path": enc_cache_path,
            "safe_max_len": args.safe_max_len,
            "long_seq_policy": args.long_seq_policy,
            "train_frac_requested": args.train_frac,
            "train_frac_actual": train_frac_actual,
            "val_per_back": args.val_per_back,
            "test_per_back": args.test_per_back,
            "val_targets_requested": args.val_targets,
            "test_targets_requested": args.test_targets,
            "probe_examples_requested": args.probe_examples,
            "probe_examples_actual": len(probe_idx),
            "unassigned_count": len(unassigned_idx),
            "val_heldout_frac": args.val_heldout_frac,
            "val_family_frac": args.val_heldout_frac,  # backwards-compatible metadata key
            "target_mod_base": args.target_mod_base if args.split_kind == "target_mod" else None,
            "train_target_mods": sorted(train_mods) if train_mods is not None else None,
            "val_target_mods": sorted(val_mods) if val_mods is not None else None,
            "test_target_mods": sorted(test_mods) if test_mods is not None else None,
            "all_back_buckets": all_back_buckets,
            "train_back_hist": hist_by_back(train_idx, bk_counts),
            "val_back_hist": hist_by_back(val_idx, bk_counts),
            "test_back_hist": hist_by_back(test_idx, bk_counts),
            "probe_back_hist": hist_by_back(probe_idx, bk_counts),
            "train_target_count": len(train_targets),
            "val_target_count": len(val_targets),
            "test_target_count": len(test_targets),
            "probe_target_count": len(probe_targets),
            "train_targets": sorted(train_targets),
            "val_targets": sorted(val_targets),
            "test_targets": sorted(test_targets),
            "probe_targets": sorted(probe_targets),
            "val_target_pool": sorted(val_pool_targets),
            "test_target_pool": sorted(test_pool_targets),
            "probe_target_pool": sorted(probe_pool_targets),
            "train_family_count": len(train_families),
            "val_family_count": len(val_families),
            "test_family_count": len(test_families),
            "probe_family_count": len(probe_families),
            "train_val_family_overlap": train_val_family_overlap,
            "train_test_family_overlap": train_test_family_overlap,
            "val_test_family_overlap": val_test_family_overlap,
            "train_probe_family_overlap": train_probe_family_overlap,
            "val_probe_family_overlap": val_probe_family_overlap,
            "test_probe_family_overlap": test_probe_family_overlap,
        },
        split_path,
    )

    logging.info("")
    logging.info(
        f"Saved custom train/val/test/probe split to: {split_path}"
        if args.split_kind == "target_random"
        else f"Saved custom train/val/test split to: {split_path}"
    )

if __name__ == "__main__":
    main()