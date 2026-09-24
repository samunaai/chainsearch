#!/usr/bin/env python3
"""
label_generated_states_for_probing_samples.py

Parse every model-generated trace saved by
``evaluate_unused_astar_samples_for_probing.py``. Reconstruct and validate the
intermediate Countdown states, deduplicate the observed states, and use exact
forward-only subset dynamic programming to determine whether each valid state
can still reach its target without BACK.

The output is a structured PyTorch artifact containing generated-trace
metadata, recorded state positions, operation metadata, BACK transitions,
deduplicated states, and exact reachability labels. No model is loaded and no
GPU is used.
"""

from __future__ import annotations

# Clamp native-library threads before importing torch.
import os
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

# Select the same tokenizer contract used by the A* stage.
os.environ["COUNTDOWN_STAGE"] = "astar"

import argparse
import hashlib
import multiprocessing as mp
import random
import time
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch
from tqdm.auto import tqdm

import pretrain_and_sft_model as pm


# =========================
# DEFAULT CONFIG
# =========================

RUN_TAG = "exp34"
ORACLE_VERSION = "forward_subset_dp_v1"

ASTAR_DIR = "/workspace/user/chainsearch_archives/data_pts/astar"
PROBING_DIR = os.path.join(ASTAR_DIR, "probing")

EVALUATED_PATH = os.path.join(
    PROBING_DIR,
    f"{RUN_TAG}_astar_unused_samples_evaluated_for_probing.pt",
)

OUTPUT_PATH = os.path.join(
    PROBING_DIR,
    f"{RUN_TAG}_generated_state_solvability_for_probing_samples.pt",
)

PARTIAL_PATH = os.path.join(
    PROBING_DIR,
    f"{RUN_TAG}_generated_state_solvability_for_probing_samples.partial.pt",
)

DEFAULT_WORKERS = min(24, os.cpu_count() or 1)
SAVE_PARTIAL_EVERY_STATES = 10_000
MAX_INTEGER_DIGITS = 18
MAX_INTEGER_VALUE = (1 << 63) - 1
RESUME_PARTIAL = True
OVERWRITE = True

QUALITATIVE_EXAMPLE_SEED = 42
QUALITATIVE_EXAMPLES_PER_BACK_BIN = 2
MAX_QUALITATIVE_EVENTS_PER_TRACE = 20
BACK_BIN_LABELS = ("0", "1", "2", "3", "4", "5", "6", "7+")

STATE_EVENT_NAMES = [
    "initial",
    "after_operation",
    "pre_back",
    "post_back",
    "pre_success",
]
STATE_EVENT_TO_ID = {name: i for i, name in enumerate(STATE_EVENT_NAMES)}


# =========================
# COMPACT COLUMN STORAGE
# =========================


class Columns:
    """Append-only compact numeric columns backed by ``array.array``."""

    def __init__(self, schema: dict[str, str]):
        self.schema = dict(schema)
        self.data = {name: array(typecode) for name, typecode in schema.items()}

    def __len__(self) -> int:
        if not self.data:
            return 0
        first = next(iter(self.data.values()))
        return len(first)

    def append(self, **values: Any) -> None:
        missing = set(self.schema) - set(values)
        extra = set(values) - set(self.schema)
        if missing or extra:
            raise KeyError(
                f"Column append mismatch. missing={sorted(missing)} extra={sorted(extra)}"
            )
        for name, column in self.data.items():
            column.append(values[name])

    def set_slice(self, name: str, start: int, end: int, values: Iterable[Any]) -> None:
        replacement = array(self.schema[name], values)
        if len(replacement) != end - start:
            raise ValueError("Replacement length does not match slice length.")
        self.data[name][start:end] = replacement

    def to_tensors(self, dtypes: dict[str, torch.dtype]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, column in self.data.items():
            out[name] = torch.tensor(column, dtype=dtypes[name])
        return out


ROLLOUT_SCHEMA = {
    "cache_idx": "q",
    "target": "h",
    "prompt_length": "h",
    "generated_length": "h",
    "n_state_events": "h",
    "n_operations": "h",
    "n_valid_operations": "h",
    "n_backtracks": "h",
    "n_valid_backtracks": "h",
    "parse_error_count": "h",
}

STATE_EVENT_SCHEMA = {
    "record_pos": "i",
    "cache_idx": "q",
    "target": "h",
    "event_type": "B",
    "state_id": "i",
    "query_id": "i",
    "generated_token_idx": "h",
    "absolute_token_idx": "h",
    "branch_idx": "h",
    "operation_idx": "h",
    "prefix_is_valid": "B",
}

OPERATION_SCHEMA = {
    "record_pos": "i",
    "cache_idx": "q",
    "target": "h",
    "branch_idx": "h",
    "operation_idx": "h",
    "operation_idx_within_branch": "h",
    "operations_in_rollout": "h",
    "operations_in_branch": "h",
    "before_state_id": "i",
    "after_state_id": "i",
    "before_query_id": "i",
    "after_query_id": "i",
    "operation_start_generated_idx": "h",
    "operator_generated_idx": "h",
    "equals_generated_idx": "h",
    "result_end_generated_idx": "h",
    "transition_arrow_generated_idx": "h",
    "after_state_end_generated_idx": "h",
    "semicolon_generated_idx": "h",
    "result_token_progress": "f",
    "state_token_progress": "f",
    "prefix_valid_before": "B",
    "stated_before_matches_active": "B",
    "operands_available": "B",
    "arithmetic_valid": "B",
    "state_transition_consistent": "B",
    "operation_is_valid": "B",
    "claimed_result": "q",
    "claimed_result_is_target": "B",
    "valid_target_hit": "B",
}

BACKTRACK_SCHEMA = {
    "record_pos": "i",
    "cache_idx": "q",
    "target": "h",
    "backtrack_idx": "h",
    "branch_idx_before": "h",
    "branch_idx_after": "h",
    "pre_state_id": "i",
    "post_state_id": "i",
    "pre_query_id": "i",
    "post_query_id": "i",
    "pre_back_generated_idx": "h",
    "back_generated_idx": "h",
    "post_state_end_generated_idx": "h",
    "semicolon_generated_idx": "h",
    "trace_token_progress": "f",
    "pre_prefix_valid": "B",
    "post_prefix_valid": "B",
    "restored_state_is_active_ancestor": "B",
}

ROLLOUT_DTYPES = {
    "cache_idx": torch.int64,
    "target": torch.int16,
    "prompt_length": torch.int16,
    "generated_length": torch.int16,
    "n_state_events": torch.int16,
    "n_operations": torch.int16,
    "n_valid_operations": torch.int16,
    "n_backtracks": torch.int16,
    "n_valid_backtracks": torch.int16,
    "parse_error_count": torch.int16,
}

STATE_EVENT_DTYPES = {
    "record_pos": torch.int32,
    "cache_idx": torch.int64,
    "target": torch.int16,
    "event_type": torch.uint8,
    "state_id": torch.int32,
    "query_id": torch.int32,
    "generated_token_idx": torch.int16,
    "absolute_token_idx": torch.int16,
    "branch_idx": torch.int16,
    "operation_idx": torch.int16,
    "prefix_is_valid": torch.bool,
}

OPERATION_DTYPES = {
    "record_pos": torch.int32,
    "cache_idx": torch.int64,
    "target": torch.int16,
    "branch_idx": torch.int16,
    "operation_idx": torch.int16,
    "operation_idx_within_branch": torch.int16,
    "operations_in_rollout": torch.int16,
    "operations_in_branch": torch.int16,
    "before_state_id": torch.int32,
    "after_state_id": torch.int32,
    "before_query_id": torch.int32,
    "after_query_id": torch.int32,
    "operation_start_generated_idx": torch.int16,
    "operator_generated_idx": torch.int16,
    "equals_generated_idx": torch.int16,
    "result_end_generated_idx": torch.int16,
    "transition_arrow_generated_idx": torch.int16,
    "after_state_end_generated_idx": torch.int16,
    "semicolon_generated_idx": torch.int16,
    "result_token_progress": torch.float32,
    "state_token_progress": torch.float32,
    "prefix_valid_before": torch.bool,
    "stated_before_matches_active": torch.bool,
    "operands_available": torch.bool,
    "arithmetic_valid": torch.bool,
    "state_transition_consistent": torch.bool,
    "operation_is_valid": torch.bool,
    "claimed_result": torch.int64,
    "claimed_result_is_target": torch.bool,
    "valid_target_hit": torch.bool,
}

BACKTRACK_DTYPES = {
    "record_pos": torch.int32,
    "cache_idx": torch.int64,
    "target": torch.int16,
    "backtrack_idx": torch.int16,
    "branch_idx_before": torch.int16,
    "branch_idx_after": torch.int16,
    "pre_state_id": torch.int32,
    "post_state_id": torch.int32,
    "pre_query_id": torch.int32,
    "post_query_id": torch.int32,
    "pre_back_generated_idx": torch.int16,
    "back_generated_idx": torch.int16,
    "post_state_end_generated_idx": torch.int16,
    "semicolon_generated_idx": torch.int16,
    "trace_token_progress": torch.float32,
    "pre_prefix_valid": torch.bool,
    "post_prefix_valid": torch.bool,
    "restored_state_is_active_ancestor": torch.bool,
}


# =========================
# BASIC HELPERS
# =========================


def safe_torch_save(obj: Any, path: str) -> None:
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, final_path)


def generated_ids_at(payload: dict[str, Any], position: int) -> list[int]:
    start = int(payload["generated_offsets"][position].item())
    length = int(payload["generated_lengths"][position].item())
    return (
        payload["generated_flat_ids"][start:start + length]
        .to(torch.int64)
        .tolist()
    )


def canonical_state(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(int(v) for v in values))


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _state_tensor(states: Sequence[tuple[int, ...]]) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.full((len(states), 6), -1, dtype=torch.int64)
    lengths = torch.empty(len(states), dtype=torch.uint8)
    for i, state in enumerate(states):
        if len(state) > 6:
            raise RuntimeError(f"State has more than six values: {state}")
        lengths[i] = len(state)
        if state:
            values[i, :len(state)] = torch.tensor(state, dtype=torch.int64)
    return values, lengths


# =========================
# TOKEN-LEVEL TRACE PARSER
# =========================


class TokenContract:
    def __init__(self, tok: pm.Tokenizer):
        self.tok = tok
        self.LBR = tok.tok2id["["]
        self.RBR = tok.tok2id["]"]
        self.COMMA = tok.tok2id[","]
        self.EQ = tok.tok2id["="]
        self.SEMI = tok.tok2id[";"]
        self.ARROW = tok.tok2id["->"]
        self.BACK = tok.tok2id["BACK"]
        self.SUCCESS = tok.tok2id["SUCCESS"]
        self.EXPR = tok.tok2id["EXPR"]
        self.OP_IDS = {
            tok.tok2id["+"]: "+",
            tok.tok2id["-"]: "-",
            tok.tok2id["*"]: "*",
            tok.tok2id["/"]: "/",
        }
        self.DIGIT_BY_ID = {tok.tok2id[str(d)]: d for d in range(10)}


def parse_number(
    ids: Sequence[int],
    pos: int,
    contract: TokenContract,
) -> tuple[int, int, int] | None:
    """Return ``(value, next_pos, final_digit_pos)`` or ``None``."""
    digits: list[str] = []
    while pos < len(ids) and ids[pos] in contract.DIGIT_BY_ID:
        digits.append(str(contract.DIGIT_BY_ID[ids[pos]]))
        pos += 1
        if len(digits) > MAX_INTEGER_DIGITS:
            return None
    if not digits:
        return None
    value = int("".join(digits))
    if value > MAX_INTEGER_VALUE:
        return None
    return value, pos, pos - 1


def parse_state(
    ids: Sequence[int],
    pos: int,
    contract: TokenContract,
) -> tuple[tuple[int, ...], int, int] | None:
    """Return ``(canonical_state, next_pos, right_bracket_pos)`` or ``None``."""
    if pos >= len(ids) or ids[pos] != contract.LBR:
        return None
    pos += 1
    values: list[int] = []

    first = parse_number(ids, pos, contract)
    if first is None:
        return None
    value, pos, _ = first
    values.append(value)

    while pos < len(ids) and ids[pos] == contract.COMMA:
        pos += 1
        parsed = parse_number(ids, pos, contract)
        if parsed is None:
            return None
        value, pos, _ = parsed
        values.append(value)
        if len(values) > 6:
            return None

    if pos >= len(ids) or ids[pos] != contract.RBR:
        return None
    right_bracket_pos = pos
    return canonical_state(values), pos + 1, right_bracket_pos


def parse_operation_at(
    ids: Sequence[int],
    start: int,
    contract: TokenContract,
) -> dict[str, Any] | None:
    """Parse one complete arithmetic transition beginning at ``start``."""
    pos = start
    if pos >= len(ids) or ids[pos] != contract.LBR:
        return None

    parsed_before = parse_state(ids, pos, contract)
    if parsed_before is None:
        return None

    stated_before, pos, _ = parsed_before

    if pos >= len(ids) or ids[pos] != contract.ARROW:
        return None
    pos += 1

    n1_parsed = parse_number(ids, pos, contract)
    if n1_parsed is None:
        return None
    n1, pos, _ = n1_parsed

    if pos >= len(ids) or ids[pos] not in contract.OP_IDS:
        return None
    operator_pos = pos
    operator = contract.OP_IDS[ids[pos]]
    pos += 1

    n2_parsed = parse_number(ids, pos, contract)
    if n2_parsed is None:
        return None
    n2, pos, _ = n2_parsed

    if pos >= len(ids) or ids[pos] != contract.EQ:
        return None
    equals_pos = pos
    pos += 1

    result_parsed = parse_number(ids, pos, contract)
    if result_parsed is None:
        return None
    result, pos, result_end_pos = result_parsed

    if pos >= len(ids) or ids[pos] != contract.ARROW:
        return None
    transition_arrow_pos = pos
    pos += 1

    parsed_after = parse_state(ids, pos, contract)
    if parsed_after is None:
        return None
    stated_after, pos, after_state_end_pos = parsed_after

    if pos >= len(ids) or ids[pos] != contract.SEMI:
        return None
    semicolon_pos = pos

    return {
        "start": start,
        "next_pos": pos + 1,
        "stated_before": stated_before,
        "n1": n1,
        "operator": operator,
        "operator_pos": operator_pos,
        "n2": n2,
        "equals_pos": equals_pos,
        "result": result,
        "result_end_pos": result_end_pos,
        "transition_arrow_pos": transition_arrow_pos,
        "stated_after": stated_after,
        "after_state_end_pos": after_state_end_pos,
        "semicolon_pos": semicolon_pos,
    }


def parse_back_at(
    ids: Sequence[int],
    start: int,
    contract: TokenContract,
) -> dict[str, Any] | None:
    if start >= len(ids) or ids[start] != contract.BACK:
        return None
    pos = start + 1
    if pos >= len(ids) or ids[pos] != contract.ARROW:
        return None
    pos += 1
    parsed_state = parse_state(ids, pos, contract)
    if parsed_state is None:
        return None
    restored_state, pos, state_end_pos = parsed_state
    if pos >= len(ids) or ids[pos] != contract.SEMI:
        return None
    return {
        "next_pos": pos + 1,
        "restored_state": restored_state,
        "state_end_pos": state_end_pos,
        "semicolon_pos": pos,
    }

def parse_success_at(
    ids: Sequence[int],
    start: int,
    contract: TokenContract,
) -> dict[str, int] | None:
    """Parse the required ``SUCCESS ;`` termination event."""
    if start >= len(ids) or ids[start] != contract.SUCCESS:
        return None

    semicolon_pos = start + 1
    if semicolon_pos >= len(ids) or ids[semicolon_pos] != contract.SEMI:
        return None

    return {
        "next_pos": semicolon_pos + 1,
        "semicolon_pos": semicolon_pos,
    }



def arithmetic_result(n1: int, operator: str, n2: int) -> int | None:
    if operator == "+":
        return n1 + n2
    if operator == "-":
        return n1 - n2 if n1 > n2 else None
    if operator == "*":
        return n1 * n2 if n1 > 1 and n2 > 1 else None
    if operator == "/":
        return n1 // n2 if n2 > 0 and n1 % n2 == 0 else None
    raise ValueError(operator)


def operands_available(state: tuple[int, ...], n1: int, n2: int) -> bool:
    counts = Counter(state)
    if n1 == n2:
        return counts[n1] >= 2
    return counts[n1] >= 1 and counts[n2] >= 1


def apply_operation(
    state: tuple[int, ...],
    n1: int,
    n2: int,
    result: int,
) -> tuple[int, ...] | None:
    values = list(state)
    try:
        values.remove(n1)
        values.remove(n2)
    except ValueError:
        return None
    values.append(result)
    return canonical_state(values)


# =========================
# STATE AND QUERY CATALOGUES
# =========================


class Catalogues:
    def __init__(self):
        self.state_to_id: dict[tuple[int, ...], int] = {}
        self.states: list[tuple[int, ...]] = []
        self.queries_by_state: dict[int, dict[int, int]] = {}
        self.query_state_id = array("i")
        self.query_target = array("h")

    def intern_state(self, state: tuple[int, ...]) -> int:
        state = canonical_state(state)
        existing = self.state_to_id.get(state)
        if existing is not None:
            return existing
        state_id = len(self.states)
        self.state_to_id[state] = state_id
        self.states.append(state)
        return state_id

    def intern_query(self, state_id: int, target: int) -> int:
        target_map = self.queries_by_state.setdefault(state_id, {})
        existing = target_map.get(target)
        if existing is not None:
            return existing
        query_id = len(self.query_state_id)
        target_map[target] = query_id
        self.query_state_id.append(state_id)
        self.query_target.append(target)
        return query_id


# =========================
# GENERATED-TRACE PARSING
# =========================


def add_state_event(
    columns: Columns,
    catalogues: Catalogues,
    *,
    record_pos: int,
    cache_idx: int,
    target: int,
    event_type: str,
    state: tuple[int, ...],
    generated_token_idx: int,
    prompt_length: int,
    branch_idx: int,
    operation_idx: int,
    prefix_is_valid: bool,
) -> int:
    state_id = catalogues.intern_state(state)
    query_id = catalogues.intern_query(state_id, target) if prefix_is_valid else -1
    absolute_idx = prompt_length + generated_token_idx
    if generated_token_idx < 0:
        absolute_idx = prompt_length - 1
    columns.append(
        record_pos=record_pos,
        cache_idx=cache_idx,
        target=target,
        event_type=STATE_EVENT_TO_ID[event_type],
        state_id=state_id,
        query_id=query_id,
        generated_token_idx=generated_token_idx,
        absolute_token_idx=absolute_idx,
        branch_idx=branch_idx,
        operation_idx=operation_idx,
        prefix_is_valid=int(prefix_is_valid),
    )
    return query_id


def parse_one_rollout(
    *,
    record_pos: int,
    cache_idx: int,
    target: int,
    prompt_length: int,
    initial_state: tuple[int, ...],
    generated_ids: list[int],
    contract: TokenContract,
    catalogues: Catalogues,
    state_events: Columns,
    operations: Columns,
    backtracks: Columns,
) -> dict[str, int | bool]:
    state_event_start = len(state_events)
    operation_start_row = len(operations)
    backtrack_start_row = len(backtracks)

    active_state = canonical_state(initial_state)
    active_prefix_valid = True
    # Valid states on the currently active branch, from root to leaf.
    # BACK may restore only an ancestor on this path.
    valid_path = [active_state]

    branch_idx = 0
    operation_idx = 0
    operation_idx_within_branch = 0
    backtrack_idx = 0
    valid_operations = 0
    valid_backtracks = 0
    parse_errors = 0

    add_state_event(
        state_events,
        catalogues,
        record_pos=record_pos,
        cache_idx=cache_idx,
        target=target,
        event_type="initial",
        state=active_state,
        generated_token_idx=-1,
        prompt_length=prompt_length,
        branch_idx=branch_idx,
        operation_idx=-1,
        prefix_is_valid=True,
    )

    branch_operation_ranges: dict[int, list[int]] = defaultdict(list)

    pos = 0
    while pos < len(generated_ids):
        token_id = generated_ids[pos]

        if token_id == contract.EXPR:
            break

        if token_id == contract.BACK:
            parsed_back = parse_back_at(generated_ids, pos, contract)
            if parsed_back is None:
                parse_errors += 1
                active_prefix_valid = False
                pos += 1
                continue

            pre_state = active_state
            pre_valid = active_prefix_valid
            pre_state_id = catalogues.intern_state(pre_state)
            pre_query_id = (
                catalogues.intern_query(pre_state_id, target) if pre_valid else -1
            )

            pre_back_idx = pos - 1
            add_state_event(
                state_events,
                catalogues,
                record_pos=record_pos,
                cache_idx=cache_idx,
                target=target,
                event_type="pre_back",
                state=pre_state,
                generated_token_idx=pre_back_idx,
                prompt_length=prompt_length,
                branch_idx=branch_idx,
                operation_idx=operation_idx - 1,
                prefix_is_valid=pre_valid,
            )

            restored_state = parsed_back["restored_state"]

            # From a valid state, BACK must move to a proper ancestor.
            # From an invalid suffix, it may return to the last valid
            # state itself.
            allowable_path = valid_path[:-1] if pre_valid else valid_path
            restore_idx = -1
            for i in range(len(allowable_path) - 1, -1, -1):
                if allowable_path[i] == restored_state:
                    restore_idx = i
                    break

            restored_is_active_ancestor = restore_idx >= 0
            post_valid = restored_is_active_ancestor

            if post_valid:
                valid_path = valid_path[:restore_idx + 1]
            post_state_id = catalogues.intern_state(restored_state)
            post_query_id = (
                catalogues.intern_query(post_state_id, target) if post_valid else -1
            )

            old_branch_idx = branch_idx
            branch_idx += 1
            operation_idx_within_branch = 0
            if post_valid:
                valid_backtracks += 1

            active_state = restored_state
            active_prefix_valid = post_valid

            add_state_event(
                state_events,
                catalogues,
                record_pos=record_pos,
                cache_idx=cache_idx,
                target=target,
                event_type="post_back",
                state=restored_state,
                generated_token_idx=parsed_back["semicolon_pos"],
                prompt_length=prompt_length,
                branch_idx=branch_idx,
                operation_idx=operation_idx - 1,
                prefix_is_valid=post_valid,
            )

            backtracks.append(
                record_pos=record_pos,
                cache_idx=cache_idx,
                target=target,
                backtrack_idx=backtrack_idx,
                branch_idx_before=old_branch_idx,
                branch_idx_after=branch_idx,
                pre_state_id=pre_state_id,
                post_state_id=post_state_id,
                pre_query_id=pre_query_id,
                post_query_id=post_query_id,
                pre_back_generated_idx=pre_back_idx,
                back_generated_idx=pos,
                post_state_end_generated_idx=parsed_back["state_end_pos"],
                semicolon_generated_idx=parsed_back["semicolon_pos"],
                trace_token_progress=_ratio(
                    parsed_back["semicolon_pos"],
                    max(1, len(generated_ids) - 1),
                ),
                pre_prefix_valid=int(pre_valid),
                post_prefix_valid=int(post_valid),
                restored_state_is_active_ancestor=int(
                    restored_is_active_ancestor
                ),
            )

            backtrack_idx += 1
            pos = parsed_back["next_pos"]
            continue

        if token_id == contract.SUCCESS:
            parsed_success = parse_success_at(
                generated_ids,
                pos,
                contract,
            )
            if parsed_success is None:
                parse_errors += 1
                active_prefix_valid = False
                pos += 1
                continue

            add_state_event(
                state_events,
                catalogues,
                record_pos=record_pos,
                cache_idx=cache_idx,
                target=target,
                event_type="pre_success",
                state=active_state,
                generated_token_idx=pos - 1,
                prompt_length=prompt_length,
                branch_idx=branch_idx,
                operation_idx=operation_idx - 1,
                prefix_is_valid=active_prefix_valid,
            )

            pos = parsed_success["next_pos"]

            if (
                pos >= len(generated_ids)
                or generated_ids[pos] != contract.EXPR
            ):
                parse_errors += 1
                active_prefix_valid = False

            break

        parsed_operation = None
        if token_id == contract.LBR:
            parsed_operation = parse_operation_at(generated_ids, pos, contract)

        if parsed_operation is None:
            # Anything other than a complete operation, BACK, SUCCESS,
            # or EXPR is malformed at an event boundary. Continue scanning
            # so that a later legal BACK can still restore an ancestor.
            parse_errors += 1
            active_prefix_valid = False
            pos += 1
            continue

        stated_before = parsed_operation["stated_before"]
        prefix_valid_before = active_prefix_valid
        before_state = active_state
        before_state_id = catalogues.intern_state(before_state)
        before_query_id = (
            catalogues.intern_query(before_state_id, target)
            if prefix_valid_before
            else -1
        )

        before_matches = stated_before == before_state
        available = operands_available(
            before_state,
            parsed_operation["n1"],
            parsed_operation["n2"],
        )
        expected_result = arithmetic_result(
            parsed_operation["n1"],
            parsed_operation["operator"],
            parsed_operation["n2"],
        )
        arithmetic_valid = expected_result == parsed_operation["result"]
        expected_after = apply_operation(
            before_state,
            parsed_operation["n1"],
            parsed_operation["n2"],
            parsed_operation["result"],
        )
        state_consistent = (
            expected_after is not None
            and expected_after == parsed_operation["stated_after"]
        )
        operation_valid = (
            prefix_valid_before
            and before_matches
            and available
            and arithmetic_valid
            and state_consistent
        )

        after_state = parsed_operation["stated_after"]
        after_state_id = catalogues.intern_state(after_state)
        after_query_id = (
            catalogues.intern_query(after_state_id, target)
            if operation_valid
            else -1
        )

        if operation_valid:
            valid_operations += 1
            active_prefix_valid = True
            valid_path.append(after_state)
        else:
            active_prefix_valid = False
        active_state = after_state

        claimed_result_is_target = parsed_operation["result"] == target
        valid_target_hit = (
            operation_valid
            and claimed_result_is_target
            and target in after_state
        )

        row_idx = len(operations)
        operations.append(
            record_pos=record_pos,
            cache_idx=cache_idx,
            target=target,
            branch_idx=branch_idx,
            operation_idx=operation_idx,
            operation_idx_within_branch=operation_idx_within_branch,
            operations_in_rollout=0,
            operations_in_branch=0,
            before_state_id=before_state_id,
            after_state_id=after_state_id,
            before_query_id=before_query_id,
            after_query_id=after_query_id,
            operation_start_generated_idx=parsed_operation["start"],
            operator_generated_idx=parsed_operation["operator_pos"],
            equals_generated_idx=parsed_operation["equals_pos"],
            result_end_generated_idx=parsed_operation["result_end_pos"],
            transition_arrow_generated_idx=parsed_operation["transition_arrow_pos"],
            after_state_end_generated_idx=parsed_operation["after_state_end_pos"],
            semicolon_generated_idx=parsed_operation["semicolon_pos"],
            result_token_progress=_ratio(
                parsed_operation["result_end_pos"],
                max(1, len(generated_ids) - 1),
            ),
            state_token_progress=_ratio(
                parsed_operation["semicolon_pos"],
                max(1, len(generated_ids) - 1),
            ),
            prefix_valid_before=int(prefix_valid_before),
            stated_before_matches_active=int(before_matches),
            operands_available=int(available),
            arithmetic_valid=int(arithmetic_valid),
            state_transition_consistent=int(state_consistent),
            operation_is_valid=int(operation_valid),
            claimed_result=parsed_operation["result"],
            claimed_result_is_target=int(claimed_result_is_target),
            valid_target_hit=int(valid_target_hit),
        )
        branch_operation_ranges[branch_idx].append(row_idx)

        add_state_event(
            state_events,
            catalogues,
            record_pos=record_pos,
            cache_idx=cache_idx,
            target=target,
            event_type="after_operation",
            state=after_state,
            generated_token_idx=parsed_operation["semicolon_pos"],
            prompt_length=prompt_length,
            branch_idx=branch_idx,
            operation_idx=operation_idx,
            prefix_is_valid=operation_valid,
        )

        operation_idx += 1
        operation_idx_within_branch += 1
        pos = parsed_operation["next_pos"]

    operation_end_row = len(operations)
    n_operations = operation_end_row - operation_start_row
    if n_operations:
        operations.set_slice(
            "operations_in_rollout",
            operation_start_row,
            operation_end_row,
            [n_operations] * n_operations,
        )
        for rows in branch_operation_ranges.values():
            branch_count = len(rows)
            for row in rows:
                operations.data["operations_in_branch"][row] = branch_count

    return {
        "n_state_events": len(state_events) - state_event_start,
        "n_operations": n_operations,
        "n_valid_operations": valid_operations,
        "n_backtracks": len(backtracks) - backtrack_start_row,
        "n_valid_backtracks": valid_backtracks,
        "parse_error_count": parse_errors,
    }


# =========================
# EXACT FORWARD-ONLY ORACLE
# =========================


def _proper_partitions(mask: int) -> Iterator[tuple[int, int]]:
    """Yield each unordered nonempty partition of ``mask`` exactly once."""
    left = (mask - 1) & mask
    while left:
        right = mask ^ left
        if right and left < right:
            yield left, right
        left = (left - 1) & mask


def minimum_steps_for_targets(
    state: tuple[int, ...],
    targets: Sequence[int],
) -> dict[int, int]:
    """
    Return exact minimum forward operations needed for each target.

    ``-1`` means no legal forward continuation reaches that target. Reaching a
    target already present in the state costs zero operations. The search uses
    every legal positive-integer Countdown calculation over every subset of the
    current state. BACK is never part of the search graph.
    """
    unique_targets = set(int(t) for t in targets)
    results = {target: -1 for target in unique_targets}
    remaining = set(unique_targets)

    for value in state:
        if value in remaining:
            results[value] = 0
            remaining.remove(value)
    if not remaining:
        return results

    n = len(state)
    if n < 2:
        return results

    dp: list[set[int]] = [set() for _ in range(1 << n)]
    for i, value in enumerate(state):
        dp[1 << i].add(int(value))

    masks_by_size: list[list[int]] = [[] for _ in range(n + 1)]
    for mask in range(1, 1 << n):
        masks_by_size[mask.bit_count()].append(mask)

    for size in range(2, n + 1):
        steps = size - 1
        for mask in masks_by_size[size]:
            values: set[int] = set()
            for left, right in _proper_partitions(mask):
                left_values = dp[left]
                right_values = dp[right]
                if not left_values or not right_values:
                    continue
                for a in left_values:
                    for b in right_values:
                        values.add(a + b)
                        if a > 1 and b > 1:
                            values.add(a * b)
                        if a > b:
                            values.add(a - b)
                        elif b > a:
                            values.add(b - a)
                        if b > 0 and a % b == 0:
                            values.add(a // b)
                        if a > 0 and b % a == 0:
                            values.add(b // a)
            dp[mask] = values

            hits = remaining.intersection(values)
            if hits:
                for target in hits:
                    results[target] = steps
                remaining.difference_update(hits)
                if not remaining:
                    return results

    return results


def _oracle_worker_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def _solve_state_task(
    task: tuple[int, tuple[int, ...], list[tuple[int, int]]],
) -> tuple[int, list[tuple[int, int]]]:
    state_id, state, query_target_pairs = task
    targets = [target for _, target in query_target_pairs]
    solved = minimum_steps_for_targets(state, targets)
    return state_id, [
        (query_id, int(solved[target]))
        for query_id, target in query_target_pairs
    ]


def query_catalog_hash(
    states: Sequence[tuple[int, ...]],
    query_state_id: Sequence[int],
    query_target: Sequence[int],
) -> str:
    digest = hashlib.sha256()
    digest.update(ORACLE_VERSION.encode("utf-8"))
    for state in states:
        digest.update(len(state).to_bytes(1, "little", signed=False))
        for value in state:
            digest.update(int(value).to_bytes(8, "little", signed=True))
    for state_id, target in zip(query_state_id, query_target):
        digest.update(int(state_id).to_bytes(4, "little", signed=True))
        digest.update(int(target).to_bytes(2, "little", signed=True))
    return digest.hexdigest()


def save_partial_oracle(
    *,
    path: str,
    catalog_hash: str,
    query_state_id: array,
    query_target: array,
    query_min_steps: array,
    states_completed: int,
) -> None:
    safe_torch_save(
        {
            "format_version": 2,
            "oracle_version": ORACLE_VERSION,
            "catalog_hash": catalog_hash,
            "query_state_id": torch.tensor(query_state_id, dtype=torch.int32),
            "query_target": torch.tensor(query_target, dtype=torch.int16),
            "query_min_steps": torch.tensor(query_min_steps, dtype=torch.int16),
            "states_completed": int(states_completed),
        },
        path,
    )


# =========================
# DERIVED LABELS AND OUTPUT
# =========================


def gather_query_values(
    query_ids: torch.Tensor,
    query_min_steps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    known = query_ids >= 0
    min_steps = torch.full(query_ids.shape, -2, dtype=torch.int16)
    if bool(known.any()):
        min_steps[known] = query_min_steps[query_ids[known].to(torch.long)]
    solvable = known & (min_steps >= 0)
    return known, solvable, min_steps


def add_derived_labels(
    *,
    state_events: dict[str, torch.Tensor],
    operations: dict[str, torch.Tensor],
    backtracks: dict[str, torch.Tensor],
    state_values: torch.Tensor,
    state_lengths: torch.Tensor,
    query_min_steps: torch.Tensor,
) -> None:
    event_known, event_solvable, event_steps = gather_query_values(
        state_events["query_id"], query_min_steps
    )
    state_events["solvability_known"] = event_known
    state_events["is_forward_solvable"] = event_solvable
    state_events["min_steps_remaining"] = event_steps

    # Primary binary probing contract
    #
    # Positive   1-5 minimum forward operations remain
    # Negative   -1, no forward solution exists
    # Excluded    0, target already appears in the state
    # Excluded   -2, generated prefix was invalid
    state_events["binary_reachability_label_mask"] = (
        event_known
        & (
            (event_steps == -1)
            | (event_steps >= 1)
        )
    )
    state_events["binary_reachability_label"] = event_steps >= 1

    after_operation_id = STATE_EVENT_TO_ID["after_operation"]
    state_events["primary_probe_candidate"] = (
        state_events["event_type"] == after_operation_id
    ) & state_events["binary_reachability_label_mask"]
    event_state_ids = state_events["state_id"].to(torch.long)
    event_targets = state_events["target"].to(torch.int64)
    contains_target = torch.zeros(len(event_state_ids), dtype=torch.bool)
    for width in range(6):
        present = state_lengths[event_state_ids] > width
        contains_target |= present & (
            state_values[event_state_ids, width] == event_targets
        )
    state_events["state_contains_target"] = contains_target

    before_known, before_solvable, before_steps = gather_query_values(
        operations["before_query_id"], query_min_steps
    )
    after_known, after_solvable, after_steps = gather_query_values(
        operations["after_query_id"], query_min_steps
    )
    operations["before_solvability_known"] = before_known
    operations["before_is_forward_solvable"] = before_solvable
    operations["before_min_steps_remaining"] = before_steps
    operations["after_solvability_known"] = after_known
    operations["after_is_forward_solvable"] = after_solvable
    operations["after_min_steps_remaining"] = after_steps


    operations["before_binary_reachability_label_mask"] = (
        before_known
        & (
            (before_steps == -1)
            | (before_steps >= 1)
        )
    )
    operations["before_binary_reachability_label"] = before_steps >= 1

    operations["after_binary_reachability_label_mask"] = (
        after_known
        & (
            (after_steps == -1)
            | (after_steps >= 1)
        )
    )
    operations["after_binary_reachability_label"] = after_steps >= 1

    valid_operation = operations["operation_is_valid"]

    # The trace was valid before this operation, but this operation itself
    # introduced an arithmetic, operand, before-state, or after-state error.
    operations["latest_operation_introduced_invalidity"] = (
        operations["prefix_valid_before"]
        & ~valid_operation
    )

    # An earlier error had already invalidated the active branch.
    operations["operation_follows_invalid_prefix"] = (
        ~operations["prefix_valid_before"]
    )

    if bool(
        (
            operations["latest_operation_introduced_invalidity"]
            & operations["operation_follows_invalid_prefix"]
        ).any()
    ):
        raise RuntimeError(
            "Invalid-operation categories unexpectedly overlap."
        )

    if not torch.equal(
        (
            operations["latest_operation_introduced_invalidity"]
            | operations["operation_follows_invalid_prefix"]
        ),
        ~valid_operation,
    ):
        raise RuntimeError(
            "Invalid-operation categories do not cover every invalid operation."
        )

    # There is exactly one after-operation state position per parsed operation.
    after_operation_rows = torch.nonzero(
        state_events["event_type"]
        == STATE_EVENT_TO_ID["after_operation"],
        as_tuple=False,
    ).flatten()

    if int(after_operation_rows.numel()) != int(valid_operation.numel()):
        raise RuntimeError(
            "After-operation state-position count does not match operation count."
        )

    if not torch.equal(
        state_events["record_pos"][after_operation_rows],
        operations["record_pos"],
    ):
        raise RuntimeError(
            "After-operation positions do not align with operation trace rows."
        )

    if not torch.equal(
        state_events["operation_idx"][after_operation_rows],
        operations["operation_idx"],
    ):
        raise RuntimeError(
            "After-operation positions do not align with operation indices."
        )

    if not torch.equal(
        state_events["cache_idx"][after_operation_rows],
        operations["cache_idx"],
    ):
        raise RuntimeError(
            "After-operation positions do not align with operation cache rows."
        )


    state_events["latest_operation_introduced_invalidity"] = (
        torch.zeros_like(state_events["prefix_is_valid"])
    )
    state_events["operation_follows_invalid_prefix"] = (
        torch.zeros_like(state_events["prefix_is_valid"])
    )

    state_events[
        "latest_operation_introduced_invalidity"
    ][after_operation_rows] = operations[
        "latest_operation_introduced_invalidity"
    ]

    state_events[
        "operation_follows_invalid_prefix"
    ][after_operation_rows] = operations[
        "operation_follows_invalid_prefix"
    ]

    impossible_operation = (
        valid_operation
        & before_known
        & after_known
        & ~before_solvable
        & after_solvable
    )

    if bool(impossible_operation.any()):
        raise RuntimeError(
            "Oracle inconsistency. A valid operation moved from an "
            "unreachable state to a reachable state."
        )

    pre_known, pre_solvable, pre_steps = gather_query_values(
        backtracks["pre_query_id"], query_min_steps
    )
    post_known, post_solvable, post_steps = gather_query_values(
        backtracks["post_query_id"], query_min_steps
    )
    backtracks["pre_solvability_known"] = pre_known
    backtracks["pre_is_forward_solvable"] = pre_solvable
    backtracks["pre_min_steps_remaining"] = pre_steps
    backtracks["post_solvability_known"] = post_known
    backtracks["post_is_forward_solvable"] = post_solvable
    backtracks["post_min_steps_remaining"] = post_steps

    backtracks["pre_binary_reachability_label_mask"] = (
        pre_known
        & (
            (pre_steps == -1)
            | (pre_steps >= 1)
        )
    )
    backtracks["pre_binary_reachability_label"] = pre_steps >= 1

    backtracks["post_binary_reachability_label_mask"] = (
        post_known
        & (
            (post_steps == -1)
            | (post_steps >= 1)
        )
    )
    backtracks["post_binary_reachability_label"] = post_steps >= 1

    impossible_backtrack = (
        pre_known
        & post_known
        & pre_solvable
        & ~post_solvable
    )
    if bool(impossible_backtrack.any()):
        raise RuntimeError(
            "Oracle inconsistency. A valid BACK moved from a reachable "
            "state to an unreachable ancestor."
        )


def build_summary(
    *,
    rollouts: dict[str, torch.Tensor],
    state_events: dict[str, torch.Tensor],
    operations: dict[str, torch.Tensor],
    backtracks: dict[str, torch.Tensor],
    state_lengths: torch.Tensor,
    n_queries: int) -> dict[str, Any]:

    state_size_distribution = Counter(
        int(x) for x in state_lengths.tolist()
    )

    position_type = state_events["event_type"]
    after_operation = (
        position_type == STATE_EVENT_TO_ID["after_operation"]
    )
    known = state_events["solvability_known"]
    steps = state_events["min_steps_remaining"]

    primary_candidate = state_events["primary_probe_candidate"]
    binary_label = state_events["binary_reachability_label"]

    latest_operation_invalid = state_events[
        "latest_operation_introduced_invalidity"
    ]
    prior_prefix_invalid = state_events[
        "operation_follows_invalid_prefix"
    ]

    minimum_step_distribution = {
        step: int(
            (
                after_operation
                & known
                & (steps == step)
            ).sum().item()
        )
        for step in range(-1, 6)
    }

    pre_known = backtracks["pre_solvability_known"]
    post_known = backtracks["post_solvability_known"]
    pre_solvable = backtracks["pre_is_forward_solvable"]
    post_solvable = backtracks["post_is_forward_solvable"]
    paired_known = pre_known & post_known

    return {
        "generated_traces": int(rollouts["cache_idx"].numel()),
        "recorded_state_positions": int(
            state_events["state_id"].numel()
        ),
        "unique_states": int(state_lengths.numel()),
        "unique_state_target_queries": int(n_queries),
        "state_size_distribution": dict(
            sorted(state_size_distribution.items())
        ),

        "reachable_binary_labels": int(
            (primary_candidate & binary_label).sum().item()
        ),
        "unreachable_binary_labels": int(
            (primary_candidate & ~binary_label).sum().item()
        ),
        "target_present_excluded": int(
            (
                after_operation
                & known
                & (steps == 0)
            ).sum().item()
        ),
        "latest_operation_invalid_excluded": int(
            (
                after_operation
                & latest_operation_invalid
            ).sum().item()
        ),
        "prior_prefix_invalid_excluded": int(
            (
                after_operation
                & prior_prefix_invalid
            ).sum().item()
        ),
        "minimum_step_distribution": minimum_step_distribution,

        "back_known_pairs": int(paired_known.sum().item()),
        "back_excluded_pairs": int((~paired_known).sum().item()),
        "back_solvable_to_solvable": int(
            (
                paired_known
                & pre_solvable
                & post_solvable
            ).sum().item()
        ),
        "back_solvable_to_unsolvable": int(
            (
                paired_known
                & pre_solvable
                & ~post_solvable
            ).sum().item()
        ),
        "back_unsolvable_to_solvable": int(
            (
                paired_known
                & ~pre_solvable
                & post_solvable
            ).sum().item()
        ),
        "back_unsolvable_to_unsolvable": int(
            (
                paired_known
                & ~pre_solvable
                & ~post_solvable
            ).sum().item()
        ),

        "parsed_operations": int(
            operations["operation_idx"].numel()
        ),
        "valid_operations": int(
            operations["operation_is_valid"].sum().item()
        ),
        "traces_with_syntax_errors": int(
            (rollouts["parse_error_count"] > 0).sum().item()
        ),
    }


def print_summary(summary: dict[str, Any], output_path: str) -> None:
    width = 92
    print()
    print("=" * width)
    print("GENERATED-STATE REACHABILITY LABEL SUMMARY".center(width))
    print("=" * width)

    print("DATA SCALE")
    print(
        f"  Generated traces parsed:               "
        f"{summary['generated_traces']:>12,}"
    )
    print(
        f"  Recorded state positions:              "
        f"{summary['recorded_state_positions']:>12,}"
    )
    print(
        f"  Unique parsed states:                  "
        f"{summary['unique_states']:>12,}"
    )
    print(
        f"  Unique state-target queries:           "
        f"{summary['unique_state_target_queries']:>12,}"
    )
    print(
        f"  Parsed-state size distribution:        "
        f"{summary['state_size_distribution']}"
    )
    print()

    print("BINARY LABELS AT AFTER-OPERATION POSITIONS")
    print(
        f"  Reachable without BACK, at least 1 step:"
        f"{summary['reachable_binary_labels']:>12,}"
    )
    print(
        f"  Unreachable without BACK, -1:          "
        f"{summary['unreachable_binary_labels']:>12,}"
    )
    print(
        f"  Target already present, excluded:      "
        f"{summary['target_present_excluded']:>12,}"
    )
    print(
        f"  Latest operation invalid, excluded:    "
        f"{summary['latest_operation_invalid_excluded']:>12,}"
    )
    print(
        f"  Earlier prefix already invalid, excluded:"
        f"{summary['prior_prefix_invalid_excluded']:>11,}"
    )
    print(
        f"  Minimum-step distribution:             "
        f"{summary['minimum_step_distribution']}"
    )
    print()

    print("BACK SOLVABILITY CHANGES")
    print(
        f"  Transitions with known pre/post labels: "
        f"{summary['back_known_pairs']:>12,}"
    )
    print(
        f"    solvable   -> solvable:               "
        f"{summary['back_solvable_to_solvable']:>12,}"
    )
    print(
        f"    solvable   -> unsolvable:             "
        f"{summary['back_solvable_to_unsolvable']:>12,}"
    )
    print(
        f"    unsolvable -> solvable:               "
        f"{summary['back_unsolvable_to_solvable']:>12,}"
    )
    print(
        f"    unsolvable -> unsolvable:             "
        f"{summary['back_unsolvable_to_unsolvable']:>12,}"
    )
    print(
        f"  Excluded from transition analysis:     "
        f"{summary['back_excluded_pairs']:>12,}"
    )
    print()

    print("TRACE QUALITY")
    print(
        f"  Operations parsed / legally valid:     "
        f"{summary['parsed_operations']:,} / "
        f"{summary['valid_operations']:,}"
    )
    print(
        f"  Generated traces with syntax errors:   "
        f"{summary['traces_with_syntax_errors']:>12,}"
    )
    print()

    print("RUNTIME")
    print(
        f"  Parsing:                                "
        f"{summary['parse_minutes']:>11.2f} min"
    )
    print(
        f"  Exact reachability oracle:              "
        f"{summary['oracle_minutes']:>11.2f} min"
    )
    print()
    print(f"Saved label artifact: {output_path}")
    print(
        'Binary label: state_occurrences["binary_reachability_label"]'
    )
    print(
        'Eligibility mask: state_occurrences["primary_probe_candidate"]'
    )
    print(
        'Activation token index: state_occurrences["absolute_token_idx"]'
    )
    print("=" * width)

# =========================
# SELF-TESTS
# =========================

def back_bin_label(n_backtracks: int) -> str:
    return str(n_backtracks) if n_backtracks <= 6 else "7+"


def select_qualitative_examples(
    n_backtracks: torch.Tensor,
) -> list[tuple[str, int]]:
    """Select two deterministic random traces from each BACK-count bin."""
    rng = random.Random(QUALITATIVE_EXAMPLE_SEED)
    candidates_by_bin = {
        label: []
        for label in BACK_BIN_LABELS
    }

    for record_pos, count in enumerate(n_backtracks.tolist()):
        label = back_bin_label(int(count))
        candidates_by_bin[label].append(record_pos)

    selected: list[tuple[str, int]] = []
    for label in BACK_BIN_LABELS:
        candidates = candidates_by_bin[label]
        rng.shuffle(candidates)
        selected.extend(
            (label, record_pos)
            for record_pos in candidates[
                :QUALITATIVE_EXAMPLES_PER_BACK_BIN
            ]
        )

    return selected


def decode_generated_span(
    tok: pm.Tokenizer,
    generated_ids: list[int],
    start: int,
    end: int,
) -> str:
    if start < 0 or end < start or end >= len(generated_ids):
        return "<invalid token span>"
    return tok.decode(generated_ids[start:end + 1]).strip()


def format_reachability(
    known: bool,
    min_steps: int,
) -> str:
    if not known:
        return "reachability unknown"
    if min_steps == -1:
        return "forward-unreachable"
    if min_steps == 0:
        return "target already present"
    return f"forward-reachable in {min_steps} step(s)"


def operation_label_description(
    operations: dict[str, torch.Tensor],
    row: int,
) -> str:
    operation_valid = bool(
        operations["operation_is_valid"][row].item()
    )

    if not operation_valid:
        reasons: list[str] = []

        if not bool(
            operations["prefix_valid_before"][row].item()
        ):
            reasons.append("earlier prefix already invalid")
        else:
            checks = (
                (
                    "stated_before_matches_active",
                    "before-state mismatch",
                ),
                (
                    "operands_available",
                    "unavailable operand",
                ),
                (
                    "arithmetic_valid",
                    "arithmetic mismatch",
                ),
                (
                    "state_transition_consistent",
                    "after-state mismatch",
                ),
            )
            for field, description in checks:
                if not bool(operations[field][row].item()):
                    reasons.append(description)

        if not reasons:
            reasons.append("unspecified validity failure")

        return (
            "INVALID transition, excluded from reachability probe "
            f"({', '.join(reasons)})"
        )

    min_steps = int(
        operations["after_min_steps_remaining"][row].item()
    )

    if min_steps == -1:
        return "forward-unreachable, binary label 0"
    if min_steps == 0:
        return "target already present, excluded from binary probe"

    return (
        f"forward-reachable in {min_steps} step(s), "
        "binary label 1"
    )


def print_qualitative_examples(
    *,
    selected_examples: list[tuple[str, int]],
    evaluated: dict[str, Any],
    tok: pm.Tokenizer,
    generated_traces: dict[str, torch.Tensor],
    operations: dict[str, torch.Tensor],
    back_transitions: dict[str, torch.Tensor],
) -> None:
    width = 92

    print()
    print("=" * width)
    print("QUALITATIVE REACHABILITY-LABEL EXAMPLES".center(width))
    print("=" * width)
    print(
        f"Selection seed {QUALITATIVE_EXAMPLE_SEED}, "
        f"up to {QUALITATIVE_EXAMPLES_PER_BACK_BIN} traces per "
        "parsed-BACK-count bin 0, 1, 2, 3, 4, 5, 6, and 7+."
    )
    print(
        f"Displaying at most {MAX_QUALITATIVE_EVENTS_PER_TRACE} "
        "operations and BACK transitions per trace."
    )

    for example_idx, (bin_label, record_pos) in enumerate(
        selected_examples,
        start=1,
    ):
        generated_ids = generated_ids_at(evaluated, record_pos)

        operation_rows = torch.nonzero(
            operations["record_pos"] == record_pos,
            as_tuple=False,
        ).flatten().tolist()

        back_rows = torch.nonzero(
            back_transitions["record_pos"] == record_pos,
            as_tuple=False,
        ).flatten().tolist()

        timeline: list[tuple[int, str, int] | None] = []

        for row in operation_rows:
            timeline.append(
                (
                    int(
                        operations[
                            "operation_start_generated_idx"
                        ][row].item()
                    ),
                    "operation",
                    row,
                )
            )

        for row in back_rows:
            timeline.append(
                (
                    int(
                        back_transitions[
                            "back_generated_idx"
                        ][row].item()
                    ),
                    "back",
                    row,
                )
            )

        timeline.sort(
            key=lambda item: item[0] if item is not None else -1
        )

        if len(timeline) > MAX_QUALITATIVE_EVENTS_PER_TRACE:
            first_count = MAX_QUALITATIVE_EVENTS_PER_TRACE // 2
            last_count = (
                MAX_QUALITATIVE_EVENTS_PER_TRACE
                - first_count
            )
            timeline = (
                timeline[:first_count]
                + [None]
                + timeline[-last_count:]
            )

        input_numbers = [
            int(x)
            for x in generated_traces[
                "input_numbers"
            ][record_pos].tolist()
        ]
        target = int(
            generated_traces["target"][record_pos].item()
        )
        back_count = int(
            generated_traces["n_backtracks"][record_pos].item()
        )
        syntax_errors = int(
            generated_traces[
                "parse_error_count"
            ][record_pos].item()
        )

        print()
        print("-" * width)
        print(
            f"EXAMPLE {example_idx} OF {len(selected_examples)} "
            f"| BACK bin {bin_label}"
        )
        print(f"Generated-trace row: {record_pos}")
        print(f"Input numbers:       {input_numbers}")
        print(f"Target:              {target}")
        print(f"Parsed BACK transitions: {back_count}")
        print(f"Syntax errors:       {syntax_errors}")

        if not timeline:
            print("  No complete operation or BACK transition was parsed.")
            continue

        for item in timeline:
            if item is None:
                print("  ... middle events omitted ...")
                continue

            _, event_kind, row = item

            if event_kind == "operation":
                start = int(
                    operations[
                        "operation_start_generated_idx"
                    ][row].item()
                )
                end = int(
                    operations[
                        "semicolon_generated_idx"
                    ][row].item()
                )
                text = decode_generated_span(
                    tok,
                    generated_ids,
                    start,
                    end,
                )
                operation_idx = int(
                    operations["operation_idx"][row].item()
                )

                print(f"  Operation {operation_idx}")
                print(f"    {text}")
                print(
                    f"    {operation_label_description(operations, row)}"
                )
                continue

            start = int(
                back_transitions[
                    "back_generated_idx"
                ][row].item()
            )
            end = int(
                back_transitions[
                    "semicolon_generated_idx"
                ][row].item()
            )
            text = decode_generated_span(
                tok,
                generated_ids,
                start,
                end,
            )

            pre_known = bool(
                back_transitions[
                    "pre_solvability_known"
                ][row].item()
            )
            post_known = bool(
                back_transitions[
                    "post_solvability_known"
                ][row].item()
            )
            pre_steps = int(
                back_transitions[
                    "pre_min_steps_remaining"
                ][row].item()
            )
            post_steps = int(
                back_transitions[
                    "post_min_steps_remaining"
                ][row].item()
            )

            print(
                f"  BACK {int(back_transitions['backtrack_idx'][row].item())}"
            )
            print(f"    {text}")
            print(
                "    "
                f"{format_reachability(pre_known, pre_steps)}"
                " -> "
                f"{format_reachability(post_known, post_steps)}"
            )

    print()
    print("=" * width)

def run_self_tests() -> None:
    # This target genuinely requires all six numbers and therefore five
    # forward operations under the current Countdown rules.
    assert minimum_steps_for_targets(
        (1, 2, 3, 4, 5, 6),
        [158],
    )[158] == 5

    assert minimum_steps_for_targets((4, 4, 12, 100), [102])[102] == 3
    assert minimum_steps_for_targets((4, 4, 112), [102])[102] == -1
    assert minimum_steps_for_targets((2, 100), [102])[102] == 1
    assert minimum_steps_for_targets((102, 999), [102])[102] == 0

    tok = pm.Tokenizer()
    contract = TokenContract(tok)

    def encode_without_eos(text: str) -> list[int]:
        ids = pm.encode_ids(tok, text)
        return ids[:-1]

    trace = (
        "[ 2, 3, 4, 4, 9, 50 ] -> 2 * 50 = 100 -> "
        "[ 3, 4, 4, 9, 100 ] ; "
        "[ 3, 4, 4, 9, 100 ] -> 3 + 9 = 12 -> "
        "[ 4, 4, 12, 100 ] ; "
        "BACK -> [ 3, 4, 4, 9, 100 ] ; SUCCESS ; EXPR (100)"
    )
    ids = encode_without_eos(trace)
    op0 = parse_operation_at(ids, 0, contract)
    assert op0 is not None and op0["result"] == 100
    op1 = parse_operation_at(ids, op0["next_pos"], contract)
    assert op1 is not None and op1["stated_after"] == (4, 4, 12, 100)
    back = parse_back_at(ids, op1["next_pos"], contract)
    assert back is not None and back["restored_state"] == (3, 4, 4, 9, 100)

    print("Self-tests passed.")


# =========================
# MAIN
# =========================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Label every valid model-generated Countdown state with exact "
            "forward-only target solvability."
        )
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of CPU worker processes (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--limit-traces",
        "--limit-rollouts",
        dest="limit_traces",
        type=int,
        default=None,
        help="Parse only the first N generated traces for testing.",
    )
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Parse and print inventory counts without running the DP oracle.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore and remove an existing partial oracle artifact.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run internal DP and parser tests, then exit.",
    )
    args = parser.parse_args()

    if args.self_test:
        run_self_tests()
        return

    width = 92
    print("=" * width)
    print("LABEL GENERATED STATES FOR PROBING SAMPLES".center(width))
    print("=" * width)
    print(f"Evaluated generated traces: {EVALUATED_PATH}")
    print(f"Label artifact:             {OUTPUT_PATH}")
    print(f"Partial oracle:             {PARTIAL_PATH}")
    print(f"CPU workers:                {args.workers}")
    print("Oracle rule:                legal forward arithmetic only, with no BACK")
    print("=" * width)
    print()

    if not os.path.isfile(EVALUATED_PATH):
        raise FileNotFoundError(EVALUATED_PATH)
    if os.path.exists(OUTPUT_PATH) and not OVERWRITE:
        raise FileExistsError(OUTPUT_PATH)

    os.makedirs(PROBING_DIR, exist_ok=True)

    print("Loading evaluated generated traces.", flush=True)
    evaluated = torch.load(
        EVALUATED_PATH,
        map_location="cpu",
        weights_only=False,
    )
    if not bool(evaluated.get("complete", False)):
        raise RuntimeError("Evaluated-generation artifact is not marked complete.")
    if evaluated.get("index_space") != "astar_tokenisation_cache_rows":
        raise RuntimeError(
            f"Unexpected evaluated index space: {evaluated.get('index_space')!r}"
        )

    required = {
        "cache_idx",
        "target",
        "input_numbers",
        "prompt_length",
        "generated_flat_ids",
        "generated_offsets",
        "generated_lengths",
    }
    missing = required - set(evaluated)
    if missing:
        raise KeyError(f"Evaluated artifact is missing keys: {sorted(missing)}")

    full_n_records = int(evaluated["cache_idx"].numel())

    for name in (
        "target",
        "prompt_length",
        "generated_offsets",
        "generated_lengths",
    ):
        actual = int(evaluated[name].numel())
        if actual != full_n_records:
            raise RuntimeError(
                f"Evaluated artifact field {name!r} has "
                f"{actual:,} rows, expected {full_n_records:,}."
            )

    if tuple(evaluated["input_numbers"].shape) != (
        full_n_records,
        6,
    ):
        raise RuntimeError(
            "Evaluated artifact input_numbers has shape "
            f"{tuple(evaluated['input_numbers'].shape)}, expected "
            f"{(full_n_records, 6)}."
        )

    generated_offsets = evaluated["generated_offsets"].to(torch.int64)
    generated_lengths = evaluated["generated_lengths"].to(torch.int64)
    flat_size = int(evaluated["generated_flat_ids"].numel())

    if bool((generated_offsets < 0).any()):
        raise RuntimeError(
            "Evaluated artifact contains negative generated offsets."
        )
    if bool((generated_lengths < 0).any()):
        raise RuntimeError(
            "Evaluated artifact contains negative generated lengths."
        )
    if bool(
        (
            generated_offsets
            + generated_lengths
            > flat_size
        ).any()
    ):
        raise RuntimeError(
            "Evaluated artifact contains a generated sequence outside "
            "generated_flat_ids."
        )

    n_records = full_n_records
    if args.limit_traces is not None:
        n_records = min(n_records, max(0, int(args.limit_traces)))
    if n_records <= 0:
        raise RuntimeError("No generated traces selected.")

    tok = pm.Tokenizer()
    contract = TokenContract(tok)
    catalogues = Catalogues()
    rollouts = Columns(ROLLOUT_SCHEMA)
    state_events = Columns(STATE_EVENT_SCHEMA)
    operations = Columns(OPERATION_SCHEMA)
    backtracks = Columns(BACKTRACK_SCHEMA)

    parse_start = time.time()
    for record_pos in tqdm(
        range(n_records),
        desc="Parsing generated states",
        dynamic_ncols=True,
    ):
        cache_idx = int(evaluated["cache_idx"][record_pos].item())
        target = int(evaluated["target"][record_pos].item())
        prompt_length = int(evaluated["prompt_length"][record_pos].item())
        initial_state = canonical_state(
            int(x) for x in evaluated["input_numbers"][record_pos].tolist()
        )
        generated_ids = generated_ids_at(evaluated, record_pos)

        parsed = parse_one_rollout(
            record_pos=record_pos,
            cache_idx=cache_idx,
            target=target,
            prompt_length=prompt_length,
            initial_state=initial_state,
            generated_ids=generated_ids,
            contract=contract,
            catalogues=catalogues,
            state_events=state_events,
            operations=operations,
            backtracks=backtracks,
        )

        rollouts.append(
            cache_idx=cache_idx,
            target=target,
            prompt_length=prompt_length,
            generated_length=len(generated_ids),
            n_state_events=parsed["n_state_events"],
            n_operations=parsed["n_operations"],
            n_valid_operations=parsed["n_valid_operations"],
            n_backtracks=parsed["n_backtracks"],
            n_valid_backtracks=parsed["n_valid_backtracks"],
            parse_error_count=parsed["parse_error_count"],
        )

    parse_seconds = time.time() - parse_start
    print()
    print("PARSE INVENTORY")
    print(f"  Generated traces:            {len(rollouts):>12,}")
    print(f"  Recorded state positions:    {len(state_events):>12,}")
    print(f"  Operations:                  {len(operations):>12,}")
    print(f"  BACK transitions:            {len(backtracks):>12,}")
    print(f"  Unique states:               {len(catalogues.states):>12,}")
    print(f"  Unique state-target queries: {len(catalogues.query_state_id):>12,}")
    print(f"  Parsing time:                {parse_seconds / 60:>11.2f} min")

    if args.inventory_only:
        print("Inventory-only mode finished. No artifact was saved.")
        return

    state_values, state_lengths = _state_tensor(catalogues.states)
    catalog_hash = query_catalog_hash(
        catalogues.states,
        catalogues.query_state_id,
        catalogues.query_target,
    )

    n_queries = len(catalogues.query_state_id)
    query_min_steps = array("h", [-2]) * n_queries
    resume = RESUME_PARTIAL and not args.no_resume

    if os.path.exists(PARTIAL_PATH):
        if resume:
            print(f"Loading partial oracle: {PARTIAL_PATH}", flush=True)
            partial = torch.load(
                PARTIAL_PATH,
                map_location="cpu",
                weights_only=False,
            )
            if partial.get("oracle_version") != ORACLE_VERSION:
                raise RuntimeError(
                    "Partial oracle uses a different oracle definition. "
                    "Run again with --no-resume to discard it."
                )
            if partial.get("catalog_hash") != catalog_hash:
                raise RuntimeError(
                    "Partial oracle was created from a different state/query catalogue. "
                    "Run again with --no-resume to discard it."
                )
            loaded = partial["query_min_steps"].to(torch.int16).tolist()
            if len(loaded) != n_queries:
                raise RuntimeError("Partial oracle query count mismatch.")
            query_min_steps = array("h", loaded)
        else:
            os.remove(PARTIAL_PATH)

    unresolved_state_ids = [
        state_id
        for state_id, target_map in catalogues.queries_by_state.items()
        if any(query_min_steps[query_id] == -2 for query_id in target_map.values())
    ]

    unresolved_state_ids.sort(
        key=lambda state_id: len(catalogues.states[state_id]),
        reverse=True,
    )

    def iter_tasks() -> Iterator[
        tuple[int, tuple[int, ...], list[tuple[int, int]]]
    ]:
        for state_id in unresolved_state_ids:
            target_map = catalogues.queries_by_state[state_id]
            unresolved = [
                (query_id, target)
                for target, query_id in target_map.items()
                if query_min_steps[query_id] == -2
            ]
            if unresolved:
                yield state_id, catalogues.states[state_id], unresolved

    print()
    print(
        f"Solving {len(unresolved_state_ids):,} unique states on "
        f"{args.workers} CPU workers."
    )
    solve_start = time.time()
    states_completed = 0

    if args.workers <= 1:
        result_iter: Iterable[tuple[int, list[tuple[int, int]]]] = (
            _solve_state_task(task) for task in iter_tasks()
        )
        pool_context = None
    else:
        ctx = mp.get_context("spawn")
        pool_context = ctx.Pool(
            processes=max(1, int(args.workers)),
            initializer=_oracle_worker_init,
        )
        result_iter = pool_context.imap_unordered(
            _solve_state_task,
            iter_tasks(),
            chunksize=2,
        )

    try:
        for _state_id, results in tqdm(
            result_iter,
            total=len(unresolved_state_ids),
            desc="Exact solvability DP",
            dynamic_ncols=True,
        ):
            for query_id, min_steps in results:
                query_min_steps[query_id] = min_steps
            states_completed += 1

            if (
                SAVE_PARTIAL_EVERY_STATES > 0
                and states_completed % SAVE_PARTIAL_EVERY_STATES == 0
            ):
                save_partial_oracle(
                    path=PARTIAL_PATH,
                    catalog_hash=catalog_hash,
                    query_state_id=catalogues.query_state_id,
                    query_target=catalogues.query_target,
                    query_min_steps=query_min_steps,
                    states_completed=states_completed,
                )
    except BaseException:
        if pool_context is not None:
            pool_context.terminate()
            pool_context.join()
        raise
    else:
        if pool_context is not None:
            pool_context.close()
            pool_context.join()

    if any(value == -2 for value in query_min_steps):
        unresolved_count = sum(value == -2 for value in query_min_steps)
        raise RuntimeError(f"Oracle left {unresolved_count:,} queries unresolved.")

    solve_seconds = time.time() - solve_start
    query_state_id = torch.tensor(catalogues.query_state_id, dtype=torch.int32)
    query_target = torch.tensor(catalogues.query_target, dtype=torch.int16)
    query_min_steps_tensor = torch.tensor(query_min_steps, dtype=torch.int16)
    query_is_solvable = query_min_steps_tensor >= 0

    query_state_lengths = state_lengths[
        query_state_id.to(torch.long)
    ].to(torch.int16)
    finite = query_min_steps_tensor >= 0

    if bool(
        (
            finite
            & (
                query_min_steps_tensor
                > query_state_lengths - 1
            )
        ).any()
    ):
        raise RuntimeError(
            "Oracle inconsistency: minimum steps exceed state length minus one."
        )

    if bool((query_min_steps_tensor > 5).any()):
        raise RuntimeError(
            "Oracle inconsistency: a six-number state cannot require more "
            "than five forward operations."
        )

    rollout_tensors = rollouts.to_tensors(ROLLOUT_DTYPES)
    state_event_tensors = state_events.to_tensors(STATE_EVENT_DTYPES)
    operation_tensors = operations.to_tensors(OPERATION_DTYPES)
    backtrack_tensors = backtracks.to_tensors(BACKTRACK_DTYPES)

    add_derived_labels(
        state_events=state_event_tensors,
        operations=operation_tensors,
        backtracks=backtrack_tensors,
        state_values=state_values,
        state_lengths=state_lengths,
        query_min_steps=query_min_steps_tensor,
    )

    if not torch.equal(
        state_event_tensors["solvability_known"],
        state_event_tensors["prefix_is_valid"],
    ):
        raise RuntimeError(
            "State labels do not align with generated-prefix validity."
        )

    if not torch.equal(
        operation_tensors["before_solvability_known"],
        operation_tensors["prefix_valid_before"],
    ):
        raise RuntimeError(
            "Before-operation labels do not align with prefix validity."
        )

    if not torch.equal(
        operation_tensors["after_solvability_known"],
        operation_tensors["operation_is_valid"],
    ):
        raise RuntimeError(
            "After-operation labels do not align with operation validity."
        )

    if not torch.equal(
        backtrack_tensors["pre_solvability_known"],
        backtrack_tensors["pre_prefix_valid"],
    ):
        raise RuntimeError(
            "Pre-BACK labels do not align with prefix validity."
        )

    if not torch.equal(
        backtrack_tensors["post_solvability_known"],
        backtrack_tensors["post_prefix_valid"],
    ):
        raise RuntimeError(
            "Post-BACK labels do not align with restoration validity."
        )

    known_positions = state_event_tensors["solvability_known"]
    if not torch.equal(
        state_event_tensors[
            "min_steps_remaining"
        ][known_positions] == 0,
        state_event_tensors[
            "state_contains_target"
        ][known_positions],
    ):
        raise RuntimeError(
            "Zero-step reachability does not match target presence."
        )

    steps = state_event_tensors["min_steps_remaining"]
    expected_binary_mask = (
        state_event_tensors["solvability_known"]
        & (
            (steps == -1)
            | (steps >= 1)
        )
    )
    if not torch.equal(
        state_event_tensors["binary_reachability_label_mask"],
        expected_binary_mask,
    ):
        raise RuntimeError(
            "Binary reachability eligibility mask is inconsistent."
        )

    after_operation = (
        state_event_tensors["event_type"]
        == STATE_EVENT_TO_ID["after_operation"]
    )
    known = state_event_tensors["solvability_known"]
    steps = state_event_tensors["min_steps_remaining"]
    primary_candidate = state_event_tensors["primary_probe_candidate"]
    binary_label = state_event_tensors["binary_reachability_label"]

    category_count = (
        (primary_candidate & binary_label).to(torch.int8)
        + (primary_candidate & ~binary_label).to(torch.int8)
        + (
            after_operation
            & known
            & (steps == 0)
        ).to(torch.int8)
        + (
            after_operation
            & state_event_tensors[
                "latest_operation_introduced_invalidity"
            ]
        ).to(torch.int8)
        + (
            after_operation
            & state_event_tensors[
                "operation_follows_invalid_prefix"
            ]
        ).to(torch.int8)
    )

    if bool((category_count[after_operation] != 1).any()):
        raise RuntimeError(
            "After-operation categories are not mutually exclusive and exhaustive."
        )


    summary = build_summary(
        rollouts=rollout_tensors,
        state_events=state_event_tensors,
        operations=operation_tensors,
        backtracks=backtrack_tensors,
        state_lengths=state_lengths,
        n_queries=int(query_min_steps_tensor.numel()),
    )
    summary["parse_minutes"] = parse_seconds / 60.0
    summary["oracle_minutes"] = solve_seconds / 60.0

    generated_trace_tensors = dict(rollout_tensors)
    generated_trace_tensors["n_state_positions"] = (
        generated_trace_tensors.pop("n_state_events")
    )
    generated_trace_tensors["input_numbers"] = (
        evaluated["input_numbers"][:n_records].to(torch.int16)
    )

    state_occurrence_tensors = dict(state_event_tensors)
    state_occurrence_tensors["position_type"] = (
        state_occurrence_tensors.pop("event_type")
    )

    operation_output_tensors = dict(operation_tensors)
    operation_output_tensors["operations_in_trace"] = (
        operation_output_tensors.pop("operations_in_rollout")
    )

    qualitative_examples = select_qualitative_examples(
        generated_trace_tensors["n_backtracks"]
    )

    payload = {
        "format_version": 4,
        "complete": True,
        "run_tag": RUN_TAG,
        "oracle_version": ORACLE_VERSION,
        "source_evaluated_path": EVALUATED_PATH,
        "source_token_cache_path": evaluated.get("source_token_cache_path"),
        "source_checkpoint_path": evaluated.get("source_checkpoint_path"),
        "oracle_definition": (
            "Exact minimum number of legal forward Countdown operations needed "
            "to place TGT in the current state. BACK is excluded. -1 means "
            "unreachable and 0 means TGT is already present."
        ),
        "operation_legality": {
            "addition": "positive integer addition",
            "subtraction": "left operand strictly greater than right operand",
            "multiplication": "both operands greater than one",
            "division": "positive exact integer division",
        },
        "state_position_type_names": list(STATE_EVENT_NAMES),
        "state_position_type_to_id": dict(STATE_EVENT_TO_ID),
        "generated_traces": generated_trace_tensors,
        "state_catalogue": {
            "values": state_values,
            "lengths": state_lengths,
        },
        "oracle_queries": {
            "state_id": query_state_id,
            "target": query_target,
            "is_forward_solvable": query_is_solvable,
            "min_steps_remaining": query_min_steps_tensor,
        },
        "state_occurrences": state_occurrence_tensors,
        "operations": operation_output_tensors,
        "back_transitions": backtrack_tensors,
        "binary_label_contract": {
            "positive": (
                "min_steps_remaining is at least 1"
            ),
            "negative": "min_steps_remaining equals -1",
            "excluded_target_present": (
                "min_steps_remaining equals 0"
            ),
            "excluded_invalid_prefix": (
                "min_steps_remaining equals -2"
            ),
            "latest_operation_invalid_field": (
                "latest_operation_introduced_invalidity"
            ),
            "prior_prefix_invalid_field": (
                "operation_follows_invalid_prefix"
            ),
            "invalid_transition_policy": (
                "Invalid transitions are retained for audit but excluded "
                "from the reachability probe."
            ),
        },
        "activation_join_contract": {
            "primary_table": "state_occurrences",
            "primary_eligibility_field": "primary_probe_candidate",
            "primary_label_field": "binary_reachability_label",
            "primary_min_steps_field": "min_steps_remaining",
            "primary_token_index_field": "absolute_token_idx",
            "row_identity": (
                "row index within state_occurrences"
            ),
            "operation_result_table": "operations",
            "operation_result_eligibility_field": (
                "after_binary_reachability_label_mask"
            ),
            "operation_result_label_field": (
                "after_binary_reachability_label"
            ),
            "operation_result_token_index_field": (
                "result_end_generated_idx"
            ),
            "operation_result_absolute_index_rule": (
                "generated_traces.prompt_length[record_pos] "
                "+ operations.result_end_generated_idx"
            ),
        },
        "qualitative_example_selection": {
            "seed": QUALITATIVE_EXAMPLE_SEED,
            "examples_per_back_bin": (
                QUALITATIVE_EXAMPLES_PER_BACK_BIN
            ),
            "back_bin_labels": [
                label
                for label, _ in qualitative_examples
            ],
            "record_pos": torch.tensor(
                [
                    record_pos
                    for _, record_pos in qualitative_examples
                ],
                dtype=torch.int32,
            ),
        },
        "summary": summary,
        "catalog_hash": catalog_hash,
    }

    safe_torch_save(payload, OUTPUT_PATH)
    if os.path.exists(PARTIAL_PATH):
        os.remove(PARTIAL_PATH)

    output_mb = os.path.getsize(OUTPUT_PATH) / (1024 ** 2)
    print_summary(summary, OUTPUT_PATH)
    print_qualitative_examples(
        selected_examples=qualitative_examples,
        evaluated=evaluated,
        tok=tok,
        generated_traces=generated_trace_tensors,
        operations=operation_output_tensors,
        back_transitions=backtrack_tensors,
    )
    print(f"Output size: {output_mb:.2f} MiB")


if __name__ == "__main__":
    main()
