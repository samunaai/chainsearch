#!/usr/bin/env python3
"""
generate_final_pretrain_traces.py
===========================
Generate A*-style prompt/trace datasets from an exact-hit HR3\\A* source bundle.

This script:
  1) streams HR3-style record dicts from a sharded directory or single .pt file,
  2) keeps only non-trivial exact-hit records
     (score == 10, expr != None, and target not already in inputs),
  3) reconstructs each record's known-valid solution path from explanation,
  4) injects randomized ancestor-backtracking detours around that path,
  5) writes an A*-style output dataset:
       {
           "prompts": [...],
           "traces":  [...],
           "vocab":   [...],
           "config":  {...},
           "format_version": 2,
           "trace_mode": "randomized_target_backtrack",
       }

Key idea
--------
The output traces are not pure A* search traces. They are randomized,
ancestor-backtracking traces wrapped around a known-valid exact solution path,
so they still finish on the original target.

Compared with the earlier local-detour script, this version is more A*-like in
two ways:
  - detours can be anchored at root or intermediate solution-prefix ancestors
  - after a detour, the trace can replay the correct prefix back to the current
    focus depth before continuing

Output shape
------------
Default mode writes a sharded bundle for memory safety:
    output_base_sharded/
        manifest.json
        shard_000000.pt
        shard_000001.pt
        ...

Each shard is a self-contained A*-style dataset dict with prompts/traces/vocab/
config/format_version/trace_mode.

If --shard-size 0 is used, a single monolithic .pt file is written instead.

Examples
--------
# Edit CONFIG near the top of this file, then run:
#   python generate_final_pretrain_traces.py &> out_pretrain_traces.txt
"""

from __future__ import annotations

# ---------------------------------------------------------------------
# Thread clamping before torch / BLAS imports
# ---------------------------------------------------------------------
import os as _os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
del _os, _v


import hashlib
import json
import multiprocessing
import os
import random
import re
import shutil
import time
from collections import Counter, defaultdict, deque
from typing import Any, Iterator

try:
    import torch
except ImportError as e:
    raise RuntimeError("This script requires torch.") from e

try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

FORMAT_VERSION = 2
TRACE_MODE = "randomized_target_backtrack"
FILE_EXT = ".pt"

 
CONFIG = {
    "input": (
        "/workspace/sam/minireason_archives/data_pts/"
        "colton_h3/hr3_samples_setminus_astar_sharded"
    ),
    "output": (
        "/workspace/sam/minireason_archives/data_pts/"
        "pretrain/randomized_target_traces.pt"
    ),
    "num_records": -1,          # -1 => all exact-hit source records
    "workers": 16,
    "seed": 42,
    "shard_size": 100_000,      # 0 => save one monolithic .pt
    "overwrite": True,
    "rand_max_backtracks": 8,
    "rand_stick_steps": 1,
    "rand_root_back_prob": 0.40,
    "rand_min_back_depth": 2,
    "rand_back_prob": 0.35,
    "rand_step_resample_tries": 64,
    "rand_branch_min_steps": 1,
    "rand_branch_max_steps": 2,
    "rand_max_total_hops": 0,   # 0 => auto
    "rand_trace_retry": 16,
    "rand_max_trace_tokens": 10_000,
    "rand_target_back_weights": [4, 8, 11, 13, 15, 15, 13, 8, 5],
    "rand_two_back_preference": 0.55,
    "compact_steps": False,
    "rand_recent_anchor_window": 4,
    "rand_allow_div_by_one": False,
    "rand_forbid_equal_division": True,
    "rand_root_back_cap": 1,
    "rand_max_two_back_per_trace": 3,
    "rand_branch_candidate_tries": 6,
    "rand_branch_huge_factor": 4,
    "maxtasks": 100,
}
# ---------------------------------------------------------------------
# Tokenizer / shared output contract
# ---------------------------------------------------------------------
class Tokenizer:
    def __init__(self):
        self.special = [
            "PAD", "EOS", "SEP", "IN:", "TGT:", "[", "]", "(", ")", ",",
            "=", ";", "->", "+", "-", "*", "/", "BACK", "SUCCESS"
        ]
        self.cmds = ["EXPR"]
        self.digits = [str(d) for d in range(10)]
        vocab = self.special + self.cmds + self.digits
        self.tok2id = {t: i for i, t in enumerate(vocab)}
        self.id2tok = {i: t for i, t in enumerate(vocab)}
        self._pad = self.tok2id["PAD"]
        self._eos = self.tok2id["EOS"]
        self._sep = self.tok2id["SEP"]
        self._tok_re = re.compile(r"(->|\[|\]|\(|\)|,|=|;|\+|\-|\*|/)")

    @property
    def vocab(self) -> list[str]:
        return list(self.tok2id.keys())

    @property
    def eos_id(self) -> int:
        return self._eos

def encode_ids(tok: Tokenizer, text: str) -> list[int]:
    spaced_text = tok._tok_re.sub(r" \1 ", text)
    tokens = spaced_text.split()
    flat = []
    for t in tokens:
        if t.isdigit() and len(t) > 1:
            flat.extend(list(t))
        else:
            flat.append(t)
    flat.append("EOS")
    return [tok.tok2id[t] for t in flat]

def token_count(tok: Tokenizer, text: str, add_eos: bool = True) -> int:
    ids = encode_ids(tok, text)
    if not add_eos and ids and ids[-1] == tok.eos_id:
        ids = ids[:-1]
    return len(ids)

def _build_canonical_prompt(numbers_tuple: tuple[int, ...], target: int) -> str:
    nums = ", ".join(map(str, numbers_tuple))
    return f"IN: [ {nums} ] TGT: {target}"

# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------
def safe_torch_save(obj: Any, path: str) -> None:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def shard_paths(path: str) -> list[str]:
    if os.path.isdir(path):
        manifest = os.path.join(path, "manifest.json")
        if os.path.exists(manifest):
            with open(manifest) as f:
                data = json.load(f)
            names = data["shards"]
            return [os.path.join(path, n) for n in names]
        return sorted(
            os.path.join(path, fn)
            for fn in os.listdir(path)
            if fn.startswith("shard_") and fn.endswith(FILE_EXT)
        )
    return [path]

def iter_hr3_records(path: str) -> Iterator[dict]:
    for p in shard_paths(path):
        data = torch.load(p, weights_only=False, map_location="cpu")
        if not isinstance(data, list):
            raise ValueError(f"Expected list[dict] in HR3 input shard, got {type(data)} from {p}")
        for record in data:
            yield record

def _build_output_obj(
    prompts: list[str],
    traces: list[str],
    tok: Tokenizer,
    cfg: dict,
) -> dict[str, Any]:
    return {
        "prompts": prompts,
        "traces": traces,
        "vocab": tok.vocab,
        "config": cfg,
        "format_version": FORMAT_VERSION,
        "trace_mode": TRACE_MODE,
    }

def _flush_output_shard(
    prompts: list[str],
    traces: list[str],
    shard_dir: str,
    shard_idx: int,
    tok: Tokenizer,
    cfg: dict,
) -> str:
    os.makedirs(shard_dir, exist_ok=True)
    path = os.path.join(shard_dir, f"shard_{shard_idx:06d}{FILE_EXT}")
    obj = _build_output_obj(prompts, traces, tok, cfg)
    safe_torch_save(obj, path)
    return path

def _write_manifest(bundle_dir: str, shard_paths_list: list[str], tok: Tokenizer, cfg: dict) -> str:
    manifest = os.path.join(bundle_dir, "manifest.json")
    meta = {
        "shards": [os.path.basename(p) for p in sorted(shard_paths_list)],
        "format_version": FORMAT_VERSION,
        "trace_mode": TRACE_MODE,
        "vocab": tok.vocab,
        "config": cfg,
    }
    with open(manifest, "w") as f:
        json.dump(meta, f, indent=2)
    return manifest

def _prepare_output(output_path: str, shard_size: int, overwrite: bool) -> tuple[str, bool]:
    """
    Returns (resolved_output_path, is_sharded_output).

    If shard_size > 0:
      output foo.pt  -> foo_sharded/
      output foo     -> foo/
    Else:
      output is treated as a single file path.
    """
    if shard_size > 0:
        base = output_path
        if base.endswith(FILE_EXT):
            base = os.path.splitext(base)[0] + "_sharded"
        if os.path.exists(base):
            if not overwrite:
                raise FileExistsError(
                    f"Output directory already exists: {base}\n"
                    "Use --overwrite to replace it."
                )
            shutil.rmtree(base)
        os.makedirs(base, exist_ok=True)
        return base, True

    # Monolithic output file
    out_file = output_path
    out_dir = os.path.dirname(os.path.abspath(out_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(out_file):
        if not overwrite:
            raise FileExistsError(
                f"Output file already exists: {out_file}\n"
                "Use --overwrite to replace it."
            )
        os.remove(out_file)
    return out_file, False

# ---------------------------------------------------------------------
# Stable seeds
# ---------------------------------------------------------------------
def _stable_u64(text: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(),
        "little",
    )

def _record_seed(record: dict, base_seed: int) -> int:
    payload = json.dumps(
        {
            "inputs": list(record.get("inputs", ())),
            "target": int(record.get("target", 0)),
            "expr": record.get("expr"),
            "num_used": record.get("num_used"),
            "seed": int(base_seed),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return int(base_seed) ^ _stable_u64(payload)

# ---------------------------------------------------------------------
# Countdown solution / trace helpers
# ---------------------------------------------------------------------
_STEP_RE = re.compile(r"^(\d+)([+\-*/])(\d+)=(\d+)$")
_OPS = ("+", "-", "*", "/")

def _parse_step(step: str) -> tuple[int, str, int, int] | None:
    m = _STEP_RE.match(step.replace(" ", ""))
    if not m:
        return None
    return int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))

def _eval_op(a: int, op: str, b: int) -> int | None:
    if op == "+":
        return a + b
    if op == "-":
        return (a - b) if a > b else None
    if op == "*":
        return a * b
    if op == "/":
        if b == 0:
            return None
        if a % b != 0:
            return None
        return a // b
    return None

def _canonicalize_op_for_trace(a: int, op: str, b: int, res: int) -> tuple[int, str, int, int]:
    """
    Canonicalize operand order to look like A* traces:
      +, * : smaller first
      -, / : larger first
    """
    if op in ("+", "*"):
        x, y = (a, b) if a <= b else (b, a)
        return (x, op, y, res)
    x, y = (a, b) if a >= b else (b, a)
    return (x, op, y, res)

def _apply_op(state: tuple[int, ...], a: int, op: str, b: int, res: int) -> tuple[int, ...]:
    s = list(state)
    s.remove(a)
    s.remove(b)
    s.append(res)
    return tuple(sorted(s))

def _state_str(state: tuple[int, ...]) -> str:
    return "[ " + ", ".join(map(str, state)) + " ]"

def _emit_descend(
    trace_lines: list[str],
    parent_state: tuple[int, ...],
    child_op: tuple[int, str, int, int],
    child_state: tuple[int, ...],
    *,
    last_after_state: tuple[int, ...] | None,
    compact: bool,
) -> str:
    a, sym, b, c = child_op
    if compact and last_after_state is not None and last_after_state == parent_state:
        line = f"{a} {sym} {b} = {c} -> {_state_str(child_state)} ;"
    else:
        line = f"{_state_str(parent_state)} -> {a} {sym} {b} = {c} -> {_state_str(child_state)} ;"
    trace_lines.append(line)
    return line

def _emit_back_to(trace_lines: list[str], state: tuple[int, ...]) -> str:
    line = f"BACK -> {_state_str(state)} ;"
    trace_lines.append(line)
    return line

def _expr_from_ops(initial_numbers: list[int], ops: list[tuple[int, str, int, int]], target: int) -> str:
    if not ops:
        return str(target)

    pool: list[tuple[int, str]] = [(x, str(x)) for x in initial_numbers]

    def take_one(val: int) -> tuple[int, str]:
        for i, (v, e) in enumerate(pool):
            if v == val:
                return pool.pop(i)
        raise AssertionError(f"value {val} not found while building EXPR")

    for a, sym, b, _res in ops:
        va, ea = take_one(a)
        vb, eb = take_one(b)
        expr = f"({ea} {sym} {eb})"
        if sym == "+":
            vc = va + vb
        elif sym == "-":
            vc = va - vb
        elif sym == "*":
            vc = va * vb
        else:
            vc = va // vb
        pool.append((vc, expr))

    return pool[-1][1] if pool else str(target)

def _reconstruct_solution_path(
    record: dict,
) -> tuple[tuple[int, ...], list[tuple[int, ...]], list[tuple[int, str, int, int]], int] | None:
    """
    Returns:
        initial_state, states_along_solution, canonical_solution_ops, target

    states_along_solution has length len(ops)+1 and includes the initial state.
    For exact-hit records, success means target appears in the final state.
    """
    if record.get("score") != 10 or record.get("expr") is None:
        return None

    inputs = tuple(sorted(int(x) for x in record["inputs"]))
    target = int(record["target"])
    explanation = list(record.get("explanation") or [])

    # Exclude trivial exact-hit prompts where the target is already present
    # in the starting multiset. These do not teach search/backtracking.
    if target in inputs:
        return None

    if not explanation:
        return None

    states = [inputs]
    ops: list[tuple[int, str, int, int]] = []
    curr = inputs

    for step in explanation:
        parsed = _parse_step(step)
        if parsed is None:
            return None
        a, op, b, res = parsed

        expected = _eval_op(a, op, b)
        if expected is None or expected != res:
            return None

        c = Counter(curr)
        if c[a] < 1:
            return None
        if a == b and c[a] < 2:
            return None
        if c[b] < 1:
            return None

        op_can = _canonicalize_op_for_trace(a, op, b, res)
        curr = _apply_op(curr, op_can[0], op_can[1], op_can[2], op_can[3])
        states.append(curr)
        ops.append(op_can)

    if target not in states[-1]:
        return None

    return inputs, states, ops, target

# ---------------------------------------------------------------------
# Random branch helpers
# --------------------------------------------------------------------- 

def _sample_random_step(
    state: tuple[int, ...],
    rnd: random.Random,
    max_tries: int = 64,
    forbid_step: tuple[int, str, int, int] | None = None,
    forbidden_steps: set[tuple[int, str, int, int]] | None = None,
    forbidden_result: int | None = None,
    allow_div_by_one: bool = False,
    forbid_equal_division: bool = True,
) -> tuple[tuple[int, str, int, int], tuple[int, ...]] | None:
    """
    Sample one legal random step from the current state.

    Legality is chosen to resemble the A* trace generator:
      + : always legal, smaller-first in trace
      - : strictly positive result, larger-first
      * : disallow *1, smaller-first
      / : exact division, larger-first; /1 can be disabled to avoid identity detours
    """
    n = len(state)
    if n < 2:
        return None

    for _ in range(max_tries):
        i, j = rnd.sample(range(n), 2)
        x, y = state[i], state[j]
        op = rnd.choice(_OPS)

        if op in ("+", "*"):
            a, b = (x, y) if x <= y else (y, x)
            if op == "*":
                if a <= 1 or b <= 1:
                    continue
                res = a * b
            else:
                res = a + b
        elif op == "-":
            a, b = (x, y) if x >= y else (y, x)
            if a <= b:
                continue
            res = a - b
        else:
            a, b = (x, y) if x >= y else (y, x)
            if b == 0:
                continue
            if (not allow_div_by_one) and b <= 1:
                continue
            if forbid_equal_division and a == b:
                continue
            if a % b != 0:
                continue
            res = a // b

        op_tuple = (a, op, b, res)
        if forbid_step is not None and op_tuple == forbid_step:
            continue
        if forbidden_steps is not None and op_tuple in forbidden_steps:
            continue
        if forbidden_result is not None and res == forbidden_result:
            continue

        new_s = list(state)
        for idx in sorted((i, j), reverse=True):
            new_s.pop(idx)
        new_s.append(res)
        return op_tuple, tuple(sorted(new_s))

    return None

def _sample_random_branch(
    start_state: tuple[int, ...],
    rnd: random.Random,
    *,
    min_steps: int,
    max_steps: int,
    step_resample_tries: int,
    forbid_first: tuple[int, str, int, int] | None = None,
    forbidden_first_steps: set[tuple[int, str, int, int]] | None = None,
    forbidden_result: int | None = None,
    allow_div_by_one: bool = False,
    forbid_equal_division: bool = True,
) -> list[tuple[tuple[int, ...], tuple[int, str, int, int], tuple[int, ...]]] | None:
    """
    Sample a random branch as a list of:
        (state_before, op_tuple, state_after)

    Returns None if it cannot reach min_steps.
    """
    if len(start_state) < 2:
        return None

    wanted = rnd.randint(min_steps, max_steps)
    branch = []
    curr = start_state

    for step_idx in range(wanted):
        sampled = _sample_random_step(
            curr,
            rnd,
            max_tries=step_resample_tries,
            forbid_step=forbid_first if step_idx == 0 else None,
            forbidden_steps=forbidden_first_steps if step_idx == 0 else None,
            forbidden_result=forbidden_result,
            allow_div_by_one=allow_div_by_one,
            forbid_equal_division=forbid_equal_division,
        )
        if sampled is None:
            break
        op_tuple, next_state = sampled
        branch.append((curr, op_tuple, next_state))
        curr = next_state
        if len(curr) < 2:
            break

    if len(branch) < min_steps:
        return None
    return branch

def _score_branch(
    start_state: tuple[int, ...],
    branch: list[tuple[tuple[int, ...], tuple[int, str, int, int], tuple[int, ...]]],
    target: int,
    *,
    huge_factor: int,
) -> float:
    """
    Heuristic score for a sampled wrong branch.

    Higher is better. We want detours that:
      - are not degenerate
      - look somewhat goal-relevant
      - avoid absurd blow-ups unless unavoidable
    """
    if not branch:
        return float("-inf")

    start_gap = min(abs(v - target) for v in start_state)
    end_state = branch[-1][2]
    end_gap = min(abs(v - target) for v in end_state)

    score = 0.0

    # Prefer branches whose frontier still looks somewhat target-relevant.
    if end_gap < start_gap:
        score += 3.0
    elif end_gap == start_gap:
        score += 1.0
    else:
        score -= 1.0

    # Penalize equal-division identity-ish moves such as 5 / 5 = 1.
    if any(op == "/" and a == b for _, (a, op, b, _), _ in branch):
        score -= 4.0

    # Mild penalty for producing 1 in a detour.
    if any(res == 1 for _, (_, _, _, res), _ in branch):
        score -= 1.5

    # Prefer branches that do not collapse too quickly.
    if len(end_state) <= 2:
        score -= 0.75

    # Penalize very large intermediates in random detours.
    huge_cap = max(2000, huge_factor * max(100, target))
    max_seen = max(max(after_state) for _, _, after_state in branch)
    if max_seen > huge_cap:
        score -= 2.5

    # Slight preference for 2-step branches over 1-step branches when both are available.
    score += 0.25 * len(branch)

    return score

def _choose_scored_branch(
    start_state: tuple[int, ...],
    rnd: random.Random,
    *,
    target: int,
    min_steps: int,
    max_steps: int,
    step_resample_tries: int,
    forbid_first: tuple[int, str, int, int] | None,
    forbidden_first_steps: set[tuple[int, str, int, int]] | None,
    allow_div_by_one: bool,
    forbid_equal_division: bool,
    branch_candidate_tries: int,
    huge_factor: int,
) -> list[tuple[tuple[int, ...], tuple[int, str, int, int], tuple[int, ...]]] | None:
    """
    Sample several candidate branches and keep the best-scoring one.

    Novel first steps are preferred first; if none are available, fallback
    to unconstrained first steps.
    """
    best_branch = None
    best_score = float("-inf")

    def _try_pool(blocked_first_steps: set[tuple[int, str, int, int]] | None) -> None:
        nonlocal best_branch, best_score
        for _ in range(max(1, branch_candidate_tries)):
            maybe_branch = _sample_random_branch(
                start_state,
                rnd,
                min_steps=min_steps,
                max_steps=max_steps,
                step_resample_tries=step_resample_tries,
                forbid_first=forbid_first,
                forbidden_first_steps=blocked_first_steps,
                forbidden_result=target,
                allow_div_by_one=allow_div_by_one,
                forbid_equal_division=forbid_equal_division,
            )
            if maybe_branch is None:
                continue
            score = _score_branch(
                start_state,
                maybe_branch,
                target,
                huge_factor=huge_factor,
            )
            if score > best_score:
                best_branch = maybe_branch
                best_score = score

    if forbidden_first_steps:
        _try_pool(forbidden_first_steps)
        if best_branch is None:
            _try_pool(None)
    else:
        _try_pool(None)

    return best_branch

def _ordered_two_back_anchor_depths(
    current_depth: int,
    rnd: random.Random,
    root_prob: float,
    recent_anchor_depths: deque[int],
) -> list[int]:
    """
    Order strict-ancestor anchors for 2-BACK detours.

    Preference:
      1) optionally front-load root
      2) prefer anchors not used recently
      3) still keep all valid ancestors as fallback
    """
    if current_depth <= 0:
        return [0]

    candidates = list(range(current_depth))  # strict ancestors only
    recent = set(recent_anchor_depths)

    novel_nonroot = [d for d in candidates if d != 0 and d not in recent]
    recent_nonroot = [d for d in candidates if d != 0 and d in recent]

    # Prefer deeper anchors first to reduce replay length, but keep some
    # randomness inside the ordering.
    novel_nonroot = sorted(novel_nonroot, key=lambda d: (-d, rnd.random()))
    recent_nonroot = sorted(recent_nonroot, key=lambda d: (-d, rnd.random()))

    ordered: list[int] = []
    if 0 in candidates and rnd.random() < root_prob:
        ordered.append(0)
    ordered.extend(novel_nonroot)
    if 0 in candidates and 0 not in ordered:
        ordered.append(0)
    ordered.extend(recent_nonroot)
    return ordered

def _sample_target_back_count(
    max_backs: int,
    rnd: random.Random,
    weights: tuple[int, ...],
) -> int:
    """
    Sample the desired BACK budget for one trace from a fixed target
    distribution over BACK counts 0..max_backs.
    """
    if max_backs < 0:
        return 0
    usable_weights = list(weights[: max_backs + 1])
    vals = list(range(len(usable_weights)))
    return rnd.choices(vals, weights=usable_weights, k=1)[0]

# ---------------------------------------------------------------------
# Core randomized trace builder
# ---------------------------------------------------------------------
def _build_randomized_trace(
    record: dict,
    cfg: dict,
    tok: Tokenizer,
) -> dict | None:
    rec = _reconstruct_solution_path(record)
    if rec is None:
        return None

    inputs, states, solution_ops, target = rec
    prompt = _build_canonical_prompt(inputs, target)
 

    base_seed = _record_seed(record, int(cfg["seed"]))
    max_backs = int(cfg["rand_max_backtracks"])
    stick_steps = int(cfg["rand_stick_steps"])
    root_back_prob = float(cfg["rand_root_back_prob"])
    min_back_depth = int(cfg["rand_min_back_depth"])
    back_prob = float(cfg["rand_back_prob"])
    step_resample_tries = int(cfg["rand_step_resample_tries"])
    branch_min_steps = int(cfg["rand_branch_min_steps"])
    branch_max_steps = int(cfg["rand_branch_max_steps"])
    max_total_hops = int(cfg["rand_max_total_hops"])
    trace_retry = int(cfg["rand_trace_retry"])
    max_trace_tokens = int(cfg["rand_max_trace_tokens"])
    compact = bool(cfg.get("compact_steps", False))

    target_back_weights = tuple(cfg["rand_target_back_weights"])
    two_back_preference = float(cfg.get("rand_two_back_preference", 0.78))
    recent_anchor_window = int(cfg.get("rand_recent_anchor_window", 2))
    allow_div_by_one = bool(cfg.get("rand_allow_div_by_one", False))
    forbid_equal_division = bool(cfg.get("rand_forbid_equal_division", True))
    root_back_cap = int(cfg.get("rand_root_back_cap", 2))
    max_two_back_per_trace = int(cfg.get("rand_max_two_back_per_trace", 3))
    branch_candidate_tries = int(cfg.get("rand_branch_candidate_tries", 6))
    huge_factor = int(cfg.get("rand_branch_huge_factor", 4))

    # Short exact solutions should still be allowed to carry BACKs.
    effective_min_back_depth = min(min_back_depth, max(0, len(solution_ops) - 1))

    if max_total_hops <= 0:
        # Base path length + replay allowance + backtrack allowance + slack
        max_total_hops = 2 * len(solution_ops) + 3 * max_backs + 20

    target_back_rng = random.Random(base_seed ^ _stable_u64("target_back_budget"))
    target_backs = _sample_target_back_count(
        max_backs=max_backs,
        rnd=target_back_rng,
        weights=target_back_weights,
    )

    best_out: dict | None = None
    best_n_backs = -1

    for retry_idx in range(trace_retry):
        rnd = random.Random(base_seed ^ _stable_u64(f"retry:{retry_idx}"))

        used_first_steps_by_anchor: dict[int, set[tuple[int, str, int, int]]] = defaultdict(set)
        recent_anchor_depths: deque[int] = deque(maxlen=max(1, recent_anchor_window))
        root_two_back_used = 0
        two_back_used = 0
        replayed_prefix_pairs: set[tuple[int, int]] = set()

        trace_lines: list[str] = []
        last_after: tuple[int, ...] | None = None
        backs_used = 0
        backs_remaining = target_backs
        hops_used = 0
        stick_remaining = 0
        ok = True

        for depth in range(len(solution_ops)):
            branched_here = False
            detours_this_depth = 0

            # Cooldown applies to FUTURE depths, not the current one where the
            # detour was just taken.
            can_branch_here = True
            if stick_remaining > 0:
                stick_remaining -= 1
                can_branch_here = False

            # Spend the sampled BACK budget deliberately rather than relying on
            # one Bernoulli trial per depth.
            while (
                can_branch_here
                and backs_remaining > 0
                and hops_used < max_total_hops
                and depth >= effective_min_back_depth
            ):
                depths_left = len(solution_ops) - depth
                must_spend_now = (
                    (depths_left == 1 and backs_remaining > 0)
                    or
                    (backs_remaining >= 2 * max(1, depths_left))
                )

                # Usually allow just one detour opportunity per depth, but
                # if we are behind budget, force more at the same depth.
                if detours_this_depth >= 1 and not must_spend_now:
                    break
                if detours_this_depth == 0 and not must_spend_now and rnd.random() >= back_prob:
                    break

                fresh_two_back: list[tuple[int, int]] = []
                stale_two_back: list[tuple[int, int]] = []

                if backs_remaining >= 2 and depth > 0 and two_back_used < max_two_back_per_trace:
                    for anchor_depth in _ordered_two_back_anchor_depths(
                        depth,
                        rnd,
                        root_back_prob,
                        recent_anchor_depths,
                    ):
                        if anchor_depth == 0 and root_two_back_used >= root_back_cap:
                            continue
                        plan = (2, anchor_depth)
                        if (anchor_depth, depth) in replayed_prefix_pairs:
                            stale_two_back.append(plan)
                        else:
                            fresh_two_back.append(plan)

                one_back: list[tuple[int, int]] = []

                if backs_remaining >= 1:
                    one_back = [(1, depth)]

                if must_spend_now:
                    # Still prefer fresh 2-BACKs when forced, but keep 1-BACK
                    # ahead of repeated replay-pair 2-BACKs.
                    candidate_plans = fresh_two_back + one_back + stale_two_back
                else:
                    prefer_two = (rnd.random() < two_back_preference)
                    if prefer_two:
                        candidate_plans = fresh_two_back + one_back + stale_two_back
                    else:
                        candidate_plans = one_back + fresh_two_back + stale_two_back

                if not candidate_plans:
                    break

                chosen_back_cost = 0
                chosen_anchor_depth = depth
                branch = None
                chosen_first_step: tuple[int, str, int, int] | None = None

                for back_cost, anchor_depth in candidate_plans:
                    forbid_first = solution_ops[anchor_depth] if anchor_depth < len(solution_ops) else None
                    used_first_steps = used_first_steps_by_anchor[anchor_depth]

                    maybe_branch = _choose_scored_branch(
                        states[anchor_depth],
                        rnd,
                        target=target,
                        min_steps=branch_min_steps,
                        max_steps=branch_max_steps,
                        step_resample_tries=step_resample_tries,
                        forbid_first=forbid_first,
                        forbidden_first_steps=used_first_steps if used_first_steps else None,
                        allow_div_by_one=allow_div_by_one,
                        forbid_equal_division=forbid_equal_division,
                        branch_candidate_tries=branch_candidate_tries,
                        huge_factor=huge_factor,
                        )

                    if maybe_branch:
                        chosen_back_cost = back_cost
                        chosen_anchor_depth = anchor_depth
                        branch = maybe_branch
                        chosen_first_step = maybe_branch[0][1]
                        break

                if not branch:
                    break

                # 2-BACK detour: first jump up to a strict ancestor.
                if chosen_back_cost == 2:
                    two_back_used += 1
                    _emit_back_to(trace_lines, states[chosen_anchor_depth])
                    last_after = states[chosen_anchor_depth]
                    backs_used += 1
                    hops_used += 1
                    if hops_used >= max_total_hops:
                        ok = False
                        break

                # Emit wrong branch from the chosen anchor.
                for before_state, op_tuple, after_state in branch:
                    _emit_descend(
                        trace_lines,
                        before_state,
                        op_tuple,
                        after_state,
                        last_after_state=last_after,
                        compact=compact,
                    )
                    last_after = after_state
                    hops_used += 1
                    if hops_used >= max_total_hops:
                        ok = False
                        break
                if not ok:
                    break

                # Always BACK to the anchor after exploring the wrong branch.
                _emit_back_to(trace_lines, states[chosen_anchor_depth])
                last_after = states[chosen_anchor_depth]
                backs_used += 1
                hops_used += 1
                if hops_used >= max_total_hops:
                    ok = False
                    break

                # 2-BACK detours replay the correct prefix back to the current focus.
                if chosen_back_cost == 2:
                    for k in range(chosen_anchor_depth, depth):
                        _emit_descend(
                            trace_lines,
                            states[k],
                            solution_ops[k],
                            states[k + 1],
                            last_after_state=last_after,
                            compact=compact,
                        )
                        last_after = states[k + 1]
                        hops_used += 1
                        if hops_used >= max_total_hops:
                            ok = False
                            break
                    if not ok:
                        break

                if chosen_first_step is not None:
                    used_first_steps_by_anchor[chosen_anchor_depth].add(chosen_first_step)
                if chosen_back_cost == 2:
                    recent_anchor_depths.append(chosen_anchor_depth)
                    replayed_prefix_pairs.add((chosen_anchor_depth, depth))
                    if chosen_anchor_depth == 0:
                        root_two_back_used += 1

                backs_remaining -= chosen_back_cost
                branched_here = True
                detours_this_depth += 1

            if not ok:
                break

            # Emit the next correct solution step.
            _emit_descend(
                trace_lines,
                states[depth],
                solution_ops[depth],
                states[depth + 1],
                last_after_state=last_after,
                compact=compact,
            )
            last_after = states[depth + 1]
            hops_used += 1

            if branched_here and stick_steps > 0:
                stick_remaining = max(stick_remaining, stick_steps)

            if hops_used >= max_total_hops:
                ok = False
                break

        if not ok:
            continue 

        expr = _expr_from_ops(list(inputs), solution_ops, target=target)
        trace_lines.append(f"SUCCESS ; EXPR ({expr})")
        trace = " ".join(trace_lines)

        trace_tokens = token_count(tok, f"{prompt} SEP {trace}", add_eos=True)
        if max_trace_tokens > 0 and trace_tokens > max_trace_tokens:
            continue

        out = {
            "prompt": prompt,
            "trace": trace,
            "n_backs": backs_used,
            "target_backs": target_backs,
            "n_hops": hops_used,
            "trace_tokens": trace_tokens,
        }

        # Best case: hit the requested BACK budget exactly.
        if backs_used == target_backs:
            return out

        # Otherwise keep the best-effort trace instead of dropping the sample.
        if out["n_backs"] > best_n_backs:
            best_out = out
            best_n_backs = out["n_backs"]

    return best_out

# ---------------------------------------------------------------------
# Multiprocessing
# ---------------------------------------------------------------------
_WORKER_CFG: dict[str, Any] = {}
_WORKER_TOK: Tokenizer | None = None

def _worker_init(cfg: dict[str, Any]) -> None:
    global _WORKER_CFG, _WORKER_TOK
    _WORKER_CFG = dict(cfg)
    _WORKER_TOK = Tokenizer()
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

def _process_record(record: dict) -> dict | None:
    global _WORKER_CFG, _WORKER_TOK
    if _WORKER_TOK is None:
        _WORKER_TOK = Tokenizer()
    return _build_randomized_trace(record, _WORKER_CFG, _WORKER_TOK)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
 
    args = CONFIG

    resolved_output, is_sharded = _prepare_output(
        args["output"],
        args["shard_size"],
        args["overwrite"],
    )

    cfg = {
        "source_path": args["input"],
        "seed": int(args["seed"]),
        "num_records": int(args["num_records"]),
        "workers": int(args["workers"]),
        "shard_size": int(args["shard_size"]),
        "compact_steps": bool(args["compact_steps"]),
        "rand_max_backtracks": int(args["rand_max_backtracks"]),
        "rand_stick_steps": int(args["rand_stick_steps"]),
        "rand_root_back_prob": float(args["rand_root_back_prob"]),
        "rand_min_back_depth": int(args["rand_min_back_depth"]),
        "rand_back_prob": float(args["rand_back_prob"]),
        "rand_step_resample_tries": int(args["rand_step_resample_tries"]),
        "rand_branch_min_steps": int(args["rand_branch_min_steps"]),
        "rand_branch_max_steps": int(args["rand_branch_max_steps"]),
        "rand_max_total_hops": int(args["rand_max_total_hops"]),
        "rand_trace_retry": int(args["rand_trace_retry"]),
        "rand_max_trace_tokens": int(args["rand_max_trace_tokens"]),
        "rand_target_back_weights": list(args["rand_target_back_weights"]),
        "rand_two_back_preference": float(args["rand_two_back_preference"]),
        "rand_recent_anchor_window": int(args["rand_recent_anchor_window"]),
        "rand_allow_div_by_one": bool(args["rand_allow_div_by_one"]),
        "rand_forbid_equal_division": bool(args["rand_forbid_equal_division"]),
        "rand_root_back_cap": int(args["rand_root_back_cap"]),
        "rand_max_two_back_per_trace": int(args["rand_max_two_back_per_trace"]),
        "rand_branch_candidate_tries": int(args["rand_branch_candidate_tries"]),
        "rand_branch_huge_factor": int(args["rand_branch_huge_factor"]),
        "format_version": FORMAT_VERSION,
        "trace_mode": TRACE_MODE,
        "source_filter": "score == 10 and expr != None and target not in inputs",
    }

    tok = Tokenizer()
    limit = args["num_records"] if args["num_records"] > 0 else None

    print(f"[info] Input             : {args['input']}")
    print(f"[info] Output            : {resolved_output}")
    print(f"[info] Output mode       : {'sharded bundle' if is_sharded else 'single .pt'}")
    print(f"[info] Records limit     : {'ALL' if limit is None else f'{limit:,}'}")
    print(f"[info] Workers           : {args['workers']}")
    print(f"[info] compact_steps     : {int(bool(args['compact_steps']))}")
    print(f"[info] rand_max_backtracks={args['rand_max_backtracks']}  "
          f"rand_back_prob={args['rand_back_prob']}  rand_root_back_prob={args['rand_root_back_prob']}  "
          f"rand_min_back_depth={args['rand_min_back_depth']}  rand_stick_steps={args['rand_stick_steps']}")
    print(f"[info] rand_branch_steps=[{args['rand_branch_min_steps']}, {args['rand_branch_max_steps']}]  "
          f"rand_step_resample_tries={args['rand_step_resample_tries']}  "
          f"rand_max_total_hops={args['rand_max_total_hops']}  rand_trace_retry={args['rand_trace_retry']}")
    print(f"[info] rand_target_back_weights={cfg['rand_target_back_weights'][:args['rand_max_backtracks'] + 1]}  "
          f"rand_two_back_preference={args['rand_two_back_preference']}  "
          f"rand_recent_anchor_window={args['rand_recent_anchor_window']}  "
          f"rand_allow_div_by_one={int(bool(args['rand_allow_div_by_one']))}  "
          f"rand_forbid_equal_division={int(bool(args['rand_forbid_equal_division']))}  "
          f"rand_root_back_cap={args['rand_root_back_cap']}  "
          f"rand_max_two_back_per_trace={args['rand_max_two_back_per_trace']}  "
          f"rand_branch_candidate_tries={args['rand_branch_candidate_tries']}  "
          f"rand_branch_huge_factor={args['rand_branch_huge_factor']}")
    print()

    source_exact_raw = 0
    source_trivial_skipped = 0

    def source_iter() -> Iterator[dict]:
        nonlocal source_exact_raw, source_trivial_skipped
        for r in iter_hr3_records(args["input"]):
            if r.get("score") != 10 or r.get("expr") is None:
                continue
            source_exact_raw += 1
            if int(r["target"]) in tuple(r["inputs"]):
                source_trivial_skipped += 1
                continue
            yield r

    records_iter = source_iter()

    prompts_buf: list[str] = []
    traces_buf: list[str] = []
    shard_paths_list: list[str] = []
    shard_idx = 0

    total_seen_exact = 0
    total_produced = 0
    total_failed = 0
    total_with_back = 0
    total_backs = 0
    total_target_backs = 0
    under_target_count = 0
    total_hops = 0
    total_tokens = 0
    backs_min: int | None = None
    backs_max = 0
    hops_min: int | None = None
    hops_max = 0
    tokens_min: int | None = None
    tokens_max = 0
    back_dist: Counter[int] = Counter()
    target_back_dist: Counter[int] = Counter()
    exact_target_back_match = 0
    sample_pair: tuple[str, str] | None = None

    t_start = time.perf_counter()

    def _wrap_tqdm(it, total=None):
        if _HAS_TQDM:
            return tqdm(it, total=total, desc="Generating", unit="rec", dynamic_ncols=True)
        return it

    def _flush_if_needed(force: bool = False) -> None:
        nonlocal prompts_buf, traces_buf, shard_idx
        if not is_sharded:
            return
        if not prompts_buf:
            return
        if (not force) and len(prompts_buf) < args["shard_size"]:
            return
        shard_path = _flush_output_shard(
            prompts_buf,
            traces_buf,
            resolved_output,
            shard_idx,
            tok,
            cfg,
        )
        shard_paths_list.append(shard_path)
        shard_idx += 1
        prompts_buf = []
        traces_buf = []

    def _commit(out: dict) -> None:
        nonlocal total_produced, total_with_back, total_backs, total_target_backs, total_hops, total_tokens
        nonlocal backs_min, backs_max, hops_min, hops_max, tokens_min, tokens_max
        nonlocal exact_target_back_match, under_target_count
        nonlocal sample_pair
        prompts_buf.append(out["prompt"])
        traces_buf.append(out["trace"])
        total_produced += 1
        total_backs += out["n_backs"]
        total_target_backs += out.get("target_backs", out["n_backs"])
        total_hops += out["n_hops"]
        total_tokens += out["trace_tokens"]
        backs_min = out["n_backs"] if backs_min is None else min(backs_min, out["n_backs"])
        backs_max = max(backs_max, out["n_backs"])
        hops_min = out["n_hops"] if hops_min is None else min(hops_min, out["n_hops"])
        hops_max = max(hops_max, out["n_hops"])
        tokens_min = out["trace_tokens"] if tokens_min is None else min(tokens_min, out["trace_tokens"])
        tokens_max = max(tokens_max, out["trace_tokens"])
        back_dist[out["n_backs"]] += 1
        target_back_dist[out.get("target_backs", out["n_backs"])] += 1
        if out["n_backs"] == out.get("target_backs", out["n_backs"]):
            exact_target_back_match += 1
        elif out["n_backs"] < out.get("target_backs", out["n_backs"]):
            under_target_count += 1
        if out["n_backs"] > 0:
            total_with_back += 1
        if sample_pair is None:
            sample_pair = (out["prompt"], out["trace"])
        _flush_if_needed(force=False)

    if args["workers"] <= 1:
        for record in _wrap_tqdm(records_iter, total=None):
            total_seen_exact += 1
            out = _build_randomized_trace(record, cfg, tok)
            if out is None:
                total_failed += 1
            else:
                _commit(out)
            if limit is not None and total_produced >= limit:
                break
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(
            processes=args["workers"],
            initializer=_worker_init,
            initargs=(cfg,),
            maxtasksperchild=args["maxtasks"],
        ) as pool:
            it = pool.imap_unordered(_process_record, records_iter, chunksize=64)
            for out in _wrap_tqdm(it, total=None):
                total_seen_exact += 1
                if out is None:
                    total_failed += 1
                else:
                    _commit(out)
                if limit is not None and total_produced >= limit:
                    pool.terminate()
                    break

    # Final flush / save
    _flush_if_needed(force=True)

    elapsed = time.perf_counter() - t_start

    if not is_sharded:
        obj = _build_output_obj(prompts_buf, traces_buf, tok, cfg)
        safe_torch_save(obj, resolved_output)

    if is_sharded:
        manifest = _write_manifest(resolved_output, shard_paths_list, tok, cfg)
    else:
        manifest = None

    print(f"\n[info] Raw exact-hit source records : {source_exact_raw:,}")
    print(f"[info] Trivial exact-hit skipped    : {source_trivial_skipped:,}")
    if source_exact_raw:
        print(f"[info] Eligible exact-hit consumed  : {total_seen_exact:,} ({100*total_seen_exact/source_exact_raw:.2f}%)")
    else:
        print(f"[info] Eligible exact-hit consumed  : {total_seen_exact:,}")

    print(f"[info] Produced output samples     : {total_produced:,}")
    print(f"[info] Failed / skipped after retry: {total_failed:,}")
    if limit is not None and total_produced < limit:
        print(f"[warning] Underfilled requested count: {total_produced:,}/{limit:,}")
    print(f"[info] Runtime                     : {elapsed:.1f}s")
    if total_produced:
        zero_back = total_produced - total_with_back
        print(f"[info] Records with 0 BACK         : {zero_back:,} ({100*zero_back/total_produced:.1f}%)")
        print(f"[info] Records with ≥1 BACK        : {total_with_back:,} ({100*total_with_back/total_produced:.1f}%)")
        print(f"[info] Target BACKs / trace       : mean {total_target_backs/total_produced:.2f}")
        print(f"[info] Realized==target BACKs     : {exact_target_back_match:,} ({100*exact_target_back_match/total_produced:.1f}%)")
        print(f"[info] Under-target BACKs        : {under_target_count:,} ({100*under_target_count/total_produced:.1f}%)")
        print(f"[info] BACKs / trace              : min {backs_min}, max {backs_max}, mean {total_backs/total_produced:.2f}")
        print(f"[info] Hops / trace               : min {hops_min}, max {hops_max}, mean {total_hops/total_produced:.2f}")
        print(f"[info] Tokens / trace             : min {tokens_min}, max {tokens_max}, mean {total_tokens/total_produced:.2f}")
        print("[info] Target BACK distribution:")
        for k in sorted(target_back_dist):
            print(f"         target {k} : {target_back_dist[k]:,}")
        print("[info] BACK count distribution:")
        for k in sorted(back_dist):
            print(f"         {k} BACKs : {back_dist[k]:,}")

    if sample_pair is not None:
        prompt, trace = sample_pair
        print("\n[sample] First generated sample:")
        print(f"  prompt : {prompt}")
        print("  trace  :")
        print("    " + trace.replace(" ;", " ;\n    "))

    if is_sharded:
        print(f"\n[info] Saved sharded bundle → {resolved_output}")
        print(f"[info] Manifest            → {manifest}")
    else:
        size_mb = os.path.getsize(resolved_output) / 1e6
        print(f"\n[info] Saved → {resolved_output}  ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()