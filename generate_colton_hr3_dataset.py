"""
generate_colton_style_dataset.py
================================
Implements Colton (2014) "Countdown Numbers Game: Solved, Analysed, Extended."

Algorithm
---------
One theory-formation session per sorted 6-number input list (13,243 total).
Each session uses subset-DP to exhaustively generate all legal arithmetic
expressions, keeping only the optimal one per (subset, value) under Colton's
three-level ordering: fewest inputs used → smallest largest intermediate →
fewest distinct operators.  All 900 targets [100-999] are then resolved from
that single table via exact hit or nearest hit (bisect).

Key design choices
------------------
- Subset-DP (not flat enumeration): dominated expressions never generate
  children, making the full 11.9M-puzzle run tractable.
- Bitmask DP keys + ops bitmask on Expr: int bitmasks for both input subsets
  and operator sets; O(1) union/intersection; lower memory per DP cell.
- Thread clamping at import time (before torch loads) + _worker_init() second
  layer: no silent BLAS thread explosion at 96-128 workers.
- Seeded shuffle (SHARD_SEED) + chunksize=1: deterministic order → stable
  resume; --limit slices after shuffle for representative benchmarking.
- Atomic shard saves (write-then-rename) + MD5-keyed filenames: crash-safe
  and resume-stable across re-runs regardless of shuffle order.
- Sharded output: workers save one backend-native shard each (.pt or .pkl);
  main process never holds
  all 11.9M records in RAM simultaneously.
- maxtasksperchild: recycles workers periodically to avoid memory fragmentation
  on long runs.

Recommended worker counts (benchmark before full run):
    32  → debug / correctness
    64  → first serious production run
    96-128 → likely sweet spot (Colton used 128)
    256 → experiment only; SMT contention may hurt

Record format (one dict per puzzle instance)
--------------------------------------------
    target      int          T  (100-999)
    answer      int|None     A, best achievable
    gap         int|None     |T - A|
    inputs      tuple[int]   I, sorted 6-number list
    expr        str|None     C, fully-parenthesised expression
    num_used    int|None     U(P)
    num_ops     int|None     N(P)
    max_interm  int|None     L(P)
    explanation list[str]    E, e.g. ["75*6=450", "450/50=9", ...]
    score       int          10 / 7 / 5 / 0

Usage
-----
    # Demo (one input list):
    python generate_colton_hr3_dataset.py --input-list 3 6 25 50 75 100

    # Quick test (first 200 lists, 32 workers):
    python generate_colton_hr3_dataset.py --limit 200 --workers 32

    # Full run (shards only, no bundling):
    python generate_colton_hr3_dataset.py --workers 96 --no-merge --shard-dir ./shards

    # Full run (manifest bundle directory):
    python generate_colton_hr3_dataset.py --workers 96 --out countdown_colton_dataset.pt

Loading
-------
    from generate_colton_hr3_dataset import load_dataset
    records = load_dataset("./countdown_colton_dataset_sharded")  # bundled dir
    records = load_dataset("./shards")                            # shard dir
"""

from __future__ import annotations

# ── Thread clamping: must happen before torch (or any BLAS lib) is imported ──
import os as _os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
del _os, _v

import argparse
import bisect
import hashlib
import itertools
import json
import multiprocessing
import os
import random
import sys
import tempfile
import time
from typing import Any
from tqdm.auto import tqdm

# ── torch / pickle fallback ──────────────────────────────────────────────────
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    import pickle as _pickle
    _HAS_TORCH = False
    print("[WARNING] torch not found – using pickle-based serialization instead of torch.save.",
          file=sys.stderr)
FILE_EXT = ".pt" if _HAS_TORCH else ".pkl"

# ── Output paths ─────────────────────────────────────────────────────────────
OUTPUT_DIR      = os.environ.get(
    "COUNTDOWN_OUTPUT_DIR",
    "/workspace/sam/minireason_archives/data_pts/colton_h3",
)
# Full runs are stored as a manifest bundle directory, not one monolithic .pt.
# Using an extension-free name avoids the automatic "*_sharded" rename.
OUTPUT_BUNDLE   = os.path.join(OUTPUT_DIR, "countdown_colton_h3_full")
OUTPUT_SHARDS   = os.path.join(OUTPUT_DIR, "shards")
DEMO_OUTPUT     = os.path.join(OUTPUT_DIR, "countdown_colton_h3_demo.pt")

# ── Tuning macros ─────────────────────────────────────────────────────────────
DEFAULT_WORKERS = 32    # start here; tune to 64 / 96 / 128 for production
                        # or: COUNTDOWN_WORKERS=96 python generate_colton_hr3_dataset.py
SHARD_SEED      = 42    # fixed seed → deterministic shuffle → stable resume

# ── Countdown pool & constants  (paper §1) ───────────────────────────────────
COUNTDOWN_POOL: list[int] = sorted(list(range(1, 11)) * 2 + [25, 50, 75, 100])
TARGET_MIN, TARGET_MAX = 100, 999
OPERATORS = ("+", "-", "*", "/")
_OP_BIT   = {"+": 1, "-": 2, "*": 4, "/": 8}   # bitmask for operator tracking
 

# Tiny popcount tables are faster than repeated bin(...).count("1")
SUBSET_POPCOUNT = [i.bit_count() for i in range(1 << 6)]   # 0..63
OPS_POPCOUNT    = [i.bit_count() for i in range(1 << 4)]   # 0..15
# ── Expression node ──────────────────────────────────────────────────────────

class Expr:
    """
    One arithmetic expression.  All fields are immutable after construction.

    Optimality key (minimise, Colton §3):  (len(inputs_used), max_interm, n_ops)

    inputs_used  int  bitmask (bit i set ↔ input index i used; n=6 → max 63).
    ops_mask     int  bitmask over _OP_BIT; popcount == n_ops.
    """
    __slots__ = ("value", "inputs_used", "max_interm", "n_ops",
                 "ops_mask", "canon", "left", "right", "op")

    def __init__(self, value, inputs_used, max_interm, ops_mask,
                 canon, left=None, right=None, op=None):
        self.value       = value
        self.inputs_used = inputs_used
        self.max_interm  = max_interm
        self.ops_mask    = ops_mask
        self.n_ops       = OPS_POPCOUNT[ops_mask]
        self.canon       = canon
        self.left        = left
        self.right       = right
        self.op          = op

    def key(self):
        return (SUBSET_POPCOUNT[self.inputs_used], self.max_interm, self.n_ops)

    def better_than(self, other: Expr) -> bool:
        return self.key() < other.key()


def _leaf(idx: int, value: int) -> Expr:
    return Expr(value, 1 << idx, 0, 0, str(value))

def _render_steps(expr: Expr) -> list[str]:
    """Reconstruct explanation lazily from backpointers."""
    out: list[str] = []

    def dfs(node: Expr) -> None:
        if node.left is None:
            return
        dfs(node.left)
        dfs(node.right)
        out.append(f"{node.left.value}{node.op}{node.right.value}={node.value}")

    dfs(expr)
    return out

def _combine(left: Expr, right: Expr, op: str) -> Expr | None:
    """
    Combine two disjoint-index expressions.  Returns None if illegal or trivial.

    Symmetry-breaking (Colton §3.1):
      +/* commutative: require lv >= rv; on tie require left.canon <= right.canon.
      -   require lv > rv  (strictly positive result).
      /   require exact division, rv > 1.
      No ×1, +0, -0.
    """
    lv, rv = left.value, right.value

    if op == "+":
        if lv < rv or rv == 0: return None
        if lv == rv and left.canon > right.canon: return None
        result = lv + rv
    elif op == "-":
        if lv <= rv: return None
        result = lv - rv
    elif op == "*":
        if lv < rv or rv <= 1: return None
        if lv == rv and left.canon > right.canon: return None
        result = lv * rv
    elif op == "/":
        if rv <= 1 or lv % rv: return None
        result = lv // rv
    else:
        raise ValueError(op)
 
    new_mask  = left.ops_mask | right.ops_mask | _OP_BIT[op]
    return Expr(
        value       = result,
        inputs_used = left.inputs_used | right.inputs_used,
        # lv & rv are the sub-results consumed by this op; they are intermediate
        # values relative to the parent. `result` is excluded, matching Colton §3
        # definition of L(P): largest intermediate NOT including the final answer.
        max_interm  = max(left.max_interm, right.max_interm, lv, rv),
        ops_mask    = new_mask,
        canon       = f"({left.canon}{op}{right.canon})",
        left        = left,
        right       = right,
        op          = op,
    )


# ── Subset-DP solver ─────────────────────────────────────────────────────────

def solve_input_list(numbers: list[int]) -> dict[int, Expr]:
    """
    Build dp[bitmask][value] = best Expr for that (input-index bitmask, value).
    Subsets grow size-by-size (1 → 6).  Only the best Expr per cell is kept,
    so dominated expressions never seed further combinations.

    Returns the globally best Expr per reachable answer value.
    """
    n = len(numbers)
    # dp keys are int bitmasks (bit i set ↔ input i used). 6 inputs → keys 0..63.
    dp: dict[int, dict[int, Expr]] = {
        1 << i: {v: _leaf(i, v)} for i, v in enumerate(numbers)
    }
    by_size: list[list[int]] = [[] for _ in range(n + 1)]
    for s in dp:
        by_size[1].append(s)

    def _update(bucket: dict, expr: Expr) -> None:
        v = expr.value
        if v not in bucket or expr.better_than(bucket[v]):
            bucket[v] = expr

    for size in range(2, n + 1):
        pending: dict[int, dict[int, Expr]] = {}
        for lsz in range(1, size):
            rsz = size - lsz
            if rsz < lsz:               # avoid double-counting symmetric pairs
                continue
            for ls in by_size[lsz]:
                for rs in by_size[rsz]:
                    if ls & rs:
                        continue
                    combined = ls | rs
                    bucket = pending.setdefault(combined, {})
                    le_list = list(dp[ls].values())
                    re_list = list(dp[rs].values())
                    for le in le_list:
                        for re in re_list:
                            for op in OPERATORS:
                                expr = _combine(le, re, op)
                                if expr: _update(bucket, expr)
                    # Non-commutative reversed role (only needed when sizes differ)
                    if lsz != rsz:
                        for re in re_list:
                            for le in le_list:
                                for op in ("-", "/"):
                                    expr = _combine(re, le, op)
                                    if expr: _update(bucket, expr)

        for sub, vdict in pending.items():
            if sub in dp:
                for v, expr in vdict.items():
                    _update(dp[sub], expr)
            else:
                dp[sub] = vdict
            by_size[size].append(sub)

    # Flatten to global best per value
    best: dict[int, Expr] = {}
    for vdict in dp.values():
        for v, expr in vdict.items():
            _update(best, expr)
    return best


# ── Worker (one per input list, runs in child process) ───────────────────────

def _worker_init() -> None:
    """Second-layer clamp: env vars already set at import time (before torch
    loaded); this re-applies torch.set_num_threads in case the worker
    process reinitialised torch internals after spawning."""
    if _HAS_TORCH:
        torch.set_num_threads(1)


def _score(gap: int) -> int:
    if gap == 0:   return 10
    if gap <= 5:   return 7
    if gap <= 10:  return 5
    return 0


def _worker(task: tuple) -> str:
    """Solve 900 targets for one input list; save shard atomically; return path."""
    shard_idx, numbers_tuple, shard_dir = task
    # Hash of the input tuple → filename stable across re-runs / reshuffles.
    _key  = hashlib.md5(str(numbers_tuple).encode()).hexdigest()[:12]
    final = os.path.join(shard_dir, f"shard_{_key}{FILE_EXT}")
    if os.path.exists(final):          # resumability: skip completed shards
        return final

    best = solve_input_list(list(numbers_tuple))
    reachable = sorted(best)
    render_cache: dict[int, tuple[str, list[str], int, int, int]] = {}

    records: list[dict] = []
    for target in range(TARGET_MIN, TARGET_MAX + 1):
        if target in best:
            expr, answer, gap = best[target], target, 0
        elif not reachable:
            records.append(dict(target=target, answer=None, gap=None,
                                inputs=numbers_tuple, expr=None, num_used=None,
                                num_ops=None, max_interm=None,
                                explanation=[], score=0))
            continue
        else:
            pos = bisect.bisect_left(reachable, target)
            cands = [reachable[p] for p in (pos, pos - 1)
                     if 0 <= p < len(reachable)]
            answer = min(cands, key=lambda v: (abs(v - target), v))
            gap    = abs(target - answer)
            expr   = best[answer]

        expr_id = id(expr)
        cached = render_cache.get(expr_id)
        if cached is None:
            cached = (
                expr.canon,
                _render_steps(expr),
                SUBSET_POPCOUNT[expr.inputs_used],
                expr.n_ops,
                expr.max_interm,
            )
            render_cache[expr_id] = cached
        expr_str, explanation, num_used, num_ops, max_interm = cached

        records.append(dict(
            target=target,
            answer=answer,
            gap=gap,
            inputs=numbers_tuple,
            expr=expr_str,
            num_used=num_used,
            num_ops=num_ops,
            max_interm=max_interm,
            explanation=list(explanation),
            score=_score(gap),
        ))

    # Atomic write: save to .tmp then rename (POSIX atomic)
    tmp = final + ".tmp"
    _pt_save(records, tmp)
    os.replace(tmp, final)
    return final


# ── Input-list enumeration ───────────────────────────────────────────────────

def enumerate_input_lists() -> list[tuple]:
    """All 13,243 sorted 6-number multisets from the Countdown pool."""
    seen: set[tuple] = set()
    for combo in itertools.combinations(range(len(COUNTDOWN_POOL)), 6):
        seen.add(tuple(sorted(COUNTDOWN_POOL[i] for i in combo)))
    return sorted(seen)


# ── I/O helpers ──────────────────────────────────────────────────────────────

def _pt_save(obj: Any, path: str) -> None:
    if _HAS_TORCH:
        torch.save(obj, path)
    else:
        with open(path, "wb") as f:
            _pickle.dump(obj, f, protocol=_pickle.HIGHEST_PROTOCOL)


def _pt_load(path: str) -> Any:
    if _HAS_TORCH:
        return torch.load(path, weights_only=False)
    with open(path, "rb") as f:
        return _pickle.load(f)


def load_dataset(path: str = "countdown_colton_dataset.pt") -> list[dict[str, Any]]:
    """
    Load the saved dataset into memory (merged file or sharded directory).
    For large datasets, prefer iter_dataset(...) to avoid high RAM use.

    Examples
    --------
    >>> records = load_dataset("./countdown_colton_dataset_sharded")  # bundled dir
    >>> records = load_dataset("./shards")                            # shard dir
    >>> r = records[0]
    >>> print(r["target"], r["expr"], r["explanation"])
    """
    if os.path.isdir(path):
        manifest = os.path.join(path, "manifest.json")
        if os.path.exists(manifest):
            with open(manifest) as f:
                names = json.load(f)["shards"]
            paths = [os.path.join(path, name) for name in names]
        else:
            paths = sorted(
                os.path.join(path, fn) for fn in os.listdir(path)
                if fn.startswith("shard_") and fn.endswith(FILE_EXT)
            )
        records: list[dict] = []
        for p in paths:
            records.extend(_pt_load(p))
        return records
    return _pt_load(path)

def iter_dataset(path: str = "countdown_colton_dataset.pt"):
    """
    Stream records from a sharded directory without materializing the full
    dataset in RAM. For a single monolithic file, the file itself is still
    loaded before iteration.

    Examples
    --------
    >>> for r in iter_dataset("./countdown_colton_dataset_sharded"):
    ...     print(r["target"], r["answer"])
    """
    if os.path.isdir(path):
        manifest = os.path.join(path, "manifest.json")
        if os.path.exists(manifest):
            with open(manifest) as f:
                names = json.load(f)["shards"]
            paths = [os.path.join(path, name) for name in names]
        else:
            paths = sorted(
                os.path.join(path, fn) for fn in os.listdir(path)
                if fn.startswith("shard_") and fn.endswith(FILE_EXT)
            )
        for p in paths:
            for record in _pt_load(p):
                yield record
    else:
        for record in _pt_load(path):
            yield record


def bundle_shards(shard_paths: list[str], out_path: str,
                  delete_shards: bool = False) -> str:
    """
    Create a manifest-based dataset bundle without concatenating all records
    into RAM.

    If `out_path` ends with `.pt`, we instead create a directory next to it:
        countdown_colton_dataset.pt  ->  countdown_colton_dataset_sharded/

    Returns the bundle directory path.
    """
    base = out_path
    if base.endswith(".pt"):
        base = os.path.splitext(base)[0] + "_sharded"

    os.makedirs(base, exist_ok=True)
    kept_paths: list[str] = []

    for src in sorted(shard_paths):
        dst = os.path.join(base, os.path.basename(src))
        if os.path.abspath(src) != os.path.abspath(dst):
            if not os.path.exists(dst):
                try:
                    os.link(src, dst)   # cheap if same filesystem
                except OSError:
                    import shutil
                    shutil.copy2(src, dst)
            if delete_shards:
                try:
                    os.remove(src)
                except OSError:
                    pass
        kept_paths.append(dst)

    manifest = os.path.join(base, "manifest.json")
    with open(manifest, "w") as f:
        json.dump({"shards": sorted(os.path.basename(p) for p in kept_paths)}, f, indent=2)

    print(f"[merge] manifest bundle with {len(kept_paths):,} shards → {base}")
    return base


def write_manifest(shard_dir: str, shard_paths: list[str]) -> str:
    """Write a manifest.json so load_dataset() can find shards without glob."""
    manifest = os.path.join(shard_dir, "manifest.json")
    with open(manifest, "w") as f:
        json.dump({"shards": sorted(os.path.basename(p) for p in shard_paths)}, f, indent=2)
    return manifest


# ── Summary statistics  (mirrors Colton Table 1) ─────────────────────────────

def print_summary(records: list[dict]) -> None:
    """In-memory summary helper (fine for demo/small runs)."""
    n = len(records)
    if not n:
        print("[summary] No records."); return
    s   = {10: 0, 7: 0, 5: 0, 0: 0}
    ez = med = hrd = dif = total_score = 0
    for r in records:
        sc = r["score"]
        s[sc] = s.get(sc, 0) + 1
        total_score += sc
        nu = r["num_used"]
        if nu is not None:
            if nu <= 3: ez += 1
            elif nu == 4: med += 1
            else: hrd += 1
        if (sc == 10 and nu == 6 and r["max_interm"] is not None
                and r["max_interm"] >= 1000 and r["num_ops"] == 4):
            dif += 1
    p   = lambda k: f"{100*k/n:.2f}%"
    print("\n--- Dataset Summary (cf. Colton 2014 Table 1) ---")
    for label, val in [("Total instances", n), ("Solvable (10 pts)", s[10]),
                       ("Solvable  (7 pts)", s[7]),  ("Solvable  (5 pts)", s[5]),
                       ("Unsolvable (0 pts)", s[0])]:
        pct = f"  ({p(val)})" if label != "Total instances" else ""
        print(f"  {label:<20}: {val:>12,}{pct}")
    print(f"  {'Expected score':<20}: {total_score/n:>12.4f}")
    for label, val in [("Easy   (U≤3)", ez), ("Medium (U=4)", med),
                       ("Hard   (U≥5)", hrd), ("Difficult", dif)]:
        print(f"  {label:<20}: {val:>12,}  ({p(val)})")
    print("-" * 49)

def print_summary_stream(records_iter) -> None:
    """Streaming summary over any record iterator."""
    n = 0
    s = {10: 0, 7: 0, 5: 0, 0: 0}
    ez = med = hrd = dif = total_score = 0

    for r in records_iter:
        n += 1
        sc = r["score"]
        s[sc] = s.get(sc, 0) + 1
        total_score += sc
        nu = r["num_used"]
        if nu is not None:
            if nu <= 3:
                ez += 1
            elif nu == 4:
                med += 1
            else:
                hrd += 1
        if (sc == 10 and nu == 6 and r["max_interm"] is not None
                and r["max_interm"] >= 1000 and r["num_ops"] == 4):
            dif += 1

    if n == 0:
        print("[summary] No records.")
        return

    p = lambda k: f"{100*k/n:.2f}%"
    print("\n--- Dataset Summary (cf. Colton 2014 Table 1) ---")
    for label, val in [("Total instances", n), ("Solvable (10 pts)", s[10]),
                       ("Solvable  (7 pts)", s[7]),  ("Solvable  (5 pts)", s[5]),
                       ("Unsolvable (0 pts)", s[0])]:
        pct = f"  ({p(val)})" if label != "Total instances" else ""
        print(f"  {label:<20}: {val:>12,}{pct}")
    print(f"  {'Expected score':<20}: {total_score/n:>12.4f}")
    for label, val in [("Easy   (U≤3)", ez), ("Medium (U=4)", med),
                       ("Hard   (U≥5)", hrd), ("Difficult", dif)]:
        print(f"  {label:<20}: {val:>12,}  ({p(val)})")
    print("-" * 49)

# ── Progress helper ───────────────────────────────────────────────────────────
 
def _update_tqdm_postfix(pbar, done: int, t0: float, recent_times: list[float]) -> None:
    elapsed = time.perf_counter() - t0
    puzzles_done = done * 900
    puzzles_per_s = puzzles_done / elapsed if elapsed else 0.0
    med_ms = (
        sorted(recent_times)[len(recent_times)//2] * 1000
        if recent_times else float("nan")
    )
    pbar.set_postfix_str(
        f"{puzzles_done:,} puzzles | {puzzles_per_s:.0f} p/s | med {med_ms:.0f} ms/list",
        refresh=False,
    )

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    pa = argparse.ArgumentParser(
        description="Generate the Colton-style Countdown Numbers Game dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    _default_w = int(os.environ.get("COUNTDOWN_WORKERS", DEFAULT_WORKERS))
    pa.add_argument("--workers",      type=int, default=_default_w,
                    help=f"Worker processes (default: {_default_w}; env: COUNTDOWN_WORKERS).")
    pa.add_argument(
        "--out",
        type=str,
        default=OUTPUT_BUNDLE,
        help="Output bundle path. This script creates a directory containing "
             "shard files plus manifest.json. If --out ends with .pt, the "
             "actual output will be a sibling directory named *_sharded. "
             f"(default: {OUTPUT_BUNDLE})"
    )
    pa.add_argument("--shard-dir",    type=str, default=OUTPUT_SHARDS,
                    help=f"Dir for per-list shards (default: {OUTPUT_SHARDS}).")
    pa.add_argument("--keep-shards",  action="store_true")
    pa.add_argument("--no-merge",     action="store_true",
                    help="Leave shards in --shard-dir; write manifest.json.")
    pa.add_argument("--input-list",   type=int, nargs=6,
                    metavar=("a","b","c","d","e","f"),
                    help="Demo: solve one sorted 6-number list.")
    pa.add_argument("--limit",        type=int, default=None,
                    help="Process only first K input lists.")
    pa.add_argument("--maxtasks",     type=int, default=25,
                    help="maxtasksperchild for worker recycling (default: 25).")
    args = pa.parse_args()

    # ── Demo mode ────────────────────────────────────────────────────────────
    if args.input_list:
        nums = tuple(sorted(args.input_list))
        print(f"[demo] {list(nums)}")
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as td:
            shard = _worker((0, nums, td))
            records = _pt_load(shard)
        print(f"[demo] {len(records)} targets in {time.perf_counter()-t0:.3f}s\n")
        for r in records[:5]:
            print(f"  T={r['target']:3d}  A={r['answer']}  gap={r['gap']}  "
                  f"score={r['score']}  U={r['num_used']}  "
                  f"L={r['max_interm']}  N={r['num_ops']}")
            print(f"    {r['expr']}")
            print(f"    {r['explanation']}")
        demo_out = args.out
        if not os.path.splitext(demo_out)[1]:
            os.makedirs(demo_out, exist_ok=True)
            demo_out = os.path.join(demo_out, os.path.basename(DEMO_OUTPUT))
        else:
            out_dir = os.path.dirname(demo_out)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
        _pt_save(records, demo_out)
        print(f"\n[demo] → {demo_out}")
        print_summary(records)
        return

    # ── Full / limited run ───────────────────────────────────────────────────
    input_lists = enumerate_input_lists()
    random.seed(SHARD_SEED)               # deterministic → stable shard names
    random.shuffle(input_lists)           # mirror Colton's random scheduling
    if args.limit:
        input_lists = input_lists[:args.limit]  # slice AFTER shuffle → random sample

    total = len(input_lists)
    print(f"[info] Input lists  : {total:,}  |  Puzzles: {total*900:,}  "
          f"|  Workers: {args.workers}  |  maxtasks: {args.maxtasks}")

    shard_dir = args.shard_dir
    os.makedirs(shard_dir, exist_ok=True)
    print(f"[info] Shard dir    : {shard_dir}\n")

    tasks         = [(i, il, shard_dir) for i, il in enumerate(input_lists)]
    shard_paths:  list[str]   = []
    recent_times: list[float] = []      # sliding window for median ms/list
    done = 0
    t_start = time.perf_counter()

    if args.workers == 1:
        with tqdm(
            total=total,
            desc="Input lists",
            unit="list",
            dynamic_ncols=True,
            mininterval=0.5,
            maxinterval=2.0,
            smoothing=0.05,
        ) as pbar:
            for task in tasks:
                t1 = time.perf_counter()
                shard_paths.append(_worker(task))
                recent_times.append(time.perf_counter() - t1)
                if len(recent_times) > 50:
                    recent_times.pop(0)
                done += 1
                pbar.update(1)
                _update_tqdm_postfix(pbar, done, t_start, recent_times)
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=args.workers, initializer=_worker_init,
                      maxtasksperchild=args.maxtasks) as pool:
            with tqdm(
                total=total,
                desc="Input lists",
                unit="list",
                dynamic_ncols=True,
                mininterval=0.5,
                maxinterval=2.0,
                smoothing=0.05,
            ) as pbar:
                for sp in pool.imap_unordered(_worker, tasks, chunksize=1):
                    t_now = time.perf_counter()
                    shard_paths.append(sp)
                    # approximate per-task time from overall throughput
                    recent_times.append((t_now - t_start) / (done + 1))
                    if len(recent_times) > 50:
                        recent_times.pop(0)
                    done += 1
                    pbar.update(1)
                    _update_tqdm_postfix(pbar, done, t_start, recent_times)


    elapsed = time.perf_counter() - t_start
    print(f"\n[info] Done: {elapsed:.1f}s  "
          f"({elapsed/total*1000:.1f}ms/list  "
          f"{elapsed/(total*900)*1000:.3f}ms/puzzle  "
          f"{total*900/elapsed:.0f} puzzles/s)")

    if args.no_merge:
        write_manifest(shard_dir, shard_paths)
        print_summary_stream(iter_dataset(shard_dir))
        return

    final_path = bundle_shards( 
        shard_paths,
        args.out,
        delete_shards=not args.keep_shards,
    )

    print_summary_stream(iter_dataset(final_path))


if __name__ == "__main__":
    main()