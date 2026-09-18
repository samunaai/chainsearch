#!/usr/bin/env python3
"""
generate_astar_dataset.py
==========================
Generate A* Countdown trace datasets and save them to .pt.

This file owns the *data contract* used by both generation and training:
  - prompt format:  IN: [ ... ] TGT: n
  - trace format:   [before] -> a op b = c -> [after] ; BACK -> [state] ; SUCCESS ; EXPR (...)
  - tokenizer / token-counting logic used to enforce trace-length caps

It supports:
  - A* traces only

The saved dataset format is:
    {
        "prompts": [...],
        "traces":  [...],
        "vocab":   [...],
        "config":  {...},
        "format_version": 2,
        "trace_mode": "astar",
    }
"""

import os
# Set thread env vars *before* importing torch for maximum reliability.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import re
import time
import math
import random
import logging
from pathlib import Path
from contextlib import contextmanager
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from heapq import heappush, heappop
from itertools import combinations
from multiprocessing import cpu_count, get_context
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

# Headless matplotlib backend to avoid server-side crashes if later imported elsewhere.
import matplotlib
os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use(os.environ["MPLBACKEND"])

# Line-buffered logs when redirected.
if not sys.stdout.isatty():
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
        sys.stderr.reconfigure(line_buffering=True, write_through=True)
    except AttributeError:
        sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
        sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)


# =========================
# CONFIGURATION
# =========================
TARGET_RANGES = {
    3: (11, 300),
    4: (11, 750),
    5: (11, 900),
    6: (100, 999),
    7: (100, 999),
    8: (100, 999),
}

BASE_PARAMS = dict(
    dataset_path=None,
    artifacts_dir="/workspace/sam/minireason_archives/data_pts",
    log_dir=os.getcwd(),
    seed=42,
)

DATA_PARAMS = dict(
    trace_mode="astar",
    small_deck=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10],
    large_deck=[25, 50, 75, 100],
    num_samples=None,
    num_input_digits_list=[6],
    target_range=TARGET_RANGES,
    slice_size=50_000,
    data_gen_workers=64,
    stats_topk=1000,
)
 

ASTAR_PARAMS = dict(
    astar_max_expansions=2500,
    astar_max_trace_tokens=10000,
    astar_max_focus_switches=8,
    astar_stick_steps=0,
    compact_steps=False,
    use_h2=False,
    h2_use_all_ops=False,
    h2_softcap_k_factor=1.0,
)

DATA_HPARAMS = BASE_PARAMS | DATA_PARAMS | ASTAR_PARAMS


def _env(k: str) -> Optional[str]:
    v = os.environ.get(k)
    return v if v is not None and v != "" else None


# =========================
# LOGGING / IO HELPERS
# =========================
@contextmanager
def attach_file_logger(path: str, mode: str = "w"):
    class FlushHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    root = logging.getLogger()
    fh = FlushHandler(path, mode=mode)
    fh.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(fh)
    try:
        yield fh
    finally:
        root.removeHandler(fh)
        fh.close()


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_torch_save(obj, path, retries=3, legacy_fallback=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    last_err = None
    for i in range(retries):
        try:
            with open(tmp, "wb") as f:
                torch.save(obj, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            return
        except Exception as e:
            last_err = e
            time.sleep(0.2 * (i + 1))
    if legacy_fallback:
        with open(tmp, "wb") as f:
            torch.save(obj, f, _use_new_zipfile_serialization=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return
    raise last_err


# =========================
# TOKENIZER / SHARED DATA CONTRACT
# =========================
class Tokenizer:
    def __init__(self):
        self.special = [
            'PAD', 'EOS', 'SEP', 'IN:', 'TGT:', '[', ']', '(', ')', ',', '=', ';', '->', '+', '-', '*', '/', 'BACK', 'SUCCESS'
        ]
        self.cmds = ['EXPR']
        self.digits = [str(d) for d in range(10)]
        vocab = self.special + self.cmds + self.digits
        self.tok2id = {t: i for i, t in enumerate(vocab)}
        self.id2tok = {i: t for i, t in enumerate(vocab)}
        self._pad = self.tok2id['PAD']
        self._eos = self.tok2id['EOS']
        self._sep = self.tok2id['SEP']
        self._tok_re = re.compile(r'(->|\[|\]|\(|\)|,|=|;|\+|\-|\*|/)')

    @property
    def pad_id(self):
        return self._pad

    @property
    def eos_id(self):
        return self._eos

    @property
    def sep_id(self):
        return self._sep

    @property
    def vocab_size(self):
        return len(self.tok2id)

    def encode(self, text, max_len):
        spaced_text = self._tok_re.sub(r' \1 ', text)
        tokens = spaced_text.split()
        flat_tokens = []
        for tok in tokens:
            if tok.isdigit() and len(tok) > 1:
                flat_tokens.extend(list(tok))
            else:
                flat_tokens.append(tok)
        flat_tokens.append('EOS')
        ids = [self.tok2id.get(t) for t in flat_tokens]
        if any(i is None for i in ids):
            raise ValueError(f"Unknown token in text: '{text}' from tokens '{flat_tokens}'")
        if len(ids) < max_len:
            ids += [self._pad] * (max_len - len(ids))
        return ids[:max_len]

    def decode(self, ids):
        tokens = [self.id2tok.get(i, '?') for i in ids if i not in (self._pad, self._eos)]
        result = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.isdigit():
                num_str = token
                i += 1
                while i < len(tokens) and tokens[i].isdigit():
                    num_str += tokens[i]
                    i += 1
                result.append(num_str)
            else:
                result.append(token)
                i += 1
        s = " ".join(result)
        s = re.sub(r"\s*,\s*", ", ", s)
        s = re.sub(r"\[\s*", "[ ", s)
        s = re.sub(r"\s*\]", " ]", s)
        s = re.sub(r"\(\s*", "(", s)
        s = re.sub(r"\s*\)", ")", s)
        s = re.sub(r"\s*->\s*", " -> ", s)
        s = re.sub(r"\s*=\s*", " = ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s


def encode_ids(tok: Tokenizer, text: str) -> list[int]:
    spaced_text = tok._tok_re.sub(r' \1 ', text)
    tokens = spaced_text.split()
    flat = []
    for t in tokens:
        if t.isdigit() and len(t) > 1:
            flat.extend(list(t))
        else:
            flat.append(t)
    flat.append('EOS')
    return [tok.tok2id[t] for t in flat]


def token_count(tok: Tokenizer, text: str, add_eos: bool = True) -> int:
    ids = encode_ids(tok, text)
    if not add_eos and ids and ids[-1] == tok.eos_id:
        ids = ids[:-1]
    return len(ids)


def parse_initial_numbers(prompt_text: str) -> Optional[List[int]]:
    match = re.search(r"IN:\s*\[\s*(.*?)\s*\]", prompt_text)
    if match:
        parts = re.split(r"[,\s]+", match.group(1).strip())
        return [int(n) for n in parts if n]
    return None


# =========================
# PATH / CONFIG HELPERS
# =========================   

def build_cache_path(cfg):
    ns_val = cfg.get("num_samples")
    ns_str = str(ns_val) if isinstance(ns_val, int) else "ALL"

    nlist = cfg.get("num_input_digits_list") or []
    if nlist:
        lo, hi = min(nlist), max(nlist)
        n_part = f"{lo}-{hi}" if len(set(nlist)) == (hi - lo + 1) else ",".join(map(str, nlist))
    else:
        n_part = "NA"

    tr = cfg["target_range"]
    if isinstance(tr, tuple):
        tr_part = f"T{int(tr[0])}to{int(tr[1])}_"
    elif isinstance(tr, dict):
        keys = cfg.get("num_input_digits_list") or sorted(tr)
        missing = [k for k in keys if k not in tr]
        if missing:
            raise ValueError(f"target_range missing keys for n={missing}")
        tr_part = "T{" + "|".join(f"{k}:{tr[k][0]}-{tr[k][1]}" for k in keys) + "}_"
    else:
        raise ValueError("target_range must be a tuple or dict")    

    compact = 1 if cfg.get("compact_steps") else 0
    base = (
        "astar_data_"
        f"N{ns_str}_S{n_part}_"
        f"{tr_part}"
        f"exp{int(cfg.get('astar_max_expansions', 0))}_"
        f"trace{int(cfg.get('astar_max_trace_tokens', 0))}_"
        f"sw{int(cfg.get('astar_max_focus_switches', 0))}_"
        f"stick{int(cfg.get('astar_stick_steps', 0))}_"
        f"compact{compact}_"
        f"useh2{1 if cfg.get('use_h2') else 0}_"
        f"h2all{1 if cfg.get('h2_use_all_ops') else 0}_"
        f"h2k{float(cfg.get('h2_softcap_k_factor', 1.0))}_"
        f"seed{int(cfg['seed'])}_fmtv2.pt"
    )
    return os.path.join(cfg["artifacts_dir"], base)


def validate_data_config(cfg: dict) -> None:
    mode = str(cfg.get("trace_mode", "")).lower()
    if mode not in {"astar", "targeted"}:
        raise ValueError(f"Only A* mode is supported, got trace_mode={mode!r}")
    n_list = cfg.get("num_input_digits_list")
    if not isinstance(n_list, list) or not n_list:
        raise ValueError("num_input_digits_list must be a non-empty list.")
    tr = cfg.get("target_range")
    if not isinstance(tr, (tuple, dict)):
        raise ValueError("target_range must be a tuple or dict.")
    if cfg.get("dataset_path") is not None and not isinstance(cfg.get("dataset_path"), str):
        raise ValueError("dataset_path must be a string path or None.")


# =========================
# DATA WORKER HELPERS
# =========================
WORKER_TOK = None


def _pool_init(num_threads: int = 1):
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    try:
        torch.set_num_threads(num_threads)
    except Exception:
        pass
    global WORKER_TOK
    WORKER_TOK = Tokenizer()


@contextmanager
def spawn_pool(n_workers: int, *, init_threads: int = 1, maxtasksperchild: int = 200):
    ctx = get_context("spawn")
    with ctx.Pool(
        processes=int(max(1, n_workers)),
        initializer=_pool_init,
        initargs=(init_threads,),
        maxtasksperchild=maxtasksperchild,
    ) as pool:
        yield pool


def _multiset_combinations(counter, r):
    items = sorted(counter.items())
    comb = []

    def dfs(i, left):
        if left == 0:
            yield tuple(sorted(x for v, c in comb for x in [v] * c))
            return
        if i >= len(items):
            return
        v, maxc = items[i]
        for c in range(0, min(maxc, left) + 1):
            if c:
                comb.append((v, c))
            yield from dfs(i + 1, left - c)
            if c:
                comb.pop()

    yield from dfs(0, r)


def _enumerate_unique_number_multisets(small_deck, large_deck, n):
    import itertools
    small_ctr = Counter(small_deck)
    large_vals = sorted(set(large_deck))
    out = set()
    max_large = min(n, len(large_vals))
    for L in range(0, max_large + 1):
        S = n - L
        large_part = list(itertools.combinations(large_vals, L))
        small_parts = [()] if S == 0 else list(_multiset_combinations(small_ctr, S))
        for sp in small_parts:
            for lp in large_part:
                tup = tuple(sorted(sp + lp))
                if len(tup) == n:
                    out.add(tup)
    return sorted(out)


def _estimate_total_prompts(numbers_sets, tgt_lo, tgt_hi):
    lo, hi = int(tgt_lo), int(tgt_hi)
    per_ns = hi - lo + 1
    total = 0
    for ns in numbers_sets:
        bad = 0
        if len(ns) > 1:
            for v in set(ns):
                if lo <= v <= hi:
                    bad += 1
        total += (per_ns - bad)
    return total


def _iter_prompt_slices(numbers_sets, tgt_lo, tgt_hi, *, seed: int, slice_size: int = 50000):
    lo, hi = int(tgt_lo), int(tgt_hi)
    ns_list = list(numbers_sets)
    rnd = random.Random(seed)
    rnd.shuffle(ns_list)
    for idx, ns in enumerate(ns_list):
        tgts = list(range(lo, hi + 1))
        rnd2 = random.Random(seed + 17 * idx)
        rnd2.shuffle(tgts)
        buf = []
        for tgt in tgts:
            if len(ns) > 1 and tgt in ns:
                continue
            buf.append((ns, tgt))
            if len(buf) >= slice_size:
                yield buf
                buf = []
        if buf:
            yield buf


def _build_canonical_prompt(numbers_tuple, target):
    nums = ", ".join(map(str, numbers_tuple))
    return f"IN: [ {nums} ] TGT: {target}"


# =========================
# A* SOLVER
# =========================
class AStarSolver:
    def __init__(self, cfg: Optional[dict] = None, tok: Optional[Tokenizer] = None):
        self._tok = tok or Tokenizer()
        self.cfg = cfg or {}

    def _count_model_toks(self, s: str) -> int:
        return token_count(self._tok, s, add_eos=False)

    class Node:
        __slots__ = ("state", "parent", "op", "g", "h", "f")

        def __init__(self, state, parent=None, op=None, g=0, h=0):
            self.state = state
            self.parent = parent
            self.op = op
            self.g, self.h = g, h
            self.f = g + h

        def __lt__(self, other):
            return (self.f, self.h, len(self.state)) < (other.f, other.h, len(other.state))

    def _heuristic_cost(
        self,
        state: Tuple[int, ...],
        result_of_last_op: Optional[int],
        T: int,
        prev_best_gap: Optional[int] = None,
    ) -> float:
        if not state:
            return 1e9
        vals = list(state)
        max_for_scaling = T + 1

        h1_abs = min(abs(v - T) for v in vals)
        if h1_abs == 0:
            return 0.0
        h1 = min(h1_abs, max_for_scaling) / max_for_scaling

        if bool(self.cfg.get("use_h2", False)):
            use_all_ops = bool(self.cfg.get("h2_use_all_ops", False))
            best = None
            for a, b in combinations(vals, 2):
                cand = []
                if use_all_ops:
                    cand.append(a + b)
                    if a > b:
                        cand.append(a - b)
                    if b > a:
                        cand.append(b - a)
                    if a > 1 and b > 1:
                        cand.append(a * b)
                    big, small = (a, b) if a > b else (b, a)
                    if small != 0 and big % small == 0:
                        cand.append(big // small)
                else:
                    if a > 1 and b > 1:
                        cand.append(a * b)
                if cand:
                    g = min(abs(x - T) for x in cand)
                    best = g if best is None or g < best else best
            if best is not None:
                k_factor = float(self.cfg.get("h2_softcap_k_factor", 1.0))
                k_soft = max(1.0, k_factor * max_for_scaling)
                h2 = best / (best + k_soft)
            else:
                h2 = 0.0
        else:
            h2 = 0.0

        cnt = Counter(vals)
        ready = False
        partial = False
        for x, freq in cnt.items():
            if x > 1 and T % x == 0:
                q = T // x
                if q != 1 and ((q in cnt) and (q != x or cnt[q] >= 2)):
                    ready = True
                    break
                partial = True
        h3 = 0.0 if ready else (0.5 if partial else 1.0)

        r = result_of_last_op
        h4 = 0.0
        if r is not None and r > 1 and abs(r - T) > 25:
            smooth = (r % 2 == 0) + (r % 3 == 0) + (r % 5 == 0)
            h4 = 0.0 if smooth >= 2 else 1.0

        h5 = 0.0
        if prev_best_gap is not None:
            STEP = 3.0
            improvement = max(0.0, prev_best_gap - h1_abs)
            h5 = max(0.0, min(1.0, (STEP - improvement) / STEP))

        return float(h1 + h2 + h3 + h4 + h5 + 1e-3 * (len(state) - 1))

    def _succ(self, state: Tuple[int, ...]):
        S = list(state)
        n = len(S)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = S[j], S[i]
                yield (b, '+', a, a + b)
                if a > 1 and b > 1:
                    yield (b, '*', a, a * b)
                if a > b:
                    yield (a, '-', b, a - b)
                if b != 0 and a % b == 0:
                    yield (a, '/', b, a // b)

    def _reconstruct(self, node):
        path = []
        cur = node
        while cur:
            path.append(cur)
            cur = cur.parent
        return list(reversed(path))

    def _lcp_len(self, A, B):
        k = 0
        while k < min(len(A), len(B)) and A[k].state == B[k].state:
            k += 1
        return k

    def _S(self, tup):
        return "[ " + ", ".join(map(str, tup)) + " ]"

    def _expr_from_ops(self, initial_numbers: List[int], ops: List[Tuple[int, str, int, int]]) -> str:
        pool = [(x, str(x)) for x in initial_numbers]

        def take_pair(val):
            for i, (v, e) in enumerate(pool):
                if v == val:
                    return pool.pop(i)
            raise AssertionError(f"value {val} not found while building EXPR")

        for a, sym, b, c in ops:
            va, ea = take_pair(a)
            vb, eb = take_pair(b)
            expr = f"({ea} {sym} {eb})"
            if sym == '+':
                vc = va + vb
            elif sym == '-':
                vc = va - vb
            elif sym == '*':
                vc = va * vb
            else:
                vc = va // vb
            pool.append((vc, expr))
        return pool[-1][1] if pool else ""

    def _emit_descend(self, trace: List[str], parent_state, child_op, child_state, *, last_after_state, compact: bool) -> str:
        a, sym, b, c = child_op
        if compact and last_after_state is not None and last_after_state == parent_state:
            line = f"{a} {sym} {b} = {c} -> {self._S(child_state)} ;"
        else:
            line = f"{self._S(parent_state)} -> {a} {sym} {b} = {c} -> {self._S(child_state)} ;"
        trace.append(line)
        return line

    def _emit_back_to(self, trace: List[str], lca_state, need_back: bool) -> Optional[str]:
        if need_back:
            line = f"BACK -> {self._S(lca_state)} ;"
            trace.append(line)
            return line
        return None

    def solve(
        self,
        numbers: List[int],
        T: int,
        max_expansions: int = 400,
        max_trace_tokens: int = 1600,
        max_focus_switches: int = 10,
        stick_steps: int = 0,
    ) -> Optional[str]:
        start = tuple(sorted(numbers))
        if T in start:
            return f"{self._S(start)} ; SUCCESS ; EXPR ({T})"
        root = AStarSolver.Node(start, None, None, g=0, h=self._heuristic_cost(start, None, T))
        root.f = root.g + root.h
        open_pq = [(root.f, id(root), root)]
        best_g = {root.state: 0}
        trace: List[str] = []
        prev_focus = [root]
        expansions = 0
        last_after = None
        stick_remaining = max(0, stick_steps)
        switches = 0
        trace_token_count = 0

        while open_pq and expansions < max_expansions:
            emitted_any_step = False
            f0, i0, n = heappop(open_pq)
            path = self._reconstruct(n)
            lca_len = self._lcp_len(prev_focus, path)
            is_switch = (len(prev_focus) > 0 and lca_len < len(prev_focus))
            if stick_remaining > 0 and is_switch:
                temp = []
                alt = None
                while open_pq:
                    f1, i1, cand = heappop(open_pq)
                    cand_path = self._reconstruct(cand)
                    if self._lcp_len(prev_focus, cand_path) == len(prev_focus):
                        alt = (f1, i1, cand)
                        break
                    temp.append((f1, i1, cand))
                for it in temp:
                    heappush(open_pq, it)
                if alt is not None:
                    heappush(open_pq, (f0, i0, n))
                    f0, i0, n = alt
                    path = self._reconstruct(n)
                    lca_len = self._lcp_len(prev_focus, path)
                    is_switch = False
            expansions += 1

            path = self._reconstruct(n)
            lca_len = self._lcp_len(prev_focus, path)
            lca_idx = max(0, lca_len - 1)
            need_back = (len(prev_focus) - 1) > lca_idx
            if not trace and need_back:
                need_back = False
            if need_back:
                switches += 1
                if switches > max_focus_switches:
                    return None
                back_line = self._emit_back_to(trace, path[lca_idx].state, True)
                if back_line:
                    trace_token_count += self._count_model_toks(back_line)
                last_after = path[lca_idx].state

            start_k = lca_idx
            for k in range(start_k, len(path) - 1):
                parent = path[k]
                child = path[k + 1]
                line = self._emit_descend(
                    trace,
                    parent.state,
                    child.op,
                    child.state,
                    last_after_state=last_after,
                    compact=bool(self.cfg.get("compact_steps", False)),
                )
                last_after = child.state
                emitted_any_step = True
                trace_token_count += self._count_model_toks(line)
            prev_focus = path

            if any(v == T for v in n.state):
                ops = [p.op for p in path[1:]]
                expr = self._expr_from_ops(list(numbers), ops)
                if not emitted_any_step and len(path) > 1:
                    for k in range(0, len(path) - 1):
                        parent = path[k]
                        child = path[k + 1]
                        line2 = self._emit_descend(
                            trace,
                            parent.state,
                            child.op,
                            child.state,
                            last_after_state=last_after,
                            compact=bool(self.cfg.get("compact_steps", False)),
                        )
                        last_after = child.state
                        trace_token_count += self._count_model_toks(line2)
                line = f"SUCCESS ; EXPR ({expr})"
                trace.append(line)
                trace_token_count += self._count_model_toks(line)
                out = " ".join(trace)
                return out if trace_token_count <= max_trace_tokens else None

            if need_back:
                stick_remaining = max(0, stick_steps)
            else:
                if stick_remaining > 0:
                    stick_remaining -= 1

            if trace_token_count > max_trace_tokens:
                return None

            parent_best_gap = min(abs(v - T) for v in n.state)
            for (a, sym, b, c) in self._succ(n.state):
                S = list(n.state)
                ia = S.index(a)
                S.pop(ia)
                if a == b:
                    S.pop(S.index(b))
                else:
                    ib = S.index(b)
                    S.pop(ib)
                S.append(c)
                next_state = tuple(sorted(S))
                g2 = n.g + 1
                if best_g.get(next_state, 1e9) <= g2:
                    continue
                hcost = self._heuristic_cost(next_state, c, T, prev_best_gap=parent_best_gap)
                child = AStarSolver.Node(next_state, n, (a, sym, b, c), g2, hcost)
                best_g[next_state] = g2
                heappush(open_pq, (child.f, id(child), child))
        return None


# =========================
# A* DATASET GENERATION
# =========================
def _solve_prompt_slice(args):
    prompts_slice, seed, cfg = args
    random.seed(seed)
    solver = AStarSolver(cfg=cfg)
    ins, outs = [], []
    rejects_solver = attempts = successful = 0
    for numbers, tgt in prompts_slice:
        attempts += 1
        if len(numbers) > 1 and tgt in numbers:
            rejects_solver += 1
            continue
        trace = solver.solve(
            list(numbers),
            tgt,
            max_expansions=int(cfg.get("astar_max_expansions", 2500)),
            max_trace_tokens=int(cfg.get("astar_max_trace_tokens", 10000)),
            max_focus_switches=int(cfg.get("astar_max_focus_switches", 8)),
            stick_steps=int(cfg.get("astar_stick_steps", 0)),
        )
        if not trace:
            rejects_solver += 1
            continue
        successful += 1
        ins.append(_build_canonical_prompt(numbers, tgt))
        outs.append(trace)
    return ins, outs, attempts, rejects_solver, successful, None


def generate_countdown_data(config, n_workers=None):
    num_samples = config['num_samples']
    num_input_digits = config['num_input_digits']
    seed = config['seed']
    small_deck, large_deck = config['small_deck'], config['large_deck']
    tgt_lo, tgt_hi = config['target_range']

    chosen = n_workers or config.get("data_gen_workers")
    if chosen is None:
        chosen = os.cpu_count() or cpu_count()
    n_workers = int(max(1, chosen))

    numbers_sets = _enumerate_unique_number_multisets(small_deck, large_deck, num_input_digits)
    total_prompts = _estimate_total_prompts(numbers_sets, tgt_lo, tgt_hi)
    if num_samples is None:
        num_samples = int(total_prompts)
    slice_size = int(config["slice_size"])
    if not numbers_sets:
        return [], []

    logging.info(f"Enumerated {len(numbers_sets):,} unique number-sets; total prompts: {total_prompts:,}.")
    logging.info(f"Starting unique-prompt solving on {n_workers} workers (streamed shards)...")

    ins, outs = [], []
    total_attempts = total_rej_s = total_success = 0

    def _tasks_iter():
        for i, sl in enumerate(_iter_prompt_slices(numbers_sets, tgt_lo, tgt_hi, seed=seed, slice_size=slice_size)):
            yield (sl, seed + 1000 * i, config)

    with spawn_pool(n_workers) as pool:
        total_slices = len(numbers_sets)
        pbar_shards = tqdm(total=total_slices, desc="Shards", dynamic_ncols=True, miniters=1)
        pbar_solved = tqdm(total=num_samples, desc="Solved", dynamic_ncols=True, miniters=1)
        for ins_chunk, outs_chunk, a, rs, succ, err in pool.imap_unordered(_solve_prompt_slice, _tasks_iter(), chunksize=1):
            pbar_shards.update(1)
            if err:
                tqdm.write(f"Data worker error: {err}")
                continue
            room = max(0, num_samples - len(ins))
            if room > 0:
                take = min(room, len(ins_chunk))
                ins.extend(ins_chunk[:take])
                outs.extend(outs_chunk[:take])
                pbar_solved.update(take)
            total_attempts += a
            total_rej_s += rs
            total_success += succ
            if total_attempts:
                hit = 100.0 * total_success / total_attempts
                pbar_solved.set_postfix(attempts=total_attempts, hit=f"{hit:.1f}%")
            if len(ins) >= num_samples:
                pbar_shards.close()
                pbar_solved.close()
                pool.terminate()
                break
        else:
            pbar_shards.close()
            pbar_solved.close()

    logging.info(f"Collected {len(ins):,} solved unique prompts out of {total_attempts:,} attempts.")
    if len(ins) < num_samples:
        logging.warning(f"Underfilled: {len(ins):,}/{num_samples:,} solved unique prompts available.")
    return ins, outs

 
# =========================
# DATASET STATS / LOGGING
# =========================
RE_STEP = re.compile(r"(?:\[\s*[^\]]*?\s*\]\s*->\s*)?\d+\s*[+\-*/]\s*\d+\s*=\s*\d+\s*->\s*\[\s*[^\]]*?\s*\]\s*;")
RE_BACK = re.compile(r"BACK\s*->\s*\[\s*[^\]]*?\s*\]\s*;")
RE_PARTS = re.compile(r"(\d+)\s*([+\-*/])\s*(\d+)\s*=\s*(\d+)")


def _trace_stats_one(tr: str) -> tuple[int, int, int, int, int, int]:
    backs = len(RE_BACK.findall(tr))
    plus = minus = times = div = 0
    for m in RE_STEP.finditer(tr):
        seg = m.group(0)
        m2 = RE_PARTS.search(seg)
        if not m2:
            continue
        op = m2.group(2)
        if op == '+':
            plus += 1
        elif op == '-':
            minus += 1
        elif op == '*':
            times += 1
        elif op == '/':
            div += 1
    hops = plus + minus + times + div + backs
    return hops, backs, plus, minus, times, div


@dataclass
class _DatasetStatsAgg:
    tok: Tokenizer
    top_k: int = 1000
    n: int = 0
    total_tokens: int = 0
    hop_min: int = 10**18
    hop_max: int = -10**18
    hop_sum: float = 0.0
    hop_sumsq: float = 0.0
    bk_min: int = 10**18
    bk_max: int = -10**18
    bk_sum: float = 0.0
    bk_sumsq: float = 0.0
    hop_hist: Counter = None
    bk_hist: Counter = None
    op_ctr: Counter = None

    def __post_init__(self):
        if self.hop_hist is None:
            self.hop_hist = Counter()
        if self.bk_hist is None:
            self.bk_hist = Counter()
        if self.op_ctr is None:
            self.op_ctr = Counter({'+': 0, '-': 0, '*': 0, '/': 0})

    def update(self, prompt: str, trace: str):
        hops, backs, plus, minus, times, div = _trace_stats_one(trace)
        self.n += 1
        self.hop_hist[hops] += 1
        self.bk_hist[backs] += 1
        self.op_ctr['+'] += plus
        self.op_ctr['-'] += minus
        self.op_ctr['*'] += times
        self.op_ctr['/'] += div
        self.total_tokens += token_count(self.tok, f"{prompt} SEP {trace}", add_eos=True)
        self.hop_min = min(self.hop_min, hops)
        self.hop_max = max(self.hop_max, hops)
        self.hop_sum += hops
        self.hop_sumsq += float(hops) * float(hops)
        self.bk_min = min(self.bk_min, backs)
        self.bk_max = max(self.bk_max, backs)
        self.bk_sum += backs
        self.bk_sumsq += float(backs) * float(backs)

    def log(self, *, label: str = ""):
        if self.n <= 0:
            logging.info("Dataset stats: no steps parsed.")
            return
        n = self.n
        mean = self.hop_sum / n
        var = (self.hop_sumsq / n) - (mean * mean)
        bk_mean = self.bk_sum / n
        bk_var = (self.bk_sumsq / n) - (bk_mean * bk_mean)
        len_mode, len_mode_freq = max(self.hop_hist.items(), key=lambda kv: (kv[1], -kv[0]))
        bk_mode, bk_mode_freq = max(self.bk_hist.items(), key=lambda kv: (kv[1], -kv[0]))
        top_k = int(self.top_k)
        top_len = self.hop_hist.most_common(min(top_k, len(self.hop_hist)))
        top_back = self.bk_hist.most_common(min(top_k, len(self.bk_hist)))

        width = 62
        lines = []
        lines.append("=" * width)
        title = "Dataset Hop Statistics" + (f" {label}" if label else "")
        lines.append(f"{title:^{width}}")
        lines.append("=" * width)
        lines.append(f"{'Samples:':<28}{n:,}")
        lines.append(f"{'Total tokens:':<28}{self.total_tokens:,}")
        lines.append(
            f"{'Hops/sample:':<28}min {self.hop_min}, max {self.hop_max}, mean {mean:.2f}, var {var:.2f}, mode {len_mode}x{len_mode_freq}"
        )
        lines.append(
            f"{'Backtracks/sample:':<28}min {self.bk_min}, max {self.bk_max}, mean {bk_mean:.2f}, var {bk_var:.2f}, mode {bk_mode}x{bk_mode_freq}"
        )
        lines.append(f"{'Operator usage:':<28}" + " ".join(f"{k}:{self.op_ctr.get(k, 0):,}" for k in ['+', '-', '*', '/']))
        lines.append("Top hop-counts:")
        for hops, cnt in top_len:
            lines.append(f"  {hops:>2} hops × {cnt:,}")
        lines.append("Top backtrack-counts:")
        for bks, cnt in top_back:
            lines.append(f"  {bks:>2} BACK × {cnt:,}")
        logging.info("\n" + "\n".join(lines))


def _log_datagen_config(cfg):
    width = 62
    ns_val = cfg.get("num_samples")
    ns_str = f"{ns_val:,}" if isinstance(ns_val, int) else "ALL"
    if isinstance(cfg["target_range"], dict):
        tr_s = ", ".join(f"n={k}:{v[0]}-{v[1]}" for k, v in sorted(cfg["target_range"].items()))
    else:
        tr = cfg["target_range"]
        tr_s = f"{tr[0]}-{tr[1]}"
    lines = [
        "=" * width,
        f"{'Data Generation Configuration (A*)':^{width}}",
        "=" * width,
        f"{'num_input_digits_list:':<32}{cfg.get('num_input_digits_list')}",
        f"{'num_samples:':<32}{ns_str}",
        f"{'target_range:':<32}{tr_s}",
        f"{'small_deck:':<32}{cfg['small_deck']}",
        f"{'large_deck:':<32}{cfg['large_deck']}",
        f"{'Data-gen workers:':<32}{cfg.get('data_gen_workers', 'auto')}",
        f"{'Seed:':<32}{cfg['seed']}",
        "-" * width,
        f"{'A* max expansions:':<32}{cfg.get('astar_max_expansions')}",
        f"{'A* max trace tokens:':<32}{cfg.get('astar_max_trace_tokens')}",
        f"{'A* max focus switches:':<32}{cfg.get('astar_max_focus_switches')}",
        f"{'A* stick steps:':<32}{cfg.get('astar_stick_steps')}",
        f"{'Compact steps:':<32}{int(bool(cfg.get('compact_steps', False)))}",
        f"{'use_h2:':<32}{int(bool(cfg.get('use_h2', False)))}",
        f"{'h2_use_all_ops:':<32}{int(bool(cfg.get('h2_use_all_ops', False)))}",
        f"{'h2_softcap_k_factor:':<32}{cfg.get('h2_softcap_k_factor', 1.0)}",
    ]
    lines.append("=" * width)
    logging.info("\n" + "\n".join(lines))


# =========================
# PUBLIC DISPATCH
# =========================
def generate_dataset(cfg: dict) -> tuple[list[str], list[str]]:
    return generate_countdown_data(cfg, n_workers=cfg.get("data_gen_workers"))


# =========================
# MAIN
# =========================
def main():
    config = DATA_HPARAMS.copy()

    _v = _env("DATASET_PATH")
    if _v:
        config["dataset_path"] = _v
    _v = _env("NUM_SAMPLES")
    if _v:
        config["num_samples"] = int(_v)
    _v = _env("N_LIST")
    if _v:
        config["num_input_digits_list"] = [int(x) for x in _v.split(",") if x.strip()]
    _v = _env("DATA_GEN_WORKERS")
    if _v:
        config["data_gen_workers"] = int(_v)

    validate_data_config(config)

    config["artifacts_dir"] = os.path.join(config["artifacts_dir"], "astar")
    os.makedirs(config["artifacts_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)

    output_path = (config.get("dataset_path") or "").strip() if isinstance(config.get("dataset_path"), str) else ""
    if not output_path:
        output_path = build_cache_path(config)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    datalog_path = os.path.join(config["log_dir"], "build_countdown_dataset_datagen.log")

    with attach_file_logger(datalog_path):
        _log_datagen_config(config)
        dg_start = time.time()
        logging.info(f"Data generation started at {_ts()}")
        logging.info(f"Saving dataset to: {output_path}")

        all_prompts, all_traces = [], []
        per_n_summary = []
        tok = Tokenizer()
        global_stats = _DatasetStatsAgg(tok, top_k=int(config.get("stats_topk", 1000)))

        for n in config.get("num_input_digits_list") or []:
            if isinstance(config["num_samples"], int) and len(all_prompts) >= config["num_samples"]:
                break

            cfg_n = config.copy()
            cfg_n["num_input_digits"] = n
            if isinstance(cfg_n["target_range"], dict):
                if n not in cfg_n["target_range"]:
                    raise ValueError(f"num_input_digits_list includes {n} but target_range lacks it")
                cfg_n["target_range"] = cfg_n["target_range"][n]
            if config["num_samples"] is None:
                cfg_n["num_samples"] = None
            else:
                cfg_n["num_samples"] = max(0, int(config["num_samples"]) - len(all_prompts))
            if cfg_n["num_samples"] == 0:
                break

            msg = "\n{line}\n{center}\n{line}".format(
                line="=" * 62,
                center=f" Generating data for n={n} input digits ".center(62),
            )
            logging.info(msg)
            ins_n, outs_n = generate_dataset(cfg_n)

            if cfg_n["num_samples"] is None:
                take_ins, take_outs = ins_n, outs_n
            else:
                take = min(cfg_n["num_samples"], len(ins_n))
                take_ins, take_outs = ins_n[:take], outs_n[:take]

            if take_ins:
                n_stats = _DatasetStatsAgg(tok, top_k=int(config.get("stats_topk", 1000)))
                for i, o in zip(take_ins, take_outs):
                    n_stats.update(i, o)
                    global_stats.update(i, o)
                n_stats.log(label=f"(n={n} taken)")
                per_n_summary.append((n, len(take_ins), n_stats.total_tokens))

            all_prompts.extend(take_ins)
            all_traces.extend(take_outs)

        if not all_prompts:
            raise RuntimeError("Data generation failed to produce any samples.")

        if per_n_summary:
            width = 62
            lines = [
                "=" * width,
                f"{'Per-n Dataset Summary':^{width}}",
                "=" * width,
                f"{'n':<8}{'samples':>12}{'tokens':>16}",
            ]
            total_s = total_t = 0
            for n, s, t in sorted(per_n_summary):
                lines.append(f"{n:<8}{s:>12,}{t:>16,}")
                total_s += s
                total_t += t
            lines.append("-" * width)
            lines.append(f"{'TOTAL':<8}{total_s:>12,}{total_t:>16,}")
            logging.info("\n" + "\n".join(lines))

        global_stats.log(label="(newly generated)")
        safe_torch_save(
            {
                "prompts": all_prompts,
                "traces": all_traces,
                "vocab": list(tok.tok2id.keys()),
                "config": config,
                "format_version": 2,
            },
            output_path,
        )
        logging.info(f"Data generation finished at {_ts()} | Duration {(time.time() - dg_start) / 60:.2f} min")
        logging.info(f"Saved {len(all_prompts):,} samples.")


if __name__ == "__main__":
    os.makedirs(DATA_HPARAMS["log_dir"], exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    start_time = time.time()
    logging.info(f"Job started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    exit_reason = "ok"
    try:
        main()
    except SystemExit:
        exit_reason = "sys-exit"
        raise
    except Exception:
        exit_reason = "error"
        logging.exception("Unhandled exception:")
        raise
    finally:
        elapsed = time.time() - start_time
        logging.info(f"\nJob finished ({exit_reason}). Total runtime: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")
