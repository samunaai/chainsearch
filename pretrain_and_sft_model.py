#!/usr/bin/env python3
"""
pretrain_model.py
========================
Train/evaluate a decoder-only Transformer on the randomized backtracking
Countdown pretraining trace dataset.

This file deliberately does NOT generate data. It consumes the saved
pretraining trace bundle (single .pt or sharded directory with manifest.json)
and reuses the tokenizer / data-format contract.
"""

import os
# Set thread env vars *before* importing torch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import json
import re
import ast
import time
import math
import logging
import random
import numpy as np
from pathlib import Path
from contextlib import contextmanager, nullcontext
from collections import defaultdict
from datetime import datetime
from functools import partial
from multiprocessing import cpu_count, get_context
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.checkpoint as checkpoint
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import matplotlib
os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use(os.environ["MPLBACKEND"])
import matplotlib.pyplot as plt
    
if __name__ == "__main__" and not sys.stdout.isatty():
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
        sys.stderr.reconfigure(line_buffering=True, write_through=True)
    except AttributeError:
        sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
        sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)


# =========================
# TOKENIZER / FORMAT CONTRACT
# =========================
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
    def vocab(self):
        return list(self.tok2id.keys())

    @property
    def vocab_size(self):
        return len(self.tok2id)

    @property
    def pad_id(self):
        return self._pad

    @property
    def eos_id(self):
        return self._eos

    @property
    def sep_id(self):
        return self._sep

    def decode(self, ids):
        tokens = [self.id2tok.get(int(i), "?") for i in ids if int(i) not in (self._pad, self._eos)]
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


def encode_ids(tok: Tokenizer, text: str):
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


def parse_initial_numbers(prompt_text: str):
    """
    Parse prompts of the form:
        IN: [ 1, 3, 9, 10, 25, 50 ] TGT: 114
    """
    match = re.search(r"IN:\s*\[\s*(.*?)\s*\]", prompt_text)
    if match:
        parts = re.split(r"[,\s]+", match.group(1).strip())
        return [int(n) for n in parts if n]
    return []

PRETRAIN_RUN_TAG = "exp31"
ASTAR_RUN_TAG = "exp35"

BASE_PARAMS = dict(
    log_dir=os.getcwd(),
    seed=42,
)

TRAIN_PARAMS = dict(
    init_from_ckpt=None,
    d_model=256,
    n_heads=8,
    n_layers=4,
    mlp_ratio=4,
    use_warmup=False,
    scheduler="linear",
    warmup_epochs=0,
    min_lr=0.0,
    divergence_scope=5,
    divergence_num_epochs=3,
    val_divergence_num_epochs=5,
    val_divergence_multiplier=1.2,
    opt="adamw",
    amsgrad=True,
    grad_log_every=0,
    dropout=0.0,
    num_workers=0,
    encode_workers=32,
    weight_decay=0.01,
    max_backtracks=8,
    use_gradient_checkpointing=False,
    stochastic_steps=0,
    temperature_ops=0.7,
    top_p_ops=1.0,
    top_k_ops=0,
    print_training_batch_for_debugging=0,
    eval_only=False,
    num_inference_examples=20,
    safe_max_len=2048,
    long_seq_policy="drop",
    resume=False,
    force_rebuild_token_cache=False,
    resume_from=None,
    save_last_every=1,
    eval_program_metrics_every=0,
    split_mode="train_val",               # "train_val" or "train_val_test"
    split_seed=42,
    test_per_back=0,
    split_path=None,
    save_eval_indices=None,
    # model_max_len=None,  
    model_max_len=1024,  # longest finetuning traces is ~990 tokens, longest pretraining ~918 
)

STAGE_OVERRIDES = dict(
    pretrain=dict(
        run_tag=f"{PRETRAIN_RUN_TAG}_pretrain",
        stage="pretrain",
        stage_label="Countdown Pretraining",
        dataset_path="/workspace/user/chainsearch_archives/data_pts/pretrain/randomized_target_traces_sharded",
        model_dir="/workspace/user/chainsearch_archives/model_pts/pretrain",
        token_cache_dir="/workspace/user/chainsearch_archives/data_pts/pretrain/tokenisation_cache",
        val_per_back=56, # validation samples per BACK bucket (e.g. 9 buckets * 56 = 504 validation samples total)        
        lr=[3e-4],
        batch_size=128,
        grad_accum_steps=1,
        epochs=2,
        patience=2,
    ),
    astar=dict(
        run_tag=f"{ASTAR_RUN_TAG}_astar",
        stage="finetune",
        stage_label="Countdown A* Finetuning",
        dataset_path="/workspace/user/chainsearch_archives/data_pts/astar/astar_data_NALL_S6-6_T{6:100-999}_exp2500_trace10000_sw8_stick0_compact0_useh20_h2all0_h2k1.0_seed42_fmtv2.pt",
        model_dir="/workspace/user/chainsearch_archives/model_pts/astar",
        init_from_ckpt=(
            "/workspace/user/chainsearch_archives/model_pts/pretrain/"
            f"{PRETRAIN_RUN_TAG}_pretrain_best.pt"
        ),
        resume=False,  # Only set True to continue an existing astar finetuning run; if a last.pt is found, it will override init_from_ckpt.
        lr=[1e-4],
        batch_size=512,
        grad_accum_steps=1,
        epochs=100,
        patience=8,
        eval_program_metrics_every=0,
        # val_per_back=100, # validation samples per BACK bucket
        split_mode="train_val_test",
        split_path=(
            "/workspace/user/chainsearch_archives/data_pts/astar/"
            f"{ASTAR_RUN_TAG}_astar_target_random_"
            "train_val_test_probe_split.pt"
        ),
        require_existing_split=True,
        val_per_back=100,
        test_per_back=1000,
        token_cache_dir="/workspace/user/chainsearch_archives/data_pts/astar/tokenisation_cache",
        save_eval_indices=(
            "/workspace/user/chainsearch_archives/data_pts/astar/"
            f"{ASTAR_RUN_TAG}_astar_target_random_eval_indices.pt"
        ),
    ),
)

STAGE = os.environ.get("COUNTDOWN_STAGE", "pretrain").strip().lower()
if STAGE not in STAGE_OVERRIDES:
    raise ValueError(f"Unknown COUNTDOWN_STAGE={STAGE!r}; expected one of {sorted(STAGE_OVERRIDES)}")

HPARAMS = BASE_PARAMS | TRAIN_PARAMS | STAGE_OVERRIDES[STAGE]

def tag(cfg: dict, name: str) -> str:
    rt = str(cfg.get("run_tag", "")).strip()
    parts = [p for p in (rt, name) if p]
    return "_".join(parts)

def stage_train_noun(cfg: dict) -> str:
    return "Finetuning" if str(cfg.get("stage", "pretrain")).lower() == "finetune" else "Pretraining"

def stage_label(cfg: dict) -> str:
    return str(cfg.get("stage_label", stage_train_noun(cfg)))

def get_run_log_path(cfg: dict) -> str:
    suffix = "eval_results" if cfg.get("eval_only", False) else stage_train_noun(cfg).lower()
    return os.path.join(cfg["log_dir"], f"{tag(cfg, 'run_countdown')}_{suffix}.log")

# =========================
# LOGGING / IO
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


def _clock() -> str:
    return datetime.now().strftime("%I:%M%p").lstrip("0").lower()


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


def save_checkpoint(obj: dict, path: str) -> None:
    safe_torch_save(obj, path)


def load_checkpoint(path: str, *, map_location: str = "cpu") -> dict:
    return torch.load(path, map_location=map_location)


# =========================
# CONFIG HELPERS
# =========================
def validate_train_config(cfg: dict) -> None:
    pol = str(cfg.get("long_seq_policy", "drop")).lower()
    if pol not in {"drop", "truncate_output"}:
        raise ValueError(f"Unknown long_seq_policy={pol!r}")
    split_mode = str(cfg.get("split_mode", "train_val")).lower()
    if split_mode not in {"train_val", "train_val_test"}:
        raise ValueError(f"Unknown split_mode={split_mode!r}")
    dp = cfg.get("dataset_path")
    if dp is not None and not isinstance(dp, str):
        raise ValueError("dataset_path must be a string path or None.")
    if cfg.get("batch_size", 0) <= 0:
        raise ValueError("batch_size must be positive")
    if int(cfg.get("val_per_back", 0)) <= 0:
        raise ValueError("val_per_back must be positive")
    if split_mode == "train_val_test" and int(cfg.get("test_per_back", 0)) <= 0:
        raise ValueError("test_per_back must be positive when split_mode='train_val_test'")

def build_enc_cache_path(
    cache_dir: str,
    safe_max_len: int,
    long_seq_policy: str = "drop",
    run_tag: str = "run",
) -> str:
    os.makedirs(cache_dir, exist_ok=True)

    tag_clean = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(run_tag).strip())
    pol = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(long_seq_policy).lower())

    return os.path.join(
        cache_dir,
        f"{tag_clean}_max{safe_max_len}_{pol}.pt"
    )


def stratified_split_by_back(ds, bk_counts, val_per_back: int, seed: int):
    rng = random.Random(seed)
    groups = defaultdict(list)
    bk_list = bk_counts.tolist() if torch.is_tensor(bk_counts) else list(bk_counts)
    for idx, b in enumerate(bk_list):
        groups[int(b)].append(idx)

    train_idx, val_idx = [], []
    for b in sorted(groups):
        idxs = groups[b]
        rng.shuffle(idxs)
        n_val = min(val_per_back, len(idxs))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return Subset(ds, train_idx), Subset(ds, val_idx)

def stratified_split_train_val_test_by_back(ds, bk_counts, val_per_back: int, test_per_back: int, seed: int):
    rng = random.Random(seed)
    groups = defaultdict(list)
    bk_list = bk_counts.tolist() if torch.is_tensor(bk_counts) else list(bk_counts)
    for idx, b in enumerate(bk_list):
        groups[int(b)].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for b in sorted(groups):
        idxs = groups[b]
        rng.shuffle(idxs)
        n_test = min(test_per_back, len(idxs))
        n_val = min(val_per_back, max(0, len(idxs) - n_test))
        test_idx.extend(idxs[:n_test])
        val_idx.extend(idxs[n_test:n_test + n_val])
        train_idx.extend(idxs[n_test + n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return Subset(ds, train_idx), Subset(ds, val_idx), Subset(ds, test_idx)


def get_stage_splits(ds, bk_counts, cfg, *, rank=0, poll_s=2.0):
    split_mode = str(cfg.get("split_mode", "train_val")).lower()
    split_seed = int(cfg.get("split_seed", cfg.get("seed", 42)))

    if split_mode == "train_val":
        tr, va = stratified_split_by_back(ds, bk_counts, int(cfg["val_per_back"]), split_seed)
        return tr, va, None

    split_path = cfg.get("split_path")
    if split_path and os.path.exists(split_path):
        split = torch.load(split_path, map_location="cpu")
        logging.info(f"Reusing saved split from: {split_path}")
        required = {"train_idx", "val_idx", "test_idx"}
        missing = required - set(split)
        if missing:
            raise KeyError(f"Saved split is missing keys {sorted(missing)}: {split_path}")
        tr = Subset(ds, split["train_idx"])
        va = Subset(ds, split["val_idx"])
        te = Subset(ds, split["test_idx"])
        return tr, va, te

    if bool(cfg.get("require_existing_split", False)):
        raise FileNotFoundError(
            f"require_existing_split=True, but split_path was not found:\n{split_path}"
        )

    if not split_path:
        return stratified_split_train_val_test_by_back(
            ds,
            bk_counts,
            int(cfg["val_per_back"]),
            int(cfg["test_per_back"]),
            split_seed,
        )

    if rank == 0:
        tr, va, te = stratified_split_train_val_test_by_back(
            ds,
            bk_counts,
            int(cfg["val_per_back"]),
            int(cfg["test_per_back"]),
            split_seed,
        )
        safe_torch_save(
            {
                "train_idx": list(tr.indices),
                "val_idx": list(va.indices),
                "test_idx": list(te.indices),
                "split_seed": split_seed,
                "val_per_back": int(cfg["val_per_back"]),
                "test_per_back": int(cfg["test_per_back"]),
            },
            split_path,
        )
        logging.info(f"Saved split to: {split_path}")
    else:
        logging.info(f"[RANK {rank}] Waiting for split: {split_path}")
        while not os.path.exists(split_path):
            time.sleep(poll_s)

    split = torch.load(split_path, map_location="cpu")
    tr = Subset(ds, split["train_idx"])
    va = Subset(ds, split["val_idx"])
    te = Subset(ds, split["test_idx"])
    return tr, va, te

def bucket_hist(indices, bk_counts):
    hist = defaultdict(int)
    bk_list = bk_counts.tolist() if torch.is_tensor(bk_counts) else list(bk_counts)
    for i in indices:
        hist[int(bk_list[i])] += 1
    return dict(sorted(hist.items()))
 
def log_program_metrics(metrics: dict, prefix: str = "Val", indent: str = "  "):
    logging.info(f"{indent}{prefix} program metrics")
    logging.info(
        f"{indent}  Program exact match (full trace exactly matches gold): "
        f"{100*metrics['program_exact_match']:.2f}%"
    )
    logging.info(
        f"{indent}  Any-step target hit (some valid generated step reaches target): "
        f"{100*metrics['any_step_hits_target_rate']:.2f}%"
    )
    logging.info(
        f"{indent}  Valid trace rate (all parsed steps valid/state-consistent + SUCCESS): "
        f"{100*metrics['valid_sequence_rate']:.2f}%"
    )
    logging.info(
        f"{indent}  Solved valid trace rate (valid trace and hits target): "
        f"{100*metrics['solved_valid_trace_rate']:.2f}%"
    )
    logging.info(
        f"{indent}  Best-step coverage (at least one valid parsed step): "
        f"{100*metrics['best_value_coverage']:.2f}%"
    )
    logging.info(
        f"{indent}  EXPR exact match (predicted EXPR string matches gold): "
        f"{100*metrics['expr_exact_match']:.2f}%"
    )
    logging.info(
        f"{indent}  EXPR target accuracy (predicted EXPR evaluates to target): "
        f"{100*metrics['expr_target_accuracy']:.2f}%"
    )
    logging.info(
        f"{indent}  EXPR parse rate (predicted EXPR parsed successfully): "
        f"{100*metrics['expr_parse_rate']:.2f}%"
    )
    logging.info(
        f"{indent}  EXPR target MAE (absolute error to target among parseable EXPRs): "
        f"{metrics['expr_target_mae']:.2f}"
    )
    logging.info(
        f"{indent}  Best-step target MAE (closest valid intermediate to target): "
        f"{metrics['best_value_mae']:.2f}"
    )
    logging.info(
        f"{indent}  BACK count MAE (absolute error in number of BACK tokens): "
        f"{metrics['back_count_mae']:.2f}"
    )
    logging.info(
        f"{indent}  BACK count bias (predicted BACK count minus gold BACK count): "
        f"{metrics['back_count_bias']:+.2f}"
    )
    logging.info(
        f"{indent}  Operation validity rate (generated arithmetic step is legal): "
        f"{100*metrics['op_valid_rate']:.2f}%"
    )
    logging.info(
        f"{indent}  Operation state-consistency rate (state update matches the operation): "
        f"{100*metrics['op_state_consistent_rate']:.2f}%"
    )
    logging.info(
        f"{indent}  Evaluation samples: {metrics['samples']:,}"
    )

def iter_trace_bundle(dataset_path: str):
    if os.path.isfile(dataset_path):
        data = torch.load(dataset_path, map_location="cpu")
        prompts = data.get("prompts") or data.get("ins")
        traces = data.get("traces") or data.get("outs")
        if not prompts or not traces:
            raise RuntimeError(f"No prompts/traces found in {dataset_path}")
        for p, t in zip(prompts, traces):
            yield p, t
        return

    manifest = os.path.join(dataset_path, "manifest.json")
    if not os.path.exists(manifest):
        raise FileNotFoundError(f"Expected manifest.json in sharded dataset dir: {dataset_path}")

    with open(manifest) as f:
        meta = json.load(f)

    for shard_name in meta["shards"]:
        shard_path = os.path.join(dataset_path, shard_name)
        shard = torch.load(shard_path, map_location="cpu")
        prompts = shard["prompts"]
        traces = shard["traces"]
        if len(prompts) != len(traces):
            raise RuntimeError(f"Length mismatch in shard: {shard_path}")
        for p, t in zip(prompts, traces):
            yield p, t


def load_trace_bundle_vocab(dataset_path: str):
    if os.path.isfile(dataset_path):
        data = torch.load(dataset_path, map_location="cpu")
        return data.get("vocab")
    manifest = os.path.join(dataset_path, "manifest.json")
    if not os.path.exists(manifest):
        return None
    with open(manifest) as f:
        return json.load(f).get("vocab")

def make_loader(subset, cfg, shuffle, sampler=None, drop_last=False, device=None, collate_fn=None):
    pin = (device is not None and getattr(device, "type", "") == "cuda")
    num_workers = int(cfg.get("num_workers", 0))
    return DataLoader(
        subset,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),
        batch_size=cfg["batch_size"],
        num_workers=num_workers,
        pin_memory=pin,
        collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),
        drop_last=drop_last,
    )


def is_ddp() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def is_main() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


# =========================
# TOKENIZATION CACHE / VARIABLE-LENGTH DATASET
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


def _iter_shards(items, shard_size: int):
    it = iter(items)
    while True:
        chunk = list(islice(it, shard_size))
        if not chunk:
            return
        yield chunk


# local import to keep top imports compact
from itertools import islice


def apply_len_policy(ids: list[int], tok: Tokenizer, safe_max_len: Optional[int], policy: str):
    if safe_max_len is None:
        return ids, False
    try:
        sep = ids.index(tok.sep_id)
    except ValueError:
        return None, False
    if sep + 2 > safe_max_len:
        return None, False
    if len(ids) <= safe_max_len:
        return ids, False
    if policy == "drop":
        return None, False
    if policy != "truncate_output":
        raise ValueError(f"Unknown long_seq_policy={policy!r}")
    prompt = ids[:sep + 1]
    tail_budget = safe_max_len - len(prompt)
    tail = ids[sep + 1: sep + 1 + tail_budget]
    if not tail:
        return None, False
    if tail[-1] != tok.eos_id:
        tail[-1] = tok.eos_id
    return prompt + tail, True


def _encode_shard(args):
    shard, safe_max_len, policy = args
    tok = WORKER_TOK or Tokenizer()
    back_id = tok.tok2id["BACK"]

    flat = []
    lengths = []
    bk = []
    ml = 0
    mi = 0
    mo = 0
    dropped = 0
    truncated = 0
    malformed = 0

    for s in shard:
        ids = encode_ids(tok, s)
        ids, was_trunc = apply_len_policy(ids, tok, safe_max_len, policy)
        if ids is None:
            dropped += 1
            continue
        if was_trunc:
            truncated += 1

        if ids.count(tok.sep_id) != 1 or ids.count(tok.eos_id) != 1:
            malformed += 1
            continue
        try:
            sep = ids.index(tok.sep_id)
            eos = ids.index(tok.eos_id)
        except ValueError:
            malformed += 1
            continue
        if sep >= eos:
            malformed += 1
            continue
        ml = max(ml, len(ids))
        mi = max(mi, sep + 1)
        mo = max(mo, len(ids) - (sep + 1))
        bk.append(ids.count(back_id))
        lengths.append(len(ids))
        flat.extend(ids)

    if lengths:
        flat_arr = np.asarray(flat, dtype=np.int32)
        lengths_arr = np.asarray(lengths, dtype=np.int32)
        bk_arr = np.asarray(bk, dtype=np.int16)
    else:
        flat_arr = np.empty((0,), dtype=np.int32)
        lengths_arr = np.empty((0,), dtype=np.int32)
        bk_arr = np.empty((0,), dtype=np.int16)

    return flat_arr, lengths_arr, bk_arr, ml, mi, mo, dropped, truncated, malformed


def compute_lengths_and_backs(enc_ids: list[list[int]], tok: Tokenizer):
    ml = 0
    mi = 0
    mo = 0
    back_id = tok.tok2id["BACK"]
    bk_counts = []
    for ids in enc_ids:
        sep = ids.index(tok.sep_id)
        ml = max(ml, len(ids))
        mi = max(mi, sep + 1)
        mo = max(mo, len(ids) - (sep + 1))
        bk_counts.append(ids.count(back_id))
    return ml, mi, mo, bk_counts


def _build_and_save_enc_blob(seqs, tok, safe_max_len, long_seq_policy, enc_cache_path, cfg):
    nproc = int(cfg.get("encode_workers", os.cpu_count() or 1))
    nproc = max(1, min(nproc, os.cpu_count() or 1))
    shard_size = int(os.environ.get("ENC_SHARD_SIZE", "50000"))

    ctx = get_context("spawn")
    flat_parts = []
    len_parts = []
    bk_parts = []
    ml = mi = mo = 0
    dropped = 0
    truncated = 0
    malformed = 0

    tasks = ((sh, safe_max_len, str(long_seq_policy).lower()) for sh in _iter_shards(seqs, shard_size))
    total_shards = math.ceil(len(seqs) / shard_size) if hasattr(seqs, "__len__") else None

    with ctx.Pool(processes=nproc, initializer=_pool_init, initargs=(1,), maxtasksperchild=200) as p:
        for flat_arr, lengths_arr, bk_arr, ml_s, mi_s, mo_s, d_s, tr_s, mf_s in tqdm(
            p.imap_unordered(_encode_shard, tasks, chunksize=1),
            total=total_shards,
            desc=f"Tokenizing shards on {nproc} CPUs",
            dynamic_ncols=True,
        ):
            if lengths_arr.size == 0:
                dropped += int(d_s)
                truncated += int(tr_s)
                malformed += int(mf_s)
                continue
            flat_parts.append(torch.from_numpy(flat_arr))
            len_parts.append(torch.from_numpy(lengths_arr))
            bk_parts.append(torch.from_numpy(bk_arr))
            ml = max(ml, int(ml_s))
            mi = max(mi, int(mi_s))
            mo = max(mo, int(mo_s))
            dropped += int(d_s)
            truncated += int(tr_s)
            malformed += int(mf_s)

    if not len_parts:
        raise RuntimeError("Tokenisation produced no kept samples.")

    lengths = torch.cat(len_parts, dim=0).to(torch.int32)
    flat_ids = torch.cat(flat_parts, dim=0).to(torch.int32)
    bk_counts = torch.cat(bk_parts, dim=0).to(torch.int16)

    offsets = torch.zeros(int(lengths.numel()), dtype=torch.int64)
    if lengths.numel() > 1:
        offsets[1:] = torch.cumsum(lengths[:-1].to(torch.int64), dim=0)

    kept = int(lengths.numel())
    logging.info(
        f"Tokenisation length policy={long_seq_policy} safe_max_len={safe_max_len} "
        f"-> kept={kept:,} dropped={dropped:,} truncated={truncated:,} malformed={malformed:,} "
        f"(shard_size={shard_size})"
    )

    blob = {
        "flat_ids": flat_ids,
        "offsets": offsets,
        "lengths": lengths,
        "ml": ml,
        "mi": mi,
        "mo": mo,
        "bk_counts": bk_counts,
        "vocab": list(tok.tok2id.keys()),
        "safe_max_len": int(safe_max_len) if safe_max_len is not None else None,
        "long_seq_policy": str(long_seq_policy),
    }
    safe_torch_save(blob, enc_cache_path)
    logging.info("Encoding complete. Saved tokenisation cache.")
    return blob


def _load_enc_blob(enc_cache_path: str, tok: Tokenizer, *, safe_max_len=None, long_seq_policy=None):
    blob = torch.load(enc_cache_path, map_location="cpu")
    if "vocab" in blob:
        cur_vocab = list(tok.tok2id.keys())
        if list(blob["vocab"]) != cur_vocab:
            raise RuntimeError(f"Tokenisation cache vocab mismatch: {enc_cache_path}")
    if safe_max_len is not None and ("safe_max_len" in blob) and blob["safe_max_len"] != safe_max_len:
        raise RuntimeError(
            f"Tokenisation cache safe_max_len mismatch: cache={blob['safe_max_len']} vs run={safe_max_len} ({enc_cache_path})"
        )
    if long_seq_policy is not None and ("long_seq_policy" in blob) and str(blob["long_seq_policy"]).lower() != str(long_seq_policy).lower():
        raise RuntimeError(
            f"Tokenisation cache long_seq_policy mismatch: cache={blob['long_seq_policy']} vs run={long_seq_policy} ({enc_cache_path})"
        )

    if all(k in blob for k in ("flat_ids", "offsets", "lengths")):
        if all(k in blob for k in ("ml", "mi", "mo", "bk_counts")):
            return blob, None, blob["ml"], blob["mi"], blob["mo"], blob["bk_counts"]
        flat_ids = blob["flat_ids"]
        offsets = blob["offsets"]
        lengths = blob["lengths"]
        enc_ids = [
            flat_ids[int(offsets[i].item()): int((offsets[i] + lengths[i]).item())].tolist()
            for i in range(int(lengths.numel()))
        ]
        ml, mi, mo, bk_counts = compute_lengths_and_backs(enc_ids, tok)
        return blob, enc_ids, ml, mi, mo, bk_counts

    if "enc_ids" in blob:
        enc_ids = blob["enc_ids"]
        if all(k in blob for k in ("ml", "mi", "mo", "bk_counts")):
            ml, mi, mo, bk_counts = blob["ml"], blob["mi"], blob["mo"], blob["bk_counts"]
        else:
            ml, mi, mo, bk_counts = compute_lengths_and_backs(enc_ids, tok)
        return blob, enc_ids, ml, mi, mo, bk_counts

    raise KeyError(f"Unrecognized tokenisation cache format: {enc_cache_path}")


def get_or_build_enc_blob(seqs, tok, safe_max_len, long_seq_policy, enc_cache_path, *, rank=0, poll_s=2.0, cfg=None):
    if os.path.exists(enc_cache_path):
        logging.info(f"[RANK {rank}] Reusing tokenisation cache at: {enc_cache_path}")
    else:
        if rank == 0:
            if seqs is None:
                raise RuntimeError("Rank 0 must provide seqs when tokenisation cache does not exist.")
            logging.info(f"[RANK {rank}] Building tokenisation cache at: {enc_cache_path}")
            _build_and_save_enc_blob(seqs, tok, safe_max_len, long_seq_policy, enc_cache_path, cfg=cfg or {})
        else:
            logging.info(f"[RANK {rank}] Waiting for tokenisation cache: {enc_cache_path}")
            while not os.path.exists(enc_cache_path):
                time.sleep(poll_s)
    return _load_enc_blob(enc_cache_path, tok, safe_max_len=safe_max_len, long_seq_policy=long_seq_policy)


class VaryLenDataset(Dataset):
    def __init__(self, flat_ids, offsets, lengths, bk_counts):
        self.flat_ids = flat_ids
        self.offsets = offsets
        self.lengths = lengths
        self.bk_counts = bk_counts

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        start = int(self.offsets[idx].item())
        length = int(self.lengths[idx].item())
        ids = self.flat_ids[start:start + length].long()
        bk = int(self.bk_counts[idx].item())
        return ids, ids.clone(), bk


def collate_batch(batch, pad_id: int):
    xs, ys, bks = zip(*batch)
    lens = [x.size(0) for x in xs]
    L = max(lens)
    B = len(xs)
    X = torch.full((B, L), pad_id, dtype=torch.long)
    Y = torch.full((B, L), pad_id, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        Li = x.size(0)
        X[i, :Li] = x
        Y[i, :Li] = y
    BK = torch.tensor(bks, dtype=torch.long)
    return X, Y, BK


# =========================
# MODEL / TRAINING
# =========================
def _init_gpt2(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        nn.init.normal_(m.in_proj_weight, mean=0.0, std=0.02)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.normal_(m.out_proj.weight, mean=0.0, std=0.02)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)


class Block(nn.Module):
    def __init__(self, d, h, max_len: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        assert d % h == 0, "d_model must be divisible by n_heads"
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.n_heads = h
        self.head_dim = d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d, d)
        mh = int(round(mlp_ratio * d))
        self.mlp = nn.Sequential(
            nn.Linear(d, mh), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mh, d), nn.Dropout(dropout),
        ) 
    def _split_heads(self, t):
        B, L, C = t.shape
        return t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, t):
        B, H, L, D = t.shape
        return t.transpose(1, 2).contiguous().view(B, L, H * D)
    
    def forward(self, x, _key_padding_mask=None):
        residual = x
        x1 = self.ln1(x)
        q, k, v = self.qkv(x1).chunk(3, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
    
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=True,
        )
        attn_output = self.out_proj(self._merge_heads(attn_output))

        x = residual + attn_output
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_step(self, x, past_k=None, past_v=None):
        _, L, _ = x.shape
        assert L == 1, "forward_step expects exactly one token"

        residual = x
        x1 = self.ln1(x)
        q, k, v = self.qkv(x1).chunk(3, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if past_k is not None:
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False,
        )
        attn_output = self.out_proj(self._merge_heads(attn_output))
        x = residual + attn_output
        x = x + self.mlp(self.ln2(x))
        return x, k, v


class DecoderOnly(nn.Module):
    def __init__(self, vocab, d, h, n_layers, max_len, pad_id, use_grad_ckpt, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.pad_id = pad_id
        self.use_gradient_checkpointing = use_grad_ckpt
        self.tok = nn.Embedding(vocab, d, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, h, max_len, mlp_ratio, dropout=dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.max_len = max_len
        self.register_buffer("pos_idx", torch.arange(max_len))
        self.apply(_init_gpt2)
        self.head.weight = self.tok.weight

    def forward_features(self, x):
        _, L = x.shape
        if L > self.max_len:
            x = x[:, -self.max_len:]
            L = self.max_len
        key_padding_mask = (x == self.pad_id)
        pos_emb = self.pos(self.pos_idx[:L].to(x.device))
        h = self.tok(x) + pos_emb
        h = h.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        for blk in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                try:
                    h = checkpoint.checkpoint(lambda a, b: blk(a, b), h, key_padding_mask, use_reentrant=False)
                except TypeError:
                    h = checkpoint.checkpoint(lambda a, b: blk(a, b), h, key_padding_mask)
            else:
                h = blk(h, key_padding_mask)
        return self.ln_f(h)

    def forward(self, x):
        h = self.forward_features(x)
        return self.head(h)

    def forward_step(self, x, pos_offset: int, past_kvs=None):
        if x.size(1) != 1:
            raise ValueError("forward_step expects shape [B, 1]")

        pos_offset = int(min(pos_offset, self.max_len - 1))
        pos_emb = self.pos(self.pos_idx[pos_offset:pos_offset + 1].to(x.device)).unsqueeze(0)
        h = self.tok(x) + pos_emb

        new_past_kvs = []
        max_past = max(1, self.max_len - 1)

        for i, blk in enumerate(self.blocks):
            past_k, past_v = (None, None) if past_kvs is None else past_kvs[i]
            h, k, v = blk.forward_step(h, past_k, past_v)
            if k.size(2) > max_past:
                k = k[:, :, -max_past:, :].contiguous()
                v = v[:, :, -max_past:, :].contiguous()
            new_past_kvs.append((k, v))

        h = self.ln_f(h)
        logits = self.head(h)
        return logits, new_past_kvs

def infer_model_max_len_from_checkpoint(path: str) -> int:
    ckpt = load_checkpoint(path, map_location="cpu")
    state = ckpt.get("model_state") or ckpt.get("state")
    if state is None:
        raise KeyError(f"Checkpoint {path} has neither 'model_state' nor 'state'.")

    for key in ("pos.weight", "module.pos.weight", "_orig_mod.pos.weight", "module._orig_mod.pos.weight"):
        if key in state and torch.is_tensor(state[key]) and state[key].ndim == 2:
            return int(state[key].shape[0])

    for key, value in state.items():
        if key.endswith("pos.weight") and torch.is_tensor(value) and value.ndim == 2:
            return int(value.shape[0])

    raise KeyError(f"Could not infer model_max_len from checkpoint: {path}")


def resolve_model_max_len(cfg: dict, dataset_max_len: int) -> int:
    if cfg.get("model_max_len") is not None:
        return int(cfg["model_max_len"])

    ckpt_path = cfg.get("init_from_ckpt") or cfg.get("resume_from")
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            return infer_model_max_len_from_checkpoint(ckpt_path)
        except Exception as e:
            logging.warning(f"Could not infer model_max_len from checkpoint {ckpt_path}: {e}")

    return int(dataset_max_len)

def _build_model(config, tok, ml, *, use_grad_ckpt=False, dropout=None):
    return DecoderOnly(
        tok.vocab_size,
        config['d_model'],
        config['n_heads'],
        config['n_layers'],
        ml,
        tok.pad_id,
        use_grad_ckpt=use_grad_ckpt,
        mlp_ratio=config.get('mlp_ratio', 4.0),
        dropout=(config.get('dropout', 0.1) if dropout is None else dropout),
    )


def _state_model(model):
    m = model.module if hasattr(model, "module") else model
    m = m._orig_mod if hasattr(m, "_orig_mod") else m
    return m

def _runtime_model(model):
    return model.module if hasattr(model, "module") else model

def _get_model_state(model) -> dict:
    return _state_model(model).state_dict()

def _load_model_state(model, state: dict) -> None:
    _state_model(model).load_state_dict(state)

def optimizer_to_device(opt: torch.optim.Optimizer, device: torch.device) -> None:
    for st in opt.state.values():
        for k, v in list(st.items()):
            if torch.is_tensor(v):
                st[k] = v.to(device, non_blocking=True)


def compute_ce_loss_from_logits(logits, targets, tok):
    logits = logits[:, :-1, :]
    tgt = targets[:, 1:]
    B, Tm1 = tgt.size()

    per_tok = F.cross_entropy(
        logits.transpose(1, 2),
        tgt,
        reduction="none",
        ignore_index=tok.pad_id,
    )

    seq = targets
    sep_mask = (seq == tok.sep_id)
    eos_mask = (seq == tok.eos_id)

    sep_cnt = sep_mask.sum(dim=1)
    eos_cnt = eos_mask.sum(dim=1)
    has_sep = sep_cnt > 0
    has_eos = eos_cnt > 0

    sep_pos = torch.where(has_sep, sep_mask.float().argmax(dim=1), torch.full((B,), -1, device=targets.device))
    eos_pos = torch.where(has_eos, eos_mask.float().argmax(dim=1), torch.full((B,), -1, device=targets.device))
    good = (sep_cnt == 1) & (eos_cnt == 1) & (sep_pos >= 0) & (eos_pos >= 0) & (sep_pos < eos_pos)

    idx = torch.arange(Tm1, device=targets.device).unsqueeze(0)
    valid = good.unsqueeze(1) & (idx >= sep_pos.unsqueeze(1)) & (idx < eos_pos.unsqueeze(1))

    sample_loss_sum = (per_tok * valid.float()).sum(dim=1)
    sample_tok_count = valid.float().sum(dim=1)
    denom = sample_tok_count.sum().clamp_min(1.0)
    total = sample_loss_sum.sum() / denom

    with torch.no_grad():
        pred = logits.argmax(-1)
        correct = int(((pred == tgt) & valid).sum().item())
        n_tok = int(valid.sum().item())
        match_per_pos = (pred == tgt) & valid
        valid_len = valid.sum(dim=1)
        sample_correct_tok = match_per_pos.sum(dim=1).to(torch.float32)
        correct_len = match_per_pos.sum(dim=1)
        em_flags = (valid_len > 0) & (correct_len == valid_len)
        em_correct = int(em_flags.sum().item())
        em_total = int((valid_len > 0).sum().item())
        sample_em_total = (valid_len > 0).to(torch.float32)
        sample_em_correct = em_flags.to(torch.float32)
    return (
        total,
        total.detach(),
        {},
        correct,
        n_tok,
        em_correct,
        em_total,
        sample_loss_sum.detach(),
        sample_tok_count.detach(),
        sample_correct_tok.detach(),  
        sample_em_correct.detach(),
        sample_em_total.detach(),
    )


def _maybe_step_scheduler(scheduler, sched_step_state):
    if scheduler is None:
        return
    steps = int((sched_step_state or {}).get("steps", 0))
    warmup = int((sched_step_state or {}).get("warmup", 0))
    if steps < warmup:
        scheduler.step()
        if sched_step_state is not None:
            sched_step_state["steps"] = steps + 1


def _optimizer_step(*, opt, model, scaler=None, scheduler=None, sched_step_state=None, clip_norm: float = 1.0, log_cb=None):
    stepped = True
    if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
        scaler.unscale_(opt)
        val = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        if not torch.isfinite(val):
            stepped = False
            opt.zero_grad(set_to_none=True)
            scaler.update()
            return stepped
        if log_cb:
            log_cb(float(val))
        scaler.step(opt)
        _maybe_step_scheduler(scheduler, sched_step_state)
        scaler.update()
    else:
        val = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        if not torch.isfinite(val):
            stepped = False
            opt.zero_grad(set_to_none=True)
            return stepped
        if log_cb:
            log_cb(float(val))
        opt.step()
        _maybe_step_scheduler(scheduler, sched_step_state)
    return stepped


def run_epoch(is_train, model, loader, device, tok, opt=None, scaler=None, amp_dtype=torch.float32, use_amp=False,
              epoch=None, debug_n=0, scheduler=None, sched_step_state=None, cfg=None,
              reduce_across_ranks=True):
    cfg = cfg or {}
    stage_name = stage_train_noun(cfg)
    model.train(is_train)
    total_loss_num = torch.zeros(1, device=device)
    total_correct_tokens = torch.zeros(1, device=device)
    total_tokens = torch.zeros(1, device=device)
    total_em_correct = torch.zeros(1, device=device)
    total_em_total = torch.zeros(1, device=device)
    debug_printed_this_epoch = False
    opt_steps = 0
    grad_log_every = int(cfg.get("grad_log_every", 0))
    run_name = cfg.get("run_tag", "run")
    grad_log_path = os.path.join(cfg["log_dir"], f"{run_name}_grad_norm.log")
    grad_logger = open(grad_log_path, "a") if (is_train and is_main() and opt is not None) else None

    def _log_grad_norm(epoch_id, step_id, value):
        if grad_logger is not None:
            grad_logger.write(f"Epoch {epoch_id} Step {step_id}: grad_norm {value:.2f}\n")
            grad_logger.flush()

    def _log_this_step(step_count: int) -> bool:
        return (grad_log_every > 0) and (step_count % grad_log_every == 0)

    def _do_opt_step() -> bool:
        nonlocal opt_steps
        stepped = _optimizer_step(
            opt=opt,
            model=model,
            scaler=scaler,
            scheduler=scheduler,
            sched_step_state=sched_step_state,
            clip_norm=1.0,
            log_cb=(lambda v: _log_grad_norm(epoch if epoch is not None else 0, opt_steps, v)) if _log_this_step(opt_steps) and is_main() else None,
        )
        opt.zero_grad(set_to_none=True)
        if stepped:
            opt_steps += 1
        return stepped

    accum = max(1, int(cfg.get("grad_accum_steps", 1)))
    step_in_accum = 0
    if is_train and opt is not None:
        opt.zero_grad(set_to_none=True)

    for bidx, (x, y, _) in enumerate(loader):
        is_last_batch = (bidx == len(loader) - 1)
        if is_train and (epoch == 0) and not debug_printed_this_epoch and isinstance(debug_n, int) and debug_n > 0 and is_main():
            logging.info("\n" + "=" * 60 + "\n--- TRAINING BATCH DEBUG (epoch 0, first batch) ---")
            logging.info(f"Batch X shape: {tuple(x.shape)} | Batch Y shape: {tuple(y.shape)}")
            k = min(debug_n, x.shape[0])
            for i in range(k):
                xi = x[i].cpu().tolist()
                yi = y[i].cpu().tolist()
                x_text = tok.decode(xi)
                y_text = tok.decode(yi)
                logging.info(f"\n--- Sample {i+1}/{k} ---")
                logging.info(f"DECODED X:\n{x_text}")
                logging.info(f"DECODED Y:\n{y_text}")
                logging.info(f"X==Y: {x_text == y_text}")
            logging.info("=" * 60 + "\n")
            debug_printed_this_epoch = True

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        sync_now = ((((step_in_accum + 1) % accum) == 0) or is_last_batch) if is_train else True
        sync_ctx = (
            model.no_sync()
            if is_train and hasattr(model, "no_sync") and not sync_now
            else nullcontext()
        )

        forward_model = model if is_train else (model.module if hasattr(model, "module") else model)

        with sync_ctx:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp), torch.set_grad_enabled(is_train):
                logits = forward_model(x)
                (
                    loss, _, _, correct, n_tokens, em_correct, em_total,
                    sample_loss_sum, _, _,
                    _, _
                ) = compute_ce_loss_from_logits(logits, y, tok)

            if is_train and n_tokens > 0:
                loss_divisor = (step_in_accum + 1) if is_last_batch else accum
                loss_to_back = loss / loss_divisor
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss_to_back).backward()
                else:
                    loss_to_back.backward()

        if is_main() and is_train and (bidx % 50 == 0):
            logging.info(
                f"  [{stage_name} | Epoch {epoch}] "
                f"Batch {bidx}/{len(loader)} | Loss: {float(loss.detach().item()):.4f} | n_tok: {n_tokens}"
            )
            for h in logging.getLogger().handlers:
                try:
                    h.flush()
                except Exception:
                    pass

        if n_tokens > 0 or is_train:
            if is_train and n_tokens > 0:
                step_in_accum += 1
                if sync_now:
                    _do_opt_step()
                    step_in_accum = 0
            if n_tokens > 0:
                total_loss_num += sample_loss_sum.sum()
            total_correct_tokens += correct
            total_tokens += n_tokens
            total_em_correct += em_correct
            total_em_total += em_total
        

    if is_train and opt is not None and step_in_accum > 0:
        _do_opt_step()
        step_in_accum = 0

    if reduce_across_ranks and is_ddp():
        dist.all_reduce(total_loss_num, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct_tokens, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_em_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_em_total, op=dist.ReduceOp.SUM)


    if grad_logger is not None:
        grad_logger.close()

    tok_acc = float((total_correct_tokens / total_tokens.clamp_min(1.0)).item()) if total_tokens.item() > 0 else 0.0
    seq_em = float((total_em_correct / total_em_total.clamp_min(1.0)).item()) if total_em_total.item() > 0 else 0.0
    avg_loss = float((total_loss_num / total_tokens.clamp_min(1.0)).item()) if total_tokens.item() > 0 else float("nan")

    return tok_acc, seq_em, avg_loss


inference_mode = getattr(torch, "inference_mode", torch.no_grad)


def eval_and_broadcast(model, loader, device, tok, use_amp, amp_dtype, cfg):
    out = torch.tensor([float("nan"), 0.0, 0.0], device=device) 
    if is_main():
        tok_acc, seq_em, loss = run_epoch(
            False, model, loader, device, tok, None, None, amp_dtype, use_amp,
            cfg=cfg, reduce_across_ranks=False
        )
        out[:] = torch.tensor([loss, tok_acc, seq_em], device=device)
    if is_ddp():
        dist.broadcast(out, src=0)
    return float(out[0].item()), float(out[1].item()), float(out[2].item())


def train_model(cfg, model, tok, _ds, tr, va, _bk_counts, device, amp_dtype, use_amp, global_rank, model_dir):
    history = defaultdict(list)
    stage_name = stage_train_noun(cfg)
    named = list(model.named_parameters())
    uniq = {id(p): (n, p) for n, p in named if p.requires_grad}
    decay, no_decay = [], []
    for n, p in uniq.values():
        nl = n.lower()
        is_bias = n.endswith("bias")
        is_ln = ("layernorm" in nl) or ("ln." in nl) or ("ln_f" in nl)
        is_emb = ("tok.weight" in n) or ("pos.weight" in n)
        is_head = ("head.weight" in n)
        if is_bias or is_ln or is_emb or is_head:
            no_decay.append(p)
        else:
            decay.append(p)
    param_groups = [
        {"params": decay, "weight_decay": cfg["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    opt_kwargs = {"lr": (cfg["lr"][0] if isinstance(cfg.get("lr"), (list, tuple)) else float(cfg["lr"])), "amsgrad": bool(cfg.get("amsgrad", False))}
    if device.type == "cuda":
        opt_kwargs["fused"] = True
    logging.info(f"Optimizer: AdamW with { {k: v for k, v in opt_kwargs.items()} }")
    try:
        opt = torch.optim.AdamW(param_groups, **opt_kwargs)
    except TypeError:
        opt_kwargs.pop("fused", None)
        opt = torch.optim.AdamW(param_groups, **opt_kwargs)

    use_fp16 = bool(use_amp and amp_dtype == torch.float16)
    if device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    else:
        scaler = None

    best_ckpt = os.path.join(model_dir, tag(cfg, "best.pt"))
    last_ckpt = os.path.join(model_dir, tag(cfg, "last.pt"))
    save_last_every = int(cfg.get("save_last_every", 1))

    lr_seq = cfg["lr"] if isinstance(cfg.get("lr"), (list, tuple)) else [float(cfg["lr"])]
    phase_idx = 0
    loss_hist = []
    val_hist = []
    div_scope = int(cfg.get("divergence_scope", 5))
    div_pat = int(cfg.get("divergence_num_epochs", 3))
    val_div_pat = int(cfg.get("val_divergence_num_epochs", div_pat))
    val_div_mult = float(cfg.get("val_divergence_multiplier", 1.2))
    best_val, patience, global_ep = float("inf"), 0, 0
    val_div_count = 0
    global_best_val = float("inf")
    resume_path = cfg.get("resume_from") or (last_ckpt if bool(cfg.get("resume", False)) and os.path.exists(last_ckpt) else None)

    if is_main():
        world = dist.get_world_size() if is_ddp() else 1
        logging.info(
            f"Effective batch: per_gpu={cfg['batch_size']} × accum={cfg.get('grad_accum_steps',1)} × world={world} = "
            f"{cfg['batch_size'] * max(1, cfg.get('grad_accum_steps', 1)) * world}"
        )

    stage_ep = 0
    subset_tr, subset_va = tr, va
    sampler = DistributedSampler(subset_tr, shuffle=True) if is_ddp() else None
    drop_last = len(subset_tr) >= cfg["batch_size"]

    pad_id = tok.pad_id
    train_loader = make_loader(subset_tr, cfg, shuffle=True, sampler=sampler, drop_last=drop_last, device=device,
                               collate_fn=partial(collate_batch, pad_id=pad_id))
    val_loader = DataLoader(
        subset_va,
        shuffle=False,
        batch_size=min(512, len(subset_va)),
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
        collate_fn=partial(collate_batch, pad_id=pad_id),
    )

    accum = max(1, int(cfg.get("grad_accum_steps", 1)))
    steps_per_epoch = math.ceil(len(train_loader) / accum) if len(train_loader) > 0 else 0
    total_steps = max(1, cfg["epochs"] * steps_per_epoch)
    we = int(cfg.get("warmup_epochs", 0))
    warmup = we * max(1, steps_per_epoch)
    base_lr = float(lr_seq[0])
    min_lr = float(cfg.get("min_lr", 0.0))
    sched_kind = str(cfg.get("scheduler", "cosine")).lower()

    def _lr_lambda(step):
        if warmup > 0 and step < warmup:
            return step / max(1, warmup)
        if sched_kind == "linear":
            return 1.0
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(max(progress, 0.0), 1.0)
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        floor = min_lr / max(base_lr, 1e-12)
        return floor + (1.0 - floor) * scale

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda) if cfg.get("use_warmup", False) else None
    sched_step_state = {"steps": 0, "warmup": warmup}

    if resume_path and os.path.exists(resume_path):
        ck = load_checkpoint(resume_path, map_location="cpu")
        if "model_state" in ck:
            _load_model_state(model, ck["model_state"])
        if "opt_state" in ck:
            opt.load_state_dict(ck["opt_state"])
            optimizer_to_device(opt, device)
        if scaler is not None and ck.get("scaler_state") is not None:
            try:
                scaler.load_state_dict(ck["scaler_state"])
            except Exception:
                pass
        if scheduler is not None and ck.get("scheduler_state") is not None:
            try:
                scheduler.load_state_dict(ck["scheduler_state"])
            except Exception:
                pass
        global_ep = int(ck.get("epoch", 0))
        stage_ep = global_ep
        phase_idx = int(ck.get("phase_idx", 0))
        best_val = float(ck.get("best_val", best_val))
        global_best_val = float(ck.get("global_best_val", global_best_val))
        patience = int(ck.get("patience", patience))
        val_div_count = int(ck.get("val_div_count", val_div_count))
        sss = ck.get("sched_step_state") or {}
        sched_step_state["steps"] = int(sss.get("steps", sched_step_state["steps"]))
        if is_main():
            logging.info(f"Resumed full training state from: {resume_path}")

    while stage_ep < cfg["epochs"]:
        stop_due_to_div = False
        if is_ddp() and hasattr(train_loader, "sampler") and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(global_ep)
        tr_tok_acc, tr_seq_em, tr_loss = run_epoch(
            True, model, train_loader, device, tok, opt, scaler, amp_dtype, use_amp,
            epoch=global_ep, debug_n=cfg.get("print_training_batch_for_debugging", 0),
            scheduler=scheduler, sched_step_state=sched_step_state, cfg=cfg,
        )
        val_loss, va_tok_acc, va_seq_em = eval_and_broadcast(model, val_loader, device, tok, use_amp, amp_dtype, cfg)
        prog_metrics = None
        if global_rank == 0 and int(cfg.get("eval_program_metrics_every", 0)) > 0 and ((global_ep + 1) % int(cfg["eval_program_metrics_every"]) == 0):
            prog_metrics = evaluate_program_metrics(
                model, tok, val_loader, device, cfg["max_output_len"], cfg, amp_dtype, use_amp, max_batches=None
            )

        tr_b = torch.tensor([float("nan")], device=device)
        if is_main():
            tr_b[0] = float(tr_loss) if not math.isnan(tr_loss) else float("nan")
        if is_ddp():
            dist.broadcast(tr_b, src=0)
        tr_loss_for_ctrl = float(tr_b[0].item())

        if global_rank == 0:
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(val_loss)
            history["train_tok_acc"].append(tr_tok_acc)
            history["val_tok_acc"].append(va_tok_acc)
            history["train_seq_em"].append(tr_seq_em)
            history["val_seq_em"].append(va_seq_em)
            history["val_program_metrics"].append(prog_metrics)
            lr_now = opt.param_groups[0]["lr"]
            logging.info(
                f"{stage_name} Epoch {global_ep+1:04d} | "
                f"TrainLoss {tr_loss:.4f} | ValLoss {val_loss:.4f} | "
                f"TrainTokAcc {100*tr_tok_acc:5.1f}% | ValTokAcc {100*va_tok_acc:5.1f}% | "
                f"ValTeacherForcedSeqEM {100*va_seq_em:5.1f}% | LR {lr_now:.2e} | {_clock()}"
            )
            if prog_metrics is not None:
                log_program_metrics(prog_metrics, prefix="Val", indent="  ") 
            for h in logging.getLogger().handlers:
                try:
                    h.flush()
                except Exception:
                    pass
            if best_val < float("inf") and val_loss > best_val * val_div_mult:
                val_div_count += 1
                logging.error(f"VAL LOSS DIVERGED ({val_div_count}/{val_div_pat}): {val_loss:.4f} > {val_div_mult:.2f}× best {best_val:.4f}.")
                if val_div_count >= val_div_pat:
                    logging.error(f"VAL LOSS DIVERGED too many times (>= {val_div_pat}). Stopping.")
                    stop_due_to_div = True
            if val_loss < best_val:
                best_val, patience = val_loss, 0
                if best_val < global_best_val:
                    global_best_val = best_val
                    save_checkpoint({"model_state": _get_model_state(model), "config": cfg}, best_ckpt)
                    logging.info("  -> New GLOBAL best val loss. best.pt saved.")
            else:
                patience += 1

            if save_last_every > 0 and ((global_ep + 1) % save_last_every == 0):
                save_checkpoint(
                    {
                        "model_state": _get_model_state(model),
                        "opt_state": opt.state_dict(),
                        "scaler_state": (scaler.state_dict() if scaler is not None else None),
                        "scheduler_state": (scheduler.state_dict() if scheduler is not None else None),
                        "sched_step_state": dict(sched_step_state),
                        "epoch": int(global_ep + 1),
                        "phase_idx": int(phase_idx),
                        "best_val": float(best_val),
                        "global_best_val": float(global_best_val),
                        "patience": int(patience),
                        "val_div_count": int(val_div_count),
                        "config": cfg,
                    },
                    last_ckpt,
                )

        warmup_done = (sched_step_state["steps"] >= sched_step_state["warmup"])
        if warmup_done and not math.isnan(tr_loss_for_ctrl):
            loss_hist.append(tr_loss_for_ctrl)
            val_hist.append(val_loss)
            if len(loss_hist) > div_scope:
                loss_hist.pop(0)
            if len(val_hist) > div_scope:
                val_hist.pop(0)
            if len(loss_hist) == div_scope and len(val_hist) == div_scope:
                train_rises = sum(1 for i in range(1, len(loss_hist)) if loss_hist[i] > loss_hist[i - 1])
                val_rises = sum(1 for i in range(1, len(val_hist)) if val_hist[i] > val_hist[i - 1])
                if (train_rises >= div_pat) and (val_rises >= div_pat) and phase_idx < len(lr_seq) - 1:
                    phase_idx += 1
                    new_lr = float(lr_seq[phase_idx])
                    for pg in opt.param_groups:
                        pg["lr"] = new_lr
                    loss_hist.clear()
                    val_hist.clear()
                    if global_rank == 0:
                        logging.info(f"↘ LR drop: train↑{train_rises}/{div_scope}, val↑{val_rises}/{div_scope} → {new_lr:.2e} (phase {phase_idx+1}/{len(lr_seq)}).")

        ctrl = torch.tensor(0, device=device)
        if stop_due_to_div:
            ctrl.fill_(2)
        if is_main() and not math.isnan(val_loss) and patience >= cfg["patience"]:
            logging.info("Stopping early.")
            ctrl.fill_(2)
        if is_ddp():
            dist.broadcast(ctrl, src=0)
        if int(ctrl.item()) == 2:
            return history

        stage_ep += 1
        global_ep += 1
    return history


# =========================
# EVAL / SAMPLING
# =========================
RE_EVENT = re.compile(
    r"BACK\s*->\s*\[\s*([^\]]*?)\s*\]\s*;"
    r"|"
    r"(?:\[\s*([^\]]*?)\s*\]\s*->\s*)?"
    r"(\d+)\s*([\+\-\*\/])\s*(\d+)\s*=\s*(\d+)\s*->\s*\[\s*([^\]]*?)\s*\]\s*;"
)
RE_EXPR = re.compile(r"EXPR\s+(.+?)\s*$")

def _ints_in(s: str) -> List[int]:
    return [int(x) for x in re.findall(r"-?\d+", s)]

def extract_expr(text: str) -> Optional[str]:
    m = RE_EXPR.search(text.strip())
    return m.group(1).strip() if m else None


def eval_expr_text(expr_text: str) -> int:
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return int(node.value)
        if hasattr(ast, "Num") and isinstance(node, ast.Num):
            return int(node.n)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_eval(node.operand)
        if isinstance(node, ast.BinOp):
            a = _eval(node.left)
            b = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                if b == 0 or a % b != 0:
                    raise ValueError("non-integer division")
                return a // b
        raise ValueError(f"Unsupported EXPR: {expr_text}")

    tree = ast.parse(expr_text, mode="eval")
    return int(_eval(tree))

@inference_mode()
def sample_one(model, tokenizer, prompt_ids, max_out, device, cfg=None, amp_dtype=torch.float32, use_amp=False):
    base_model = _runtime_model(model)
    base_model.eval()
    gen = []
    stop_ids = {tokenizer.eos_id, tokenizer.pad_id}
    cfg = cfg or {}
    stochastic_steps = int(cfg.get("stochastic_steps", 0))
    temp = float(cfg.get("temperature_ops", 1.0))
    top_p = float(cfg.get("top_p_ops", 1.0))
    top_k = int(cfg.get("top_k_ops", 0))

    def sample_from_logits(logits: torch.Tensor) -> int:
        probs = torch.softmax(logits / max(1e-6, temp), dim=-1)
        if top_k > 0:
            _, idxs = torch.topk(probs, k=min(top_k, probs.numel()))
            mask = torch.zeros_like(probs).scatter_(-1, idxs, 1.0)
            probs = (probs * mask)
            probs = probs / probs.sum().clamp_min(1e-9)
        if top_p < 1.0:
            sp, si = torch.sort(probs, descending=True)
            csum = torch.cumsum(sp, dim=-1)
            cutoff = (csum <= top_p).nonzero(as_tuple=False).flatten()
            k = int(cutoff[-1].item()) + 1 if cutoff.numel() else 1
            keep = si[:k]
            mask = torch.zeros_like(probs)
            mask[keep] = 1.0
            probs = probs * mask
            probs = probs / probs.sum().clamp_min(1e-9)
        return int(torch.multinomial(probs, 1).item())

    if not prompt_ids:
        return gen

    past_kvs = None
    total_len = 0
    last_logits = None

    for tok_id in prompt_ids:
        x = torch.tensor([[tok_id]], device=device)
        pos_offset = min(total_len, base_model.max_len - 1)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits, past_kvs = base_model.forward_step(x, pos_offset, past_kvs)
        last_logits = logits[:, -1, :].squeeze(0).float()
        total_len += 1

    max_tokens = max(0, int(max_out))
    for t in range(max_tokens):
        nxt = sample_from_logits(last_logits) if t < stochastic_steps else int(last_logits.argmax(-1).item())

        if nxt in stop_ids:
            break
        gen.append(nxt)
        x = torch.tensor([[nxt]], device=device)
        pos_offset = min(total_len, base_model.max_len - 1)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits, past_kvs = base_model.forward_step(x, pos_offset, past_kvs)
        last_logits = logits[:, -1, :].squeeze(0).float()
        total_len += 1
    return gen

@inference_mode()
def sample_batch(model, tokenizer, batch_prompt_ids, max_out, device, cfg=None, amp_dtype=torch.float32, use_amp=False):
    if not batch_prompt_ids:
        return []

    base_model = _runtime_model(model)
    base_model.eval()

    cfg = cfg or {}
    stochastic_steps = int(cfg.get("stochastic_steps", 0))
    temp = float(cfg.get("temperature_ops", 1.0))
    top_p = float(cfg.get("top_p_ops", 1.0))
    top_k = int(cfg.get("top_k_ops", 0))
    stop_ids = {tokenizer.eos_id, tokenizer.pad_id}

    all_gens = [[] for _ in range(len(batch_prompt_ids))]
    max_tokens = max(0, int(max_out))

    def sample_from_logits(logits_row: torch.Tensor) -> int:
        probs = torch.softmax(logits_row / max(1e-6, temp), dim=-1)
        if top_k > 0:
            _, idxs = torch.topk(probs, k=min(top_k, probs.numel()))
            mask = torch.zeros_like(probs).scatter_(-1, idxs, 1.0)
            probs = probs * mask
            probs = probs / probs.sum().clamp_min(1e-9)
        if top_p < 1.0:
            sp, si = torch.sort(probs, descending=True)
            csum = torch.cumsum(sp, dim=-1)
            cutoff = (csum <= top_p).nonzero(as_tuple=False).flatten()
            k = int(cutoff[-1].item()) + 1 if cutoff.numel() else 1
            keep = si[:k]
            mask = torch.zeros_like(probs)
            mask[keep] = 1.0
            probs = probs * mask
            probs = probs / probs.sum().clamp_min(1e-9)
        return int(torch.multinomial(probs, 1).item())

    # Group by exact prompt length so we can do true batched KV-cache decoding
    # without padding-contaminating the cache.
    groups = defaultdict(list)
    for orig_idx, prompt_ids in enumerate(batch_prompt_ids):
        groups[len(prompt_ids)].append((orig_idx, prompt_ids))

    for prompt_len, items in groups.items():
        orig_indices = [orig_idx for orig_idx, _ in items]
        prompts = [prompt_ids for _, prompt_ids in items]
        B = len(prompts)

        prompt_tensor = torch.tensor(prompts, dtype=torch.long, device=device)
        local_gens = [[] for _ in range(B)]

        # Prefill prompt with batched forward_step + KV cache.
        past_kvs = None
        last_logits = None
        total_len = 0
        for pos in range(prompt_len):
            x_step = prompt_tensor[:, pos:pos + 1]
            pos_offset = min(total_len, base_model.max_len - 1)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits, past_kvs = base_model.forward_step(x_step, pos_offset, past_kvs)
            last_logits = logits[:, -1, :].float()
            total_len += 1

        # active_slots maps rows in current tensors -> original local row ids
        active_slots = torch.arange(B, device=device)

        for t in range(max_tokens):
            if active_slots.numel() == 0:
                break

            if t < stochastic_steps:
                next_tokens = torch.empty(active_slots.numel(), dtype=torch.long, device=device)
                for i in range(active_slots.numel()):
                    next_tokens[i] = sample_from_logits(last_logits[i])
            else:
                next_tokens = last_logits.argmax(dim=-1)

            stop_mask = torch.zeros(next_tokens.size(0), dtype=torch.bool, device=device)
            for sid in stop_ids:
                stop_mask |= (next_tokens == sid)

            keep_mask = ~stop_mask

            # Save generated tokens for rows that are still alive this step.
            for cur_i in keep_mask.nonzero(as_tuple=False).flatten().tolist():
                local_row = int(active_slots[cur_i].item())
                local_gens[local_row].append(int(next_tokens[cur_i].item()))

            if not bool(keep_mask.any()):
                break

            # Shrink cache/state down to still-active rows only.
            x_step = next_tokens[keep_mask].unsqueeze(1)
            active_slots = active_slots[keep_mask]
            if past_kvs is not None:
                past_kvs = [(k[keep_mask], v[keep_mask]) for (k, v) in past_kvs]

            pos_offset = min(total_len, base_model.max_len - 1)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits, past_kvs = base_model.forward_step(x_step, pos_offset, past_kvs)
            last_logits = logits[:, -1, :].float()
            total_len += 1

        for local_i, orig_idx in enumerate(orig_indices):
            all_gens[orig_idx] = local_gens[local_i]

    return all_gens


@inference_mode()
def evaluate_program_metrics(model, tok, val_loader, device, max_out, cfg, amp_dtype, use_amp, max_batches=None):
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    n_prog_em, n, n_final_ok = 0, 0, 0
    n_expr_em, n_expr_target_ok, n_expr_parse_ok = 0, 0, 0
    expr_abs_err_sum, expr_abs_err_n = 0.0, 0
    step_ok = step_total = step_state_ok = step_state_total = 0
    n_valid_seq = 0
    n_solved_valid_seq = 0
    back_abs_err_sum = 0.0
    back_bias_sum = 0.0
    best_value_err_sum = 0.0
    best_value_err_n = 0

    it = iter(val_loader)
    batch_idx = 0
    while True:
        if max_batches is not None and batch_idx >= max_batches:
            break
        if is_main():
            if max_batches is None:
                logging.info(f"[eval] starting batch {batch_idx+1}")
            else:
                logging.info(f"[eval] starting batch {batch_idx+1}/{max_batches}")
        try:
            x_batch, _, _ = next(it)
        except StopIteration:
            break
        batch_idx += 1  
        if x_batch.numel() == 0:
            continue
        batch_prompt_ids = []
        batch_meta = []
        for x in x_batch:
            ids = x.tolist()
            if tok.sep_id not in ids:
                continue
            sep_idx = ids.index(tok.sep_id)
            prompt_ids = [t for t in ids[:sep_idx + 1] if t != tok.pad_id]
            tgt_text = tok.decode(ids[sep_idx + 1:])
            prompt_text = tok.decode(prompt_ids).replace(' SEP', '').strip()
            initial_nums = parse_initial_numbers(prompt_text)
            if not initial_nums:
                continue
            batch_prompt_ids.append(prompt_ids)
            batch_meta.append((tgt_text, prompt_text, initial_nums))

        batch_gen_ids = sample_batch(
            model, tok, batch_prompt_ids, max_out, device,
            cfg=cfg, amp_dtype=amp_dtype, use_amp=use_amp
        )

        for gen_ids, (tgt_text, prompt_text, initial_nums) in zip(batch_gen_ids, batch_meta):
            gen_text = tok.decode(gen_ids) if gen_ids else ""

            n += 1
            if is_main() and (n % 50 == 0):
                logging.info(f"[eval] processed {n} samples...")
            has_success = ("SUCCESS" in gen_text)
            if norm(gen_text) == norm(tgt_text):
                n_prog_em += 1
            gold_back = tgt_text.count("BACK")
            pred_back = gen_text.count("BACK")
            back_abs_err_sum += abs(pred_back - gold_back)
            back_bias_sum += (pred_back - gold_back)

            tgt_val = None
            tgt_m = re.search(r"TGT:\s*(\d+)", prompt_text)
            if tgt_m:
                tgt_val = int(tgt_m.group(1))

                gold_expr = extract_expr(tgt_text)
                pred_expr = extract_expr(gen_text)
                if gold_expr is not None and pred_expr is not None and norm(pred_expr) == norm(gold_expr):
                    n_expr_em += 1
                if pred_expr is not None:
                    try:
                        pred_val = eval_expr_text(pred_expr)
                        n_expr_parse_ok += 1
                        expr_abs_err_sum += abs(pred_val - tgt_val)
                        expr_abs_err_n += 1
                        if pred_val == tgt_val:
                            n_expr_target_ok += 1
                    except Exception:
                        pass
 
            any_step_hits_target = False
            best_step_err = None

            s_step_ok = s_step_total = s_step_state_ok = s_step_state_total = 0
            current_state = defaultdict(int)
            for v in initial_nums:
                current_state[v] += 1
            any_step = False
            for m in RE_EVENT.finditer(gen_text):
                back_state_str = m.group(1)
                if back_state_str is not None:
                    current_state = defaultdict(int)
                    for v in _ints_in(back_state_str):
                        current_state[v] += 1
                    continue

                before_str = m.group(2)
                n1_str, op, n2_str, res_str, after_str = m.group(3), m.group(4), m.group(5), m.group(6), m.group(7)
                any_step = True
                n1, n2, res = int(n1_str), int(n2_str), int(res_str)
                is_arith_ok = (
                    (op == '+' and n1 + n2 == res) or
                    (op == '-' and n1 > n2 and n1 - n2 == res) or
                    (op == '*' and n1 > 1 and n2 > 1 and n1 * n2 == res) or
                    (op == '/' and n2 > 0 and n1 % n2 == 0 and n1 // n2 == res)
                )
                step_total += 1
                s_step_total += 1
                if is_arith_ok:
                    step_ok += 1
                    s_step_ok += 1

                step_state_total += 1
                s_step_state_total += 1

                stated_before = defaultdict(int)
                if before_str:
                    for v in _ints_in(before_str):
                        stated_before[v] += 1
                else:
                    stated_before = current_state.copy()
                stated_after = defaultdict(int)
                for v in _ints_in(after_str):
                    stated_after[v] += 1

                if dict(stated_before) != dict(current_state) or not is_arith_ok:
                    current_state = stated_after
                    continue

                if n1 == n2:
                    if current_state.get(n1, 0) < 2:
                        current_state = stated_after
                        continue
                else:
                    if current_state.get(n1, 0) < 1 or current_state.get(n2, 0) < 1:
                        current_state = stated_after
                        continue

                temp_state = []
                for k, c in current_state.items():
                    temp_state.extend([k] * c)
                temp_state.remove(n1)
                if n1 == n2:
                    temp_state.remove(n1)
                else:
                    temp_state.remove(n2)
                temp_state.append(res)
                expected_after = defaultdict(int)
                for v in temp_state:
                    expected_after[v] += 1

                if dict(expected_after) == dict(stated_after):
                    step_state_ok += 1
                    s_step_state_ok += 1

                    if tgt_val is not None:
                        err = abs(res - tgt_val)
                        best_step_err = err if best_step_err is None else min(best_step_err, err)
                        if res == tgt_val:
                            any_step_hits_target = True
                current_state = stated_after

            if any_step_hits_target:
                n_final_ok += 1

            if best_step_err is not None:
                best_value_err_sum += best_step_err
                best_value_err_n += 1

            if any_step and (s_step_ok == s_step_total) and (s_step_state_ok == s_step_state_total) and has_success:
                n_valid_seq += 1

                if any_step_hits_target:
                    n_solved_valid_seq += 1

    if n == 0:
        raise RuntimeError("Evaluation failed: No valid samples were processed.")
    out = dict(
        samples=n,
        program_exact_match=n_prog_em / n,
        any_step_hits_target_rate=n_final_ok / n,
        valid_sequence_rate=n_valid_seq / n,
        solved_valid_trace_rate=n_solved_valid_seq / n,
        back_count_mae=back_abs_err_sum / n,
        back_count_bias=back_bias_sum / n,
        best_value_coverage=(best_value_err_n / n) if n else 0.0,
        best_value_mae=(best_value_err_sum / best_value_err_n) if best_value_err_n else float("nan"),
        expr_exact_match=n_expr_em / n,
        expr_target_accuracy=n_expr_target_ok / n,
        expr_parse_rate=n_expr_parse_ok / n,
        expr_target_mae=(expr_abs_err_sum / expr_abs_err_n) if expr_abs_err_n else float("nan"),
        op_valid_rate=(step_ok / step_total) if step_total else 0.0,
        op_state_consistent_rate=(step_state_ok / step_state_total) if step_state_total else 0.0,
    ) 
    return out



def _pretty_trace_for_log(text: str, *, indent: str = "  ") -> str:
    """
    Pretty-print a decoded trace for logs only.

    This does NOT affect training data, tokenization, loss computation, or
    model targets. It only turns a one-line decoded trace into a readable
    multi-line display by splitting at semicolons.
    """
    s = re.sub(r"\s+", " ", str(text)).strip()
    if not s:
        return f"{indent}<empty>"

    # Most trace events end with semicolons:
    #   state transition ;
    #   BACK -> ... ;
    #   SUCCESS ;
    # EXPR usually follows the final semicolon and should not receive one.
    parts = [p.strip() for p in re.split(r"\s*;\s*", s) if p.strip()]
    if not parts:
        return f"{indent}{s}"

    lines = []
    for part in parts:
        if part.startswith("EXPR"):
            lines.append(f"{indent}{part}")
        else:
            lines.append(f"{indent}{part} ;")
    return "\n".join(lines)


def _format_example_for_log(
    *,
    example_num: int,
    problem: str,
    true_sol: str,
    generated_sol: str,
    width: int = 80,
) -> str:
    """
    Build a readable inference-example block for logs.

    The separators and newlines are display-only. They are not part of the
    model input/output format.
    """
    line = "=" * width
    return (
        f"\n{line}\n"
        f"Example #{example_num}\n"
        f"{line}\n"
        f"Problem: {problem}\n"
        f"\n"
        f"*True Sol:\n"
        f"{_pretty_trace_for_log(true_sol)}\n"
        f"\n"
        f"*Generated Sol:\n"
        f"{_pretty_trace_for_log(generated_sol)}\n"
        f"{line}"
    )




def _log_metrics_and_examples(config, final_model, tok, eval_ds, device, amp_dtype, use_amp, split_name="val"):
    val_loader_eval = DataLoader(
        eval_ds,
        shuffle=False,
        batch_size=min(512, len(eval_ds)),
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
        collate_fn=partial(collate_batch, pad_id=tok.pad_id),
    )
    metrics = evaluate_program_metrics(final_model, tok, val_loader_eval, device,
                                       config['max_output_len'], config, amp_dtype, use_amp,
                                       max_batches=None)
    logging.info(f"\n--- {stage_label(config)}: Final Evaluation Metrics ({split_name}) ---")
    log_program_metrics(metrics, prefix=f"Final {split_name}", indent="  ")
 
    logging.info(f"\n--- {stage_label(config)}: Inference Examples ({split_name}) ---")
    n_show = int(config.get("num_inference_examples", 5))
    for i in range(min(n_show, len(eval_ds))):
        x, _, _ = eval_ds[i]
        ids = x.tolist()
        sep_idx = ids.index(tok.sep_id)
        prompt_ids = [t for t in ids[:sep_idx + 1] if t != tok.pad_id]
        gen_ids = sample_one(final_model, tok, prompt_ids, config['max_output_len'], device, cfg=config, amp_dtype=amp_dtype, use_amp=use_amp)
        problem = tok.decode(prompt_ids).replace(" SEP", "").strip()
        true_sol = tok.decode([t for t in ids[sep_idx + 1:] if t != tok.pad_id])
        generated_sol = tok.decode(gen_ids)
        logging.info(
            _format_example_for_log(
                example_num=i + 1,
                problem=problem,
                true_sol=true_sol,
                generated_sol=generated_sol,
            )
        )


def plot_training_results(history, filename="countdown_training_plot.png", title_prefix="Training"):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title_prefix}: Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(epochs, [a * 100 for a in history['train_tok_acc']], 'bo-', label='Training Token Accuracy')
    ax2.plot(epochs, [a * 100 for a in history['val_tok_acc']], 'ro-', label='Validation Token Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Token Accuracy (%)')
    ax2.set_title(f'{title_prefix}: Training and Validation Token Accuracy')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    logging.info(f"\nTraining plot saved to {os.path.abspath(filename)}")

# =========================
# DDP HELPERS
# =========================
def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank, dist.get_rank(), dist.get_world_size()


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# =========================
# MAIN
# =========================
def main():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            _sdpa_cm = sdpa_kernel(
                backends=[
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH,
                ]
            )
            _sdpa_cm.__enter__()
            import atexit
            atexit.register(lambda: _sdpa_cm.__exit__(None, None, None))
        except Exception:
            sdp = torch.backends.cuda
            if hasattr(sdp, "enable_flash_sdp"):
                sdp.enable_flash_sdp(True)
            if hasattr(sdp, "enable_mem_efficient_sdp"):
                sdp.enable_mem_efficient_sdp(True)
            if hasattr(sdp, "enable_math_sdp"):
                sdp.enable_math_sdp(True)
        if hasattr(torch.backends, "cudnn"):
            try:
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass

    random_seed = HPARAMS['seed']
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)
    det = bool(int(os.environ.get("DETERMINISTIC", "0")))
    torch.backends.cudnn.deterministic = det
    torch.backends.cudnn.benchmark = not det
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    config = HPARAMS.copy() 

    validate_train_config(config)

    is_ddp_env = int(os.environ.get("WORLD_SIZE", "1")) > 1
    env_rank = int(os.environ.get("RANK", "0"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_rank = env_local_rank if is_ddp_env else 0
    global_rank = env_rank if is_ddp_env else 0

    if global_rank == 0:
        os.makedirs(config["model_dir"], exist_ok=True)
        os.makedirs(config["log_dir"], exist_ok=True)
    model_dir = config["model_dir"]

    dataset_path = (config.get("dataset_path") or "").strip() if isinstance(config.get("dataset_path"), str) else ""
    if not dataset_path:
        raise ValueError("dataset_path must be set explicitly for the active trace dataset.")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}\n"
            f"Expected a sharded trace bundle directory or a .pt dataset file."
        )

    enc_cache_path = build_enc_cache_path(config["token_cache_dir"], int(config.get("safe_max_len", 2048)), str(config.get("long_seq_policy", "drop")), run_tag=config["run_tag"])

    if global_rank == 0 and bool(config.get("force_rebuild_token_cache", False)) and os.path.exists(enc_cache_path):
        logging.info(f"[RANK {global_rank}] Removing existing tokenisation cache: {enc_cache_path}")
        os.remove(enc_cache_path)
 
    if is_ddp_env:
        local_rank, global_rank, _world_size = setup_ddp()

    try:
        device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        use_amp = (device.type == 'cuda')
        use_bf16 = use_amp and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        config["num_workers"] = int(config.get("num_workers", min(4, cpu_count())))
 
        with nullcontext():
            if global_rank == 0:
                logging.info(f"Dataset path: {dataset_path}")
                logging.info(f"Encoding cache path: {enc_cache_path}")
            tok = Tokenizer()
            vocab_cached = load_trace_bundle_vocab(dataset_path)
            if vocab_cached is not None:
                cur_vocab = list(tok.tok2id.keys())
                assert vocab_cached == cur_vocab, f"vocab mismatch with cached dataset. cached[:10]={vocab_cached[:10]} current[:10]={cur_vocab[:10]}"
            config['vocab_size'] = tok.vocab_size
            tok_start_ts = time.time()
            seqs = None
            building_tok_cache = (global_rank == 0 and not os.path.exists(enc_cache_path))
            if building_tok_cache:
                logging.info(f"[RANK {global_rank}] building tokenisation cache from sharded trace bundle at {_ts()}")
                seqs = (f"{inp} SEP {out}" for inp, out in iter_trace_bundle(dataset_path))

            blob, _, ml, mi, mo, bk_counts = get_or_build_enc_blob(
                seqs, tok, int(config.get("safe_max_len", 2048)), str(config.get("long_seq_policy", "drop")), enc_cache_path,
                rank=global_rank, cfg=config,
            )
            tok_elapsed_min = (time.time() - tok_start_ts) / 60.0
            if global_rank == 0:
                if building_tok_cache:
                    logging.info(f"Tokenisation cache build finished at {_ts()} | Duration {tok_elapsed_min:.2f} min")
                else:
                    logging.info(f"Tokenisation cache load finished at {_ts()} | Duration {tok_elapsed_min:.2f} min")
            logging.info(f"[RANK {global_rank}] tokenisation loaded; building dataset/model... {_ts()}")

            config['max_input_len'], config['max_output_len'] = mi, mo
            config['model_max_len'] = resolve_model_max_len(config, ml)
            if int(config['model_max_len']) < int(ml):
                raise ValueError(
                    f"model_max_len={config['model_max_len']} is smaller than dataset max sequence length {ml}."
                )
            assert len(bk_counts) == int(blob["lengths"].numel()), "bk_counts must align with encoded dataset."
            ds = VaryLenDataset(blob["flat_ids"], blob["offsets"], blob["lengths"], blob["bk_counts"])
            tr, va, test_ds = get_stage_splits(ds, blob["bk_counts"], config, rank=global_rank)
            if global_rank == 0 and config.get("save_eval_indices"):
                payload = {"val_idx": list(va.indices)}
                if test_ds is not None:
                    payload["test_idx"] = list(test_ds.indices)
                safe_torch_save(payload, config["save_eval_indices"])
            eval_ds = test_ds if test_ds is not None else va
            eval_split_name = "test" if test_ds is not None else "val"

            if global_rank == 0:
                tr_hist = bucket_hist(tr.indices, blob["bk_counts"])
                va_hist = bucket_hist(va.indices, blob["bk_counts"])
                logging.info(f"Encoded dataset size: {len(ds):,}")
                logging.info(f"Train split size:     {len(tr):,}")
                logging.info(f"Validation size:      {len(va):,}")
                if test_ds is not None:
                    te_hist = bucket_hist(test_ds.indices, blob["bk_counts"])
                    logging.info(f"Test size:            {len(test_ds):,}")
                logging.info(f"Train BACK histogram: {tr_hist}")
                logging.info(f"Val BACK histogram:   {va_hist}")
                if test_ds is not None:
                    logging.info(f"Test BACK histogram:  {te_hist}")

            model = _build_model(config, tok, config['model_max_len'], use_grad_ckpt=config['use_gradient_checkpointing']).to(device)
            if config.get("init_from_ckpt"):
                ck0 = torch.load(config["init_from_ckpt"], map_location="cpu")
                if "model_state" in ck0:
                    _load_model_state(model, ck0["model_state"])
                elif "state" in ck0:
                    _load_model_state(model, ck0["state"])
                else:
                    raise KeyError(f"Unrecognized checkpoint format: {config['init_from_ckpt']}")
                if is_main():
                    logging.info(f"Initialized model weights from: {config['init_from_ckpt']}")

            best_ckpt_path = os.path.join(model_dir, tag(config, "best.pt"))
            last_ckpt_path = os.path.join(model_dir, tag(config, "last.pt"))

            # Compile the raw model before wrapping with DDP.
            # try:
            #     model = torch.compile(model)
            #     logging.info("Model compiled successfully.")
            # except Exception as e:
            #     logging.warning(f"Could not compile model: {e}")
            logging.info("Model compile disabled.")
            if is_ddp():
                if device.type == "cuda":
                    model = nn.parallel.DistributedDataParallel(
                        model,
                        device_ids=[torch.cuda.current_device()],
                        gradient_as_bucket_view=True,
                    )
                else:
                    model = nn.parallel.DistributedDataParallel(model)
            if config.get("use_warmup", False):
                logging.info(f"LR scheduler: warmup+{config.get('scheduler', 'cosine')} active.")
            else:
                lr0_print = (config["lr"][0] if isinstance(config.get("lr"), (list, tuple)) else float(config["lr"]))
                logging.info(f"LR fixed at {lr0_print:.2e} (no scheduler).")
            if isinstance(config.get("lr"), (list, tuple)) and len(config["lr"]) > 1:
                logging.info(
                    f"LR policy: warmup (epochs={config.get('warmup_epochs', 0)}) then hold; drop after warmup when BOTH train and val losses rise "
                    f"≥{config.get('divergence_num_epochs', 3)} times within a window of {config.get('divergence_scope', 5)}. Phases={config['lr']}."
                )

            if global_rank == 0:
                width = 62
                seen, param_total = set(), 0
                for p in model.parameters():
                    if p.requires_grad and id(p) not in seen:
                        seen.add(id(p))
                        param_total += p.numel()
                test_size_str = f"{len(test_ds):,}" if test_ds is not None else "N/A"
                log_msg = f"""
{'='*width}
{f'Run Configuration ({stage_label(config)})':^{width}}
{'='*width}
{f'PyTorch Version:':<32} {torch.__version__}
{f'DDP Enabled:':<32} {'Yes' if is_ddp() else 'No'}
{f'Total Dataset Size:':<32} {len(ds):,} samples
{f'Train / Val Size:':<32} {len(tr):,} / {len(va):,}
{f'Test Size:':<32} {test_size_str}
{f'Vocabulary Size:':<32} {config['vocab_size']} tokens
{f'd_model / n_heads / n_layers / mlp_ratio:':<32} {config['d_model']} / {config['n_heads']} / {config['n_layers']} / {config.get('mlp_ratio', 4.0)}
{f'Total Parameters:':<32} {param_total:,}
{f'Max Sequence Length:':<32} {config['model_max_len']} (dataset={ml}, prompt={mi}, solution={mo})
{f'Batch Size / LR:':<32} {config['batch_size']} / {config['lr']}
{f'Val samples per BACK:':<32} {config['val_per_back']}
{f'Loss:':<32} cross-entropy on SEP→EOS
{'='*width}"""
                logging.info(log_msg)

            if config.get("eval_only", False):
                if is_ddp() and global_rank != 0:
                    done = torch.tensor([0], device=device)
                    dist.broadcast(done, src=0)
                    return
                legacy_best = os.path.join(model_dir, tag(config, "best_model.pt"))
                load_path = None
                if os.path.exists(best_ckpt_path):
                    load_path = best_ckpt_path
                elif os.path.exists(legacy_best):
                    load_path = legacy_best
                elif os.path.exists(last_ckpt_path):
                    load_path = last_ckpt_path

                if load_path is None:
                    logging.warning("\nNo checkpoint found. Using in-memory model for evaluation.")
                    final_model = model.module if is_ddp() else model
                else:
                    ckpt = torch.load(load_path, map_location=device)
                    final_model = _build_model(config, tok, config['model_max_len'], use_grad_ckpt=False, dropout=0.0).to(device)
                    if "model_state" in ckpt:
                        final_model.load_state_dict(ckpt["model_state"])
                    elif "state" in ckpt:
                        final_model.load_state_dict(ckpt["state"])
                    else:
                        raise KeyError(f"Unrecognized checkpoint format: {load_path}")
                    try:
                        final_model = torch.compile(final_model, mode="default")
                        logging.info("Compiled final_model for evaluation.")
                    except Exception as e:
                        logging.warning(f"Could not compile final_model: {e}")
                _log_metrics_and_examples(config, final_model, tok, eval_ds, device, amp_dtype, use_amp, split_name=eval_split_name)
                if is_ddp():
                    done = torch.tensor([1], device=device)
                    dist.broadcast(done, src=0)
                return

            train_start_ts = time.time()
            stage_noun = stage_train_noun(config)
            logging.info(f"{stage_noun} started at {_ts()}")
            logging.info(f"Starting {stage_noun.lower()} on rank {global_rank}...")
            history = train_model(
                config, model, tok, ds, tr, va, bk_counts,
                device, amp_dtype, use_amp, global_rank, model_dir
            )
            logging.info(f"{stage_noun} finished at {_ts()} | Duration {(time.time()-train_start_ts)/60:.2f} min")

            # Keep all ranks alive until rank 0 finishes final-only work.
            if is_ddp():
                dist.barrier()

            if global_rank == 0:
                plot_training_results(
                    history,
                    filename=os.path.join(config["log_dir"], tag(config, "training_plot.png")),
                    title_prefix=stage_label(config),
                )
                legacy_best = os.path.join(model_dir, tag(config, "best_model.pt"))
                load_path = best_ckpt_path if os.path.exists(best_ckpt_path) else (legacy_best if os.path.exists(legacy_best) else None)
                if load_path is None:
                    logging.warning("\nNo checkpoint saved. Using last in-memory model for evaluation and examples.")
                    final_model = model.module if is_ddp() else model
                else:
                    logging.info(f"\nLoading best model for final evaluation from {load_path}...")
                    ckpt = torch.load(load_path, map_location=device)
                    final_model = _build_model(config, tok, config['model_max_len'], use_grad_ckpt=False, dropout=0.0).to(device)
                    if "model_state" in ckpt:
                        final_model.load_state_dict(ckpt["model_state"])
                    else:
                        final_model.load_state_dict(ckpt["state"])
                    try:
                        final_model = torch.compile(final_model)
                        logging.info("Compiled final_model for evaluation.")
                    except Exception as e:
                        logging.warning(f"Could not compile final_model: {e}")
                _log_metrics_and_examples(config, final_model, tok, eval_ds, device, amp_dtype, use_amp, split_name=eval_split_name)
                if is_ddp():
                    done = torch.tensor([1], device=device)
                    dist.broadcast(done, src=0)
            elif is_ddp():
                # Wait here while rank 0 does final eval/examples.
                done = torch.tensor([0], device=device)
                dist.broadcast(done, src=0)
    finally:
        if is_ddp():
            cleanup_ddp()


if __name__ == "__main__":
    os.makedirs(HPARAMS["log_dir"], exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    top_log_cm = attach_file_logger(get_run_log_path(HPARAMS), mode="w") if int(os.environ.get("RANK", "0")) == 0 else nullcontext()
    with top_log_cm:
        start_time = time.time()
        if int(os.environ.get("RANK", "0")) == 0:
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
            if int(os.environ.get("RANK", "0")) == 0:
                logging.info(f"\nJob finished ({exit_reason}). Total runtime: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")


