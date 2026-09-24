#!/usr/bin/env python3
"""Validation-selected attention-head steering comparison for ChainSearch.

This is a standalone experiment driver.  It compares unit-normalised Probe,
CAA, and (for final-answer correctness only) Direct Optimisation directions at
the same CAA-selected attention-head locations.  Intervention energy is held
constant across methods by scaling every unit direction with the corresponding
CAA raw-vector norm.

The protocol deliberately avoids a full Cartesian test-set sweep:

1. a head-selection validation subset ranks attention heads by CAA projection
   AUROC;
2. a disjoint coarse-validation subset selects one beta per method and target,
   aggregating over semantic checkpoints at a fixed reference head count;
3. a disjoint refined-validation subset selects one reference checkpoint at
   that already locked beta and head count;
4. a fresh held-out subset evaluates two predeclared one-factor sweeps: head
   count at the locked checkpoint, and checkpoint at the locked head count.

The fresh test subset is explicitly checked against the completed CAA-only
ablation so that this post-hoc experiment cannot reuse its inspected test
problems.  Positive interventions and matched negative-direction controls are
both evaluated.  Model loading, prefix construction, hooks, greedy generation,
scoring, selection, persistence, and plotting are implemented in this file.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["COUNTDOWN_STAGE"] = "astar"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

import build_probe_dataset as builder
import compute_linear_probe_vectors as probe
import evaluate_unused_astar_samples_for_probing as evaluator
import label_generated_states_for_probing_samples as reachability

pm = builder.pm


# =============================================================================
# STANDALONE STEERING CORE
# =============================================================================

SOURCE_RUN_TAG = probe.RUN_TAG
COMPARISON_RUN_TAG = "exp34-attention-head-method-comparison-v1"
ASTAR_DIR = probe.ASTAR_DIR
CAA_PATH = os.path.join(ASTAR_DIR, "caa", f"{SOURCE_RUN_TAG}_astar_caa_vectors.pt")
PROBE_PATH = probe.RESULTS_PATH
DIRECT_PATH = os.path.join(
    ASTAR_DIR,
    "direct_optimisation",
    f"{SOURCE_RUN_TAG}_astar_direct_optimisation_vectors.pt",
)
SEED = 42
SHARD_DIR = ""
ZERO_EPS = 1e-12
UNIT_NORM_TOL = 1e-4

FINAL_TASKS = {
    "trace_outcome_operation_decisions",
    "trace_outcome_trajectory",
    "trace_outcome_back_decisions",
    "terminal_outcome_controls",
}
REACHABILITY_TASKS = {"operation_reachability", "back_reachability"}
BOOL_SCORE_FIELDS = (
    "program_exact_match", "any_step_target_hit", "valid_trace", "solved_valid_trace",
    "best_step_coverage", "expr_exact_match", "expr_target_correct", "expr_parseable",
    "has_success", "primary_correct", "next_state_viable", "next_state_observed",
    "ever_future_unreachable",
)
INT_SCORE_FIELDS = (
    "generated_length", "generated_back_count", "op_valid_count", "op_total",
    "state_consistent_count", "state_total", "failure_category",
)
FLOAT_SCORE_FIELDS = ("expr_target_abs_error", "best_step_target_error", "future_reachable_state_fraction")
FAILURE_NAMES = dict(evaluator.FAILURE_CATEGORY_NAMES)

METHOD_PANEL_DISPLAY = {
    "probe": "Probe Steering",
    "caa": "CAA Steering",
    "direct": "Direct Optimisation",
}

OUTCOME_PANEL_DISPLAY = {
    "reachability": "Reachability",
    "final_answer": "Final-Answer Accuracy",
}

@dataclass(frozen=True)
class SteeringVector:
    key: str
    task: str
    label_family: str
    checkpoint: str
    representation_type: str
    location_name: str
    layer: int
    head: int
    feature_dim: int
    direction: torch.Tensor
    raw_vector: torch.Tensor
    raw_norm: float
    zero_norm: bool


@dataclass(frozen=True)
class PrefixUnit:
    task: str
    source_row: int
    record_pos: int
    cache_idx: int
    target: int
    generated_idx: int
    prompt_ids: tuple[int, ...]
    generated_prefix: tuple[int, ...]
    original_suffix: tuple[int, ...]
    prompt_text: str
    gold_text: str
    gold_back_count: int
    weight: float
    initial_state: tuple[int, ...]

    @property
    def prefix_ids(self) -> tuple[int, ...]:
        return self.prompt_ids + self.generated_prefix


@dataclass
class PrefixBatch:
    selected: tuple[int, ...]
    max_new: int
    last_token: torch.Tensor
    past_kvs: list[tuple[torch.Tensor, torch.Tensor]] | None
    total_len: int


@dataclass(frozen=True)
class SteeringCondition:
    condition_id: str
    method: str
    vector_key: str
    task: str
    label_family: str
    checkpoint: str
    representation_type: str
    location_name: str
    layer: int
    head: int
    beta: float
    scale: float
    alpha: float
    native_direct: bool
    shared_optimisation_id: str | None
    member_keys: tuple[str, ...] = ()
    member_scales: tuple[float, ...] = ()
    selection_variant: str = "single"
    head_count: int = 1
    head_selection: str = "single"
    head_scaling: str = "single"


class HookGroup:
    def __init__(self, handles: list[Any]):
        self.handles = handles

    def remove(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in (SEED, *parts))
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big") % (2**63 - 1)


def vector_key(task: str, checkpoint: str, location_name: str) -> str:
    return f"{task}:{checkpoint}:{location_name}"


def condition_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def realpath(path: str) -> str:
    return os.path.realpath(os.path.expanduser(path))


def file_signature(path: str) -> dict[str, Any]:
    resolved = realpath(path)
    stat = os.stat(resolved)
    return {"path": resolved, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def script_signature() -> dict[str, Any]:
    path = Path(__file__).resolve()
    return {
        **file_signature(str(path)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def atomic_save(payload: dict[str, Any], path: str | Path, attempts: int = 3) -> None:
    for attempt in range(1, attempts + 1):
        try:
            probe.safe_torch_save(payload, str(path))
            return
        except OSError:
            if attempt == attempts:
                raise
            time.sleep(2 ** (attempt - 1))


def load_archive(path: str, name: str, expected_run_tag: str = SOURCE_RUN_TAG) -> dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    archive = torch.load(path, map_location="cpu", weights_only=False)
    if not bool(archive.get("complete", False)):
        raise RuntimeError(f"{name} archive is incomplete: {path}")
    if archive.get("run_tag") != expected_run_tag:
        raise RuntimeError(
            f"{name} run tag {archive.get('run_tag')!r} != {expected_run_tag!r}."
        )
    return archive


def canonicalise_caa_vectors(archive: dict[str, Any]) -> dict[str, SteeringVector]:
    source = archive.get("vector_bank")
    if not isinstance(source, dict) or not source:
        raise RuntimeError("CAA archive contains no vector bank.")
    output: dict[str, SteeringVector] = {}
    for entry in source.values():
        task = str(entry["task"])
        checkpoint = str(entry["checkpoint"])
        location = str(entry["location_name"])
        key = vector_key(task, checkpoint, location)
        if key in output:
            raise RuntimeError(f"Duplicate CAA vector key: {key}")
        raw = entry["vector_raw_space"].detach().cpu().float().reshape(-1)
        direction = entry["direction_raw_space_normalized"].detach().cpu().float().reshape(-1)
        raw_norm, direction_norm = float(raw.norm()), float(direction.norm())
        zero = bool(entry.get("zero_norm", False)) or raw_norm <= ZERO_EPS or direction_norm <= ZERO_EPS
        if entry.get("direction_sign") != "label=True minus label=False":
            raise RuntimeError(f"CAA vector {key} has an unexpected direction convention.")
        if not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(direction).all()):
            raise RuntimeError(f"Non-finite values in CAA vector {key}.")
        if not zero and abs(direction_norm - 1.0) > UNIT_NORM_TOL:
            raise RuntimeError(f"CAA direction is not unit-normalised for {key}.")
        output[key] = SteeringVector(
            key=key,
            task=task,
            label_family="reachability" if task in REACHABILITY_TASKS else "final_answer",
            checkpoint=checkpoint,
            representation_type=str(entry["representation_type"]),
            location_name=location,
            layer=int(entry["layer"]),
            head=int(entry["head"]),
            feature_dim=int(entry["feature_dim"]),
            direction=torch.zeros_like(direction) if zero else direction / direction_norm,
            raw_vector=raw,
            raw_norm=raw_norm,
            zero_norm=zero,
        )
    return output


def canonicalise_probe_vectors(archive: dict[str, Any]) -> dict[str, SteeringVector]:
    source = archive.get("probe_bank")
    if not isinstance(source, dict) or not source:
        raise RuntimeError("Probe archive contains no probe bank.")
    output: dict[str, SteeringVector] = {}
    for entry in source.values():
        if str(entry["representation_type"]) != "attention_head":
            continue
        task = str(entry["task"])
        checkpoint = str(entry["checkpoint"])
        location = str(entry["location_name"])
        key = vector_key(task, checkpoint, location)
        raw = entry["weight_raw_space"].detach().cpu().float().reshape(-1)
        direction = entry["direction_raw_space_normalized"].detach().cpu().float().reshape(-1)
        raw_norm, direction_norm = float(raw.norm()), float(direction.norm())
        zero = raw_norm <= ZERO_EPS or direction_norm <= ZERO_EPS
        if entry.get("direction_sign") != "larger logit means label=True":
            raise RuntimeError(f"Probe vector {key} has an unexpected direction convention.")
        if not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(direction).all()):
            raise RuntimeError(f"Non-finite values in Probe vector {key}.")
        if not zero and abs(direction_norm - 1.0) > UNIT_NORM_TOL:
            raise RuntimeError(f"Probe direction is not unit-normalised for {key}.")
        if key in output:
            raise RuntimeError(f"Duplicate Probe vector key: {key}")
        output[key] = SteeringVector(
            key=key,
            task=task,
            label_family="reachability" if task in REACHABILITY_TASKS else "final_answer",
            checkpoint=checkpoint,
            representation_type="attention_head",
            location_name=location,
            layer=int(entry["layer"]),
            head=int(entry["head"]),
            feature_dim=int(entry["feature_dim"]),
            direction=torch.zeros_like(direction) if zero else direction / direction_norm,
            raw_vector=raw,
            raw_norm=raw_norm,
            zero_norm=zero,
        )
    return output


def canonicalise_direct_vectors(archive: dict[str, Any]) -> dict[str, SteeringVector]:
    source = archive.get("vector_bank")
    if not isinstance(source, dict) or not source:
        raise RuntimeError("Direct Optimisation archive contains no vector bank.")
    output: dict[str, SteeringVector] = {}
    for entry in source.values():
        if str(entry["representation_type"]) != "attention_head":
            continue
        task = str(entry["task"])
        checkpoint = str(entry["checkpoint"])
        location = str(entry["location_name"])
        key = vector_key(task, checkpoint, location)
        raw = entry["vector_raw_space"].detach().cpu().float().reshape(-1)
        direction = entry["direction_raw_space_normalized"].detach().cpu().float().reshape(-1)
        raw_norm, direction_norm = float(raw.norm()), float(direction.norm())
        zero = bool(entry.get("zero_norm", False)) or raw_norm <= ZERO_EPS or direction_norm <= ZERO_EPS
        if entry.get("direction_sign") != "increases selected valid final-expression likelihood":
            raise RuntimeError(f"Direct vector {key} has an unexpected direction convention.")
        if not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(direction).all()):
            raise RuntimeError(f"Non-finite values in Direct vector {key}.")
        if not zero and abs(direction_norm - 1.0) > UNIT_NORM_TOL:
            raise RuntimeError(f"Direct direction is not unit-normalised for {key}.")
        if key in output:
            raise RuntimeError(f"Duplicate Direct vector key: {key}")
        output[key] = SteeringVector(
            key=key,
            task=task,
            label_family="final_answer",
            checkpoint=checkpoint,
            representation_type="attention_head",
            location_name=location,
            layer=int(entry["layer"]),
            head=int(entry["head"]),
            feature_dim=int(entry["feature_dim"]),
            direction=torch.zeros_like(direction) if zero else direction / direction_norm,
            raw_vector=raw,
            raw_norm=raw_norm,
            zero_norm=zero,
        )
    return output


def load_model(
    data: dict[str, Any], tok: Any, device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], int]:
    checkpoint_path = str(data["source_checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state") or checkpoint.get("state")
    if state is None:
        raise KeyError("Checkpoint has neither 'model_state' nor 'state'.")
    config = dict(data["model_config"])
    checkpoint_config = checkpoint.get("config") or {}
    for name in ("d_model", "n_heads", "n_layers", "mlp_ratio"):
        if name in config and name in checkpoint_config and config[name] != checkpoint_config[name]:
            raise RuntimeError(f"Checkpoint and probe metadata disagree on {name}.")
    model_max_len = int(data["model_max_len"])
    if pm.infer_model_max_len_from_checkpoint(checkpoint_path) != model_max_len:
        raise RuntimeError("Checkpoint and probe metadata disagree on model_max_len.")
    model = pm._build_model(config, tok, model_max_len, use_grad_ckpt=False, dropout=0.0)
    pm._load_model_state(model, state)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, config, model_max_len


def load_sources(device: torch.device, methods: tuple[str, ...]) -> dict[str, Any]:
    data = probe.load_probe_dataset()
    probe.audit_probe_dataset(data)
    evaluated = load_archive(str(data["source_evaluated_path"]), "Evaluated generations")
    token_cache = torch.load(data["source_token_cache_path"], map_location="cpu", weights_only=False)
    required_cache = {"flat_ids", "offsets", "lengths", "bk_counts"}
    missing_cache = required_cache - set(token_cache)
    if missing_cache:
        raise KeyError(f"Token cache is missing fields: {sorted(missing_cache)}")
    caa_archive = load_archive(CAA_PATH, "CAA")
    source_dataset = caa_archive.get("source_probe_dataset_path")
    if source_dataset is None or realpath(str(source_dataset)) != realpath(probe.DATASET_PATH):
        raise RuntimeError("CAA archive points to a different probe dataset.")
    source_checkpoint = caa_archive.get("source_checkpoint_path")
    if source_checkpoint is None or realpath(str(source_checkpoint)) != realpath(str(data["source_checkpoint_path"])):
        raise RuntimeError("CAA archive and probe dataset refer to different model checkpoints.")
    archives: dict[str, dict[str, Any]] = {"caa": caa_archive}
    banks: dict[str, dict[str, SteeringVector]] = {
        "caa": canonicalise_caa_vectors(caa_archive),
    }
    paths: dict[str, str] = {"caa": CAA_PATH}
    if "probe" in methods:
        probe_archive = load_archive(PROBE_PATH, "Probe")
        archives["probe"] = probe_archive
        banks["probe"] = canonicalise_probe_vectors(probe_archive)
        paths["probe"] = PROBE_PATH
    if "direct" in methods:
        direct_archive = load_archive(DIRECT_PATH, "Direct Optimisation")
        archives["direct"] = direct_archive
        banks["direct"] = canonicalise_direct_vectors(direct_archive)
        paths["direct"] = DIRECT_PATH
    expected_dataset = realpath(probe.DATASET_PATH)
    expected_checkpoint = realpath(str(data["source_checkpoint_path"]))
    for method, archive in archives.items():
        archive_dataset = archive.get("source_probe_dataset_path")
        archive_checkpoint = archive.get("source_checkpoint_path")
        if archive_dataset is None or realpath(str(archive_dataset)) != expected_dataset:
            raise RuntimeError(f"{method} archive points to a different probe dataset.")
        if archive_checkpoint is not None and realpath(str(archive_checkpoint)) != expected_checkpoint:
            raise RuntimeError(f"{method} archive points to a different model checkpoint.")
        if method != "probe" and archive_checkpoint is None:
            raise RuntimeError(f"{method} archive does not identify its model checkpoint.")
    tok = pm.Tokenizer()
    model, model_config, model_max_len = load_model(data, tok, device)
    return {
        "data": data,
        "evaluated": evaluated,
        "token_cache": token_cache,
        "tok": tok,
        "model": model,
        "model_config": model_config,
        "model_max_len": model_max_len,
        "paths": paths,
        "archives": archives,
        "banks": banks,
        "vector_audit": {
            f"{method}_attention_vectors": len(bank)
            for method, bank in banks.items()
        },
        "trace_outcome_lookup": probe.build_trace_outcome_lookup(data),
        "unit_tables": {},
    }


def select_rows(rows: torch.Tensor, cap: int, offset: int, seed: int) -> torch.Tensor:
    rows = rows.to(torch.int64)
    if offset < 0 or cap <= 0:
        raise ValueError("Selection offset must be non-negative and cap must be positive.")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(rows.numel(), generator=generator)
    selected = rows[order[offset:offset + cap]]
    if selected.numel() != cap:
        raise RuntimeError(
            f"Requested {cap} rows at offset {offset}, but only {selected.numel()} are available."
        )
    return selected


def pack_sequences(sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.int32)
    offsets = torch.zeros(len(sequences), dtype=torch.int64)
    if lengths.numel() > 1:
        offsets[1:] = torch.cumsum(lengths[:-1].to(torch.int64), dim=0)
    flat = torch.tensor([token for sequence in sequences for token in sequence], dtype=torch.int16)
    return flat, offsets, lengths


def unpack_sequence(
    flat: torch.Tensor, offsets: torch.Tensor, lengths: torch.Tensor, index: int,
) -> list[int]:
    start, length = int(offsets[index]), int(lengths[index])
    return flat[start:start + length].to(torch.int64).tolist()


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> float:
    values, weights = values.to(torch.float64), weights.to(torch.float64)
    return float((values * weights).sum() / weights.sum().clamp_min(1e-12))


def weighted_count_rate(
    numerator: torch.Tensor, denominator: torch.Tensor, weights: torch.Tensor,
) -> float:
    numerator = numerator.to(torch.float64)
    denominator = denominator.to(torch.float64)
    weights = weights.to(torch.float64)
    total = (denominator * weights).sum()
    return float((numerator * weights).sum() / total) if float(total) > 0 else 0.0


def broadcast_delta(delta: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    if delta.ndim == 1:
        return delta.to(hidden.dtype).view(1, 1, -1)
    if delta.ndim == 2 and delta.shape[0] == hidden.shape[0]:
        return delta.to(hidden.dtype).unsqueeze(1)
    raise RuntimeError(
        f"Cannot broadcast delta shape {tuple(delta.shape)} over hidden shape {tuple(hidden.shape)}."
    )


def install_attention_hooks(
    model: torch.nn.Module,
    vectors: list[SteeringVector],
    deltas: list[torch.Tensor],
) -> HookGroup:
    """Install one pre-hook per layer, even when several heads are steered."""
    if len(vectors) != len(deltas):
        raise RuntimeError("Vector and delta counts differ.")
    by_layer: dict[int, list[tuple[int, torch.Tensor]]] = defaultdict(list)
    for vector, delta in zip(vectors, deltas):
        if vector.representation_type != "attention_head":
            raise RuntimeError(f"Standalone ablation only supports attention heads, got {vector.representation_type}.")
        by_layer[vector.layer].append((vector.head, delta))

    handles = []
    for layer, interventions in sorted(by_layer.items()):
        block = model.blocks[layer]
        head_dim = int(block.head_dim)

        def hook(
            _module: torch.nn.Module,
            inputs: tuple[Any, ...],
            interventions: tuple[tuple[int, torch.Tensor], ...] = tuple(interventions),
            head_dim: int = head_dim,
        ) -> tuple[Any, ...]:
            hidden = inputs[0].clone()
            for head, delta in interventions:
                start, end = head * head_dim, (head + 1) * head_dim
                head_hidden = hidden[:, :, start:end]
                hidden[:, :, start:end] = head_hidden + broadcast_delta(delta, head_hidden)
            return (hidden, *inputs[1:])

        handles.append(block.out_proj.register_forward_pre_hook(hook))
    return HookGroup(handles)


@torch.inference_mode()
def build_prefix_batches(
    model: torch.nn.Module,
    units: list[PrefixUnit],
    max_output_tokens: int,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    max_new_override: int,
) -> list[PrefixBatch]:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, unit in enumerate(units):
        remaining = max(0, max_output_tokens - len(unit.generated_prefix))
        if max_new_override > 0:
            remaining = min(remaining, max_new_override)
        groups[(len(unit.prefix_ids), remaining)].append(index)

    batches: list[PrefixBatch] = []
    for (prefix_len, max_new), group_indices in groups.items():
        for start in range(0, len(group_indices), batch_size):
            selected = tuple(group_indices[start:start + batch_size])
            prefix_tensor = torch.tensor(
                [units[index].prefix_ids for index in selected], dtype=torch.long, device=device,
            )
            past_kvs, total_len = None, 0
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                for position in range(prefix_len - 1):
                    x_step = prefix_tensor[:, position:position + 1]
                    _, past_kvs = model.forward_step(
                        x_step, min(total_len, model.max_len - 1), past_kvs,
                    )
                    total_len += 1
            batches.append(PrefixBatch(
                selected=selected,
                max_new=max_new,
                last_token=prefix_tensor[:, -1:],
                past_kvs=past_kvs,
                total_len=total_len,
            ))
    return batches


@torch.inference_mode()
def generate_suffixes(
    model: torch.nn.Module,
    units: list[PrefixUnit],
    prefix_batches: list[PrefixBatch],
    vectors: list[SteeringVector],
    deltas: list[torch.Tensor],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    eos_id: int,
    pad_id: int,
) -> list[list[int]]:
    outputs: list[list[int]] = [[] for _ in units]
    stop_ids = torch.tensor([int(eos_id), int(pad_id)], dtype=torch.long, device=device)

    for prefix_batch in prefix_batches:
        selected = prefix_batch.selected
        past_kvs = None if prefix_batch.past_kvs is None else [
            (key.clone(), value.clone()) for key, value in prefix_batch.past_kvs
        ]
        total_len = prefix_batch.total_len
        hooks = install_attention_hooks(model, vectors, deltas) if vectors else None
        try:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits, past_kvs = model.forward_step(
                    prefix_batch.last_token,
                    min(total_len, model.max_len - 1),
                    past_kvs,
                )
        finally:
            if hooks is not None:
                hooks.remove()

        last_logits = logits[:, -1, :].float()
        total_len += 1
        batch_count = len(selected)
        generated = torch.full(
            (batch_count, prefix_batch.max_new), int(pad_id), dtype=torch.long, device=device,
        )
        lengths = torch.zeros(batch_count, dtype=torch.int64, device=device)
        active = torch.ones(batch_count, dtype=torch.bool, device=device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            for step in range(prefix_batch.max_new):
                next_tokens = last_logits.argmax(dim=-1)
                stop = (next_tokens.unsqueeze(1) == stop_ids.unsqueeze(0)).any(dim=1)
                emit = active & ~stop
                generated[:, step] = torch.where(emit, next_tokens, generated[:, step])
                lengths += emit.to(torch.int64)
                active &= ~stop
                x_step = torch.where(
                    active, next_tokens, torch.full_like(next_tokens, int(pad_id)),
                ).unsqueeze(1)
                logits, past_kvs = model.forward_step(
                    x_step, min(total_len, model.max_len - 1), past_kvs,
                )
                last_logits = logits[:, -1, :].float()
                total_len += 1
                if (step + 1) % 16 == 0 and not bool(active.any()):
                    break

        generated_cpu, lengths_cpu = generated.cpu(), lengths.cpu()
        for local_index, global_index in enumerate(selected):
            outputs[global_index] = generated_cpu[
                local_index, :int(lengths_cpu[local_index])
            ].tolist()
    return outputs


@functools.lru_cache(maxsize=200000)
def state_is_viable(state: tuple[int, ...], target: int) -> bool:
    return reachability.minimum_steps_for_targets(state, (target,))[target] >= 0


def rollout_viability(
    generated_ids: list[int],
    intervention_index: int,
    initial_state: tuple[int, ...],
    target: int,
    contract: Any,
) -> dict[str, Any]:
    """Score the first completed post-intervention state as valid and reachable."""
    active_state = tuple(sorted(initial_state))
    valid_path = [active_state]
    prefix_valid = True
    future_viability: list[bool] = []
    pos = 0
    while pos < len(generated_ids):
        token_id = generated_ids[pos]
        if token_id == contract.EXPR:
            break
        if token_id == contract.BACK:
            parsed = reachability.parse_back_at(generated_ids, pos, contract)
            if parsed is None:
                prefix_valid = False
                pos += 1
                continue
            allowable = valid_path[:-1] if prefix_valid else valid_path
            restore_index = next(
                (index for index in range(len(allowable) - 1, -1, -1)
                 if allowable[index] == parsed["restored_state"]),
                -1,
            )
            prefix_valid = restore_index >= 0
            active_state = parsed["restored_state"]
            if prefix_valid:
                valid_path = valid_path[:restore_index + 1]
            if parsed["semicolon_pos"] > intervention_index:
                future_viability.append(prefix_valid and state_is_viable(active_state, target))
                break
            pos = parsed["next_pos"]
            continue
        if token_id == contract.SUCCESS:
            break
        parsed = reachability.parse_operation_at(generated_ids, pos, contract) if token_id == contract.LBR else None
        if parsed is None:
            prefix_valid = False
            pos += 1
            continue
        expected_result = reachability.arithmetic_result(parsed["n1"], parsed["operator"], parsed["n2"])
        expected_after = reachability.apply_operation(
            active_state, parsed["n1"], parsed["n2"], parsed["result"],
        )
        operation_valid = (
            prefix_valid
            and parsed["stated_before"] == active_state
            and reachability.operands_available(active_state, parsed["n1"], parsed["n2"])
            and expected_result == parsed["result"]
            and expected_after == parsed["stated_after"]
        )
        active_state = parsed["stated_after"]
        prefix_valid = operation_valid
        if operation_valid:
            valid_path.append(active_state)
        if parsed["semicolon_pos"] > intervention_index:
            future_viability.append(operation_valid and state_is_viable(active_state, target))
            break
        pos = parsed["next_pos"]
    observed = bool(future_viability)
    return {
        "next_state_observed": observed,
        "next_state_viable": future_viability[0] if observed else False,
        "ever_future_unreachable": observed and not all(future_viability),
        "future_reachable_state_fraction": sum(future_viability) / len(future_viability) if observed else 0.0,
    }


def score_suffixes(
    units: list[PrefixUnit], suffixes: list[list[int]], tok: Any,
) -> dict[str, Any]:
    labels: dict[str, list[Any]] = {
        name: [] for name in (*BOOL_SCORE_FIELDS, *INT_SCORE_FIELDS, *FLOAT_SCORE_FIELDS)
    }
    full_sequences: list[list[int]] = []
    contract = reachability.TokenContract(tok)
    score_reachability = bool(units) and units[0].task in REACHABILITY_TASKS
    for unit, suffix in zip(units, suffixes):
        generated_ids = list(unit.generated_prefix) + suffix
        generated_text = tok.decode(generated_ids).strip() if generated_ids else ""
        result = evaluator.label_generation(
            prompt_text=unit.prompt_text,
            gold_text=unit.gold_text,
            generated_text=generated_text,
            target=unit.target,
            gold_back_count=unit.gold_back_count,
        )
        result["generated_length"] = len(generated_ids)
        if score_reachability:
            result.update(rollout_viability(
                generated_ids, unit.generated_idx, unit.initial_state, unit.target, contract,
            ))
        else:
            result.update(
                next_state_viable=False,
                next_state_observed=False,
                ever_future_unreachable=False,
                future_reachable_state_fraction=0.0,
            )
        full_sequences.append(generated_ids)
        for name in BOOL_SCORE_FIELDS:
            labels[name].append(bool(result[name]))
        for name in INT_SCORE_FIELDS:
            labels[name].append(int(result[name]))
        for name in FLOAT_SCORE_FIELDS:
            labels[name].append(float(result[name]))
    tensors = {
        name: torch.tensor(values, dtype=torch.bool)
        for name, values in labels.items() if name in BOOL_SCORE_FIELDS
    }
    tensors.update({name: torch.tensor(labels[name], dtype=torch.int32) for name in INT_SCORE_FIELDS})
    tensors.update({name: torch.tensor(labels[name], dtype=torch.float32) for name in FLOAT_SCORE_FIELDS})
    return {"labels": tensors, "full_sequences": full_sequences, "suffixes": suffixes}


def cluster_bootstrap_delta(
    record_pos: torch.Tensor,
    delta: torch.Tensor,
    weights: torch.Tensor,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    unique, inverse = torch.unique(record_pos.to(torch.int64), sorted=True, return_inverse=True)
    weighted_delta = delta.to(torch.float64) * weights.to(torch.float64)
    trace_numerator = torch.zeros(unique.numel(), dtype=torch.float64)
    trace_denominator = torch.zeros(unique.numel(), dtype=torch.float64)
    trace_numerator.scatter_add_(0, inverse, weighted_delta)
    trace_denominator.scatter_add_(0, inverse, weights.to(torch.float64))
    generator = torch.Generator().manual_seed(seed)
    estimates = torch.empty(samples, dtype=torch.float64)
    for start in range(0, samples, 256):
        end = min(start + 256, samples)
        draw = torch.randint(unique.numel(), (end - start, unique.numel()), generator=generator)
        numerator = trace_numerator[draw].sum(dim=1)
        denominator = trace_denominator[draw].sum(dim=1).clamp_min(1e-12)
        estimates[start:end] = numerator / denominator
    return float(torch.quantile(estimates, 0.025)), float(torch.quantile(estimates, 0.975))


def summarise_result(
    units: list[PrefixUnit],
    baseline: dict[str, Any],
    result: dict[str, Any],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    weights = torch.tensor([unit.weight for unit in units], dtype=torch.float64)
    base, steered = baseline["labels"], result["labels"]
    summary: dict[str, Any] = {
        "units": len(units),
        "unique_traces": len({unit.record_pos for unit in units}),
    }
    if not bool(torch.allclose(weights, torch.ones_like(weights))):
        raise RuntimeError(
            "Exact paired problem counts require one unit of weight per ChainSearch problem."
        )
    for field in BOOL_SCORE_FIELDS:
        baseline_rate = weighted_mean(base[field].float(), weights)
        steered_rate = weighted_mean(steered[field].float(), weights)
        summary[f"baseline_{field}_rate"] = baseline_rate
        summary[f"steered_{field}_rate"] = steered_rate
        summary[f"delta_{field}_rate"] = steered_rate - baseline_rate
        gain = (~base[field]) & steered[field]
        loss = base[field] & (~steered[field])
        summary[f"{field}_gain_count"] = int(gain.sum())
        summary[f"{field}_loss_count"] = int(loss.sum())
        summary[f"{field}_net_gain_count"] = int(gain.sum() - loss.sum())
        summary[f"{field}_unchanged_count"] = int((base[field] == steered[field]).sum())
    baseline_correct, steered_correct = base["expr_target_correct"], steered["expr_target_correct"]
    rescue = (~baseline_correct) & steered_correct
    damage = baseline_correct & (~steered_correct)
    summary["rescue_rate"] = weighted_mean(rescue.float(), weights)
    summary["damage_rate"] = weighted_mean(damage.float(), weights)
    summary["net_rescue_rate"] = summary["rescue_rate"] - summary["damage_rate"]
    summary["rescue_count"] = int(rescue.sum())
    summary["damage_count"] = int(damage.sum())
    summary["net_rescue_count"] = int(rescue.sum() - damage.sum())
    continuation_changed = torch.tensor(
        [suffix != baseline_suffix
         for suffix, baseline_suffix in zip(result["suffixes"], baseline["suffixes"])],
        dtype=torch.bool,
    )
    summary["continuation_change_rate"] = weighted_mean(continuation_changed.float(), weights)
    for field in ("generated_length", "generated_back_count"):
        summary[f"baseline_{field}_mean"] = weighted_mean(base[field].float(), weights)
        summary[f"steered_{field}_mean"] = weighted_mean(steered[field].float(), weights)
        summary[f"delta_{field}_mean"] = (
            summary[f"steered_{field}_mean"] - summary[f"baseline_{field}_mean"]
        )
    field = "future_reachable_state_fraction"
    summary[f"baseline_{field}_mean"] = weighted_mean(base[field], weights)
    summary[f"steered_{field}_mean"] = weighted_mean(steered[field], weights)
    summary[f"delta_{field}_mean"] = summary[f"steered_{field}_mean"] - summary[f"baseline_{field}_mean"]
    summary["baseline_operation_validity_rate"] = weighted_count_rate(
        base["op_valid_count"], base["op_total"], weights,
    )
    summary["steered_operation_validity_rate"] = weighted_count_rate(
        steered["op_valid_count"], steered["op_total"], weights,
    )
    summary["delta_operation_validity_rate"] = (
        summary["steered_operation_validity_rate"] - summary["baseline_operation_validity_rate"]
    )
    summary["baseline_state_consistency_rate"] = weighted_count_rate(
        base["state_consistent_count"], base["state_total"], weights,
    )
    summary["steered_state_consistency_rate"] = weighted_count_rate(
        steered["state_consistent_count"], steered["state_total"], weights,
    )
    summary["delta_state_consistency_rate"] = (
        summary["steered_state_consistency_rate"] - summary["baseline_state_consistency_rate"]
    )
    summary["baseline_failure_counts"] = dict(sorted(
        (FAILURE_NAMES[int(code)], int((base["failure_category"] == code).sum()))
        for code in torch.unique(base["failure_category"])
    ))
    summary["steered_failure_counts"] = dict(sorted(
        (FAILURE_NAMES[int(code)], int((steered["failure_category"] == code).sum()))
        for code in torch.unique(steered["failure_category"])
    ))
    if bootstrap_samples > 0:
        for field in ("expr_target_correct", "solved_valid_trace", "valid_trace", "next_state_viable"):
            low, high = cluster_bootstrap_delta(
                torch.tensor([unit.record_pos for unit in units]),
                steered[field].float() - base[field].float(),
                weights,
                bootstrap_samples,
                seed + stable_seed(field),
            )
            summary[f"delta_{field}_ci_low"] = low
            summary[f"delta_{field}_ci_high"] = high
        if units[0].task in REACHABILITY_TASKS:
            field = "future_reachable_state_fraction"
            low, high = cluster_bootstrap_delta(
                torch.tensor([unit.record_pos for unit in units]),
                steered[field] - base[field],
                weights,
                bootstrap_samples,
                seed + stable_seed(field),
            )
            summary[f"delta_{field}_ci_low"] = low
            summary[f"delta_{field}_ci_high"] = high
    return summary


def result_shard(
    phase: str,
    condition_id: str,
    condition: SteeringCondition | None,
    units: list[PrefixUnit],
    result: dict[str, Any],
    summary: dict[str, Any],
    save_tokens: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": 2,
        "complete": True,
        "run_tag": COMPARISON_RUN_TAG,
        "phase": phase,
        "condition_id": condition_id,
        "condition": condition.__dict__ if condition is not None else None,
        "source_row": torch.tensor([unit.source_row for unit in units], dtype=torch.int64),
        "record_pos": torch.tensor([unit.record_pos for unit in units], dtype=torch.int32),
        "cache_idx": torch.tensor([unit.cache_idx for unit in units], dtype=torch.int64),
        "generated_idx": torch.tensor([unit.generated_idx for unit in units], dtype=torch.int16),
        "weights": torch.tensor([unit.weight for unit in units], dtype=torch.float32),
        "labels": result["labels"],
        "summary": summary,
    }
    if save_tokens:
        flat, offsets, lengths = pack_sequences(result["full_sequences"])
        suffix_flat, suffix_offsets, suffix_lengths = pack_sequences(result["suffixes"])
        payload.update(
            generated_flat_ids=flat,
            generated_offsets=offsets,
            generated_lengths=lengths,
            suffix_flat_ids=suffix_flat,
            suffix_offsets=suffix_offsets,
            suffix_lengths=suffix_lengths,
        )
    return payload


def shard_path(phase: str, identifier: str) -> str:
    return os.path.join(SHARD_DIR, f"{phase}_{condition_hash(identifier)}.pt")


def load_shard(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    shard = torch.load(path, map_location="cpu", weights_only=False)
    if not bool(shard.get("complete", False)) or shard.get("run_tag") != COMPARISON_RUN_TAG:
        raise RuntimeError(f"Invalid attention-head comparison shard: {path}")
    return shard


def generate_and_score(
    sources: dict[str, Any],
    units: list[PrefixUnit],
    prefix_batches: list[PrefixBatch],
    vector_lookup: dict[tuple[str, str], SteeringVector],
    condition: SteeringCondition | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    vectors: list[SteeringVector] = []
    deltas: list[torch.Tensor] = []
    if condition is not None:
        keys = condition.member_keys or (condition.vector_key,)
        scales = condition.member_scales or (condition.scale,)
        if len(keys) != len(scales):
            raise RuntimeError(f"Condition {condition.condition_id} has mismatched members and scales.")
        vectors = [vector_lookup[(condition.method, key)] for key in keys]
        deltas = [
            (condition.beta * scale * vector.direction).to(args.device)
            for vector, scale in zip(vectors, scales)
        ]
    suffixes = generate_suffixes(
        sources["model"], units, prefix_batches, vectors, deltas,
        args.device, args.use_amp, args.amp_dtype,
        sources["tok"].eos_id, sources["tok"].pad_id,
    )
    return score_suffixes(units, suffixes, sources["tok"])


def baseline_identifier(split: str, task: str, checkpoint: str) -> str:
    return f"baseline:{split}:{task}:{checkpoint}"


def ensure_baseline(
    sources: dict[str, Any],
    units: list[PrefixUnit],
    prefix_batches: list[PrefixBatch],
    split: str,
    task: str,
    checkpoint: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    identifier = baseline_identifier(split, task, checkpoint)
    path = shard_path("baseline", identifier)
    if os.path.isfile(path):
        shard = load_shard(path)
        expected_ids = torch.tensor([unit.cache_idx for unit in units], dtype=torch.int64)
        if not torch.equal(shard["cache_idx"].to(torch.int64), expected_ids):
            raise RuntimeError(f"Baseline shard problem IDs do not match {split}/{task}/{checkpoint}.")
        return shard, shard["summary"], path
    result = generate_and_score(sources, units, prefix_batches, {}, None, args)
    summary = summarise_result(units, result, result, 0, stable_seed(identifier))
    reproduction = torch.tensor(
        [suffix == list(unit.original_suffix) for unit, suffix in zip(units, result["suffixes"])],
        dtype=torch.float32,
    )
    next_token_reproduction = torch.tensor([
        (suffix[0] if suffix else None) ==
        (unit.original_suffix[0] if unit.original_suffix else None)
        for unit, suffix in zip(units, result["suffixes"])
    ], dtype=torch.float32)
    unit_weights = torch.tensor([unit.weight for unit in units], dtype=torch.float64)
    summary["original_suffix_reproduction_rate"] = weighted_mean(reproduction, unit_weights)
    summary["original_next_token_reproduction_rate"] = weighted_mean(
        next_token_reproduction, unit_weights,
    )
    atomic_save(result_shard("baseline", identifier, None, units, result, summary, True), path)
    return load_shard(path), summary, path


def baseline_result(shard: dict[str, Any]) -> dict[str, Any]:
    suffixes = [
        unpack_sequence(
            shard["suffix_flat_ids"], shard["suffix_offsets"], shard["suffix_lengths"], index,
        )
        for index in range(int(shard["suffix_lengths"].numel()))
    ]
    return {"labels": shard["labels"], "full_sequences": [], "suffixes": suffixes}


def evaluate_condition(
    sources: dict[str, Any],
    units: list[PrefixUnit],
    prefix_batches: list[PrefixBatch],
    baseline: dict[str, Any],
    vector_lookup: dict[tuple[str, str], SteeringVector],
    condition: SteeringCondition,
    phase: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    result = generate_and_score(sources, units, prefix_batches, vector_lookup, condition, args)
    summary = summarise_result(
        units,
        baseline,
        result,
        args.bootstrap_samples if phase == "test" else 0,
        stable_seed(phase, condition.condition_id),
    )
    row = {**condition.__dict__, **summary, "phase": phase}
    save_shard = phase == "test" or args.save_validation_tokens
    path = shard_path(phase, condition.condition_id) if save_shard else ""
    if save_shard:
        atomic_save(
            result_shard(phase, condition.condition_id, condition, units, result, summary, True),
            path,
        )
    row["shard_path"] = path or None
    return row, path


def primary_effect(row: dict[str, Any]) -> float:
    if row["label_family"] == "reachability":
        return float(row["delta_next_state_viable_rate"])
    return float(row["delta_expr_target_correct_rate"])


# =============================================================================
# EXPERIMENT CONTRACT
# =============================================================================

TASK_BY_TRACK = {
    "reachability": "operation_reachability",
    "final_answer": "trace_outcome_operation_decisions",
}

OPERATION_CHECKPOINTS = (
    "pre_first_operand",
    "first_operand_end",
    "operator",
    "second_operand_end",
    "result_end",
)

CHECKPOINT_DISPLAY = {
    "pre_first_operand": "Before first operand",
    "first_operand_end": "First operand",
    "operator": "Operator",
    "second_operand_end": "Second operand",
    "result_end": "Result",
}

METHODS = ("probe", "caa", "direct")
METHOD_DISPLAY = {
    "probe": "Probe",
    "caa": "CAA",
    "direct": "Direct",
}
METHOD_COLORS = {
    "probe": "#1C4587",
    "caa": "#7CA6DE",
    "direct": "#D57758",
}

# Paper figure palette. Keep aligned with plot_probe_results.py.
PAPER_FIGURE_BG = "#FCFCFD"
PAPER_CANVAS_BG = "#FFFFFF"
PAPER_PANEL_BG = "#F2F6FF"
PAPER_DEEP_BLUE = "#1C4587"
PAPER_GAIN_FILL = "#7EA5D6"
PAPER_LOSS_FILL = "#E8A98B"
PAPER_DARK_TEXT = "#26364F"
PAPER_GRID = "#C9D6EE"
PAPER_REFLINE = "#9AA7BD"
PAPER_TITLE = "#000000"

DEFAULT_BETAS = (0.125, 0.25, 0.5, 1.0, 2.0)
DEFAULT_HEAD_COUNTS = (1, 2, 4, 8)
DEFAULT_REFERENCE_HEAD_COUNT = 4
FORMAT_VERSION = 1
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "chainsearch_archives" / "attention_head_steering_comparison"
)


# =============================================================================
# ARGUMENTS AND PATHS
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a controlled Probe/CAA/Direct attention-head steering comparison."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS),
    )
    parser.add_argument(
        "--tracks",
        nargs="+",
        choices=tuple(TASK_BY_TRACK),
        default=list(TASK_BY_TRACK),
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        choices=OPERATION_CHECKPOINTS,
        default=list(OPERATION_CHECKPOINTS),
    )
    parser.add_argument("--head-counts", nargs="+", type=int, default=list(DEFAULT_HEAD_COUNTS))
    parser.add_argument("--betas", nargs="+", type=float, default=list(DEFAULT_BETAS))
    parser.add_argument("--reference-head-count", type=int, default=DEFAULT_REFERENCE_HEAD_COUNT)
    parser.add_argument("--head-selection-units", type=int, default=128)
    parser.add_argument("--coarse-validation-units", type=int, default=32)
    parser.add_argument("--refined-validation-units", type=int, default=128)
    parser.add_argument("--test-units", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=0,
        help="Optional paired bootstrap replicates for machine-readable rate intervals; plots use exact counts.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--maximum-validation-validity-drop",
        type=float,
        default=0.02,
        help="Reject a validation candidate if valid-trace rate falls by more than this amount.",
    )
    parser.add_argument(
        "--previous-caa-result",
        type=Path,
        default=(
            Path(__file__).resolve().parent.parent
            / "chainsearch_archives" / "attention_head_ablations"
            / "attention_head_ablations.pt"
        ),
        help="Completed inspected CAA run whose test problem IDs must be excluded.",
    )
    parser.add_argument("--minimum-reproduction-rate", type=float, default=0.95)
    parser.add_argument("--hard-minimum-reproduction-rate", type=float, default=0.80)
    parser.add_argument("--save-validation-tokens", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate CSV, summary, and figures from an existing complete result.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.methods = tuple(method for method in METHODS if method in set(args.methods))
    args.tracks = tuple(dict.fromkeys(args.tracks))
    args.checkpoints = tuple(
        checkpoint for checkpoint in OPERATION_CHECKPOINTS if checkpoint in set(args.checkpoints)
    )
    args.head_counts = tuple(sorted(set(args.head_counts)))
    args.betas = tuple(sorted(set(float(beta) for beta in args.betas)))

    if not args.methods or not args.checkpoints:
        raise ValueError("At least one method and checkpoint are required.")
    if not args.head_counts or any(count <= 0 for count in args.head_counts):
        raise ValueError("--head-counts must contain positive integers.")
    if not args.betas or any(not math.isfinite(beta) or beta <= 0 for beta in args.betas):
        raise ValueError("--betas must contain finite positive values; negative controls are added automatically.")
    for name in (
        "head_selection_units",
        "coarse_validation_units",
        "refined_validation_units",
        "test_units",
        "reference_head_count",
        "batch_size",
        "save_every",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be non-negative.")
    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be non-negative.")
    if args.reference_head_count not in args.head_counts:
        raise ValueError("--reference-head-count must be one of --head-counts.")
    if not 0.0 <= args.maximum_validation_validity_drop <= 1.0:
        raise ValueError("--maximum-validation-validity-drop must lie in [0, 1].")
    if not 0.0 <= args.hard_minimum_reproduction_rate <= args.minimum_reproduction_rate <= 1.0:
        raise ValueError("Replay thresholds must satisfy 0 <= hard <= preferred <= 1.")

    if args.smoke:
        args.head_counts = (1, 2)
        args.reference_head_count = 2
        args.betas = (0.25, 1.0)
        args.head_selection_units = 128
        args.coarse_validation_units = 2
        args.refined_validation_units = 2
        args.test_units = 2
        args.bootstrap_samples = min(args.bootstrap_samples, 50)
        args.save_every = 1
        args.maximum_validation_validity_drop = 1.0
        if args.output_dir == DEFAULT_OUTPUT_DIR:
            args.output_dir = DEFAULT_OUTPUT_DIR.with_name(
                f"{DEFAULT_OUTPUT_DIR.name}_smoke"
            )

    args.output_dir = args.output_dir.resolve()
    args.previous_caa_result = args.previous_caa_result.resolve()
    if args.plot_only:
        args.device = "cpu"
    args.device = torch.device(args.device)
    if args.device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        torch.cuda.set_device(args.device)
    args.use_amp = args.device.type == "cuda"
    args.amp_dtype = (
        torch.bfloat16
        if args.use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    return args


def experiment_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "result": args.output_dir / "attention_head_steering_comparison.pt",
        "partial": args.output_dir / "attention_head_steering_comparison.partial.pt",
        "shards": args.output_dir / "generation_shards",
        "ranking_csv": args.output_dir / "common_caa_head_ranking.csv",
        "coarse_csv": args.output_dir / "coarse_validation_dose_results.csv",
        "refined_csv": args.output_dir / "refined_validation_position_results.csv",
        "effects_csv": args.output_dir / "heldout_method_comparison_results.csv",
        "transition_csv": args.output_dir / "heldout_changed_problem_manifest.csv",
        "summary": args.output_dir / "attention_head_steering_comparison_summary.txt",
        "figures": args.output_dir / "figures",
    }


def load_excluded_test_problem_ids(
    path: Path,
    tasks: tuple[str, ...],
) -> dict[str, set[int]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"The previous inspected CAA result is required to exclude its test problems: {path}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not bool(payload.get("complete", False)):
        raise RuntimeError(f"Previous CAA result is incomplete: {path}")
    problem_ids = payload.get("problem_ids", {})
    output: dict[str, set[int]] = {}
    for task in tasks:
        key = f"test:{task}"
        if key not in problem_ids:
            raise RuntimeError(f"Previous CAA result has no problem manifest for {key}.")
        output[task] = {int(value) for value in problem_ids[key]}
        if not output[task]:
            raise RuntimeError(f"Previous CAA test manifest is empty for {task}.")
    return output


def archive_legacy_partial(paths: dict[str, Path]) -> None:
    """Preserve, but do not attempt to resume, an incompatible v1 run."""
    partial = paths["partial"]
    if not partial.is_file():
        return
    state = torch.load(partial, map_location="cpu", weights_only=False)
    version = int(state.get("format_version", 0)) if isinstance(state, dict) else 0
    if version == FORMAT_VERSION:
        return
    suffix = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    archived_partial = partial.with_name(f"{partial.stem}.legacy-v{version}-{suffix}{partial.suffix}")
    os.replace(partial, archived_partial)
    archived_shards = None
    if paths["shards"].is_dir():
        archived_shards = paths["shards"].with_name(
            f"{paths['shards'].name}.legacy-v{version}-{suffix}"
        )
        os.replace(paths["shards"], archived_shards)
    print(
        f"Archived incompatible v{version} partial state at {archived_partial}"
        + (f" and shards at {archived_shards}." if archived_shards else "."),
        flush=True,
    )


# =============================================================================
# CAA SEPARATION AND HEAD RANKING
# =============================================================================

def unique_problem_rows(
    rows: torch.Tensor,
    units: Any,
    data: dict[str, Any],
) -> torch.Tensor:
    """Keep the earliest operation unit for each distinct ChainSearch problem."""
    rows = rows.to(torch.int64)
    activation = units.activation_index[rows, 0].to(torch.int64)
    problem = units.cache_idx[rows].to(torch.int64)
    position = data["activation_positions"]["generated_token_idx"][activation].to(torch.int64)
    order = torch.argsort(position, stable=True)
    order = order[torch.argsort(problem[order], stable=True)]
    ordered_problem = problem[order]
    first = torch.ones(order.numel(), dtype=torch.bool)
    if first.numel() > 1:
        first[1:] = ordered_problem[1:] != ordered_problem[:-1]
    return rows[order[first]]


def build_problem_prefix_units(
    sources: dict[str, Any],
    task: str,
    checkpoint: str,
    split: str,
    cap: int,
    offset: int,
    excluded_problem_ids: set[int] | None = None,
) -> list[Any]:
    """Build one deterministic prefix per distinct ChainSearch problem."""
    data = sources["data"]
    evaluated = sources["evaluated"]
    cache = sources["token_cache"]
    tok = sources["tok"]
    trace_label, trace_known = sources["trace_outcome_lookup"]
    if task not in sources["unit_tables"]:
        sources["unit_tables"][task] = probe.build_unit_table(
            data,
            task,
            trace_label,
            trace_known,
        )
    units = sources["unit_tables"][task]
    checkpoint_index = units.checkpoints.index(checkpoint)
    split_to_id = {str(name): int(value) for name, value in data["split_to_id"].items()}
    rows = probe.split_unit_rows(units, split_to_id)[split]

    activation = units.activation_index[rows, checkpoint_index].to(torch.int64)
    problem = units.cache_idx[rows].to(torch.int64)
    position = data["activation_positions"]["generated_token_idx"][activation].to(torch.int64)
    order = torch.argsort(position, stable=True)
    order = order[torch.argsort(problem[order], stable=True)]
    ordered_problem = problem[order]
    first = torch.ones(order.numel(), dtype=torch.bool)
    if first.numel() > 1:
        first[1:] = ordered_problem[1:] != ordered_problem[:-1]
    rows = rows[order[first]]
    if excluded_problem_ids:
        keep = torch.tensor(
            [int(units.cache_idx[row]) not in excluded_problem_ids for row in rows.tolist()],
            dtype=torch.bool,
        )
        rows = rows[keep]
    rows = select_rows(
        rows,
        cap,
        offset,
        stable_seed(task, split, "problem_unit_order"),
    )

    record_pos = units.record_pos[rows].to(torch.int64)
    activation_index = units.activation_index[rows, checkpoint_index].to(torch.int64)
    positions = data["activation_positions"]
    if not torch.equal(
        positions["record_pos"][activation_index].to(torch.int64),
        record_pos,
    ):
        raise RuntimeError(f"Activation-position record mismatch for {task}/{checkpoint}.")
    generated_idx = positions["generated_token_idx"][activation_index].to(torch.int64)

    cache_parts: dict[int, tuple[list[int], str, str, int]] = {}
    output = []
    for local, row in enumerate(rows.tolist()):
        record = int(record_pos[local])
        cache_idx = int(units.cache_idx[row])
        if cache_idx not in cache_parts:
            prompt_ids, gold_ids = evaluator.cache_row_parts(cache, cache_idx, tok)
            cache_parts[cache_idx] = (
                prompt_ids,
                tok.decode(prompt_ids[:-1]).strip(),
                tok.decode(gold_ids).strip(),
                int(cache["bk_counts"][cache_idx]),
            )
        prompt_ids, prompt_text, gold_text, gold_back = cache_parts[cache_idx]
        original = builder.generated_ids_at(evaluated, record)
        position_index = int(generated_idx[local])
        if not 0 <= position_index < len(original):
            raise RuntimeError(
                f"Invalid generated checkpoint index {position_index} for record {record}."
            )
        output.append(PrefixUnit(
            task=task,
            source_row=int(units.source_row[row]),
            record_pos=record,
            cache_idx=cache_idx,
            target=int(units.target[row]),
            generated_idx=position_index,
            prompt_ids=tuple(prompt_ids),
            generated_prefix=tuple(original[:position_index + 1]),
            original_suffix=tuple(original[position_index + 1:]),
            prompt_text=prompt_text,
            gold_text=gold_text,
            gold_back_count=gold_back,
            weight=1.0,
            initial_state=tuple(
                sorted(int(value) for value in evaluated["input_numbers"][record].tolist())
            ),
        ))
    if len(output) != cap or len({unit.cache_idx for unit in output}) != cap:
        raise RuntimeError(
            f"Expected {cap} unique ChainSearch problems for {task}/{checkpoint}/{split}, "
            f"found {len({unit.cache_idx for unit in output})}."
        )
    return output


def rank_percentiles(values: list[float]) -> list[float]:
    if len(values) <= 1:
        return [1.0] * len(values)
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    output = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and math.isclose(
            values[order[end]], values[order[start]], rel_tol=0.0, abs_tol=1e-12
        ):
            end += 1
        average_rank = 0.5 * (start + end - 1)
        percentile = average_rank / (len(order) - 1)
        for local in range(start, end):
            output[order[local]] = percentile
        start = end
    return output


def load_selected_attention_features(
    data: dict[str, Any],
    requested: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Read only requested attention positions while scanning each shard once."""
    if not requested:
        return {}
    flattened = [indices.to(torch.int64).reshape(-1) for indices in requested.values()]
    unique_indices = torch.unique(torch.cat(flattened), sorted=True)
    expected_shape = tuple(data["attention_head_output_pre_projection_shape"])
    if expected_shape[1:] != (4, 8, 32):
        raise RuntimeError(f"Unexpected attention activation shape: {expected_shape}")
    if unique_indices.numel() == 0:
        raise RuntimeError("No attention activation positions were requested.")
    if int(unique_indices[0]) < 0 or int(unique_indices[-1]) >= expected_shape[0]:
        raise IndexError("A requested attention activation index is out of range.")

    selected = torch.empty((unique_indices.numel(), 4, 8, 32), dtype=torch.float16)
    filled = torch.zeros(unique_indices.numel(), dtype=torch.bool)
    shard_dir = Path(data["activation_shard_dir"])
    expected_start = 0
    for shard_info in tqdm(
        data["activation_shards"],
        desc="Scanning attention activation shards",
        dynamic_ncols=True,
    ):
        start, end = int(shard_info["start"]), int(shard_info["end"])
        if start != expected_start or end <= start:
            raise RuntimeError("Activation shards contain a gap or overlap.")
        path = shard_dir / str(shard_info["file"])
        try:
            shard = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except (TypeError, RuntimeError):
            shard = torch.load(path, map_location="cpu", weights_only=False)
        if int(shard["start"]) != start or int(shard["end"]) != end:
            raise RuntimeError(f"Activation shard range mismatch in {path}.")
        mask = (unique_indices >= start) & (unique_indices < end)
        slots = torch.nonzero(mask, as_tuple=False).flatten()
        if slots.numel():
            local = unique_indices[slots] - start
            attention = shard["attention_head_output_pre_projection"]
            if attention.dtype != torch.float16 or tuple(attention.shape[1:]) != (4, 8, 32):
                raise RuntimeError(f"Unexpected attention tensor in {path}.")
            values = attention[local]
            if not bool(torch.isfinite(values).all()):
                raise RuntimeError(f"Selected attention activations are non-finite in {path}.")
            selected.index_copy_(0, slots, values)
            filled[slots] = True
            del attention, values
        expected_start = end
        del shard
    if expected_start != expected_shape[0] or not bool(filled.all()):
        raise RuntimeError("Activation shards did not cover every requested attention position.")
    if not bool(torch.isfinite(selected).all()):
        raise RuntimeError("Collected attention-feature buffer contains non-finite values.")


    output: dict[str, torch.Tensor] = {}
    for name, indices in requested.items():
        slots = torch.searchsorted(unique_indices, indices.to(torch.int64).reshape(-1))
        if not torch.equal(unique_indices[slots], indices.to(torch.int64).reshape(-1)):
            raise RuntimeError(f"Could not map selected activation positions for {name}.")
        n_units, n_checkpoints = indices.shape
        output[name] = selected[slots].reshape(n_units, n_checkpoints * 32, 32)
    return output


@torch.inference_mode()
def compute_caa_head_metrics(
    sources: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    data = sources["data"]
    split_to_id = {str(name): int(value) for name, value in data["split_to_id"].items()}
    if "val" not in split_to_id:
        raise KeyError("The probe dataset has no validation split.")

    trace_label, trace_known = probe.build_trace_outcome_lookup(data)
    caa_bank = sources["banks"]["caa"]
    task_specs: dict[str, dict[str, Any]] = {}
    requested: dict[str, torch.Tensor] = {}
    sources["head_selection_problem_ids"] = {}

    for track in args.tracks:
        task = TASK_BY_TRACK[track]
        units = probe.build_unit_table(data, task, trace_label, trace_known)
        sources["unit_tables"][task] = units
        missing = set(args.checkpoints) - set(units.checkpoints)
        if missing:
            raise RuntimeError(f"{task} is missing requested checkpoints: {sorted(missing)}")
        validation_rows = probe.split_unit_rows(units, split_to_id)["val"]
        validation_rows = unique_problem_rows(validation_rows, units, data)
        validation_rows = select_rows(
            validation_rows,
            args.head_selection_units,
            0,
            stable_seed(task, "val", "problem_unit_order"),
        )
        labels = units.label[validation_rows].to(torch.bool)
        if not bool(labels.any()) or not bool((~labels).any()):
            raise RuntimeError(f"Head-selection subset for {task} does not contain both classes.")
        indices = units.activation_index[validation_rows].to(torch.int64)
        task_specs[task] = {
            "track": track,
            "units": units,
            "rows": validation_rows,
            "labels": labels,
        }
        requested[task] = indices
        sources["head_selection_problem_ids"][task] = sorted(
            int(value) for value in units.cache_idx[validation_rows].tolist()
        )

    selected_features = load_selected_attention_features(data, requested)
    rows_out: list[dict[str, Any]] = []

    for track in args.tracks:
        task = TASK_BY_TRACK[track]
        units = task_specs[task]["units"]
        labels = task_specs[task]["labels"]
        # Head ranking is small. Compute it deterministically in CPU float64,
        # avoiding reduced-precision GPU matrix-multiplication paths.
        features = selected_features[task].detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(features).all()):
            raise RuntimeError(f"Non-finite selected features for task {task}.")
        n_locations = int(probe.REPRESENTATION_SPECS["attention_head"].locations)
        directions: list[torch.Tensor] = []
        metadata: list[tuple[str, dict[str, Any], Any]] = []
        for checkpoint in units.checkpoints:
            for location_index in range(n_locations):
                location = probe.location_info("attention_head", location_index)
                key = vector_key(task, checkpoint, location["location_name"])
                vector = caa_bank.get(key)
                if vector is None or vector.zero_norm:
                    raise RuntimeError(f"CAA direction is unavailable for {key}.")
                directions.append(vector.direction)
                metadata.append((checkpoint, location, vector))
        direction_matrix = torch.stack(directions).detach().to(
            device="cpu", dtype=torch.float64
        )
        if not bool(torch.isfinite(direction_matrix).all()):
            raise RuntimeError(f"Non-finite CAA directions for task {task}.")

        scores_tensor = (features * direction_matrix.unsqueeze(0)).sum(dim=-1)
        if not bool(torch.isfinite(scores_tensor).all()):
            raise RuntimeError(
                f"Non-finite CAA projection scores for task {task}; "
                f"max|feature|={features.abs().max().item():.6g}, "
                f"max|direction|={direction_matrix.abs().max().item():.6g}."
            )
        scores = scores_tensor.numpy()
        label_array = labels.cpu().numpy().astype(np.int8)

        for probe_index, (checkpoint, location, vector) in enumerate(metadata):
            if checkpoint not in args.checkpoints:
                continue
            score = scores[:, probe_index]
            positive, negative = score[label_array == 1], score[label_array == 0]
            pooled = math.sqrt(0.5 * (float(positive.var()) + float(negative.var())))
            gap = (float(positive.mean()) - float(negative.mean())) / max(pooled, 1e-12)
            rows_out.append({
                "track": track,
                "task": task,
                "checkpoint": checkpoint,
                "location_name": location["location_name"],
                "layer": int(location["layer"]),
                "head": int(location["head"]),
                "head_selection_units": int(labels.numel()),
                "head_selection_positive": int(labels.sum()),
                "head_selection_negative": int((~labels).sum()),
                "caa_projection_auroc": float(roc_auc_score(label_array, score)),
                "caa_standardized_mean_gap": gap,
                "caa_raw_norm": float(vector.raw_norm),
            })

        del features, direction_matrix, scores_tensor, scores
        if args.device.type == "cuda":
            torch.cuda.empty_cache()

    caa_bank = sources["banks"]["caa"]
    for row in rows_out:
        task = row["task"]
        checkpoint = row["checkpoint"]
        location = row["location_name"]
        checkpoint_index = args.checkpoints.index(checkpoint)
        neighbours = []
        if checkpoint_index > 0:
            neighbours.append(args.checkpoints[checkpoint_index - 1])
        if checkpoint_index + 1 < len(args.checkpoints):
            neighbours.append(args.checkpoints[checkpoint_index + 1])
        vector = caa_bank[vector_key(task, checkpoint, location)].direction
        cosines = [
            float(torch.dot(
                vector,
                caa_bank[vector_key(task, neighbour, location)].direction,
            ))
            for neighbour in neighbours
        ]
        row["local_successive_stability"] = float(np.mean(cosines)) if cosines else 1.0

    for task in {row["task"] for row in rows_out}:
        for checkpoint in args.checkpoints:
            group = [
                row for row in rows_out
                if row["task"] == task and row["checkpoint"] == checkpoint
            ]
            auc_percentile = rank_percentiles([row["caa_projection_auroc"] for row in group])
            stability_percentile = rank_percentiles([row["local_successive_stability"] for row in group])
            for row, auc_rank, stability_rank in zip(group, auc_percentile, stability_percentile):
                row["caa_auroc_percentile"] = auc_rank
                row["stability_percentile"] = stability_rank

    del selected_features
    return sorted(
        rows_out,
        key=lambda row: (row["task"], args.checkpoints.index(row["checkpoint"]), row["location_name"]),
    )


def build_common_head_orders(
    metric_rows: list[dict[str, Any]],
    sources: dict[str, Any],
    args: argparse.Namespace,
) -> dict[tuple[str, str], list[str]]:
    """Rank CAA-separating heads, restricted to all compared methods' support."""
    output: dict[tuple[str, str], list[str]] = {}
    tasks = [TASK_BY_TRACK[track] for track in args.tracks]
    for task in tasks:
        eligible_methods = tuple(
            method for method in args.methods
            if method != "direct" or task not in REACHABILITY_TASKS
        )
        for checkpoint in args.checkpoints:
            group = [
                row for row in metric_rows
                if row["task"] == task and row["checkpoint"] == checkpoint
            ]
            group = [
                row for row in group
                if all(
                    vector_key(task, checkpoint, row["location_name"]) in sources["banks"][method]
                    and not sources["banks"][method][
                        vector_key(task, checkpoint, row["location_name"])
                    ].zero_norm
                    for method in eligible_methods
                )
            ]
            if len(group) < max(args.head_counts):
                raise RuntimeError(
                    f"Only {len(group)} nonzero attention heads are common to "
                    f"{eligible_methods} at {task}/{checkpoint}; need {max(args.head_counts)}."
                )
            separation = sorted(
                group,
                key=lambda row: (
                    -row["caa_projection_auroc"],
                    -row["caa_standardized_mean_gap"],
                    row["location_name"],
                ),
            )
            output[(task, checkpoint)] = [row["location_name"] for row in separation]
    return output


# =============================================================================
# CONDITIONS AND VALIDATION SELECTION
# =============================================================================

def condition_id(
    method: str,
    task: str,
    checkpoint: str,
    head_count: int,
    beta: float,
) -> str:
    return (
        f"head-method-comparison:{method}:{task}:{checkpoint}:"
        f"k={head_count}:beta={repr(float(beta))}"
    )


def eligible_method_tasks(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    output = []
    for method in args.methods:
        for track in args.tracks:
            task = TASK_BY_TRACK[track]
            if method == "direct" and track != "final_answer":
                continue
            output.append((method, track, task))
    if not output:
        raise RuntimeError("No method/target combinations are eligible.")
    return output


def build_conditions(
    sources: dict[str, Any],
    head_orders: dict[tuple[str, str], list[str]],
    args: argparse.Namespace,
) -> list[Any]:
    caa_bank = sources["banks"]["caa"]
    output = []
    for method, track, task in eligible_method_tasks(args):
        for checkpoint in args.checkpoints:
            order = head_orders[(task, checkpoint)]
            for head_count in args.head_counts:
                names = order[:head_count]
                member_keys = tuple(
                    vector_key(task, checkpoint, name) for name in names
                )
                divisor = math.sqrt(head_count)
                # All methods receive equal energy at matched locations.  The
                # CAA raw norm defines the shared per-head unit of intervention.
                member_scales = tuple(
                    float(caa_bank[key].raw_norm) / divisor for key in member_keys
                )
                if any(key not in sources["banks"][method] for key in member_keys):
                    raise RuntimeError(
                        f"{method} is missing a vector required by the common head set: {member_keys}"
                    )
                for magnitude in args.betas:
                    for beta in (magnitude, -magnitude):
                        output.append(SteeringCondition(
                            condition_id=condition_id(method, task, checkpoint, head_count, beta),
                            method=method,
                            vector_key=f"{task}:{checkpoint}:common-caa-top{head_count}",
                            task=task,
                            label_family=track,
                            checkpoint=checkpoint,
                            representation_type="attention_head",
                            location_name="+".join(names),
                            layer=-1,
                            head=-1,
                            beta=float(beta),
                            scale=1.0,
                            alpha=float(beta),
                            native_direct=False,
                            shared_optimisation_id=None,
                            member_keys=member_keys,
                            member_scales=member_scales,
                            selection_variant=f"common_caa_separation_top{head_count}",
                            head_count=head_count,
                            head_selection="common_caa_separation",
                            head_scaling="matched_caa_energy",
                        ))
    ids = [condition.condition_id for condition in output]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Attention-head comparison generated duplicate condition IDs.")
    return output


def audit_condition_lattice(
    conditions: list[SteeringCondition], args: argparse.Namespace,
) -> dict[str, int]:
    combinations = len(eligible_method_tasks(args))
    expected_total = (
        combinations * len(args.checkpoints) * len(args.head_counts) * len(args.betas) * 2
    )
    if len(conditions) != expected_total:
        raise RuntimeError(
            f"Condition lattice has {len(conditions)} rows, expected {expected_total}."
        )
    grouped: dict[tuple[str, str, int, float], list[SteeringCondition]] = defaultdict(list)
    for condition in conditions:
        grouped[(
            condition.task,
            condition.checkpoint,
            condition.head_count,
            condition.beta,
        )].append(condition)
    for key, group in grouped.items():
        reference_members = group[0].member_keys
        reference_scales = group[0].member_scales
        if any(item.member_keys != reference_members for item in group):
            raise RuntimeError(f"Methods do not share head locations in matched cell {key}.")
        if any(item.member_scales != reference_scales for item in group):
            raise RuntimeError(f"Methods do not share intervention energy in matched cell {key}.")
    positive_coarse = combinations * len(args.checkpoints) * len(args.betas)
    refined = combinations * len(args.checkpoints)
    positive_test = combinations * (len(args.checkpoints) + len(args.head_counts) - 1)
    task_count = len({task for _method, _track, task in eligible_method_tasks(args)})
    baseline_continuations = task_count * len(args.checkpoints) * (
        args.coarse_validation_units + args.refined_validation_units + args.test_units
    )
    steered_continuations = (
        positive_coarse * args.coarse_validation_units
        + refined * args.refined_validation_units
        + 2 * positive_test * args.test_units
    )
    return {
        "all_signed_conditions": expected_total,
        "coarse_validation_conditions": positive_coarse,
        "refined_validation_conditions": refined,
        "positive_test_conditions": positive_test,
        "negative_test_conditions": positive_test,
        "baseline_continuations": baseline_continuations,
        "steered_continuations": steered_continuations,
        "total_generated_continuations": baseline_continuations + steered_continuations,
    }


def validation_rank(row: dict[str, Any], args: argparse.Namespace) -> tuple[Any, ...]:
    validity = float(row["delta_valid_trace_rate"])
    eligible = validity >= -args.maximum_validation_validity_drop
    return (
        eligible,
        float(primary_effect(row)),
        float(row["delta_expr_target_correct_rate"]),
        validity,
        -abs(float(row["beta"])),
        str(row["condition_id"]),
    )


def eligible_validation_rows(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    context: str,
) -> list[dict[str, Any]]:
    eligible = [
        row for row in rows
        if float(row["delta_valid_trace_rate"]) >= -args.maximum_validation_validity_drop
    ]
    if not eligible:
        raise RuntimeError(
            f"No validity-eligible candidate for {context}; either inspect the failures or "
            "explicitly relax --maximum-validation-validity-drop."
        )
    return eligible


def selection_key(method: str, task: str) -> str:
    return f"{method}|{task}"


def select_global_betas_and_refined_ids(
    coarse_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[str]]:
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in coarse_rows:
        if int(row["head_count"]) != args.reference_head_count or float(row["beta"]) <= 0:
            continue
        groups[(row["method"], row["task"], float(row["beta"]))].append(row)
    selected_betas: dict[str, float] = {}
    refined_ids: list[str] = []
    by_method_task: dict[tuple[str, str], list[tuple[float, list[dict[str, Any]]]]] = defaultdict(list)
    for (method, task, beta), rows in groups.items():
        if len(rows) != len(args.checkpoints):
            raise RuntimeError(
                f"Coarse dose cell {method}/{task}/beta={beta:g} covers {len(rows)} "
                f"checkpoints, expected {len(args.checkpoints)}."
            )
        by_method_task[(method, task)].append((beta, rows))
    for (method, task), candidates in sorted(by_method_task.items()):
        def aggregate_rank(item: tuple[float, list[dict[str, Any]]]) -> tuple[Any, ...]:
            beta, rows = item
            validity = [float(row["delta_valid_trace_rate"]) for row in rows]
            effects = [float(primary_effect(row)) for row in rows]
            all_eligible = min(validity) >= -args.maximum_validation_validity_drop
            return (
                all_eligible,
                float(np.median(effects)),
                float(np.mean(effects)),
                min(validity),
                -abs(beta),
            )
        winner_beta, winner_rows = max(candidates, key=aggregate_rank)
        if min(float(row["delta_valid_trace_rate"]) for row in winner_rows) < -args.maximum_validation_validity_drop:
            print(
                f"WARNING: no beta for {method}/{task} preserves validity at every checkpoint; "
                f"selected the robust best fallback beta={winner_beta:g}.",
                flush=True,
            )
        selected_betas[selection_key(method, task)] = float(winner_beta)
        refined_ids.extend(row["condition_id"] for row in winner_rows)
    expected = len(eligible_method_tasks(args))
    if len(selected_betas) != expected:
        raise RuntimeError(f"Selected {len(selected_betas)} method/task betas, expected {expected}.")
    return selected_betas, sorted(set(refined_ids))


def select_reference_checkpoints(
    refined_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, str]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in refined_rows:
        groups[(row["method"], row["task"])].append(row)
    output: dict[str, str] = {}
    for (method, task), rows in sorted(groups.items()):
        candidates = eligible_validation_rows(rows, args, f"refined checkpoint selection {method}/{task}")
        winner = max(candidates, key=lambda row: validation_rank(row, args))
        output[selection_key(method, task)] = str(winner["checkpoint"])
    return output


def select_test_ids(
    conditions: list[Any],
    selected_betas: dict[str, float],
    selected_checkpoints: dict[str, str],
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, list[str]]]:
    lookup = {condition.condition_id: condition for condition in conditions}
    roles: dict[str, list[str]] = defaultdict(list)
    for method, _track, task in eligible_method_tasks(args):
        key = selection_key(method, task)
        beta = float(selected_betas[key])
        reference_checkpoint = selected_checkpoints[key]
        for checkpoint in args.checkpoints:
            cid = condition_id(
                method, task, checkpoint, args.reference_head_count, beta,
            )
            if cid not in lookup:
                raise RuntimeError(f"Missing position-sweep condition: {cid}")
            roles[cid].append("position")
        for head_count in args.head_counts:
            cid = condition_id(method, task, reference_checkpoint, head_count, beta)
            if cid not in lookup:
                raise RuntimeError(f"Missing breadth-sweep condition: {cid}")
            roles[cid].append("breadth")
    return sorted(roles), dict(roles)


def add_negative_controls(
    positive_ids: list[str],
    condition_lookup: dict[str, Any],
) -> tuple[list[str], dict[str, str]]:
    identity_lookup = {
        (
            condition.method,
            condition.task,
            condition.checkpoint,
            condition.head_count,
            float(condition.beta),
        ): condition.condition_id
        for condition in condition_lookup.values()
    }
    negatives: dict[str, str] = {}
    output = list(positive_ids)
    for positive_id in positive_ids:
        positive = condition_lookup[positive_id]
        key = (
            positive.method,
            positive.task,
            positive.checkpoint,
            positive.head_count,
            -float(positive.beta),
        )
        negative_id = identity_lookup.get(key)
        if negative_id is None:
            raise RuntimeError(f"No matched negative control exists for {positive_id}.")
        negatives[negative_id] = positive_id
        output.append(negative_id)
    return list(dict.fromkeys(output)), negatives


# =============================================================================
# EVALUATION AND PERSISTENCE
# =============================================================================

def make_signature(
    sources: dict[str, Any],
    conditions: list[Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    source_paths = dict(sources["paths"])
    source_paths.update({
        "probe_dataset": probe.DATASET_PATH,
        "evaluated_generations": str(sources["data"]["source_evaluated_path"]),
        "token_cache": str(sources["data"]["source_token_cache_path"]),
        "checkpoint": str(sources["data"]["source_checkpoint_path"]),
        "previous_caa_result": str(args.previous_caa_result),
    })
    return {
        "format_version": FORMAT_VERSION,
        "experiment": "attention-head-steering-comparison",
        "comparison_run_tag": COMPARISON_RUN_TAG,
        "source_run_tag": SOURCE_RUN_TAG,
        "methods": list(args.methods),
        "tracks": list(args.tracks),
        "tasks": sorted({task for _method, _track, task in eligible_method_tasks(args)}),
        "checkpoints": list(args.checkpoints),
        "head_counts": list(args.head_counts),
        "betas": list(args.betas),
        "head_selection_units": args.head_selection_units,
        "coarse_validation_units": args.coarse_validation_units,
        "refined_validation_units": args.refined_validation_units,
        "test_units": args.test_units,
        "reference_head_count": args.reference_head_count,
        "batch_size": args.batch_size,
        "bootstrap_samples": args.bootstrap_samples,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "maximum_validation_validity_drop": args.maximum_validation_validity_drop,
        "unique_chainsearch_problems": True,
        "head_selection": "common_caa_separation",
        "head_scaling": "matched_caa_energy",
        "fresh_test_exclusion": "previous_caa_heldout_manifest",
        "conditions": [
            {
                "condition_id": condition.condition_id,
                "method": condition.method,
                "task": condition.task,
                "checkpoint": condition.checkpoint,
                "head_selection": condition.head_selection,
                "head_count": condition.head_count,
                "beta": condition.beta,
                "member_keys": list(condition.member_keys),
                "member_scales": list(condition.member_scales),
            }
            for condition in conditions
        ],
        "script": script_signature(),
        "source_files": {
            name: file_signature(str(path)) for name, path in source_paths.items()
        },
    }


def empty_state(signature: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "complete": False,
        "signature": signature,
        "head_metric_rows": [],
        "coarse_rows": [],
        "selected_betas": {},
        "refined_condition_ids": [],
        "refined_rows": [],
        "selected_checkpoints": {},
        "positive_test_ids": [],
        "test_roles": {},
        "negative_control_of": {},
        "test_rows": [],
        "baseline_rows": [],
        "problem_ids": {},
        "excluded_previous_test_problem_ids": {},
        "completed_ids": set(),
    }


def load_or_initialize_state(
    paths: dict[str, Path],
    signature: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if paths["result"].exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {paths['result']}. Use --overwrite to replace it.")
    if args.overwrite:
        for key in ("result", "partial"):
            if paths[key].exists():
                paths[key].unlink()
        if paths["shards"].is_dir():
            shutil.rmtree(paths["shards"])
    if args.no_resume:
        if paths["partial"].exists():
            paths["partial"].unlink()
        if paths["shards"].is_dir():
            shutil.rmtree(paths["shards"])
    if paths["partial"].is_file():
        state = torch.load(paths["partial"], map_location="cpu", weights_only=False)
        if state.get("signature") != signature:
            raise RuntimeError("Partial attention-head comparison has a different configuration.")
        state["completed_ids"] = set(state.get("completed_ids", set()))
        state.setdefault("problem_ids", {})
        return state
    if paths["shards"].is_dir() and any(paths["shards"].iterdir()):
        raise RuntimeError(
            "Generation shards exist without a matching partial state. "
            "Use --overwrite to start a clean run."
        )
    return empty_state(signature)


def save_state(state: dict[str, Any], path: Path) -> None:
    atomic_save(state, str(path))


def check_reproduction(
    baseline_rows: list[dict[str, Any]],
    phase: str,
    args: argparse.Namespace,
) -> None:
    rows = [row for row in baseline_rows if row["phase"] == phase]
    if not rows:
        raise RuntimeError(f"No {phase} baseline rows were produced.")
    minimum = min(float(row["original_next_token_reproduction_rate"]) for row in rows)
    if minimum < args.hard_minimum_reproduction_rate:
        raise RuntimeError(
            f"Minimum {phase} next-token reproduction {minimum:.4f} is below "
            f"the hard floor {args.hard_minimum_reproduction_rate:.4f}."
        )
    if minimum < args.minimum_reproduction_rate:
        print(
            f"WARNING: minimum {phase} next-token reproduction {minimum:.4f} is below "
            f"the preferred {args.minimum_reproduction_rate:.4f}."
        )


def run_evaluations(
    sources: dict[str, Any],
    conditions: list[Any],
    state: dict[str, Any],
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    condition_lookup = {condition.condition_id: condition for condition in conditions}
    vector_lookup = {
        (method, key): vector
        for method, bank in sources["banks"].items()
        for key, vector in bank.items()
        if vector.representation_type == "attention_head"
    }
    offsets = {
        "coarse": args.head_selection_units,
        "refined": args.head_selection_units + args.coarse_validation_units,
        "test": 0,
    }
    caps = {
        "coarse": args.coarse_validation_units,
        "refined": args.refined_validation_units,
        "test": args.test_units,
    }

    def evaluate_ids(ids: list[str], phase: str, target: str) -> None:
        ordered = sorted(
            ids,
            key=lambda cid: (
                args.methods.index(condition_lookup[cid].method),
                condition_lookup[cid].task,
                args.checkpoints.index(condition_lookup[cid].checkpoint),
                condition_lookup[cid].head_count,
                condition_lookup[cid].beta,
            ),
        )
        pending = [cid for cid in ordered if f"{phase}:{cid}" not in state["completed_ids"]]
        cells: dict[tuple[str, str], list[str]] = defaultdict(list)
        for cid in pending:
            condition = condition_lookup[cid]
            cells[(condition.task, condition.checkpoint)].append(cid)

        progress = tqdm(total=len(pending), desc=f"{phase.title()} conditions", dynamic_ncols=True)
        for (task, checkpoint), cell_ids in cells.items():
            split = "test" if phase == "test" else "val"
            units = build_problem_prefix_units(
                sources,
                task,
                checkpoint,
                split,
                caps[phase],
                offsets[phase],
                excluded_problem_ids=(
                    sources["excluded_test_problem_ids"].get(task, set())
                    if phase == "test" else None
                ),
            )
            problem_ids = [int(unit.cache_idx) for unit in units]
            problem_key = f"{phase}:{task}"
            previous = state["problem_ids"].get(problem_key)
            if previous is not None and list(previous) != problem_ids:
                raise RuntimeError(
                    f"Problem selection changed across checkpoints for {problem_key}."
                )
            state["problem_ids"][problem_key] = problem_ids

            prefix_batches = build_prefix_batches(
                sources["model"],
                units,
                int(sources["evaluated"]["max_output_tokens"]),
                args.batch_size,
                args.device,
                args.use_amp,
                args.amp_dtype,
                args.max_new_tokens,
            )
            shard, summary, baseline_path = ensure_baseline(
                sources, units, prefix_batches, phase, task, checkpoint, args,
            )
            baseline = baseline_result(shard)
            key = (task, checkpoint, phase)
            state["baseline_rows"] = [
                row for row in state["baseline_rows"]
                if (row["task"], row["checkpoint"], row["phase"]) != key
            ]
            state["baseline_rows"].append({
                "task": task,
                "checkpoint": checkpoint,
                "phase": phase,
                "shard_path": baseline_path,
                **summary,
            })
            check_reproduction([state["baseline_rows"][-1]], phase, args)

            execution_groups: dict[tuple[Any, ...], list[str]] = defaultdict(list)
            for condition_id_value in cell_ids:
                condition = condition_lookup[condition_id_value]
                members = tuple(sorted(zip(condition.member_keys, condition.member_scales)))
                execution_groups[(
                    condition.method,
                    condition.task,
                    condition.checkpoint,
                    condition.beta,
                    members,
                )].append(
                    condition_id_value
                )

            for aliases in execution_groups.values():
                execution_condition_id = aliases[0]
                condition = condition_lookup[execution_condition_id]
                generated_row, _ = evaluate_condition(
                    sources,
                    units,
                    prefix_batches,
                    baseline,
                    vector_lookup,
                    condition,
                    phase,
                    args,
                )
                for condition_id_value in aliases:
                    alias_condition = condition_lookup[condition_id_value]
                    row = {
                        **generated_row,
                        **alias_condition.__dict__,
                        "execution_condition_id": execution_condition_id,
                        "duplicate_execution": condition_id_value != execution_condition_id,
                        "unique_problems": len(set(problem_ids)),
                        "baseline_shard_path": baseline_path,
                    }
                    if phase == "test":
                        row["negative_control_of"] = state["negative_control_of"].get(condition_id_value)
                        row["test_roles"] = list(state["test_roles"].get(
                            state["negative_control_of"].get(condition_id_value, condition_id_value),
                            [],
                        ))
                    state[target] = [
                        old for old in state[target]
                        if old.get("condition_id") != condition_id_value
                    ]
                    state[target].append(row)
                    state["completed_ids"].add(f"{phase}:{condition_id_value}")
                    progress.update(1)
                    if len(state["completed_ids"]) % args.save_every == 0:
                        save_state(state, paths["partial"])

            del baseline, shard, prefix_batches, units
            if args.device.type == "cuda":
                torch.cuda.empty_cache()
        save_state(state, paths["partial"])
        progress.close()
        check_reproduction(state["baseline_rows"], phase, args)

    positive_conditions = [condition for condition in conditions if condition.beta > 0]
    coarse_ids = [
        condition.condition_id for condition in positive_conditions
        if condition.head_count == args.reference_head_count
    ]
    evaluate_ids(coarse_ids, "coarse", "coarse_rows")

    if not state["refined_condition_ids"]:
        selected_betas, refined_ids = select_global_betas_and_refined_ids(
            state["coarse_rows"], args,
        )
        state["selected_betas"] = selected_betas
        state["refined_condition_ids"] = refined_ids
        save_state(state, paths["partial"])
    evaluate_ids(state["refined_condition_ids"], "refined", "refined_rows")

    if not state["positive_test_ids"]:
        if not state.get("selected_betas"):
            selected_betas, _ = select_global_betas_and_refined_ids(
                state["coarse_rows"], args,
            )
            state["selected_betas"] = selected_betas
        state["selected_checkpoints"] = select_reference_checkpoints(
            state["refined_rows"], args,
        )
        positive_ids, roles = select_test_ids(
            conditions,
            state["selected_betas"],
            state["selected_checkpoints"],
            args,
        )
        test_ids, negative_control_of = add_negative_controls(positive_ids, condition_lookup)
        state["positive_test_ids"] = positive_ids
        state["test_roles"] = roles
        state["negative_control_of"] = negative_control_of
        state["test_condition_ids"] = test_ids
        save_state(state, paths["partial"])
    test_ids = list(state["test_condition_ids"])
    evaluate_ids(test_ids, "test", "test_rows")
    return state


# =============================================================================
# TABLES, FIGURES, AND SUMMARY
# =============================================================================

def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, sort_keys=True)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def primary_outcome_field(row: dict[str, Any]) -> str:
    return "next_state_viable" if row["label_family"] == "reachability" else "expr_target_correct"


def paired_counts(
    row: dict[str, Any], field: str | None = None,
) -> tuple[int, int, int]:
    field = field or primary_outcome_field(row)
    gain = int(row[f"{field}_gain_count"])
    loss = int(row[f"{field}_loss_count"])
    return gain, loss, gain - loss


def save_figure(fig: plt.Figure, stem: str, paths: dict[str, Path], args: argparse.Namespace) -> None:
    paths["figures"].mkdir(parents=True, exist_ok=True)
    for extension in args.formats:
        fig.savefig(paths["figures"] / f"{stem}.{extension}", dpi=args.dpi, bbox_inches="tight",
                    facecolor=PAPER_CANVAS_BG, transparent=False,)
    plt.close(fig)


def style_paper_figure(
    fig: plt.Figure,
    axes: list[plt.Axes] | tuple[plt.Axes, ...] | None = None,
    *,
    grid: bool = True,
) -> None:
    """Apply the blue paper aesthetic to the supplied data axes."""
    fig.patch.set_facecolor(PAPER_CANVAS_BG)
    target_axes = list(fig.axes if axes is None else axes)
    legends = list(fig.legends)
    for axis in target_axes:
        axis.set_facecolor(PAPER_PANEL_BG)
        axis.title.set_color(PAPER_DEEP_BLUE)
        axis.xaxis.label.set_color(PAPER_DEEP_BLUE)
        axis.yaxis.label.set_color(PAPER_DEEP_BLUE)
        axis.tick_params(axis="both", colors=PAPER_DEEP_BLUE)
        axis.spines["left"].set_color(PAPER_DEEP_BLUE)
        axis.spines["bottom"].set_color(PAPER_DEEP_BLUE)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if grid:
            axis.grid(True, axis="y", color=PAPER_GRID, alpha=0.55)
        else:
            axis.grid(False)
        legend = axis.get_legend()
        if legend is not None:
            legends.append(legend)
    for legend in legends:
        legend.get_frame().set_facecolor(PAPER_FIGURE_BG)
        legend.get_frame().set_edgecolor(PAPER_DEEP_BLUE)
        legend.get_frame().set_alpha(0.90)
        for text_item in legend.get_texts():
            text_item.set_color(PAPER_DARK_TEXT)


def format_selected_settings(
    state: dict[str, Any],
    methods: list[str],
    task: str,
    *,
    include_checkpoint: bool,
) -> str:
    """Format validation-selected settings as a compact centred annotation."""
    items = []
    for method in methods:
        key = selection_key(method, task)
        item = f"{METHOD_DISPLAY[method]} β={state['selected_betas'][key]:g}"
        if include_checkpoint:
            item += f" at {CHECKPOINT_DISPLAY[state['selected_checkpoints'][key]]}"
        items.append(item)
    wrap_after = 2 if include_checkpoint else 3
    first_line = "   •   ".join(items[:wrap_after])
    text = f"Validation-selected   •   {first_line}"
    if len(items) > wrap_after:
        text += "\n" + "   •   ".join(items[wrap_after:])
    return text


def positive_test_rows(state: dict[str, Any], role: str) -> list[dict[str, Any]]:
    positive = set(state["positive_test_ids"])
    return [
        row for row in state["test_rows"]
        if row["condition_id"] in positive and role in row.get("test_roles", [])
    ]


def signed_count_limit(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> int:
    maximum = 1
    for row in rows:
        for field in fields:
            gain, loss, _net = paired_counts(row, field)
            maximum = max(maximum, gain, loss)
    return maximum


def draw_paired_count_bars(
    axis: plt.Axes,
    categories: list[Any],
    methods: list[str],
    lookup: dict[tuple[Any, str], dict[str, Any]],
    field: str,
    *,
    annotation_fontsize: float = 9.2,
) -> None:
    width = 0.78 / max(1, len(methods))
    centre = (len(methods) - 1) / 2.0
    all_rows = list(lookup.values())
    limit = signed_count_limit(all_rows, (field,))
    label_pad = max(0.12, 0.045 * limit)
    zero_label_available = True
    for method_index, method in enumerate(methods):
        offset = (method_index - centre) * width
        for category_index, category in enumerate(categories):
            row = lookup.get((category, method))
            if row is None:
                continue
            gain, loss, _net = paired_counts(row, field)
            x = category_index + offset
            axis.bar(
                x, gain, width=width * 0.88, color=METHOD_COLORS[method],
                label=METHOD_DISPLAY[method] if category_index == 0 else None,
            )
            axis.bar(
                x, -loss, width=width * 0.88, color=METHOD_COLORS[method],
                alpha=0.34, hatch="//", edgecolor=METHOD_COLORS[method], linewidth=0.7,
            )
            if gain:
                axis.text(x, gain + label_pad, f"{gain}", ha="center", va="bottom", fontsize=annotation_fontsize)
            if loss:
                axis.text(x, -loss - label_pad, f"−{loss}", ha="center", va="top", fontsize=annotation_fontsize)
            if not gain and not loss:
                axis.plot(
                    x,
                    0,
                    marker="D",
                    markersize=5.0,
                    markerfacecolor=PAPER_PANEL_BG,
                    markeredgecolor=PAPER_DEEP_BLUE,
                    markeredgewidth=1.0,
                    linestyle="none",
                    label="No changes" if zero_label_available else None,
                    zorder=3,
                )
                zero_label_available = False
    axis.axhline(0.0, color=PAPER_REFLINE, linestyle="--", linewidth=1.0, zorder=1,)
    axis.set_xlim(-0.55, len(categories) - 0.45)
    axis.set_ylim(-1.35 * limit, 1.35 * limit)
    axis.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda value, _position: (f"−{abs(value):g}" if value < 0 else f"{value:g}"))
    ) 
    axis.grid(True, axis="y", alpha=0.28)


def plot_validation_dose_counts(
    state: dict[str, Any], paths: dict[str, Path], args: argparse.Namespace,
) -> None:
    panels = eligible_method_tasks(args)
    panel_lookup = {
        (method, track): (method, track, task)
        for method, track, task in panels
    }
    fig = plt.figure(figsize=(13.2, 7.8))
    grid = fig.add_gridspec(4, 3, hspace=1.55, wspace=0.12)
    slots = (
        (("probe", "reachability"), grid[0:2, 0]),
        (("caa", "reachability"), grid[0:2, 1]),
        (("probe", "final_answer"), grid[2:4, 0]),
        (("caa", "final_answer"), grid[2:4, 1]),
        (("direct", "final_answer"), grid[1:3, 2]),
    )
    panel_axes: list[tuple[plt.Axes, tuple[str, str, str]]] = []
    for key, slot in slots:
        if key in panel_lookup:
            panel_axes.append((fig.add_subplot(slot), panel_lookup[key]))

    all_nets = [paired_counts(row)[2] for row in state["coarse_rows"]]
    bound = max(1, max((abs(value) for value in all_nets), default=1))
    colour_norm = matplotlib.colors.TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    colour_map = matplotlib.colors.LinearSegmentedColormap.from_list(
        "loss_neutral_gain", (PAPER_LOSS_FILL, PAPER_PANEL_BG, PAPER_GAIN_FILL),
    )

    for axis, (method, track, task) in panel_axes:
        lookup = {
            (row["checkpoint"], float(row["beta"])): row
            for row in state["coarse_rows"]
            if row["method"] == method and row["task"] == task
        }
        matrix = np.asarray([
            [paired_counts(lookup[(checkpoint, beta)])[2] for beta in args.betas]
            for checkpoint in args.checkpoints
        ], dtype=float)
        axis.imshow(matrix, aspect="auto", cmap=colour_map, norm=colour_norm)
        for y, checkpoint in enumerate(args.checkpoints):
            for x, beta in enumerate(args.betas):
                gain, loss, net = paired_counts(lookup[(checkpoint, beta)])
                if not gain and not loss:
                    axis.plot(
                        x,
                        y,
                        marker="D",
                        markersize=4.6,
                        markerfacecolor=PAPER_PANEL_BG,
                        markeredgecolor=PAPER_DEEP_BLUE,
                        markeredgewidth=0.9,
                        linestyle="none",
                    )
                else: 
                    axis.text(
                        x,
                        y,
                        f"+{gain}\n−{loss}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color=PAPER_DARK_TEXT,
                    )

        selected_beta = float(state["selected_betas"][selection_key(method, task)])
        selected_column = list(args.betas).index(selected_beta)
        axis.add_patch(plt.Rectangle(
            (selected_column - 0.5, -0.5), 1.0, len(args.checkpoints),
            fill=False, edgecolor=PAPER_DEEP_BLUE, linewidth=1.0,
        ))
        axis.set_xticks(range(len(args.betas)), [f"{beta:g}" for beta in args.betas])
        axis.set_yticks(
            range(len(args.checkpoints)),
            [CHECKPOINT_DISPLAY[checkpoint] for checkpoint in args.checkpoints],
        )
        axis.tick_params(axis="both", labelsize=12)
        if method != "probe":
            axis.set_yticklabels([])
        axis.set_xlabel("Intervention strength β", fontsize=12.5) 
        panel_title = (
            f"{METHOD_PANEL_DISPLAY[method]}\n"
            f"for {OUTCOME_PANEL_DISPLAY[track]}"
            if track == "final_answer"
            else
            f"{METHOD_PANEL_DISPLAY[method]} "
            f"for {OUTCOME_PANEL_DISPLAY[track]}"
        )
        axis.set_title(
            panel_title,
            fontsize=13.5,
            pad=8,
            linespacing=1.05,
        )

    data_axes = [axis for axis, _panel in panel_axes]
    style_paper_figure(fig, data_axes, grid=False)

    fig.suptitle(
        f"Effect of intervention strength at the {args.reference_head_count} heads with the strongest CAA class separation", # prev f"Effect of intervention strength at {args.reference_head_count} CAA-selected heads",
        y=0.985,
        fontsize=16,
        color=PAPER_TITLE,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.975,
        bottom=0.090,
        top=0.900,
    )
    save_figure(fig, "validation_dose_response_counts", paths, args)


def plot_heldout_headcount_counts(
    state: dict[str, Any], paths: dict[str, Path], args: argparse.Namespace,
) -> None:
    source_rows = positive_test_rows(state, "breadth")
    available_tracks = [
        track for track in args.tracks
        if any(row["task"] == TASK_BY_TRACK[track] for row in source_rows)
    ]
    fig, axes = plt.subplots(
        1, len(available_tracks), figsize=(6.0 * len(available_tracks), 4.7), squeeze=False,
    )
    for axis, track in zip(axes[0], available_tracks):
        task = TASK_BY_TRACK[track]
        methods = [method for method in args.methods if any(
            row["task"] == task and row["method"] == method for row in source_rows
        )]
        lookup = {
            (int(row["head_count"]), row["method"]): row
            for row in source_rows if row["task"] == task
        }
        draw_paired_count_bars(axis, list(args.head_counts), methods, lookup, primary_outcome_field(next(iter(lookup.values()))))
        axis.set_xticks(range(len(args.head_counts)), [str(value) for value in args.head_counts])
        axis.tick_params(axis="both", labelsize=11, colors=PAPER_DEEP_BLUE)
        panel_title = ("Next-state reachability" if track == "reachability" else "Final-answer correctness")
        axis.set_title(panel_title, fontsize=13.0)
        axis.legend(fontsize=10, loc="best")
    fig.suptitle("Effect of attention-head count on steering", x=0.53, y=0.985, fontsize=14.75, color=PAPER_TITLE, )
    fig.supxlabel("Number of steered attention heads", y=0.045, fontsize=12, color=PAPER_DEEP_BLUE,)
    fig.supylabel("Changed test problems", x=0.025, y=0.52, fontsize=12.5, color=PAPER_DEEP_BLUE,)
    fig.subplots_adjust( left=0.08,  right=0.985,  top=0.85,  bottom=0.18,  wspace=0.12,)
    style_paper_figure(fig)
    save_figure(fig, "heldout_headcount_counts", paths, args)


def plot_heldout_position_counts(
    state: dict[str, Any], paths: dict[str, Path], args: argparse.Namespace,
) -> None:
    source_rows = positive_test_rows(state, "position")
    available_tracks = [
        track for track in args.tracks
        if any(row["task"] == TASK_BY_TRACK[track] for row in source_rows)
    ]
    fig, axes = plt.subplots(
        1, len(available_tracks), figsize=(6.0 * len(available_tracks), 5.2), squeeze=False,
    )
    for axis, track in zip(axes[0], available_tracks):
        task = TASK_BY_TRACK[track]
        methods = [method for method in args.methods if any(
            row["task"] == task and row["method"] == method for row in source_rows
        )]
        lookup = {
            (row["checkpoint"], row["method"]): row
            for row in source_rows if row["task"] == task
        }
        draw_paired_count_bars(axis, list(args.checkpoints), methods, lookup, primary_outcome_field(next(iter(lookup.values()))))
        axis.set_xticks(
            range(len(args.checkpoints)),
            [CHECKPOINT_DISPLAY[value] for value in args.checkpoints],
            rotation=30,
            ha="right",
        )
        axis.tick_params( axis="both", labelsize=11, colors=PAPER_DEEP_BLUE,)
        panel_title = (
            "Steering next-state reachability\nusing reachability directions"
            if track == "reachability"
            else "Steering final-answer correctness\nusing final-answer directions"
        )
        axis.set_title(panel_title, fontsize=13.0)

        handles, labels = axis.get_legend_handles_labels()

        setting_labels = {
            METHOD_DISPLAY[method]: (
                f"{METHOD_DISPLAY[method]} "
                f"({args.reference_head_count} heads, "
                f"β={state['selected_betas'][selection_key(method, task)]:g})"
            )
            for method in methods
        }

        labels = [
            setting_labels.get(label, label)
            for label in labels
        ]

        axis.legend(
            handles,
            labels,
            fontsize=8.5,
            loc="best",
        )
    fig.suptitle(
        "Effects of intervention position under target-matched steering",
        x=0.53,
        y=0.985,
        fontsize=14.75,
        color=PAPER_TITLE,
    )
    fig.supxlabel("Intervention position",  x=0.53, y=-0.010, fontsize=12, color=PAPER_DEEP_BLUE,)
    fig.supylabel("Changed test problems", x=0.03, y=0.555, fontsize=12.5, color=PAPER_DEEP_BLUE,)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.83, bottom=0.25, wspace=0.12,)
    style_paper_figure(fig)
    save_figure(fig, "heldout_position_counts", paths, args)


def plot_reachability_local_global_counts(
    state: dict[str, Any], paths: dict[str, Path], args: argparse.Namespace,
) -> None:
    task = TASK_BY_TRACK["reachability"]
    rows = [
        row for row in positive_test_rows(state, "position")
        if row["task"] == task
    ]
    if not rows:
        return
    methods = [method for method in args.methods if any(row["method"] == method for row in rows)]
    setting_labels = {
        METHOD_DISPLAY[method]: (
            f"{METHOD_DISPLAY[method]} "
            f"({args.reference_head_count} heads, "
            f"β={state['selected_betas'][selection_key(method, task)]:g})"
        )
        for method in methods
    }
    lookup = {(row["checkpoint"], row["method"]): row for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.1), sharex=True)
    for axis, field, title in (
        (
            axes[0],
            "next_state_viable",
            "Steering next-state reachability\nusing reachability directions",
        ),
        (
            axes[1],
            "expr_target_correct",
            "Steering final-answer correctness\nusing reachability directions",
        ),
    ):
        draw_paired_count_bars(axis, list(args.checkpoints), methods, lookup, 
            field, annotation_fontsize=9.2,)
        axis.set_title(title, fontsize=13.0)
        axis.tick_params(axis="both", labelsize=11.0, colors=PAPER_DEEP_BLUE)
        handles, labels = axis.get_legend_handles_labels()

        labels = [
            setting_labels.get(label, label)
            for label in labels
        ]

        axis.legend(
            handles,
            labels,
            fontsize=8.8,
            loc="upper right",
        )
    tick_labels = [CHECKPOINT_DISPLAY[value] for value in args.checkpoints]
    for axis in axes:
        axis.set_xticks(range(len(args.checkpoints)), tick_labels, rotation=30, ha="right")

    fig.supylabel("Changed test problems", x=0.03, y=0.57, fontsize=12.5, color=PAPER_DEEP_BLUE)
    fig.supxlabel("Intervention position", y=0.008, fontsize=12.0, color=PAPER_DEEP_BLUE,)

    fig.suptitle(
        "Local-to-global transfer of reachability steering",
        x=0.53,
        y=0.985,
        fontsize=14.75,   
        color=PAPER_TITLE,
    )  
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.29, top=0.83, wspace=0.12)

    style_paper_figure(fig)
    save_figure(fig, "reachability_local_to_global_counts", paths, args)


def plot_local_global_position_counts(
    state: dict[str, Any], paths: dict[str, Path], args: argparse.Namespace,
) -> None:
    source_rows = positive_test_rows(state, "position")
    if not source_rows:
        return

    fig = plt.figure(figsize=(12.0, 5.2))
    grid = fig.add_gridspec(1, 2)
    panels = (
        (
            fig.add_subplot(grid[0, 0]),
            "final_answer",
            "expr_target_correct",
            "Steering final-answer correctness\nusing final-answer directions",
        ),
        (
            fig.add_subplot(grid[0, 1]),
            "reachability",
            "expr_target_correct",
            "Steering final-answer correctness\nusing reachability directions",
        ),
    )

    final_answer_axes = []
    for axis, track, field, title in panels:
        task = TASK_BY_TRACK[track]
        rows = [row for row in source_rows if row["task"] == task]
        if not rows:
            axis.set_visible(False)
            continue

        methods = [
            method for method in args.methods
            if any(row["method"] == method for row in rows)
        ]
        lookup = {
            (row["checkpoint"], row["method"]): row
            for row in rows
        }
        draw_paired_count_bars(
            axis, list(args.checkpoints), methods, lookup, field,
        )
        axis.set_title(title, fontsize=13.0)
        axis.set_xticks(
            range(len(args.checkpoints)),
            [CHECKPOINT_DISPLAY[value] for value in args.checkpoints],
            rotation=30,
            ha="right",
        )
        axis.tick_params(axis="both", labelsize=11, colors=PAPER_DEEP_BLUE)

        setting_labels = {
            METHOD_DISPLAY[method]: (
                f"{METHOD_DISPLAY[method]} "
                f"({args.reference_head_count} heads, "
                f"β={state['selected_betas'][selection_key(method, task)]:g})"
            )
            for method in methods
        }
        handles, labels = axis.get_legend_handles_labels()
        axis.legend(
            handles,
            [setting_labels.get(label, label) for label in labels],
            fontsize=8.5,
            loc="best",
        )
        if field == "expr_target_correct":
            final_answer_axes.append(axis)

    # Compare both final-answer panels on the same scale.
    if final_answer_axes:
        lower = min(axis.get_ylim()[0] for axis in final_answer_axes)
        upper = max(axis.get_ylim()[1] for axis in final_answer_axes)
        for axis in final_answer_axes:
            axis.set_ylim(lower, upper)

    fig.suptitle(
        "Local and global effects of steering",
        x=0.53, y=0.98, fontsize=15, color=PAPER_TITLE,
    )
    fig.supxlabel(
        "Intervention position",
        x=0.53, y=-0.02, fontsize=12, color=PAPER_DEEP_BLUE,
    )
    fig.subplots_adjust(
        left=0.08, right=0.985, top=0.80, bottom=0.29,
        wspace=0.30, 
    )

    # One shared y-axis label, centered beside the left panel.
    for axis in (panels[0][0],):
        bounds = axis.get_position()
        fig.text(
            bounds.x0 - 0.055,
            (bounds.y0 + bounds.y1) / 2,
            "Changed test problems",
            rotation=90,
            ha="center",
            va="center",
            fontsize=12.5,
            color=PAPER_DEEP_BLUE,
        )

    style_paper_figure(fig)
    save_figure(fig, "heldout_position_local_global_counts", paths, args)



def plot_directionality_counts(
    state: dict[str, Any], paths: dict[str, Path], args: argparse.Namespace,
) -> None:
    row_by_id = {row["condition_id"]: row for row in state["test_rows"]}
    reference_positive = [
        row for row in state["test_rows"]
        if row["condition_id"] in set(state["positive_test_ids"])
        and set(row.get("test_roles", [])) == {"position", "breadth"}
    ]
    available_tracks = [
        track for track in args.tracks
        if any(row["task"] == TASK_BY_TRACK[track] for row in reference_positive)
    ]
    fig, axes = plt.subplots(
        1, len(available_tracks), figsize=(6.0 * len(available_tracks), 4.5), squeeze=False,
    )
    for axis, track in zip(axes[0], available_tracks):
        task = TASK_BY_TRACK[track]
        methods = [method for method in args.methods if any(
            row["task"] == task and row["method"] == method for row in reference_positive
        )]
        lookup: dict[tuple[str, str], dict[str, Any]] = {}
        for positive in reference_positive:
            if positive["task"] != task:
                continue
            lookup[("Selected direction", positive["method"])] = positive
            negative_id = next(
                negative for negative, matched in state["negative_control_of"].items()
                if matched == positive["condition_id"]
            )
            lookup[("Reversed direction", positive["method"])] = row_by_id[negative_id]
        field = primary_outcome_field(next(iter(lookup.values())))
        draw_paired_count_bars(axis, ["Selected direction", "Reversed direction"], methods, lookup, field)
        axis.set_xticks(range(2), ["Selected direction", "Reversed direction"])
        panel_title = (
            "Next-state reachability"
            if track == "reachability"
            else "Final-answer correctness"
        )
        axis.set_title(panel_title, fontsize=13.0)
        axis.tick_params(
            axis="both",
            labelsize=11,
            colors=PAPER_DEEP_BLUE,
        )
        axis.legend(fontsize=10, loc="best") 
    fig.suptitle(
        "Effects of direction reversal across steering methods",
        x=0.53,
        y=0.985,
        fontsize=14.75,
        color=PAPER_TITLE,
    )
    fig.supylabel(
        "Changed test problems",
        x=0.025, y=0.50,
        fontsize=12.5,
        color=PAPER_DEEP_BLUE,
    )
    fig.subplots_adjust(
        left=0.08, right=0.985,
        top=0.85, bottom=0.16,
        wspace=0.12,
    )
    style_paper_figure(fig)
    save_figure(fig, "heldout_directionality_counts", paths, args)


def write_summary(
    state: dict[str, Any],
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> None:
    lines = [
        "ATTENTION-HEAD STEERING METHOD COMPARISON",
        "",
        "The same CAA-ranked attention-head locations are used for every method.",
        "Each method uses a unit direction with matched CAA-defined intervention energy.",
        "Head ranking, coarse beta selection, and refined checkpoint selection use disjoint validation problems.",
        "Held-out problems are fresh: none appeared in selection or in the earlier inspected CAA test set.",
        "All human-readable effects below are exact paired ChainSearch-problem counts.",
        "",
        f"Methods: {', '.join(METHOD_DISPLAY[method] for method in args.methods)}",
        f"Head-selection problems per task: {args.head_selection_units}",
        f"Coarse-validation problems per condition: {args.coarse_validation_units}",
        f"Refined-validation problems per condition: {args.refined_validation_units}",
        f"Held-out test problems per condition: {args.test_units}",
        f"Positive held-out configurations: {len(state['positive_test_ids'])}",
        f"Matched negative controls: {len(state['negative_control_of'])}",
        "",
        "VALIDATION-SELECTED SETTINGS",
    ]
    for method, track, task in eligible_method_tasks(args):
        key = selection_key(method, task)
        lines.append(
            f"{METHOD_DISPLAY[method]} | {track.replace('_', ' ')} | "
            f"beta={state['selected_betas'][key]:g} | "
            f"reference checkpoint={CHECKPOINT_DISPLAY[state['selected_checkpoints'][key]]} | "
            f"reference heads={args.reference_head_count}"
        )
    lines.extend([
        "",
        "HELD-OUT POSITIVE RESULTS",
        "Format: gains are false→true; losses are true→false; net = gains − losses.",
    ])
    rows = [
        row for row in state["test_rows"]
        if row["condition_id"] in set(state["positive_test_ids"])
    ]
    for row in sorted(rows, key=lambda current: current["condition_id"]):
        field = primary_outcome_field(row)
        gain, loss, net = paired_counts(row, field)
        final_gain, final_loss, final_net = paired_counts(row, "expr_target_correct")
        lines.append(
            f"{METHOD_DISPLAY[row['method']]} | {row['task']} | {row['checkpoint']} | "
            f"k={row['head_count']} | beta={row['beta']:g} | "
            f"primary gains/losses/net={gain}/{loss}/{net:+d} of {row['units']} | "
            f"final gains/losses/net={final_gain}/{final_loss}/{final_net:+d} of {row['units']} | "
            f"roles={','.join(row.get('test_roles', []))}"
        )
    paths["summary"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_changed_problem_manifest(
    state: dict[str, Any], paths: dict[str, Path], args: argparse.Namespace,
) -> None:
    """Index every held-out paired transition so its exact generations can be inspected."""
    rows_out: list[dict[str, Any]] = []
    positive = set(state["positive_test_ids"])
    for row in sorted(state["test_rows"], key=lambda current: current["condition_id"]):
        if row["condition_id"] not in positive:
            continue
        steered_path = Path(str(row["shard_path"]))
        baseline_path = Path(str(row["baseline_shard_path"]))
        if not steered_path.is_file() or not baseline_path.is_file():
            raise FileNotFoundError(
                f"A paired transition shard is missing: {baseline_path} or {steered_path}"
            )
        steered = load_shard(str(steered_path))
        baseline = load_shard(str(baseline_path))
        if not torch.equal(
            steered["cache_idx"].to(torch.int64), baseline["cache_idx"].to(torch.int64)
        ):
            raise RuntimeError(f"Paired shard problem IDs differ for {row['condition_id']}.")
        fields = ["expr_target_correct"]
        if row["label_family"] == "reachability":
            fields.insert(0, "next_state_viable")
        for field in fields:
            before = baseline["labels"][field].to(torch.bool)
            after = steered["labels"][field].to(torch.bool)
            for index in torch.nonzero(before != after, as_tuple=False).flatten().tolist():
                rows_out.append({
                    "condition_id": row["condition_id"],
                    "method": row["method"],
                    "task": row["task"],
                    "checkpoint": row["checkpoint"],
                    "head_count": int(row["head_count"]),
                    "beta": float(row["beta"]),
                    "test_roles": row.get("test_roles", []),
                    "outcome": field,
                    "transition": "gain" if bool(after[index]) else "loss",
                    "baseline_value": int(before[index]),
                    "steered_value": int(after[index]),
                    "problem_id": int(steered["cache_idx"][index]),
                    "source_row": int(steered["source_row"][index]),
                    "record_pos": int(steered["record_pos"][index]),
                    "baseline_shard_path": str(baseline_path),
                    "steered_shard_path": str(steered_path),
                    "shard_row": int(index),
                })
    if rows_out:
        write_csv(paths["transition_csv"], rows_out)
    else:
        fields = (
            "condition_id,method,task,checkpoint,head_count,beta,test_roles,outcome,"
            "transition,baseline_value,steered_value,problem_id,source_row,record_pos,"
            "baseline_shard_path,steered_shard_path,shard_row\n"
        )
        paths["transition_csv"].write_text(fields, encoding="utf-8")


def final_audit(state: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    positive = set(state["positive_test_ids"])
    test_ids = {row["condition_id"] for row in state["test_rows"]}
    expected = positive | set(state["negative_control_of"])
    if test_ids != expected:
        raise RuntimeError("Held-out test rows do not match selected positives and negative controls.")
    if not set(state["negative_control_of"].values()) <= positive:
        raise RuntimeError("A negative control does not reference a selected positive condition.")
    if any(int(row["units"]) != args.test_units for row in state["test_rows"]):
        raise RuntimeError("A held-out row does not contain the requested number of problems.")
    if any(int(row["unique_problems"]) != args.test_units for row in state["test_rows"]):
        raise RuntimeError("A held-out row contains repeated ChainSearch problems.")
    tasks = sorted({task for _method, _track, task in eligible_method_tasks(args)})
    combinations = len(eligible_method_tasks(args))
    expected_coarse = combinations * len(args.checkpoints) * len(args.betas)
    expected_refined = combinations * len(args.checkpoints)
    expected_positive = combinations * (
        len(args.checkpoints) + len(args.head_counts) - 1
    )
    if len(state["coarse_rows"]) != expected_coarse:
        raise RuntimeError(
            f"Coarse validation has {len(state['coarse_rows'])} rows, expected {expected_coarse}."
        )
    if len(state["refined_rows"]) != expected_refined:
        raise RuntimeError(
            f"Refined validation has {len(state['refined_rows'])} rows, expected {expected_refined}."
        )
    if len(positive) != expected_positive:
        raise RuntimeError(
            f"Held-out positive sweep has {len(positive)} rows, expected {expected_positive}."
        )
    required_problem_keys = {
        f"{phase}:{task}"
        for task in tasks
        for phase in ("head_selection", "coarse", "refined", "test")
    }
    missing = required_problem_keys - set(state.get("problem_ids", {}))
    if missing:
        raise RuntimeError(f"Problem-ID audit is missing roles: {sorted(missing)}")
    selection_ids: set[int] = set()
    test_problem_ids: set[int] = set()
    for task in tasks:
        role_sets = {
            phase: set(int(value) for value in state["problem_ids"][f"{phase}:{task}"])
            for phase in ("head_selection", "coarse", "refined", "test")
        }
        for phase, expected_count in (
            ("head_selection", args.head_selection_units),
            ("coarse", args.coarse_validation_units),
            ("refined", args.refined_validation_units),
            ("test", args.test_units),
        ):
            if len(role_sets[phase]) != expected_count:
                raise RuntimeError(
                    f"{task}/{phase} has {len(role_sets[phase])} unique problems, "
                    f"expected {expected_count}."
                )
        if role_sets["head_selection"] & role_sets["coarse"]:
            raise RuntimeError(f"Head-selection and coarse validation overlap for {task}.")
        if role_sets["head_selection"] & role_sets["refined"]:
            raise RuntimeError(f"Head-selection and refined validation overlap for {task}.")
        if role_sets["coarse"] & role_sets["refined"]:
            raise RuntimeError(f"Coarse and refined validation overlap for {task}.")
        task_selection = role_sets["head_selection"] | role_sets["coarse"] | role_sets["refined"]
        if task_selection & role_sets["test"]:
            raise RuntimeError(f"Validation and test problem IDs overlap for {task}.")
        excluded = set(int(value) for value in state["excluded_previous_test_problem_ids"][task])
        if role_sets["test"] & excluded:
            raise RuntimeError(f"Fresh {task} test set overlaps the earlier inspected CAA test set.")
        selection_ids |= task_selection
        test_problem_ids |= role_sets["test"]
    if selection_ids & test_problem_ids:
        raise RuntimeError("A held-out test problem was used anywhere in selection.")
    return {
        "head_selection_used_test": bool(
            set().union(*(
                set(state["problem_ids"][f"head_selection:{task}"]) for task in tasks
            )) & test_problem_ids
        ),
        "steering_selection_used_test": bool(selection_ids & test_problem_ids),
        "problem_role_disjointness_verified": True,
        "previous_caa_test_exclusion_verified": True,
        "unique_chainsearch_problems_per_condition": args.test_units,
        "positive_test_conditions": len(positive),
        "negative_test_controls": len(state["negative_control_of"]),
        "overall_status": "PASS",
    }


def render_outputs(
    state: dict[str, Any], paths: dict[str, Path], args: argparse.Namespace,
) -> None:
    metric_rows = list(state["head_metric_rows"])
    write_csv(paths["ranking_csv"], metric_rows)
    write_csv(paths["coarse_csv"], state["coarse_rows"])
    write_csv(paths["refined_csv"], state["refined_rows"])
    write_csv(paths["effects_csv"], state["test_rows"])
    write_changed_problem_manifest(state, paths, args)
    plot_validation_dose_counts(state, paths, args)
    plot_heldout_headcount_counts(state, paths, args)
    plot_heldout_position_counts(state, paths, args)
    plot_reachability_local_global_counts(state, paths, args)
    plot_local_global_position_counts(state, paths, args)
    plot_directionality_counts(state, paths, args)
    write_summary(state, paths, args)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    global SEED, SHARD_DIR
    args = parse_args()
    paths = experiment_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_legacy_partial(paths)
    paths["figures"].mkdir(parents=True, exist_ok=True)

    SEED = args.seed
    SHARD_DIR = str(paths["shards"])

    if args.plot_only:
        if not paths["result"].is_file():
            raise FileNotFoundError(f"No complete result exists at {paths['result']}.")
        payload = torch.load(paths["result"], map_location="cpu", weights_only=False)
        if not bool(payload.get("complete", False)):
            raise RuntimeError(f"Result is incomplete: {paths['result']}")
        saved = payload.get("signature", {})
        for name in ("methods", "tracks", "checkpoints", "head_counts", "betas"):
            if name in saved:
                setattr(args, name, tuple(saved[name]))
        for name in (
            "head_selection_units",
            "coarse_validation_units",
            "refined_validation_units",
            "test_units",
            "reference_head_count",
        ):
            if name in saved:
                setattr(args, name, int(saved[name]))
        render_outputs(payload, paths, args)
        probe.heading("ATTENTION-HEAD METHOD-COMPARISON OUTPUTS REGENERATED")
        return

    if paths["result"].exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {paths['result']}. Use --overwrite to replace it.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    started = time.monotonic()
    probe.heading("ATTENTION-HEAD STEERING METHOD COMPARISON")
    probe.print_rows("RUN CONFIGURATION", [
        ("Methods", ", ".join(METHOD_DISPLAY[method] for method in args.methods)),
        ("Tracks", ", ".join(args.tracks)),
        ("Checkpoints", ", ".join(args.checkpoints)),
        ("Head selection", "common CAA projection separation"),
        ("Head counts", list(args.head_counts)),
        ("Reference head count", args.reference_head_count),
        ("Positive betas", list(args.betas)),
        (
            "Head / coarse / refined / test problems",
            f"{args.head_selection_units} / {args.coarse_validation_units} / "
            f"{args.refined_validation_units} / {args.test_units}",
        ),
        ("Output directory", args.output_dir),
        ("Device", args.device),
    ])

    sources = load_sources(args.device, args.methods)
    tasks = tuple(sorted({task for _method, _track, task in eligible_method_tasks(args)}))
    sources["excluded_test_problem_ids"] = load_excluded_test_problem_ids(
        args.previous_caa_result, tasks,
    )
    resume_preview = None
    if paths["partial"].is_file() and not args.overwrite and not args.no_resume:
        resume_preview = torch.load(paths["partial"], map_location="cpu", weights_only=False)
    if (
        isinstance(resume_preview, dict)
        and resume_preview.get("format_version") == FORMAT_VERSION
        and resume_preview.get("head_metric_rows")
        and resume_preview.get("problem_ids")
    ):
        head_metric_rows = list(resume_preview["head_metric_rows"])
        sources["head_selection_problem_ids"] = {
            task: list(resume_preview["problem_ids"][f"head_selection:{task}"])
            for task in (TASK_BY_TRACK[track] for track in args.tracks)
        }
        print("Reusing saved validation-only attention-head rankings.", flush=True)
    else:
        head_metric_rows = compute_caa_head_metrics(sources, args)
    head_orders = build_common_head_orders(head_metric_rows, sources, args)
    conditions = build_conditions(sources, head_orders, args)
    protocol = audit_condition_lattice(conditions, args)
    probe.print_rows("PRE-RUN PROTOCOL AUDIT", [
        ("All signed condition definitions", protocol["all_signed_conditions"]),
        ("Coarse validation conditions", protocol["coarse_validation_conditions"]),
        ("Refined validation conditions", protocol["refined_validation_conditions"]),
        ("Positive / reversed held-out conditions", (
            f"{protocol['positive_test_conditions']} / {protocol['negative_test_conditions']}"
        )),
        ("Expected generated continuations", protocol["total_generated_continuations"]),
        ("Matched locations and energy", "PASS"),
    ])
    signature = make_signature(sources, conditions, args)
    state = load_or_initialize_state(paths, signature, args)
    paths["shards"].mkdir(parents=True, exist_ok=True)
    state["head_metric_rows"] = head_metric_rows
    state["excluded_previous_test_problem_ids"] = {
        task: sorted(problem_ids)
        for task, problem_ids in sources["excluded_test_problem_ids"].items()
    }
    for task, problem_ids in sources["head_selection_problem_ids"].items():
        state["problem_ids"][f"head_selection:{task}"] = list(problem_ids)
    save_state(state, paths["partial"])

    state = run_evaluations(sources, conditions, state, paths, args)
    audit = final_audit(state, args)
    payload = {
        **state,
        "complete": True,
        "audit": audit,
        "elapsed_seconds": time.monotonic() - started,
        "output_paths": {name: str(path) for name, path in paths.items()},
    }
    payload.pop("completed_ids", None)
    save_state(payload, paths["result"])

    render_outputs(payload, paths, args)

    if paths["partial"].exists():
        paths["partial"].unlink()

    probe.heading("ATTENTION-HEAD STEERING METHOD COMPARISON COMPLETE")
    probe.print_rows("RESULTS", [
        ("CAA head-ranking rows", len(head_metric_rows)),
        ("Coarse validation conditions", len(state["coarse_rows"])),
        ("Refined validation conditions", len(state["refined_rows"])),
        ("Positive held-out conditions", audit["positive_test_conditions"]),
        ("Matched negative controls", audit["negative_test_controls"]),
        ("Unique test problems per condition", audit["unique_chainsearch_problems_per_condition"]),
        ("Test used for selection", "no"),
        ("Output", paths["result"]),
        ("OVERALL STATUS", audit["overall_status"]),
    ])


if __name__ == "__main__":
    main()
