#!/usr/bin/env python3
"""
attention_head_ablations.py

Run a focused sparse-attention-head CAA steering experiment for ChainSearch.

This is a self-contained experiment driver. It asks whether attention heads
with strong validation-set separation along their CAA directions are also
causally useful. The experiment keeps three data roles
strictly separate:

1. head-selection validation examples rank heads using CAA projection AUROC;
2. disjoint coarse and refined validation examples tune beta, checkpoint, and
   sparse head count;
3. held-out test examples are evaluated once after all choices are fixed.

The intervention is applied once at a stored semantic checkpoint. Varying the
number of simultaneous head locations is the sparse-head ablation here. Model
loading, prefix construction, attention-head hooks, greedy generation, scoring,
bootstrap confidence intervals, selection, persistence, and plotting are all
implemented in this file. Only the project's foundational data, model, parser,
and reachability utilities are imported.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
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
ABLATION_RUN_TAG = "exp34-attention-head-ablation-v2"
ASTAR_DIR = probe.ASTAR_DIR
CAA_PATH = os.path.join(ASTAR_DIR, "caa", f"{SOURCE_RUN_TAG}_astar_caa_vectors.pt")
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


def load_sources(device: torch.device) -> dict[str, Any]:
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
    bank = canonicalise_caa_vectors(caa_archive)
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
        "paths": {"caa": CAA_PATH},
        "archives": {"caa": caa_archive},
        "banks": {"caa": bank},
        "vector_audit": {
            "caa_vectors": len(bank),
            "caa_attention_vectors": sum(v.representation_type == "attention_head" for v in bank.values()),
            "caa_zero_vectors": sum(v.zero_norm for v in bank.values()),
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
    for field in BOOL_SCORE_FIELDS:
        baseline_rate = weighted_mean(base[field].float(), weights)
        steered_rate = weighted_mean(steered[field].float(), weights)
        summary[f"baseline_{field}_rate"] = baseline_rate
        summary[f"steered_{field}_rate"] = steered_rate
        summary[f"delta_{field}_rate"] = steered_rate - baseline_rate
    baseline_correct, steered_correct = base["expr_target_correct"], steered["expr_target_correct"]
    rescue = (~baseline_correct) & steered_correct
    damage = baseline_correct & (~steered_correct)
    summary["rescue_rate"] = weighted_mean(rescue.float(), weights)
    summary["damage_rate"] = weighted_mean(damage.float(), weights)
    summary["net_rescue_rate"] = summary["rescue_rate"] - summary["damage_rate"]
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
        "run_tag": ABLATION_RUN_TAG,
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
    if not bool(shard.get("complete", False)) or shard.get("run_tag") != ABLATION_RUN_TAG:
        raise RuntimeError(f"Invalid attention-head ablation shard: {path}")
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

SELECTION_RULES = (
    "caa_separation",
    "caa_separation_stable",
    "random",
)

RULE_DISPLAY = {
    "caa_separation": "CAA separation",
    "caa_separation_stable": "CAA separation + stability",
    "random": "Random heads",
}

RULE_COLORS = {
    "caa_separation": "#1C4587",
    "caa_separation_stable": "#7CA6DE",
    "random": "#8A94A6",
}

PAPER_FIGURE_BG = "#FCFCFD"
PAPER_PANEL_BG = "#F2F6FF"
PAPER_DEEP_BLUE = "#1C4587"
PAPER_DARK_TEXT = "#26364F"
PAPER_GRID = "#C9D6EE"
PAPER_REFLINE = "#9AA7BD"
PAPER_TITLE = "#000000"

DIAGNOSTIC_FIGURE_STEMS = (
    "caa_attention_head_separation",
    "sparse_attention_head_heldout_effects",
    "attention_head_local_to_global",
    "caa_separation_vs_steering_effect",
)

DEFAULT_BETAS = (0.125, 0.25, 0.5, 1.0, 2.0)
DEFAULT_HEAD_COUNTS = (1, 2, 4, 8)
FORMAT_VERSION = 2


# =============================================================================
# ARGUMENTS AND PATHS
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run validation-selected sparse attention-head CAA steering."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parent.parent
            / "minireason_archives" / "attention_head_ablations"
        ),
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
    parser.add_argument(
        "--selection-rules",
        nargs="+",
        choices=SELECTION_RULES,
        default=list(SELECTION_RULES),
    )
    parser.add_argument("--head-counts", nargs="+", type=int, default=list(DEFAULT_HEAD_COUNTS))
    parser.add_argument("--betas", nargs="+", type=float, default=list(DEFAULT_BETAS))
    parser.add_argument("--head-selection-units", type=int, default=128)
    parser.add_argument("--coarse-validation-units", type=int, default=32)
    parser.add_argument("--refined-validation-units", type=int, default=128)
    parser.add_argument("--test-units", type=int, default=512)
    parser.add_argument("--refine-k", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
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
        "--stability-weight",
        type=float,
        default=0.25,
        help="Rank-percentile weight assigned to local successive-position stability.",
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

    args.tracks = tuple(dict.fromkeys(args.tracks))
    args.checkpoints = tuple(
        checkpoint for checkpoint in OPERATION_CHECKPOINTS if checkpoint in set(args.checkpoints)
    )
    args.selection_rules = tuple(
        rule for rule in SELECTION_RULES if rule in set(args.selection_rules)
    )
    args.head_counts = tuple(sorted(set(args.head_counts)))
    args.betas = tuple(sorted(set(float(beta) for beta in args.betas)))

    if not args.checkpoints or not args.selection_rules:
        raise ValueError("At least one checkpoint and selection rule are required.")
    if not args.head_counts or any(count <= 0 for count in args.head_counts):
        raise ValueError("--head-counts must contain positive integers.")
    if not args.betas or any(not math.isfinite(beta) or beta <= 0 for beta in args.betas):
        raise ValueError("--betas must contain finite positive values; negative controls are added automatically.")
    for name in (
        "head_selection_units",
        "coarse_validation_units",
        "refined_validation_units",
        "test_units",
        "refine_k",
        "batch_size",
        "save_every",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive because the figures require confidence intervals.")
    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be non-negative.")
    if not 0.0 <= args.stability_weight <= 1.0:
        raise ValueError("--stability-weight must lie in [0, 1].")
    if not 0.0 <= args.maximum_validation_validity_drop <= 1.0:
        raise ValueError("--maximum-validation-validity-drop must lie in [0, 1].")
    if not 0.0 <= args.hard_minimum_reproduction_rate <= args.minimum_reproduction_rate <= 1.0:
        raise ValueError("Replay thresholds must satisfy 0 <= hard <= preferred <= 1.")

    if args.smoke:
        args.head_counts = (1, 2)
        args.betas = (0.25, 1.0)
        args.head_selection_units = 128
        args.coarse_validation_units = 2
        args.refined_validation_units = 2
        args.test_units = 2
        args.refine_k = 1
        args.bootstrap_samples = min(args.bootstrap_samples, 50)
        args.save_every = 1
        args.maximum_validation_validity_drop = 1.0

    args.output_dir = args.output_dir.resolve()
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
        "result": args.output_dir / "attention_head_ablations.pt",
        "partial": args.output_dir / "attention_head_ablations.partial.pt",
        "shards": args.output_dir / "generation_shards",
        "ranking_csv": args.output_dir / "caa_head_ranking.csv",
        "effects_csv": args.output_dir / "attention_head_ablation_test_effects.csv",
        "counts_csv": args.output_dir / "attention_head_ablation_transition_counts.csv",
        "changed_csv": args.output_dir / "attention_head_changed_problems.csv",
        "summary": args.output_dir / "attention_head_ablation_summary.txt",
        "figures": args.output_dir / "clear_interpretation",
        "interpretation_tbd": args.output_dir / "interpretation_tbd",
    }


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
                row["separation_stability_score"] = (
                    (1.0 - args.stability_weight) * auc_rank
                    + args.stability_weight * stability_rank
                )

    del selected_features
    return sorted(
        rows_out,
        key=lambda row: (row["task"], args.checkpoints.index(row["checkpoint"]), row["location_name"]),
    )


def build_head_orders(
    metric_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[tuple[str, str, str], list[str]]:
    output: dict[tuple[str, str, str], list[str]] = {}
    tasks = [TASK_BY_TRACK[track] for track in args.tracks]
    for task in tasks:
        for checkpoint in args.checkpoints:
            group = [
                row for row in metric_rows
                if row["task"] == task and row["checkpoint"] == checkpoint
            ]
            if len(group) < max(args.head_counts):
                raise RuntimeError(
                    f"Only {len(group)} attention heads are available for {task}/{checkpoint}; "
                    f"need {max(args.head_counts)}."
                )
            separation = sorted(
                group,
                key=lambda row: (
                    -row["caa_projection_auroc"],
                    -row["caa_standardized_mean_gap"],
                    row["location_name"],
                ),
            )
            stable = sorted(
                group,
                key=lambda row: (
                    -row["separation_stability_score"],
                    -row["caa_projection_auroc"],
                    row["location_name"],
                ),
            )
            random_rows = sorted(group, key=lambda row: row["location_name"])
            random.Random(
                stable_seed("attention_head_ablation", task, checkpoint, "random")
            ).shuffle(random_rows)
            output[(task, checkpoint, "caa_separation")] = [row["location_name"] for row in separation]
            output[(task, checkpoint, "caa_separation_stable")] = [row["location_name"] for row in stable]
            output[(task, checkpoint, "random")] = [row["location_name"] for row in random_rows]
    return output


# =============================================================================
# CONDITIONS AND VALIDATION SELECTION
# =============================================================================

def condition_id(
    task: str,
    checkpoint: str,
    rule: str,
    head_count: int,
    beta: float,
) -> str:
    return (
        f"caa-head-ablation:{task}:{checkpoint}:{rule}:"
        f"k={head_count}:beta={repr(float(beta))}"
    )


def build_conditions(
    sources: dict[str, Any],
    head_orders: dict[tuple[str, str, str], list[str]],
    args: argparse.Namespace,
) -> list[Any]:
    caa_bank = sources["banks"]["caa"]
    output = []
    for track in args.tracks:
        task = TASK_BY_TRACK[track]
        for checkpoint in args.checkpoints:
            for rule in args.selection_rules:
                order = head_orders[(task, checkpoint, rule)]
                for head_count in args.head_counts:
                    names = order[:head_count]
                    member_keys = tuple(
                        vector_key(task, checkpoint, name) for name in names
                    )
                    divisor = math.sqrt(head_count)
                    member_scales = tuple(
                        float(caa_bank[key].raw_norm) / divisor for key in member_keys
                    )
                    for magnitude in args.betas:
                        for beta in (magnitude, -magnitude):
                            output.append(SteeringCondition(
                                condition_id=condition_id(task, checkpoint, rule, head_count, beta),
                                method="caa",
                                vector_key=f"{task}:{checkpoint}:{rule}:k={head_count}",
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
                                selection_variant=f"{rule}_top{head_count}",
                                head_count=head_count,
                                head_selection=rule,
                                head_scaling="fixed_energy",
                            ))
    ids = [condition.condition_id for condition in output]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Attention-head ablation generated duplicate condition IDs.")
    return output


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


def select_refined_ids(
    coarse_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[str]:
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in coarse_rows:
        groups[(
            row["task"],
            row["checkpoint"],
            row["head_selection"],
            int(row["head_count"]),
        )].append(row)
    selected = []
    for key in sorted(groups):
        candidates = [
            row for row in groups[key]
            if float(row["delta_valid_trace_rate"])
            >= -args.maximum_validation_validity_drop
        ]
        if candidates:
            ranked = sorted(
                candidates,
                key=lambda row: validation_rank(row, args),
                reverse=True,
            )
        else:
            best_validity = max(
                float(row["delta_valid_trace_rate"]) for row in groups[key]
            )
            print(
                f"WARNING: coarse cell {key} has no candidate within the "
                f"{100.0 * args.maximum_validation_validity_drop:.3f} pp validity limit; "
                f"advancing the least validity-damaging candidates "
                f"(best change {100.0 * best_validity:+.3f} pp) to refined validation.",
                flush=True,
            )
            ranked = sorted(
                groups[key],
                key=lambda row: (
                    float(row["delta_valid_trace_rate"]),
                    float(primary_effect(row)),
                    float(row["delta_expr_target_correct_rate"]),
                    -abs(float(row["beta"])),
                    str(row["condition_id"]),
                ),
                reverse=True,
            )
        selected.extend(row["condition_id"] for row in ranked[:args.refine_k])
    return list(dict.fromkeys(selected))


def select_test_ids(
    refined_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, list[str]]]:
    roles: dict[str, list[str]] = defaultdict(list)

    breadth_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in refined_rows:
        breadth_groups[(row["task"], row["head_selection"], int(row["head_count"]))].append(row)
    for key, group in breadth_groups.items():
        candidates = eligible_validation_rows(group, args, f"refined breadth cell {key}")
        winner = max(candidates, key=lambda row: validation_rank(row, args))
        roles[winner["condition_id"]].append("breadth")

    trajectory_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in refined_rows:
        if row["head_selection"] == "caa_separation":
            trajectory_groups[(row["task"], row["checkpoint"])].append(row)
    for key, group in trajectory_groups.items():
        candidates = eligible_validation_rows(group, args, f"refined trajectory cell {key}")
        winner = max(candidates, key=lambda row: validation_rank(row, args))
        roles[winner["condition_id"]].append("trajectory")

    return sorted(roles), dict(roles)


def add_negative_controls(
    positive_ids: list[str],
    condition_lookup: dict[str, Any],
) -> tuple[list[str], dict[str, str]]:
    identity_lookup = {
        (
            condition.task,
            condition.checkpoint,
            condition.head_selection,
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
            positive.task,
            positive.checkpoint,
            positive.head_selection,
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
    })
    return {
        "format_version": FORMAT_VERSION,
        "experiment": "attention-head-ablation",
        "ablation_run_tag": ABLATION_RUN_TAG,
        "source_run_tag": SOURCE_RUN_TAG,
        "tracks": list(args.tracks),
        "tasks": [TASK_BY_TRACK[track] for track in args.tracks],
        "checkpoints": list(args.checkpoints),
        "selection_rules": list(args.selection_rules),
        "head_counts": list(args.head_counts),
        "betas": list(args.betas),
        "head_selection_units": args.head_selection_units,
        "coarse_validation_units": args.coarse_validation_units,
        "refined_validation_units": args.refined_validation_units,
        "test_units": args.test_units,
        "refine_k": args.refine_k,
        "batch_size": args.batch_size,
        "bootstrap_samples": args.bootstrap_samples,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "maximum_validation_validity_drop": args.maximum_validation_validity_drop,
        "stability_weight": args.stability_weight,
        "unique_chainsearch_problems": True,
        "head_scaling": "fixed_energy",
        "conditions": [
            {
                "condition_id": condition.condition_id,
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
        "refined_condition_ids": [],
        "refined_rows": [],
        "positive_test_ids": [],
        "test_roles": {},
        "negative_control_of": {},
        "test_rows": [],
        "baseline_rows": [],
        "problem_ids": {},
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
            raise RuntimeError("Partial attention-head ablation has a different configuration.")
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
        ("caa", key): vector
        for key, vector in sources["banks"]["caa"].items()
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
                condition_lookup[cid].task,
                args.checkpoints.index(condition_lookup[cid].checkpoint),
                condition_lookup[cid].head_selection,
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
                sources, task, checkpoint, split, caps[phase], offsets[phase],
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
                execution_groups[(condition.task, condition.checkpoint, condition.beta, members)].append(
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
    coarse_ids = [condition.condition_id for condition in positive_conditions]
    evaluate_ids(coarse_ids, "coarse", "coarse_rows")

    if not state["refined_condition_ids"]:
        state["refined_condition_ids"] = select_refined_ids(state["coarse_rows"], args)
        save_state(state, paths["partial"])
    evaluate_ids(state["refined_condition_ids"], "refined", "refined_rows")

    if not state["positive_test_ids"]:
        positive_ids, roles = select_test_ids(state["refined_rows"], args)
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


def effect_and_ci(row: dict[str, Any]) -> tuple[float, float, float]:
    if row["label_family"] == "reachability":
        field = "delta_next_state_viable_rate"
    else:
        field = "delta_expr_target_correct_rate"
    return (
        float(row[field]),
        float(row[f"{field.removesuffix('_rate')}_ci_low"]),
        float(row[f"{field.removesuffix('_rate')}_ci_high"]),
    )


def save_figure(
    fig: plt.Figure,
    stem: str,
    paths: dict[str, Path],
    args: argparse.Namespace,
    directory_key: str = "figures",
) -> None:
    output_dir = paths[directory_key]
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in args.formats:
        fig.savefig(output_dir / f"{stem}.{extension}", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def style_clear_interpretation_figure(fig: plt.Figure) -> None:
    """Match the blue paper aesthetic without changing diagnostic figures."""
    fig.patch.set_facecolor(PAPER_FIGURE_BG)
    legends = list(fig.legends)
    for axis in fig.axes:
        axis.set_facecolor(PAPER_PANEL_BG)
        axis.title.set_color(PAPER_DEEP_BLUE)
        axis.xaxis.label.set_color(PAPER_DEEP_BLUE)
        axis.yaxis.label.set_color(PAPER_DEEP_BLUE)
        axis.tick_params(axis="both", colors=PAPER_DEEP_BLUE)
        axis.spines["left"].set_color(PAPER_DEEP_BLUE)
        axis.spines["bottom"].set_color(PAPER_DEEP_BLUE)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(True, axis="y", color=PAPER_GRID, alpha=0.55)
        legend = axis.get_legend()
        if legend is not None:
            legends.append(legend)
    for legend in legends:
        legend.get_frame().set_facecolor(PAPER_FIGURE_BG)
        legend.get_frame().set_edgecolor(PAPER_DEEP_BLUE)
        legend.get_frame().set_alpha(0.9)
        for text in legend.get_texts():
            text.set_color(PAPER_DARK_TEXT)


def relocate_existing_diagnostic_figures(paths: dict[str, Path]) -> None:
    """Move earlier diagnostic renders out of the paper-facing figure directory."""
    paths["interpretation_tbd"].mkdir(parents=True, exist_ok=True)
    for stem in DIAGNOSTIC_FIGURE_STEMS:
        for extension in ("png", "pdf"):
            source = paths["figures"] / f"{stem}.{extension}"
            if source.is_file():
                os.replace(source, paths["interpretation_tbd"] / source.name)


def plot_head_ranking(
    metric_rows: list[dict[str, Any]],
    head_orders: dict[tuple[str, str, str], list[str]],
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> None:
    tasks = [TASK_BY_TRACK[track] for track in args.tracks]
    fig, axes = plt.subplots(len(tasks), 1, figsize=(13.0, 3.2 * len(tasks)), squeeze=False)
    locations = sorted(
        {row["location_name"] for row in metric_rows},
        key=lambda name: (
            next(row["layer"] for row in metric_rows if row["location_name"] == name),
            next(row["head"] for row in metric_rows if row["location_name"] == name),
        ),
    )
    for axis, task in zip(axes[:, 0], tasks):
        lookup = {
            (row["checkpoint"], row["location_name"]): row["caa_projection_auroc"]
            for row in metric_rows if row["task"] == task
        }
        matrix = np.asarray([
            [lookup[(checkpoint, location)] for location in locations]
            for checkpoint in args.checkpoints
        ])
        image = axis.imshow(matrix, aspect="auto", vmin=0.5, vmax=1.0, cmap="Blues")
        axis.set_yticks(range(len(args.checkpoints)), [CHECKPOINT_DISPLAY[c] for c in args.checkpoints])
        axis.set_xticks(range(len(locations)), locations, rotation=75, ha="right", fontsize=7)
        track = next(track for track, current in TASK_BY_TRACK.items() if current == task)
        axis.set_title(f"{track.replace('_', ' ').title()} CAA projection AUROC")
        for row_index, checkpoint in enumerate(args.checkpoints):
            selected = set(head_orders[(task, checkpoint, "caa_separation")][:max(args.head_counts)])
            for column, location in enumerate(locations):
                if location in selected:
                    axis.add_patch(plt.Rectangle(
                        (column - 0.5, row_index - 0.5), 1, 1,
                        fill=False, edgecolor="#D57758", linewidth=1.0,
                    ))
        fig.colorbar(image, ax=axis, fraction=0.018, pad=0.012)
    fig.suptitle("Validation-only CAA separation at individual attention heads")
    fig.subplots_adjust(hspace=0.55, bottom=0.22)
    save_figure(
        fig, "caa_attention_head_separation", paths, args,
        directory_key="interpretation_tbd",
    )


def positive_test_rows(state: dict[str, Any], role: str) -> list[dict[str, Any]]:
    positive = set(state["positive_test_ids"])
    return [
        row for row in state["test_rows"]
        if row["condition_id"] in positive and role in row.get("test_roles", [])
    ]


COUNT_FIELDS = ("next_state_viable", "expr_target_correct")


def existing_shard_path(candidates: list[str | Path | None], description: str) -> Path:
    checked = []
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        checked.append(str(path))
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Cannot construct exact-count plots because the {description} shard is missing. "
        f"Checked: {checked}. Preserve the generation_shards directory produced by the run."
    )


def compact_count_shard(path: Path) -> dict[str, Any]:
    shard = load_shard(str(path))
    missing = [field for field in COUNT_FIELDS if field not in shard["labels"]]
    if missing:
        raise RuntimeError(f"Shard {path} is missing count fields: {missing}")
    return {
        "path": str(path),
        "cache_idx": shard["cache_idx"].to(torch.int64),
        "source_row": shard["source_row"].to(torch.int64),
        "record_pos": shard["record_pos"].to(torch.int64),
        "generated_idx": shard["generated_idx"].to(torch.int64),
        "labels": {
            field: shard["labels"][field].to(torch.bool)
            for field in COUNT_FIELDS
        },
    }


def build_paired_test_cache(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Load only the identifiers and Boolean labels needed for exact paired counts."""
    baseline_rows = {
        (row["task"], row["checkpoint"]): row
        for row in state.get("baseline_rows", [])
        if row.get("phase") == "test"
    }
    baseline_cache: dict[tuple[str, str], dict[str, Any]] = {}
    execution_cache: dict[str, dict[str, Any]] = {}
    paired: dict[str, dict[str, Any]] = {}

    for row in state["test_rows"]:
        task = str(row["task"])
        checkpoint = str(row["checkpoint"])
        baseline_key = (task, checkpoint)
        if baseline_key not in baseline_cache:
            saved = baseline_rows.get(baseline_key, {}).get("shard_path")
            baseline_path = existing_shard_path(
                [
                    saved,
                    shard_path(
                        "baseline",
                        baseline_identifier("test", task, checkpoint),
                    ),
                ],
                f"test baseline for {task}/{checkpoint}",
            )
            baseline_cache[baseline_key] = compact_count_shard(baseline_path)

        execution_id = str(row.get("execution_condition_id") or row["condition_id"])
        if execution_id not in execution_cache:
            steered_path = existing_shard_path(
                [row.get("shard_path"), shard_path("test", execution_id)],
                f"test result for {execution_id}",
            )
            execution_cache[execution_id] = compact_count_shard(steered_path)

        baseline = baseline_cache[baseline_key]
        steered = execution_cache[execution_id]
        for field in ("cache_idx", "source_row", "record_pos", "generated_idx"):
            if not torch.equal(baseline[field], steered[field]):
                raise RuntimeError(
                    f"Paired identifiers differ for {row['condition_id']} in field {field}."
                )
        paired[str(row["condition_id"])] = {
            "baseline": baseline,
            "steered": steered,
        }
    return paired


def paired_transition(
    row: dict[str, Any],
    field: str,
    paired_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pair = paired_cache[str(row["condition_id"])]
    before = pair["baseline"]["labels"][field]
    after = pair["steered"]["labels"][field]
    improved = (~before) & after
    worsened = before & (~after)
    return {
        "before": before,
        "after": after,
        "improved_mask": improved,
        "worsened_mask": worsened,
        "improved": int(improved.sum().item()),
        "worsened": int(worsened.sum().item()),
        "net": int(improved.sum().item() - worsened.sum().item()),
        "unchanged": int((before == after).sum().item()),
        "n": int(before.numel()),
        "baseline_positive": int(before.sum().item()),
        "steered_positive": int(after.sum().item()),
        "baseline_path": pair["baseline"]["path"],
        "steered_path": pair["steered"]["path"],
    }


def transition_count_rows(
    state: dict[str, Any],
    paired_cache: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    positive = set(state["positive_test_ids"])
    output = []
    for row in state["test_rows"]:
        condition_id_value = str(row["condition_id"])
        direction = "selected" if condition_id_value in positive else "reversed"
        for field in COUNT_FIELDS:
            counts = paired_transition(row, field, paired_cache)
            output.append({
                "condition_id": condition_id_value,
                "positive_condition_id": (
                    condition_id_value
                    if direction == "selected"
                    else state["negative_control_of"][condition_id_value]
                ),
                "direction": direction,
                "test_roles": row.get("test_roles", []),
                "task": row["task"],
                "label_family": row["label_family"],
                "checkpoint": row["checkpoint"],
                "head_selection": row["head_selection"],
                "head_count": int(row["head_count"]),
                "beta": float(row["beta"]),
                "metric": field,
                "baseline_positive": counts["baseline_positive"],
                "steered_positive": counts["steered_positive"],
                "improved": counts["improved"],
                "worsened": counts["worsened"],
                "net": counts["net"],
                "unchanged": counts["unchanged"],
                "problems": counts["n"],
            })
    return output


def changed_problem_rows(
    state: dict[str, Any],
    paired_cache: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    positive = set(state["positive_test_ids"])
    output = []
    for row in state["test_rows"]:
        condition_id_value = str(row["condition_id"])
        direction = "selected" if condition_id_value in positive else "reversed"
        pair = paired_cache[condition_id_value]
        for field in COUNT_FIELDS:
            changes = paired_transition(row, field, paired_cache)
            changed_indices = torch.nonzero(
                changes["improved_mask"] | changes["worsened_mask"],
                as_tuple=False,
            ).flatten()
            for index_tensor in changed_indices:
                index = int(index_tensor.item())
                improved = bool(changes["improved_mask"][index].item())
                output.append({
                    "condition_id": condition_id_value,
                    "positive_condition_id": (
                        condition_id_value
                        if direction == "selected"
                        else state["negative_control_of"][condition_id_value]
                    ),
                    "direction": direction,
                    "test_roles": row.get("test_roles", []),
                    "task": row["task"],
                    "checkpoint": row["checkpoint"],
                    "head_selection": row["head_selection"],
                    "head_count": int(row["head_count"]),
                    "beta": float(row["beta"]),
                    "location_name": row["location_name"],
                    "metric": field,
                    "transition": "failure_to_success" if improved else "success_to_failure",
                    "problem_id": int(pair["baseline"]["cache_idx"][index].item()),
                    "source_row": int(pair["baseline"]["source_row"][index].item()),
                    "record_pos": int(pair["baseline"]["record_pos"][index].item()),
                    "generated_idx": int(pair["baseline"]["generated_idx"][index].item()),
                    "baseline_value": int(changes["before"][index].item()),
                    "steered_value": int(changes["after"][index].item()),
                    "baseline_shard": changes["baseline_path"],
                    "steered_shard": changes["steered_path"],
                })
    return output


def finish_count_axis(
    axis: plt.Axes,
    positive_maximum: int,
    negative_maximum: int,
) -> float:
    positive_limit = max(0, int(positive_maximum))
    negative_limit = max(0, int(negative_maximum))
    scale = max(1, positive_limit, negative_limit)
    padding = max(0.18, 0.07 * scale)
    upper = positive_limit + 2.5 * padding if positive_limit else 2.5 * padding
    lower = -negative_limit - 2.5 * padding if negative_limit else -2.5 * padding
    axis.set_ylim(lower, upper)
    axis.axhline(0.0, color=PAPER_REFLINE, linestyle="--", linewidth=1.0, zorder=1)
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(True, axis="y", alpha=0.35)
    return padding


def annotate_diverging_bar(
    axis: plt.Axes,
    x: float,
    improved: int,
    worsened: int,
    padding: float,
) -> None:
    if improved:
        axis.text(x, improved + padding, f"+{improved}", ha="center", va="bottom", fontsize=9.2)
    if worsened:
        axis.text(x, -worsened - padding, f"−{worsened}", ha="center", va="top", fontsize=9.2)


def head_count_text(head_count: int) -> str:
    noun = "head" if int(head_count) == 1 else "heads"
    return f"{int(head_count)} {noun}"


NO_CHANGE_LABEL = "No changes"
NO_CHANGE_EDGE = PAPER_DEEP_BLUE


def draw_no_change_markers(
    axis: plt.Axes,
    positions: list[float] | np.ndarray,
    improved: list[int],
    worsened: list[int],
) -> bool:
    """Mark genuine 0/0 transitions without inventing nonzero bar heights."""
    zero_positions = [
        float(position)
        for position, gained, lost in zip(positions, improved, worsened)
        if int(gained) == 0 and int(lost) == 0
    ]
    if not zero_positions:
        return False
    axis.scatter(
        zero_positions,
        np.zeros(len(zero_positions)),
        marker="D",
        s=28,
        facecolors=PAPER_PANEL_BG,
        edgecolors=NO_CHANGE_EDGE,
        linewidths=1.1,
        zorder=5,
    )
    return True


def no_change_legend_handle() -> Line2D:
    return Line2D(
        [], [],
        linestyle="none",
        marker="D",
        markersize=5.5,
        markerfacecolor=PAPER_PANEL_BG,
        markeredgecolor=NO_CHANGE_EDGE,
        markeredgewidth=1.1,
        label=NO_CHANGE_LABEL,
    )


def plot_sparse_breadth_counts(
    state: dict[str, Any],
    paired_cache: dict[str, dict[str, Any]],
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> None:
    rows = positive_test_rows(state, "breadth")
    rules = [rule for rule in ("caa_separation", "random") if rule in args.selection_rules]
    if not rows or not rules:
        return
    fig, axes = plt.subplots(
        1, len(args.tracks), figsize=(6.0 * len(args.tracks), 4.8), squeeze=False,
    )
    x = np.arange(len(args.head_counts), dtype=float)
    width = 0.34 if len(rules) > 1 else 0.52
    offsets = np.linspace(-width / 2.0, width / 2.0, len(rules)) if len(rules) > 1 else [0.0]
    legend_handles = []
    figure_has_no_change = False

    for axis, track in zip(axes[0], args.tracks):
        task = TASK_BY_TRACK[track]
        field = "next_state_viable" if track == "reachability" else "expr_target_correct"
        lookups = {
            rule: {
                int(row["head_count"]): row
                for row in rows
                if row["task"] == task and row["head_selection"] == rule
            }
            for rule in rules
        }
        all_improved = []
        all_worsened = []
        bars_to_annotate = []
        for offset, rule in zip(offsets, rules):
            lookup = lookups[rule]
            positions, improved, worsened = [], [], []
            for index, head_count in enumerate(args.head_counts):
                row = lookup.get(int(head_count))
                if row is None:
                    continue
                counts = paired_transition(row, field, paired_cache)
                positions.append(float(x[index] + offset))
                improved.append(counts["improved"])
                worsened.append(counts["worsened"])
            color = RULE_COLORS[rule]
            axis.bar(positions, improved, width=width, color=color, alpha=0.92)
            axis.bar(
                positions, [-value for value in worsened], width=width,
                color=color, alpha=0.45, hatch="///", edgecolor=color,
            )
            figure_has_no_change |= draw_no_change_markers(
                axis, positions, improved, worsened,
            )
            all_improved.extend(improved)
            all_worsened.extend(worsened)
            bars_to_annotate.extend(zip(positions, improved, worsened))

        padding = finish_count_axis(
            axis,
            max(all_improved, default=0),
            max(all_worsened, default=0),
        )
        for position, improved, worsened in bars_to_annotate:
            annotate_diverging_bar(axis, position, improved, worsened, padding)
        setting_labels = []
        for head_count in args.head_counts:
            beta_values = []
            for rule in rules:
                row = lookups[rule].get(int(head_count))
                beta_values.append(
                    "—" if row is None else f"{abs(float(row['beta'])):g}"
                )
            setting_labels.append(
                f"{int(head_count)}\nβ {' / '.join(beta_values)}"
            )
        axis.set_xticks(x, setting_labels)
        axis.tick_params(
            axis="both",
            labelsize=11,
            colors=PAPER_DEEP_BLUE,
        )
        axis.set_xlabel(
            "Number of steered attention heads",
            fontsize=12.5,
        )
        axis.set_title(
            "Next-state reachability"
            if track == "reachability"
            else "Final-answer correctness",
            fontsize=13.0,
        )

    for rule in rules:
        legend_handles.extend([
            Patch(
                facecolor=RULE_COLORS[rule], alpha=0.92,
                label=f"{RULE_DISPLAY[rule]}  failure → success",
            ),
            Patch(
                facecolor=RULE_COLORS[rule], alpha=0.45,
                edgecolor=RULE_COLORS[rule], hatch="///",
                label=f"{RULE_DISPLAY[rule]}  success → failure",
            ),
        ])
    if figure_has_no_change:
        legend_handles.append(no_change_legend_handle())
    fig.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.535, 0.945),
        ncol=min(3, len(legend_handles)), fontsize=11, frameon=True,
    )
    fig.supylabel(
        "Changed test problems", x=0.03, y=0.50, fontsize=12.5,
        color=PAPER_DEEP_BLUE,
    )
    fig.suptitle(
        "Effects of attention-head selection and count",
        x=0.535,
        y=0.990,
        fontsize=14.75,
        color=PAPER_TITLE,
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.735, bottom=0.25, wspace=0.12)
    style_clear_interpretation_figure(fig)
    save_figure(fig, "sparse_attention_head_heldout_counts", paths, args)


def plot_local_to_global_counts(
    state: dict[str, Any],
    paired_cache: dict[str, dict[str, Any]],
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> None:
    if "reachability" not in args.tracks:
        return
    rows = [
        row for row in positive_test_rows(state, "trajectory")
        if row["task"] == TASK_BY_TRACK["reachability"]
        and row["head_selection"] == "caa_separation"
    ]
    lookup = {row["checkpoint"]: row for row in rows}
    checkpoints = [checkpoint for checkpoint in args.checkpoints if checkpoint in lookup]
    if not checkpoints:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharex=True)
    specifications = (
        (
            "next_state_viable",
            "Next-state reachability",
            "Unreachable → reachable",
            "Reachable → unreachable",
        ),
        (
            "expr_target_correct",
            "Final-answer correctness",
            "Incorrect → correct",
            "Correct → incorrect",
        ),
    )
    x = np.arange(len(checkpoints), dtype=float)
    for axis, (field, title, gain_label, loss_label) in zip(axes, specifications):
        counts = [paired_transition(lookup[checkpoint], field, paired_cache) for checkpoint in checkpoints]
        improved = [value["improved"] for value in counts]
        worsened = [value["worsened"] for value in counts]
        axis.bar(x, improved, width=0.56, color=RULE_COLORS["caa_separation"], label=gain_label)
        axis.bar(
            x, [-value for value in worsened], width=0.56,
            color="#AEB6C5", hatch="///", edgecolor="#7C879A", label=loss_label,
        )
        has_no_change = draw_no_change_markers(axis, x, improved, worsened)
        padding = finish_count_axis(
            axis,
            max(improved, default=0),
            max(worsened, default=0),
        )
        for position, gained, lost in zip(x, improved, worsened):
            annotate_diverging_bar(axis, float(position), gained, lost, padding)
        axis.set_title(title, fontsize=13.0)
        handles, labels = axis.get_legend_handles_labels()
        if has_no_change:
            handles.append(no_change_legend_handle())
            labels.append(NO_CHANGE_LABEL)
        axis.legend(handles, labels, loc="best", fontsize=10)

    lower, upper = axes[1].get_ylim()
    axes[1].set_ylim(lower - 0.25, upper + 0.25)

    tick_labels = [
        f"{CHECKPOINT_DISPLAY[checkpoint]}\n"
        f"{head_count_text(int(lookup[checkpoint]['head_count']))}, "
        f"β={float(lookup[checkpoint]['beta']):g}"
        for checkpoint in checkpoints
    ]

    for axis in axes:
        axis.set_xticks(
            x,
            tick_labels,
            rotation=30,
            ha="right",
        )
        axis.tick_params(
            axis="x",
            labelsize=11,
            colors=PAPER_DEEP_BLUE,
        )

    fig.supxlabel(
        "Intervention position",
        y=0.008,
        fontsize=12.5,
        color=PAPER_DEEP_BLUE,
    )
    fig.suptitle(
        "Effects of CAA steering across intervention positions",
        x=0.53,
        y=0.985,
        fontsize=14.75,
        color=PAPER_TITLE,
    )
    fig.supylabel(
        "Changed test problems", x=0.03, y=0.575, fontsize=12.5,
        color=PAPER_DEEP_BLUE,
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.30, top=0.86, wspace=0.12)
    style_clear_interpretation_figure(fig)
    save_figure(fig, "attention_head_local_to_global_counts", paths, args)


def plot_directionality_counts(
    state: dict[str, Any],
    paired_cache: dict[str, dict[str, Any]],
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> None:
    if "reachability" not in args.tracks:
        return
    positives = [
        row for row in positive_test_rows(state, "trajectory")
        if row["task"] == TASK_BY_TRACK["reachability"]
        and row["head_selection"] == "caa_separation"
    ]
    positive_lookup = {row["checkpoint"]: row for row in positives}
    negative_by_positive = {
        positive_id: negative_id
        for negative_id, positive_id in state["negative_control_of"].items()
    }
    row_lookup = {row["condition_id"]: row for row in state["test_rows"]}
    checkpoints = [
        checkpoint for checkpoint in args.checkpoints
        if checkpoint in positive_lookup
        and positive_lookup[checkpoint]["condition_id"] in negative_by_positive
    ]
    if not checkpoints:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), squeeze=False)
    specifications = (
        ("next_state_viable", "Next-state reachability"),
        ("expr_target_correct", "Final-answer correctness"),
    )
    directions = (
        ("CAA direction (+β)", "#1C4587"),
        ("Reversed CAA direction (−β)", "#D57758"),
    )
    x = np.arange(len(checkpoints), dtype=float)
    width = 0.36

    for axis, (field, title) in zip(axes[0], specifications):
        positive_extents = []
        negative_extents = []
        annotations = []
        axis_has_no_change = False
        for direction_index, (label, color) in enumerate(directions):
            values, gains, losses = [], [], []
            positions = x + (-width / 2.0 if direction_index == 0 else width / 2.0)
            for checkpoint in checkpoints:
                positive_row = positive_lookup[checkpoint]
                if direction_index == 0:
                    row = positive_row
                else:
                    row = row_lookup[negative_by_positive[positive_row["condition_id"]]]
                counts = paired_transition(row, field, paired_cache)
                values.append(counts["net"])
                gains.append(counts["improved"])
                losses.append(counts["worsened"])
            axis.bar(positions, values, width=width, color=color, alpha=0.88, label=label)
            axis_has_no_change |= draw_no_change_markers(
                axis, positions, gains, losses,
            )
            positive_extents.extend(max(0, value) for value in values)
            negative_extents.extend(max(0, -value) for value in values)
            annotations.extend(zip(positions, values, gains, losses))

        padding = finish_count_axis(
            axis,
            max(positive_extents, default=0),
            max(negative_extents, default=0),
        )
        for position, net, gained, lost in annotations:
            if not gained and not lost:
                continue
            vertical = net + padding if net >= 0 else net - padding
            axis.text(
                float(position), vertical, f"↑+{gained}\n↓−{lost}",
                ha="center", va="bottom" if net >= 0 else "top", fontsize=9.2,
            )
        axis.set_xticks(
            x,
            [
                f"{CHECKPOINT_DISPLAY[checkpoint]}\n"
                f"{head_count_text(int(positive_lookup[checkpoint]['head_count']))}, "
                f"β=±{abs(float(positive_lookup[checkpoint]['beta'])):g}"
                for checkpoint in checkpoints
            ],
            rotation=22,
            ha="right",
        )
        axis.set_title(title, fontsize=13.0)
        axis.tick_params(
            axis="both",
            labelsize=11,
            colors=PAPER_DEEP_BLUE,
        )
        handles, labels = axis.get_legend_handles_labels()
        if axis_has_no_change:
            handles.append(no_change_legend_handle())
            labels.append(NO_CHANGE_LABEL)
        axis.legend(handles, labels, loc="lower left", fontsize=10)

    final_answer_axis = axes[0, 1]
    final_answer_axis.set_ylim(-5, 7)
    final_answer_axis.set_yticks(np.arange(-4, 7, 1))
    
    fig.supxlabel(
        "Intervention position",
        y=0.008,
        fontsize=12.5,
        color=PAPER_DEEP_BLUE,
    )

    fig.suptitle(
        "Effects of reversing CAA directions",
        x=0.53, y=0.985, fontsize=14.75,
        color=PAPER_TITLE,
    )
    fig.supylabel(
        "Changed test problems", x=0.03, y=0.575, fontsize=12.5,
        color=PAPER_DEEP_BLUE,
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.30, wspace=0.12)
    style_clear_interpretation_figure(fig)
    save_figure(fig, "attention_head_directionality_counts", paths, args)


def plot_sparse_breadth(state: dict[str, Any], paths: dict[str, Path], args: argparse.Namespace) -> None:
    rows = positive_test_rows(state, "breadth")
    fig, axes = plt.subplots(1, len(args.tracks), figsize=(6.0 * len(args.tracks), 4.4), squeeze=False)
    for axis, track in zip(axes[0], args.tracks):
        task = TASK_BY_TRACK[track]
        subset = [row for row in rows if row["task"] == task]
        for rule in args.selection_rules:
            rule_rows = sorted(
                [row for row in subset if row["head_selection"] == rule],
                key=lambda row: int(row["head_count"]),
            )
            if not rule_rows:
                continue
            x, y, lower, upper = [], [], [], []
            for row in rule_rows:
                effect, low, high = effect_and_ci(row)
                effect, low, high = 100.0 * effect, 100.0 * low, 100.0 * high
                x.append(int(row["head_count"]))
                y.append(effect)
                lower.append(max(0.0, effect - low))
                upper.append(max(0.0, high - effect))
            axis.errorbar(
                x, y, yerr=[lower, upper], marker="o", capsize=3,
                color=RULE_COLORS[rule], label=RULE_DISPLAY[rule],
            )
        axis.axhline(0.0, color="#8A94A6", linestyle="--", linewidth=1.0)
        axis.set_xticks(args.head_counts)
        axis.set_xlabel("Simultaneously steered attention heads")
        axis.set_ylabel("Held-out primary effect (percentage points)")
        axis.set_title(track.replace("_", " ").title())
        axis.grid(True, axis="y", alpha=0.4)
        axis.legend(fontsize=8)
    fig.suptitle("Sparse CAA attention-head steering on held-out ChainSearch problems")
    fig.subplots_adjust(top=0.82, wspace=0.28)
    save_figure(
        fig, "sparse_attention_head_heldout_effects", paths, args,
        directory_key="interpretation_tbd",
    )


def plot_local_to_global(state: dict[str, Any], paths: dict[str, Path], args: argparse.Namespace) -> None:
    if "reachability" not in args.tracks:
        return
    rows = [
        row for row in positive_test_rows(state, "trajectory")
        if row["task"] == TASK_BY_TRACK["reachability"]
    ]
    lookup = {row["checkpoint"]: row for row in rows}
    checkpoints = [checkpoint for checkpoint in args.checkpoints if checkpoint in lookup]
    if not checkpoints:
        return
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.4), sharex=True)
    specifications = (
        (
            "delta_next_state_viable_rate",
            "Valid-and-reachable next-state change (pp)",
            "Local effect",
        ),
        ("delta_expr_target_correct_rate", "Final-answer accuracy change (pp)", "Global effect"),
    )
    for axis, (field, ylabel, title) in zip(axes, specifications):
        values, lower, upper = [], [], []
        for checkpoint in checkpoints:
            row = lookup[checkpoint]
            value = 100.0 * float(row[field])
            low = 100.0 * float(row[f"{field.removesuffix('_rate')}_ci_low"])
            high = 100.0 * float(row[f"{field.removesuffix('_rate')}_ci_high"])
            values.append(value)
            lower.append(max(0.0, value - low))
            upper.append(max(0.0, high - value))
        axis.errorbar(range(len(checkpoints)), values, yerr=[lower, upper], marker="o", capsize=3)
        axis.axhline(0.0, color="#8A94A6", linestyle="--", linewidth=1.0)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.4)
    axes[1].set_xticks(
        range(len(checkpoints)),
        [CHECKPOINT_DISPLAY[checkpoint] for checkpoint in checkpoints],
        rotation=22,
        ha="right",
    )
    axes[1].set_xlabel("Semantic position within the arithmetic operation")
    fig.suptitle("Validation-selected sparse CAA reachability steering")
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.19, top=0.88, hspace=0.32)
    save_figure(
        fig, "attention_head_local_to_global", paths, args,
        directory_key="interpretation_tbd",
    )


def plot_separation_vs_validation_effect(
    state: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> None:
    metric_lookup = {
        (row["task"], row["checkpoint"], row["location_name"]): row
        for row in metric_rows
    }
    fig, axes = plt.subplots(1, len(args.tracks), figsize=(6.0 * len(args.tracks), 4.4), squeeze=False)
    for axis, track in zip(axes[0], args.tracks):
        task = TASK_BY_TRACK[track]
        for rule in args.selection_rules:
            rows = [
                row for row in state["refined_rows"]
                if row["task"] == task and row["head_selection"] == rule
            ]
            x, y, sizes = [], [], []
            for row in rows:
                names = str(row["location_name"]).split("+")
                aurocs = [
                    metric_lookup[(task, row["checkpoint"], name)]["caa_projection_auroc"]
                    for name in names
                ]
                x.append(float(np.mean(aurocs)))
                y.append(100.0 * float(primary_effect(row)))
                sizes.append(18.0 + 7.0 * int(row["head_count"]))
            if x:
                axis.scatter(x, y, s=sizes, alpha=0.55, color=RULE_COLORS[rule], label=RULE_DISPLAY[rule])
        axis.axhline(0.0, color="#8A94A6", linestyle="--", linewidth=1.0)
        axis.set_xlabel("Mean validation CAA projection AUROC")
        axis.set_ylabel("Refined-validation primary effect (pp)")
        axis.set_title(track.replace("_", " ").title())
        axis.grid(True, alpha=0.35)
        axis.legend(fontsize=8)
    fig.suptitle("Representation-level separation versus validation steering effect")
    fig.subplots_adjust(top=0.82, wspace=0.28)
    save_figure(
        fig, "caa_separation_vs_steering_effect", paths, args,
        directory_key="interpretation_tbd",
    )


def write_summary(
    state: dict[str, Any],
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> None:
    lines = [
        "ATTENTION-HEAD CAA ABLATION",
        "",
        "Head ranking used only head-selection validation examples.",
        "Beta and configuration selection used disjoint steering-validation examples.",
        "Held-out test examples were not used for selection.",
        "Every reported test unit is one unique ChainSearch problem.",
        "",
        f"Head-selection problems per task: {args.head_selection_units}",
        f"Coarse-validation problems per condition: {args.coarse_validation_units}",
        f"Refined-validation problems per condition: {args.refined_validation_units}",
        f"Held-out test problems per condition: {args.test_units}",
        f"Positive held-out configurations: {len(state['positive_test_ids'])}",
        f"Matched negative controls: {len(state['negative_control_of'])}",
        "",
        "HELD-OUT POSITIVE RESULTS",
    ]
    rows = [
        row for row in state["test_rows"]
        if row["condition_id"] in set(state["positive_test_ids"])
    ]
    for row in sorted(rows, key=lambda current: current["condition_id"]):
        lines.append(
            f"{row['task']} | {row['checkpoint']} | {row['head_selection']} | "
            f"k={row['head_count']} | beta={row['beta']:g} | "
            f"primary={100.0 * primary_effect(row):+.3f} pp | "
            f"final={100.0 * float(row['delta_expr_target_correct_rate']):+.3f} pp | "
            f"roles={','.join(row.get('test_roles', []))}"
        )
    paths["summary"].write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    tasks = [TASK_BY_TRACK[track] for track in args.tracks]
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
        "unique_chainsearch_problems_per_condition": args.test_units,
        "positive_test_conditions": len(positive),
        "negative_test_controls": len(state["negative_control_of"]),
        "overall_status": "PASS",
    }


def render_outputs(
    state: dict[str, Any], paths: dict[str, Path], args: argparse.Namespace,
) -> None:
    metric_rows = list(state["head_metric_rows"])
    head_orders = build_head_orders(metric_rows, args)
    write_csv(paths["ranking_csv"], metric_rows)
    write_csv(paths["effects_csv"], state["test_rows"])
    relocate_existing_diagnostic_figures(paths)
    plot_head_ranking(metric_rows, head_orders, paths, args)
    plot_sparse_breadth(state, paths, args)
    plot_local_to_global(state, paths, args)
    plot_separation_vs_validation_effect(state, metric_rows, paths, args)

    paired_cache = build_paired_test_cache(state)
    write_csv(paths["counts_csv"], transition_count_rows(state, paired_cache))
    write_csv(paths["changed_csv"], changed_problem_rows(state, paired_cache))
    plot_sparse_breadth_counts(state, paired_cache, paths, args)
    plot_local_to_global_counts(state, paired_cache, paths, args)
    plot_directionality_counts(state, paired_cache, paths, args)
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
        for name in ("tracks", "checkpoints", "selection_rules", "head_counts"):
            if name in saved:
                setattr(args, name, tuple(saved[name]))
        for name in (
            "head_selection_units",
            "coarse_validation_units",
            "refined_validation_units",
            "test_units",
        ):
            if name in saved:
                setattr(args, name, int(saved[name]))
        render_outputs(payload, paths, args)
        probe.heading("ATTENTION-HEAD ABLATION OUTPUTS REGENERATED")
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
    probe.heading("ATTENTION-HEAD CAA ABLATION")
    probe.print_rows("RUN CONFIGURATION", [
        ("Tracks", ", ".join(args.tracks)),
        ("Checkpoints", ", ".join(args.checkpoints)),
        ("Selection rules", ", ".join(args.selection_rules)),
        ("Head counts", list(args.head_counts)),
        ("Positive betas", list(args.betas)),
        (
            "Head / coarse / refined / test problems",
            f"{args.head_selection_units} / {args.coarse_validation_units} / "
            f"{args.refined_validation_units} / {args.test_units}",
        ),
        ("Output directory", args.output_dir),
        ("Device", args.device),
    ])

    sources = load_sources(args.device)
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
    head_orders = build_head_orders(head_metric_rows, args)
    conditions = build_conditions(sources, head_orders, args)
    signature = make_signature(sources, conditions, args)
    state = load_or_initialize_state(paths, signature, args)
    paths["shards"].mkdir(parents=True, exist_ok=True)
    state["head_metric_rows"] = head_metric_rows
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

    probe.heading("ATTENTION-HEAD ABLATION COMPLETE")
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
