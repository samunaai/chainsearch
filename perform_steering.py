#!/usr/bin/env python3
"""
perform_steering.py

Run causal continuation steering with Probe, CAA, and Direct Optimisation
vectors. Interventions are applied at stored or dynamically detected semantic
checkpoints; the remaining suffix is generated greedily and scored with the
existing ChainSearch evaluator.
"""

from __future__ import annotations

import argparse
import copy
import functools
import gc
import hashlib
import math
import os
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["COUNTDOWN_STAGE"] = "astar"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from tqdm.auto import tqdm

import build_probe_dataset as builder
import compute_linear_probe_vectors as probe
import evaluate_unused_astar_samples_for_probing as evaluator
import label_generated_states_for_probing_samples as reachability

pm = builder.pm


# =============================================================================
# CONFIGURATION
# =============================================================================

SOURCE_RUN_TAG = probe.RUN_TAG
STEERING_RUN_TAG = "exp34"
SEED = 42
ASTAR_DIR = probe.ASTAR_DIR
EVALUATION_DIR = os.path.join(ASTAR_DIR, "steering_evaluations")
STEERING_EXAMPLE_DIR = "/workspace/sam/minireason_archives/steering_examples"
CAA_PATH = os.path.join(ASTAR_DIR, "caa", f"{SOURCE_RUN_TAG}_astar_caa_vectors.pt")
DIRECT_PATH = os.path.join(ASTAR_DIR, "direct_optimisation", f"{SOURCE_RUN_TAG}_astar_direct_optimisation_vectors.pt")
 

def output_paths(smoke: bool, experiment: str = "residual-pilot") -> tuple[str, str, str]:
    suffix = "_smoke" if smoke else ""
    experiment_suffix = "" if experiment == "residual-pilot" else f"_{experiment.replace('-', '_')}"
    stem = f"{STEERING_RUN_TAG}_steering_results{experiment_suffix}{suffix}"
    output = os.path.join(EVALUATION_DIR, f"{stem}.pt")
    partial = os.path.abspath(f"{stem}.partial.pt")
    shards = os.path.join(EVALUATION_DIR, f"{stem}_generation_shards")
    return output, partial, shards

def qualitative_output_path(smoke: bool, experiment: str = "residual-pilot") -> str:
    suffix = "_smoke" if smoke else ""
    experiment_suffix = "" if experiment == "residual-pilot" else f"_{experiment.replace('-', '_')}"
    return os.path.join(STEERING_EXAMPLE_DIR, f"{STEERING_RUN_TAG}_steering_examples{experiment_suffix}{suffix}.txt")

def suite_output_path(smoke: bool) -> str:
    suffix = "_smoke" if smoke else ""
    return os.path.join(EVALUATION_DIR, f"{STEERING_RUN_TAG}_steering_suite{suffix}.pt")

OUTPUT_PATH, PARTIAL_PATH, SHARD_DIR = output_paths(False)
QUALITATIVE_PATH = qualitative_output_path(False)

FINAL_TASKS = {
    "trace_outcome_operation_decisions",
    "trace_outcome_trajectory",
    "trace_outcome_back_decisions",
    "terminal_outcome_controls",
}
REACHABILITY_TASKS = {"operation_reachability", "back_reachability"}
METHOD_ORDER = ("probe", "caa", "optimised")
REPRESENTATION_ORDER = ("attention_head", "residual_stream", "mlp_output", "final_hidden")
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
CAUSAL_CHECKPOINTS = {
    "pre_first_operand", "first_operand_end", "operator", "second_operand_end",
    "result_end", "resulting_state_semicolon", "pre_back", "back_token",
    "restored_state_semicolon", "success_token",
}
BALANCED_CHECKPOINTS = {
    "pre_first_operand", "resulting_state_semicolon", "pre_back",
    "back_token", "restored_state_semicolon", "success_token",
}
BALANCED_REPRESENTATIONS = {"residual_stream"}
PRIMARY_CHECKPOINTS = {
    "pre_first_operand", "resulting_state_semicolon", "pre_back", "restored_state_semicolon",
}
LOCAL_TO_GLOBAL_CHECKPOINTS = (
    "pre_first_operand", "first_operand_end", "operator", "second_operand_end", "result_end",
)
EXPERIMENTS = ("residual-pilot", "steering-atlas", "headcount", "staying-power", "local-to-global")
SUITE_EXPERIMENTS = ("steering-atlas", "headcount", "staying-power", "local-to-global")
TEST_NAMES = {
    "residual-pilot": "Residual Pilot",
    "steering-atlas": "Where and When Test",
    "headcount": "How Many Heads Test",
    "staying-power": "Intervention Frequency Test",
    "local-to-global": "Attention-Head Local-to-Global Test",
}
ATLAS_TOP_HEADS = 4
EXPECTED_VECTOR_COUNTS = {"probe": probe.EXPECTED_PROBES, "caa": probe.EXPECTED_PROBES, "optimised": 300}
EXPECTED_DIRECT_OPTIMISATIONS = 270
ZERO_EPS = 1e-12
UNIT_NORM_TOL = 1e-4

DEFAULT_BETAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
DEFAULTS = {
    "coarse_validation_units": 16,
    "refined_validation_units": 96,
    "test_units": 192,
    "batch_size": 32,
    "refine_k": 2,
    "top_k": 1,
    "bootstrap_samples": 1000,
    "qualitative_examples_per_condition": 2,
    "save_every": 25,
}
SMOKE_DEFAULTS = {
    "coarse_validation_units": 2,
    "refined_validation_units": 2,
    "test_units": 2,
    "batch_size": 2,
    "refine_k": 1,
    "top_k": 1,
    "bootstrap_samples": 50,
    "qualitative_examples_per_condition": 1,
    "save_every": 1,
}


# =============================================================================
# DATA CONTAINERS
# =============================================================================

@dataclass(frozen=True)
class SteeringVector:
    method: str
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
    shared_optimisation_id: str | None
    validation_auroc: float


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


@dataclass(frozen=True)
class ScheduledCondition(SteeringCondition):
    schedule: str = "single"
    event_members: tuple[tuple[str, str, float], ...] = ()
    source_validation_rows: tuple[str, ...] = ()


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in (SEED, *parts))
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big") % (2**63 - 1)


def vector_key(task: str, checkpoint: str, location_name: str) -> str:
    return f"{task}:{checkpoint}:{location_name}"


def parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    unknown = set(methods) - set(METHOD_ORDER)
    if not methods or unknown:
        raise argparse.ArgumentTypeError(f"Methods must be comma-separated members of {METHOD_ORDER}; got {sorted(unknown)}.")
    return methods


def condition_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def realpath(path: str) -> str:
    return os.path.realpath(os.path.expanduser(path))


def resolve(value: int | None, name: str, smoke: bool) -> int:
    return int(value if value is not None else (SMOKE_DEFAULTS if smoke else DEFAULTS)[name])


def pack_sequences(sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.int32)
    offsets = torch.zeros(len(sequences), dtype=torch.int64)
    if lengths.numel() > 1:
        offsets[1:] = torch.cumsum(lengths[:-1].to(torch.int64), dim=0)
    flat = torch.tensor([token for sequence in sequences for token in sequence], dtype=torch.int16)
    return flat, offsets, lengths


def unpack_sequence(flat: torch.Tensor, offsets: torch.Tensor, lengths: torch.Tensor, index: int) -> list[int]:
    start, length = int(offsets[index]), int(lengths[index])
    return flat[start:start + length].to(torch.int64).tolist()


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> float:
    values, weights = values.to(torch.float64), weights.to(torch.float64)
    return float((values * weights).sum() / weights.sum().clamp_min(1e-12))


def weighted_count_rate(numerator: torch.Tensor, denominator: torch.Tensor, weights: torch.Tensor) -> float:
    numerator, denominator, weights = numerator.to(torch.float64), denominator.to(torch.float64), weights.to(torch.float64)
    total = (denominator * weights).sum()
    return float((numerator * weights).sum() / total) if float(total) > 0 else 0.0


def file_signature(path: str) -> dict[str, Any]:
    resolved = realpath(path)
    stat = os.stat(resolved)
    return {"path": resolved, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def atomic_save(payload: dict[str, Any], path: str, attempts: int = 3) -> None:
    for attempt in range(1, attempts + 1):
        try:
            probe.safe_torch_save(payload, path)
            return
        except OSError:
            if attempt == attempts:
                raise
            time.sleep(2 ** (attempt - 1))


# =============================================================================
# SOURCE LOADING
# =============================================================================

def load_archive(path: str, name: str, expected_run_tag: str = SOURCE_RUN_TAG) -> dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    archive = torch.load(path, map_location="cpu", weights_only=False)
    if not bool(archive.get("complete", False)):
        raise RuntimeError(f"{name} archive is incomplete: {path}")
    if archive.get("run_tag") != expected_run_tag:
        raise RuntimeError(f"{name} run tag {archive.get('run_tag')!r} != {expected_run_tag!r}.")
    return archive


def probe_validation_aurocs(archive: dict[str, Any]) -> dict[str, float]:
    rows = archive.get("metrics")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("Probe archive contains no metrics rows for validation-only location ranking.")
    output: dict[str, float] = {}
    for row in rows:
        key = vector_key(str(row["task"]), str(row["checkpoint"]), str(row["location_name"]))
        score = float(row["val_auroc"])
        if not math.isfinite(score):
            raise RuntimeError(f"Non-finite validation AUROC for {key}.")
        if key in output:
            raise RuntimeError(f"Duplicate validation AUROC row for {key}.")
        output[key] = score
    return output


def canonicalise_vectors(
    method: str,
    archive: dict[str, Any],
    validation_aurocs: dict[str, float],
) -> dict[str, SteeringVector]:
    source = archive.get("probe_bank") if method == "probe" else archive.get("vector_bank")
    if not isinstance(source, dict) or not source:
        raise RuntimeError(f"{method} archive contains no vector bank.")
    output: dict[str, SteeringVector] = {}
    for entry in source.values():
        task = str(entry["task"])
        if task not in FINAL_TASKS | REACHABILITY_TASKS:
            raise KeyError(f"Unknown task {task!r} in {method} archive.")
        checkpoint = str(entry["checkpoint"])
        location = str(entry["location_name"])
        key = vector_key(task, checkpoint, location)
        if key in output:
            raise RuntimeError(f"Duplicate canonical vector key in {method}: {key}")
        raw_name = "weight_raw_space" if method == "probe" else "vector_raw_space"
        raw = entry[raw_name].detach().cpu().float().reshape(-1)
        direction = entry["direction_raw_space_normalized"].detach().cpu().float().reshape(-1)
        raw_norm, direction_norm = float(raw.norm()), float(direction.norm())
        zero = bool(entry.get("zero_norm", False)) or raw_norm <= ZERO_EPS or direction_norm <= ZERO_EPS
        if not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(direction).all()):
            raise RuntimeError(f"Non-finite vector values for {method}/{key}.")
        output[key] = SteeringVector(
            method=method,
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
            shared_optimisation_id=entry.get("shared_optimisation_id"),
            validation_auroc=validation_aurocs[key],
        )
    return output


def audit_vector_banks(banks: dict[str, dict[str, SteeringVector]]) -> dict[str, Any]:
    for method, expected in EXPECTED_VECTOR_COUNTS.items():
        if len(banks[method]) != expected:
            raise RuntimeError(f"Expected {expected} {method} vectors, found {len(banks[method])}.")
        for key, vector in banks[method].items():
            if vector.direction.numel() != vector.feature_dim or vector.raw_vector.numel() != vector.feature_dim:
                raise RuntimeError(f"Wrong vector dimension for {method}/{key}.")
            if not vector.zero_norm and abs(float(vector.direction.norm()) - 1.0) > UNIT_NORM_TOL:
                raise RuntimeError(f"Direction is not unit-normalised for {method}/{key}.")
    probe_keys, caa_keys, direct_keys = set(banks["probe"]), set(banks["caa"]), set(banks["optimised"])
    if probe_keys != caa_keys:
        raise RuntimeError(f"Probe and CAA key sets differ: probe-only={len(probe_keys-caa_keys)}, CAA-only={len(caa_keys-probe_keys)}.")
    if not direct_keys <= probe_keys:
        raise RuntimeError(f"Direct keys missing from Probe/CAA banks: {len(direct_keys-probe_keys)}.")
    for key in probe_keys:
        reference = banks["probe"][key]
        records = [banks["caa"][key]] + ([banks["optimised"][key]] if key in direct_keys else [])
        for record in records:
            for field in ("task", "checkpoint", "representation_type", "location_name", "layer", "head", "feature_dim"):
                if getattr(record, field) != getattr(reference, field):
                    raise RuntimeError(f"Vector metadata mismatch for {key} field {field!r}.")
    if any(vector.label_family != "final_answer" for vector in banks["optimised"].values()):
        raise RuntimeError("Direct Optimisation contains non-final-answer vectors.")
    direct_ids = {vector.shared_optimisation_id for vector in banks["optimised"].values() if vector.shared_optimisation_id is not None}
    if len(direct_ids) != EXPECTED_DIRECT_OPTIMISATIONS:
        raise RuntimeError(f"Expected {EXPECTED_DIRECT_OPTIMISATIONS} unique Direct optimisations, found {len(direct_ids)}.")
    return {
        "vectors_by_method": {method: len(bank) for method, bank in banks.items()},
        "zero_vectors_by_method": {method: sum(vector.zero_norm for vector in bank.values()) for method, bank in banks.items()},
        "direct_unique_optimisations": len(direct_ids),
        "probe_caa_keys_identical": True,
        "direct_keys_subset": True,
    }


def load_model(data: dict[str, Any], tok: Any, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any], int]:
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
    tok = pm.Tokenizer()
    model, model_config, model_max_len = load_model(data, tok, device)
    paths = {
        "probe": probe.RESULTS_PATH,
        "caa": CAA_PATH,
        "optimised": DIRECT_PATH,
    }
    archives = {method: load_archive(paths[method], method) for method in METHOD_ORDER}
    for method, archive in archives.items():
        source_dataset = archive.get("source_probe_dataset_path")
        if source_dataset is None or realpath(str(source_dataset)) != realpath(probe.DATASET_PATH):
            raise RuntimeError(f"{method} archive points to a different probe dataset.")
    validation_aurocs = probe_validation_aurocs(archives["probe"])
    banks = {
        method: canonicalise_vectors(method, archives[method], validation_aurocs)
        for method in METHOD_ORDER
    }
    vector_audit = audit_vector_banks(banks)
    checkpoint_paths = {realpath(str(data["source_checkpoint_path"]))}
    for method in ("caa", "optimised"):
        source_checkpoint = archives[method].get("source_checkpoint_path")
        if source_checkpoint is None:
            raise RuntimeError(f"{method} archive lacks source_checkpoint_path.")
        checkpoint_paths.add(realpath(str(source_checkpoint)))
    if len(checkpoint_paths) != 1:
        raise RuntimeError(f"Sources refer to different model checkpoints: {sorted(checkpoint_paths)}")
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
        "vector_audit": vector_audit,
        "trace_outcome_lookup": probe.build_trace_outcome_lookup(data),
        "unit_tables": {},
    }


# =============================================================================
# CANDIDATE VECTORS, PREFIXES, AND CONDITIONS
# =============================================================================

def candidate_vectors(
    banks: dict[str, dict[str, SteeringVector]],
    methods: tuple[str, ...],
    include_reachability: bool,
    include_pairwise_extra: bool,
    checkpoints: set[str],
    representations: set[str],
    smoke: bool,
) -> list[SteeringVector]:
    final_common = set.intersection(*(set(banks[method]) for method in methods))
    final_common = {
        key for key in final_common
        if banks[methods[0]][key].label_family == "final_answer"
        and banks[methods[0]][key].checkpoint in checkpoints
        and banks[methods[0]][key].representation_type in representations
        and all(not banks[method][key].zero_norm for method in methods)
    }
    selected: dict[tuple[str, str], SteeringVector] = {}
    pairwise_extra = {
        key for key in set(banks["probe"]) & set(banks["caa"])
        if banks["probe"][key].label_family == "final_answer"
        and banks["probe"][key].checkpoint in checkpoints
        and banks["probe"][key].representation_type in representations
        and not banks["probe"][key].zero_norm
        and not banks["caa"][key].zero_norm
    }
    for method in methods:
        keys = pairwise_extra if include_pairwise_extra and method in {"probe", "caa"} else final_common
        for key in keys:
            vector = banks[method][key]
            if not vector.zero_norm:
                selected[(method, key)] = vector

    reach_methods = tuple(method for method in methods if method in {"probe", "caa"})
    if include_reachability and reach_methods:
        reach_common = set.intersection(*(set(banks[method]) for method in reach_methods))
        reach_common = {
            key for key in reach_common
            if banks[reach_methods[0]][key].label_family == "reachability"
            and banks[reach_methods[0]][key].checkpoint in checkpoints
            and banks[reach_methods[0]][key].representation_type in representations
            and all(not banks[method][key].zero_norm for method in reach_methods)
        }
        for method in reach_methods:
            for key in reach_common:
                selected[(method, key)] = banks[method][key]

    output = list(selected.values())
    if smoke:
        allowed = {"operator", "success_token"}
        output = [vector for vector in output if vector.checkpoint in allowed]
        first: dict[tuple[str, str, str, str], SteeringVector] = {}
        for vector in sorted(output, key=lambda value: (value.task, value.location_name)):
            first.setdefault(
                (vector.method, vector.task, vector.checkpoint, vector.representation_type),
                vector,
            )
        output = list(first.values())
    if not output:
        raise RuntimeError("No steering vectors remain after filtering.")
    return sorted(output, key=lambda value: (value.method, value.task, value.checkpoint, value.location_name))


def methods_for_label(methods: tuple[str, ...], label_family: str) -> tuple[str, ...]:
    if label_family == "reachability":
        return tuple(method for method in methods if method in {"probe", "caa"})
    return methods


def validation_ranked_head_names(
    vectors: list[SteeringVector],
    methods: tuple[str, ...],
    task: str,
    checkpoint: str,
    limit: int,
) -> list[str]:
    rows = [
        vector for vector in vectors
        if vector.task == task and vector.checkpoint == checkpoint and vector.representation_type == "attention_head"
    ]
    if not rows:
        raise RuntimeError(f"No attention-head vectors for {task}/{checkpoint}.")
    label_family = rows[0].label_family
    required_methods = methods_for_label(methods, label_family)
    names_by_method = {
        method: {vector.location_name for vector in rows if vector.method == method and not vector.zero_norm}
        for method in required_methods
    }
    common = set.intersection(*(names_by_method[method] for method in required_methods))
    ranked = sorted(
        common,
        key=lambda name: (-next(vector.validation_auroc for vector in rows if vector.location_name == name), name),
    )
    if limit <= 0:
        return ranked
    if len(ranked) < limit:
        raise RuntimeError(f"Only {len(ranked)} common heads are available for {task}/{checkpoint}; need {limit}.")
    return ranked[:limit]


def atlas_vectors(vectors: list[SteeringVector]) -> list[SteeringVector]:
    selected_heads: dict[tuple[str, str, str], set[str]] = {}
    cells = sorted({(vector.task, vector.checkpoint, vector.method) for vector in vectors})
    for task, checkpoint, method in cells:
        rows = [
            vector for vector in vectors
            if vector.task == task and vector.checkpoint == checkpoint and vector.method == method
            and vector.representation_type == "attention_head"
        ]
        ranked = sorted(rows, key=lambda vector: (-vector.validation_auroc, vector.location_name))
        if len(ranked) < ATLAS_TOP_HEADS:
            raise RuntimeError(f"Only {len(ranked)} heads are available for {method}/{task}/{checkpoint}.")
        selected_heads[(task, checkpoint, method)] = {vector.location_name for vector in ranked[:ATLAS_TOP_HEADS]}

    output: list[SteeringVector] = []
    for vector in vectors:
        if vector.representation_type == "attention_head":
            if vector.location_name not in selected_heads[(vector.task, vector.checkpoint, vector.method)]:
                continue
        elif vector.method == "optimised" and vector.representation_type == "mlp_output":
            # Direct residual/MLP entries share an optimisation and add the same
            # block-output displacement, so count the residual intervention once.
            continue
        output.append(vector)
    if not output:
        raise RuntimeError("The Where and When Test representative panel is empty.")
    return output


def local_to_global_vectors(
    vectors: list[SteeringVector],
    methods: tuple[str, ...],
) -> list[SteeringVector]:
    reachability_methods = methods_for_label(methods, "reachability")
    if not reachability_methods:
        raise RuntimeError("The local-to-global experiment requires Probe or CAA.")

    filtered = [
        vector for vector in vectors
        if vector.task == "operation_reachability"
        and vector.label_family == "reachability"
        and vector.representation_type == "attention_head"
    ]
    available_cells = {
        (vector.method, vector.checkpoint)
        for vector in filtered
    }
    missing_cells = [
        (method, checkpoint)
        for method in reachability_methods
        for checkpoint in LOCAL_TO_GLOBAL_CHECKPOINTS
        if (method, checkpoint) not in available_cells
    ]
    if missing_cells:
        formatted = ", ".join(f"{method}/{checkpoint}" for method, checkpoint in missing_cells)
        raise RuntimeError(
            "The local-to-global experiment lacks operation-reachability "
            f"attention-head vectors for: {formatted}."
        )
    return atlas_vectors(filtered)


def select_rows(rows: torch.Tensor, cap: int, offset: int, seed: int) -> torch.Tensor:
    rows = rows.to(torch.int64)
    if offset < 0 or cap <= 0:
        raise ValueError(f"Invalid row selection offset={offset}, cap={cap}.")
    required = offset + cap
    if rows.numel() < required:
        raise RuntimeError(
            f"Requested {cap} rows after offset {offset}, "
            f"but only {rows.numel()} rows are available."
        )
    generator = torch.Generator().manual_seed(seed) 
    shuffled = rows[torch.randperm(rows.numel(), generator=generator)]
    selected = shuffled[offset:required]
    if selected.numel() != cap:
        raise RuntimeError(f"Expected {cap} selected rows, found {selected.numel()}.")    
    return selected


def trace_weights(record_pos: torch.Tensor, trace_normalized: bool) -> torch.Tensor:
    if not trace_normalized:
        return torch.ones(record_pos.numel(), dtype=torch.float32)
    _, inverse, counts = torch.unique(record_pos.to(torch.int64), return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse].to(torch.float32)


def build_prefix_units(
    sources: dict[str, Any],
    task: str,
    checkpoint: str,
    split: str,
    cap: int,
    offset: int = 0,
    unique_trace: bool = False,
) -> list[PrefixUnit]:
    data, evaluated, cache, tok = sources["data"], sources["evaluated"], sources["token_cache"], sources["tok"]
    trace_label, trace_known = sources["trace_outcome_lookup"]
    if task not in sources["unit_tables"]:
        sources["unit_tables"][task] = probe.build_unit_table(data, task, trace_label, trace_known)
    units = sources["unit_tables"][task]
    checkpoint_index = units.checkpoints.index(checkpoint)
    rows = probe.split_unit_rows(units, {str(k): int(v) for k, v in data["split_to_id"].items()})[split]
    if unique_trace:
        candidate_activation = units.activation_index[rows, checkpoint_index].to(torch.int64)
        candidate_record = units.record_pos[rows].to(torch.int64)
        candidate_position = data["activation_positions"]["generated_token_idx"][candidate_activation].to(torch.int64)
        order = torch.argsort(candidate_position, stable=True)
        order = order[torch.argsort(candidate_record[order], stable=True)]
        ordered_record = candidate_record[order]
        first = torch.ones(order.numel(), dtype=torch.bool)
        first[1:] = ordered_record[1:] != ordered_record[:-1]
        rows = rows[order[first]]
    rows = select_rows(rows, cap, offset, stable_seed(task, split, "unit_order"))
    record_pos = units.record_pos[rows].to(torch.int64)
    activation_index = units.activation_index[rows, checkpoint_index].to(torch.int64)

    positions = data["activation_positions"]
    n_positions = int(positions["generated_token_idx"].numel())
    if bool((activation_index < 0).any()) or bool((activation_index >= n_positions).any()):
        raise RuntimeError(f"Invalid activation index for {task}/{checkpoint}.")
    if not torch.equal(
        positions["record_pos"][activation_index].to(torch.int64),
        record_pos,
    ):
        raise RuntimeError(f"Activation-position record mismatch for {task}/{checkpoint}.")
    generated_idx = positions["generated_token_idx"][activation_index].to(torch.int64)

    weights = trace_weights(record_pos, probe.TASK_SPECS[task].trace_normalized)
    cache_parts: dict[int, tuple[list[int], list[int], str, str, int]] = {}
    output: list[PrefixUnit] = []
    for local, row in enumerate(rows.tolist()):
        record = int(record_pos[local])
        cache_idx = int(units.cache_idx[row])
        if cache_idx not in cache_parts:
            prompt_ids, gold_ids = evaluator.cache_row_parts(cache, cache_idx, tok)
            prompt_text = tok.decode(prompt_ids[:-1]).strip()
            gold_text = tok.decode(gold_ids).strip()
            gold_back = int(cache["bk_counts"][cache_idx])
            cache_parts[cache_idx] = (prompt_ids, gold_ids, prompt_text, gold_text, gold_back)
        prompt_ids, _, prompt_text, gold_text, gold_back = cache_parts[cache_idx]
        original = builder.generated_ids_at(evaluated, record)
        position = int(generated_idx[local])
        if not 0 <= position < len(original):
            raise RuntimeError(f"Invalid generated checkpoint index {position} for record {record}.")
        output.append(PrefixUnit(
            task=task,
            source_row=int(units.source_row[row]),
            record_pos=record,
            cache_idx=cache_idx,
            target=int(units.target[row]),
            generated_idx=position,
            prompt_ids=tuple(prompt_ids),
            generated_prefix=tuple(original[:position + 1]),
            original_suffix=tuple(original[position + 1:]),
            prompt_text=prompt_text,
            gold_text=gold_text,
            gold_back_count=gold_back,
            weight=float(weights[local]),
            initial_state=tuple(sorted(int(value) for value in evaluated["input_numbers"][record].tolist())),
        ))
    if not output:
        raise RuntimeError(f"No {split} prefix units for {task}/{checkpoint}.")
    return output


def build_conditions(
    vectors: list[SteeringVector],
    banks: dict[str, dict[str, SteeringVector]],
    betas: tuple[float, ...],
    scale_mode: str,
    include_native_direct: bool,
    max_per_group: int,
) -> list[SteeringCondition]:
    grouped: dict[tuple[str, str, str, str], list[SteeringVector]] = defaultdict(list)
    for vector in vectors:
        grouped[(vector.method, vector.task, vector.checkpoint, vector.representation_type)].append(vector)
    selected_vectors: list[SteeringVector] = []
    for key in sorted(grouped):
        values = sorted(grouped[key], key=lambda value: value.location_name)
        selected_vectors.extend(values[:max_per_group] if max_per_group > 0 else values)
    conditions: list[SteeringCondition] = []
    for vector in selected_vectors:
        caa = banks.get("caa", {}).get(vector.key)
        if scale_mode == "caa_norm":
            if caa is None or caa.zero_norm:
                raise RuntimeError(f"CAA scale is unavailable for {vector.key}.")
            scale = caa.raw_norm
        else:
            scale = 1.0
        for beta in betas:
            alpha = float(beta) * scale
            cid = f"{vector.method}:{vector.key}:beta={repr(float(beta))}:scale={scale_mode}"
            conditions.append(SteeringCondition(
                condition_id=cid,
                method=vector.method,
                vector_key=vector.key,
                task=vector.task,
                label_family=vector.label_family,
                checkpoint=vector.checkpoint,
                representation_type=vector.representation_type,
                location_name=vector.location_name,
                layer=vector.layer,
                head=vector.head,
                beta=float(beta),
                scale=float(scale),
                alpha=alpha,
                native_direct=False,
                shared_optimisation_id=vector.shared_optimisation_id,
                member_keys=(vector.key,),
                member_scales=(float(scale),),
            ))
        if include_native_direct and vector.method == "optimised":
            cid = f"optimised:{vector.key}:native"
            conditions.append(SteeringCondition(
                condition_id=cid,
                method=vector.method,
                vector_key=vector.key,
                task=vector.task,
                label_family=vector.label_family,
                checkpoint=vector.checkpoint,
                representation_type=vector.representation_type,
                location_name=vector.location_name,
                layer=vector.layer,
                head=vector.head,
                beta=float("nan"),
                scale=vector.raw_norm,
                alpha=1.0,
                native_direct=True,
                shared_optimisation_id=vector.shared_optimisation_id,
                member_keys=(vector.key,),
                member_scales=(vector.raw_norm,),
            ))
    return conditions


def build_headcount_conditions(
    vectors: list[SteeringVector],
    banks: dict[str, dict[str, SteeringVector]],
    methods: tuple[str, ...],
    betas: tuple[float, ...],
) -> list[SteeringCondition]:
    vector_by_method_key = {(vector.method, vector.key): vector for vector in vectors}
    conditions: list[SteeringCondition] = []
    cells = sorted({(vector.task, vector.checkpoint, vector.label_family) for vector in vectors})

    for task, checkpoint, label_family in cells:
        required_methods = methods_for_label(methods, label_family)
        ranked_names = validation_ranked_head_names(vectors, methods, task, checkpoint, 0)
        if len(ranked_names) < 12:
            raise RuntimeError(f"Only {len(ranked_names)} common heads are available for {task}/{checkpoint}.")
        random_names = ranked_names.copy()
        random.Random(stable_seed("random_heads", task, checkpoint)).shuffle(random_names)
        tested_counts = (4, 12, min(24, len(ranked_names)))
        variants = [(f"top{k}_fixed", ranked_names[:k], "top", "fixed_energy") for k in tested_counts]
        variants.extend([
            ("random12_fixed", random_names[:12], "random", "fixed_energy"),
            ("top12_per_head", ranked_names[:12], "top", "per_head"),
        ])

        for method in required_methods:
            for variant, names, selection, scaling in variants:
                member_keys = tuple(vector_key(task, checkpoint, name) for name in names)
                members = [vector_by_method_key[(method, key)] for key in member_keys]
                caa_scales = tuple(float(banks["caa"][key].raw_norm) for key in member_keys)
                divisor = math.sqrt(len(members)) if scaling == "fixed_energy" else 1.0
                member_scales = tuple(scale / divisor for scale in caa_scales)
                aggregate_key = f"{task}:{checkpoint}:{variant}"
                location_name = "+".join(names)

                for beta in betas:
                    conditions.append(SteeringCondition(
                        condition_id=f"{method}:{aggregate_key}:beta={repr(float(beta))}:scale=caa_norm",
                        method=method,
                        vector_key=aggregate_key,
                        task=task,
                        label_family=label_family,
                        checkpoint=checkpoint,
                        representation_type="attention_head",
                        location_name=location_name,
                        layer=-1,
                        head=-1,
                        beta=float(beta),
                        scale=1.0,
                        alpha=float(beta),
                        native_direct=False,
                        shared_optimisation_id=None,
                        member_keys=member_keys,
                        member_scales=member_scales,
                        selection_variant=variant,
                        head_count=len(members),
                        head_selection=selection,
                        head_scaling=scaling,
                    ))
    if not conditions:
        raise RuntimeError("No How Many Heads Test conditions were constructed.")
    return conditions


def primary_effect(row: dict[str, Any]) -> float:
    if row["label_family"] != "reachability":
        return float(row["delta_expr_target_correct_rate"])
    scheduled = row["representation_type"] == "scheduled"
    field = "delta_future_reachable_state_fraction_mean" if scheduled else "delta_next_state_viable_rate"
    return float(row[field])


def select_headcount_anchors(atlas: dict[str, Any], methods: tuple[str, ...]) -> dict[str, tuple[str, str]]:
    selected_ids = set(atlas["refined_condition_ids"])
    rows = [
        row for row in atlas["refined_validation_rows"]
        if row["condition_id"] in selected_ids and row["representation_type"] == "attention_head"
        and not row["native_direct"] and float(row["beta"]) > 0
    ]
    anchors: dict[str, tuple[str, str]] = {}
    for label_family in ("reachability", "final_answer"):
        required_methods = methods_for_label(methods, label_family)
        cells = sorted({(row["task"], row["checkpoint"]) for row in rows if row["label_family"] == label_family})
        cell_scores: list[tuple[float, str, str]] = []
        for task, checkpoint in cells:
            method_scores = []
            for method in required_methods:
                candidates = [
                    row for row in rows
                    if row["method"] == method and row["task"] == task and row["checkpoint"] == checkpoint
                ]
                if not candidates:
                    break
                method_scores.append(max(primary_effect(row) for row in candidates))
            if len(method_scores) == len(required_methods):
                cell_scores.append((sum(method_scores) / len(method_scores), task, checkpoint))
        if not cell_scores:
            raise RuntimeError(f"The Where and When Test has no common attention-head anchor for {label_family}.")
        _, task, checkpoint = max(cell_scores)
        anchors[label_family] = (task, checkpoint)
    return anchors


STAYING_CELLS = {
    "reachability": (
        ("pre_first_operand", "operation_reachability", "pre_first_operand"),
        ("resulting_state_semicolon", "operation_reachability", "resulting_state_semicolon"),
        ("restored_state_semicolon", "back_reachability", "restored_state_semicolon"),
    ),
    "final_answer": (
        ("pre_first_operand", "trace_outcome_operation_decisions", "pre_first_operand"),
        ("resulting_state_semicolon", "trace_outcome_operation_decisions", "resulting_state_semicolon"),
        ("restored_state_semicolon", "trace_outcome_back_decisions", "restored_state_semicolon"),
    ),
}


def row_rank(row: dict[str, Any]) -> tuple[float, float, float, float, str]:
    if row["label_family"] == "reachability":
        return (
            primary_effect(row),
            float(row["delta_next_state_observed_rate"]),
            float(row["delta_expr_target_correct_rate"]),
            -abs(float(row["beta"])),
            str(row["location_name"]),
        )
    return (
        float(row["delta_expr_target_correct_rate"]),
        float(row["delta_solved_valid_trace_rate"]),
        float(row["delta_valid_trace_rate"]),
        -abs(float(row["beta"])),
        str(row["location_name"]),
    )


def is_local_to_global_candidate(row: dict[str, Any]) -> bool:
    return (
        row["task"] == "operation_reachability"
        and row["label_family"] == "reachability"
        and row["representation_type"] == "attention_head"
        and row["checkpoint"] in LOCAL_TO_GLOBAL_CHECKPOINTS
    )


def build_staying_conditions(atlas: dict[str, Any], methods: tuple[str, ...]) -> list[ScheduledCondition]:
    if not bool(atlas.get("complete", False)) or atlas.get("run_tag") != STEERING_RUN_TAG:
        raise RuntimeError("The Intervention Frequency Test requires a complete matching Where and When Test artifact.")
    selected_ids = set(atlas["validation_selected_condition_ids"])
    rows = [
        row for row in atlas["refined_validation_rows"]
        if row["condition_id"] in selected_ids and not row["native_direct"] and float(row["beta"]) > 0
    ]
    conditions: list[ScheduledCondition] = []
    for label_family, cells in STAYING_CELLS.items():
        for method in methods_for_label(methods, label_family):
            chosen: dict[str, dict[str, Any]] = {}
            for event, task, checkpoint in cells:
                candidates = [
                    row for row in rows
                    if row["method"] == method and row["task"] == task and row["checkpoint"] == checkpoint
                ]
                if not candidates:
                    raise RuntimeError(f"The Where and When Test selected no positive {method} condition for {task}/{checkpoint}.")
                chosen[event] = max(candidates, key=row_rank)

            initial = chosen["pre_first_operand"]
            members_by_schedule = {
                "single": (("pre_first_operand", initial),),
                "repeated": (("pre_first_operand", initial),),
                "multipoint": tuple((event, chosen[event]) for event, _, _ in cells),
            }
            for schedule, members in members_by_schedule.items():
                event_members = tuple(
                    (event, str(row["vector_key"]), float(row["beta"]) * float(row["scale"]))
                    for event, row in members
                )
                locations = "+".join(f"{event}={row['location_name']}" for event, row in members)
                source_rows = tuple(str(row["condition_id"]) for _, row in members)
                for polarity in (1.0, -1.0):
                    sign = "positive" if polarity > 0 else "negative"
                    conditions.append(ScheduledCondition(
                        condition_id=f"staying:{method}:{label_family}:{schedule}:{sign}",
                        method=method,
                        vector_key=f"{label_family}:{schedule}",
                        task=str(initial["task"]),
                        label_family=label_family,
                        checkpoint="pre_first_operand",
                        representation_type="scheduled",
                        location_name=locations,
                        layer=-1,
                        head=-1,
                        beta=polarity,
                        scale=1.0,
                        alpha=polarity,
                        native_direct=False,
                        shared_optimisation_id=None,
                        member_keys=tuple(key for _, key, _ in event_members),
                        member_scales=tuple(scale for _, _, scale in event_members),
                        selection_variant=schedule,
                        head_count=0,
                        head_selection="atlas",
                        head_scaling="checkpoint_matched",
                        schedule=schedule,
                        event_members=event_members,
                        source_validation_rows=source_rows,
                    ))
    return conditions


def add_matched_negative_controls(
    selected_ids: list[str],
    condition_lookup: dict[str, SteeringCondition],
) -> tuple[list[str], dict[str, str]]:
    by_vector_and_beta = {
        (condition.method, condition.vector_key, condition.beta): condition.condition_id
        for condition in condition_lookup.values()
        if not condition.native_direct
    }
    output = list(selected_ids)
    negative_control_of: dict[str, str] = {}
    for condition_id in selected_ids:
        condition = condition_lookup[condition_id]
        if condition.native_direct or not math.isfinite(condition.beta) or condition.beta <= 0:
            continue
        negative_id = by_vector_and_beta.get((condition.method, condition.vector_key, -condition.beta))
        if negative_id is None:
            raise RuntimeError(f"Missing matched negative control for {condition_id}.")
        output.append(negative_id)
        negative_control_of[negative_id] = condition_id
    return list(dict.fromkeys(output)), negative_control_of


# =============================================================================
# INTERVENTION, GENERATION, AND SCORING
# =============================================================================

class TemporaryMethodPatch:
    def __init__(self, module: torch.nn.Module, name: str, replacement: Any):
        self.module, self.name = module, name
        self.had_instance_value = name in module.__dict__
        self.instance_value = module.__dict__.get(name)
        setattr(module, name, replacement)

    def remove(self) -> None:
        if self.had_instance_value:
            setattr(self.module, self.name, self.instance_value)
        elif self.name in self.module.__dict__:
            delattr(self.module, self.name)


class HookGroup:
    def __init__(self, handles: list[Any]):
        self.handles = handles

    def remove(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()


def broadcast_delta(delta: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    if delta.ndim == 1:
        return delta.to(hidden.dtype).view(1, 1, -1)
    if delta.ndim == 2 and delta.shape[0] == hidden.shape[0]:
        return delta.to(hidden.dtype).unsqueeze(1)
    raise RuntimeError(f"Cannot broadcast delta shape {tuple(delta.shape)} over hidden shape {tuple(hidden.shape)}.")


def install_hook(model: torch.nn.Module, vector: SteeringVector, delta: torch.Tensor):
    layer, head = vector.layer, vector.head
    if vector.representation_type == "attention_head":
        block = model.blocks[layer]
        start, end = head * int(block.head_dim), (head + 1) * int(block.head_dim)

        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            hidden = inputs[0].clone()
            hidden[:, :, start:end] = hidden[:, :, start:end] + broadcast_delta(delta, hidden)
            return (hidden, *inputs[1:])

        return block.out_proj.register_forward_pre_hook(hook)
    if vector.representation_type == "residual_stream":
        block = model.blocks[layer]
        original = block.forward_step

        def forward_step(hidden: torch.Tensor, past_k: torch.Tensor | None = None, past_v: torch.Tensor | None = None):
            output, key, value = original(hidden, past_k, past_v)
            return output + broadcast_delta(delta, output), key, value

        return TemporaryMethodPatch(block, "forward_step", forward_step)
    if vector.representation_type == "mlp_output":
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
            return output + broadcast_delta(delta, output)

        return model.blocks[layer].mlp.register_forward_hook(hook)
    if vector.representation_type == "final_hidden":
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
            return output + broadcast_delta(delta, output)

        return model.ln_f.register_forward_hook(hook)
    raise KeyError(vector.representation_type)


def install_hooks(model: torch.nn.Module, vectors: list[SteeringVector], deltas: list[torch.Tensor]) -> HookGroup:
    if len(vectors) != len(deltas):
        raise RuntimeError("Vector and delta counts differ.")
    return HookGroup([install_hook(model, vector, delta) for vector, delta in zip(vectors, deltas)])


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
            prefix_tensor = torch.tensor([units[index].prefix_ids for index in selected], dtype=torch.long, device=device)
            past_kvs, total_len = None, 0
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                for position in range(prefix_len - 1):
                    x_step = prefix_tensor[:, position:position + 1]
                    logits, past_kvs = model.forward_step(x_step, min(total_len, model.max_len - 1), past_kvs)
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
        max_new = prefix_batch.max_new
        past_kvs = None if prefix_batch.past_kvs is None else [
            (key.clone(), value.clone()) for key, value in prefix_batch.past_kvs
        ]
        total_len = prefix_batch.total_len
        handle = install_hooks(model, vectors, deltas) if vectors else None
        try:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits, past_kvs = model.forward_step(
                    prefix_batch.last_token,
                    min(total_len, model.max_len - 1),
                    past_kvs,
                )
        finally:
            if handle is not None:
                handle.remove()

        last_logits = logits[:, -1, :].float()
        total_len += 1
        batch_count = len(selected)
        generated = torch.full((batch_count, max_new), int(pad_id), dtype=torch.long, device=device)
        lengths = torch.zeros(batch_count, dtype=torch.int64, device=device)
        active = torch.ones(batch_count, dtype=torch.bool, device=device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            for step in range(max_new):
                next_tokens = last_logits.argmax(dim=-1)
                stop = (next_tokens.unsqueeze(1) == stop_ids.unsqueeze(0)).any(dim=1)
                emit = active & ~stop
                generated[:, step] = torch.where(emit, next_tokens, generated[:, step])
                lengths += emit.to(torch.int64)
                active &= ~stop
                x_step = torch.where(active, next_tokens, torch.full_like(next_tokens, int(pad_id))).unsqueeze(1)
                logits, past_kvs = model.forward_step(x_step, min(total_len, model.max_len - 1), past_kvs)
                last_logits = logits[:, -1, :].float()
                total_len += 1

                # Avoid synchronizing the CPU and GPU every token. At most 15
                # harmless masked steps are performed after all rows finish.
                if (step + 1) % 16 == 0 and not bool(active.any()):
                    break

        generated_cpu = generated.cpu()
        lengths_cpu = lengths.cpu()
        for local_index, global_index in enumerate(selected):
            outputs[global_index] = generated_cpu[local_index, :int(lengths_cpu[local_index])].tolist()
    return outputs


def semantic_event_after_latest_token(ids: list[int], contract: Any) -> str | None:
    if not ids:
        return None
    last = ids[-1]
    if last == contract.BACK:
        return "back_token"
    if last not in {contract.ARROW, contract.SEMI}:
        return None
    previous_semicolon = next((index for index in range(len(ids) - 2, -1, -1) if ids[index] == contract.SEMI), -1)
    statement_start = previous_semicolon + 1
    if last == contract.ARROW:
        parsed_state = reachability.parse_state(ids, statement_start, contract)
        if parsed_state is not None and parsed_state[1] == len(ids) - 1:
            return "pre_first_operand"
        return None
    parsed_operation = reachability.parse_operation_at(ids, statement_start, contract)
    if parsed_operation is not None and parsed_operation["next_pos"] == len(ids):
        return "resulting_state_semicolon"
    parsed_back = reachability.parse_back_at(ids, statement_start, contract)
    if parsed_back is not None and parsed_back["next_pos"] == len(ids):
        return "restored_state_semicolon"
    return None


def scheduled_hooks(
    model: torch.nn.Module,
    event_plan: dict[str, tuple[SteeringVector, torch.Tensor]],
    events: list[str | None],
) -> HookGroup | None:
    vectors: list[SteeringVector] = []
    deltas: list[torch.Tensor] = []
    for event in dict.fromkeys(event for event in events if event is not None):
        if event not in event_plan:
            continue
        vector, delta = event_plan[event]
        batched_delta = torch.zeros((len(events), delta.numel()), dtype=delta.dtype, device=delta.device)
        rows = [index for index, current in enumerate(events) if current == event]
        batched_delta[rows] = delta
        vectors.append(vector)
        deltas.append(batched_delta)
    return install_hooks(model, vectors, deltas) if vectors else None


@torch.inference_mode()
def generate_scheduled_suffixes(
    model: torch.nn.Module,
    units: list[PrefixUnit],
    prefix_batches: list[PrefixBatch],
    vector_lookup: dict[tuple[str, str], SteeringVector],
    condition: ScheduledCondition,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    eos_id: int,
    pad_id: int,
    tok: Any,
) -> list[list[int]]:
    outputs: list[list[int]] = [[] for _ in units]
    histories = [list(unit.generated_prefix) for unit in units]
    stop_ids = torch.tensor([int(eos_id), int(pad_id)], dtype=torch.long, device=device)
    event_plan = {
        event: (
            vector_lookup[(condition.method, key)],
            condition.beta * scale * vector_lookup[(condition.method, key)].direction.to(device),
        )
        for event, key, scale in condition.event_members
    }
    contract = reachability.TokenContract(tok)

    for prefix_batch in prefix_batches:
        selected, max_new = prefix_batch.selected, prefix_batch.max_new
        past_kvs = None if prefix_batch.past_kvs is None else [(key.clone(), value.clone()) for key, value in prefix_batch.past_kvs]
        total_len = prefix_batch.total_len
        handle = scheduled_hooks(model, event_plan, ["pre_first_operand"] * len(selected))
        try:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits, past_kvs = model.forward_step(prefix_batch.last_token, min(total_len, model.max_len - 1), past_kvs)
        finally:
            if handle is not None:
                handle.remove()

        last_logits = logits[:, -1, :].float()
        total_len += 1
        generated = torch.full((len(selected), max_new), int(pad_id), dtype=torch.long, device=device)
        lengths = torch.zeros(len(selected), dtype=torch.int64, device=device)
        active = torch.ones(len(selected), dtype=torch.bool, device=device)

        for step in range(max_new):
            next_tokens = last_logits.argmax(dim=-1)
            stop = (next_tokens.unsqueeze(1) == stop_ids.unsqueeze(0)).any(dim=1)
            emit = active & ~stop
            generated[:, step] = torch.where(emit, next_tokens, generated[:, step])
            lengths += emit.to(torch.int64)
            token_values, emit_values = next_tokens.cpu().tolist(), emit.cpu().tolist()
            events: list[str | None] = []
            for local, global_index in enumerate(selected):
                if emit_values[local]:
                    histories[global_index].append(token_values[local])
                    event = semantic_event_after_latest_token(histories[global_index], contract)
                    events.append(event if condition.schedule != "single" else None)
                else:
                    events.append(None)
            active &= ~stop
            if step + 1 >= max_new or not bool(active.any()):
                break
            x_step = torch.where(active, next_tokens, torch.full_like(next_tokens, int(pad_id))).unsqueeze(1)
            handle = scheduled_hooks(model, event_plan, events)
            try:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    logits, past_kvs = model.forward_step(x_step, min(total_len, model.max_len - 1), past_kvs)
            finally:
                if handle is not None:
                    handle.remove()
            last_logits = logits[:, -1, :].float()
            total_len += 1

        generated_cpu, lengths_cpu = generated.cpu(), lengths.cpu()
        for local, global_index in enumerate(selected):
            outputs[global_index] = generated_cpu[local, :int(lengths_cpu[local])].tolist()
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
    full_trajectory: bool,
) -> dict[str, Any]:
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
                (index for index in range(len(allowable) - 1, -1, -1) if allowable[index] == parsed["restored_state"]),
                -1,
            )
            prefix_valid = restore_index >= 0
            active_state = parsed["restored_state"]
            if prefix_valid:
                valid_path = valid_path[:restore_index + 1]
            if parsed["semicolon_pos"] > intervention_index:
                future_viability.append(prefix_valid and state_is_viable(active_state, target))
                if not full_trajectory:
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
        expected_after = reachability.apply_operation(active_state, parsed["n1"], parsed["n2"], parsed["result"])
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
            if not full_trajectory:
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
    units: list[PrefixUnit],
    suffixes: list[list[int]],
    tok: Any,
    full_reachability_trajectory: bool = False,
) -> dict[str, Any]:
    labels: dict[str, list[Any]] = {name: [] for name in (*BOOL_SCORE_FIELDS, *INT_SCORE_FIELDS, *FLOAT_SCORE_FIELDS)}
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
                full_reachability_trajectory,
            ))
        else:
            result.update(
                next_state_viable=False, next_state_observed=False, ever_future_unreachable=False,
                future_reachable_state_fraction=0.0,
            )
        full_sequences.append(generated_ids)
        for name in BOOL_SCORE_FIELDS:
            labels[name].append(bool(result[name]))
        for name in INT_SCORE_FIELDS:
            labels[name].append(int(result[name]))
        for name in FLOAT_SCORE_FIELDS:
            labels[name].append(float(result[name]))
    tensors = {name: torch.tensor(values, dtype=torch.bool) for name, values in labels.items() if name in BOOL_SCORE_FIELDS}
    tensors.update({name: torch.tensor(labels[name], dtype=torch.int32) for name in INT_SCORE_FIELDS})
    tensors.update({name: torch.tensor(labels[name], dtype=torch.float32) for name in FLOAT_SCORE_FIELDS})
    return {"labels": tensors, "full_sequences": full_sequences, "suffixes": suffixes}


def summarise_result(
    units: list[PrefixUnit],
    baseline: dict[str, Any],
    result: dict[str, Any],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    weights = torch.tensor([unit.weight for unit in units], dtype=torch.float64)
    base, steered = baseline["labels"], result["labels"]
    summary: dict[str, Any] = {"units": len(units), "unique_traces": len({unit.record_pos for unit in units})}
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
        [suffix != baseline_suffix for suffix, baseline_suffix in zip(result["suffixes"], baseline["suffixes"])],
        dtype=torch.bool,
    )
    summary["continuation_change_rate"] = weighted_mean(continuation_changed.float(), weights)
    for field in ("generated_length", "generated_back_count"):
        summary[f"baseline_{field}_mean"] = weighted_mean(base[field].float(), weights)
        summary[f"steered_{field}_mean"] = weighted_mean(steered[field].float(), weights)
        summary[f"delta_{field}_mean"] = summary[f"steered_{field}_mean"] - summary[f"baseline_{field}_mean"]
    field = "future_reachable_state_fraction"
    summary[f"baseline_{field}_mean"] = weighted_mean(base[field], weights)
    summary[f"steered_{field}_mean"] = weighted_mean(steered[field], weights)
    summary[f"delta_{field}_mean"] = summary[f"steered_{field}_mean"] - summary[f"baseline_{field}_mean"]
    summary["baseline_operation_validity_rate"] = weighted_count_rate(base["op_valid_count"], base["op_total"], weights)
    summary["steered_operation_validity_rate"] = weighted_count_rate(steered["op_valid_count"], steered["op_total"], weights)
    summary["delta_operation_validity_rate"] = summary["steered_operation_validity_rate"] - summary["baseline_operation_validity_rate"]
    summary["baseline_state_consistency_rate"] = weighted_count_rate(base["state_consistent_count"], base["state_total"], weights)
    summary["steered_state_consistency_rate"] = weighted_count_rate(steered["state_consistent_count"], steered["state_total"], weights)
    summary["delta_state_consistency_rate"] = summary["steered_state_consistency_rate"] - summary["baseline_state_consistency_rate"]
    summary["baseline_failure_counts"] = dict(sorted((FAILURE_NAMES[int(code)], int((base["failure_category"] == code).sum())) for code in torch.unique(base["failure_category"])))
    summary["steered_failure_counts"] = dict(sorted((FAILURE_NAMES[int(code)], int((steered["failure_category"] == code).sum())) for code in torch.unique(steered["failure_category"])))
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
                torch.tensor([unit.record_pos for unit in units]), steered[field] - base[field], weights,
                bootstrap_samples, seed + stable_seed(field),
            )
            summary[f"delta_{field}_ci_low"] = low
            summary[f"delta_{field}_ci_high"] = high
    return summary


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
        "format_version": 1,
        "complete": True,
        "run_tag": STEERING_RUN_TAG,
        "phase": phase,
        "condition_id": condition_id,
        "condition": condition.__dict__ if condition is not None else None,
        "source_row": torch.tensor([unit.source_row for unit in units], dtype=torch.int64),
        "record_pos": torch.tensor([unit.record_pos for unit in units], dtype=torch.int32),
        "cache_idx": torch.tensor([unit.cache_idx for unit in units], dtype=torch.int64),
        "generated_idx": torch.tensor([unit.generated_idx for unit in units], dtype=torch.int16),
        "target": torch.tensor([unit.target for unit in units], dtype=torch.int16),
        "initial_state": torch.tensor([unit.initial_state for unit in units], dtype=torch.int16),
        "weights": torch.tensor([unit.weight for unit in units], dtype=torch.float32),
        "labels": result["labels"],
        "summary": summary,
    }
    if save_tokens:
        flat, offsets, lengths = pack_sequences(result["full_sequences"])
        suffix_flat, suffix_offsets, suffix_lengths = pack_sequences(result["suffixes"])
        payload.update(
            generated_flat_ids=flat, generated_offsets=offsets, generated_lengths=lengths,
            suffix_flat_ids=suffix_flat, suffix_offsets=suffix_offsets, suffix_lengths=suffix_lengths,
        )
    return payload

def unpack_generated_sequences(shard: dict[str, Any]) -> list[list[int]]:
    return [
        unpack_sequence(
            shard["generated_flat_ids"],
            shard["generated_offsets"],
            shard["generated_lengths"],
            index,
        )
        for index in range(int(shard["generated_lengths"].numel()))
    ]


def format_trace_text(token_ids: list[int], tok: Any) -> str:
    """Decode a trace and put each completed statement on its own line."""
    text = tok.decode(token_ids).strip()
    text = re.sub(r"\s*;\s*", " ;\n", text)
    text = re.sub(r"(?<!^)\s+(?=EXPR\b)", "\n", text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def example_case(baseline_correct: bool, steered_correct: bool, changed: bool) -> str:
    if not baseline_correct and steered_correct:
        return "rescue"
    if baseline_correct and not steered_correct:
        return "damage"
    if changed:
        return "changed continuation, same correctness"
    return "unchanged continuation"


def select_example_indices(
    baseline_shard: dict[str, Any],
    steered_shard: dict[str, Any],
    count: int,
) -> list[int]:
    baseline_sequences = unpack_generated_sequences(baseline_shard)
    steered_sequences = unpack_generated_sequences(steered_shard)
    baseline_correct = baseline_shard["labels"]["expr_target_correct"].bool()
    steered_correct = steered_shard["labels"]["expr_target_correct"].bool()

    if not (
        len(baseline_sequences)
        == len(steered_sequences)
        == int(baseline_correct.numel())
        == int(steered_correct.numel())
    ):
        raise RuntimeError("Qualitative-example shards have inconsistent row counts.")

    buckets: dict[str, list[int]] = {
        "rescue": [],
        "damage": [],
        "changed continuation, same correctness": [],
        "unchanged continuation": [],
    }

    for index, (baseline_ids, steered_ids) in enumerate(zip(baseline_sequences, steered_sequences)):
        changed = baseline_ids != steered_ids
        case = example_case(bool(baseline_correct[index]), bool(steered_correct[index]), changed)
        buckets[case].append(index)

    case_order = (
        "rescue",
        "damage",
        "changed continuation, same correctness",
        "unchanged continuation",
    )

    selected: list[int] = []

    # First choose one example from each available outcome type.
    for case in case_order:
        if buckets[case]:
            selected.append(buckets[case][0])
            if len(selected) >= count:
                return selected

    # Fill any remaining positions using the same priority order.
    for case in case_order:
        for index in buckets[case][1:]:
            if len(selected) >= count:
                return selected
            selected.append(index)

    return selected


def write_qualitative_examples(
    path: str,
    tok: Any,
    conditions: list[SteeringCondition],
    units_by_key: dict[tuple[str, str], list[PrefixUnit]],
    baseline_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    examples_per_condition: int,
) -> None:
    if examples_per_condition <= 0:
        return

    baseline_paths = {
        (str(row["task"]), str(row["checkpoint"])): str(row["shard_path"])
        for row in baseline_rows
        if row["phase"] == "test"
    }
    test_paths = {str(row["condition_id"]): str(row["shard_path"]) for row in test_rows}

    lines = [
        "CHAINSEARCH STEERING QUALITATIVE EXAMPLES",
        "",
        "Each section compares the regenerated no-intervention baseline with the paired steered continuation.",
        "Examples prioritize rescues, damage cases, and other changed continuations.",
        "",
    ]

    for condition_number, condition in enumerate(conditions, start=1):
        key = (condition.task, condition.checkpoint)

        if key not in baseline_paths or condition.condition_id not in test_paths or key not in units_by_key:
            raise RuntimeError(f"Missing qualitative-example inputs for {condition.condition_id}.")

        baseline_shard = load_shard(baseline_paths[key])
        steered_shard = load_shard(test_paths[condition.condition_id])

        for field in ("source_row", "record_pos", "cache_idx", "generated_idx"):
            if not torch.equal(baseline_shard[field], steered_shard[field]):
                raise RuntimeError(f"Baseline/steered row mismatch for {condition.condition_id}: {field}.")

        units = units_by_key[key]
        baseline_sequences = unpack_generated_sequences(baseline_shard)
        steered_sequences = unpack_generated_sequences(steered_shard)
        chosen = select_example_indices(baseline_shard, steered_shard, examples_per_condition)
        baseline_labels = baseline_shard["labels"]
        steered_labels = steered_shard["labels"]
        beta_text = "native" if condition.native_direct else str(condition.beta)

        lines.extend([
            "=" * 120,
            f"CONDITION {condition_number}",
            f"Condition ID  {condition.condition_id}",
            f"Method  {condition.method}",
            f"Task  {condition.task}",
            f"Checkpoint  {condition.checkpoint}",
            f"Location  {condition.location_name}",
            f"Beta  {beta_text}",
            f"Alpha  {condition.alpha:.8g}",
            "",
        ])

        for example_number, index in enumerate(chosen, start=1):
            baseline_correct = bool(baseline_labels["expr_target_correct"][index])
            steered_correct = bool(steered_labels["expr_target_correct"][index])
            baseline_valid = bool(baseline_labels["valid_trace"][index])
            steered_valid = bool(steered_labels["valid_trace"][index])
            changed = baseline_sequences[index] != steered_sequences[index]
            case = example_case(baseline_correct, steered_correct, changed)

            lines.extend([
                "-" * 120,
                f"EXAMPLE {example_number}",
                f"Target  {units[index].target}",
                f"Outcome case  {case}",
                f"Baseline answer correct  {'yes' if baseline_correct else 'no'}",
                f"Steered answer correct  {'yes' if steered_correct else 'no'}",
                f"Baseline valid trace  {'yes' if baseline_valid else 'no'}",
                f"Steered valid trace  {'yes' if steered_valid else 'no'}",
                f"Continuation changed  {'yes' if changed else 'no'}",
                "",
                "BASELINE TRACE",
                format_trace_text(baseline_sequences[index], tok),
                "",
                "STEERED TRACE",
                format_trace_text(steered_sequences[index], tok),
                "",
            ])

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary_path = f"{path}.tmp.{os.getpid()}"
    Path(temporary_path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    os.replace(temporary_path, path)



# =============================================================================
# BASELINES, CONDITIONS, SELECTION, AND PERSISTENCE
# =============================================================================

def shard_path(phase: str, identifier: str) -> str:
    return os.path.join(SHARD_DIR, f"{phase}_{condition_hash(identifier)}.pt")


def load_shard(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    shard = torch.load(path, map_location="cpu", weights_only=False)
    if not bool(shard.get("complete", False)) or shard.get("run_tag") != STEERING_RUN_TAG:
        raise RuntimeError(f"Invalid steering shard: {path}")
    return shard


def generate_and_score(
    sources: dict[str, Any],
    units: list[PrefixUnit],
    prefix_batches: list[PrefixBatch],
    vector_lookup: dict[tuple[str, str], SteeringVector],
    condition: SteeringCondition | ScheduledCondition | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if isinstance(condition, ScheduledCondition):
        return generate_scheduled_and_score(sources, units, prefix_batches, vector_lookup, condition, args)
    vectors: list[SteeringVector] = []
    deltas: list[torch.Tensor] = []
    if condition is not None:
        keys = condition.member_keys or (condition.vector_key,)
        scales = condition.member_scales or (condition.scale,)
        if len(keys) != len(scales):
            raise RuntimeError(f"Condition {condition.condition_id} has mismatched member keys and scales.")
        vectors = [vector_lookup[(condition.method, key)] for key in keys]
        if condition.native_direct:
            deltas = [vector.raw_vector.to(args.device) for vector in vectors]
        else:
            deltas = [
                (condition.beta * scale * vector.direction).to(args.device)
                for vector, scale in zip(vectors, scales)
            ]
    suffixes = generate_suffixes(
        sources["model"],
        units,
        prefix_batches,
        vectors,
        deltas,
        args.device,
        args.use_amp,
        args.amp_dtype,
        sources["tok"].eos_id,
        sources["tok"].pad_id,
    )
    return score_suffixes(
        units, suffixes, sources["tok"],
        full_reachability_trajectory=args.experiment == "staying-power",
    )


def generate_scheduled_and_score(
    sources: dict[str, Any],
    units: list[PrefixUnit],
    prefix_batches: list[PrefixBatch],
    vector_lookup: dict[tuple[str, str], SteeringVector],
    condition: ScheduledCondition,
    args: argparse.Namespace,
) -> dict[str, Any]:
    suffixes = generate_scheduled_suffixes(
        sources["model"], units, prefix_batches, vector_lookup, condition, args.device, args.use_amp,
        args.amp_dtype, sources["tok"].eos_id, sources["tok"].pad_id, sources["tok"],
    )
    return score_suffixes(units, suffixes, sources["tok"], full_reachability_trajectory=True)


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
        return shard, shard["summary"], path
    result = generate_and_score(sources, units, prefix_batches, {}, None, args)
    summary = summarise_result(units, result, result, 0, stable_seed(identifier))
    reproduction = torch.tensor(
        [suffix == list(unit.original_suffix) for unit, suffix in zip(units, result["suffixes"])],
        dtype=torch.float32,)
    next_token_reproduction = torch.tensor(
        [
            (suffix[0] if suffix else None) == (unit.original_suffix[0] if unit.original_suffix else None)
            for unit, suffix in zip(units, result["suffixes"])
        ],
        dtype=torch.float32,
    )
    unit_weights = torch.tensor([unit.weight for unit in units], dtype=torch.float64)
    summary["original_suffix_reproduction_rate"] = weighted_mean(reproduction, unit_weights)
    summary["original_next_token_reproduction_rate"] = weighted_mean(next_token_reproduction, unit_weights)
    atomic_save(result_shard("baseline", identifier, None, units, result, summary, True), path)
    return load_shard(path), summary, path


def baseline_result(shard: dict[str, Any]) -> dict[str, Any]:
    suffixes = [
        unpack_sequence(shard["suffix_flat_ids"], shard["suffix_offsets"], shard["suffix_lengths"], index)
        for index in range(int(shard["suffix_lengths"].numel()))
    ]
    return {"labels": shard["labels"], "full_sequences": [], "suffixes": suffixes}


def evaluate_condition(
    sources: dict[str, Any],
    units: list[PrefixUnit],
    prefix_batches: list[PrefixBatch],
    baseline: dict[str, Any],
    vector_lookup: dict[tuple[str, str], SteeringVector],
    condition: SteeringCondition | ScheduledCondition,
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
    row = {
        **condition.__dict__,
        **summary,
        "phase": phase,
    }
    save_shard = phase == "test" or args.save_validation_tokens
    path = shard_path(phase, condition.condition_id) if save_shard else ""
    if save_shard:
        atomic_save(result_shard(phase, condition.condition_id, condition, units, result, summary, True), path)
    row["shard_path"] = path or None
    return row, path


def select_conditions(
    rows: list[dict[str, Any]],
    top_k: int,
    include_native: bool,
    positive_only: bool = True,
    require_nonnegative_local_reachability: bool = False,
) -> list[str]:

    row_ids = [str(row["condition_id"]) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise RuntimeError("Condition-selection rows contain duplicate condition IDs.")

    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    native_by_vector: dict[str, str] = {}

    for row in rows:
        if row["native_direct"]:
            vector_key_value = str(row["vector_key"])
            if vector_key_value in native_by_vector:
                raise RuntimeError(f"Multiple native Direct rows for {vector_key_value}.")
            native_by_vector[vector_key_value] = str(row["condition_id"])
            continue
        if positive_only and float(row["beta"]) <= 0:
            continue
        if (
            require_nonnegative_local_reachability
            and is_local_to_global_candidate(row)
            # Exclude validation regressions, while allowing a null local
            # effect to proceed to the held-out local-to-global test.
            and primary_effect(row) < 0.0
        ):
            continue
        key = (
            row["method"],
            row["label_family"],
            row["checkpoint"],
            row["representation_type"],
            row.get("selection_variant", "single"),
        )
        groups[key].append(row)

    winners: list[dict[str, Any]] = []
    for key in sorted(groups):
        winners.extend(sorted(groups[key], key=row_rank, reverse=True)[:top_k])

    selected = [str(row["condition_id"]) for row in winners]

    if include_native:
        for row in winners:
            if row["method"] != "optimised":
                continue
            native_id = native_by_vector.get(str(row["vector_key"]))
            if native_id is None:
                raise RuntimeError(
                    f"Missing native Direct companion for {row['vector_key']}."
                )
            selected.append(native_id)
    return list(dict.fromkeys(selected))

def save_partial(signature: dict[str, Any], state: dict[str, Any]) -> None:
    atomic_save({"format_version": 1, "complete": False, "signature": signature, **state}, PARTIAL_PATH)


def load_partial(signature: dict[str, Any]) -> dict[str, Any]:
    partial = torch.load(PARTIAL_PATH, map_location="cpu", weights_only=False)
    partial_signature = partial.get("signature")
    legacy_signature = dict(signature)
    validation_gate = legacy_signature.pop("local_to_global_validation_gate")
    legacy_signature["local_to_global_positive_validation_only"] = (
        validation_gate != "not_applicable"
    )
    if partial_signature != signature and partial_signature != legacy_signature:
        raise RuntimeError("Partial steering artifact was created with a different configuration.")
    return {
        "baseline_rows": list(partial.get("baseline_rows", [])),
        "coarse_rows": list(partial.get("coarse_rows", [])),
        "refined_condition_ids": list(partial.get("refined_condition_ids", [])),
        "refined_rows": list(partial.get("refined_rows", [])),
        "selected_condition_ids": list(partial.get("selected_condition_ids", [])),
        "test_rows": list(partial.get("test_rows", [])),
        "completed_ids": set(partial.get("completed_ids", [])),
    }


def build_signature(args: argparse.Namespace, sources: dict[str, Any], conditions: list[SteeringCondition]) -> dict[str, Any]:
    source_files = {
        **sources["paths"],
        "probe_dataset": probe.DATASET_PATH,
        "evaluated_generations": str(sources["data"]["source_evaluated_path"]),
        "token_cache": str(sources["data"]["source_token_cache_path"]),
        "checkpoint": str(sources["data"]["source_checkpoint_path"]),
    }
    return {
        "signature_version": 7,
        "run_tag": STEERING_RUN_TAG,
        "source_run_tag": SOURCE_RUN_TAG,
        "experiment": args.experiment,
        "experiment_title": TEST_NAMES[args.experiment],
        "methods": list(args.methods),
        "betas": list(args.betas),
        "checkpoints": list(args.checkpoints),
        "representations": list(args.representations),
        "scale_mode": args.scale_mode,
        "include_reachability": args.include_reachability,
        "include_pairwise_extra": args.include_pairwise_extra,
        "include_native_direct": args.include_native_direct,
        "negative_control_policy": "matched_selected_only",
        "selection_objective": "label_family_primary_effect",
        "local_to_global_validation_gate": ( "nonnegative" if args.experiment == "local-to-global" else "not_applicable"),
        "coarse_validation_units": args.coarse_validation_units,
        "refined_validation_units": args.refined_validation_units,
        "test_units": args.test_units,
        "refine_k": args.refine_k,
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "bootstrap_samples": args.bootstrap_samples,
        "minimum_reproduction_rate": args.minimum_reproduction_rate,
        "hard_minimum_reproduction_rate": args.hard_minimum_reproduction_rate,
        "max_new_tokens": args.max_new_tokens,
        "save_validation_tokens": args.save_validation_tokens,
        "qualitative_examples_per_condition": args.qualitative_examples_per_condition,
        "save_every": args.save_every,
        "condition_ids": [condition.condition_id for condition in conditions],
        "source_files": {name: file_signature(path) for name, path in source_files.items()},
    }


# =============================================================================
# AUDIT AND MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Causally steer ChainSearch continuations with saved activation directions.")
    parser.add_argument("--suite", choices=("all", "single"), default="all")
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="residual-pilot")
    parser.add_argument("--methods", type=parse_methods, default=parse_methods("probe,caa,optimised"))
    parser.add_argument("--betas", nargs="+", type=float, default=list(DEFAULT_BETAS),
                        help="Positive CAA-norm multipliers; matched negative test controls are added automatically.")
    parser.add_argument("--checkpoints", nargs="+", choices=sorted(CAUSAL_CHECKPOINTS), default=sorted(BALANCED_CHECKPOINTS))
    parser.add_argument("--representations", nargs="+", choices=REPRESENTATION_ORDER, default=sorted(BALANCED_REPRESENTATIONS))
    parser.add_argument("--scale-mode", choices=("caa_norm", "unit"), default="caa_norm")
    parser.add_argument("--include-reachability", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-pairwise-extra", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-native-direct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--coarse-validation-units", type=int)
    parser.add_argument("--refined-validation-units", type=int)
    parser.add_argument("--test-units", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--refine-k", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument("--minimum-reproduction-rate", type=float, default=0.95)
    parser.add_argument(
        "--hard-minimum-reproduction-rate",
        type=float,
        default=0.80,
        help="Abort below this replay rate. Rates below the preferred threshold but above this floor only warn.",
    )
    parser.add_argument("--max-candidates-per-group", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--save-validation-tokens", action="store_true")
    parser.add_argument("--qualitative-examples-per-condition", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--expected-direct-max-steps",
        type=int,
        default=1500,
        help="Fail if the Direct archive was produced with another step ceiling.",
    )
    args = parser.parse_args()
    args.coarse_validation_units = resolve(args.coarse_validation_units, "coarse_validation_units", args.smoke)
    args.refined_validation_units = resolve(args.refined_validation_units, "refined_validation_units", args.smoke)
    args.test_units = resolve(args.test_units, "test_units", args.smoke)
    args.batch_size = resolve(args.batch_size, "batch_size", args.smoke)
    args.refine_k = resolve(args.refine_k, "refine_k", args.smoke)
    args.top_k = resolve(args.top_k, "top_k", args.smoke)
    args.bootstrap_samples = resolve(args.bootstrap_samples, "bootstrap_samples", args.smoke)
    args.qualitative_examples_per_condition = resolve(
        args.qualitative_examples_per_condition,
        "qualitative_examples_per_condition",
        args.smoke,
    )
    args.save_every = resolve(args.save_every, "save_every", args.smoke)
    args.checkpoints = tuple(dict.fromkeys(args.checkpoints))
    args.representations = tuple(dict.fromkeys(args.representations))
    parsed_betas = [float(beta) for beta in args.betas]
    if any(not math.isfinite(beta) for beta in parsed_betas):
        raise ValueError("--betas must all be finite.")
    args.betas = tuple(dict.fromkeys(parsed_betas))
    if not args.betas:
        raise ValueError("At least one positive beta is required.")
    if any(beta <= 0.0 for beta in args.betas):
        raise ValueError("--betas must contain positive strengths only; beta=0 is the baseline.")

    for name in ("coarse_validation_units", "refined_validation_units", "test_units", "batch_size", "refine_k", "top_k"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.bootstrap_samples < 0 or args.max_candidates_per_group < 0 or args.max_new_tokens < 0 or args.qualitative_examples_per_condition < 0:
        raise ValueError("Bootstrap samples, candidate caps, token caps, and example counts must be non-negative.")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive.")

    if not 0.0 <= args.minimum_reproduction_rate <= 1.0:
        raise ValueError("--minimum-reproduction-rate must lie in [0, 1].")

    if not 0.0 <= args.hard_minimum_reproduction_rate <= 1.0:
        raise ValueError("--hard-minimum-reproduction-rate must lie in [0, 1].")
    if args.hard_minimum_reproduction_rate > args.minimum_reproduction_rate:
        raise ValueError("--hard-minimum-reproduction-rate cannot exceed --minimum-reproduction-rate.")

    args.device = torch.device(args.device)
    if args.device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        torch.cuda.set_device(args.device)
    args.use_amp = args.device.type == "cuda"
    args.amp_dtype = torch.bfloat16 if args.use_amp and torch.cuda.is_bf16_supported() else torch.float16
    return args


def experiment_args(base: argparse.Namespace, experiment: str) -> argparse.Namespace:
    args = copy.deepcopy(base)
    args.suite = "single"
    args.experiment = experiment
    if experiment == "residual-pilot":
        return args
    if experiment == "steering-atlas":
        args.checkpoints = tuple(sorted(PRIMARY_CHECKPOINTS))
        args.representations = REPRESENTATION_ORDER
        args.include_pairwise_extra = True
        args.max_candidates_per_group = 0
        args.test_units = 2 if args.smoke else 256
        return args
    if experiment == "headcount":
        args.checkpoints = tuple(sorted(PRIMARY_CHECKPOINTS))
        args.representations = ("attention_head",)
        args.include_pairwise_extra = False
        args.include_native_direct = False
        args.max_candidates_per_group = 0
        args.test_units = 2 if args.smoke else 256
        return args
    if experiment == "staying-power":
        args.checkpoints = tuple(sorted(PRIMARY_CHECKPOINTS))
        args.representations = REPRESENTATION_ORDER
        args.include_pairwise_extra = True
        args.include_native_direct = False
        args.max_candidates_per_group = 0
        args.refine_k = 1
        args.top_k = 1
        args.test_units = 2 if args.smoke else 512
        return args
    if experiment == "local-to-global":
        args.checkpoints = LOCAL_TO_GLOBAL_CHECKPOINTS
        args.representations = ("attention_head",)
        args.include_reachability = True
        args.include_pairwise_extra = False
        args.include_native_direct = False
        args.max_candidates_per_group = 0
        args.top_k = 1
        args.test_units = 2 if args.smoke else 256
        return args
    raise KeyError(experiment)


def final_audit(
    payload: dict[str, Any],
    selection_conditions: list[SteeringCondition],
    preferred_reproduction_rate: float,
    hard_reproduction_floor: float,
) -> dict[str, Any]:
    expected_coarse_ids = {condition.condition_id for condition in selection_conditions}
    coarse_ids = {row["condition_id"] for row in payload["coarse_validation_rows"]}
    refined_selected = set(payload["refined_condition_ids"])
    refined_ids = {row["condition_id"] for row in payload["refined_validation_rows"]}
    validation_selected = set(payload["validation_selected_condition_ids"])
    expected_test_ids = set(payload["selected_condition_ids"])
    test_ids = {row["condition_id"] for row in payload["test_condition_rows"]}
    negative_control_of = dict(payload["negative_control_of"])
    if coarse_ids != expected_coarse_ids:
        raise RuntimeError("Coarse-validation condition coverage is incomplete.")
    if refined_ids != refined_selected:
        raise RuntimeError("Refined-validation conditions do not match coarse selection.")
    if not validation_selected <= refined_selected:
        raise RuntimeError("A test condition bypassed refined validation.")
    if test_ids != expected_test_ids:
        raise RuntimeError("Test rows do not match selected conditions plus negative controls.")
    if expected_test_ids != validation_selected | set(negative_control_of):
        raise RuntimeError("Negative-control bookkeeping does not match the test condition set.")
    if not set(negative_control_of.values()) <= validation_selected:
        raise RuntimeError("A negative control is not paired with a selected positive condition.")
    required_shards = [row["shard_path"] for row in payload["baseline_rows"]]
    required_shards += [row["shard_path"] for row in payload["test_condition_rows"]]
    missing_shards = [path for path in required_shards if not path or not os.path.isfile(path)]
    if missing_shards:
        raise RuntimeError(f"Missing required steering shards: {missing_shards[:5]}")
    suffix_reproduction = [row["original_suffix_reproduction_rate"] for row in payload["baseline_rows"]]
    next_token_reproduction = [row["original_next_token_reproduction_rate"] for row in payload["baseline_rows"]]

    if min(next_token_reproduction) < hard_reproduction_floor:
        raise RuntimeError("A baseline replay rate fell below the hard safety floor during final audit.")

    return {
        "coarse_validation_conditions": len(coarse_ids),
        "refined_validation_conditions": len(refined_ids),
        "selected_test_conditions": len(test_ids),
        "selected_positive_or_native_conditions": len(validation_selected),
        "matched_negative_test_controls": len(negative_control_of),
        "minimum_baseline_suffix_reproduction_rate": min(suffix_reproduction),
        "minimum_baseline_next_token_reproduction_rate": min(next_token_reproduction),
        "preferred_baseline_next_token_reproduction_rate": preferred_reproduction_rate,
        "hard_baseline_next_token_reproduction_floor": hard_reproduction_floor,
        "baseline_replay_warning_count": sum(
            rate < preferred_reproduction_rate for rate in next_token_reproduction
        ),
        "test_used_for_selection": False,
        "all_sources_complete": True,
        "overall_status": "PASS",
    }


def run_experiment(args: argparse.Namespace) -> str:
    global OUTPUT_PATH, PARTIAL_PATH, SHARD_DIR, QUALITATIVE_PATH
    OUTPUT_PATH, PARTIAL_PATH, SHARD_DIR = output_paths(args.smoke, args.experiment)
    QUALITATIVE_PATH = qualitative_output_path(args.smoke, args.experiment)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if args.device.type == "cuda":
        torch.cuda.set_device(args.device)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    if os.path.exists(OUTPUT_PATH) and not args.overwrite:
        raise FileExistsError(f"Output exists: {OUTPUT_PATH}. Use --overwrite to replace it.")
    if args.no_resume and os.path.exists(PARTIAL_PATH):
        os.remove(PARTIAL_PATH)
    if args.no_resume or not os.path.isfile(PARTIAL_PATH):
        if os.path.isdir(SHARD_DIR):
            shutil.rmtree(SHARD_DIR)
    Path(SHARD_DIR).mkdir(parents=True, exist_ok=True)

    probe.heading("PERFORM A* CAUSAL STEERING")
    probe.print_rows("RUN CONFIGURATION", [
        ("Experiment", TEST_NAMES[args.experiment]),
        ("Methods", ", ".join(args.methods)),
        ("Positive betas", list(args.betas)),
        ("Negative controls", "matched to selected test conditions only"),
        ("Checkpoints", ", ".join(args.checkpoints)),
        ("Representations", ", ".join(args.representations)),
        ("Scale mode", args.scale_mode),
        ("Coarse / refined / test units", f"{args.coarse_validation_units} / {args.refined_validation_units} / {args.test_units}"),
        ("Refine K / test K", f"{args.refine_k} / {args.top_k}"),
        (
            "Preferred / hard replay rates",
            f"{args.minimum_reproduction_rate:.2f} / {args.hard_minimum_reproduction_rate:.2f}",
        ),
        ("Examples per test condition", args.qualitative_examples_per_condition),
        ("Save every", f"{args.save_every} conditions"),
        ("Partial state", PARTIAL_PATH),
        ("Output", OUTPUT_PATH),
        ("Device", args.device),
    ])

    sources = load_sources(args.device)

    direct_archive = sources["archives"]["optimised"]
    run_config = direct_archive.get("run_config", {})
    direct_config = (
        run_config.get("config")
        or direct_archive.get("optimisation_config", {})
    )
    direct_max_steps = direct_config.get("max_steps")
    if args.expected_direct_max_steps is not None:
        if direct_max_steps is None:
            raise RuntimeError("Direct archive does not record max_steps.")
        if int(direct_max_steps) != args.expected_direct_max_steps:
            raise RuntimeError(
                f"Expected a {args.expected_direct_max_steps}-step Direct archive, "
                f"but found max_steps={direct_max_steps}."
            )
    sources["vector_audit"]["direct_max_steps"] = (
        int(direct_max_steps) if direct_max_steps is not None else None
    )
    vectors = candidate_vectors(
        sources["banks"], args.methods, args.include_reachability,
        args.include_pairwise_extra, set(args.checkpoints), set(args.representations),
        args.smoke and args.experiment == "residual-pilot",
    )
    headcount_anchors: dict[str, tuple[str, str]] | None = None
    source_atlas_path: str | None = None
    signed_betas = args.betas + tuple(-beta for beta in args.betas)
    if args.experiment == "steering-atlas":
        vectors = atlas_vectors(vectors)
    if args.experiment == "local-to-global":
        vectors = local_to_global_vectors(vectors, args.methods)
    if args.experiment in {"headcount", "staying-power"}:
        source_atlas_path = output_paths(args.smoke, "steering-atlas")[0]
        atlas = load_archive(source_atlas_path, "Where and When Test", STEERING_RUN_TAG)
    if args.experiment == "headcount":
        headcount_anchors = select_headcount_anchors(atlas, args.methods)
        vectors = [
            vector for vector in vectors
            if (vector.task, vector.checkpoint) == headcount_anchors[vector.label_family]
        ]
        conditions = build_headcount_conditions(vectors, sources["banks"], args.methods, signed_betas)
    elif args.experiment == "staying-power":
        conditions = build_staying_conditions(atlas, args.methods)
        needed = {(condition.method, key) for condition in conditions for _, key, _ in condition.event_members}
        vectors = [sources["banks"][method][key] for method, key in sorted(needed)]
    else:
        conditions = build_conditions(
            vectors, sources["banks"], signed_betas, args.scale_mode,
            args.include_native_direct, args.max_candidates_per_group,
        )
    selection_conditions = [
        condition for condition in conditions
        if condition.native_direct or condition.beta > 0
    ]

    condition_ids = [condition.condition_id for condition in conditions]
    if len(condition_ids) != len(set(condition_ids)):
        duplicates = [
            condition_id
            for condition_id, count in Counter(condition_ids).items()
            if count > 1
        ]
        raise RuntimeError(f"Duplicate steering condition IDs: {duplicates[:5]}.")

    condition_lookup = {condition.condition_id: condition for condition in conditions}
    vector_lookup = {(vector.method, vector.key): vector for vector in vectors}
    signature = build_signature(args, sources, conditions)
    if source_atlas_path is not None:
        signature["source_atlas"] = file_signature(source_atlas_path)
    if headcount_anchors is not None:
        signature["headcount_anchors"] = headcount_anchors
    probe.print_rows("CANDIDATE COUNTS", [
        ("Vectors", len(vectors)),
        ("All signed conditions", len(conditions)),
        ("Coarse selection conditions", len(selection_conditions)),
        ("Headcount anchors", headcount_anchors or "not applicable"),
    ])
    state = load_partial(signature) if os.path.isfile(PARTIAL_PATH) and not args.no_resume else {
        "baseline_rows": [],
        "coarse_rows": [],
        "refined_condition_ids": [],
        "refined_rows": [],
        "selected_condition_ids": [],
        "test_rows": [],
        "completed_ids": set(),
    }

    unit_cache: dict[tuple[str, str, str], list[PrefixUnit]] = {}
    baseline_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    active_prefix_key: tuple[str, str, str] | None = None
    active_prefix_batches: list[PrefixBatch] | None = None

    def units_for(task: str, checkpoint: str, phase: str) -> list[PrefixUnit]:
        key = (task, checkpoint, phase)
        if key not in unit_cache:
            if phase == "coarse":
                split, cap, offset = "val", args.coarse_validation_units, 0
            elif phase == "refined":
                split, cap, offset = "val", args.refined_validation_units, args.coarse_validation_units
            elif phase == "test":
                split, cap, offset = "test", args.test_units, 0
            else:
                raise KeyError(phase)
            unit_cache[key] = build_prefix_units(
                sources, task, checkpoint, split, cap, offset,
                unique_trace=args.experiment == "staying-power",
            )
        return unit_cache[key]

    def prefix_batches_for(task: str, checkpoint: str, phase: str) -> list[PrefixBatch]:
        nonlocal active_prefix_key, active_prefix_batches
        key = (task, checkpoint, phase)
        if active_prefix_key != key or active_prefix_batches is None:
            active_prefix_key = key
            active_prefix_batches = build_prefix_batches(
                sources["model"],
                units_for(task, checkpoint, phase),
                int(sources["evaluated"]["max_output_tokens"]),
                args.batch_size,
                args.device,
                args.use_amp,
                args.amp_dtype,
                args.max_new_tokens,
            )
        return active_prefix_batches

    def baseline_for(task: str, checkpoint: str, phase: str) -> dict[str, Any]:
        key = (task, checkpoint, phase)
        if key not in baseline_cache:
            units = units_for(task, checkpoint, phase)
            shard, summary, path = ensure_baseline(
                sources, units, prefix_batches_for(task, checkpoint, phase), phase, task, checkpoint, args,
            )
            baseline_cache[key] = baseline_result(shard)
            row = {"phase": phase, "task": task, "checkpoint": checkpoint, "shard_path": path, **summary}
            state["baseline_rows"] = [
                current for current in state["baseline_rows"]
                if not (current["phase"] == phase and current["task"] == task and current["checkpoint"] == checkpoint)
            ]
            state["baseline_rows"].append(row)
        return baseline_cache[key]

    def save_progress(force: bool = False) -> None:
        if force or len(state["completed_ids"]) % args.save_every == 0:
            save_partial(signature, state)
 

    def check_reproduction(phase: str) -> None:
        rows = [row for row in state["baseline_rows"] if row["phase"] == phase]
        if not rows:
            raise RuntimeError(f"No {phase} archived replay results were recorded.")

        probe.heading(f"{phase.upper()} ARCHIVE REPLAY AUDIT")
        for row in sorted(
            rows,
            key=lambda value: (
                float(value["original_next_token_reproduction_rate"]),
                str(value["task"]),
                str(value["checkpoint"]),
            ),
        ):
            print(
                f"  {row['task']} / {row['checkpoint']}"
                f"  next={float(row['original_next_token_reproduction_rate']):.4f}"
                f"  suffix={float(row['original_suffix_reproduction_rate']):.4f}"
            )

        observed = min(float(row["original_next_token_reproduction_rate"]) for row in rows)

        if observed < args.hard_minimum_reproduction_rate:
            raise RuntimeError(
                f"Minimum {phase} archived next-token reproduction rate {observed:.4f} is below "
                f"the hard safety floor {args.hard_minimum_reproduction_rate:.4f}."
            )

        if observed < args.minimum_reproduction_rate:
            print(
                f"WARNING: Minimum {phase} archived next-token reproduction rate {observed:.4f} "
                f"is below the preferred {args.minimum_reproduction_rate:.4f}. "
                "Steering comparisons will still use paired regenerated baselines."
            )

    baseline_keys = sorted({(condition.task, condition.checkpoint) for condition in selection_conditions})
    for task, checkpoint in baseline_keys:
        baseline_for(task, checkpoint, "coarse")
    check_reproduction("coarse")

    ordered_coarse_conditions = sorted(
        selection_conditions,
        key=lambda condition: (condition.task, condition.checkpoint, condition.method, condition.location_name, condition.beta),
    )
    coarse_progress = tqdm(ordered_coarse_conditions, desc="Coarse validation conditions", dynamic_ncols=True)
    for condition in coarse_progress:
        completed_key = f"coarse:{condition.condition_id}"
        if completed_key in state["completed_ids"]:
            continue
        units = units_for(condition.task, condition.checkpoint, "coarse")
        row, _ = evaluate_condition(
            sources, units, prefix_batches_for(condition.task, condition.checkpoint, "coarse"),
            baseline_for(condition.task, condition.checkpoint, "coarse"),
            vector_lookup, condition, "coarse", args,
        )
        state["coarse_rows"].append(row)
        state["completed_ids"].add(completed_key)
        save_progress()
        coarse_progress.set_postfix(delta=f"{primary_effect(row):+.4f}")

    coarse_progress.close()
    save_progress(force=True)

    if not state["refined_condition_ids"]:
        state["refined_condition_ids"] = select_conditions(
            state["coarse_rows"], args.refine_k, args.include_native_direct,
            positive_only=True,
        )
        save_progress(force=True)

    refined_conditions = [condition_lookup[condition_id] for condition_id in state["refined_condition_ids"]]
    for task, checkpoint in sorted({(condition.task, condition.checkpoint) for condition in refined_conditions}):
        baseline_for(task, checkpoint, "refined")
    check_reproduction("refined")

    ordered_refined_ids = sorted(
        state["refined_condition_ids"],
        key=lambda condition_id: (
            condition_lookup[condition_id].task,
            condition_lookup[condition_id].checkpoint,
            condition_lookup[condition_id].method,
            condition_lookup[condition_id].location_name,
            condition_lookup[condition_id].beta,
        ),
    )
    refined_progress = tqdm(ordered_refined_ids, desc="Refined validation conditions", dynamic_ncols=True)
    for condition_id in refined_progress:
        completed_key = f"refined:{condition_id}"
        if completed_key in state["completed_ids"]:
            continue
        condition = condition_lookup[condition_id]
        units = units_for(condition.task, condition.checkpoint, "refined")
        row, _ = evaluate_condition(
            sources, units, prefix_batches_for(condition.task, condition.checkpoint, "refined"),
            baseline_for(condition.task, condition.checkpoint, "refined"),
            vector_lookup, condition, "refined", args,
        )
        state["refined_rows"].append(row)
        state["completed_ids"].add(completed_key)
        save_progress()
        refined_progress.set_postfix(delta=f"{primary_effect(row):+.4f}")

    refined_progress.close()
    save_progress(force=True)

    if not state["selected_condition_ids"]:
        state["selected_condition_ids"] = select_conditions(
            state["refined_rows"], args.top_k, args.include_native_direct,
            positive_only=True,
            require_nonnegative_local_reachability=args.experiment == "local-to-global",
        )
        save_progress(force=True)

    test_condition_ids, negative_control_of = add_matched_negative_controls(
        state["selected_condition_ids"], condition_lookup,
    )
    test_conditions = [condition_lookup[condition_id] for condition_id in test_condition_ids]
    for task, checkpoint in sorted({(condition.task, condition.checkpoint) for condition in test_conditions}):
        baseline_for(task, checkpoint, "test")
    check_reproduction("test")

    ordered_test_ids = sorted(
        test_condition_ids,
        key=lambda condition_id: (
            condition_lookup[condition_id].task,
            condition_lookup[condition_id].checkpoint,
            condition_lookup[condition_id].method,
            condition_lookup[condition_id].location_name,
            condition_lookup[condition_id].beta,
        ),
    )
    test_progress = tqdm(ordered_test_ids, desc="Test steering conditions", dynamic_ncols=True)
    for condition_id in test_progress:
        completed_key = f"test:{condition_id}"
        if completed_key in state["completed_ids"]:
            continue
        condition = condition_lookup[condition_id]
        units = units_for(condition.task, condition.checkpoint, "test")
        row, _ = evaluate_condition(
            sources, units, prefix_batches_for(condition.task, condition.checkpoint, "test"),
            baseline_for(condition.task, condition.checkpoint, "test"),
            vector_lookup, condition, "test", args,
        )
        row["negative_control_of"] = negative_control_of.get(condition_id)
        state["test_rows"].append(row)
        state["completed_ids"].add(completed_key)
        save_progress()
        test_progress.set_postfix(delta=f"{primary_effect(row):+.4f}")

    test_progress.close()
    save_progress(force=True)

    qualitative_path: str | None = None
    if args.qualitative_examples_per_condition > 0:
        units_by_key = {
            (condition.task, condition.checkpoint): units_for(condition.task, condition.checkpoint, "test")
            for condition in test_conditions
        }
        write_qualitative_examples(
            QUALITATIVE_PATH,
            sources["tok"],
            test_conditions,
            units_by_key,
            state["baseline_rows"],
            state["test_rows"],
            args.qualitative_examples_per_condition,
        )
        qualitative_path = QUALITATIVE_PATH

    payload = {
        "format_version": 1,
        "complete": True,
        "run_tag": STEERING_RUN_TAG,
        "source_run_tag": SOURCE_RUN_TAG,
        "source_paths": {
            **sources["paths"],
            "probe_dataset": probe.DATASET_PATH,
            "evaluated_generations": sources["data"]["source_evaluated_path"],
            "token_cache": sources["data"]["source_token_cache_path"],
            "checkpoint": sources["data"]["source_checkpoint_path"],
        },
        "run_config": signature,
        "failure_category_names": FAILURE_NAMES,
        "baseline_rows": state["baseline_rows"],
        "coarse_validation_rows": state["coarse_rows"],
        "refined_condition_ids": state["refined_condition_ids"],
        "refined_validation_rows": state["refined_rows"],
        "validation_selected_condition_ids": state["selected_condition_ids"],
        "selected_condition_ids": test_condition_ids,
        "test_condition_ids": test_condition_ids,
        "negative_control_of": negative_control_of,
        "test_condition_rows": state["test_rows"],
        "generation_shard_dir": SHARD_DIR,
        "qualitative_examples_path": qualitative_path,
        "vector_audit": sources["vector_audit"],
    }
    payload["audit"] = final_audit(
        payload,
        selection_conditions,
        args.minimum_reproduction_rate,
        args.hard_minimum_reproduction_rate,
    )
    atomic_save(payload, OUTPUT_PATH)
    if os.path.exists(PARTIAL_PATH):
        os.remove(PARTIAL_PATH)

    probe.heading("FINAL STEERING AUDIT")
    probe.print_rows("RESULTS", [
        ("Coarse validation conditions", payload["audit"]["coarse_validation_conditions"]),
        ("Refined validation conditions", payload["audit"]["refined_validation_conditions"]),
        ("Selected test conditions", payload["audit"]["selected_test_conditions"]),
        ("Selected positive/native conditions", payload["audit"]["selected_positive_or_native_conditions"]),
        ("Matched negative test controls", payload["audit"]["matched_negative_test_controls"]),
        ("Minimum next-token reproduction", f"{payload['audit']['minimum_baseline_next_token_reproduction_rate']:.4f}"),
        ("Minimum full-suffix reproduction", f"{payload['audit']['minimum_baseline_suffix_reproduction_rate']:.4f}"),
        ("Replay warnings", payload["audit"]["baseline_replay_warning_count"]),
        ("Test used for selection", "no"),
        ("Qualitative examples", qualitative_path or "disabled"),
        ("Output", OUTPUT_PATH),
        ("OVERALL STATUS", "PASS"),
    ])
    print("=" * probe.WIDTH)
    return OUTPUT_PATH


def main() -> None:
    args = parse_args()
    experiments = SUITE_EXPERIMENTS if args.suite == "all" else (args.experiment,)
    suite_started = time.monotonic()

    outputs: dict[str, str] = {}
    for index, experiment in enumerate(experiments, start=1):

        probe.heading(f"STEERING SUITE {index}/{len(experiments)} — {TEST_NAMES[experiment].upper()}")
        existing_output = output_paths(args.smoke, experiment)[0]
        if args.suite == "all" and not args.no_resume and not args.overwrite and os.path.isfile(existing_output):
            existing = load_archive(existing_output, experiment, STEERING_RUN_TAG)
            config = existing.get("run_config", {})
            if config.get("signature_version") != 7 or config.get("experiment") != experiment:
                raise RuntimeError(f"Existing {experiment} output was produced by another suite configuration.")
            outputs[experiment] = existing_output
            print(f"Already complete; skipping {existing_output}")
            continue
        output = run_experiment(experiment_args(args, experiment))
        outputs[experiment] = output
        gc.collect()
        if args.device.type == "cuda":
            torch.cuda.empty_cache()
    if args.suite == "all":
        manifest_path = suite_output_path(args.smoke)
        atomic_save({
            "format_version": 1,
            "complete": True,
            "run_tag": STEERING_RUN_TAG,
            "source_run_tag": SOURCE_RUN_TAG,
            "experiments": outputs,
            "experiment_titles": {name: TEST_NAMES[name] for name in outputs},
            "experiment_signatures": {name: file_signature(path) for name, path in outputs.items()},
            "elapsed_minutes": (time.monotonic() - suite_started) / 60.0,
            "overall_status": "PASS",
        }, manifest_path)
        probe.heading("STEERING SUITE COMPLETE")
        probe.print_rows("RESULTS", [
            ("Experiments", ", ".join(TEST_NAMES[name] for name in outputs)),
            ("Elapsed minutes", f"{(time.monotonic() - suite_started) / 60.0:.1f}"),
            ("Manifest", manifest_path),
            ("OVERALL STATUS", "PASS"),
        ])


if __name__ == "__main__":
    main()
