#!/usr/bin/env python3
"""
compute_direct_optimisation_vectors.py

Learn reusable local activation vectors through the frozen exp34 A* model.
The loss is mean token cross-entropy over a valid final EXPR sequence. When a
model-generated EXPR is correct, it is used as its own target. Otherwise, the
corresponding A* reference EXPR is used.

Direct Optimisation uses final-answer supervision only. It covers the same 10
final-answer checkpoints as probe/CAA, except expr_final_parenthesis, and only
model locations with a differentiable path to later answer-token logits. In
this four-layer model those are attention heads, residual streams, and MLP
outputs in layers 0-2. Layer 3 and final_hidden remain valid probe/CAA steering
sites, but their discrete next-token route cannot carry answer-loss gradients.

The model is frozen throughout. Only one additive vector is optimised at a
time. MLP-output and post-block residual interventions are exactly equivalent
for this architecture, so each shared block vector is optimised once and saved
under both matched keys.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import build_probe_dataset as builder
import compute_linear_probe_vectors as probe

pm = builder.pm


# =============================================================================
# CONFIGURATION
# =============================================================================

RUN_TAG = probe.RUN_TAG
SEED = 42

DIRECT_DIR = os.path.join(probe.ASTAR_DIR, "direct_optimisation")
CAA_PATH = os.path.join(probe.ASTAR_DIR, "caa", f"{RUN_TAG}_astar_caa_vectors.pt")

DIRECT_CHECKPOINTS = {
    "trace_outcome_operation_decisions": probe.OPERATION_CHECKPOINTS,
    "trace_outcome_trajectory": ("pre_back",),
    "trace_outcome_back_decisions": probe.BACK_CHECKPOINTS,
    "terminal_outcome_controls": ("success_token",),
}

GENERATED_TARGET = 0
REFERENCE_TARGET = 1
ZERO_NORM_EPS = 1e-12
GRAD_EPS = 1e-12
MIN_DELTA = 1e-5
DIRECTION_STABLE_COSINE = 0.999

FULL_DEFAULTS = {
    "batch_size": 16,
    "eval_batch_size": 128,
    "max_steps": 1500,
    "val_every": 20,
    "patience": 4,
    "val_units": 2048,
    "learning_rate": 1e-2,
    "l2": 1e-4,
    "gradient_clip": 1.0,
}

SMOKE_DEFAULTS = {
    "batch_size": 2,
    "eval_batch_size": 4,
    "max_steps": 5,
    "val_every": 1,
    "patience": 3,
    "val_units": 8,
    "learning_rate": 1e-2,
    "l2": 1e-4,
    "gradient_clip": 1.0,
}


# =============================================================================
# COMPACT DATA CONTAINERS
# =============================================================================

@dataclass
class TargetBank:
    flat_ids: torch.Tensor
    offsets: torch.Tensor
    lengths: torch.Tensor
    answer_start: torch.Tensor
    answer_end: torch.Tensor
    target_source: torch.Tensor

    def sequence(self, record_pos: int) -> torch.Tensor:
        start = int(self.offsets[record_pos])
        length = int(self.lengths[record_pos])
        if start < 0 or length <= 0:
            raise KeyError(f"No Direct Optimisation target for record_pos={record_pos}.")
        return self.flat_ids[start:start + length]


@dataclass
class CheckpointRows:
    task: str
    checkpoint: str
    checkpoint_index: int
    units: probe.UnitTable
    train_rows: torch.Tensor
    val_rows: torch.Tensor
    trace_normalized: bool


@dataclass
class ExampleSet:
    record_pos: torch.Tensor
    intervention_pos: torch.Tensor
    weights: torch.Tensor
    target_source: torch.Tensor

    @property
    def n(self) -> int:
        return int(self.record_pos.numel())


@dataclass(frozen=True)
class OptimisationSpec:
    kind: str
    layer: int
    head: int
    feature_dim: int
    location_id: str
    aliases: tuple[tuple[str, int], ...]


@dataclass
class ReplayBatch:
    input_ids: torch.Tensor
    intervention_pos: torch.Tensor
    answer_start: torch.Tensor
    answer_end: torch.Tensor
    weights: torch.Tensor
    target_source: torch.Tensor

    def to(self, device: torch.device) -> "ReplayBatch":
        return ReplayBatch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            intervention_pos=self.intervention_pos.to(device, non_blocking=True),
            answer_start=self.answer_start.to(device, non_blocking=True),
            answer_end=self.answer_end.to(device, non_blocking=True),
            weights=self.weights.to(device, non_blocking=True),
            target_source=self.target_source.to(device, non_blocking=True),
        )


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in (SEED, *parts))
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big") % (2**63 - 1)


def model_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def vector_key(task: str, checkpoint: str, location_name: str) -> str:
    return f"{task}:{checkpoint}:{location_name}"


def output_paths(smoke: bool) -> tuple[str, str]:
    suffix = "_smoke" if smoke else ""
    stem = f"{RUN_TAG}_astar_direct_optimisation_vectors{suffix}"
    return os.path.join(DIRECT_DIR, f"{stem}.pt"), os.path.join(DIRECT_DIR, f"{stem}.partial.pt")


def resolve_arg(value: Any, name: str, smoke: bool) -> Any:
    return value if value is not None else (SMOKE_DEFAULTS if smoke else FULL_DEFAULTS)[name]


# =============================================================================
# SOURCE LOADING AND TARGET CONSTRUCTION
# =============================================================================

def load_sources(device: torch.device) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any, torch.nn.Module, dict[str, Any]]:
    data = probe.load_probe_dataset()
    probe.audit_probe_dataset(data)

    if data.get("run_tag") != RUN_TAG:
        raise RuntimeError(f"Probe metadata run tag {data.get('run_tag')!r} != {RUN_TAG!r}.")

    checkpoint_path = str(data["source_checkpoint_path"])
    if os.path.basename(checkpoint_path) != f"{RUN_TAG}_astar_best.pt":
        raise RuntimeError(f"Unexpected source checkpoint: {checkpoint_path}")

    evaluated = torch.load(data["source_evaluated_path"], map_location="cpu", weights_only=False)
    token_cache = torch.load(data["source_token_cache_path"], map_location="cpu", weights_only=False)

    if not bool(evaluated.get("complete", False)):
        raise RuntimeError("Evaluated-generation artifact is incomplete.")
    if evaluated.get("index_space") != "astar_tokenisation_cache_rows":
        raise RuntimeError("Unexpected evaluated-generation index space.")

    required_evaluated = {
        "cache_idx", "target", "prompt_length", "generated_flat_ids",
        "generated_offsets", "generated_lengths", "expr_target_correct", "expr_parseable",
    }
    missing = required_evaluated - set(evaluated)
    if missing:
        raise KeyError(f"Evaluated artifact is missing fields: {sorted(missing)}")

    required_cache = {"flat_ids", "offsets", "lengths"}
    missing = required_cache - set(token_cache)
    if missing:
        raise KeyError(f"Token cache is missing fields: {sorted(missing)}")

    tok = pm.Tokenizer()
    if token_cache.get("vocab") is not None and list(token_cache["vocab"]) != list(tok.tok2id):
        raise RuntimeError("Token-cache vocabulary differs from the runtime tokenizer.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state") or checkpoint.get("state")
    if state is None:
        raise KeyError("Checkpoint has neither 'model_state' nor 'state'.")

    config = dict(data["model_config"])
    checkpoint_config = checkpoint.get("config") or {}
    for name in ("d_model", "n_heads", "n_layers", "mlp_ratio"):
        if name in checkpoint_config and name in config and checkpoint_config[name] != config[name]:
            raise RuntimeError(f"Checkpoint and probe metadata disagree on {name}.")

    model_max_len = int(data["model_max_len"])
    inferred_max_len = pm.infer_model_max_len_from_checkpoint(checkpoint_path)
    if inferred_max_len != model_max_len:
        raise RuntimeError(f"Checkpoint max length {inferred_max_len} != probe metadata {model_max_len}.")

    model = pm._build_model(config, tok, model_max_len, use_grad_ckpt=False, dropout=0.0)
    pm._load_model_state(model, state)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    n_layers = len(model.blocks)
    n_heads = int(model.blocks[0].n_heads)
    head_dim = int(model.blocks[0].head_dim)
    d_model = n_heads * head_dim
    if (n_layers, n_heads, head_dim, d_model) != (4, 8, 32, 256):
        raise RuntimeError(
            "The exp34 probe contract expects 4 layers, 8 heads, 32 dimensions per head, and d_model=256, "
            f"but the loaded model has {(n_layers, n_heads, head_dim, d_model)}."
        )

    n_records = int(evaluated["cache_idx"].numel())
    for field in ("target", "prompt_length", "generated_offsets", "generated_lengths", "expr_target_correct"):
        if int(evaluated[field].numel()) != n_records:
            raise RuntimeError(f"Evaluated field {field!r} is misaligned.")

    return data, evaluated, token_cache, tok, model, config


def expression_tokens(ids: list[int], tok: Any) -> tuple[int, list[int]]:
    sequence = [int(token_id) for token_id in ids]
    if sequence and sequence[-1] == tok.eos_id:
        sequence = sequence[:-1]

    expr_positions = [index for index, token_id in enumerate(sequence) if token_id == tok.tok2id["EXPR"]]
    if len(expr_positions) != 1:
        raise RuntimeError(f"Expected one EXPR token, found {len(expr_positions)}.")

    expr_index = expr_positions[0]
    answer = sequence[expr_index + 1:]
    if not answer:
        raise RuntimeError("EXPR has no expression tokens.")
    return expr_index, answer


def build_target_bank(
    record_positions: torch.Tensor,
    evaluated: dict[str, Any],
    token_cache: dict[str, Any],
    tok: Any,
    model_max_len: int,
) -> tuple[TargetBank, dict[str, int]]:
    n_records = int(evaluated["cache_idx"].numel())
    offsets = torch.full((n_records,), -1, dtype=torch.int64)
    lengths = torch.zeros(n_records, dtype=torch.int16)
    answer_start = torch.full((n_records,), -1, dtype=torch.int16)
    answer_end = torch.full((n_records,), -1, dtype=torch.int16)
    target_source = torch.full((n_records,), -1, dtype=torch.int8)

    flat_parts: list[torch.Tensor] = []
    flat_buffer: list[int] = []
    flushed_tokens = 0
    generated_count = 0
    reference_count = 0

    for record_pos in tqdm(record_positions.tolist(), desc="Building answer targets", dynamic_ncols=True):
        cache_idx = int(evaluated["cache_idx"][record_pos])
        target = int(evaluated["target"][record_pos])
        prompt_ids = builder.prompt_ids_from_cache(token_cache, cache_idx, tok)
        generated_ids = builder.generated_ids_at(evaluated, record_pos)
        cache_ids = builder.cache_row_ids(token_cache, cache_idx)

        if len(prompt_ids) != int(evaluated["prompt_length"][record_pos]):
            raise RuntimeError(f"Prompt length mismatch at record {record_pos}.")

        generated_expr_index, generated_expr = expression_tokens(generated_ids, tok)
        _, reference_expr = expression_tokens(cache_ids, tok)
        generated_correct = bool(evaluated["expr_target_correct"][record_pos])

        if generated_correct:
            selected_expr = generated_expr
            source = GENERATED_TARGET
            generated_count += 1
        else:
            selected_expr = reference_expr
            source = REFERENCE_TARGET
            reference_count += 1

        expression_text = tok.decode(selected_expr)
        try:
            expression_value = pm.eval_expr_text(expression_text)
        except Exception as exc:
            raise RuntimeError(f"Target expression failed to parse at record {record_pos}: {expression_text!r}") from exc
        if expression_value != target:
            raise RuntimeError(f"Target expression evaluates to {expression_value}, expected {target}, at record {record_pos}.")

        generated_prefix = generated_ids[:generated_expr_index + 1]
        replay_ids = prompt_ids + generated_prefix + selected_expr + [tok.eos_id]
        if len(replay_ids) > model_max_len:
            raise RuntimeError(
                f"Hybrid replay at record {record_pos} has length {len(replay_ids)}, exceeding model_max_len={model_max_len}."
            )

        start = len(prompt_ids) + len(generated_prefix)
        end = start + len(selected_expr)
        if not (0 <= start < end < len(replay_ids)):
            raise RuntimeError(f"Invalid answer span at record {record_pos}.")

        offsets[record_pos] = flushed_tokens + len(flat_buffer)
        lengths[record_pos] = len(replay_ids)
        answer_start[record_pos] = start
        answer_end[record_pos] = end
        target_source[record_pos] = source
        flat_buffer.extend(replay_ids)

        if len(flat_buffer) >= 1_000_000:
            flat_parts.append(torch.tensor(flat_buffer, dtype=torch.int16))
            flushed_tokens += len(flat_buffer)
            flat_buffer.clear()

    if flat_buffer:
        flat_parts.append(torch.tensor(flat_buffer, dtype=torch.int16))
    if not flat_parts:
        raise RuntimeError("Target construction produced no replay sequences.")

    flat_ids = torch.cat(flat_parts)
    bank = TargetBank(flat_ids, offsets, lengths, answer_start, answer_end, target_source)
    summary = {
        "records": int(record_positions.numel()),
        "generated_targets": generated_count,
        "reference_targets": reference_count,
        "packed_tokens": int(flat_ids.numel()),
    }
    return bank, summary


# =============================================================================
# CHECKPOINT AND BATCH PREPARATION
# =============================================================================

def selected_checkpoints(smoke: bool) -> dict[str, tuple[str, ...]]:
    if not smoke:
        return DIRECT_CHECKPOINTS
    return {
        "trace_outcome_operation_decisions": ("operator",),
        "terminal_outcome_controls": ("success_token",),
    }



def smoke_rows(rows: torch.Tensor, labels: torch.Tensor, limit: int) -> torch.Tensor:
    rows = rows.to(torch.int64)
    labels = labels.to(torch.bool)
    selected: list[torch.Tensor] = []
    per_class = max(1, limit // 2)
    for value in (True, False):
        class_rows = rows[labels[rows] == value]
        if class_rows.numel():
            selected.append(class_rows[:per_class])
    combined = torch.cat(selected) if selected else rows[:0]
    if combined.numel() < min(limit, rows.numel()):
        used = torch.zeros(labels.numel(), dtype=torch.bool)
        used[combined] = True
        combined = torch.cat([combined, rows[~used[rows]][:limit - combined.numel()]])
    return combined[:limit]

def build_checkpoint_rows(data: dict[str, Any], smoke: bool) -> list[CheckpointRows]:
    trace_label, trace_known = probe.build_trace_outcome_lookup(data)
    split_to_id = {str(name): int(value) for name, value in data["split_to_id"].items()}
    selected = selected_checkpoints(smoke)
    output: list[CheckpointRows] = []

    for task, checkpoints in selected.items():
        task_spec = probe.TASK_SPECS[task]
        units = probe.build_unit_table(data, task, trace_label, trace_known)
        split_rows = probe.split_unit_rows(units, split_to_id)

        for checkpoint in checkpoints:
            if checkpoint not in units.checkpoints:
                raise KeyError(f"Checkpoint {checkpoint!r} is not available for task {task!r}.")
            checkpoint_index = units.checkpoints.index(checkpoint)
            train_rows = split_rows["train"]
            val_rows = split_rows["val"]
            if smoke:
                train_rows = smoke_rows(train_rows, units.label, 32)
                val_rows = smoke_rows(val_rows, units.label, 16)

            output.append(
                CheckpointRows(
                    task=task,
                    checkpoint=checkpoint,
                    checkpoint_index=checkpoint_index,
                    units=units,
                    train_rows=train_rows,
                    val_rows=val_rows,
                    trace_normalized=task_spec.trace_normalized,
                )
            )

    return output


def example_weights(record_pos: torch.Tensor, trace_normalized: bool) -> torch.Tensor:
    if not trace_normalized:
        return torch.ones(record_pos.numel(), dtype=torch.float32)
    _, inverse, counts = torch.unique(record_pos.to(torch.int64), return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse].to(torch.float32)


def make_example_set(
    checkpoint: CheckpointRows,
    rows: torch.Tensor,
    data: dict[str, Any],
    bank: TargetBank,
) -> ExampleSet:
    units = checkpoint.units
    rows = rows.to(torch.int64)
    record_pos = units.record_pos[rows].to(torch.int64)
    activation_index = units.activation_index[rows, checkpoint.checkpoint_index].to(torch.int64)
    intervention_pos = data["activation_positions"]["absolute_token_idx"][activation_index].to(torch.int64)
    weights = example_weights(record_pos, checkpoint.trace_normalized)
    source = bank.target_source[record_pos].to(torch.int64)

    if bool((bank.offsets[record_pos] < 0).any()):
        raise RuntimeError(f"Checkpoint {checkpoint.task}/{checkpoint.checkpoint} references a missing target.")
    if bool((intervention_pos >= bank.answer_start[record_pos].to(torch.int64)).any()):
        raise RuntimeError(f"An intervention is not strictly before the answer span for {checkpoint.task}/{checkpoint.checkpoint}.")

    expected_correct = source == GENERATED_TARGET
    observed_correct = units.label[rows].to(torch.bool)
    if not torch.equal(expected_correct, observed_correct):
        raise RuntimeError(f"Target-source policy disagrees with expr_target_correct for {checkpoint.task}/{checkpoint.checkpoint}.")

    return ExampleSet(record_pos, intervention_pos, weights, source)


def select_validation_indices(n: int, cap: int, seed: int) -> torch.Tensor:
    if n <= 0:
        raise RuntimeError("Validation set is empty.")
    if cap <= 0 or cap >= n:
        return torch.arange(n, dtype=torch.int64)
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=generator)[:cap]


def train_schedule(n: int, steps: int, batch_size: int, seed: int) -> torch.Tensor:
    if n <= 0:
        raise RuntimeError("Training set is empty.")
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(n, (steps, batch_size), generator=generator, dtype=torch.int64)


def make_batch(bank: TargetBank, examples: ExampleSet, indices: torch.Tensor, tok: Any) -> ReplayBatch:
    indices = indices.to(torch.int64)
    records = examples.record_pos[indices]
    lengths = bank.lengths[records].to(torch.int64)
    max_length = int(lengths.max())
    input_ids = torch.full((indices.numel(), max_length), tok.pad_id, dtype=torch.long)

    for batch_index, record_pos in enumerate(records.tolist()):
        sequence = bank.sequence(record_pos).to(torch.long)
        input_ids[batch_index, :sequence.numel()] = sequence

    return ReplayBatch(
        input_ids=input_ids,
        intervention_pos=examples.intervention_pos[indices].to(torch.int64),
        answer_start=bank.answer_start[records].to(torch.int64),
        answer_end=bank.answer_end[records].to(torch.int64),
        weights=examples.weights[indices].to(torch.float32),
        target_source=examples.target_source[indices].to(torch.int64),
    )


# =============================================================================
# INTERVENTIONS AND LOSS
# =============================================================================

def forward_with_vector(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    intervention_pos: torch.Tensor,
    kind: str,
    layer: int,
    head: int,
    vector: torch.Tensor,
) -> torch.Tensor:
    batch_index = torch.arange(input_ids.shape[0], device=input_ids.device)

    if kind == "attention":
        block = model.blocks[layer]
        start = head * int(block.head_dim)
        end = start + int(block.head_dim)

        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            hidden = inputs[0].clone()
            hidden[batch_index, intervention_pos, start:end] = hidden[batch_index, intervention_pos, start:end] + vector
            return (hidden, *inputs[1:])

        handle = block.out_proj.register_forward_pre_hook(hook)
    elif kind == "block":
        block = model.blocks[layer]

        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
            hidden = output.clone()
            hidden[batch_index, intervention_pos] = hidden[batch_index, intervention_pos] + vector
            return hidden

        handle = block.register_forward_hook(hook)
    elif kind == "final_hidden":
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
            hidden = output.clone()
            hidden[batch_index, intervention_pos] = hidden[batch_index, intervention_pos] + vector
            return hidden

        handle = model.ln_f.register_forward_hook(hook)
    else:
        raise KeyError(f"Unknown intervention kind: {kind!r}")

    try:
        return model(input_ids)
    finally:
        handle.remove()


def answer_loss(logits: torch.Tensor, batch: ReplayBatch) -> tuple[torch.Tensor, torch.Tensor]:
    shifted_logits = logits[:, :-1]
    targets = batch.input_ids[:, 1:]
    per_token = F.cross_entropy(shifted_logits.transpose(1, 2), targets, reduction="none")

    target_positions = torch.arange(1, batch.input_ids.shape[1], device=logits.device).unsqueeze(0)
    mask = (target_positions >= batch.answer_start[:, None]) & (target_positions < batch.answer_end[:, None])
    token_counts = mask.sum(dim=1)
    if bool((token_counts <= 0).any()):
        raise RuntimeError("An answer span contains no target tokens.")

    per_example = (per_token * mask).sum(dim=1) / token_counts
    total = (per_example * batch.weights).sum() / batch.weights.sum().clamp_min(1e-12)
    return total, per_example


# =============================================================================
# LOCATION SPECS AND OPTIMISATION
# =============================================================================

def optimisation_specs(model: torch.nn.Module, smoke: bool) -> list[OptimisationSpec]:
    n_layers = len(model.blocks)
    if n_layers < 2:
        raise RuntimeError("Direct Optimisation requires at least two Transformer layers.")

    n_heads = int(model.blocks[0].n_heads)
    head_dim = int(model.blocks[0].head_dim)
    d_model = n_heads * head_dim
    eligible_layers = list(range(n_layers - 1))

    if smoke:
        last_eligible = eligible_layers[-1]
        attention_pairs = list(dict.fromkeys([(0, 0), (last_eligible, 0)]))
        block_layers = [min(1, last_eligible)]
    else:
        attention_pairs = [(layer, head) for layer in eligible_layers for head in range(n_heads)]
        block_layers = eligible_layers

    specs = [
        OptimisationSpec(
            kind="attention",
            layer=layer,
            head=head,
            feature_dim=head_dim,
            location_id=f"attention_L{layer}_H{head}",
            aliases=(("attention_head", layer * n_heads + head),),
        )
        for layer, head in attention_pairs
    ]

    specs.extend(
        OptimisationSpec(
            kind="block",
            layer=layer,
            head=-1,
            feature_dim=d_model,
            location_id=f"block_L{layer}",
            aliases=(("residual_stream", layer), ("mlp_output", layer)),
        )
        for layer in block_layers
    )
    return specs


def evaluate_vector(
    model: torch.nn.Module,
    bank: TargetBank,
    examples: ExampleSet,
    indices: torch.Tensor,
    tok: Any,
    device: torch.device,
    batch_size: int,
    spec: OptimisationSpec | None = None,
    vector: torch.Tensor | None = None,
) -> dict[str, float]:
    total_loss = 0.0
    total_mass = 0.0
    group_loss = {GENERATED_TARGET: 0.0, REFERENCE_TARGET: 0.0}
    group_mass = {GENERATED_TARGET: 0.0, REFERENCE_TARGET: 0.0}

    with torch.inference_mode():
        for start in range(0, int(indices.numel()), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch = make_batch(bank, examples, batch_indices, tok).to(device)
            logits = model(batch.input_ids) if spec is None else forward_with_vector(
                model, batch.input_ids, batch.intervention_pos, spec.kind, spec.layer, spec.head, vector
            )
            _, per_example = answer_loss(logits, batch)
            weighted = per_example * batch.weights
            total_loss += float(weighted.sum())
            total_mass += float(batch.weights.sum())

            for source in (GENERATED_TARGET, REFERENCE_TARGET):
                mask = batch.target_source == source
                if bool(mask.any()):
                    group_loss[source] += float(weighted[mask].sum())
                    group_mass[source] += float(batch.weights[mask].sum())

    if total_mass <= 0:
        raise RuntimeError("Validation evaluation has zero weight mass.")

    return {
        "loss": total_loss / total_mass,
        "generated_loss": group_loss[GENERATED_TARGET] / group_mass[GENERATED_TARGET] if group_mass[GENERATED_TARGET] else float("nan"),
        "reference_loss": group_loss[REFERENCE_TARGET] / group_mass[REFERENCE_TARGET] if group_mass[REFERENCE_TARGET] else float("nan"),
        "units": int(indices.numel()),
    }


def optimise_vector(
    model: torch.nn.Module,
    bank: TargetBank,
    train_examples: ExampleSet,
    val_examples: ExampleSet,
    schedule: torch.Tensor,
    val_indices: torch.Tensor,
    baseline: dict[str, float],
    spec: OptimisationSpec,
    tok: Any,
    device: torch.device,
    config: dict[str, Any],
    description: str,
) -> dict[str, Any]:
    vector = torch.zeros(spec.feature_dim, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([vector], lr=float(config["learning_rate"]))

    best_vector = vector.detach().clone()
    best_metrics = dict(baseline)
    best_step = 0
    best_train_loss = float("nan")
    stale = 0
    initial_grad_norm = float("nan")
    maximum_grad_norm = 0.0
    steps_completed = 0
    previous_validation_direction = None
    best_step_direction_cosine = float("nan")
    validation_history: list[dict[str, Any]] = []

    progress = tqdm(range(1, int(config["max_steps"]) + 1), desc=description, leave=False, dynamic_ncols=True)
    for step in progress:
        batch = make_batch(bank, train_examples, schedule[step - 1], tok).to(device)
        logits = forward_with_vector(
            model, batch.input_ids, batch.intervention_pos, spec.kind, spec.layer, spec.head, vector
        )
        data_loss, _ = answer_loss(logits, batch)
        regularisation = float(config["l2"]) * vector.square().mean()
        loss = data_loss + regularisation

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if vector.grad is None or not bool(torch.isfinite(vector.grad).all()):
            raise RuntimeError(f"Vector gradient is missing or non-finite for {description}.")

        grad_norm = float(vector.grad.norm())
        if step == 1:
            initial_grad_norm = grad_norm
        maximum_grad_norm = max(maximum_grad_norm, grad_norm)

        torch.nn.utils.clip_grad_norm_([vector], float(config["gradient_clip"]))
        optimizer.step()
        if not bool(torch.isfinite(vector).all()):
            raise RuntimeError(f"Vector became non-finite for {description}.")

        steps_completed = step
        validate = step % int(config["val_every"]) == 0 or step == int(config["max_steps"])
        if validate:
            metrics = evaluate_vector(
                model, bank, val_examples, val_indices, tok, device, int(config["eval_batch_size"]), spec, vector.detach()
            )

            current_vector = vector.detach()
            current_norm = float(current_vector.norm())
            current_direction = current_vector / current_norm if current_norm > ZERO_NORM_EPS else None
            direction_cosine = float("nan")
            if current_direction is not None and previous_validation_direction is not None:
                direction_cosine = float(torch.dot(current_direction, previous_validation_direction))
                direction_cosine = max(-1.0, min(1.0, direction_cosine))
            previous_validation_direction = current_direction.clone() if current_direction is not None else None

            improved = metrics["loss"] < best_metrics["loss"] - MIN_DELTA
            if improved:
                best_metrics = metrics
                best_vector = vector.detach().clone()
                best_step = step
                best_train_loss = float(data_loss.detach())
                best_step_direction_cosine = direction_cosine
                stale = 0
            else:
                stale += 1

            validation_history.append({
                "step": step,
                "loss": metrics["loss"],
                "vector_norm": current_norm,
                "direction_cosine_to_previous_validation": direction_cosine,
                "improved": improved,
            })


            progress.set_postfix(
                train=f"{float(data_loss.detach()):.4f}",
                val=f"{metrics['loss']:.4f}",
                norm=f"{current_norm:.3f}",
            )
            if stale >= int(config["patience"]):
                break

    return {
        "vector": best_vector.cpu(),
        "baseline": baseline,
        "best_validation": best_metrics,
        "best_step": best_step,
        "best_train_loss": best_train_loss,
        "steps_completed": steps_completed,
        "initial_grad_norm": initial_grad_norm,
        "maximum_grad_norm": maximum_grad_norm,
        "best_step_direction_cosine": best_step_direction_cosine,
        "validation_history": validation_history,
    }


# =============================================================================
# GRADIENT AND ARCHIVE AUDITS
# =============================================================================

def audit_gradient_paths(
    model: torch.nn.Module,
    bank: TargetBank,
    checkpoints: list[CheckpointRows],
    data: dict[str, Any],
    tok: Any,
    device: torch.device,
) -> dict[str, Any]:
    n_layers = len(model.blocks)
    d_model = int(model.blocks[0].n_heads) * int(model.blocks[0].head_dim)
    head_dim = int(model.blocks[0].head_dim)
    rows: list[dict[str, Any]] = []

    for checkpoint in tqdm(checkpoints, desc="Auditing gradient paths", dynamic_ncols=True):
        examples = make_example_set(checkpoint, checkpoint.val_rows, data, bank)
        lengths = bank.lengths[examples.record_pos].to(torch.int64)
        count = min(4, examples.n)
        selected = torch.argsort(lengths)[:count]
        batch = make_batch(bank, examples, selected, tok).to(device)

        cases = [("attention", layer, 0, head_dim) for layer in range(n_layers)]
        cases += [("block", layer, -1, d_model) for layer in range(n_layers)]
        cases += [("final_hidden", -1, -1, d_model)]

        for kind, layer, head, feature_dim in cases:
            vector = torch.zeros(feature_dim, dtype=torch.float32, device=device, requires_grad=True)
            logits = forward_with_vector(model, batch.input_ids, batch.intervention_pos, kind, layer, head, vector)
            loss, _ = answer_loss(logits, batch)
            loss.backward()
            grad_norm = float(vector.grad.norm()) if vector.grad is not None else 0.0
            expected_nonzero = kind != "final_hidden" and layer < n_layers - 1

            if not math.isfinite(grad_norm):
                raise RuntimeError("Gradient-path audit produced a non-finite gradient.")
            if expected_nonzero and grad_norm <= GRAD_EPS:
                raise RuntimeError(f"Expected nonzero gradient for {checkpoint.task}/{checkpoint.checkpoint}/{kind}/L{layer}.")
            if not expected_nonzero and grad_norm > GRAD_EPS:
                raise RuntimeError(f"Expected zero gradient for {checkpoint.task}/{checkpoint.checkpoint}/{kind}/L{layer}, got {grad_norm}.")

            rows.append({
                "task": checkpoint.task,
                "checkpoint": checkpoint.checkpoint,
                "kind": kind,
                "layer": layer,
                "head": head,
                "gradient_norm": grad_norm,
                "expected_nonzero": expected_nonzero,
            })

    return {
        "complete": True,
        "threshold": GRAD_EPS,
        "rows": rows,
        "nonzero_cases": sum(bool(row["expected_nonzero"]) for row in rows),
        "zero_cases": sum(not bool(row["expected_nonzero"]) for row in rows),
    }


def external_key_audit(keys: set[str]) -> dict[str, Any]:
    result = {"probe_checked": False, "caa_checked": False}

    if os.path.isfile(probe.RESULTS_PATH):
        probe_results = torch.load(probe.RESULTS_PATH, map_location="cpu", weights_only=False)
        probe_keys = {
            vector_key(entry["task"], entry["checkpoint"], entry["location_name"])
            for entry in probe_results.get("probe_bank", {}).values()
        }
        if not keys <= probe_keys:
            raise RuntimeError("Direct Optimisation keys are not a subset of probe keys.")
        result["probe_checked"] = True

    if os.path.isfile(CAA_PATH):
        caa = torch.load(CAA_PATH, map_location="cpu", weights_only=False)
        caa_keys = set(caa.get("vector_bank", {}))
        if not keys <= caa_keys:
            raise RuntimeError("Direct Optimisation keys are not a subset of CAA keys.")
        result["caa_checked"] = True

    return result

def build_convergence_audit(
    optimisation_summaries: dict[str, dict[str, Any]],
    config: dict[str, Any],) -> dict[str, Any]:
    if not optimisation_summaries:
        raise RuntimeError("No optimisation summaries were saved.")

    summaries = list(optimisation_summaries.values())
    total = len(summaries)
    max_steps = int(config["max_steps"])
    patience = int(config["patience"])
    best_steps = [int(summary["best_step"]) for summary in summaries]
    step_counts = Counter(best_steps)

    zero_summaries = [
        summary for summary in summaries
        if float(summary["vector_norm"]) <= ZERO_NORM_EPS
    ]
    best_step_zero = sum(step == 0 for step in best_steps)
    zero_with_nonzero_best_step = sum(int(summary["best_step"]) != 0 for summary in zero_summaries)
    nonzero_with_zero_best_step = sum(
        int(summary["best_step"]) == 0 and float(summary["vector_norm"]) > ZERO_NORM_EPS
        for summary in summaries
    )
    if zero_with_nonzero_best_step or nonzero_with_zero_best_step:
        raise RuntimeError(
            "Zero-vector and best-step selection are inconsistent: "
            f"zero_with_nonzero_best_step={zero_with_nonzero_best_step}, "
            f"nonzero_with_zero_best_step={nonzero_with_zero_best_step}."
        )

    reached_max = sum(int(summary["steps_completed"]) == max_steps for summary in summaries)
    best_at_max = sum(int(summary["best_step"]) == max_steps for summary in summaries)

    budget_limited = []
    for summary in summaries:
        if int(summary["steps_completed"]) != max_steps:
            continue
        best_step = int(summary["best_step"])
        checks_after_best = sum(
            int(row["step"]) > best_step
            for row in summary.get("validation_history", [])
        )
        if checks_after_best < patience:
            budget_limited.append(summary)

    budget_cosines = [
        float(summary["validation_history"][-1]["direction_cosine_to_previous_validation"])
        for summary in budget_limited
        if summary.get("validation_history")
        and math.isfinite(float(summary["validation_history"][-1]["direction_cosine_to_previous_validation"]))
    ]

    median_budget_cosine = median(budget_cosines) if budget_cosines else float("nan")
    stable_budget_directions = sum(cosine >= DIRECTION_STABLE_COSINE for cosine in budget_cosines)
    budget_fraction = len(budget_limited) / total
    horizon_warning = len(budget_limited) > 0

    if not horizon_warning:
        horizon_assessment = "PASS"
    elif len(budget_cosines) == len(budget_limited) and all(
        cosine >= DIRECTION_STABLE_COSINE for cosine in budget_cosines
    ):
        horizon_assessment = "REVIEW: loss horizon active, directions stable"
    else:
        horizon_assessment = "WARN: horizon active and directions still changing"

    sorted_steps = sorted(best_steps)
    percentile_90_index = max(0, math.ceil(0.90 * total) - 1)
    most_common_step, most_common_count = step_counts.most_common(1)[0]

    return {
        "unique_optimisations": total,
        "best_step_counts": dict(sorted(step_counts.items())),
        "best_step_zero": best_step_zero,
        "median_best_step": float(median(best_steps)),
        "percentile_90_best_step": sorted_steps[percentile_90_index],
        "most_common_best_step": most_common_step,
        "most_common_best_step_count": most_common_count,
        "reached_max_steps": reached_max,
        "best_at_max_steps": best_at_max,
        "budget_limited_runs": len(budget_limited),
        "budget_limited_fraction": budget_fraction,
        "unique_zero_optimisations": len(zero_summaries),
        "zero_best_step_consistency": True,
        "budget_direction_cosine_count": len(budget_cosines),
        "median_budget_direction_cosine": median_budget_cosine,
        "stable_budget_directions": stable_budget_directions,
        "direction_stability_threshold": DIRECTION_STABLE_COSINE,
        "horizon_warning": horizon_warning,
        "horizon_assessment": horizon_assessment,
    }


def audit_archive(
    vector_bank: dict[str, dict[str, Any]],
    optimisation_summaries: dict[str, dict[str, Any]],
    expected_keys: set[str],
    expected_unique_runs: int,
    model_hash_before: str,
    model_hash_after: str,
    smoke: bool,
    config: dict[str, Any],
) -> dict[str, Any]:
    if set(vector_bank) != expected_keys:
        missing = expected_keys - set(vector_bank)
        extra = set(vector_bank) - expected_keys
        raise RuntimeError(f"Vector-key mismatch. Missing={len(missing)}, extra={len(extra)}.")
    if model_hash_before != model_hash_after:
        raise RuntimeError("Frozen model parameters changed during Direct Optimisation.")

    counts = {"attention_head": 0, "residual_stream": 0, "mlp_output": 0, "final_hidden": 0}
    shared_ids: set[str] = set()
    shared_vectors: dict[str, torch.Tensor] = {}
    zero_norm = 0

    for key, entry in vector_bank.items():
        counts[entry["representation_type"]] += 1
        shared_id = entry["shared_optimisation_id"]
        shared_ids.add(shared_id)

        if entry["label_family"] != "final_answer" or entry["label_name"] != "expr_target_correct":
            raise RuntimeError(f"Unexpected label policy for {key}.")
        if entry["checkpoint"] == "expr_final_parenthesis":
            raise RuntimeError("Final-parenthesis Direct Optimisation vector was saved.")
        if int(entry["layer"]) >= 3 or entry["representation_type"] == "final_hidden":
            raise RuntimeError(f"Excluded model location was saved for {key}.")

        vector = entry["vector_raw_space"]
        direction = entry["direction_raw_space_normalized"]
        if int(vector.numel()) != int(entry["feature_dim"]):
            raise RuntimeError(f"Wrong vector dimension for {key}.")
        if not bool(torch.isfinite(vector).all()) or not bool(torch.isfinite(direction).all()):
            raise RuntimeError(f"Non-finite vector values for {key}.")
        if shared_id in shared_vectors and not torch.equal(shared_vectors[shared_id], vector):
            raise RuntimeError(f"Aliased vectors differ for shared optimisation {shared_id}.")
        shared_vectors.setdefault(shared_id, vector)
        zero_norm += int(entry["zero_norm"])

    if len(shared_ids) != expected_unique_runs:
        raise RuntimeError(f"Expected {expected_unique_runs} unique optimisation IDs, got {len(shared_ids)}.")

    external = external_key_audit(set(vector_bank))
    convergence = build_convergence_audit(optimisation_summaries, config)
    return {
        "smoke": smoke,
        "saved_vectors": len(vector_bank),
        "unique_optimisation_runs": len(shared_ids),
        "vectors_by_representation": counts,
        "zero_norm_vectors": zero_norm,
        "model_hash_unchanged": True,
        "test_used_for_selection": False,
        "convergence": convergence,
        **external,
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute final-answer Direct Optimisation vectors through the frozen exp34 A* model.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--val-every", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--val-units", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--l2", type=float)
    parser.add_argument("--gradient-clip", type=float)
    args = parser.parse_args()

    config = {
        name: resolve_arg(getattr(args, name), name, args.smoke)
        for name in FULL_DEFAULTS
    }
    for name, value in config.items():
        if float(value) <= 0 and name != "l2":
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if float(config["l2"]) < 0:
        raise ValueError("--l2 must be non-negative.")

    output_path, partial_path = output_paths(args.smoke)
    if os.path.exists(output_path) and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")
    if args.no_resume and os.path.exists(partial_path):
        os.remove(partial_path)

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    probe.heading("COMPUTE A* DIRECT OPTIMISATION VECTORS")
    probe.print_rows(
        "RUN CONFIGURATION",
        [
            ("Run tag", RUN_TAG),
            ("Mode", "smoke" if args.smoke else "full"),
            ("Device", device),
            ("Output", output_path),
            ("Partial", partial_path),
            ("Batch / eval batch", f"{config['batch_size']} / {config['eval_batch_size']}"),
            ("Maximum steps", config["max_steps"]),
            ("Validation interval / patience", f"{config['val_every']} / {config['patience']}"),
            ("Validation units", config["val_units"]),
            ("Learning rate / L2", f"{config['learning_rate']} / {config['l2']}"),
        ],
    )
    print("=" * probe.WIDTH)

    data, evaluated, token_cache, tok, model, model_config = load_sources(device)
    model_hash_before = model_hash(model)
    checkpoints = build_checkpoint_rows(data, args.smoke)
    record_positions = torch.unique(
        torch.cat([
            checkpoint.units.record_pos[torch.cat([checkpoint.train_rows, checkpoint.val_rows]).to(torch.int64)]
            for checkpoint in checkpoints
        ]).to(torch.int64)
    )
    bank, target_summary = build_target_bank(record_positions, evaluated, token_cache, tok, int(data["model_max_len"]))
    specs = optimisation_specs(model, args.smoke)

    expected_keys: list[str] = []
    for checkpoint in checkpoints:
        for spec in specs:
            for representation_type, location_index in spec.aliases:
                location = probe.location_info(representation_type, location_index)
                expected_keys.append(vector_key(checkpoint.task, checkpoint.checkpoint, location["location_name"]))
    if len(expected_keys) != len(set(expected_keys)):
        raise RuntimeError("Expected Direct Optimisation vector keys are not unique.")
    expected_key_to_id = {key: index for index, key in enumerate(expected_keys)}
    expected_unique_runs = len(checkpoints) * len(specs)
    if not args.smoke and (len(checkpoints), len(specs), len(expected_keys), expected_unique_runs) != (10, 27, 300, 270):
        raise RuntimeError(
            "Full Direct Optimisation contract should contain 10 checkpoints, 27 unique location jobs, "
            f"300 saved entries, and 270 total runs, got {(len(checkpoints), len(specs), len(expected_keys), expected_unique_runs)}."
        )

    job_alias_keys = {}
    for checkpoint in checkpoints:
        for spec in specs:
            job_id = f"{checkpoint.task}:{checkpoint.checkpoint}:{spec.location_id}"
            job_alias_keys[job_id] = [
                vector_key(checkpoint.task, checkpoint.checkpoint, probe.location_info(kind, index)["location_name"])
                for kind, index in spec.aliases
            ]

    run_config = {
        "run_tag": RUN_TAG,
        "smoke": args.smoke,
        "source_probe_dataset_path": probe.DATASET_PATH,
        "source_evaluated_path": data["source_evaluated_path"],
        "source_token_cache_path": data["source_token_cache_path"],
        "source_checkpoint_path": data["source_checkpoint_path"],
        "direct_checkpoints": selected_checkpoints(args.smoke),
        "optimisation_specs": [spec.location_id for spec in specs],
        "target_policy": "generated expression when correct, otherwise A* reference expression",
        "answer_span_policy": "tokens after EXPR through end of expression",
        "config": config,
        "seed": SEED,
    }

    vector_bank: dict[str, dict[str, Any]] = {}
    optimisation_summaries: dict[str, Any] = {}
    task_summaries: dict[str, Any] = {}
    completed_ids: set[str] = set()

    if os.path.exists(partial_path):
        partial = torch.load(partial_path, map_location="cpu", weights_only=False)
        if partial.get("run_config") != run_config:
            raise RuntimeError("Partial artifact was created with a different configuration. Use --no-resume to discard it.")
        vector_bank = dict(partial.get("vector_bank", {}))
        optimisation_summaries = dict(partial.get("optimisation_summaries", {}))
        task_summaries = dict(partial.get("task_summaries", {}))
        completed_ids = set(partial.get("completed_ids", []))
        if not completed_ids <= set(job_alias_keys):
            raise RuntimeError("Partial artifact contains unknown optimisation IDs.")
        if not set(vector_bank) <= set(expected_keys):
            raise RuntimeError("Partial artifact contains unknown vector keys.")
        for completed_id in completed_ids:
            if not set(job_alias_keys[completed_id]) <= set(vector_bank):
                raise RuntimeError(f"Partial artifact is missing aliases for completed optimisation {completed_id}.")
        print(f"Resuming {len(completed_ids):,} completed optimisation runs.")

    gradient_audit = audit_gradient_paths(model, bank, checkpoints, data, tok, device)
    print("Gradient-path audit: PASS")

    total_progress = tqdm(total=expected_unique_runs, initial=len(completed_ids), desc="Direct optimisation jobs", dynamic_ncols=True)

    for checkpoint in checkpoints:
        pending = [spec for spec in specs if f"{checkpoint.task}:{checkpoint.checkpoint}:{spec.location_id}" not in completed_ids]
        if not pending:
            continue

        train_examples = make_example_set(checkpoint, checkpoint.train_rows, data, bank)
        val_examples = make_example_set(checkpoint, checkpoint.val_rows, data, bank)
        seed = stable_seed(checkpoint.task, checkpoint.checkpoint)
        schedule = train_schedule(train_examples.n, int(config["max_steps"]), int(config["batch_size"]), seed)
        val_indices = select_validation_indices(val_examples.n, int(config["val_units"]), seed + 1)
        if checkpoint.trace_normalized:
            val_examples.weights[val_indices] = example_weights(
                val_examples.record_pos[val_indices],
                True,
            )
        val_lengths = bank.lengths[val_examples.record_pos[val_indices]].to(torch.int64)
        val_indices = val_indices[torch.argsort(val_lengths)]
        baseline = evaluate_vector(
            model, bank, val_examples, val_indices, tok, device, int(config["eval_batch_size"])
        )

        task_summaries[f"{checkpoint.task}:{checkpoint.checkpoint}"] = {
            "train_units": train_examples.n,
            "val_units": val_examples.n,
            "validation_units_used": int(val_indices.numel()),
            "train_generated_targets": int((train_examples.target_source == GENERATED_TARGET).sum()),
            "train_reference_targets": int((train_examples.target_source == REFERENCE_TARGET).sum()),
            "baseline_validation": baseline,
        }

        for spec in pending:
            optimisation_id = f"{checkpoint.task}:{checkpoint.checkpoint}:{spec.location_id}"
            result = optimise_vector(
                model=model,
                bank=bank,
                train_examples=train_examples,
                val_examples=val_examples,
                schedule=schedule,
                val_indices=val_indices,
                baseline=baseline,
                spec=spec,
                tok=tok,
                device=device,
                config=config,
                description=f"{checkpoint.checkpoint} | {spec.location_id}",
            )

            vector = result["vector"].float()
            norm = float(vector.norm())
            direction = vector / norm if norm > ZERO_NORM_EPS else torch.zeros_like(vector)

            for representation_type, location_index in spec.aliases:
                location = probe.location_info(representation_type, location_index)
                key = vector_key(checkpoint.task, checkpoint.checkpoint, location["location_name"])
                vector_bank[key] = {
                    "vector_id": f"optimised_{expected_key_to_id[key]:06d}",
                    "vector_key": key,
                    "method": "direct_optimisation",
                    "task": checkpoint.task,
                    "source_task": probe.source_task_name(checkpoint.task),
                    "label_family": "final_answer",
                    "label_name": "expr_target_correct",
                    "checkpoint": checkpoint.checkpoint,
                    "checkpoint_index": checkpoint.checkpoint_index,
                    **location,
                    "location_index": location_index,
                    "global_location_index": probe.REPRESENTATION_SPECS[representation_type].global_offset + location_index,
                    "feature_dim": spec.feature_dim,
                    "trace_normalized": checkpoint.trace_normalized,
                    "train_units": train_examples.n,
                    "val_units": val_examples.n,
                    "validation_units_used": int(val_indices.numel()),
                    "train_generated_target_units": int((train_examples.target_source == GENERATED_TARGET).sum()),
                    "train_reference_target_units": int((train_examples.target_source == REFERENCE_TARGET).sum()),
                    "baseline_val_loss": result["baseline"]["loss"],
                    "baseline_val_generated_loss": result["baseline"]["generated_loss"],
                    "baseline_val_reference_loss": result["baseline"]["reference_loss"],
                    "best_val_loss": result["best_validation"]["loss"],
                    "best_val_generated_loss": result["best_validation"]["generated_loss"],
                    "best_val_reference_loss": result["best_validation"]["reference_loss"],
                    "best_train_loss": result["best_train_loss"],
                    "best_step": result["best_step"],
                    "steps_completed": result["steps_completed"],
                    "initial_gradient_norm": result["initial_grad_norm"],
                    "maximum_gradient_norm": result["maximum_grad_norm"],
                    "best_step_direction_cosine": result["best_step_direction_cosine"],
                    "vector_raw_space": vector.clone(),
                    "vector_norm": norm,
                    "direction_raw_space_normalized": direction.clone(),
                    "direction_sign": "increases selected valid final-expression likelihood",
                    "zero_norm": norm <= ZERO_NORM_EPS,
                    "shared_optimisation_id": optimisation_id,
                }

            optimisation_summaries[optimisation_id] = {
                "aliases": [vector_key(checkpoint.task, checkpoint.checkpoint, probe.location_info(kind, index)["location_name"]) for kind, index in spec.aliases],
                "best_step": result["best_step"],
                "steps_completed": result["steps_completed"],
                "baseline_val_loss": result["baseline"]["loss"],
                "best_val_loss": result["best_validation"]["loss"],
                "vector_norm": norm,
                "maximum_gradient_norm": result["maximum_grad_norm"],
                "best_step_direction_cosine": result["best_step_direction_cosine"],
                "validation_history": result["validation_history"],
            }
            completed_ids.add(optimisation_id)

            partial_payload = {
                "format_version": 1,
                "complete": False,
                "run_config": run_config,
                "target_summary": target_summary,
                "gradient_path_audit": gradient_audit,
                "task_summaries": task_summaries,
                "vector_bank": vector_bank,
                "optimisation_summaries": optimisation_summaries,
                "completed_ids": sorted(completed_ids),
            }
            probe.safe_torch_save(partial_payload, partial_path)
            total_progress.update(1)

        del train_examples, val_examples, schedule, val_indices
        probe.clear_memory(device)

    total_progress.close()
    model_hash_after = model_hash(model)
    audit = audit_archive(
        vector_bank=vector_bank,
        optimisation_summaries=optimisation_summaries,
        expected_keys=set(expected_keys),
        expected_unique_runs=expected_unique_runs,
        model_hash_before=model_hash_before,
        model_hash_after=model_hash_after,
        smoke=args.smoke,
        config=config,
    )

    payload = {
        "format_version": 1,
        "complete": True,
        "method": "direct_optimisation",
        "run_tag": RUN_TAG,
        "source_probe_dataset_path": probe.DATASET_PATH,
        "source_evaluated_path": data["source_evaluated_path"],
        "source_token_cache_path": data["source_token_cache_path"],
        "source_checkpoint_path": data["source_checkpoint_path"],
        "model_config": model_config,
        "model_max_len": int(data["model_max_len"]),
        "target_policy": run_config["target_policy"],
        "answer_span_policy": run_config["answer_span_policy"],
        "loss_policy": "mean answer-token cross-entropy per example, weighted equally per trace, plus mean-square vector regularisation",
        "run_config": run_config,
        "target_summary": target_summary,
        "eligible_checkpoints": selected_checkpoints(args.smoke),
        "eligible_locations": [spec.location_id for spec in specs],
        "excluded_locations": {
            "attention": [f"attention_L{len(model.blocks) - 1}_H{head}" for head in range(int(model.blocks[0].n_heads))],
            "residual": [f"residual_L{len(model.blocks) - 1}"],
            "mlp": [f"mlp_L{len(model.blocks) - 1}"],
            "final_hidden": ["final_hidden"],
        },
        "gradient_path_audit": gradient_audit,
        "task_summaries": task_summaries,
        "optimisation_summaries": optimisation_summaries,
        "vector_bank": vector_bank,
        "audit": audit,
    }

    probe.safe_torch_save(payload, output_path)
    if os.path.exists(partial_path):
        os.remove(partial_path)

    convergence = audit["convergence"]
    budget_cosine = convergence["median_budget_direction_cosine"]
    budget_cosine_text = f"{budget_cosine:.6f}" if math.isfinite(budget_cosine) else "n/a"
    overall_status = "PASS" if not convergence["horizon_warning"] else "PASS WITH CONVERGENCE REVIEW"

    probe.heading("FINAL DIRECT OPTIMISATION AUDIT")
    probe.print_rows(
        "AUDIT RESULTS",
        [
            ("Saved vectors", audit["saved_vectors"]),
            ("Unique optimisation runs", audit["unique_optimisation_runs"]),
            ("Attention vectors", audit["vectors_by_representation"]["attention_head"]),
            ("Residual vectors", audit["vectors_by_representation"]["residual_stream"]),
            ("MLP vectors", audit["vectors_by_representation"]["mlp_output"]),
            ("Final-hidden vectors", audit["vectors_by_representation"]["final_hidden"]),
            ("Zero-norm vectors", audit["zero_norm_vectors"]),
            ("Model unchanged", "yes"),
            ("Test split used", "no"),
            ("Probe-key check", "PASS" if audit["probe_checked"] else "not available"),
            ("CAA-key check", "PASS" if audit["caa_checked"] else "not available"),
            ("Output", output_path),
        ],
    )

    probe.print_rows(
        "OPTIMISATION CONVERGENCE",
        [
            ("Median / p90 best step", f"{convergence['median_best_step']} / {convergence['percentile_90_best_step']}"),
            (
                "Most common best step",
                f"{convergence['most_common_best_step']} ({convergence['most_common_best_step_count']} runs)",
            ),
            (
                "Reached maximum steps",
                f"{convergence['reached_max_steps']}/{convergence['unique_optimisations']}",
            ),
            (
                "Best step = maximum",
                f"{convergence['best_at_max_steps']}/{convergence['unique_optimisations']}",
            ),
            (
                "Budget-limited runs",
                f"{convergence['budget_limited_runs']}/{convergence['unique_optimisations']} "
                f"({100 * convergence['budget_limited_fraction']:.1f}%)",
            ),
            ("Unique zero optimisations", convergence["unique_zero_optimisations"]),
            ("Best step = 0", convergence["best_step_zero"]),
            ("Zero / best-step consistency", "PASS"),
            ("Budget direction cosine median", budget_cosine_text),
            (
                "Stable budget directions",
                f"{convergence['stable_budget_directions']}/{convergence['budget_direction_cosine_count']} "
                f"(cosine >= {DIRECTION_STABLE_COSINE})",
            ),
            ("Horizon assessment", convergence["horizon_assessment"]),
            ("Best-step counts", str(convergence["best_step_counts"])),
            ("OVERALL STATUS", overall_status),
        ],
    )
    print("=" * probe.WIDTH)


if __name__ == "__main__":
    main()
