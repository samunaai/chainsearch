#!/usr/bin/env python3
"""
compute_linear_probe_vectors.py

Compute linear-probe vectors and evaluate the corresponding A* probe tasks.

The current probe sweep contains 779 final classifiers

    19 task/checkpoint combinations
    x
    (32 attention heads + 4 residual layers + 1 final hidden representation + 4 MLP layers)

The classifiers are trained in 24 vectorized GPU blocks

    6 task groups x 4 representation families

The script saves

    1. A 779-row master metrics table
    2. A probe bank containing fitted parameters and steering directions
    3. One compact validation/test prediction cache per task
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm.auto import tqdm


# =============================================================================
# CONFIGURATION
# =============================================================================

RUN_TAG = "exp34"
SEED = 42

ASTAR_DIR = "/workspace/sam/minireason_archives/data_pts/astar"
PROBING_DIR = os.path.join(ASTAR_DIR, "probing")

DATASET_PATH = os.path.join(PROBING_DIR, f"{RUN_TAG}_astar_probe_dataset.pt")
RESULTS_PATH = os.path.join(PROBING_DIR, f"{RUN_TAG}_astar_probe_results.pt")
METRICS_CSV_PATH = os.path.join(PROBING_DIR, f"{RUN_TAG}_astar_probe_metrics.csv")
PREDICTION_DIR = os.path.join(PROBING_DIR, f"{RUN_TAG}_astar_probe_predictions")

OVERWRITE = True

L2_GRID = (0.0, 1e-4, 1e-3, 1e-2)
LEARNING_RATE = 3e-2
MAX_EPOCHS = 25
PATIENCE = 5
MIN_DELTA = 1e-5

TRAIN_BATCH_SIZE = 32768
EVAL_BATCH_SIZE = 65536
FEATURE_COPY_BATCH_SIZE = 32768
STD_EPS = 1e-5


@dataclass(frozen=True)
class TaskSpec:
    checkpoints: tuple[str, ...]
    trace_normalized: bool = False
    source_task: str | None = None
    relabel_with_trace_outcome: bool = False

@dataclass(frozen=True)
class RepresentationSpec:
    locations: int
    feature_dim: int
    global_offset: int


OPERATION_CHECKPOINTS = (
    "pre_first_operand",
    "first_operand_end",
    "operator",
    "second_operand_end",
    "result_end",
    "resulting_state_semicolon",
)

BACK_CHECKPOINTS = (
    "back_token",
    "restored_state_semicolon",
)

TASK_SPECS = {
    "operation_reachability": TaskSpec(
        checkpoints=OPERATION_CHECKPOINTS,
    ),
    "back_reachability": TaskSpec(
        checkpoints=BACK_CHECKPOINTS,
    ),
    "trace_outcome_trajectory": TaskSpec(
        checkpoints=("pre_back",),
        trace_normalized=True,
    ),
    "terminal_outcome_controls": TaskSpec(
        checkpoints=(
            "success_token",
            "expr_final_parenthesis",
        ),
    ),

    # Reuse the already-cached operation activations, but replace the
    # reachability target with the eventual trace-correctness target.
    "trace_outcome_operation_decisions": TaskSpec(
        checkpoints=OPERATION_CHECKPOINTS,
        trace_normalized=True,
        source_task="operation_reachability",
        relabel_with_trace_outcome=True,
    ),

    # Reuse the already-cached BACK/restoration activations with the
    # eventual trace-correctness target.
    "trace_outcome_back_decisions": TaskSpec(
        checkpoints=BACK_CHECKPOINTS,
        trace_normalized=True,
        source_task="back_reachability",
        relabel_with_trace_outcome=True,
    ),
}

REPRESENTATION_SPECS = {
    "attention_head": RepresentationSpec(locations=32, feature_dim=32, global_offset=0),
    "residual_stream": RepresentationSpec(locations=4, feature_dim=256, global_offset=32),
    "final_hidden": RepresentationSpec(locations=1, feature_dim=256, global_offset=36),
    "mlp_output": RepresentationSpec(locations=4, feature_dim=256, global_offset=37),
}

REPRESENTATION_LOCATION_NAMES = (
    *(f"attention_L{layer}_H{head}" for layer in range(4) for head in range(8)),
    *(f"residual_L{layer}" for layer in range(4)),
    "final_hidden",
    *(f"mlp_L{layer}" for layer in range(4)),
)

EXPECTED_CHECKPOINTS = sum(len(spec.checkpoints) for spec in TASK_SPECS.values())
EXPECTED_PROBES = EXPECTED_CHECKPOINTS * len(REPRESENTATION_LOCATION_NAMES)
EXPECTED_PROBES_BY_TYPE = {
    "attention_head": EXPECTED_CHECKPOINTS * 32,
    "residual_stream": EXPECTED_CHECKPOINTS * 4,
    "final_hidden": EXPECTED_CHECKPOINTS,
    "mlp_output": EXPECTED_CHECKPOINTS * 4,
}


# =============================================================================
# COMPACT DATA CONTAINERS
# =============================================================================

@dataclass
class UnitTable:
    """One row per operation, BACK transition, pre-BACK event, or trace."""

    task: str
    checkpoints: tuple[str, ...]
    activation_index: torch.Tensor  # [units, checkpoints]
    label: torch.Tensor
    split_id: torch.Tensor
    family_id: torch.Tensor
    source_row: torch.Tensor
    record_pos: torch.Tensor
    cache_idx: torch.Tensor
    target: torch.Tensor

    @property
    def n_units(self) -> int:
        return int(self.label.numel())

    @property
    def n_checkpoints(self) -> int:
        return len(self.checkpoints)


@dataclass
class ActivationBank:
    attention: torch.Tensor
    residual: torch.Tensor
    final_hidden: torch.Tensor
    mlp: torch.Tensor


# =============================================================================
# GENERAL HELPERS
# =============================================================================

WIDTH = 104


def heading(title: str, char: str = "=") -> None:
    print(char * WIDTH)
    print(title.center(WIDTH))
    print(char * WIDTH)


def print_rows(title: str, rows: Iterable[tuple[str, Any]], indent: int = 2) -> None:
    print(title)
    rows = list(rows)
    label_width = max((len(label) for label, _ in rows), default=0)
    prefix = " " * indent
    for label, value in rows:
        print(f"{prefix}{label:<{label_width}}  {value}")

def print_table(
    headers: tuple[str, ...],
    rows: list[tuple[Any, ...]],
    right_aligned: frozenset[int] = frozenset(),
) -> None:
    """Print a compact table with widths inferred from its contents."""

    text_rows = [tuple(str(value) for value in row) for row in rows]
    widths = [
        max([len(headers[index]), *(len(row[index]) for row in text_rows)])
        for index in range(len(headers))
    ]

    def render(row: tuple[str, ...]) -> str:
        return "  ".join(
            f"{value:>{widths[index]}}"
            if index in right_aligned
            else f"{value:<{widths[index]}}"
            for index, value in enumerate(row)
        ).rstrip()

    print(render(headers))
    print(render(tuple("-" * width for width in widths)))

    for row in text_rows:
        print(render(row))


def safe_torch_save(obj: Any, path: str) -> None:
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")

    with open(tmp_path, "wb") as handle:
        torch.save(obj, handle)
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(tmp_path, final_path)


def safe_csv_save(rows: list[dict[str, Any]], path: str) -> None:
    if not rows:
        raise RuntimeError("Cannot save an empty metrics table.")

    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")

    with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    os.replace(tmp_path, final_path)


def require_fields(table: dict[str, Any], fields: Iterable[str], name: str) -> None:
    missing = set(fields) - set(table)
    if missing:
        raise KeyError(f"{name} is missing fields: {sorted(missing)}")


def bytes_to_gib(n_bytes: int) -> float:
    return n_bytes / (1024**3)


def clear_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

def source_task_name(task: str) -> str:
    spec = TASK_SPECS[task]
    return spec.source_task or task


def task_label_name(
    data: dict[str, Any],
    task: str,
) -> str:
    spec = TASK_SPECS[task]

    if spec.relabel_with_trace_outcome:
        return "expr_target_correct"

    return str(
        data["task_label_names"][source_task_name(task)]
    )



def location_info(representation_type: str, location: int) -> dict[str, Any]:
    if representation_type == "attention_head":
        layer, head = divmod(location, 8)
        return {
            "representation_type": representation_type,
            "location_name": f"attention_L{layer}_H{head}",
            "layer": layer,
            "head": head,
        }

    if representation_type == "residual_stream":
        return {
            "representation_type": representation_type,
            "location_name": f"residual_L{location}",
            "layer": location,
            "head": -1,
        }

    if representation_type == "final_hidden":
        return {
            "representation_type": representation_type,
            "location_name": "final_hidden",
            "layer": -1,
            "head": -1,
        }

    if representation_type == "mlp_output":
        return {
            "representation_type": representation_type,
            "location_name": f"mlp_L{location}",
            "layer": location,
            "head": -1,
        }

    raise KeyError(f"Unknown representation type: {representation_type!r}")


# =============================================================================
# METRICS
# =============================================================================


def choose_validation_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Return the logit threshold with maximum validation balanced accuracy."""

    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    balanced = 0.5 * (true_positive_rate + 1.0 - false_positive_rate)
    threshold = float(thresholds[int(np.nanargmax(balanced))])

    if math.isfinite(threshold):
        return threshold

    maximum = float(np.max(scores))
    return maximum + max(1e-6, abs(maximum) * 1e-6)


def calculate_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    positive_auprc = average_precision_score(labels, scores)
    negative_auprc = average_precision_score(1 - labels, -scores)
    prevalence = float(labels.mean())

    return {
        "examples": int(labels.size),
        "positive": int(labels.sum()),
        "negative": int(labels.size - labels.sum()),
        "positive_prevalence": prevalence,
        "majority_accuracy": max(prevalence, 1.0 - prevalence),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "auroc": float(roc_auc_score(labels, scores)),
        "positive_auprc": float(positive_auprc),
        "negative_auprc": float(negative_auprc),
        "macro_auprc": float(0.5 * (positive_auprc + negative_auprc)),
        "positive_recall": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "negative_recall": float(recall_score(labels, predictions, pos_label=0, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "positive_f1": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def prefix_metrics(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{name}": value for name, value in values.items()}


# =============================================================================
# LOAD AND AUDIT SOURCE DATA
# =============================================================================


def load_probe_dataset() -> dict[str, Any]:
    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(DATASET_PATH)

    data = torch.load(DATASET_PATH, map_location="cpu", weights_only=False)

    if not bool(data.get("complete", False)):
        raise RuntimeError("Probe dataset is not marked complete.")
    if int(data.get("format_version", -1)) < 3:
        raise RuntimeError(
            "Probe dataset format version must be at least 3."
        )
    if data.get("run_tag") != RUN_TAG:
        raise RuntimeError(f"Dataset run tag {data.get('run_tag')!r} != {RUN_TAG!r}.")
    if data.get("activation_storage") != "sharded":
        raise RuntimeError("Expected sharded activation storage.")

    require_fields(
        data,
        {
            "examples",
            "activation_positions",
            "input_family_table",
            "activation_shard_dir",
            "activation_shards",
            "attention_head_output_pre_projection_shape",
            "mlp_output_before_residual_shape",
            "residual_stream_after_block_shape",
            "final_layer_normalized_hidden_shape",
            "task_to_id",
            "checkpoint_to_id",
            "task_label_names",
            "split_to_id",
            "activation_audit",
            "shard_audit",
        },
        "probe dataset",
    )
    return data


def audit_probe_dataset(data: dict[str, Any]) -> None:
    examples = data["examples"]
    families = data["input_family_table"]

    require_fields(
        examples,
        {
            "task_id",
            "checkpoint_id",
            "source_row",
            "record_pos",
            "cache_idx",
            "target",
            "label",
            "split_id",
            "family_id",
            "activation_index",
        },
        "examples",
    )
    require_fields(families, {"values", "split_id"}, "input family table")

    n_examples = int(examples["task_id"].numel())
    n_positions = int(data["activation_positions"]["record_pos"].numel())

    for name, value in examples.items():
        if torch.is_tensor(value) and int(value.numel()) != n_examples:
            raise RuntimeError(f"Example field {name!r} is misaligned.")

    activation_index = examples["activation_index"].to(torch.int64)
    if bool((activation_index < 0).any()) or bool((activation_index >= n_positions).any()):
        raise RuntimeError("An example has an invalid activation index.")

    family_id = examples["family_id"].to(torch.int64)
    family_split = families["split_id"].to(torch.int64)
    if not torch.equal(examples["split_id"].to(torch.int64), family_split[family_id]):
        raise RuntimeError("An input family crosses probe train/validation/test splits.")

    if not bool(data["activation_audit"].get("all_values_finite", False)):
        raise RuntimeError("The builder finite-value audit did not pass.")
    if not bool(data["shard_audit"].get("complete", False)):
        raise RuntimeError("The builder shard audit did not pass.")

    print_rows(
        "SOURCE AUDIT",
        [
            ("Logical examples", f"{n_examples:,}"),
            ("Activation positions", f"{n_positions:,}"),
            ("Every activation index valid", "yes"),
            ("Input-family split disjointness", "yes"),
            ("Builder activation/shard audits", "PASS"),
        ],
    )


def build_trace_outcome_lookup(
    data: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return dense trace-level final-correctness labels.

    known[record_pos] says whether the trace has an eligible terminal
    final-expression row. label[record_pos] is expr_target_correct.
    """

    examples = data["examples"]

    terminal_task_id = int(
        data["task_to_id"]["terminal_outcome_controls"]
    )
    final_expr_checkpoint_id = int(
        data["checkpoint_to_id"]["expr_final_parenthesis"]
    )

    rows = torch.nonzero(
        (
            examples["task_id"].to(torch.int64)
            == terminal_task_id
        )
        & (
            examples["checkpoint_id"].to(torch.int64)
            == final_expr_checkpoint_id
        ),
        as_tuple=False,
    ).flatten()

    if not rows.numel():
        raise RuntimeError(
            "No terminal final-expression labels were found."
        )

    record_pos = examples["record_pos"][rows].to(torch.int64)
    outcome = examples["label"][rows].to(torch.bool)

    if int(torch.unique(record_pos).numel()) != int(
        record_pos.numel()
    ):
        raise RuntimeError(
            "A trace has multiple final-expression outcome rows."
        )

    position_records = data[
        "activation_positions"
    ]["record_pos"].to(torch.int64)

    n_records = int(position_records.max().item()) + 1

    label = torch.zeros(n_records, dtype=torch.bool)
    known = torch.zeros(n_records, dtype=torch.bool)

    label[record_pos] = outcome
    known[record_pos] = True

    if not bool(label[known].any()) or not bool(
        (~label[known]).any()
    ):
        raise RuntimeError(
            "Trace-outcome lookup does not contain both classes."
        )

    return label, known


def load_activation_bank(data: dict[str, Any]) -> ActivationBank:
    attention_shape = tuple(data["attention_head_output_pre_projection_shape"])
    mlp_shape = tuple(data["mlp_output_before_residual_shape"])
    residual_shape = tuple(data["residual_stream_after_block_shape"])
    final_shape = tuple(data["final_layer_normalized_hidden_shape"])

    if attention_shape[1:] != (4, 8, 32):
        raise RuntimeError(f"Unexpected attention shape: {attention_shape}")
    if mlp_shape[1:] != (4, 256):
        raise RuntimeError(f"Unexpected MLP shape: {mlp_shape}")
    if residual_shape[1:] != (4, 256):
        raise RuntimeError(f"Unexpected residual shape: {residual_shape}")
    if final_shape[1:] != (256,):
        raise RuntimeError(f"Unexpected final-hidden shape: {final_shape}")

    n_positions = attention_shape[0]
    if not (
        mlp_shape[0]
        == residual_shape[0]
        == final_shape[0]
        == n_positions
    ):
        raise RuntimeError(
            "Activation families have different position counts."
        )

    attention = torch.empty(attention_shape, dtype=torch.float16)
    mlp = torch.empty(mlp_shape, dtype=torch.float16)
    residual = torch.empty(residual_shape, dtype=torch.float16)
    final_hidden = torch.empty(final_shape, dtype=torch.float16)

    shard_dir = Path(data["activation_shard_dir"])
    expected_start = 0

    for shard_info in tqdm(
        data["activation_shards"],
        desc="Loading activation shards",
        dynamic_ncols=True,
    ):
        start = int(shard_info["start"])
        end = int(shard_info["end"])
        if start != expected_start or end <= start:
            raise RuntimeError("Activation shards contain a gap or overlap.")

        shard_path = shard_dir / str(shard_info["file"])
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        if int(shard["start"]) != start or int(shard["end"]) != end:
            raise RuntimeError(f"Shard range mismatch in {shard_path}")

        shard_attention = shard["attention_head_output_pre_projection"]
        shard_mlp = shard["mlp_output_before_residual"]
        shard_residual = shard["residual_stream_after_block"]
        shard_final = shard["final_layer_normalized_hidden"]

        for name, tensor in (
            ("attention", shard_attention),
            ("MLP", shard_mlp),
            ("residual", shard_residual),
            ("final hidden", shard_final),
        ):
            if tensor.dtype != torch.float16:
                raise RuntimeError(f"{name} shard has dtype {tensor.dtype}, expected float16.")
            if not bool(torch.isfinite(tensor).all()):
                raise RuntimeError(f"{name} shard contains non-finite values.")

        attention[start:end].copy_(shard_attention)
        mlp[start:end].copy_(shard_mlp)
        residual[start:end].copy_(shard_residual)
        final_hidden[start:end].copy_(shard_final)
        expected_start = end

        del shard, shard_attention, shard_mlp, shard_residual, shard_final

    if expected_start != n_positions:
        raise RuntimeError("Activation shards do not cover every activation position.")

    return ActivationBank(attention=attention, residual=residual, final_hidden=final_hidden, mlp=mlp)

# =============================================================================
# PAIR CHECKPOINTS INTO LOGICAL UNITS
# =============================================================================


def build_unit_table(
    data: dict[str, Any],
    task: str,
    trace_outcome_label: torch.Tensor,
    trace_outcome_known: torch.Tensor,
) -> UnitTable:
    examples = data["examples"]
    task_spec = TASK_SPECS[task]
    source_task = source_task_name(task)
    task_id = int(data["task_to_id"][source_task])
    checkpoints = task_spec.checkpoints
    checkpoint_ids = [int(data["checkpoint_to_id"][name]) for name in checkpoints]
    n_checkpoints = len(checkpoints)

    task_rows = torch.nonzero(
        examples["task_id"].to(torch.int64) == task_id,
        as_tuple=False,
    ).flatten()
    if not task_rows.numel():
        raise RuntimeError(f"Task {task!r} has no examples.")

    checkpoint_id = examples["checkpoint_id"][task_rows].to(torch.int64)
    source_row = examples["source_row"][task_rows].to(torch.int64)

    max_checkpoint_id = max(int(value) for value in data["checkpoint_to_id"].values())
    local_lookup = torch.full((max_checkpoint_id + 1,), -1, dtype=torch.int64)
    for local_id, checkpoint in enumerate(checkpoint_ids):
        local_lookup[checkpoint] = local_id

    local_checkpoint = local_lookup[checkpoint_id]
    if bool((local_checkpoint < 0).any()):
        raise RuntimeError(f"Task {task} contains an unexpected checkpoint.")

    key = source_row * n_checkpoints + local_checkpoint
    if int(torch.unique(key).numel()) != int(key.numel()):
        raise RuntimeError(f"Task {task} contains duplicate source/checkpoint rows.")
    if int(key.numel()) % n_checkpoints:
        raise RuntimeError(f"Task {task} cannot be paired across all checkpoints.")

    order = torch.argsort(key)
    paired_rows = task_rows[order].reshape(-1, n_checkpoints)
    paired_source = source_row[order].reshape(-1, n_checkpoints)
    paired_local = local_checkpoint[order].reshape(-1, n_checkpoints)

    expected_local = torch.arange(n_checkpoints, dtype=torch.int64).expand_as(paired_local)
    if not torch.equal(paired_local, expected_local):
        raise RuntimeError(f"Task {task} has incomplete checkpoint pairing.")
    if not torch.equal(paired_source, paired_source[:, :1].expand_as(paired_source)):
        raise RuntimeError(f"Task {task} pairs different underlying source rows.")


    record_pos_matrix = examples[
        "record_pos"
    ][paired_rows].to(torch.int64)

    if not torch.equal(
        record_pos_matrix,
        record_pos_matrix[:, :1].expand_as(record_pos_matrix),
    ):
        raise RuntimeError(
            f"Task {task} changes record_pos across checkpoints."
        )

    if task_spec.relabel_with_trace_outcome:
        keep = trace_outcome_known[
            record_pos_matrix[:, 0]
        ]

        paired_rows = paired_rows[keep]
        paired_source = paired_source[keep]

        if not paired_rows.numel():
            raise RuntimeError(
                f"Task {task} has no traces with known final outcome."
            )


    def constant_field(name: str, dtype: torch.dtype) -> torch.Tensor:
        values = examples[name][paired_rows].to(dtype)
        if not torch.equal(values, values[:, :1].expand_as(values)):
            raise RuntimeError(f"Task {task} field {name!r} changes across checkpoints.")
        return values[:, 0].contiguous()

    record_pos = constant_field(
        "record_pos",
        torch.int32,
    )

    if task_spec.relabel_with_trace_outcome:
        label = trace_outcome_label[
            record_pos.to(torch.int64)
        ].contiguous()
    else:
        label = constant_field(
            "label",
            torch.bool,
        )

    return UnitTable(
        task=task,
        checkpoints=checkpoints,
        activation_index=examples["activation_index"][paired_rows].to(torch.int32),
        label=label,
        split_id=constant_field("split_id", torch.int8),
        family_id=constant_field("family_id", torch.int32),
        source_row=paired_source[:, 0].contiguous(),
        record_pos=record_pos,
        cache_idx=constant_field("cache_idx", torch.int64),
        target=constant_field("target", torch.int16),
    )


def split_unit_rows(units: UnitTable, split_to_id: dict[str, int]) -> dict[str, torch.Tensor]:
    return {
        name: torch.nonzero(
            units.split_id.to(torch.int64) == int(split_id),
            as_tuple=False,
        ).flatten()
        for name, split_id in split_to_id.items()
    }


def balanced_weights(
    labels: torch.Tensor,
    record_pos: torch.Tensor,
    trace_normalized: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Give each class half of the total loss mass without discarding samples."""

    labels = labels.to(torch.bool)
    n = int(labels.numel())
    if n <= 0:
        raise RuntimeError("Cannot build weights for an empty split.")

    base = torch.ones(n, dtype=torch.float64)
    if trace_normalized:
        record_pos = record_pos.to(torch.int64)
        events_per_trace = torch.bincount(record_pos)
        base = 1.0 / events_per_trace[record_pos].to(torch.float64)

    positive_mass = float(base[labels].sum())
    negative_mass = float(base[~labels].sum())
    if positive_mass <= 0 or negative_mass <= 0:
        raise RuntimeError("Both classes are required in every split.")

    positive_factor = (n / 2.0) / positive_mass
    negative_factor = (n / 2.0) / negative_mass

    weights = base.clone()
    weights[labels] *= positive_factor
    weights[~labels] *= negative_factor

    expected = n / 2.0
    if not math.isclose(float(weights[labels].sum()), expected, rel_tol=1e-8):
        raise RuntimeError("Positive weighted mass is not 50 percent.")
    if not math.isclose(float(weights[~labels].sum()), expected, rel_tol=1e-8):
        raise RuntimeError("Negative weighted mass is not 50 percent.")

    return weights.to(torch.float32), {
        "positive_factor": positive_factor,
        "negative_factor": negative_factor,
    }


# =============================================================================
# MATERIALIZE AND STANDARDIZE ONE REPRESENTATION BLOCK
# =============================================================================


def materialize_features(
    bank: ActivationBank,
    activation_index: torch.Tensor,
    representation_type: str,
    device: torch.device,
) -> torch.Tensor:
    n_units, n_checkpoints = activation_index.shape
    spec = REPRESENTATION_SPECS[representation_type]
    n_probes = n_checkpoints * spec.locations

    source = {
        "attention_head": bank.attention,
        "residual_stream": bank.residual,
        "final_hidden": bank.final_hidden,
        "mlp_output": bank.mlp,
    }[representation_type]

    output = torch.empty(
        (n_units, n_probes, spec.feature_dim),
        dtype=torch.float16,
        device=device,
    )

    for start in tqdm(
        range(0, n_units, FEATURE_COPY_BATCH_SIZE),
        desc=f"Preparing {representation_type}",
        leave=False,
        dynamic_ncols=True,
    ):
        end = min(start + FEATURE_COPY_BATCH_SIZE, n_units)
        index = activation_index[start:end].to(torch.int64)
        chunk = source[index].reshape(end - start, n_probes, spec.feature_dim)
        output[start:end].copy_(chunk.to(device))
        del chunk

    return output


def feature_statistics(
    features: torch.Tensor,
    train_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Compute training-only mean and standard deviation in bounded batches."""

    train_rows = train_rows.to(features.device, dtype=torch.int64)
    feature_sum = torch.zeros(features.shape[1:], dtype=torch.float64, device=features.device)
    square_sum = torch.zeros_like(feature_sum)
    count = 0

    for start in range(0, int(train_rows.numel()), EVAL_BATCH_SIZE):
        rows = train_rows[start : start + EVAL_BATCH_SIZE]
        batch = features[rows].to(torch.float32)
        feature_sum += batch.sum(dim=0, dtype=torch.float64)
        square_sum += batch.square().sum(dim=0, dtype=torch.float64)
        count += int(batch.shape[0])
        del batch

    mean64 = feature_sum / max(1, count)
    variance = (square_sum / max(1, count) - mean64.square()).clamp_min(0.0)
    zero_variance = int((variance < STD_EPS**2).sum().item())
    std = variance.clamp_min(STD_EPS**2).sqrt().to(torch.float32)

    return mean64.to(torch.float32), std, zero_variance


# =============================================================================
# VECTORIZED LOGISTIC REGRESSION
# =============================================================================


def candidate_validation_loss(
    features: torch.Tensor,
    rows: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    weights: torch.Tensor,
    biases: torch.Tensor,
) -> torch.Tensor:
    rows = rows.to(features.device, dtype=torch.int64)
    labels = labels.to(features.device, dtype=torch.float32)
    sample_weights = sample_weights.to(features.device, dtype=torch.float32)

    total_loss = torch.zeros(weights.shape[:2], dtype=torch.float64, device=features.device)
    total_mass = sample_weights.sum().clamp_min(1e-12).to(torch.float64)

    with torch.inference_mode():
        for start in range(0, int(rows.numel()), EVAL_BATCH_SIZE):
            batch_rows = rows[start : start + EVAL_BATCH_SIZE]
            batch_features = (features[batch_rows].float() - feature_mean) / feature_std
            logits = torch.einsum("bpd,kpd->bkp", batch_features, weights) + biases[None]

            batch_labels = labels[start : start + EVAL_BATCH_SIZE, None, None]
            batch_weights = sample_weights[start : start + EVAL_BATCH_SIZE, None, None]
            losses = F.binary_cross_entropy_with_logits(
                logits,
                batch_labels.expand_as(logits),
                reduction="none",
            )
            total_loss += (losses * batch_weights).sum(dim=0).to(torch.float64)

    return (total_loss / total_mass).to(torch.float32)


def selected_logits(
    features: torch.Tensor,
    rows: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    weights: torch.Tensor,
    biases: torch.Tensor,
) -> torch.Tensor:
    rows = rows.to(features.device, dtype=torch.int64)
    pieces: list[torch.Tensor] = []

    with torch.inference_mode():
        for start in range(0, int(rows.numel()), EVAL_BATCH_SIZE):
            batch_rows = rows[start : start + EVAL_BATCH_SIZE]
            batch_features = (features[batch_rows].float() - feature_mean) / feature_std
            logits = torch.einsum("bpd,pd->bp", batch_features, weights) + biases[None]
            pieces.append(logits.cpu())

    return torch.cat(pieces, dim=0)


def all_candidate_logits(
    features: torch.Tensor,
    rows: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    weights: torch.Tensor,
    biases: torch.Tensor,
) -> torch.Tensor:
    rows = rows.to(features.device, dtype=torch.int64)
    pieces: list[torch.Tensor] = []

    with torch.inference_mode():
        for start in range(0, int(rows.numel()), EVAL_BATCH_SIZE):
            batch_rows = rows[start : start + EVAL_BATCH_SIZE]
            batch_features = (features[batch_rows].float() - feature_mean) / feature_std
            logits = torch.einsum("bpd,kpd->bkp", batch_features, weights) + biases[None]
            pieces.append(logits.cpu())

    return torch.cat(pieces, dim=0)


def train_probe_block(
    features: torch.Tensor,
    labels: torch.Tensor,
    split_rows: dict[str, torch.Tensor],
    train_weights: torch.Tensor,
    validation_weights: torch.Tensor,
    task: str,
    representation_type: str,
    max_epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    device = features.device
    feature_mean, feature_std, zero_variance = feature_statistics(features, split_rows["train"])

    if not bool(torch.isfinite(feature_mean).all()) or not bool(torch.isfinite(feature_std).all()):
        raise RuntimeError("Feature standardization produced non-finite values.")

    n_candidates = len(L2_GRID)
    n_probes = int(features.shape[1])
    feature_dim = int(features.shape[2])

    weights = torch.zeros(
        (n_candidates, n_probes, feature_dim),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    biases = torch.zeros(
        (n_candidates, n_probes),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    best_weights = torch.zeros_like(weights)
    best_biases = torch.zeros_like(biases)
    best_validation_loss = torch.full(
        (n_candidates, n_probes),
        float("inf"),
        dtype=torch.float32,
        device=device,
    )
    best_epoch = torch.full(
        (n_candidates, n_probes),
        -1,
        dtype=torch.int16,
        device=device,
    )

    l2_values = torch.tensor(L2_GRID, dtype=torch.float32, device=device)
    try:
        optimizer = torch.optim.Adam([weights, biases], lr=LEARNING_RATE, fused=device.type == "cuda")
    except TypeError:
        optimizer = torch.optim.Adam([weights, biases], lr=LEARNING_RATE)

    train_rows_device = split_rows["train"].to(device, dtype=torch.int64)
    all_labels = labels.to(device, dtype=torch.float32)
    all_train_weights = torch.zeros(labels.numel(), dtype=torch.float32, device=device)
    all_train_weights[train_rows_device] = train_weights.to(device)

    generator = torch.Generator(device=device)
    generator.manual_seed(SEED)

    stale_epochs = 0
    epochs_completed = 0

    progress = tqdm(
        range(max_epochs),
        desc=f"{task} | {representation_type}",
        leave=False,
        dynamic_ncols=True,
    )

    for epoch in progress:
        permutation = train_rows_device[
            torch.randperm(train_rows_device.numel(), generator=generator, device=device)
        ]
        train_loss_sum = torch.zeros(
            (n_candidates, n_probes),
            dtype=torch.float64,
            device=device,
        )
        train_mass = 0.0

        for start in range(0, int(permutation.numel()), batch_size):
            batch_rows = permutation[start : start + batch_size]
            batch_features = (features[batch_rows].float() - feature_mean) / feature_std
            batch_labels = all_labels[batch_rows, None, None]
            batch_weights = all_train_weights[batch_rows, None, None]

            logits = torch.einsum("bpd,kpd->bkp", batch_features, weights) + biases[None]
            losses = F.binary_cross_entropy_with_logits(
                logits,
                batch_labels.expand_as(logits),
                reduction="none",
            )

            # The sample weights are globally normalized so that their sum
            # equals the number of training examples. Taking the minibatch
            # mean therefore gives an unbiased estimate of the intended
            # globally class-balanced objective.
            data_loss = (losses * batch_weights).mean(dim=0)
            regularization = 0.5 * l2_values[:, None] * weights.square().sum(dim=-1)
            total_loss = (data_loss + regularization).mean()

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()

            if not bool(torch.isfinite(weights.grad).all()) or not bool(torch.isfinite(biases.grad).all()):
                raise RuntimeError("Probe gradients became non-finite.")

            optimizer.step()
            train_loss_sum += (losses.detach() * batch_weights).sum(dim=0).to(torch.float64)
            train_mass += int(batch_rows.numel())

        validation_loss = candidate_validation_loss(
            features=features,
            rows=split_rows["val"],
            labels=labels[split_rows["val"]],
            sample_weights=validation_weights,
            feature_mean=feature_mean,
            feature_std=feature_std,
            weights=weights,
            biases=biases,
        )

        improved = validation_loss < best_validation_loss - MIN_DELTA
        if bool(improved.any()):
            best_validation_loss[improved] = validation_loss[improved]
            best_weights[improved] = weights.detach()[improved]
            best_biases[improved] = biases.detach()[improved]
            best_epoch[improved] = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1

        mean_train = float((train_loss_sum / max(train_mass, 1e-12)).mean())
        mean_val = float(validation_loss.mean())
        progress.set_postfix(train=f"{mean_train:.4f}", val=f"{mean_val:.4f}")

        epochs_completed = epoch + 1
        if stale_epochs >= PATIENCE:
            break

    if bool((best_epoch < 0).any()):
        raise RuntimeError("At least one candidate never saved a validation state.")

    candidate_val_logits = all_candidate_logits(
        features=features,
        rows=split_rows["val"],
        feature_mean=feature_mean,
        feature_std=feature_std,
        weights=best_weights,
        biases=best_biases,
    )
    validation_labels = labels[split_rows["val"]].numpy().astype(np.int8)
    candidate_auroc = torch.empty((n_candidates, n_probes), dtype=torch.float32)

    for probe_index in tqdm(
        range(n_probes),
        desc="Selecting L2",
        leave=False,
        dynamic_ncols=True,
    ):
        for candidate in range(n_candidates):
            candidate_auroc[candidate, probe_index] = roc_auc_score(
                validation_labels,
                candidate_val_logits[:, candidate, probe_index].numpy(),
            )

    selected_candidate = torch.argmax(candidate_auroc, dim=0)
    probe_index = torch.arange(n_probes)

    best_weights_cpu = best_weights.detach().cpu()
    best_biases_cpu = best_biases.detach().cpu()
    best_epoch_cpu = best_epoch.detach().cpu()
    best_validation_loss_cpu = best_validation_loss.detach().cpu()

    gather_index = selected_candidate.view(1, 1, n_probes).expand(
        candidate_val_logits.shape[0],
        1,
        n_probes,
    )

    return {
        "feature_mean": feature_mean.detach().cpu(),
        "feature_std": feature_std.detach().cpu(),
        "zero_variance_dimensions": zero_variance,
        "selected_candidate": selected_candidate,
        "selected_weights": best_weights_cpu[selected_candidate, probe_index],
        "selected_biases": best_biases_cpu[selected_candidate, probe_index],
        "selected_l2": torch.tensor(L2_GRID, dtype=torch.float32)[selected_candidate],
        "selected_epoch": best_epoch_cpu[selected_candidate, probe_index],
        "selected_validation_loss": best_validation_loss_cpu[selected_candidate, probe_index],
        "candidate_validation_auroc": candidate_auroc,
        "candidate_best_validation_loss": best_validation_loss_cpu,
        "candidate_best_epoch": best_epoch_cpu,
        "validation_logits": candidate_val_logits.gather(1, gather_index).squeeze(1),
        "epochs_completed": epochs_completed,
    }


# =============================================================================
# RESULT ASSEMBLY
# =============================================================================


def finalize_block(
    *,
    data: dict[str, Any],
    units: UnitTable,
    rows: dict[str, torch.Tensor],
    representation_type: str,
    representation_spec: RepresentationSpec,
    trained: dict[str, Any],
    test_logits: torch.Tensor,
    probe_counter: int,
    validation_logits_all: torch.Tensor,
    test_logits_all: torch.Tensor,
    thresholds: torch.Tensor,
    probe_ids: list[list[str | None]],
    metrics_rows: list[dict[str, Any]],
    probe_bank: dict[str, dict[str, Any]],
) -> tuple[int, list[dict[str, Any]]]:
    n_locations = representation_spec.locations
    n_probes = units.n_checkpoints * n_locations
    validation_labels = units.label[rows["val"]].numpy().astype(np.int8)
    test_labels = units.label[rows["test"]].numpy().astype(np.int8)
    block_rows: list[dict[str, Any]] = []

    for local_probe_index in tqdm(
        range(n_probes),
        desc="Computing metrics",
        leave=False,
        dynamic_ncols=True,
    ):
        checkpoint_local, location_local = divmod(local_probe_index, n_locations)
        checkpoint = units.checkpoints[checkpoint_local]
        location = location_info(representation_type, location_local)

        validation_scores = trained["validation_logits"][:, local_probe_index].numpy()
        test_scores = test_logits[:, local_probe_index].numpy()
        threshold = choose_validation_threshold(validation_labels, validation_scores)

        validation_metrics = calculate_metrics(validation_labels, validation_scores, threshold)
        test_metrics = calculate_metrics(test_labels, test_scores, threshold)

        probe_id = f"probe_{probe_counter:06d}"
        prediction_cache_id = f"{units.task}:{checkpoint}:{location['location_name']}"

        standardized_weight = trained["selected_weights"][local_probe_index].float()
        standardized_bias = float(trained["selected_biases"][local_probe_index])
        mean = trained["feature_mean"][local_probe_index].float()
        std = trained["feature_std"][local_probe_index].float()

        raw_weight = standardized_weight / std
        raw_bias = standardized_bias - float((raw_weight * mean).sum())
        normalized_direction = raw_weight / raw_weight.norm().clamp_min(1e-12)

        probe_bank[probe_id] = {
            "probe_id": probe_id,
            "task": units.task,
            "label_name": task_label_name(data, units.task),
            "checkpoint": checkpoint,
            **location,
            "feature_dim": representation_spec.feature_dim,
            "selected_l2": float(trained["selected_l2"][local_probe_index]),
            "selected_epoch": int(trained["selected_epoch"][local_probe_index]),
            "selected_threshold_logit": threshold,
            "weight_standardized": standardized_weight,
            "bias_standardized": standardized_bias,
            "feature_mean": mean,
            "feature_std": std,
            "weight_raw_space": raw_weight,
            "bias_raw_space": raw_bias,
            "direction_raw_space_normalized": normalized_direction,
            "direction_sign": "larger logit means label=True",
            "candidate_l2_grid": torch.tensor(L2_GRID, dtype=torch.float32),
            "candidate_validation_auroc": trained["candidate_validation_auroc"][:, local_probe_index].clone(),
            "candidate_best_validation_loss": trained["candidate_best_validation_loss"][:, local_probe_index].clone(),
            "candidate_best_epoch": trained["candidate_best_epoch"][:, local_probe_index].clone(),
        }

        row = {
            "probe_id": probe_id,
            "prediction_cache_id": prediction_cache_id,
            "task": units.task,
            "label_name": task_label_name(data, units.task),
            "checkpoint": checkpoint,
            "representation_type": location["representation_type"],
            "location_name": location["location_name"],
            "layer": int(location["layer"]),
            "head": int(location["head"]),
            "feature_dim": representation_spec.feature_dim,
            "selected_l2": float(trained["selected_l2"][local_probe_index]),
            "selected_epoch": int(trained["selected_epoch"][local_probe_index]),
            "selected_validation_loss": float(trained["selected_validation_loss"][local_probe_index]),
            "selected_threshold_logit": threshold,
            "train_units": int(rows["train"].numel()),
            "train_positive": int(units.label[rows["train"]].sum()),
            "train_negative": int((~units.label[rows["train"]]).sum()),
            **prefix_metrics("val", validation_metrics),
            **prefix_metrics("test", test_metrics),
        }

        if row["test_tn"] + row["test_fp"] + row["test_fn"] + row["test_tp"] != row["test_examples"]:
            raise RuntimeError("A confusion matrix does not sum to the test count.")

        metrics_rows.append(row)
        block_rows.append(row)

        global_location = representation_spec.global_offset + location_local
        validation_logits_all[:, checkpoint_local, global_location] = trained["validation_logits"][:, local_probe_index]
        test_logits_all[:, checkpoint_local, global_location] = test_logits[:, local_probe_index]
        thresholds[checkpoint_local, global_location] = threshold
        probe_ids[checkpoint_local][global_location] = probe_id
        probe_counter += 1

    return probe_counter, block_rows


def print_block_summary(
    *,
    block_number: int,
    total_blocks: int,
    task: str,
    representation_type: str,
    units: UnitTable,
    spec: RepresentationSpec,
    trained: dict[str, Any],
    block_rows: list[dict[str, Any]],
) -> None:
    selected_counts = torch.bincount(
        trained["selected_candidate"],
        minlength=len(L2_GRID),
    )
    val_aurocs = np.asarray(
        [row["val_auroc"] for row in block_rows],
        dtype=np.float64,
    )
    test_bacc = np.asarray(
        [row["test_balanced_accuracy"] for row in block_rows],
        dtype=np.float64,
    )
    best = max(block_rows, key=lambda row: row["val_auroc"])

    print()
    print("-" * WIDTH)
    print_rows(
        f"BLOCK {block_number} / {total_blocks}",
        [
            ("Task", task),
            ("Representation family", representation_type),
            ("Positions trained together", units.n_checkpoints),
            ("Representation locations", spec.locations),
            ("Final probes in block", units.n_checkpoints * spec.locations),
            ("Candidate probes in block", units.n_checkpoints * spec.locations * len(L2_GRID)),
            ("Feature dimension", spec.feature_dim),
            ("Epochs completed", trained["epochs_completed"]),
            (
                "Best epoch range",
                f"{int(trained['selected_epoch'].min())}-{int(trained['selected_epoch'].max())}",
            ),
            ("Zero-variance dimensions", trained["zero_variance_dimensions"]),
            ("Losses, gradients, and weights finite", "yes"),
        ],
    )

    print("  Selected L2 counts")
    for index, l2_value in enumerate(L2_GRID):
        print(f"    {l2_value:<10g}  {int(selected_counts[index]):>6,}")

    print()
    print_rows(
        "  Score ranges",
        [
            (
                "Validation AUROC min / median / max",
                f"{val_aurocs.min():.4f} / "
                f"{np.median(val_aurocs):.4f} / "
                f"{val_aurocs.max():.4f}",
            ),
            (
                "Test balanced accuracy min / median / max",
                f"{test_bacc.min():.4f} / "
                f"{np.median(test_bacc):.4f} / "
                f"{test_bacc.max():.4f}",
            ),
        ],
        indent=4,
    )

    print_rows(
        "  Best validation-ranked probe",
        [
            ("Position", best["checkpoint"]),
            ("Location", best["location_name"]),
            ("Validation AUROC", f"{best['val_auroc']:.4f}"),
            ("Test AUROC", f"{best['test_auroc']:.4f}"),
            ("Test balanced accuracy", f"{best['test_balanced_accuracy']:.4f}"),
        ],
        indent=4,
    )
    print("-" * WIDTH)

def print_global_summary(
    metrics_rows: list[dict[str, Any]],
    max_epochs: int,
) -> None:
    """
    Print one global row per position.

    Representation columns contain median test AUROC. The best probe is
    selected using validation AUROC only, after which its test metrics are
    displayed.
    """

    task_names = {
        "operation_reachability": "operation",
        "back_reachability": "back",
        "trace_outcome_trajectory": "trajectory",
        "terminal_outcome_controls": "terminal",
        "trace_outcome_operation_decisions": "decision-op",
        "trace_outcome_back_decisions": "decision-back",
    }

    rows_by_position: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = {}

    for row in metrics_rows:
        key = (row["task"], row["checkpoint"])
        rows_by_position.setdefault(key, []).append(row)

    table_rows: list[tuple[Any, ...]] = []

    for task, task_spec in TASK_SPECS.items():
        for position in task_spec.checkpoints:
            position_rows = rows_by_position[(task, position)]

            if len(position_rows) != len(REPRESENTATION_LOCATION_NAMES):
                raise RuntimeError(
                    f"Position {task}/{position} has "
                    f"{len(position_rows)} probes instead of "
                    f"{len(REPRESENTATION_LOCATION_NAMES)}."
                )

            def median_test_auroc(representation_type: str) -> float:
                values = [
                    row["test_auroc"]
                    for row in position_rows
                    if row["representation_type"] == representation_type
                ]
                return float(np.median(values))

            best = max(
                position_rows,
                key=lambda row: row["val_auroc"],
            )

            table_rows.append(
                (
                    task_names[task],
                    position,
                    f"{median_test_auroc('attention_head'):.4f}",
                    f"{median_test_auroc('residual_stream'):.4f}",
                    f"{median_test_auroc('mlp_output'):.4f}",
                    f"{median_test_auroc('final_hidden'):.4f}",
                    best["location_name"],
                    f"{best['val_auroc']:.4f}",
                    f"{best['test_auroc']:.4f}",
                    f"{best['test_balanced_accuracy']:.4f}",
                    f"{best['test_macro_auprc']:.4f}",
                    f"{best['test_mcc']:.4f}",
                    best["selected_epoch"],
                    f"{best['selected_l2']:g}",
                )
            )

    best_overall = max(
        metrics_rows,
        key=lambda row: row["val_auroc"],
    )

    auroc_gaps = np.asarray(
        [
            abs(row["val_auroc"] - row["test_auroc"])
            for row in metrics_rows
        ],
        dtype=np.float64,
    )

    l2_counts = [
        (
            l2,
            sum(
                math.isclose(
                    float(row["selected_l2"]),
                    l2,
                    rel_tol=0.0,
                    abs_tol=1e-8,
                )
                for row in metrics_rows
            ),
        )
        for l2 in L2_GRID
    ]

    if sum(count for _, count in l2_counts) != len(metrics_rows):
        raise RuntimeError(
            "L2 selection counts do not sum to the number of probes."
        )

    print()
    heading("GLOBAL PROBE PERFORMANCE SUMMARY")
    print(
        "Representation columns show median test AUROC. "
        "Best probes are ranked using validation AUROC only."
    )
    print()

    print_table(
        (
            "Task",
            "Position",
            "Attn med",
            "Resid med",
            "MLP med",
            "Final",
            "Best location",
            "Val AUC",
            "Test AUC",
            "Test BAcc",
            "Macro AP",
            "MCC",
            "Epoch",
            "L2",
        ),
        table_rows,
        right_aligned=frozenset(
            {2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13}
        ),
    )

    print()
    print_rows(
        "GLOBAL SELECTION SUMMARY",
        [
            ("Positions summarized", len(table_rows)),
            ("Probes summarized", len(metrics_rows)),
            (
                "Best overall validation-ranked probe",
                f"{task_names[best_overall['task']]} / "
                f"{best_overall['checkpoint']} / "
                f"{best_overall['location_name']}",
            ),
            (
                "Validation / test AUROC of that probe",
                f"{best_overall['val_auroc']:.4f} / "
                f"{best_overall['test_auroc']:.4f}",
            ),
            (
                "Test balanced accuracy of that probe",
                f"{best_overall['test_balanced_accuracy']:.4f}",
            ),
            (
                "Median absolute validation-test AUROC gap",
                f"{float(np.median(auroc_gaps)):.4f}",
            ),
            (
                "Maximum absolute validation-test AUROC gap",
                f"{float(auroc_gaps.max()):.4f}",
            ),
            (
                "Probes selected at maximum epoch",
                f"{sum(row['selected_epoch'] == max_epochs for row in metrics_rows)} "
                f"/ {len(metrics_rows)}",
            ),
            (
                "L2 selection counts",
                ", ".join(
                    f"{l2:g}={count}"
                    for l2, count in l2_counts
                ),
            ),
        ],
    )
    print("=" * WIDTH)


def build_prediction_payload(
    *,
    data: dict[str, Any],
    units: UnitTable,
    rows: dict[str, torch.Tensor],
    probe_ids: list[list[str | None]],
    thresholds: torch.Tensor,
    validation_logits: torch.Tensor,
    test_logits: torch.Tensor,
) -> dict[str, Any]:
    def split_payload(split_name: str, logits: torch.Tensor) -> dict[str, Any]:
        split_rows = rows[split_name]
        return {
            "unit_row": split_rows.to(torch.int32),
            "source_row": units.source_row[split_rows],
            "record_pos": units.record_pos[split_rows],
            "cache_idx": units.cache_idx[split_rows],
            "target": units.target[split_rows],
            "family_id": units.family_id[split_rows],
            "label": units.label[split_rows],
            "activation_index": units.activation_index[split_rows],
            "logits": logits,
        }

    return {
        "format_version": 1,
        "complete": True,
        "run_tag": RUN_TAG,
        "task": units.task,
        "label_name": task_label_name(data, units.task),
        "checkpoint_names": list(units.checkpoints),
        "representation_location_names": list(REPRESENTATION_LOCATION_NAMES),
        "probe_ids": probe_ids,
        "thresholds_logit": thresholds,
        "validation": split_payload("val", validation_logits),
        "test": split_payload("test", test_logits),
    }


# =============================================================================
# FINAL AUDIT
# =============================================================================


def audit_final_archive(
    metrics_rows: list[dict[str, Any]],
    probe_bank: dict[str, dict[str, Any]],
    prediction_paths: dict[str, str],
) -> set[str]:
    if len(metrics_rows) != EXPECTED_PROBES:
        raise RuntimeError(f"Expected {EXPECTED_PROBES} metrics rows, got {len(metrics_rows)}.")
    if len(probe_bank) != EXPECTED_PROBES:
        raise RuntimeError(f"Expected {EXPECTED_PROBES} probe entries, got {len(probe_bank)}.")
    if len(prediction_paths) != len(TASK_SPECS):
        raise RuntimeError("The number of prediction files is incorrect.")

    probe_ids = {row["probe_id"] for row in metrics_rows}
    if len(probe_ids) != EXPECTED_PROBES:
        raise RuntimeError("Probe IDs are not unique.")

    observed_by_type = {name: 0 for name in REPRESENTATION_SPECS}
    for row in metrics_rows:
        if row["probe_id"] not in probe_bank:
            raise RuntimeError("A metric row has no matching probe-bank entry.")
        if row["test_tn"] + row["test_fp"] + row["test_fn"] + row["test_tp"] != row["test_examples"]:
            raise RuntimeError("A test confusion matrix does not sum correctly.")
        observed_by_type[row["representation_type"]] += 1

    if observed_by_type != EXPECTED_PROBES_BY_TYPE:
        raise RuntimeError(
            f"Probe-type counts differ. Expected {EXPECTED_PROBES_BY_TYPE}, got {observed_by_type}."
        )

    for probe in probe_bank.values():
        for field in (
            "weight_standardized",
            "feature_mean",
            "feature_std",
            "weight_raw_space",
            "direction_raw_space_normalized",
        ):
            if not bool(torch.isfinite(probe[field]).all()):
                raise RuntimeError(f"Probe field {field!r} contains non-finite values.")

        for field in (
            "bias_standardized",
            "bias_raw_space",
            "selected_l2",
            "selected_threshold_logit",
        ):
            if not math.isfinite(float(probe[field])):
                raise RuntimeError(f"Probe scalar {field!r} is non-finite.")

    return probe_ids


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute and evaluate every A* linear-probe vector.")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()

    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.max_epochs <= 0:
        raise ValueError("--max-epochs must be positive.")

    torch.set_num_threads(min(args.workers, os.cpu_count() or args.workers))
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    prediction_tmp = Path(PREDICTION_DIR + ".tmp")
    if prediction_tmp.exists():
        shutil.rmtree(prediction_tmp)
    prediction_tmp.mkdir(parents=True, exist_ok=False)

    total_blocks = len(TASK_SPECS) * len(REPRESENTATION_SPECS)

    heading("COMPUTE A* LINEAR-PROBE VECTORS")
    run_rows = [
        ("Run tag", RUN_TAG),
        ("Probe dataset", DATASET_PATH),
        ("Device", device),
    ]
    if device.type == "cuda":
        run_rows.append(("GPU", torch.cuda.get_device_name(device)))
    run_rows.extend(
        [
            ("CPU workers", args.workers),
            ("Task groups", len(TASK_SPECS)),
            ("Task/checkpoint combinations", EXPECTED_CHECKPOINTS),
            ("Representation locations per position", len(REPRESENTATION_LOCATION_NAMES)),
            ("Final selected probes", EXPECTED_PROBES),
            ("Vectorized training blocks", total_blocks),
            ("L2 candidates per probe", len(L2_GRID)),
            ("L2 grid", list(L2_GRID)),
            ("Maximum epochs", args.max_epochs),
            ("Training batch size", args.batch_size),
            ("Training objective", "all units, class-balanced logistic loss"),
            ("L2 selection", "validation AUROC"),
            ("Threshold selection", "validation balanced accuracy"),
            ("Final reporting split", "test"),
        ]
    )
    print_rows("RUN CONFIGURATION", run_rows)
    print("=" * WIDTH)

    data = load_probe_dataset()
    audit_probe_dataset(data)

    (
        trace_outcome_label,
        trace_outcome_known,
    ) = build_trace_outcome_lookup(data)

    print_rows(
        "TRACE-OUTCOME LABEL AUDIT",
        [
            (
                "Traces with known final outcome",
                f"{int(trace_outcome_known.sum()):,}",
            ),
            (
                "Correct traces",
                f"{int(trace_outcome_label[trace_outcome_known].sum()):,}",
            ),
            (
                "Incorrect traces",
                f"{int((~trace_outcome_label[trace_outcome_known]).sum()):,}",
            ),
        ],
    )

    activation_bytes = 2 * sum(
        math.prod(tuple(data[name]))
        for name in (
            "attention_head_output_pre_projection_shape",
            "mlp_output_before_residual_shape",
            "residual_stream_after_block_shape",
            "final_layer_normalized_hidden_shape",
        )
    )
    memory_rows = [("CPU activation bank", f"{bytes_to_gib(activation_bytes):.2f} GiB")]
    if device.type == "cuda":
        gpu_bytes = torch.cuda.get_device_properties(device).total_memory
        memory_rows.append(("GPU capacity", f"{bytes_to_gib(gpu_bytes):.2f} GiB"))
    memory_rows.append(("GPU loading policy", "one task/representation block at a time"))
    print()
    print_rows("MEMORY PLAN", memory_rows)

    bank = load_activation_bank(data)
    print()
    print_rows(
        "ACTIVATION LOAD AUDIT",
        [
            ("Attention shape", tuple(bank.attention.shape)),
            ("MLP shape", tuple(bank.mlp.shape)),
            ("Residual shape", tuple(bank.residual.shape)),
            ("Final-hidden shape", tuple(bank.final_hidden.shape)),
            ("All loaded values finite", "yes"),
        ],
    )

    split_to_id = {str(name): int(value) for name, value in data["split_to_id"].items()}
    for required_split in ("train", "val", "test"):
        if required_split not in split_to_id:
            raise KeyError(f"Missing split ID for {required_split!r}.")

    metrics_rows: list[dict[str, Any]] = []
    probe_bank: dict[str, dict[str, Any]] = {}
    prediction_paths: dict[str, str] = {}
    task_inventories: dict[str, Any] = {}

    probe_counter = 0
    block_counter = 0

    for task, task_spec in tqdm(
        TASK_SPECS.items(),
        total=len(TASK_SPECS),
        desc="Task groups",
        dynamic_ncols=True,
    ):
        units = build_unit_table(
            data,
            task,
            trace_outcome_label,
            trace_outcome_known,
        )
        rows = split_unit_rows(units, split_to_id)

        inventory: dict[str, Any] = {}
        inventory_rows: list[tuple[str, Any]] = []
        for split_name in ("train", "val", "test"):
            split_labels = units.label[rows[split_name]]
            positive = int(split_labels.sum())
            negative = int(split_labels.numel() - positive)
            inventory[split_name] = {
                "units": int(split_labels.numel()),
                "positive": positive,
                "negative": negative,
            }
            inventory_rows.append(
                (
                    split_name,
                    f"units={int(split_labels.numel()):,}, positive={positive:,}, negative={negative:,}",
                )
            )

        inventory_rows.extend(
            [
                ("positions", units.n_checkpoints),
                ("position pairing complete", "yes"),
            ]
        )

        if task_spec.trace_normalized:
            train_record_pos = units.record_pos[rows["train"]].to(torch.int64)
            event_counts = torch.bincount(train_record_pos)
            event_counts = event_counts[event_counts > 0].float()
            inventory_rows.extend(
                [
                    ("contributing train traces", f"{int(event_counts.numel()):,}"),
                    ("mean events per train trace", f"{float(event_counts.mean()):.2f}"),
                    ("median events per train trace", f"{float(event_counts.median()):.2f}"),
                    ("trace-normalized loss", "yes"),
                ]
            )

        print()
        print_rows(f"TASK INVENTORY - {task}", inventory_rows)
        task_inventories[task] = inventory

        train_weights, train_weight_stats = balanced_weights(
            labels=units.label[rows["train"]],
            record_pos=units.record_pos[rows["train"]],
            trace_normalized=task_spec.trace_normalized,
        )
        validation_weights, _ = balanced_weights(
            labels=units.label[rows["val"]],
            record_pos=units.record_pos[rows["val"]],
            trace_normalized=task_spec.trace_normalized,
        )

        print_rows(
            "  TRAINING WEIGHTS",
            [
                ("positive factor", f"{train_weight_stats['positive_factor']:.6f}"),
                ("negative factor", f"{train_weight_stats['negative_factor']:.6f}"),
                ("weighted class mass", "50.00% / 50.00%"),
            ],
            indent=4,
        )

        n_checkpoints = units.n_checkpoints
        n_locations = len(REPRESENTATION_LOCATION_NAMES)
        validation_logits_all = torch.empty(
            (int(rows["val"].numel()), n_checkpoints, n_locations),
            dtype=torch.float32,
        )
        test_logits_all = torch.empty(
            (int(rows["test"].numel()), n_checkpoints, n_locations),
            dtype=torch.float32,
        )
        thresholds = torch.empty((n_checkpoints, n_locations), dtype=torch.float32)
        probe_ids: list[list[str | None]] = [
            [None for _ in REPRESENTATION_LOCATION_NAMES]
            for _ in range(n_checkpoints)
        ]

        for representation_type, representation_spec in REPRESENTATION_SPECS.items():
            block_counter += 1
            features = materialize_features(
                bank=bank,
                activation_index=units.activation_index,
                representation_type=representation_type,
                device=device,
            )

            trained = train_probe_block(
                features=features,
                labels=units.label,
                split_rows=rows,
                train_weights=train_weights,
                validation_weights=validation_weights,
                task=task,
                representation_type=representation_type,
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
            )

            test_logits = selected_logits(
                features=features,
                rows=rows["test"],
                feature_mean=trained["feature_mean"].to(device),
                feature_std=trained["feature_std"].to(device),
                weights=trained["selected_weights"].to(device),
                biases=trained["selected_biases"].to(device),
            )

            probe_counter, block_rows = finalize_block(
                data=data,
                units=units,
                rows=rows,
                representation_type=representation_type,
                representation_spec=representation_spec,
                trained=trained,
                test_logits=test_logits,
                probe_counter=probe_counter,
                validation_logits_all=validation_logits_all,
                test_logits_all=test_logits_all,
                thresholds=thresholds,
                probe_ids=probe_ids,
                metrics_rows=metrics_rows,
                probe_bank=probe_bank,
            )

            print_block_summary(
                block_number=block_counter,
                total_blocks=total_blocks,
                task=task,
                representation_type=representation_type,
                units=units,
                spec=representation_spec,
                trained=trained,
                block_rows=block_rows,
            )

            del features, trained, test_logits, block_rows
            clear_memory(device)

        if any(probe_id is None for row in probe_ids for probe_id in row):
            raise RuntimeError(f"Task {task} has an unfilled probe location.")
        if not bool(torch.isfinite(validation_logits_all).all()):
            raise RuntimeError(f"Task {task} has non-finite validation logits.")
        if not bool(torch.isfinite(test_logits_all).all()):
            raise RuntimeError(f"Task {task} has non-finite test logits.")

        prediction_payload = build_prediction_payload(
            data=data,
            units=units,
            rows=rows,
            probe_ids=probe_ids,
            thresholds=thresholds,
            validation_logits=validation_logits_all,
            test_logits=test_logits_all,
        )
        temporary_prediction_path = prediction_tmp / f"{task}.pt"
        safe_torch_save(prediction_payload, str(temporary_prediction_path))

        reloaded = torch.load(temporary_prediction_path, map_location="cpu", weights_only=False)
        if not bool(reloaded.get("complete", False)):
            raise RuntimeError("Saved prediction file did not reload as complete.")
        if tuple(reloaded["validation"]["logits"].shape) != tuple(validation_logits_all.shape):
            raise RuntimeError("Reloaded validation logits have the wrong shape.")
        if tuple(reloaded["test"]["logits"].shape) != tuple(test_logits_all.shape):
            raise RuntimeError("Reloaded test logits have the wrong shape.")

        prediction_paths[task] = os.path.join(PREDICTION_DIR, f"{task}.pt")
        del units, prediction_payload, reloaded, validation_logits_all, test_logits_all
        clear_memory(device)

    if probe_counter != EXPECTED_PROBES:
        raise RuntimeError(f"Expected {EXPECTED_PROBES} probes, created {probe_counter}.")

    unique_probe_ids = audit_final_archive(metrics_rows, probe_bank, prediction_paths)

    final_prediction_dir = Path(PREDICTION_DIR)
    if final_prediction_dir.exists():
        if not OVERWRITE:
            raise FileExistsError(final_prediction_dir)
        shutil.rmtree(final_prediction_dir)
    os.replace(prediction_tmp, final_prediction_dir)

    results_payload = {
        "format_version": 1,
        "complete": True,
        "run_tag": RUN_TAG,
        "seed": SEED,
        "source_probe_dataset_path": DATASET_PATH,
        "source_activation_shard_dir": data["activation_shard_dir"],
        "task_specs": {
            name: {
                "checkpoints": list(spec.checkpoints),
                "trace_normalized": spec.trace_normalized,
                "source_task": source_task_name(name),
                "relabel_with_trace_outcome": (
                    spec.relabel_with_trace_outcome
                ),
            }
            for name, spec in TASK_SPECS.items()
        },
        "representation_specs": {
            name: {
                "locations": spec.locations,
                "feature_dim": spec.feature_dim,
                "global_offset": spec.global_offset,
            }
            for name, spec in REPRESENTATION_SPECS.items()
        },
        "representation_location_names": list(REPRESENTATION_LOCATION_NAMES),
        "l2_grid": list(L2_GRID),
        "learning_rate": LEARNING_RATE,
        "max_epochs": args.max_epochs,
        "train_batch_size": args.batch_size,
        "selection_policy": {
            "best_epoch": "minimum class-balanced validation binary cross-entropy",
            "l2": "maximum validation AUROC",
            "threshold": "maximum validation balanced accuracy",
            "test_used_for_selection": False,
        },
        "task_inventories": task_inventories,
        "metrics": metrics_rows,
        "probe_bank": probe_bank,
        "prediction_paths": prediction_paths,
        "summary": {
            "final_probes": EXPECTED_PROBES,
            "attention_probes": EXPECTED_PROBES_BY_TYPE["attention_head"],
            "residual_probes": EXPECTED_PROBES_BY_TYPE["residual_stream"],
            "final_hidden_probes": EXPECTED_PROBES_BY_TYPE["final_hidden"],
            "mlp_probes": EXPECTED_PROBES_BY_TYPE["mlp_output"],
            "prediction_files": len(prediction_paths),
        },
    }

    safe_torch_save(results_payload, RESULTS_PATH)
    safe_csv_save(metrics_rows, METRICS_CSV_PATH)

    print()
    heading("FINAL PROBE ARCHIVE AUDIT")
    print_rows(
        "AUDIT RESULTS",
        [
            ("Expected final probes", EXPECTED_PROBES),
            ("Master metric rows", len(metrics_rows)),
            ("Probe-bank entries", len(probe_bank)),
            ("Unique probe IDs", len(unique_probe_ids)),
            ("Duplicate probe IDs", 0),
            ("Attention probes", EXPECTED_PROBES_BY_TYPE["attention_head"]),
            ("Residual probes", EXPECTED_PROBES_BY_TYPE["residual_stream"]),
            ("Final-hidden probes", EXPECTED_PROBES_BY_TYPE["final_hidden"]),
            ("MLP probes", EXPECTED_PROBES_BY_TYPE["mlp_output"]),
            ("Prediction task files", len(prediction_paths)),
            ("Every metric row has a probe entry", "yes"),
            ("Every confusion matrix sums correctly", "yes"),
            ("All saved weights and directions finite", "yes"),
            ("Hyperparameters use validation only", "yes"),
            ("Thresholds use validation only", "yes"),
            ("Test data used for probe selection", "no"),
            ("Family overlap across splits", 0),
            ("OVERALL STATUS", "PASS"),
        ],
    )
    print()
    print_rows(
        "SAVED OUTPUTS",
        [
            ("Results archive", RESULTS_PATH),
            ("Metrics table", METRICS_CSV_PATH),
            ("Prediction files", PREDICTION_DIR),
        ],
    )
    print_global_summary(
        metrics_rows,
        max_epochs=args.max_epochs,
    )


if __name__ == "__main__":
    main()
