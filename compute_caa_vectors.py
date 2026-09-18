#!/usr/bin/env python3
"""
compute_caa_vectors.py

Compute one local CAA vector for every task, semantic position, and model
component used by compute_linear_probe_vectors.py.

Each vector uses training examples only:

    mean(activation | label=True) - mean(activation | label=False)

For trace-normalized tasks, each trace contributes equal total weight even
when it contains multiple eligible events.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import torch
from tqdm.auto import tqdm

import compute_linear_probe_vectors as probe


# =============================================================================
# CONFIGURATION
# =============================================================================

CAA_DIR = os.path.join(probe.ASTAR_DIR, "caa")
OUTPUT_PATH = os.path.join(CAA_DIR, f"{probe.RUN_TAG}_astar_caa_vectors.pt")

MEAN_BATCH_SIZE = 65536
ZERO_NORM_EPS = 1e-12


# =============================================================================
# CAA COMPUTATION
# =============================================================================


def example_weights(record_pos: torch.Tensor, trace_normalized: bool) -> torch.Tensor:
    """Return equal-event weights, or equal-total-weight-per-trace weights."""

    if not trace_normalized:
        return torch.ones(record_pos.numel(), dtype=torch.float32)

    record_pos = record_pos.to(torch.int64)
    counts = torch.bincount(record_pos)
    return 1.0 / counts[record_pos].to(torch.float32)


@torch.inference_mode()
def compute_caa(
    features: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """
    Compute every CAA vector in one task/representation block.

    features: [training units, positions * locations, feature_dim]
    returns:  [positions * locations, feature_dim]
    """

    if features.shape[0] != labels.numel() or labels.numel() != weights.numel():
        raise RuntimeError("Features, labels, and weights have different lengths.")

    device = features.device
    positive_sum = torch.zeros(features.shape[1:], dtype=torch.float64, device=device)
    negative_sum = torch.zeros_like(positive_sum)
    positive_mass = torch.zeros((), dtype=torch.float64, device=device)
    negative_mass = torch.zeros_like(positive_mass)

    for start in range(0, labels.numel(), batch_size):
        end = min(start + batch_size, labels.numel())

        x = features[start:end].float()
        y = labels[start:end].to(device=device, dtype=torch.bool)
        w = weights[start:end].to(device=device, dtype=torch.float32)

        positive_weights = w * y.float()
        negative_weights = w * (~y).float()

        positive_sum += (x * positive_weights[:, None, None]).sum(dim=0, dtype=torch.float64)
        negative_sum += (x * negative_weights[:, None, None]).sum(dim=0, dtype=torch.float64)
        positive_mass += positive_weights.sum(dtype=torch.float64)
        negative_mass += negative_weights.sum(dtype=torch.float64)

    if float(positive_mass) <= 0 or float(negative_mass) <= 0:
        raise RuntimeError("Both label classes are required to compute a CAA vector.")

    positive_mean = (positive_sum / positive_mass).float()
    negative_mean = (negative_sum / negative_mass).float()
    vectors = positive_mean - negative_mean

    norms = vectors.norm(dim=-1)
    directions = vectors / norms.clamp_min(ZERO_NORM_EPS)[:, None]

    for name, tensor in (
        ("positive means", positive_mean),
        ("negative means", negative_mean),
        ("CAA vectors", vectors),
        ("CAA directions", directions),
        ("CAA norms", norms),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"{name} contain non-finite values.")

    return {
        "positive_mean": positive_mean.cpu(),
        "negative_mean": negative_mean.cpu(),
        "vector": vectors.cpu(),
        "norm": norms.cpu(),
        "direction": directions.cpu(),
    }


def vector_key(task: str, checkpoint: str, location_name: str) -> str:
    return f"{task}:{checkpoint}:{location_name}"


# =============================================================================
# FINAL AUDIT
# =============================================================================


def audit_vector_bank(vector_bank: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if len(vector_bank) != probe.EXPECTED_PROBES:
        raise RuntimeError(f"Expected {probe.EXPECTED_PROBES} CAA vectors, got {len(vector_bank)}.")

    counts = {name: 0 for name in probe.REPRESENTATION_SPECS}
    zero_norm = 0

    for key, entry in vector_bank.items():
        if key != entry["vector_key"]:
            raise RuntimeError("A vector-bank key does not match its entry.")

        counts[entry["representation_type"]] += 1

        for field in (
            "positive_mean_raw_space",
            "negative_mean_raw_space",
            "vector_raw_space",
            "direction_raw_space_normalized",
        ):
            tensor = entry[field]

            if tensor.numel() != entry["feature_dim"]:
                raise RuntimeError(f"{key} field {field!r} has the wrong feature dimension.")

            if not bool(torch.isfinite(tensor).all()):
                raise RuntimeError(f"{key} field {field!r} contains non-finite values.")

        if not torch.isfinite(torch.tensor(entry["vector_norm"])):
            raise RuntimeError(f"{key} has a non-finite vector norm.")

        zero_norm += int(entry["zero_norm"])

    if counts != probe.EXPECTED_PROBES_BY_TYPE:
        raise RuntimeError(
            f"CAA representation counts differ. Expected {probe.EXPECTED_PROBES_BY_TYPE}, got {counts}."
        )

    return {
        "vectors": len(vector_bank),
        "vectors_by_representation": counts,
        "zero_norm_vectors": zero_norm,
        "all_values_finite": True,
    }


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute local CAA vectors from cached A* activations.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=MEAN_BATCH_SIZE)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    probe.heading("COMPUTE A* CAA VECTORS")
    probe.print_rows(
        "RUN CONFIGURATION",
        [
            ("Run tag", probe.RUN_TAG),
            ("Probe activation data", probe.DATASET_PATH),
            ("Output", OUTPUT_PATH),
            ("Device", device),
            ("Mean batch size", args.batch_size),
            ("Construction", "mean(label=True) - mean(label=False)"),
            ("Examples used", "training split only"),
        ],
    )
    print("=" * probe.WIDTH)

    data = probe.load_probe_dataset()
    probe.audit_probe_dataset(data)

    trace_outcome_label, trace_outcome_known = probe.build_trace_outcome_lookup(data)
    bank = probe.load_activation_bank(data)

    split_to_id = {str(name): int(value) for name, value in data["split_to_id"].items()}

    if "train" not in split_to_id:
        raise KeyError("The activation data has no training split.")

    vector_bank: dict[str, dict[str, Any]] = {}
    task_summaries: dict[str, Any] = {}
    vector_counter = 0

    for task, task_spec in tqdm(
        probe.TASK_SPECS.items(),
        total=len(probe.TASK_SPECS),
        desc="CAA task groups",
        dynamic_ncols=True,
    ):
        units = probe.build_unit_table(
            data=data,
            task=task,
            trace_outcome_label=trace_outcome_label,
            trace_outcome_known=trace_outcome_known,
        )

        train_rows = probe.split_unit_rows(units, split_to_id)["train"]
        train_labels = units.label[train_rows].to(torch.bool)
        train_weights = example_weights(
            units.record_pos[train_rows],
            trace_normalized=task_spec.trace_normalized,
        )

        train_positive = int(train_labels.sum())
        train_negative = int(train_labels.numel() - train_positive)
        task_vector_count = 0

        for representation_type, representation_spec in probe.REPRESENTATION_SPECS.items():
            features = probe.materialize_features(
                bank=bank,
                activation_index=units.activation_index[train_rows],
                representation_type=representation_type,
                device=device,
            )

            block = compute_caa(
                features=features,
                labels=train_labels,
                weights=train_weights,
                batch_size=args.batch_size,
            )

            n_locations = representation_spec.locations
            expected_vectors = units.n_checkpoints * n_locations

            if block["vector"].shape[0] != expected_vectors:
                raise RuntimeError(
                    f"{task}/{representation_type} produced {block['vector'].shape[0]} "
                    f"vectors instead of {expected_vectors}."
                )

            for local_index in range(expected_vectors):
                checkpoint_index, location_index = divmod(local_index, n_locations)
                checkpoint = units.checkpoints[checkpoint_index]
                location = probe.location_info(representation_type, location_index)
                key = vector_key(task, checkpoint, location["location_name"])

                if key in vector_bank:
                    raise RuntimeError(f"Duplicate CAA vector key: {key}")

                norm = float(block["norm"][local_index])

                vector_bank[key] = {
                    "vector_id": f"caa_{vector_counter:06d}",
                    "vector_key": key,
                    "method": "caa",
                    "task": task,
                    "source_task": probe.source_task_name(task),
                    "label_name": probe.task_label_name(data, task),
                    "checkpoint": checkpoint,
                    "checkpoint_index": checkpoint_index,
                    **location,
                    "location_index": location_index,
                    "global_location_index": representation_spec.global_offset + location_index,
                    "feature_dim": representation_spec.feature_dim,
                    "trace_normalized": task_spec.trace_normalized,
                    "train_units": int(train_rows.numel()),
                    "train_positive": train_positive,
                    "train_negative": train_negative,
                    "positive_mean_raw_space": block["positive_mean"][local_index],
                    "negative_mean_raw_space": block["negative_mean"][local_index],
                    "vector_raw_space": block["vector"][local_index],
                    "vector_norm": norm,
                    "direction_raw_space_normalized": block["direction"][local_index],
                    "direction_sign": "label=True minus label=False",
                    "zero_norm": norm <= ZERO_NORM_EPS,
                }

                vector_counter += 1
                task_vector_count += 1

            del features, block
            probe.clear_memory(device)

        task_summaries[task] = {
            "train_units": int(train_rows.numel()),
            "train_positive": train_positive,
            "train_negative": train_negative,
            "trace_normalized": task_spec.trace_normalized,
            "vectors": task_vector_count,
        }

        probe.print_rows(
            f"CAA TASK - {task}",
            [
                ("Training units", f"{int(train_rows.numel()):,}"),
                ("Positive / negative", f"{train_positive:,} / {train_negative:,}"),
                ("Trace-normalized", "yes" if task_spec.trace_normalized else "no"),
                ("CAA vectors", f"{task_vector_count:,}"),
            ],
        )

        del units, train_weights
        probe.clear_memory(device)

    audit = audit_vector_bank(vector_bank)

    payload = {
        "format_version": 1,
        "complete": True,
        "method": "caa",
        "run_tag": probe.RUN_TAG,
        "source_probe_dataset_path": probe.DATASET_PATH,
        "source_activation_shard_dir": data["activation_shard_dir"],
        "source_checkpoint_path": data.get("source_checkpoint_path"),
        "construction": "training-split mean(label=True) minus mean(label=False)",
        "trace_weighting": "equal event weight except for trace-normalized tasks, where each trace has equal total weight",
        "task_specs": {
            name: {
                "checkpoints": list(spec.checkpoints),
                "trace_normalized": spec.trace_normalized,
                "source_task": probe.source_task_name(name),
                "relabel_with_trace_outcome": spec.relabel_with_trace_outcome,
            }
            for name, spec in probe.TASK_SPECS.items()
        },
        "representation_specs": {
            name: {
                "locations": spec.locations,
                "feature_dim": spec.feature_dim,
                "global_offset": spec.global_offset,
            }
            for name, spec in probe.REPRESENTATION_SPECS.items()
        },
        "representation_location_names": list(probe.REPRESENTATION_LOCATION_NAMES),
        "task_summaries": task_summaries,
        "vector_bank": vector_bank,
        "audit": audit,
    }

    probe.safe_torch_save(payload, OUTPUT_PATH)

    print()
    probe.heading("FINAL CAA VECTOR AUDIT")
    probe.print_rows(
        "AUDIT RESULTS",
        [
            ("Expected vectors", probe.EXPECTED_PROBES),
            ("Saved vectors", audit["vectors"]),
            ("Attention vectors", audit["vectors_by_representation"]["attention_head"]),
            ("Residual vectors", audit["vectors_by_representation"]["residual_stream"]),
            ("Final-hidden vectors", audit["vectors_by_representation"]["final_hidden"]),
            ("MLP vectors", audit["vectors_by_representation"]["mlp_output"]),
            ("Zero-norm vectors", audit["zero_norm_vectors"]),
            ("All values finite", "yes"),
            ("Training split only", "yes"),
            ("Output", OUTPUT_PATH),
            ("OVERALL STATUS", "PASS"),
        ],
    )
    print("=" * probe.WIDTH)


if __name__ == "__main__":
    main()