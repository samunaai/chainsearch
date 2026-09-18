#!/usr/bin/env python3
"""
compare_steering_method_vectors.py

Compare saved Probe, CAA, and Direct Optimisation directions before any
free-generation steering experiment. The script reconciles the three archive
formats, audits their shared keys, computes five geometric analyses, and saves
plot-ready long-form rows for plot_steering_method_vectors.py.
"""

from __future__ import annotations

import argparse
import math
import os
from collections import Counter, defaultdict
from typing import Any

import torch

import compute_linear_probe_vectors as probe


# =============================================================================
# CONFIGURATION
# =============================================================================

RUN_TAG = probe.RUN_TAG
VECTOR_COMPARISON_DIR = os.path.join(probe.ASTAR_DIR, "vector_types_comparison")
DEFAULT_OUTPUT_PATH = os.path.join(
    VECTOR_COMPARISON_DIR,
    f"{RUN_TAG}_vector_types_comparison.pt",
)

DEFAULT_CAA_PATH = os.path.join(
    probe.ASTAR_DIR,
    "caa",
    f"{RUN_TAG}_astar_caa_vectors.pt",
)

DEFAULT_DIRECT_PATH = os.path.join(
    probe.ASTAR_DIR,
    "direct_optimisation",
    f"{RUN_TAG}_astar_direct_optimisation_vectors.pt",
) 

ZERO_NORM_EPS = 1e-12
UNIT_NORM_TOL = 1e-4
SPAN_COLLINEAR_COSINE = 0.999
EXPECTED_DIRECT_VECTORS = 300
EXPECTED_DIRECT_OPTIMISATIONS = 270
EXPECTED_DIRECT_MAX_STEPS = 1500
EXPECTED_DIRECT_BY_REPRESENTATION = {
    "attention_head": 240,
    "residual_stream": 30,
    "mlp_output": 30,
}
EXPECTED_ROW_COUNTS = {
    "matched_alignment_rows": 1379,
    "common_axis_rows": 672,
    "reachability_final_rows": 656,
    "direction_evolution_rows": 1276,
    "direct_span_rows": 300,
}

METHOD_ORDER = ("probe", "caa", "optimised")
METHOD_PAIRS = (
    ("probe", "caa"),
    ("probe", "optimised"),
    ("caa", "optimised"),
)
REPRESENTATION_ORDER = (
    "attention_head",
    "residual_stream",
    "mlp_output",
    "final_hidden",
)

OPERATION_SEQUENCE = probe.OPERATION_CHECKPOINTS
BACK_REACHABILITY_SEQUENCE = probe.BACK_CHECKPOINTS
BACK_FINAL_SEQUENCE = ("pre_back", *probe.BACK_CHECKPOINTS)
REACHABILITY_CHECKPOINTS = (*OPERATION_SEQUENCE, *BACK_REACHABILITY_SEQUENCE)
FINAL_SHARED_CHECKPOINTS = (*OPERATION_SEQUENCE, *BACK_FINAL_SEQUENCE, "success_token")
FINAL_ALL_CHECKPOINTS = (*FINAL_SHARED_CHECKPOINTS, "expr_final_parenthesis")

REACHABILITY_TASKS = {"operation_reachability", "back_reachability"}
FINAL_ANSWER_TASKS = {
    "trace_outcome_operation_decisions",
    "trace_outcome_trajectory",
    "trace_outcome_back_decisions",
    "terminal_outcome_controls",
}
LABEL_FAMILY_BY_TASK = {
    **{task: "reachability" for task in REACHABILITY_TASKS},
    **{task: "final_answer" for task in FINAL_ANSWER_TASKS},
}
REACHABILITY_FINAL_TASK_PAIRS = {
    "operation_reachability": "trace_outcome_operation_decisions",
    "back_reachability": "trace_outcome_back_decisions",
}
CHECKPOINT_DISPLAY = {
    "pre_first_operand": "Before first operand",
    "first_operand_end": "After first operand",
    "operator": "After operator",
    "second_operand_end": "After second operand",
    "result_end": "After result",
    "resulting_state_semicolon": "After resulting state",
    "pre_back": "Before BACK",
    "back_token": "After BACK",
    "restored_state_semicolon": "After restored state",
    "success_token": "After SUCCESS",
    "expr_final_parenthesis": "After final expression",
}


# =============================================================================
# LOADING AND CANONICALISATION
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Probe, CAA, and Direct Optimisation steering-vector geometry."
    )
    parser.add_argument("--probe-path", default=probe.RESULTS_PATH)
    parser.add_argument("--caa-path", default=DEFAULT_CAA_PATH)
    parser.add_argument("--direct-path", default=DEFAULT_DIRECT_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--span-collinear-cosine", type=float, default=SPAN_COLLINEAR_COSINE)
    parser.add_argument(
        "--expected-direct-max-steps",
        type=int,
        default=EXPECTED_DIRECT_MAX_STEPS,
        help="Fail if the Direct archive was produced with another step ceiling.",
    )
    return parser.parse_args()


def load_archive(path: str, name: str, expected_method: str | None = None) -> dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    archive = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(archive, dict):
        raise TypeError(f"{name} archive is not a dictionary: {path}")
    if int(archive.get("format_version", -1)) < 1:
        raise RuntimeError(f"{name} archive has no supported format version.")
    if not bool(archive.get("complete", False)):
        raise RuntimeError(f"{name} archive is not marked complete: {path}")
    if archive.get("run_tag") != RUN_TAG:
        raise RuntimeError(f"{name} run tag {archive.get('run_tag')!r} != {RUN_TAG!r}.")
    if expected_method is not None and archive.get("method") != expected_method:
        raise RuntimeError(
            f"{name} method {archive.get('method')!r} != {expected_method!r}."
        )
    return archive


def vector_key(task: str, checkpoint: str, location_name: str) -> str:
    return f"{task}:{checkpoint}:{location_name}"


def canonicalise_archive(method: str, archive: dict[str, Any]) -> dict[str, dict[str, Any]]:
    source_bank = archive.get("probe_bank") if method == "probe" else archive.get("vector_bank")
    if not isinstance(source_bank, dict) or not source_bank:
        raise RuntimeError(f"{method} archive contains no vector bank.")

    output: dict[str, dict[str, Any]] = {}
    for source_key, entry in source_bank.items():
        task = str(entry["task"])
        if task not in LABEL_FAMILY_BY_TASK:
            raise KeyError(f"Unknown task {task!r} in {method} archive.")
        checkpoint = str(entry["checkpoint"])
        location_name = str(entry["location_name"])
        key = vector_key(task, checkpoint, location_name)
        if key in output:
            raise RuntimeError(f"Duplicate canonical key in {method}: {key}")

        raw_field = (
            "weight_raw_space"
            if method == "probe"
            else "vector_raw_space"
        )
        feature_dim = int(entry["feature_dim"])
        raw = entry[raw_field].detach().cpu().float().reshape(-1)
        direction = entry["direction_raw_space_normalized"].detach().cpu().float().reshape(-1)

        if raw.numel() != feature_dim or direction.numel() != feature_dim:
            raise RuntimeError(f"Wrong vector dimension for {method}/{key}.")
        if not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(direction).all()):
            raise RuntimeError(f"Non-finite vector values for {method}/{key}.")

        raw_norm = float(raw.norm())
        direction_norm = float(direction.norm())
        raw_zero = raw_norm <= ZERO_NORM_EPS
        direction_zero = direction_norm <= ZERO_NORM_EPS

        if raw_zero != direction_zero:
            raise RuntimeError(
                f"Raw and normalised zero status disagree for {method}/{key}."
            )
        if "zero_norm" in entry and bool(entry["zero_norm"]) != raw_zero:
            raise RuntimeError(
                f"Saved zero_norm flag disagrees with the tensor for {method}/{key}."
            )

        zero_norm = raw_zero
        if zero_norm:
            normalized_direction = torch.zeros_like(direction)
        else:
            normalized_raw = raw / raw_norm
            normalized_direction = direction / direction_norm
            if not torch.allclose(
                normalized_raw,
                normalized_direction,
                atol=UNIT_NORM_TOL,
                rtol=UNIT_NORM_TOL,
            ):
                raise RuntimeError(
                    f"Saved direction is not the normalised raw vector for {method}/{key}."
                )

        output[key] = {
            "method": method,
            "source_key": str(source_key),
            "key": key,
            "task": task,
            "source_task": str(entry.get("source_task", task)),
            "label_family": LABEL_FAMILY_BY_TASK[task],
            "label_name": str(entry.get("label_name", "unknown")),
            "checkpoint": checkpoint,
            "representation_type": str(entry["representation_type"]),
            "location_name": location_name,
            "layer": int(entry["layer"]),
            "head": int(entry["head"]),
            "feature_dim": feature_dim,
            "raw_norm": raw_norm,
            "raw_finite": True,
            "direction_norm": direction_norm,
            "zero_norm": zero_norm,
            "direction": normalized_direction,
            "shared_optimisation_id": entry.get("shared_optimisation_id"),
        }
    return output


def load_vector_banks(
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any], dict[str, str]]:
    paths = {
        "probe": args.probe_path,
        "caa": args.caa_path,
        "optimised": args.direct_path,
    }
    archives = {
        "probe": load_archive(paths["probe"], "Probe"),
        "caa": load_archive(paths["caa"], "CAA", "caa"),
        "optimised": load_archive(
            paths["optimised"],
            "Direct Optimisation",
            "direct_optimisation",
        ),
    }
    banks = {method: canonicalise_archive(method, archives[method]) for method in METHOD_ORDER}
    return banks, archives, paths


# =============================================================================
# AUDITS AND BASIC GEOMETRY
# =============================================================================

def metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": record["task"],
        "label_family": record["label_family"],
        "checkpoint": record["checkpoint"],
        "checkpoint_display": CHECKPOINT_DISPLAY[record["checkpoint"]],
        "representation_type": record["representation_type"],
        "location_name": record["location_name"],
        "layer": record["layer"],
        "head": record["head"],
        "feature_dim": record["feature_dim"],
    }


def safe_cosine(a: dict[str, Any], b: dict[str, Any]) -> tuple[float, bool, str]:
    if a["feature_dim"] != b["feature_dim"]:
        raise RuntimeError(f"Dimension mismatch for {a['key']} and {b['key']}.")
    if a["zero_norm"] or b["zero_norm"]:
        names = [record["method"] for record in (a, b) if record["zero_norm"]]
        return float("nan"), False, "zero direction: " + ", ".join(names)
    value = float(torch.dot(a["direction"], b["direction"]))
    return max(-1.0, min(1.0, value)), True, ""


def audit_vector_banks(
    banks: dict[str, dict[str, dict[str, Any]]],
    archives: dict[str, Any],
) -> dict[str, Any]:
    expected_counts = {
        "probe": probe.EXPECTED_PROBES,
        "caa": probe.EXPECTED_PROBES,
        "optimised": EXPECTED_DIRECT_VECTORS,
    }
    counts: dict[str, int] = {}
    zeros: dict[str, int] = {}

    for method, bank in banks.items():
        counts[method] = len(bank)
        zeros[method] = sum(record["zero_norm"] for record in bank.values())
        if len(bank) != expected_counts[method]:
            raise RuntimeError(f"Expected {expected_counts[method]} {method} vectors, found {len(bank)}.")
        for key, record in bank.items():
            direction = record["direction"]
            if direction.numel() != record["feature_dim"]:
                raise RuntimeError(f"Wrong direction dimension for {method}/{key}.")
            if not record["raw_finite"] or not bool(torch.isfinite(direction).all()):
                raise RuntimeError(f"Non-finite vector values for {method}/{key}.")
            if record["zero_norm"]:
                if float(direction.norm()) > ZERO_NORM_EPS:
                    raise RuntimeError(f"Zero direction has nonzero normalized value for {method}/{key}.")
            elif abs(float(direction.norm()) - 1.0) > UNIT_NORM_TOL:
                raise RuntimeError(f"Direction is not unit-normalized for {method}/{key}.")

    probe_keys = set(banks["probe"])
    caa_keys = set(banks["caa"])
    direct_keys = set(banks["optimised"])

    three_way_nonzero_keys = {
        key
        for key in direct_keys
        if all(not banks[method][key]["zero_norm"] for method in METHOD_ORDER)
    }

    if caa_keys != probe_keys:
        raise RuntimeError(
            f"Probe and CAA key sets differ: probe-only={len(probe_keys-caa_keys)}, "
            f"CAA-only={len(caa_keys-probe_keys)}."
        )
    if not direct_keys <= probe_keys:
        raise RuntimeError(f"Direct keys missing from Probe bank: {len(direct_keys-probe_keys)}.")
    if not direct_keys <= caa_keys:
        raise RuntimeError(f"Direct keys missing from CAA bank: {len(direct_keys-caa_keys)}.")

    direct_counts = Counter(
        record["representation_type"]
        for record in banks["optimised"].values()
    )
    if dict(direct_counts) != EXPECTED_DIRECT_BY_REPRESENTATION:
        raise RuntimeError(
            "Direct representation counts differ. "
            f"Expected {EXPECTED_DIRECT_BY_REPRESENTATION}, got {dict(direct_counts)}."
        )

    for key, record in banks["optimised"].items():
        if record["checkpoint"] == "expr_final_parenthesis":
            raise RuntimeError(f"Excluded final-parenthesis vector found: {key}")
        if record["representation_type"] == "final_hidden" or record["layer"] >= 3:
            raise RuntimeError(f"Excluded Direct location found: {key}")
        if record["shared_optimisation_id"] is None:
            raise RuntimeError(f"Direct vector has no shared optimisation ID: {key}")

    direct_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in banks["optimised"].values():
        direct_groups[str(record["shared_optimisation_id"])].append(record)

    for shared_id, records in direct_groups.items():
        if len(records) == 2:
            aliases = {record["representation_type"] for record in records}
            if aliases != {"residual_stream", "mlp_output"}:
                raise RuntimeError(
                    f"Unexpected aliases for Direct optimisation {shared_id}: {aliases}"
                )
        elif len(records) != 1:
            raise RuntimeError(
                f"Direct optimisation {shared_id} has {len(records)} aliases."
            )

        reference = records[0]
        for record in records[1:]:
            if not torch.equal(reference["direction"], record["direction"]):
                raise RuntimeError(
                    f"Aliased Direct directions differ for {shared_id}."
                )
            if not math.isclose(
                reference["raw_norm"],
                record["raw_norm"],
                rel_tol=0.0,
                abs_tol=UNIT_NORM_TOL,
            ):
                raise RuntimeError(
                    f"Aliased Direct vector norms differ for {shared_id}."
                )

    for key in probe_keys:
        records = [banks["probe"][key], banks["caa"][key]]
        if key in direct_keys:
            records.append(banks["optimised"][key])
        reference = records[0]
        for record in records[1:]:
            for field in (
                "task",
                "checkpoint",
                "representation_type",
                "location_name",
                "layer",
                "head",
                "feature_dim",
            ):
                if record[field] != reference[field]:
                    raise RuntimeError(f"Metadata mismatch for {key} field {field!r}.")

    if any(record["label_family"] != "final_answer" for record in banks["optimised"].values()):
        raise RuntimeError("Direct Optimisation bank contains a non-final-answer vector.")

    source_dataset_paths = {
        method: archive.get("source_probe_dataset_path")
        for method, archive in archives.items()
    }
    missing_source_paths = [
        method
        for method, path in source_dataset_paths.items()
        if path is None
    ]
    if missing_source_paths:
        raise RuntimeError(
            f"Archives lack source_probe_dataset_path: {missing_source_paths}"
        )

    resolved_source_paths = {
        os.path.realpath(str(path))
        for path in source_dataset_paths.values()
    }
    if len(resolved_source_paths) != 1:
        raise RuntimeError(
            f"Vector archives use different probe datasets: {source_dataset_paths}"
        )

    checkpoint_paths = {
        method: archives[method].get("source_checkpoint_path")
        for method in ("caa", "optimised")
    }
    if any(path is None for path in checkpoint_paths.values()):
        raise RuntimeError(
            f"CAA or Direct archive lacks source_checkpoint_path: {checkpoint_paths}"
        )

    resolved_checkpoint_paths = {
        os.path.realpath(str(path))
        for path in checkpoint_paths.values()
    }
    if len(resolved_checkpoint_paths) != 1:
        raise RuntimeError(
            f"CAA and Direct use different checkpoints: {checkpoint_paths}"
        )

    direct_ids = {
        record["shared_optimisation_id"]
        for record in banks["optimised"].values()
        if record["shared_optimisation_id"] is not None
    }
    if len(direct_ids) != EXPECTED_DIRECT_OPTIMISATIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_DIRECT_OPTIMISATIONS} unique Direct optimisations, found {len(direct_ids)}."
        )

    return {
        "vectors_by_method": counts,
        "zero_vectors_by_method": zeros,
        "probe_caa_keys_identical": True,
        "direct_keys_subset_of_probe": True,
        "direct_keys_subset_of_caa": True,
        "direct_compatible_keys": len(direct_keys),
        "three_way_nonzero_keys": len(three_way_nonzero_keys),
        "direct_unique_optimisations": len(direct_ids),
        "direct_unique_zero_optimisations": len({
            str(record["shared_optimisation_id"])
            for record in banks["optimised"].values()
            if record["zero_norm"]
        }),
        "source_probe_dataset_path": next(iter(resolved_source_paths)),
        "source_checkpoint_path": next(iter(resolved_checkpoint_paths)),
        "all_values_finite": True,
    }


def semantic_index(bank: dict[str, dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    output: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in bank.values():
        key = (record["label_family"], record["checkpoint"], record["location_name"])
        if key in output:
            raise RuntimeError(f"Duplicate semantic key for {record['method']}: {key}")
        output[key] = record
    return output


# =============================================================================
# FIVE GEOMETRIC ANALYSES
# =============================================================================

def build_matched_alignment_rows(
    banks: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    direct_keys = set(banks["optimised"])

    three_way_nonzero_keys = {
        key
        for key in direct_keys
        if all(not banks[method][key]["zero_norm"] for method in METHOD_ORDER)
    }

    for method_a, method_b in METHOD_PAIRS:
        for key in sorted(set(banks[method_a]) & set(banks[method_b])):
            a, b = banks[method_a][key], banks[method_b][key]
            direct_record = banks["optimised"].get(key)
            cosine, valid, reason = safe_cosine(a, b)
            rows.append({
                **metadata(a),
                "key": key,
                "method_a": method_a,
                "method_b": method_b,
                "method_pair": f"{method_a}_{method_b}",
                "comparison_scope": (
                    "three_way_shared"
                    if key in three_way_nonzero_keys
                    else "three_way_zero_excluded"
                    if key in direct_keys
                    else "probe_caa_only"
                ),
                "in_three_way_support": key in three_way_nonzero_keys,
                "signed_cosine": cosine,
                "absolute_cosine": abs(cosine) if valid else float("nan"),
                "valid": valid,
                "exclusion_reason": reason,
                "direct_shared_optimisation_id": (
                    direct_record.get("shared_optimisation_id")
                    if direct_record is not None
                    else None
                ),
            })
    return rows


def axis_metrics(records: list[dict[str, Any]]) -> tuple[float, float]:
    directions = torch.stack([record["direction"] for record in records])
    resultant = float(directions.mean(dim=0).norm())
    count = len(records)
    coherence = (count * resultant * resultant - 1.0) / (count - 1)
    return resultant, max(-1.0, min(1.0, coherence))


def axis_row(
    method: str,
    family: str,
    scope: str,
    location_name: str,
    checkpoints: tuple[str, ...],
    index: dict[tuple[str, str, str], dict[str, Any]],
    shared_checkpoints: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    selected = checkpoints if shared_checkpoints is None else shared_checkpoints
    records = [index.get((family, checkpoint, location_name)) for checkpoint in selected]
    missing = [checkpoint for checkpoint, record in zip(selected, records) if record is None]
    zero = [
        checkpoint
        for checkpoint, record in zip(selected, records)
        if record is not None and record["zero_norm"]
    ]
    shared_excluded = [checkpoint for checkpoint in checkpoints if checkpoint not in selected]
    valid_records = [record for record in records if record is not None and not record["zero_norm"]]
    template = next((record for record in records if record is not None), None)
    if template is None:
        template = next(
            (
                index.get((family, checkpoint, location_name))
                for checkpoint in checkpoints
                if index.get((family, checkpoint, location_name)) is not None
            ),
            None,
        )
    valid = len(valid_records) >= 2
    resultant, coherence = axis_metrics(valid_records) if valid else (float("nan"), float("nan"))
    base = metadata(template) if template is not None else {
        "task": "multiple",
        "label_family": family,
        "checkpoint": "multiple",
        "checkpoint_display": "Multiple checkpoints",
        "representation_type": "unknown",
        "location_name": location_name,
        "layer": -1,
        "head": -1,
        "feature_dim": -1,
    }
    return {
        **base,
        "task": "multiple",
        "checkpoint": "multiple",
        "checkpoint_display": "Multiple checkpoints",
        "method": method,
        "scope": scope,
        "requested_checkpoints": list(checkpoints),
        "used_checkpoints": [record["checkpoint"] for record in valid_records],
        "missing_checkpoints": missing,
        "zero_checkpoints": zero,
        "shared_excluded_checkpoints": shared_excluded,
        "requested_checkpoint_count": len(checkpoints),
        "checkpoint_count": len(valid_records),
        "resultant_strength": resultant,
        "axis_coherence": coherence,
        "complete_scope": not missing and not zero and not shared_excluded,
        "valid": valid,
        "exclusion_reason": "" if valid else "fewer than two nonzero directions",
    }


def build_common_axis_rows(
    banks: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    indices = {method: semantic_index(bank) for method, bank in banks.items()}
    scopes = (
        ("operation", "reachability", OPERATION_SEQUENCE),
        ("operation", "final_answer", OPERATION_SEQUENCE),
        ("back", "reachability", BACK_REACHABILITY_SEQUENCE),
        ("back", "final_answer", BACK_FINAL_SEQUENCE),
        ("all", "reachability", REACHABILITY_CHECKPOINTS),
        ("all", "final_answer", FINAL_ALL_CHECKPOINTS),
    )
    rows: list[dict[str, Any]] = []

    for method in METHOD_ORDER:
        index = indices[method]
        locations = sorted({record["location_name"] for record in banks[method].values()})
        for scope, family, checkpoints in scopes:
            if method == "optimised" and family == "reachability":
                continue
            for location_name in locations:
                if any((family, checkpoint, location_name) in index for checkpoint in checkpoints):
                    rows.append(axis_row(method, family, scope, location_name, checkpoints, index))

    direct_locations = sorted({record["location_name"] for record in banks["optimised"].values()})
    for location_name in direct_locations:
        shared = tuple(
            checkpoint
            for checkpoint in FINAL_SHARED_CHECKPOINTS
            if all(
                (record := indices[method].get(("final_answer", checkpoint, location_name)))
                is not None
                and not record["zero_norm"]
                for method in METHOD_ORDER
            )
        )
        for method in METHOD_ORDER:
            rows.append(axis_row(
                method,
                "final_answer",
                "final_answer_three_way_shared",
                location_name,
                FINAL_SHARED_CHECKPOINTS,
                indices[method],
                shared_checkpoints=shared,
            ))
    return rows


def build_reachability_final_rows(
    banks: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in ("probe", "caa"):
        bank = banks[method]
        for reachability_task, final_task in REACHABILITY_FINAL_TASK_PAIRS.items():
            for checkpoint in probe.TASK_SPECS[reachability_task].checkpoints:
                for location_name in probe.REPRESENTATION_LOCATION_NAMES:
                    reach_key = vector_key(reachability_task, checkpoint, location_name)
                    final_key = vector_key(final_task, checkpoint, location_name)
                    if reach_key not in bank or final_key not in bank:
                        continue
                    reach, final = bank[reach_key], bank[final_key]
                    cosine, valid, reason = safe_cosine(reach, final)
                    rows.append({
                        **metadata(reach),
                        "method": method,
                        "reachability_task": reachability_task,
                        "final_answer_task": final_task,
                        "signed_cosine": cosine,
                        "absolute_cosine": abs(cosine) if valid else float("nan"),
                        "valid": valid,
                        "exclusion_reason": reason,
                    })
    return rows


def build_direction_evolution_rows(
    banks: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    indices = {method: semantic_index(bank) for method, bank in banks.items()}
    sequences = (
        ("operation", "reachability", OPERATION_SEQUENCE),
        ("operation", "final_answer", OPERATION_SEQUENCE),
        ("back", "reachability", BACK_REACHABILITY_SEQUENCE),
        ("back", "final_answer", BACK_FINAL_SEQUENCE),
    )
    rows: list[dict[str, Any]] = []

    direct_locations = {
        record["location_name"]
        for record in banks["optimised"].values()
    }
    three_way_transition_support: set[tuple[str, str, str, str]] = set()

    for sequence_name, family, checkpoints in sequences:
        if family != "final_answer":
            continue
        for start, end in zip(checkpoints, checkpoints[1:]):
            for location_name in direct_locations:
                records = [
                    indices[method].get((family, checkpoint, location_name))
                    for method in METHOD_ORDER
                    for checkpoint in (start, end)
                ]
                if all(
                    record is not None and not record["zero_norm"]
                    for record in records
                ):
                    three_way_transition_support.add(
                        (sequence_name, start, end, location_name)
                    )


    for method in METHOD_ORDER:
        index = indices[method]
        locations = sorted({record["location_name"] for record in banks[method].values()})
        for sequence_name, family, checkpoints in sequences:
            if method == "optimised" and family == "reachability":
                continue
            for transition_index, (start, end) in enumerate(zip(checkpoints, checkpoints[1:])):
                for location_name in locations:
                    a = index.get((family, start, location_name))
                    b = index.get((family, end, location_name))
                    if a is None or b is None:
                        continue
                    cosine, valid, reason = safe_cosine(a, b)

                    if family == "reachability":
                        comparison_scope = "probe_caa_shared"
                        in_three_way_support = False
                    else:
                        support_key = (
                            sequence_name,
                            start,
                            end,
                            location_name,
                        )
                        in_three_way_support = (
                            support_key in three_way_transition_support
                        )
                        if in_three_way_support:
                            comparison_scope = "three_way_shared"
                        elif location_name in direct_locations:
                            comparison_scope = "three_way_zero_excluded"
                        else:
                            comparison_scope = "probe_caa_only"

                    rows.append({
                        **metadata(a),
                        "method": method,
                        "sequence": sequence_name,
                        "transition_index": transition_index,
                        "from_checkpoint": start,
                        "from_display": CHECKPOINT_DISPLAY[start],
                        "to_checkpoint": end,
                        "to_display": CHECKPOINT_DISPLAY[end],
                        "transition": f"{start}_to_{end}",
                        "comparison_scope": comparison_scope,
                        "in_three_way_support": in_three_way_support,
                        "signed_cosine": cosine,
                        "absolute_cosine": abs(cosine) if valid else float("nan"),
                        "valid": valid,
                        "exclusion_reason": reason,
                    })
    return rows


def projection_onto_probe_caa_span(
    probe_record: dict[str, Any],
    caa_record: dict[str, Any],
    direct_record: dict[str, Any],
    collinear_cosine: float,
) -> dict[str, float | int]:
    probe_caa, _, _ = safe_cosine(probe_record, caa_record)
    matrix = torch.stack([probe_record["direction"], caa_record["direction"]], dim=1)
    u, _, _ = torch.linalg.svd(matrix, full_matrices=False)
    rank = 1 if abs(probe_caa) >= collinear_cosine else 2
    basis = u[:, :rank]
    projection_r2 = float((basis.T @ direct_record["direction"]).square().sum())
    projection_r2 = max(0.0, min(1.0, projection_r2))
    chance_r2 = rank / direct_record["feature_dim"]
    above_chance = (projection_r2 - chance_r2) / max(1.0 - chance_r2, ZERO_NORM_EPS)
    return {
        "probe_caa_span_rank": rank,
        "direct_projection_r2": projection_r2,
        "direct_orthogonal_residual": math.sqrt(max(0.0, 1.0 - projection_r2)),
        "chance_projection_r2": chance_r2,
        "projection_above_chance": above_chance,
    }


def build_direct_span_rows(
    banks: dict[str, dict[str, dict[str, Any]]],
    collinear_cosine: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(banks["optimised"]):
        p, c, d = banks["probe"][key], banks["caa"][key], banks["optimised"][key]
        pc, pc_valid, pc_reason = safe_cosine(p, c)
        pd, pd_valid, pd_reason = safe_cosine(p, d)
        cd, cd_valid, cd_reason = safe_cosine(c, d)
        valid = pc_valid and pd_valid and cd_valid
        span = projection_onto_probe_caa_span(p, c, d, collinear_cosine) if valid else {
            "probe_caa_span_rank": 0,
            "direct_projection_r2": float("nan"),
            "direct_orthogonal_residual": float("nan"),
            "chance_projection_r2": float("nan"),
            "projection_above_chance": float("nan"),
        }
        reasons = [reason for reason in (pc_reason, pd_reason, cd_reason) if reason]
        rows.append({
            **metadata(d),
            "key": key,
            "probe_caa_cosine": pc,
            "probe_caa_absolute_cosine": abs(pc) if pc_valid else float("nan"),
            "probe_optimised_cosine": pd,
            "caa_optimised_cosine": cd,
            **span,
            "collinear_cosine_threshold": collinear_cosine,
            "valid": valid,
            "exclusion_reason": "; ".join(sorted(set(reasons))),
            "direct_shared_optimisation_id": d.get("shared_optimisation_id"),
        })
    return rows


# =============================================================================
# SUMMARIES, SAVING, AND OUTPUT
# =============================================================================

def describe(values: list[float]) -> dict[str, float | int]:
    finite = torch.tensor([value for value in values if math.isfinite(value)], dtype=torch.float64)
    if finite.numel() == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "q1": float("nan"),
            "q3": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(finite.numel()),
        "mean": float(finite.mean()),
        "median": float(torch.quantile(finite, 0.50)),
        "q1": float(torch.quantile(finite, 0.25)),
        "q3": float(torch.quantile(finite, 0.75)),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def grouped_summary(
    rows: list[dict[str, Any]],
    value_field: str,
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    totals: dict[tuple[Any, ...], int] = defaultdict(int)
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        totals[key] += 1
        value = float(row.get(value_field, float("nan")))
        if bool(row.get("valid", True)) and math.isfinite(value):
            groups[key].append(value)

    output = []
    for key in sorted(totals, key=lambda item: tuple(str(value) for value in item)):
        output.append({
            **dict(zip(group_fields, key)),
            "total_rows": totals[key],
            "excluded_rows": totals[key] - len(groups[key]),
            **describe(groups[key]),
        })
    return output


def build_summaries(rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "matched_alignment": grouped_summary(
            rows["matched_alignment_rows"],
            "signed_cosine",
            ("method_pair", "comparison_scope", "label_family", "representation_type"),
        ),
        "common_axis": grouped_summary(
            rows["common_axis_rows"],
            "axis_coherence",
            ("method", "scope", "label_family", "representation_type"),
        ),
        "reachability_final": grouped_summary(
            rows["reachability_final_rows"],
            "signed_cosine",
            ("method", "reachability_task", "representation_type"),
        ),
        "direction_evolution": grouped_summary(
            rows["direction_evolution_rows"],
            "signed_cosine",
            ("method", "comparison_scope", "label_family", "sequence", "representation_type"),
        ),
        "direct_span": grouped_summary(
            rows["direct_span_rows"],
            "direct_projection_r2",
            ("representation_type", "probe_caa_span_rank"),
        ),
    }


def find_summary(rows: list[dict[str, Any]], **filters: Any) -> dict[str, Any] | None:
    return next(
        (row for row in rows if all(row.get(key) == value for key, value in filters.items())),
        None,
    )


def format_summary(row: dict[str, Any] | None) -> str:
    if row is None or not row["count"]:
        return "n/a"
    return f"{row['median']:.4f} [{row['q1']:.4f}, {row['q3']:.4f}] (n={row['count']})"


def print_results(
    paths: dict[str, str],
    audit: dict[str, Any],
    rows: dict[str, list[dict[str, Any]]],
    summaries: dict[str, Any],
    output_path: str,
) -> None:
    probe.heading("COMPARE A* STEERING METHOD VECTORS")
    probe.print_rows(
        "SOURCE AUDIT",
        [
            ("Probe archive", paths["probe"]),
            ("CAA archive", paths["caa"]),
            ("Direct archive", paths["optimised"]),
            ("Probe vectors", audit["vectors_by_method"]["probe"]),
            ("CAA vectors", audit["vectors_by_method"]["caa"]),
            ("Direct vectors", audit["vectors_by_method"]["optimised"]),
            ("Direct unique optimisations", audit["direct_unique_optimisations"]),
            ("Direct zero vectors", audit["zero_vectors_by_method"]["optimised"]),
            ("Direct-compatible keys", audit["direct_compatible_keys"]),
            ("Nonzero three-way keys", audit["three_way_nonzero_keys"]),
            ("Direct max steps", audit["direct_max_steps"]),
            ("Direct horizon assessment", audit["direct_horizon_assessment"]),
        ],
    )

    matched = summaries["matched_alignment"]
    common = summaries["common_axis"]
    reach = summaries["reachability_final"]
    evolution = summaries["direction_evolution"]
    span = summaries["direct_span"]

    matched_rows = []
    for pair in ("probe_caa", "probe_optimised", "caa_optimised"):
        for representation in REPRESENTATION_ORDER:
            row = find_summary(
                matched,
                method_pair=pair,
                comparison_scope="three_way_shared",
                label_family="final_answer",
                representation_type=representation,
            )
            if row is not None:
                matched_rows.append((f"{pair} / {representation}", format_summary(row)))
    probe.print_rows("MATCHED FINAL-ANSWER COSINE - MEDIAN [IQR]", matched_rows)

    axis_rows = []
    for method in METHOD_ORDER:
        for representation in REPRESENTATION_ORDER:
            row = find_summary(
                common,
                method=method,
                scope="final_answer_three_way_shared",
                label_family="final_answer",
                representation_type=representation,
            )
            if row is not None:
                axis_rows.append((f"{method} / {representation}", format_summary(row)))
    probe.print_rows("COMMON-AXIS COHERENCE - MEDIAN [IQR]", axis_rows)

    reach_rows = []
    for method in ("probe", "caa"):
        for reachability_task in REACHABILITY_FINAL_TASK_PAIRS:
            for representation in REPRESENTATION_ORDER:
                row = find_summary(
                    reach,
                    method=method,
                    reachability_task=reachability_task,
                    representation_type=representation,
                )
                if row is not None:
                    reach_rows.append((
                        f"{method} / {reachability_task} / {representation}",
                        format_summary(row),
                    ))
    probe.print_rows("REACHABILITY VS FINAL ANSWER - MEDIAN [IQR]", reach_rows)
    evolution_rows = []
    for family, methods, scope in (
        ("reachability", ("probe", "caa"), "probe_caa_shared"),
        ("final_answer", METHOD_ORDER, "three_way_shared"),
    ):
        for sequence in ("operation", "back"):
            for method in methods:
                for representation in REPRESENTATION_ORDER:
                    row = find_summary(
                        evolution,
                        method=method,
                        comparison_scope=scope,
                        label_family=family,
                        sequence=sequence,
                        representation_type=representation,
                    )
                    if row is not None:
                        evolution_rows.append((
                            f"{method} / {family} / {sequence} / {representation}",
                            format_summary(row),
                        ))
    probe.print_rows("DIRECTION EVOLUTION - MEDIAN [IQR]", evolution_rows)

    span_rows = []
    for representation in REPRESENTATION_ORDER:
        for rank in (1, 2):
            row = find_summary(
                span,
                representation_type=representation,
                probe_caa_span_rank=rank,
            )
            if row is not None:
                span_rows.append((
                    f"{representation} / rank {rank}",
                    format_summary(row),
                ))
    probe.print_rows("DIRECT PROJECTION R2 - MEDIAN [IQR]", span_rows)

    probe.print_rows(
        "OUTPUT AUDIT",
        [
            ("Matched-alignment rows", len(rows["matched_alignment_rows"])),
            ("Common-axis rows", len(rows["common_axis_rows"])),
            ("Reachability/final rows", len(rows["reachability_final_rows"])),
            ("Direction-evolution rows", len(rows["direction_evolution_rows"])),
            ("Direct-span rows", len(rows["direct_span_rows"])),
            ("Output", output_path),
            ("OVERALL STATUS", "PASS"),
        ],
    )
    print("=" * probe.WIDTH)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.span_collinear_cosine <= 1.0:
        raise ValueError("--span-collinear-cosine must be in (0, 1].")

    banks, archives, paths = load_vector_banks(args)
    direct_config = (
        archives["optimised"]
        .get("run_config", {})
        .get("config", {})
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

    audit = audit_vector_banks(banks, archives)
    audit["direct_max_steps"] = (
        int(direct_max_steps) if direct_max_steps is not None else None
    )
    audit["direct_horizon_assessment"] = (
        archives["optimised"]
        .get("audit", {})
        .get("convergence", {})
        .get("horizon_assessment", "not available")
    )
    rows = {
        "matched_alignment_rows": build_matched_alignment_rows(banks),
        "common_axis_rows": build_common_axis_rows(banks),
        "reachability_final_rows": build_reachability_final_rows(banks),
        "direction_evolution_rows": build_direction_evolution_rows(banks),
        "direct_span_rows": build_direct_span_rows(banks, args.span_collinear_cosine),
    }

    for name, expected in EXPECTED_ROW_COUNTS.items():
        actual = len(rows[name])
        if actual != expected:
            raise RuntimeError(
                f"Expected {expected} {name}, produced {actual}."
            )

    summaries = build_summaries(rows)

    payload = {
        "format_version": 1,
        "complete": True,
        "run_tag": RUN_TAG,
        "source_paths": paths,
        "configuration": {
            "zero_norm_epsilon": ZERO_NORM_EPS,
            "unit_norm_tolerance": UNIT_NORM_TOL,
            "span_collinear_cosine": args.span_collinear_cosine,
        },
        "plot_contract": {
            "method_order": list(METHOD_ORDER),
            "method_pairs": [f"{a}_{b}" for a, b in METHOD_PAIRS],
            "representation_order": list(REPRESENTATION_ORDER),
            "operation_sequence": list(OPERATION_SEQUENCE),
            "back_reachability_sequence": list(BACK_REACHABILITY_SEQUENCE),
            "back_final_sequence": list(BACK_FINAL_SEQUENCE),
            "final_shared_checkpoints": list(FINAL_SHARED_CHECKPOINTS),
            "checkpoint_display": CHECKPOINT_DISPLAY,
            "primary_common_axis_scope": "final_answer_three_way_shared",
            "primary_common_axis_metric": "axis_coherence",
        },
        "metric_definitions": {
            "matched_alignment": "signed and absolute cosine between matched method directions",
            "common_axis": "resultant strength and checkpoint-count-adjusted mean pairwise cosine",
            "reachability_final": "cosine between matched reachability and final-answer directions",
            "direction_evolution": "cosine between adjacent semantic checkpoints",
            "direct_span": "squared projection of Direct Optimisation into the Probe-CAA span",
        },
        "audit": audit,
        **rows,
        "summaries": summaries,
    }
    probe.safe_torch_save(payload, args.output)
    print_results(paths, audit, rows, summaries, args.output)


if __name__ == "__main__":
    main()
