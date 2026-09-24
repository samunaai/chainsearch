#!/usr/bin/env python3
"""
make_steering_results_tables.py

Build paper-ready CSV and LaTeX tables from the completed Steering Atlas,
Headcount, and Staying-Power artifacts produced by perform_steering.py. Also
decode representative paired test traces for every reported steering effect.

This script performs CPU-only postprocessing. It never loads the model or the
steering-vector banks and never recomputes generations.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ["COUNTDOWN_STAGE"] = "astar"

import torch

import build_probe_dataset as builder


# =============================================================================
# CONFIGURATION
# =============================================================================

RUN_TAG = "exp34"
ASTAR_DIR = Path("/workspace/user/chainsearch_archives/data_pts/astar")
EVALUATION_DIR = ASTAR_DIR / "steering_evaluations"
EXPERIMENT_ORDER = ("steering-atlas", "headcount", "staying-power")
EXPERIMENT_DISPLAY = {
    "steering-atlas": "Steering Atlas",
    "headcount": "Headcount",
    "staying-power": "Staying Power",
}
METHOD_DISPLAY = {"probe": "Probe", "caa": "CAA", "optimised": "Direct"}
REPRESENTATION_DISPLAY = {
    "attention_head": "Attention head",
    "residual_stream": "Residual stream",
    "mlp_output": "MLP output",
    "final_hidden": "Final hidden",
    "scheduled": "Scheduled",
}

IDENTITY_COLUMNS = (
    "experiment",
    "condition_id",
    "negative_control_of",
    "polarity",
    "method",
    "label_family",
    "task",
    "checkpoint",
    "representation_type",
    "location_name",
    "layer",
    "head",
    "beta",
    "alpha",
    "native_direct",
    "selection_variant",
    "head_count",
    "head_selection",
    "head_scaling",
    "schedule",
    "source_validation_rows",
    "units",
    "unique_traces",
)

EFFECT_COLUMNS = (
    "primary_measure",
    "baseline_primary",
    "steered_primary",
    "primary_effect",
    "primary_ci_low",
    "primary_ci_high",
    "baseline_expr_target_correct_rate",
    "steered_expr_target_correct_rate",
    "delta_expr_target_correct_rate",
    "delta_expr_target_correct_ci_low",
    "delta_expr_target_correct_ci_high",
    "rescue_rate",
    "damage_rate",
    "net_rescue_rate",
    "baseline_valid_trace_rate",
    "steered_valid_trace_rate",
    "delta_valid_trace_rate",
    "delta_valid_trace_ci_low",
    "delta_valid_trace_ci_high",
    "baseline_solved_valid_trace_rate",
    "steered_solved_valid_trace_rate",
    "delta_solved_valid_trace_rate",
    "delta_solved_valid_trace_ci_low",
    "delta_solved_valid_trace_ci_high",
    "baseline_next_state_viable_rate",
    "steered_next_state_viable_rate",
    "delta_next_state_viable_rate",
    "delta_next_state_viable_ci_low",
    "delta_next_state_viable_ci_high",
    "baseline_future_reachable_state_fraction_mean",
    "steered_future_reachable_state_fraction_mean",
    "delta_future_reachable_state_fraction_mean",
    "delta_future_reachable_state_fraction_ci_low",
    "delta_future_reachable_state_fraction_ci_high",
    "baseline_operation_validity_rate",
    "steered_operation_validity_rate",
    "delta_operation_validity_rate",
    "baseline_state_consistency_rate",
    "steered_state_consistency_rate",
    "delta_state_consistency_rate",
    "continuation_change_rate",
    "delta_generated_length_mean",
    "delta_generated_back_count_mean",
    "baseline_failure_counts",
    "steered_failure_counts",
)

LATEX_EFFECT_COLUMNS = (
    "method",
    "label_family",
    "checkpoint",
    "representation_type",
    "location_name",
    "selection_variant",
    "polarity",
    "beta",
    "baseline_primary",
    "steered_primary",
    "primary_effect",
    "primary_ci_low",
    "primary_ci_high",
    "rescue_rate",
    "damage_rate",
    "delta_valid_trace_rate",
)

DIRECTIONALITY_COLUMNS = (
    "experiment",
    "positive_condition_id",
    "negative_condition_id",
    "method",
    "label_family",
    "task",
    "checkpoint",
    "representation_type",
    "location_name",
    "selection_variant",
    "schedule",
    "positive_beta",
    "negative_beta",
    "primary_measure",
    "positive_primary_effect",
    "negative_primary_effect",
    "positive_minus_negative_primary_effect",
    "positive_accuracy_effect",
    "negative_accuracy_effect",
    "positive_minus_negative_accuracy_effect",
    "positive_validity_effect",
    "negative_validity_effect",
    "positive_minus_negative_validity_effect",
    "positive_rescue_rate",
    "negative_rescue_rate",
    "positive_damage_rate",
    "negative_damage_rate",
)

COVERAGE_COLUMNS = (
    "experiment",
    "coarse_validation_conditions",
    "refined_validation_conditions",
    "selected_positive_or_native_conditions",
    "matched_negative_test_controls",
    "selected_test_conditions",
    "test_baseline_cells",
    "test_baseline_units",
    "test_condition_units",
    "minimum_baseline_next_token_reproduction_rate",
    "minimum_baseline_suffix_reproduction_rate",
    "overall_status",
)


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build systematic steering-result tables and paired trace examples."
    )
    parser.add_argument("--input", type=Path, help="Completed steering-suite manifest.")
    parser.add_argument("--run-tag", default=RUN_TAG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--examples-per-effect", type=int, default=5)
    parser.add_argument("--no-examples", action="store_true")
    args = parser.parse_args()
    if args.examples_per_effect < 0:
        raise ValueError("--examples-per-effect must be non-negative.")
    if args.input is None:
        args.input = EVALUATION_DIR / f"{args.run_tag}_steering_suite.pt"
    if args.output_dir is None:
        args.output_dir = Path(f"{args.run_tag}_steering_result_tables")
    return args


def load_pt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a dictionary in {path}.")
    return value

def file_signature(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }

def finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def difference(left: Any, right: Any) -> float:
    left_value, right_value = finite_float(left), finite_float(right)
    if not math.isfinite(left_value) or not math.isfinite(right_value):
        return float("nan")
    return left_value - right_value


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.8g}"
    if isinstance(value, (tuple, list)):
        return "; ".join(stringify(item) for item in value)
    if isinstance(value, dict):
        return "; ".join(f"{key}={value[key]}" for key in sorted(value))
    return str(value)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: stringify(row.get(column)) for column in columns})
    os.replace(temporary, path)


def latex_escape(value: Any) -> str:
    text = stringify(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def write_latex(path: Path, rows: list[dict[str, Any]], columns: Sequence[str]) -> None:
    alignment = "l" * len(columns)
    lines = [
        r"\begin{tabular}{" + alignment + "}",
        r"\toprule",
        " & ".join(latex_escape(column) for column in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(row.get(column)) for column in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    atomic_write_text(path, "\n".join(lines))


def unique_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        condition_id = str(row["condition_id"])
        if condition_id in seen:
            raise RuntimeError(f"Duplicate test condition row {condition_id!r}.")
        seen.add(condition_id)
        output.append(row)
    return output


# =============================================================================
# SUITE LOADING AND TABLE ROWS
# =============================================================================

def load_suite(path: Path, run_tag: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = load_pt(path)
    required = {"complete", "run_tag", "experiments", "experiment_signatures", "overall_status"}
    missing = required - set(manifest)
    if missing:
        raise KeyError(f"Suite manifest is missing fields {sorted(missing)}.")
    if not bool(manifest["complete"]) or manifest["overall_status"] != "PASS":
        raise RuntimeError("The steering-suite manifest is incomplete or did not pass its audit.")
    if manifest["run_tag"] != run_tag:
        raise RuntimeError(f"Suite run tag {manifest['run_tag']!r} != {run_tag!r}.")

    experiment_paths = dict(manifest["experiments"])
    experiment_signatures = dict(manifest["experiment_signatures"])
    missing_experiments = set(EXPERIMENT_ORDER) - (
        set(experiment_paths) & set(experiment_signatures)
    )
    if missing_experiments:
        raise KeyError(
            f"Suite manifest is missing experiment paths or signatures "
            f"{sorted(missing_experiments)}."
        )

    artifacts: dict[str, dict[str, Any]] = {}
    for experiment in EXPERIMENT_ORDER:
        artifact_path = Path(experiment_paths[experiment])
        observed_signature = file_signature(artifact_path)
        if observed_signature != experiment_signatures[experiment]:
            raise RuntimeError(
                f"{experiment} artifact no longer matches the suite manifest."
            )
        artifact = load_pt(artifact_path)
        required_artifact = {
            "complete",
            "run_tag",
            "run_config",
            "baseline_rows",
            "coarse_validation_rows",
            "refined_validation_rows",
            "validation_selected_condition_ids",
            "negative_control_of",
            "test_condition_rows",
            "audit",
        }
        missing = required_artifact - set(artifact)
        if missing:
            raise KeyError(f"{experiment} artifact is missing fields {sorted(missing)}.")
        if not bool(artifact["complete"]) or artifact["run_tag"] != run_tag:
            raise RuntimeError(f"{experiment} artifact is incomplete or has the wrong run tag.")
        if artifact["run_config"].get("experiment") != experiment:
            raise RuntimeError(f"Artifact labeled {experiment!r} has a different experiment configuration.")
        if artifact["audit"].get("overall_status") != "PASS":
            raise RuntimeError(f"{experiment} artifact did not pass its final audit.")
        artifacts[experiment] = artifact
    return manifest, artifacts


def primary_fields(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    if row.get("label_family") != "reachability":
        stem = "expr_target_correct"
        return (
            stem,
            f"baseline_{stem}_rate",
            f"steered_{stem}_rate",
            f"delta_{stem}_rate",
            f"delta_{stem}_ci",
        )
    if row.get("representation_type") == "scheduled":
        stem = "future_reachable_state_fraction"
        return (
            stem,
            f"baseline_{stem}_mean",
            f"steered_{stem}_mean",
            f"delta_{stem}_mean",
            f"delta_{stem}_ci",
        )
    stem = "next_state_viable"
    return (
        stem,
        f"baseline_{stem}_rate",
        f"steered_{stem}_rate",
        f"delta_{stem}_rate",
        f"delta_{stem}_ci",
    )


def effect_row(
    experiment: str,
    row: dict[str, Any],
    negative_control_of: dict[str, str],
    negative_condition_for: dict[str, str],
) -> dict[str, Any]:
    condition_id = str(row["condition_id"])
    native = bool(row.get("native_direct", False))
    if native:
        polarity = "native"
    elif condition_id in negative_control_of:
        polarity = "negative"
    else:
        polarity = "positive"
    measure, baseline_field, steered_field, delta_field, ci_stem = primary_fields(row)
    output = {
        column: row.get(column)
        for column in (*IDENTITY_COLUMNS, *EFFECT_COLUMNS)
        if column in row
    }
    output.update(
        experiment=experiment,
        condition_id=condition_id,
        negative_control_of=negative_control_of.get(condition_id),
        negative_condition_id=negative_condition_for.get(condition_id),
        polarity=polarity,
        source_validation_rows=row.get("source_validation_rows", ()),
        primary_measure=measure,
        baseline_primary=row.get(baseline_field),
        steered_primary=row.get(steered_field),
        primary_effect=row.get(delta_field),
        primary_ci_low=row.get(f"{ci_stem}_low"),
        primary_ci_high=row.get(f"{ci_stem}_high"),
    )
    return output


def all_effect_rows(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for experiment in EXPERIMENT_ORDER:
        artifact = artifacts[experiment]
        controls = {str(key): str(value) for key, value in artifact["negative_control_of"].items()}
        inverse = {positive: negative for negative, positive in controls.items()}
        if len(inverse) != len(controls):
            raise RuntimeError(f"{experiment} has more than one negative control for a positive condition.")
        for row in unique_rows(artifact["test_condition_rows"]):
            output.append(effect_row(experiment, row, controls, inverse))
    return output


def directionality_rows(
    artifacts: dict[str, dict[str, Any]],
    effect_by_key: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for experiment in EXPERIMENT_ORDER:
        controls = artifacts[experiment]["negative_control_of"]
        for negative_id, positive_id in sorted(controls.items(), key=lambda item: str(item[1])):
            positive = effect_by_key[(experiment, str(positive_id))]
            negative = effect_by_key[(experiment, str(negative_id))]
            if positive["primary_measure"] != negative["primary_measure"]:
                raise RuntimeError(f"Primary-measure mismatch for {positive_id} and {negative_id}.")
            output.append({
                "experiment": experiment,
                "positive_condition_id": str(positive_id),
                "negative_condition_id": str(negative_id),
                "method": positive.get("method"),
                "label_family": positive.get("label_family"),
                "task": positive.get("task"),
                "checkpoint": positive.get("checkpoint"),
                "representation_type": positive.get("representation_type"),
                "location_name": positive.get("location_name"),
                "selection_variant": positive.get("selection_variant"),
                "schedule": positive.get("schedule"),
                "positive_beta": positive.get("beta"),
                "negative_beta": negative.get("beta"),
                "primary_measure": positive["primary_measure"],
                "positive_primary_effect": positive.get("primary_effect"),
                "negative_primary_effect": negative.get("primary_effect"),
                "positive_minus_negative_primary_effect": difference(
                    positive.get("primary_effect"), negative.get("primary_effect")
                ),
                "positive_accuracy_effect": positive.get("delta_expr_target_correct_rate"),
                "negative_accuracy_effect": negative.get("delta_expr_target_correct_rate"),
                "positive_minus_negative_accuracy_effect": difference(
                    positive.get("delta_expr_target_correct_rate"),
                    negative.get("delta_expr_target_correct_rate"),
                ),
                "positive_validity_effect": positive.get("delta_valid_trace_rate"),
                "negative_validity_effect": negative.get("delta_valid_trace_rate"),
                "positive_minus_negative_validity_effect": difference(
                    positive.get("delta_valid_trace_rate"),
                    negative.get("delta_valid_trace_rate"),
                ),
                "positive_rescue_rate": positive.get("rescue_rate"),
                "negative_rescue_rate": negative.get("rescue_rate"),
                "positive_damage_rate": positive.get("damage_rate"),
                "negative_damage_rate": negative.get("damage_rate"),
            })
    return output


def coverage_rows(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for experiment in EXPERIMENT_ORDER:
        artifact = artifacts[experiment]
        audit = artifact["audit"]
        baseline_rows = [row for row in artifact["baseline_rows"] if row.get("phase") == "test"]
        test_rows = artifact["test_condition_rows"]
        output.append({
            "experiment": experiment,
            "coarse_validation_conditions": audit["coarse_validation_conditions"],
            "refined_validation_conditions": audit["refined_validation_conditions"],
            "selected_positive_or_native_conditions": audit["selected_positive_or_native_conditions"],
            "matched_negative_test_controls": audit["matched_negative_test_controls"],
            "selected_test_conditions": audit["selected_test_conditions"],
            "test_baseline_cells": len(baseline_rows),
            "test_baseline_units": sum(int(row["units"]) for row in baseline_rows),
            "test_condition_units": sum(int(row["units"]) for row in test_rows),
            "minimum_baseline_next_token_reproduction_rate": audit[
                "minimum_baseline_next_token_reproduction_rate"
            ],
            "minimum_baseline_suffix_reproduction_rate": audit[
                "minimum_baseline_suffix_reproduction_rate"
            ],
            "overall_status": audit["overall_status"],
        })
    return output


# =============================================================================
# REPRESENTATIVE PAIRED TRACES
# =============================================================================

def load_shard(path: Any, run_tag: str) -> dict[str, Any]:
    shard = load_pt(Path(str(path)))
    if not bool(shard.get("complete", False)) or shard.get("run_tag") != run_tag:
        raise RuntimeError(f"Invalid steering shard {path}.")
    required = {
        "source_row",
        "record_pos",
        "cache_idx",
        "generated_idx",
        "weights",
        "target",
        "initial_state",
        "labels",
        "generated_flat_ids",
        "generated_offsets",
        "generated_lengths",
    }
    missing = required - set(shard)
    if missing:
        raise KeyError(f"Shard {path} is missing fields {sorted(missing)}.")
    return shard


def unpack_sequences(shard: dict[str, Any]) -> list[list[int]]:
    flat = shard["generated_flat_ids"]
    offsets = shard["generated_offsets"]
    lengths = shard["generated_lengths"]
    return [
        flat[int(offsets[index]): int(offsets[index]) + int(lengths[index])]
        .to(torch.int64)
        .tolist()
        for index in range(int(lengths.numel()))
    ]


def format_trace(token_ids: list[int], tokenizer: Any) -> str:
    text = tokenizer.decode(token_ids).strip()
    text = re.sub(r"\s*;\s*", " ;\n", text)
    text = re.sub(r"(?<!^)\s+(?=EXPR\b)", "\n", text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def tensor_item(shard: dict[str, Any], field: str, index: int) -> Any:
    if field not in shard:
        return None
    value = shard[field][index]
    if isinstance(value, torch.Tensor):
        return value.tolist() if value.ndim else value.item()
    return value


def outcome_bucket(
    baseline_labels: dict[str, torch.Tensor],
    steered_labels: dict[str, torch.Tensor],
    baseline_ids: list[int],
    steered_ids: list[int],
    index: int,
) -> str:
    baseline_correct = bool(baseline_labels["expr_target_correct"][index])
    steered_correct = bool(steered_labels["expr_target_correct"][index])
    baseline_valid = bool(baseline_labels["valid_trace"][index])
    steered_valid = bool(steered_labels["valid_trace"][index])
    changed = baseline_ids != steered_ids
    if not baseline_correct and steered_correct and steered_valid:
        return "valid rescue"
    if not baseline_correct and steered_correct:
        return "answer rescue with invalid trace"
    if baseline_correct and not steered_correct and baseline_valid:
        return "damage to a valid baseline"
    if baseline_correct and not steered_correct:
        return "answer damage"
    if changed and baseline_valid and steered_valid:
        return "changed continuation, both valid"
    if changed:
        return "changed continuation, same correctness"
    return "unchanged continuation"


EXAMPLE_BUCKET_ORDER = (
    "valid rescue",
    "answer rescue with invalid trace",
    "damage to a valid baseline",
    "answer damage",
    "changed continuation, both valid",
    "changed continuation, same correctness",
    "unchanged continuation",
)


def choose_example_indices(
    baseline_shard: dict[str, Any],
    steered_shard: dict[str, Any],
    count: int,
) -> tuple[list[int], list[list[int]], list[list[int]], dict[int, str]]:
    baseline_sequences = unpack_sequences(baseline_shard)
    steered_sequences = unpack_sequences(steered_shard)
    baseline_labels = baseline_shard["labels"]
    steered_labels = steered_shard["labels"]
    if len(baseline_sequences) != len(steered_sequences):
        raise RuntimeError("Baseline and steered shards contain different sequence counts.")

    buckets = {name: [] for name in EXAMPLE_BUCKET_ORDER}
    case_by_index: dict[int, str] = {}
    for index, (baseline_ids, steered_ids) in enumerate(zip(baseline_sequences, steered_sequences)):
        case = outcome_bucket(
            baseline_labels, steered_labels, baseline_ids, steered_ids, index
        )
        buckets[case].append(index)
        case_by_index[index] = case

    chosen: list[int] = []
    for case in EXAMPLE_BUCKET_ORDER:
        if buckets[case]:
            chosen.append(buckets[case][0])
            if len(chosen) == count:
                return chosen, baseline_sequences, steered_sequences, case_by_index
    for case in EXAMPLE_BUCKET_ORDER:
        for index in buckets[case][1:]:
            chosen.append(index)
            if len(chosen) == count:
                return chosen, baseline_sequences, steered_sequences, case_by_index
    return chosen, baseline_sequences, steered_sequences, case_by_index


def baseline_paths(artifact: dict[str, Any]) -> dict[tuple[str, str], str]:
    paths = {
        (str(row["task"]), str(row["checkpoint"])): str(row["shard_path"])
        for row in artifact["baseline_rows"]
        if row.get("phase") == "test"
    }
    if not paths:
        raise RuntimeError("Artifact contains no held-out test baseline shards.")
    return paths


def validate_paired_shards(baseline: dict[str, Any], steered: dict[str, Any], condition_id: str) -> None:
    for field in ( "source_row", "record_pos", "cache_idx", "generated_idx", "weights", "target", "initial_state"):
        if not torch.equal(baseline[field], steered[field]):
            raise RuntimeError(f"Baseline/steered {field} mismatch for {condition_id}.")


def write_representative_traces(
    path: Path,
    artifacts: dict[str, dict[str, Any]],
    run_tag: str,
    examples_per_effect: int,
) -> tuple[int, int]:
    tokenizer = builder.pm.Tokenizer()
    lines = [
        "CHAINSEARCH STEERING REPRESENTATIVE PAIRED TRACES",
        "",
        f"Run tag  {run_tag}",
        f"Examples requested per effect  {examples_per_effect}",
        "Selection prioritizes outcome diversity, valid rescues, and damage to valid baselines.",
        "These examples are descriptive. Effect rates come from the paired held-out tables.",
        "",
    ]
    effect_count = 0
    example_count = 0

    for experiment in EXPERIMENT_ORDER:
        artifact = artifacts[experiment]
        baselines = baseline_paths(artifact)
        controls = {str(key): str(value) for key, value in artifact["negative_control_of"].items()}
        for row in unique_rows(artifact["test_condition_rows"]):
            condition_id = str(row["condition_id"])
            key = (str(row["task"]), str(row["checkpoint"]))
            if key not in baselines:
                raise RuntimeError(f"No test baseline for {condition_id} at {key}.")
            baseline = load_shard(baselines[key], run_tag)
            steered = load_shard(row["shard_path"], run_tag)
            validate_paired_shards(baseline, steered, condition_id)
            chosen, baseline_sequences, steered_sequences, cases = choose_example_indices(
                baseline, steered, examples_per_effect
            )
            effect_count += 1

            lines.extend([
                "=" * 120,
                f"EFFECT {effect_count}",
                f"Experiment  {EXPERIMENT_DISPLAY[experiment]}",
                f"Condition ID  {condition_id}",
                f"Negative control of  {controls.get(condition_id, '')}",
                f"Method  {METHOD_DISPLAY.get(str(row['method']), row['method'])}",
                f"Label family  {row['label_family']}",
                f"Task  {row['task']}",
                f"Checkpoint  {row['checkpoint']}",
                f"Representation  {REPRESENTATION_DISPLAY.get(str(row['representation_type']), row['representation_type'])}",
                f"Location  {row['location_name']}",
                f"Selection variant  {row.get('selection_variant', 'single')}",
                f"Schedule  {row.get('schedule', '')}",
                f"Beta  {stringify(row.get('beta'))}",
                f"Final-answer effect  {stringify(row.get('delta_expr_target_correct_rate'))}",
                f"Rescue rate  {stringify(row.get('rescue_rate'))}",
                f"Damage rate  {stringify(row.get('damage_rate'))}",
                f"Valid-trace effect  {stringify(row.get('delta_valid_trace_rate'))}",
                "",
            ])

            for local_number, index in enumerate(chosen, start=1):
                baseline_labels = baseline["labels"]
                steered_labels = steered["labels"]
                target = tensor_item(steered, "target", index)
                initial_state = tensor_item(steered, "initial_state", index)
                lines.extend([
                    "-" * 120,
                    f"EXAMPLE {local_number}",
                    f"Outcome case  {cases[index]}",
                    f"Source row  {tensor_item(steered, 'source_row', index)}",
                    f"Record position  {tensor_item(steered, 'record_pos', index)}",
                    f"Intervention generated-token index  {tensor_item(steered, 'generated_idx', index)}",
                    f"Target  {target}",
                    f"Initial state  {initial_state}",
                    f"Baseline answer correct  {'yes' if bool(baseline_labels['expr_target_correct'][index]) else 'no'}",
                    f"Steered answer correct  {'yes' if bool(steered_labels['expr_target_correct'][index]) else 'no'}",
                    f"Baseline valid trace  {'yes' if bool(baseline_labels['valid_trace'][index]) else 'no'}",
                    f"Steered valid trace  {'yes' if bool(steered_labels['valid_trace'][index]) else 'no'}",
                    "",
                    "BASELINE TRACE",
                    format_trace(baseline_sequences[index], tokenizer),
                    "",
                    "STEERED TRACE",
                    format_trace(steered_sequences[index], tokenizer),
                    "",
                ])
                example_count += 1

    atomic_write_text(path, "\n".join(lines).rstrip() + "\n")
    return effect_count, example_count


# =============================================================================
# OUTPUT
# =============================================================================

def output_table_pair(
    output_dir: Path,
    stem: str,
    rows: list[dict[str, Any]],
    csv_columns: Sequence[str],
    latex_columns: Sequence[str],
) -> list[Path]:
    csv_path = output_dir / f"{stem}.csv"
    latex_path = output_dir / f"{stem}.tex"
    write_csv(csv_path, rows, csv_columns)
    write_latex(latex_path, rows, latex_columns)
    return [csv_path, latex_path]


def main() -> None:
    args = parse_args()
    manifest, artifacts = load_suite(args.input, args.run_tag)
    effects = all_effect_rows(artifacts)
    effect_by_key = {
        (str(row["experiment"]), str(row["condition_id"])): row
        for row in effects
    }
    if len(effect_by_key) != len(effects):
        raise RuntimeError("Effect rows are not unique by experiment and condition ID.")
    contrasts = directionality_rows(artifacts, effect_by_key)
    coverage = coverage_rows(artifacts)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    all_columns = (*IDENTITY_COLUMNS, "negative_condition_id", *EFFECT_COLUMNS)
    outputs.extend(output_table_pair(
        args.output_dir,
        "all_test_effects",
        effects,
        all_columns,
        ("experiment", *LATEX_EFFECT_COLUMNS),
    ))

    for experiment in EXPERIMENT_ORDER:
        experiment_rows = [row for row in effects if row["experiment"] == experiment]
        outputs.extend(output_table_pair(
            args.output_dir,
            f"{experiment.replace('-', '_')}_effects",
            experiment_rows,
            all_columns,
            LATEX_EFFECT_COLUMNS,
        ))

    outputs.extend(output_table_pair(
        args.output_dir,
        "directionality_contrasts",
        contrasts,
        DIRECTIONALITY_COLUMNS,
        (
            "experiment",
            "method",
            "label_family",
            "checkpoint",
            "representation_type",
            "selection_variant",
            "positive_primary_effect",
            "negative_primary_effect",
            "positive_minus_negative_primary_effect",
        ),
    ))
    outputs.extend(output_table_pair(
        args.output_dir,
        "coverage",
        coverage,
        COVERAGE_COLUMNS,
        COVERAGE_COLUMNS,
    ))

    effect_count = example_count = 0
    if not args.no_examples and args.examples_per_effect > 0:
        trace_path = args.output_dir / "representative_paired_traces.txt"
        effect_count, example_count = write_representative_traces(
            trace_path,
            artifacts,
            args.run_tag,
            args.examples_per_effect,
        )
        outputs.append(trace_path)

    manifest_path = args.output_dir / "table_manifest.txt"
    manifest_lines = [
        f"Input suite  {args.input}",
        f"Run tag  {args.run_tag}",
        f"Suite elapsed minutes  {stringify(manifest.get('elapsed_minutes'))}",
        f"Test effects  {len(effects)}",
        f"Directionality contrasts  {len(contrasts)}",
        f"Trace effect sections  {effect_count}",
        f"Representative paired traces  {example_count}",
        "",
        "OUTPUTS",
        *[str(path) for path in outputs],
    ]
    atomic_write_text(manifest_path, "\n".join(manifest_lines) + "\n")

    print(f"Input suite:  {args.input}")
    print(f"Output:       {args.output_dir}")
    print(f"Test effects: {len(effects)}")
    print(f"Contrasts:    {len(contrasts)}")
    print(f"Examples:     {example_count}")
    print(f"Manifest:     {manifest_path}")


if __name__ == "__main__":
    main()
