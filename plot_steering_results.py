#!/usr/bin/env python3
"""
plot_steering_results.py

Plot validation and held-out test results saved by perform_steering.py. This
script reads the completed steering suite, extracts exact paired validation
and held-out counts, and never loads the model or the original vector banks.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch


# =============================================================================
# CONFIGURATION
# =============================================================================

RUN_TAG = "exp34"
ASTAR_DIR = Path("/workspace/sam/minireason_archives/data_pts/astar")
DEFAULT_INPUT = ASTAR_DIR / "steering_evaluations" / f"{RUN_TAG}_steering_suite.pt"
DEFAULT_OUTPUT_DIR = Path(f"{RUN_TAG}_steering_results_plots")
DEFAULT_DPI = 220

EXPERIMENT_ORDER = ("steering-atlas", "headcount", "staying-power")
REQUIRED_EXPERIMENTS = EXPERIMENT_ORDER
EXPERIMENT_DISPLAY = {
    "steering-atlas": "Where and When Test",
    "headcount": "How Many Heads Test",
    "staying-power": "Intervention Frequency Test",
}

COLOR_BG = "#FCFCFD"
COLOR_PANEL = "#F2F6FF"
COLOR_GRID = "#C9D6EE"
COLOR_TEXT = "#26364F"
COLOR_BLUE = "#1C4587"
COLOR_LIGHT_BLUE = "#7CA6DE"
COLOR_RED = "#D57758"
COLOR_SLATE = "#5F7896"
COLOR_GREY = "#8A94A6"
COLOR_GAIN_FILL = "#7EA5D6"
COLOR_LOSS_FILL = "#E8A98B"
COLOR_GAIN_HEATMAP = "#AFC9EA"
COLOR_LOSS_HEATMAP = "#F2C2AD"
COLOR_TOKEN_POSITION = "#44474D"

COLOR_TITLE = "#000000"
METHOD_ORDER = ("probe", "caa", "optimised")
METHOD_DISPLAY = {"probe": "Probe", "caa": "CAA", "optimised": "Direct"}
METHOD_COLOR = {"probe": COLOR_BLUE, "caa": COLOR_LIGHT_BLUE, "optimised": COLOR_RED}
FAMILY_ORDER = ("final_answer", "reachability")
FAMILY_DISPLAY = {"final_answer": "Final answer", "reachability": "Reachability"}
REPRESENTATION_ORDER = ("attention_head", "residual_stream", "mlp_output", "final_hidden", "scheduled")
REPRESENTATION_DISPLAY = {
    "attention_head": "Attention heads",
    "residual_stream": "Residual stream",
    "mlp_output": "MLP output",
    "final_hidden": "Final hidden",
    "scheduled": "Scheduled",
}
HEAD_VARIANT_ORDER = ("top4_fixed", "top12_fixed", "top12_per_head", "random12_fixed", "top22_fixed", "top24_fixed")
HEAD_VARIANT_DISPLAY = {
    "top4_fixed": "Top 4\nfixed total",
    "top12_fixed": "Top 12\nfixed total",
    "top12_per_head": "Top 12\nper head",
    "random12_fixed": "Random 12\nfixed total",
    "top22_fixed": "All 22\nfixed total",
    "top24_fixed": "All 24\nfixed total",
}
SCHEDULE_ORDER = ("single", "repeated", "multipoint")
SCHEDULE_DISPLAY = {"single": "Single", "repeated": "Repeated", "multipoint": "Multipoint"}

FIGURES_DIR = "figures"
EVIDENCE_DIR = "evidence"
CHECKPOINT_ORDER = (
    "pre_first_operand", "first_operand_end", "operator", "second_operand_end",
    "result_end", "resulting_state_semicolon", "pre_back", "back_token",
    "restored_state_semicolon", "success_token",
)
CHECKPOINT_DISPLAY = {
    "pre_first_operand": "Before first operand",
    "first_operand_end": "First operand",
    "operator": "Operator",
    "second_operand_end": "Second operand",
    "result_end": "Result",
    "resulting_state_semicolon": "Resulting state",
    "pre_back": "Before BACK",
    "back_token": "BACK",
    "restored_state_semicolon": "Restored state",
    "success_token": "SUCCESS",
}

plt.rcParams.update({
    "font.size": 10.5,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.titlesize": 14,
    "figure.facecolor": COLOR_BG,
    "savefig.facecolor": COLOR_BG,
    "axes.facecolor": COLOR_PANEL,
    "axes.edgecolor": COLOR_BLUE,
    "axes.labelcolor": COLOR_TEXT,
    "axes.titlecolor": COLOR_BLUE,
    "text.color": COLOR_TEXT,
    "xtick.color": COLOR_BLUE,
    "ytick.color": COLOR_BLUE,
    "grid.color": COLOR_GRID,
    "legend.facecolor": COLOR_BG,
    "legend.edgecolor": COLOR_BLUE,
    "legend.framealpha": 0.92,
})


# =============================================================================
# LOADING AND HELPERS
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot saved causal steering results.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plotlist", nargs="+", default=["all"])
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--list-plots", action="store_true")
    return parser.parse_args()


def file_signature(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {"path": str(resolved), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def load_results(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "complete", "run_tag", "run_config", "baseline_rows", "coarse_validation_rows",
        "refined_validation_rows", "negative_control_of", "test_condition_rows", "audit",
    }
    missing = required - set(data)
    if missing:
        raise KeyError(f"Steering artifact is missing fields: {sorted(missing)}")
    if not bool(data["complete"]) or data["run_tag"] != RUN_TAG:
        raise RuntimeError("Steering artifact is incomplete or has the wrong run tag.")
    if data.get("audit", {}).get("overall_status") != "PASS":
        raise RuntimeError("Steering artifact did not pass its final audit.")
    return data


def load_suite(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = torch.load(path, map_location="cpu", weights_only=False)
    required = {"complete", "run_tag", "experiments", "experiment_signatures", "overall_status"}
    missing = required - set(manifest)
    if missing:
        raise KeyError(f"Steering-suite manifest is missing fields: {sorted(missing)}")
    if not bool(manifest["complete"]) or manifest["run_tag"] != RUN_TAG:
        raise RuntimeError("Steering-suite manifest is incomplete or has the wrong run tag.")
    if manifest["overall_status"] != "PASS":
        raise RuntimeError("Steering suite did not pass its final audit.")

    paths = dict(manifest["experiments"])
    signatures = dict(manifest["experiment_signatures"])
    artifacts: dict[str, dict[str, Any]] = {}
    for experiment in REQUIRED_EXPERIMENTS:
        if experiment not in paths or experiment not in signatures:
            raise KeyError(f"Steering-suite manifest is missing {experiment!r}.")
        artifact_path = Path(paths[experiment])
        if file_signature(artifact_path) != signatures[experiment]:
            raise RuntimeError(f"{experiment} artifact no longer matches the suite manifest.")
        artifact = load_results(artifact_path)
        if artifact["run_config"].get("experiment") != experiment:
            raise RuntimeError(f"Artifact labeled {experiment!r} has a different configuration.")
        artifacts[experiment] = artifact
    return manifest, artifacts


def rows_where(rows: list[dict[str, Any]], **filters: Any) -> list[dict[str, Any]]:
    return [row for row in rows if all(row.get(name) == value for name, value in filters.items())]

def unique_condition_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        condition_id = str(row["condition_id"])
        if condition_id in seen:
            raise RuntimeError(f"Duplicate condition row: {condition_id}")
        seen.add(condition_id)
        output.append(row)
    return output


COUNT_COLUMNS = (
    "experiment",
    "phase",
    "condition_id",
    "method",
    "label_family",
    "task",
    "checkpoint",
    "representation_type",
    "location_name",
    "beta",
    "native_direct",
    "selection_variant",
    "schedule",
    "raw_paired_case_count",
    "raw_unique_trace_count",
    "raw_unique_problem_count",
    "raw_baseline_correct_count",
    "raw_steered_correct_count",
    "raw_rescue_count",
    "raw_damage_count",
    "raw_net_correct_count",
    "raw_changed_continuation_count",
    "raw_affected_unique_trace_count",
    "raw_affected_unique_problem_count",
    "raw_valid_rescue_count",
    "raw_valid_damage_count",
    "raw_net_valid_count",
    "raw_next_state_viable_gain_count",
    "raw_next_state_viable_loss_count",
    "raw_net_next_state_viable_count",
    "raw_future_all_reachable_gain_count",
    "raw_future_all_reachable_loss_count",
    "raw_net_future_all_reachable_count",
    "raw_length_increase_count",
    "raw_length_decrease_count",
    "raw_length_unchanged_count",
    "raw_back_increase_count",
    "raw_back_decrease_count",
    "raw_back_unchanged_count",
    "raw_parseable_gain_count",
    "raw_parseable_loss_count",
    "raw_net_parseable_count",
)


def load_paired_shard(path_value: Any) -> dict[str, Any]:
    path = Path(str(path_value))
    if not path.is_file():
        raise FileNotFoundError(path)
    shard = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "complete", "run_tag", "source_row", "record_pos", "cache_idx",
        "generated_idx", "target", "initial_state", "weights", "labels",
        "generated_flat_ids", "generated_offsets", "generated_lengths",
    }
    missing = required - set(shard)
    if missing:
        raise KeyError(f"Steering shard {path} is missing fields {sorted(missing)}.")
    if not bool(shard["complete"]) or shard["run_tag"] != RUN_TAG:
        raise RuntimeError(f"Steering shard {path} is incomplete or has the wrong run tag.")
    return shard


def unpack_generated_sequences(shard: dict[str, Any]) -> list[list[int]]:
    flat = shard["generated_flat_ids"]
    offsets = shard["generated_offsets"]
    lengths = shard["generated_lengths"]
    return [
        flat[int(offsets[index]): int(offsets[index]) + int(lengths[index])]
        .to(torch.int64)
        .tolist()
        for index in range(int(lengths.numel()))
    ]


def format_trace_text(token_ids: list[int], tokenizer: Any) -> str:
    text = tokenizer.decode(token_ids).strip()
    text = re.sub(r"\s*;\s*", " ;\n", text)
    text = re.sub(r"(?<!^)\s+(?=EXPR\b)", "\n", text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def baseline_phase_paths(
    artifact: dict[str, Any],
    phase: str,
) -> dict[tuple[str, str], str]:
    paths: dict[tuple[str, str], str] = {}
    for row in artifact["baseline_rows"]:
        if row.get("phase") != phase:
            continue
        key = (str(row["task"]), str(row["checkpoint"]))
        if key in paths:
            raise RuntimeError(f"Duplicate {phase} baseline shard for {key}.")
        paths[key] = str(row["shard_path"])
    if not paths:
        raise RuntimeError(f"Steering artifact contains no {phase} baseline shards.")
    return paths


def boolean_label(shard: dict[str, Any], field: str, count: int) -> torch.Tensor:
    labels = shard["labels"]
    if field not in labels:
        raise KeyError(f"Steering shard has no {field!r} label.")
    values = labels[field].bool().cpu()
    if values.ndim != 1 or int(values.numel()) != count:
        raise RuntimeError(f"Label {field!r} does not contain exactly {count} cases.")
    return values


def numeric_label(shard: dict[str, Any], field: str, count: int) -> torch.Tensor:
    labels = shard["labels"]
    if field not in labels:
        raise KeyError(f"Steering shard has no {field!r} label.")
    values = torch.as_tensor(labels[field]).cpu()
    if values.ndim != 1 or int(values.numel()) != count:
        raise RuntimeError(f"Label {field!r} does not contain exactly {count} cases.")
    return values


def future_all_reachable_label(shard: dict[str, Any], count: int) -> torch.Tensor:
    """A trace-level Boolean counterpart to the continuous future-state fraction."""
    observed = boolean_label(shard, "next_state_observed", count)
    ever_unreachable = boolean_label(shard, "ever_future_unreachable", count)
    return observed & ~ever_unreachable


def transition_counts(
    baseline_values: torch.Tensor,
    steered_values: torch.Tensor,
) -> tuple[int, int, int]:
    gain = int(((~baseline_values) & steered_values).sum())
    loss = int((baseline_values & (~steered_values)).sum())
    return gain, loss, gain - loss


def ordered_change_counts(
    baseline_values: torch.Tensor,
    steered_values: torch.Tensor,
) -> tuple[int, int, int]:
    delta = steered_values - baseline_values
    increase = int((delta > 0).sum())
    decrease = int((delta < 0).sum())
    unchanged = int((delta == 0).sum())
    return increase, decrease, unchanged


def paired_count_record(
    experiment: str,
    row: dict[str, Any],
    baseline: dict[str, Any],
    steered: dict[str, Any],
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, list[list[int]], list[list[int]]]:
    identity_fields = (
        "source_row", "record_pos", "cache_idx", "generated_idx",
        "target", "initial_state", "weights",
    )
    for field in identity_fields:
        if not torch.equal(baseline[field], steered[field]):
            raise RuntimeError(
                f"Baseline and steered shards disagree on {field!r} for {row['condition_id']}."
            )

    count = int(steered["generated_lengths"].numel())
    if count <= 0:
        raise RuntimeError(f"Condition {row['condition_id']} has no paired held-out cases.")
    if row.get("units") is not None and int(row["units"]) != count:
        raise RuntimeError(
            f"Condition {row['condition_id']} reports {row['units']} units but stores {count}."
        )

    baseline_correct = boolean_label(baseline, "expr_target_correct", count)
    steered_correct = boolean_label(steered, "expr_target_correct", count)
    rescue = ~baseline_correct & steered_correct
    damage = baseline_correct & ~steered_correct
    affected = rescue | damage

    baseline_valid = boolean_label(baseline, "valid_trace", count)
    steered_valid = boolean_label(steered, "valid_trace", count)
    valid_rescue = ~baseline_valid & steered_valid
    valid_damage = baseline_valid & ~steered_valid

    baseline_next_state = boolean_label(baseline, "next_state_viable", count)
    steered_next_state = boolean_label(steered, "next_state_viable", count)
    next_gain, next_loss, next_net = transition_counts(
        baseline_next_state, steered_next_state,
    )

    baseline_future_all = future_all_reachable_label(baseline, count)
    steered_future_all = future_all_reachable_label(steered, count)
    future_gain, future_loss, future_net = transition_counts(
        baseline_future_all, steered_future_all,
    )

    baseline_length = numeric_label(baseline, "generated_length", count)
    steered_length = numeric_label(steered, "generated_length", count)
    length_increase, length_decrease, length_unchanged = ordered_change_counts(
        baseline_length, steered_length,
    )

    baseline_back = numeric_label(baseline, "generated_back_count", count)
    steered_back = numeric_label(steered, "generated_back_count", count)
    back_increase, back_decrease, back_unchanged = ordered_change_counts(
        baseline_back, steered_back,
    )

    baseline_parseable = boolean_label(baseline, "expr_parseable", count)
    steered_parseable = boolean_label(steered, "expr_parseable", count)
    parseable_gain, parseable_loss, parseable_net = transition_counts(
        baseline_parseable, steered_parseable,
    )

    baseline_sequences = unpack_generated_sequences(baseline)
    steered_sequences = unpack_generated_sequences(steered)
    if len(baseline_sequences) != count or len(steered_sequences) != count:
        raise RuntimeError(f"Condition {row['condition_id']} has inconsistent sequence counts.")
    changed = torch.tensor(
        [left != right for left, right in zip(baseline_sequences, steered_sequences)],
        dtype=torch.bool,
    )

    weights = steered["weights"].to(torch.float64).cpu()
    if weights.ndim != 1 or int(weights.numel()) != count or float(weights.sum()) <= 0.0:
        raise RuntimeError(f"Condition {row['condition_id']} has invalid held-out weights.")
    weighted_delta = float(
        ((steered_correct.to(torch.float64) - baseline_correct.to(torch.float64)) * weights).sum()
        / weights.sum()
    )
    weighted_rescue = float((rescue.to(torch.float64) * weights).sum() / weights.sum())
    weighted_damage = float((damage.to(torch.float64) * weights).sum() / weights.sum())
    checks = (
        ("delta_expr_target_correct_rate", weighted_delta),
        ("rescue_rate", weighted_rescue),
        ("damage_rate", weighted_damage),
    )
    for field, calculated in checks:
        reported = float(row[field])
        if not math.isclose(calculated, reported, rel_tol=1e-6, abs_tol=1e-7):
            raise RuntimeError(
                f"Condition {row['condition_id']} has inconsistent {field}: "
                f"artifact={reported}, paired shards={calculated}."
            )

    record_pos = steered["record_pos"].cpu()
    cache_idx = steered["cache_idx"].cpu()
    raw_fields = {
        "raw_paired_case_count": count,
        "raw_unique_trace_count": int(torch.unique(record_pos).numel()),
        "raw_unique_problem_count": int(torch.unique(cache_idx).numel()),
        "raw_baseline_correct_count": int(baseline_correct.sum()),
        "raw_steered_correct_count": int(steered_correct.sum()),
        "raw_rescue_count": int(rescue.sum()),
        "raw_damage_count": int(damage.sum()),
        "raw_net_correct_count": int(rescue.sum()) - int(damage.sum()),
        "raw_changed_continuation_count": int(changed.sum()),
        "raw_affected_unique_trace_count": int(torch.unique(record_pos[affected]).numel()),
        "raw_affected_unique_problem_count": int(torch.unique(cache_idx[affected]).numel()),
        "raw_valid_rescue_count": int(valid_rescue.sum()),
        "raw_valid_damage_count": int(valid_damage.sum()),
        "raw_net_valid_count": int(valid_rescue.sum()) - int(valid_damage.sum()),
        "raw_next_state_viable_gain_count": next_gain,
        "raw_next_state_viable_loss_count": next_loss,
        "raw_net_next_state_viable_count": next_net,
        "raw_future_all_reachable_gain_count": future_gain,
        "raw_future_all_reachable_loss_count": future_loss,
        "raw_net_future_all_reachable_count": future_net,
        "raw_length_increase_count": length_increase,
        "raw_length_decrease_count": length_decrease,
        "raw_length_unchanged_count": length_unchanged,
        "raw_back_increase_count": back_increase,
        "raw_back_decrease_count": back_decrease,
        "raw_back_unchanged_count": back_unchanged,
        "raw_parseable_gain_count": parseable_gain,
        "raw_parseable_loss_count": parseable_loss,
        "raw_net_parseable_count": parseable_net,
    }
    if raw_fields["raw_net_correct_count"] != (
        raw_fields["raw_steered_correct_count"] - raw_fields["raw_baseline_correct_count"]
    ):
        raise RuntimeError(f"Raw count identity failed for {row['condition_id']}.")
    row.update(raw_fields)
    record = {
        "experiment": experiment,
        **{field: row.get(field) for field in COUNT_COLUMNS if field != "experiment"},
    }
    return record, rescue, damage, baseline_sequences, steered_sequences


def prepare_paired_counts(
    artifacts: dict[str, dict[str, Any]],
    output_dir: Path,
    *,
    include_validation: bool,
) -> list[Path]:
    """Attach exact held-out counts and export their complete paired evidence."""
    os.environ["COUNTDOWN_STAGE"] = "astar"
    import build_probe_dataset as builder

    evidence_dir = output_dir / EVIDENCE_DIR
    evidence_dir.mkdir(parents=True, exist_ok=True)
    csv_path = evidence_dir / "steering_condition_counts.csv"
    validation_csv_path = evidence_dir / "steering_validation_condition_counts.csv"
    trace_path = evidence_dir / "steering_rescue_damage_traces.txt"
    records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    trace_lines = [
        "CHAINSEARCH EXACT PAIRED STEERING TRANSITIONS",
        "",
        "Every listed case changes final-answer correctness under steering.",
        "Counts are raw paired held-out intervention cases; weighted rates remain in the artifacts.",
        "",
    ]

    tokenizer = builder.pm.Tokenizer()
    for experiment in REQUIRED_EXPERIMENTS:
        artifact = artifacts[experiment]
        baselines = baseline_phase_paths(artifact, "test")
        for row in unique_condition_rows(artifact["test_condition_rows"]):
            key = (str(row["task"]), str(row["checkpoint"]))
            if key not in baselines:
                raise RuntimeError(f"No held-out baseline shard for {experiment} {key}.")
            baseline = load_paired_shard(baselines[key])
            steered = load_paired_shard(row["shard_path"])
            record, rescue, damage, baseline_sequences, steered_sequences = paired_count_record(
                experiment, row, baseline, steered,
            )
            records.append(record)

            affected_indices = torch.nonzero(rescue | damage, as_tuple=False).flatten().tolist()
            if not affected_indices:
                continue
            trace_lines.extend([
                "=" * 120,
                f"Experiment  {EXPERIMENT_DISPLAY.get(experiment, experiment)}",
                f"Condition ID  {row['condition_id']}",
                f"Method  {METHOD_DISPLAY.get(str(row['method']), row['method'])}",
                f"Task  {row['task']}",
                f"Checkpoint  {row['checkpoint']}",
                f"Representation  {row['representation_type']}",
                f"Location  {row['location_name']}",
                f"Exact outcome  {record['raw_rescue_count']} rescues, "
                f"{record['raw_damage_count']} damages, net {record['raw_net_correct_count']}",
                f"Denominator  {record['raw_paired_case_count']} paired held-out cases",
                f"Unique problems  {record['raw_unique_problem_count']}",
                "",
            ])
            for example_number, index in enumerate(affected_indices, start=1):
                outcome = "rescue" if bool(rescue[index]) else "damage"
                trace_lines.extend([
                    "-" * 120,
                    f"Example  {example_number}",
                    f"Outcome  {outcome}",
                    f"Paired-case index  {index}",
                    f"Source row  {int(steered['source_row'][index])}",
                    f"Record position  {int(steered['record_pos'][index])}",
                    f"ChainSearch cache index  {int(steered['cache_idx'][index])}",
                    f"Intervention generated-token index  {int(steered['generated_idx'][index])}",
                    "",
                    "BASELINE TRACE",
                    format_trace_text(baseline_sequences[index], tokenizer),
                    "",
                    "STEERED TRACE",
                    format_trace_text(steered_sequences[index], tokenizer),
                    "",
                ])

    if include_validation:
        atlas = artifacts["steering-atlas"]
        for phase, rows_key in (
            ("coarse", "coarse_validation_rows"),
            ("refined", "refined_validation_rows"),
        ):
            validation_baselines = baseline_phase_paths(atlas, phase)
            validation_rows = positive_scaled_rows(atlas[rows_key])
            for row in unique_condition_rows(validation_rows):
                if not row.get("shard_path"):
                    raise RuntimeError(
                        "Exact validation plots require saved validation shards. "
                        "Rerun perform_steering.py with --save-validation-tokens."
                    )
                key = (str(row["task"]), str(row["checkpoint"]))
                if key not in validation_baselines:
                    raise RuntimeError(
                        f"No {phase}-validation baseline shard for steering-atlas {key}."
                    )
                baseline = load_paired_shard(validation_baselines[key])
                steered = load_paired_shard(row["shard_path"])
                record, _rescue, _damage, _baseline_sequences, _steered_sequences = paired_count_record(
                    "steering-atlas", row, baseline, steered,
                )
                validation_records.append(record)

    temporary_csv = csv_path.with_name(f"{csv_path.name}.tmp.{os.getpid()}")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COUNT_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary_csv, csv_path)

    if include_validation:
        temporary_validation_csv = validation_csv_path.with_name(
            f"{validation_csv_path.name}.tmp.{os.getpid()}"
        )
        with temporary_validation_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COUNT_COLUMNS)
            writer.writeheader()
            writer.writerows(validation_records)
        os.replace(temporary_validation_csv, validation_csv_path)

    temporary_trace = trace_path.with_name(f"{trace_path.name}.tmp.{os.getpid()}")
    temporary_trace.write_text("\n".join(trace_lines).rstrip() + "\n", encoding="utf-8")
    os.replace(temporary_trace, trace_path)
    outputs = [csv_path, trace_path]
    if include_validation:
        outputs.insert(1, validation_csv_path)
    return outputs


def integer_count_axis(ax: plt.Axes, *, horizontal: bool = False) -> None:
    locator = MaxNLocator(integer=True)
    if horizontal:
        ax.xaxis.set_major_locator(locator)
    else:
        ax.yaxis.set_major_locator(locator)


def positive_scaled_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if not bool(row.get("native_direct", False))
        and row.get("beta") is not None
        and float(row["beta"]) > 0.0
    ]


def available_methods(rows: list[dict[str, Any]]) -> list[str]:
    present = {row["method"] for row in rows}
    return [method for method in METHOD_ORDER if method in present]


def available_representations(rows: list[dict[str, Any]]) -> list[str]:
    present = {row["representation_type"] for row in rows}
    return [representation for representation in REPRESENTATION_ORDER if representation in present]


def display_location(name: str) -> str:
    parts = name.split("_")
    if name.startswith("attention_L") and len(parts) == 3:
        return f"Attention {parts[1]} {parts[2]}"
    if name.startswith("residual_L") and len(parts) == 2:
        return f"Residual {parts[1]}"
    if name.startswith("mlp_L") and len(parts) == 2:
        return f"MLP {parts[1]}"
    if name == "final_hidden":
        return "Final hidden"
    return name


COUNT_OUTCOME_FIELDS = {
    "final_answer": ("raw_rescue_count", "raw_damage_count"),
    "valid_trace": ("raw_valid_rescue_count", "raw_valid_damage_count"),
    "next_state_viable": (
        "raw_next_state_viable_gain_count",
        "raw_next_state_viable_loss_count",
    ),
    "future_all_reachable": (
        "raw_future_all_reachable_gain_count",
        "raw_future_all_reachable_loss_count",
    ),
    "parseable": ("raw_parseable_gain_count", "raw_parseable_loss_count"),
}


def paired_transition_counts(row: dict[str, Any], outcome: str) -> tuple[int, int, int]:
    if outcome not in COUNT_OUTCOME_FIELDS:
        raise KeyError(f"Unknown paired-count outcome {outcome!r}.")
    gain_field, loss_field = COUNT_OUTCOME_FIELDS[outcome]
    gain = int(row[gain_field])
    loss = int(row[loss_field])
    return gain, loss, gain - loss


def primary_count_outcome(row: dict[str, Any]) -> str:
    if row.get("label_family") != "reachability":
        return "final_answer"
    if row.get("representation_type") == "scheduled":
        return "future_all_reachable"
    return "next_state_viable"


def paired_total_label(label: str, rows: list[dict[str, Any]]) -> str:
    totals = sorted({int(row["raw_paired_case_count"]) for row in rows})
    if len(totals) == 1:
        return f"{label}\n(out of {totals[0]})"
    return f"{label}\n(total shown per row)"


def paired_case_note(rows: list[dict[str, Any]], phase: str = "held-out") -> str:
    cases = sorted({int(row["raw_paired_case_count"]) for row in rows})
    problems = sorted({int(row["raw_unique_problem_count"]) for row in rows})
    if len(cases) == 1:
        note = f"Counts are out of {cases[0]} paired {phase} cases per condition"
    else:
        note = "Condition-specific paired-case totals"
    if len(problems) == 1:
        note += f"   •   {problems[0]} unique problems"
    return note


def transition_sign_cmap() -> tuple[Any, Any]:
    cmap = matplotlib.colors.ListedColormap(
        [COLOR_LOSS_FILL, COLOR_PANEL, COLOR_GAIN_FILL],
    )
    cmap.set_bad(COLOR_BG)
    norm = matplotlib.colors.BoundaryNorm(
        [-1.5, -0.5, 0.5, 1.5], cmap.N,
    )
    return cmap, norm


def draw_transition_count_matrix(
    ax: plt.Axes,
    row_groups: list[list[dict[str, Any] | None]],
    column_labels: list[str],
    row_labels: list[str],
    outcome: str,
    *,
    outlined_rows: set[int] | None = None,
) -> None:
    matrix = np.full((len(row_groups), len(column_labels)), np.nan, dtype=float)
    for row_index, group in enumerate(row_groups):
        for col_index, row in enumerate(group):
            if row is None:
                continue
            gain, loss, net = paired_transition_counts(row, outcome)
            matrix[row_index, col_index] = float(np.sign(net))

    cmap, norm = transition_sign_cmap()
    ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    for row_index, group in enumerate(row_groups):
        for col_index, row in enumerate(group):
            if row is None:
                continue
            gain, loss, _net = paired_transition_counts(row, outcome)
            if not gain and not loss:
                ax.plot(
                    col_index, row_index, marker="D", markersize=4.8,
                    markerfacecolor=COLOR_PANEL, markeredgecolor=COLOR_BLUE,
                    markeredgewidth=0.9, linestyle="none",
                )
            else:
                ax.text(
                    col_index, row_index, f"+{gain}\n−{loss}",
                    ha="center", va="center", fontsize=7.0, color=COLOR_TEXT,
                )
    for row_index in outlined_rows or set():
        ax.add_patch(Rectangle(
            (-0.5, row_index - 0.5), len(column_labels), 1.0,
            fill=False, edgecolor=COLOR_BLUE, linewidth=1.1,
        ))
    ax.set_xticks(range(len(column_labels)), column_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(axis="y", labelsize=7.2)
    style_paper_axis(ax, grid_axis="both")
    ax.grid(False)


def sign_tally(values: list[int]) -> tuple[int, int, int]:
    gains = sum(value > 0 for value in values)
    losses = sum(value < 0 for value in values)
    ties = len(values) - gains - losses
    return gains, losses, ties


def draw_diverging_summary_bars(
    ax: plt.Axes,
    rows: list[tuple[str, int, int, int, str]],
    *,
    gain_label: str,
    loss_label: str,
    tie_label: str = "equal",
    show_direction_labels: bool = True,
    signed_ticks: bool = False,
    plus_both_sides: bool = False,
) -> None:
    """Draw compact exact-count bars with gains right, losses left, and ties at zero."""
    maximum = max(
        1,
        max((max(gains, losses) for _label, gains, losses, _ties, _color in rows), default=1),
    )
    label_pad = max(0.12, 0.045 * maximum)
    y_positions = np.arange(len(rows), dtype=float)
    for y, (_label, gains, losses, ties, color) in zip(y_positions, rows):
        ax.barh(
            y, gains, height=0.56,
            color=color, edgecolor=color, linewidth=0.8, zorder=3,
        )
        ax.barh(
            y, -losses, height=0.56,
            color=color, edgecolor=color, linewidth=0.8,
            alpha=0.34, hatch="//", zorder=3,
        )
        if gains:
            ax.text(
                gains + label_pad, y,
                f"+{gains}" if signed_ticks or plus_both_sides else str(gains),
                ha="left", va="center", fontsize=9.2,
            )
        if losses:
            if signed_ticks:
                loss_text = f"−{losses}"
            elif plus_both_sides:
                loss_text = f"+{losses}"
            else:
                loss_text = str(losses)
            ax.text(
                -losses - label_pad, y,
                loss_text,
                ha="right", va="center", fontsize=9.2,
            )
        ax.plot(
            0.0, y, marker="D", markersize=5.2,
            markerfacecolor=COLOR_PANEL, markeredgecolor=color,
            markeredgewidth=1.1, linestyle="none", zorder=5,
        )
        if ties:
            ax.annotate(
                f"{ties} {tie_label}",
                (0.0, y), xytext=(0, 8), textcoords="offset points",
                ha="center", va="bottom", fontsize=8.8, color=COLOR_TEXT,
                bbox={
                    "facecolor": COLOR_PANEL,
                    "edgecolor": "none",
                    "alpha": 0.90,
                    "pad": 1.0,
                },
                zorder=7,
            )
    ax.axvline(0.0, color=COLOR_GREY, linestyle="--", linewidth=1.0, zorder=2)
    ax.set_xlim(-1.34 * maximum, 1.34 * maximum)
    ax.set_yticks(y_positions, [label for label, *_rest in rows])
    ax.invert_yaxis()
    if signed_ticks:
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda value, _position: (
                "0" if np.isclose(value, 0.0)
                else f"−{abs(value):g}" if value < 0
                else f"+{value:g}"
            )
        ))
    elif plus_both_sides:
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda value, _position: (
                "0" if np.isclose(value, 0.0)
                else f"+{abs(value):g}"
            )
        ))
    else:
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda value, _position: f"{abs(value):g}")
        )
    integer_count_axis(ax, horizontal=True)
    style_paper_axis(ax, grid_axis="x")
    ax.tick_params(axis="both", labelsize=10.5)
    label_box = {
        "facecolor": COLOR_PANEL,
        "edgecolor": "none",
        "alpha": 0.88,
        "pad": 1.2,
    }
    if show_direction_labels:
        ax.text(
            0.02, 0.975, loss_label, transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color=COLOR_BLUE,
            bbox=label_box, zorder=6,
        )
        ax.text(
            0.98, 0.975, gain_label, transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=COLOR_BLUE,
            bbox=label_box, zorder=6,
        )


def style_paper_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.set_facecolor(COLOR_PANEL)
    ax.title.set_color(COLOR_BLUE)
    ax.xaxis.label.set_color(COLOR_BLUE)
    ax.yaxis.label.set_color(COLOR_BLUE)
    ax.tick_params(axis="both", colors=COLOR_BLUE)
    for spine in ax.spines.values():
        spine.set_color(COLOR_BLUE)
        spine.set_linewidth(0.9)
    ax.grid(True, axis=grid_axis, color=COLOR_GRID, alpha=0.55)


def transition_legend_handles(methods: list[str]) -> list[Any]:
    handles: list[Any] = [
        Patch(
            facecolor=METHOD_COLOR[method],
            edgecolor=METHOD_COLOR[method],
            label=METHOD_DISPLAY[method],
        )
        for method in methods
    ]
    handles.append(
        Line2D(
            [0], [0], marker="D", linestyle="none", markersize=5.5,
            markerfacecolor=COLOR_PANEL, markeredgecolor=COLOR_BLUE,
            label="No changes",
        )
    )
    return handles


def draw_grouped_transition_bars(
    ax: plt.Axes,
    categories: list[str],
    methods: list[str],
    lookup: dict[tuple[str, str], dict[str, Any]],
    outcome: str,
    *,
    show_direction_labels: bool = True,
) -> None:
    width = 0.78 / max(1, len(methods))
    centre = (len(methods) - 1) / 2.0
    maximum = max(
        (
            max(paired_transition_counts(row, outcome)[:2])
            for row in lookup.values()
        ),
        default=1,
    )
    maximum = max(1, maximum)
    label_pad = max(0.12, 0.045 * maximum)
    for method_index, method in enumerate(methods):
        offset = (method_index - centre) * width
        for category_index, category in enumerate(categories):
            row = lookup.get((category, method))
            if row is None:
                continue
            gain, loss, _net = paired_transition_counts(row, outcome)
            x = category_index + offset
            ax.bar(
                x, gain, width=width * 0.86,
                color=METHOD_COLOR[method], edgecolor=METHOD_COLOR[method],
                linewidth=0.7, zorder=3,
            )
            ax.bar(
                x, -loss, width=width * 0.86,
                color=METHOD_COLOR[method], edgecolor=METHOD_COLOR[method],
                alpha=0.34, hatch="//", linewidth=0.7, zorder=3,
            )
            if gain:
                ax.text(x, gain + label_pad, str(gain), ha="center", va="bottom", fontsize=9.5)
            if loss:
                ax.text(x, -loss - label_pad, f"−{loss}", ha="center", va="top", fontsize=9.5)
            if not gain and not loss:
                ax.plot(
                    x, 0, marker="D", markersize=5.0,
                    markerfacecolor=COLOR_PANEL, markeredgecolor=COLOR_BLUE,
                    markeredgewidth=1.0, linestyle="none", zorder=4,
                )
    ax.axhline(0.0, color=COLOR_GREY, linestyle="--", linewidth=1.0, zorder=2)
    ax.set_xlim(-0.55, len(categories) - 0.45)
    ax.set_ylim(-1.35 * maximum, 1.35 * maximum)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda value, _position: (f"−{abs(value):g}" if value < 0 else f"{value:g}"))
    )
    integer_count_axis(ax)
    style_paper_axis(ax)
    ax.tick_params(axis="both", labelsize=11.5)
    if show_direction_labels:
        ax.text(
            0.012, 0.97, "Gains", transform=ax.transAxes,
            ha="left", va="top", fontsize=9.5, color=COLOR_BLUE,
        )
        ax.text(
            0.012, 0.03, "Losses", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9.5, color=COLOR_BLUE,
        )


def condition_label(row: dict[str, Any]) -> str:
    checkpoint = CHECKPOINT_DISPLAY.get(str(row["checkpoint"]), str(row["checkpoint"]))
    location = display_location(str(row["location_name"])).replace("Attention ", "Attn. ")
    return f"{checkpoint}\n{location}"


def short_location(name: str) -> str:
    return display_location(name).replace("Attention ", "").replace("Residual ", "").replace("MLP ", "")


def save_figure(
    fig: plt.Figure,
    stem: str,
    args: argparse.Namespace,
    denominator_rows: list[dict[str, Any]] | None = None,
) -> list[Path]:
    del denominator_rows  # Retained for compatibility with older local calls.
    output_dir = args.output_dir / FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in args.formats:
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


# =============================================================================
# PLOTS
# =============================================================================


def paired_directionality_rows(
    artifacts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for experiment in EXPERIMENT_ORDER:
        artifact = artifacts[experiment]
        by_id = {
            str(row["condition_id"]): row
            for row in unique_condition_rows(artifact["test_condition_rows"])
        }
        for negative_id, positive_id in artifact["negative_control_of"].items():
            positive = by_id[str(positive_id)]
            negative = by_id[str(negative_id)]
            if (positive["method"], positive["label_family"]) != (negative["method"], negative["label_family"]):
                raise RuntimeError(f"Directionality-pair mismatch: {positive_id} and {negative_id}.")
            if int(positive["units"]) != int(negative["units"]):
                raise RuntimeError(
                    f"Directionality pair {positive_id} and {negative_id} has different sample totals."
                )
            output.append({
                "experiment": experiment,
                "method": positive["method"],
                "label_family": positive["label_family"],
                "positive": positive,
                "negative": negative,
            })
    return output


def plot_directionality_summary(
    artifacts: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> list[Path]:
    rows = paired_directionality_rows(artifacts)
    fig, axes = plt.subplots(
        len(EXPERIMENT_ORDER), len(FAMILY_ORDER),
        figsize=(11.0, 8.7), squeeze=False,
    )
    for row_index, experiment in enumerate(EXPERIMENT_ORDER):
        for col_index, family in enumerate(FAMILY_ORDER):
            ax = axes[row_index, col_index]
            subset = rows_where(rows, experiment=experiment, label_family=family)
            if not subset:
                ax.set_visible(False)
                continue
            methods = available_methods(subset)
            summary_rows: list[tuple[str, int, int, int, str]] = []
            for method in methods:
                method_pairs = rows_where(subset, method=method)
                comparisons: list[int] = []
                for item in method_pairs:
                    outcome = primary_count_outcome(item["positive"])
                    selected_net = paired_transition_counts(
                        item["positive"], outcome,
                    )[2]
                    reversed_net = paired_transition_counts(
                        item["negative"], outcome,
                    )[2]
                    comparisons.append(selected_net - reversed_net)
                selected_better, reversed_better, ties = sign_tally(comparisons)
                summary_rows.append((
                    f"{METHOD_DISPLAY[method]}\n(out of {len(method_pairs)} pairs)",
                    selected_better,
                    reversed_better,
                    ties,
                    METHOD_COLOR[method],
                ))
            draw_diverging_summary_bars(
                ax, summary_rows,
                gain_label="Selected direction better",
                loss_label="Reversed direction better",
                signed_ticks=False,
                plus_both_sides=True,
            )
            ax.set_title(
                f"{EXPERIMENT_DISPLAY[experiment]}   •   {FAMILY_DISPLAY[family]}",
                fontsize=13.5, pad=8,
            )
            ax.tick_params(axis="both", labelsize=12, colors=COLOR_BLUE)
    fig.supxlabel(
        "Number of direction pairs",
        x=0.55,
        y=0.020,
        fontsize=13.5,
        color=COLOR_BLUE,
    )
    fig.legend(
        handles=[
            Patch(
                facecolor=COLOR_SLATE, edgecolor=COLOR_SLATE,
                label="Selected direction better",
            ),
            Patch(
                facecolor=COLOR_SLATE, edgecolor=COLOR_SLATE,
                alpha=0.34, hatch="//", label="Reversed direction better",
            ),
            Line2D(
                [0], [0], marker="D", linestyle="none", markersize=5.5,
                markerfacecolor=COLOR_PANEL, markeredgecolor=COLOR_BLUE,
                label="Equal net effect   •   tie count shown",
            ),
        ],
        loc="upper center", ncol=3, bbox_to_anchor=(0.55, 0.938),
        fontsize=10.5,
    )
    fig.suptitle(
        "Effects of reversing the steering directions",
        x=0.55, y=0.972, color=COLOR_TITLE, fontsize=15.5,
    )
    fig.subplots_adjust(
        left=0.12, right=0.985, bottom=0.11, top=0.835,
        hspace=0.42, wspace=0.40,
    )
    return save_figure(fig, "steering_directionality", args)


def plot_dose_response(data: dict[str, Any], args: argparse.Namespace) -> list[Path]:
    """Show exact validation gains and losses at fixed validation-selected sites."""
    outputs: list[Path] = []
    for label_family in ("final_answer", "reachability"):
        coarse_rows = positive_scaled_rows([
            row for row in data["coarse_validation_rows"]
            if row["label_family"] == label_family
        ])
        selected_rows = positive_scaled_rows([
            row for row in data["test_condition_rows"]
            if row["label_family"] == label_family
        ])
        if not coarse_rows or not selected_rows:
            continue

        methods = available_methods(selected_rows)
        representations = available_representations(selected_rows)
        betas = sorted({float(row["beta"]) for row in coarse_rows})
        plot_betas = [0.0, *betas]
        beta_tick_labels = {
            0.25: "¼",
            0.5: "½",
        }
        outcome = "final_answer" if label_family == "final_answer" else "next_state_viable"

        panel_data: dict[tuple[str, str], tuple[list[dict[str, Any]], np.ndarray, dict[tuple[int, int], dict[str, Any]]]] = {}
        all_nets: list[int] = []
        for method in methods:
            for representation in representations:
                anchors = sorted(
                    rows_where(
                        selected_rows,
                        method=method,
                        representation_type=representation,
                    ),
                    key=lambda row: CHECKPOINT_ORDER.index(row["checkpoint"]),
                )
                if not anchors:
                    continue
                matrix = np.full((len(anchors), len(plot_betas)), np.nan, dtype=float)
                lookup: dict[tuple[int, int], dict[str, Any]] = {}
                matrix[:, 0] = 0.0
                for row_index, anchor in enumerate(anchors):
                    for beta_index, beta in enumerate(betas, start=1):
                        matches = [
                            row for row in coarse_rows
                            if row["method"] == method
                            and row["task"] == anchor["task"]
                            and row["checkpoint"] == anchor["checkpoint"]
                            and row["representation_type"] == representation
                            and row["location_name"] == anchor["location_name"]
                            and float(row["beta"]) == beta
                        ]
                        if len(matches) != 1:
                            raise RuntimeError(
                                "Expected one coarse-validation dose row for "
                                f"{method}/{anchor['checkpoint']}/{anchor['location_name']}/β={beta:g}; "
                                f"found {len(matches)}."
                            )
                        gain, loss, net = paired_transition_counts(matches[0], outcome)
                        matrix[row_index, beta_index] = net
                        lookup[(row_index, beta_index)] = matches[0]
                        all_nets.append(net)
                panel_data[(method, representation)] = (anchors, matrix, lookup)

        if not panel_data:
            continue

        bound = max(1, max((abs(value) for value in all_nets), default=1))
        colour_norm = matplotlib.colors.TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
        colour_map = matplotlib.colors.LinearSegmentedColormap.from_list(
            "steering_loss_neutral_gain",
            (COLOR_LOSS_FILL, COLOR_PANEL, COLOR_GAIN_FILL),
        )
        colour_map.set_bad(COLOR_BG)
        family_title = "Final-answer" if label_family == "final_answer" else "Reachability"
        fig, axes = plt.subplots(
            len(methods), len(representations),
            figsize=(3.3 * len(representations), 2.10 * len(methods) + 1.75),
            squeeze=False,
        )
        image = None
        for row_index, method in enumerate(methods):
            for col_index, representation in enumerate(representations):
                ax = axes[row_index, col_index]
                key = (method, representation)
                if key not in panel_data:
                    ax.set_visible(False)
                    continue
                anchors, matrix, lookup = panel_data[key]
                image = ax.imshow(matrix, aspect="auto", cmap=colour_map, norm=colour_norm)
                for anchor_index, anchor in enumerate(anchors):
                    ax.plot(
                        0, anchor_index, marker="D", markersize=5.4,
                        markerfacecolor=COLOR_PANEL, markeredgecolor=COLOR_BLUE,
                        markeredgewidth=0.9, linestyle="none",
                    )
                    selected_beta_index = plot_betas.index(float(anchor["beta"]))
                    ax.add_patch(Rectangle(
                        (selected_beta_index - 0.5, anchor_index - 0.5),
                        1.0, 1.0, fill=False, edgecolor=COLOR_BLUE, linewidth=1.0,
                    ))
                    for beta_index in range(1, len(plot_betas)):
                        row = lookup[(anchor_index, beta_index)]
                        gain, loss, _net = paired_transition_counts(row, outcome)
                        if not gain and not loss:
                            ax.plot(
                                beta_index, anchor_index, marker="D", markersize=5.4,
                                markerfacecolor=COLOR_PANEL, markeredgecolor=COLOR_BLUE,
                                markeredgewidth=0.9, linestyle="none",
                            )
                        else:
                            ax.text(
                                beta_index, anchor_index, f"+{gain}\n−{loss}",
                                ha="center", va="center", fontsize=10.5, color=COLOR_TEXT,
                            )
                ax.set_xticks(
                    range(len(plot_betas)),
                    [beta_tick_labels.get(beta, f"{beta:g}") for beta in plot_betas],
                )
                ax.set_yticks(
                    range(len(anchors)),
                    [
                        f"{CHECKPOINT_DISPLAY[anchor['checkpoint']]}\n"
                        f"{short_location(str(anchor['location_name']))}"
                        for anchor in anchors
                    ],
                )
                if col_index > 0:
                    ax.set_yticklabels([])
                if row_index == len(methods) - 1:
                    ax.set_xlabel(r"Intervention strength $\beta$", fontsize=14)
                if row_index == 0:
                    ax.set_title(
                        REPRESENTATION_DISPLAY[representation],
                        fontsize=15, pad=7,
                    )
                if col_index == 0:
                    ax.set_ylabel(METHOD_DISPLAY[method], fontsize=14)
                style_paper_axis(ax, grid_axis="both")
                ax.grid(False)
                ax.tick_params(axis="x", labelsize=12.0)
                ax.tick_params(axis="y", labelsize=12.5)

        if image is not None:
            colorbar_axis = fig.add_axes((0.925, 0.18, 0.012, 0.64))
            colorbar = fig.colorbar(image, cax=colorbar_axis)
            colorbar.set_label(
                f"Changed validation problems",
                color=COLOR_BLUE, fontsize=14,
            )
            colorbar.ax.tick_params(colors=COLOR_BLUE, labelsize=12.5)
            colorbar.locator = MaxNLocator(integer=True, symmetric=True)
            colorbar.update_ticks()
            colorbar.outline.set_edgecolor(COLOR_BLUE)

        panel_top = 0.90 if label_family == "final_answer" else 0.87
        
        outcome_title = (
            "final-answer correctness"
            if label_family == "final_answer"
            else "next-state reachability"
        )

        fig.suptitle(
            f"Effects of intervention strength on {outcome_title}",
            y=0.985, color=COLOR_TITLE, fontsize=15.5,
        )
        fig.subplots_adjust(
            left=0.14, right=0.89, bottom=0.10, top=panel_top,
            hspace=0.26, wspace=0.14,
        )
        outputs.extend(save_figure(fig, f"where_and_when_dose_response_{label_family}", args))
    return outputs


def plot_validation_stability(data: dict[str, Any], args: argparse.Namespace) -> list[Path]:
    coarse = {
        str(row["condition_id"]): row
        for row in positive_scaled_rows(data["coarse_validation_rows"])
        if row["label_family"] == "final_answer"
    }
    refined = unique_condition_rows([
        row for row in positive_scaled_rows(data["refined_validation_rows"])
        if row["label_family"] == "final_answer"
        and str(row["condition_id"]) in coarse
    ])
    if not refined:
        return []
    methods = available_methods(refined)
    selected_ids = {
        str(row["condition_id"])
        for row in positive_scaled_rows(data["test_condition_rows"])
        if row["label_family"] == "final_answer"
    }
    fig, axes = plt.subplots(
        1, len(methods),
        figsize=(4.33 * len(methods), 4.9),
        sharey=True, squeeze=False,
    )
    for method_index, (ax, method) in enumerate(zip(axes[0], methods)):
        method_rows = sorted(
            rows_where(refined, method=method),
            key=lambda row: (
                CHECKPOINT_ORDER.index(str(row["checkpoint"])),
                REPRESENTATION_ORDER.index(str(row["representation_type"])),
                str(row["location_name"]),
                float(row["beta"]),
            ),
        )
        retained_rows = [
            row for row in method_rows
            if str(row["condition_id"]) in selected_ids
        ]
        summary_rows: list[tuple[str, int, int, int, str]] = []
        for label, group in (
            ("Advanced to refinement", method_rows),
            ("Retained for test", retained_rows),
        ):
            effects = [
                paired_transition_counts(row, "final_answer")[2]
                for row in group
            ]
            gains, losses, ties = sign_tally(effects)
            summary_rows.append((label, gains, losses, ties, METHOD_COLOR[method]))
        draw_diverging_summary_bars(
            ax, summary_rows,
            gain_label="",
            loss_label="",
            tie_label="zero",
            show_direction_labels=False,
            signed_ticks=True,
        )
        for annotation in ax.texts:
            annotation.set_fontsize(11.5)
        ax.set_title(METHOD_DISPLAY[method], fontsize=15.0, pad=7)
        ax.tick_params(axis="both", labelsize=13.0, colors=COLOR_BLUE)
        if method_index > 0:
            ax.tick_params(axis="y", labelleft=False)
    fig.legend(
        handles=[
            Patch(
                facecolor=COLOR_SLATE, edgecolor=COLOR_SLATE,
                label="Positive net effect",
            ),
            Patch(
                facecolor=COLOR_SLATE, edgecolor=COLOR_SLATE,
                alpha=0.34, hatch="//", label="Negative net effect",
            ),
            Line2D(
                [0], [0], marker="D", linestyle="none", markersize=5.5,
                markerfacecolor=COLOR_PANEL, markeredgecolor=COLOR_BLUE,
                label="Zero net effect (count shown)",
            ),
        ],
        loc="upper center", ncol=3, bbox_to_anchor=(0.575, 0.982),
        fontsize=12.0,
    )
    fig.suptitle(
        "How many steering configurations help or hurt final-answer correctness?",
        x=0.575, y=1.03, color=COLOR_TITLE, fontsize=15.5,
    ) 
    axes[0, 1].set_xlabel(
        "Number of steering settings",
        fontsize=14.0,
        color=COLOR_BLUE,
        labelpad=12.5,
    )
    fig.subplots_adjust(
        left=0.16, right=0.985, bottom=0.20, top=0.83, wspace=0.18,
    )
    return save_figure(fig, "steering_validation_stability", args)


def plot_test_comparison(data: dict[str, Any], args: argparse.Namespace) -> list[Path]:
    rows = unique_condition_rows(positive_scaled_rows([
        row for row in data["test_condition_rows"] if row["label_family"] == "final_answer"
    ]))
    if not rows:
        return []
    methods = available_methods(rows)
    representations = available_representations(rows)
    fig, axes = plt.subplots(
        2, 2, figsize=(12.2, 9.4), squeeze=False,
    )
    for ax, representation in zip(axes.flat, representations):
        subset = rows_where(rows, representation_type=representation)
        checkpoints = [
            checkpoint for checkpoint in CHECKPOINT_ORDER
            if any(row["checkpoint"] == checkpoint for row in subset)
        ]
        lookup: dict[tuple[str, str], dict[str, Any]] = {}
        for checkpoint in checkpoints:
            for method in methods:
                matches = rows_where(subset, checkpoint=checkpoint, method=method)
                if len(matches) > 1:
                    raise RuntimeError(
                        f"Expected at most one selected {representation}/{checkpoint}/{method} condition; "
                        f"found {len(matches)}."
                    )
                if matches:
                    lookup[(checkpoint, method)] = matches[0]
        draw_grouped_transition_bars(
            ax, checkpoints, methods, lookup, "final_answer",
            show_direction_labels=False,
        )
        for annotation in ax.texts:
            annotation.set_fontsize(10.5)
        ax.set_xticks(
            range(len(checkpoints)),
            [CHECKPOINT_DISPLAY[checkpoint] for checkpoint in checkpoints],
            rotation=22, ha="right",
        )
        ax.set_ylabel("Changed test problems", fontsize=13.5)
        ax.set_title(REPRESENTATION_DISPLAY[representation], fontsize=14, pad=7)
        ax.tick_params(axis="both", labelsize=12.5, colors=COLOR_BLUE)
    for ax in axes.flat[len(representations):]:
        ax.set_visible(False)
    fig.legend(
        handles=transition_legend_handles(methods),
        loc="upper center", ncol=len(methods) + 1,
        bbox_to_anchor=(0.54, 0.958), fontsize=10.5,
    )
    fig.suptitle(
        "Final-answer steering effects by component and checkpoint",
        x=0.54, y=0.985, color=COLOR_TITLE, fontsize=15,
    )
    fig.supxlabel(
        "Token position",
        x=0.53,
        y=0.025,
        fontsize=14.0,
        color=COLOR_BLUE,
    )
    fig.subplots_adjust(
        left=0.08, right=0.98, bottom=0.18, top=0.86,
        hspace=0.50, wspace=0.16,
    )
    return save_figure(fig, "steering_test_comparison", args)


def plot_rescue_damage(data: dict[str, Any], args: argparse.Namespace) -> list[Path]:
    rows = unique_condition_rows(positive_scaled_rows([
        row for row in data["test_condition_rows"] if row["label_family"] == "final_answer"
    ]))
    if not rows:
        return []
    methods = available_methods(rows)
    maximum = max(
        max(int(row["raw_rescue_count"]), int(row["raw_damage_count"]))
        for row in rows
    )
    maximum = max(1, maximum)
    label_pad = max(0.15, 0.035 * maximum)
    fig, axes_grid = plt.subplots(
        len(methods), 1, figsize=(13.2, 12.2), sharey=True, squeeze=False,
    )
    axes = axes_grid[:, 0]

    for ax, method in zip(axes, methods):
        method_rows = sorted(
            rows_where(rows, method=method),
            key=lambda row: (
                CHECKPOINT_ORDER.index(str(row["checkpoint"])),
                REPRESENTATION_ORDER.index(str(row["representation_type"])),
                str(row["location_name"]),
            ),
        )
        x = np.arange(len(method_rows), dtype=float)
        for row_index, row in enumerate(method_rows):
            rescue = int(row["raw_rescue_count"])
            damage = int(row["raw_damage_count"])
            ax.bar(
                row_index, rescue, width=0.66,
                color=METHOD_COLOR[method], edgecolor=METHOD_COLOR[method], linewidth=0.7,
            )
            ax.bar(
                row_index, -damage, width=0.66,
                color=METHOD_COLOR[method], edgecolor=METHOD_COLOR[method],
                alpha=0.34, hatch="//", linewidth=0.7,
            )
            if rescue:
                ax.text(
                    row_index, rescue + label_pad, f"+{rescue}",
                    ha="center", va="bottom", fontsize=12.0,
                )
            if damage:
                ax.text(
                    row_index, -damage - label_pad, f"−{damage}",
                    ha="center", va="top", fontsize=12.0,
                )
            if not rescue and not damage:
                ax.plot(
                    row_index, 0, marker="D", markersize=5.0,
                    markerfacecolor=COLOR_PANEL, markeredgecolor=COLOR_BLUE,
                    markeredgewidth=1.0, linestyle="none", zorder=4,
                )
        # Separate successive checkpoint groups visually.
        for boundary in range(1, len(method_rows)):
            previous_checkpoint = method_rows[boundary - 1]["checkpoint"]
            current_checkpoint = method_rows[boundary]["checkpoint"]

            if current_checkpoint != previous_checkpoint:
                ax.axvline(
                    boundary - 0.5,
                    color=COLOR_GRID,
                    linewidth=1.3,
                    alpha=0.95,
                    zorder=0.5,
                )
        ax.axhline(0.0, color=COLOR_GREY, linestyle="--", linewidth=1.0)
        ax.set_xlim(-0.65, len(method_rows) - 0.35)
        ax.set_ylim(-1.30 * maximum, 1.30 * maximum)

        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda value, _position: f"−{abs(value):g}" if value < 0 else f"{value:g}"
        ))
        integer_count_axis(ax)
        style_paper_axis(ax, grid_axis="y")
        ax.tick_params(axis="y", labelsize=12.5, colors=COLOR_BLUE)

        # First x-axis level: token-position groups.
        group_centres = []
        group_labels = []

        for checkpoint in CHECKPOINT_ORDER:
            checkpoint_indices = [
                index
                for index, row in enumerate(method_rows)
                if str(row["checkpoint"]) == checkpoint
            ]

            if checkpoint_indices:
                group_centres.append(float(np.mean(checkpoint_indices)))
                group_labels.append(CHECKPOINT_DISPLAY[checkpoint])

        ax.set_xticks(group_centres, group_labels)
        ax.tick_params(
            axis="x",
            which="major",
            labelsize=11.5,
            colors=COLOR_TOKEN_POSITION,
            pad=10,
            length=0,
        )
        for label in ax.get_xticklabels():
            label.set_fontstyle("italic")
        ax.set_xticks(x, minor=True)
        ax.tick_params(
            axis="x",
            which="minor",
            bottom=True,
            top=False,
            length=5,
            width=1.0,
            color=COLOR_BLUE,
            direction="out",
        )
        # Second x-axis level: model components and selected locations.
        location_labels = [
            display_location(str(row["location_name"])).replace(
                "Attention ", "Attn. "
            )
            for row in method_rows
        ]

        component_axis = ax.secondary_xaxis("bottom")
        component_axis.spines["bottom"].set_position(("outward", 30))
        component_axis.spines["bottom"].set_visible(False)
        component_axis.set_xticks(x, location_labels)
        component_axis.tick_params(
            axis="x",
            labelsize=11.5,
            colors=COLOR_BLUE,
            pad=2,
            length=0,
            rotation=34,
        )

        for label in component_axis.get_xticklabels():
            label.set_horizontalalignment("right")
            label.set_rotation_mode("anchor")


        ax.set_ylabel("Changed test problems", fontsize=13.0)
        ax.set_title(
            f"{METHOD_DISPLAY[method]}   •   {len(method_rows)} settings",
            fontsize=13.5, pad=7,
        )
    fig.legend(
        handles=transition_legend_handles([]),
        loc="upper center", ncol=1, bbox_to_anchor=(0.53, 0.957),
        fontsize=11.5,
    )
    fig.suptitle(
        "Final-answer effects of individual steering settings",
        x=0.54, y=0.983, color=COLOR_TITLE, fontsize=15.5,
    ) 

    fig.text(
        0.515,
        0.008,
        "Token position",
        ha="right",
        va="center",
        fontsize=13.0,
        fontstyle="italic",
        color=COLOR_TOKEN_POSITION,
    )

    fig.text(
        0.53,
        0.008,
        "•",
        ha="center",
        va="center",
        fontsize=13.0,
        color=COLOR_TOKEN_POSITION,
    )

    fig.text(
        0.545,
        0.008,
        "Model component",
        ha="left",
        va="center",
        fontsize=13.0,
        color=COLOR_BLUE,
    )
    fig.subplots_adjust(
        left=0.075, right=0.985, bottom=0.15, top=0.885,
        hspace=1.00,
    )
    return save_figure(fig, "steering_rescue_damage", args)



def plot_validity_tradeoff(data: dict[str, Any], args: argparse.Namespace) -> list[Path]:
    rows = unique_condition_rows(positive_scaled_rows([
        row for row in data["test_condition_rows"] if row["label_family"] == "final_answer"
    ]))
    if not rows:
        return []
    methods = available_methods(rows)
    pastel_method_color = {
        "probe": "#9FBCE3",
        "caa": "#C8DCF4",
        "optimised": "#F2C2AD",
    }
    gain_rows = [
        row for row in rows
        if int(row["raw_net_correct_count"]) > 0
    ]
    validity_signs = (1, 0, -1)
    validity_labels = (
        "Validity gain",
        "No net validity change",
        "Validity loss",
    )
    counts = {
        (validity_sign, method): sum(
            1
            for row in gain_rows
            if int(np.sign(int(row["raw_net_valid_count"]))) == validity_sign
            and str(row["method"]) == method
        )
        for validity_sign in validity_signs
        for method in methods
    }

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    x_positions = np.arange(len(validity_signs))
    bottoms = np.zeros(len(validity_signs), dtype=float)
    for method in methods:
        values = np.asarray(
            [counts[(validity_sign, method)] for validity_sign in validity_signs],
            dtype=float,
        )
        bars = ax.bar(
            x_positions, values, bottom=bottoms, width=0.62,
            color=pastel_method_color[method], edgecolor=METHOD_COLOR[method],
            linewidth=0.7, label=METHOD_DISPLAY[method],
        )
        for bar, value, bottom in zip(bars, values, bottoms):
            if value:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bottom + value / 2.0,
                    f"{int(value)}",
                    ha="center", va="center", fontsize=12.0,
                    color=COLOR_TEXT,
                )
        bottoms += values

    for x_position, total in zip(x_positions, bottoms):
        if total:
            ax.text(
                x_position, total + 0.16, f"{int(total)} total",
                ha="center", va="bottom", fontsize=12.0,
                color=COLOR_TEXT,
            )
        else:
            ax.plot(
                x_position, 0, marker="D", markersize=6.2,
                markerfacecolor=COLOR_PANEL, markeredgecolor=COLOR_BLUE,
                markeredgewidth=1.0, linestyle="none", zorder=4,
            )
            ax.text(
                x_position, 0.22, "0 total",
                ha="center", va="bottom", fontsize=12.0,
                color=COLOR_TEXT,
            )

    ax.set_xticks(x_positions, validity_labels)
    ax.set_ylabel("Number of steering settings", fontsize=13.5, labelpad=10)
    ax.tick_params(axis="x", labelsize=12.5, colors=COLOR_BLUE, pad=7)
    ax.tick_params(axis="y", labelsize=12.0, colors=COLOR_BLUE)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, max(1.0, float(max(bottoms, default=0.0)) + 1.0))
    style_paper_axis(ax, grid_axis="y") 
    fig.legend(
        loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5525, 0.91),
        fontsize=11.5,
    )
    fig.suptitle(
        "Trace-validity outcomes for 10 settings with final-answer gains",
        x=0.5525, y=0.965,
        color=COLOR_TITLE, fontsize=14.5,
    )
    ax.set_xlabel(
        "Net trace-validity outcome",
        fontsize=13.0,
        labelpad=10,
    )
    fig.subplots_adjust(
        left=0.12, right=0.985, bottom=0.18, top=0.79,
    )
    return save_figure(fig, "steering_validity_tradeoff", args)


def plot_validation_heatmap(data: dict[str, Any], args: argparse.Namespace) -> list[Path]:
    rows = positive_scaled_rows([
        row for row in data["coarse_validation_rows"]
        if row["label_family"] == "final_answer"
    ])
    if not rows:
        return []
    methods = available_methods(rows)
    checkpoints = [
        checkpoint for checkpoint in CHECKPOINT_ORDER
        if any(row["checkpoint"] == checkpoint for row in rows)
    ]
    location_representations = {
        str(row["location_name"]): str(row["representation_type"])
        for row in rows
    }
    locations = sorted(
        location_representations,
        key=lambda name: (
            REPRESENTATION_ORDER.index(location_representations[name]),
            name,
        ),
    )
    selected_locations = {
        (
            str(row["method"]),
            str(row["checkpoint"]),
            str(row["location_name"]),
        )
        for row in positive_scaled_rows(data["test_condition_rows"])
        if row["label_family"] == "final_answer"
    }

    panels: list[tuple[str, np.ndarray, dict[tuple[int, int], dict[str, Any]]]] = []
    all_nets: list[int] = []
    for method in methods:
        matrix = np.full((len(checkpoints), len(locations)), np.nan, dtype=float)
        lookup: dict[tuple[int, int], dict[str, Any]] = {}
        method_rows = rows_where(rows, method=method)
        for row_index, checkpoint in enumerate(checkpoints):
            for col_index, location in enumerate(locations):
                candidates = rows_where(
                    method_rows,
                    checkpoint=checkpoint,
                    location_name=location,
                )
                if not candidates:
                    continue
                chosen = max(
                    candidates,
                    key=lambda row: (
                        float(row["delta_expr_target_correct_rate"]),
                        float(row["delta_valid_trace_rate"]),
                        -abs(float(row["beta"])),
                        str(row["condition_id"]),
                    ),
                )
                gain, loss, net = paired_transition_counts(chosen, "final_answer")
                matrix[row_index, col_index] = net
                lookup[(row_index, col_index)] = chosen
                all_nets.append(net)
        panels.append((method, matrix, lookup))

    bound = max(1, max((abs(value) for value in all_nets), default=1))
    colour_norm = matplotlib.colors.TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    colour_map = matplotlib.colors.LinearSegmentedColormap.from_list(
        "validation_loss_neutral_gain",
        (COLOR_LOSS_FILL, COLOR_PANEL, COLOR_GAIN_FILL),
    )
    colour_map.set_bad(COLOR_BG)
    fig, axes = plt.subplots(
        len(methods), 1,
        figsize=(max(13.5, 0.58 * len(locations)), 3.7 * len(methods) + 1.8),
        sharex=True, squeeze=False,
    )
    image = None
    for ax, (method, matrix, lookup) in zip(axes[:, 0], panels):
        image = ax.imshow(matrix, cmap=colour_map, norm=colour_norm, aspect="auto")
        for (row_index, col_index), row in lookup.items():
            gain, loss, _net = paired_transition_counts(row, "final_answer")
            if not gain and not loss:
                ax.plot(
                    col_index, row_index, marker="D", markersize=5.4,
                    markerfacecolor=COLOR_PANEL, markeredgecolor=COLOR_BLUE,
                    markeredgewidth=0.8, linestyle="none",
                )
            else:
                ax.text(
                    col_index, row_index,
                    f"+{gain}\n−{loss}\nβ={float(row['beta']):g}",
                    ha="center", va="center", fontsize=10.5, color=COLOR_TEXT,
                    linespacing=1.0,
                )
            if (
                method,
                str(row["checkpoint"]),
                str(row["location_name"]),
            ) in selected_locations:
                ax.add_patch(Rectangle(
                    (col_index - 0.5, row_index - 0.5), 1.0, 1.0,
                    fill=False, edgecolor=COLOR_BLUE, linewidth=1.2,
                ))
        ax.set_yticks(
            range(len(checkpoints)),
            [CHECKPOINT_DISPLAY[checkpoint] for checkpoint in checkpoints],
        )
        ax.set_ylabel(METHOD_DISPLAY[method], fontsize=14.0)
        style_paper_axis(ax, grid_axis="both")
        ax.grid(False)
        ax.tick_params(axis="y", labelsize=12.5, colors=COLOR_BLUE)
    axes[-1, 0].set_xticks(
        range(len(locations)),
        [display_location(location) for location in locations],
        rotation=90,
    )
    axes[-1, 0].tick_params(axis="x", labelsize=12.0, colors=COLOR_BLUE)
    if image is not None:
        colorbar_axis = fig.add_axes((0.915, 0.205, 0.014, 0.685))
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label(
            "Changed validation problems",
            color=COLOR_BLUE,
            fontsize=14.0,
        )
        colorbar.ax.tick_params(colors=COLOR_BLUE, labelsize=12.0)
        colorbar.outline.set_edgecolor(COLOR_BLUE)
        colorbar.locator = MaxNLocator(integer=True, symmetric=True)
        colorbar.update_ticks()
    fig.suptitle(
        "Initial validation search across token positions and model locations",
        x=0.50, y=0.955, color=COLOR_TITLE, fontsize=15.5,
    )
    fig.supxlabel(
        "Model location",
        x=0.50,
        y=0.015,
        fontsize=14.0,
        color=COLOR_BLUE,
    )
    fig.subplots_adjust(left=0.11, right=0.89, bottom=0.23, top=0.91, hspace=0.12)
    return save_figure(fig, "steering_validation_heatmap", args)


def plot_side_effects(data: dict[str, Any], args: argparse.Namespace) -> list[Path]:
    rows = unique_condition_rows(positive_scaled_rows([
        row for row in data["test_condition_rows"] if row["label_family"] == "final_answer"
    ]))
    if not rows:
        return []
    methods = available_methods(rows)
    checkpoints = [
        checkpoint for checkpoint in CHECKPOINT_ORDER
        if any(row["checkpoint"] == checkpoint for row in rows)
    ]
    representations = available_representations(rows)
    row_specs = [
        (method, checkpoint)
        for checkpoint in checkpoints
        for method in methods
    ]
    panel_specs = (
        (
            "raw_length_increase_count", "raw_length_decrease_count",
            "Generated length", "↑", "↓",
        ),
        (
            "raw_back_increase_count", "raw_back_decrease_count",
            "Generated BACK count", "↑", "↓",
        ),
        (
            "raw_parseable_gain_count", "raw_parseable_loss_count",
            "EXPR parseability", "+", "−",
        ),
    )
    cmap = matplotlib.colors.ListedColormap(
        [COLOR_LOSS_HEATMAP, COLOR_PANEL, COLOR_GAIN_HEATMAP]
    )
    cmap.set_bad(COLOR_BG)
    norm = matplotlib.colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 7.8), sharey=True)
    for panel_index, (
        increase_field, decrease_field, title, increase_symbol, decrease_symbol,
    ) in enumerate(panel_specs):
        ax = axes[panel_index]
        matrix = np.full((len(row_specs), len(representations)), np.nan, dtype=float)
        lookup: dict[tuple[int, int], dict[str, Any]] = {}
        for row_index, (method, checkpoint) in enumerate(row_specs):
            for col_index, representation in enumerate(representations):
                matches = rows_where(
                    rows,
                    method=method,
                    checkpoint=checkpoint,
                    representation_type=representation,
                )
                if len(matches) > 1:
                    raise RuntimeError(
                        f"Expected at most one selected {method}/{checkpoint}/{representation} condition; "
                        f"found {len(matches)}."
                    )
                if not matches:
                    continue
                row = matches[0]
                increase = int(row[increase_field])
                decrease = int(row[decrease_field])
                matrix[row_index, col_index] = float(np.sign(increase - decrease))
                lookup[(row_index, col_index)] = row
        ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
        for (row_index, col_index), row in lookup.items():
            increase = int(row[increase_field])
            decrease = int(row[decrease_field])
            if not increase and not decrease:
                ax.plot(
                    col_index, row_index, marker="D", markersize=4.8,
                    markerfacecolor=COLOR_PANEL, markeredgecolor=COLOR_BLUE,
                    markeredgewidth=0.9, linestyle="none",
                )
            else:
                ax.text(
                    col_index, row_index,
                    f"{increase_symbol}{increase}\n{decrease_symbol}{decrease}",
                    ha="center", va="center", fontsize=12.0, color=COLOR_TEXT,
                )
        ax.set_xticks(
            range(len(representations)),
            [REPRESENTATION_DISPLAY[value] for value in representations],
            rotation=22, ha="right",
        )
        ax.set_yticks(
            range(len(row_specs)),
            [
                f"{CHECKPOINT_DISPLAY[checkpoint]}\n{METHOD_DISPLAY[method]}"
                for method, checkpoint in row_specs
            ],
        )
        ax.set_title(title, fontsize=14.5, pad=7)
        style_paper_axis(ax, grid_axis="both")
        ax.grid(False)
        ax.tick_params(axis="x", labelsize=13.5, colors=COLOR_BLUE)
        ax.tick_params(axis="y", labelsize=13.5, colors=COLOR_BLUE)
    fig.legend(
        handles=[
            Patch(
                facecolor=COLOR_GAIN_HEATMAP, edgecolor=COLOR_BLUE,
                label="Net increase or improvement",
            ),
            Patch(facecolor=COLOR_PANEL, edgecolor=COLOR_BLUE, label="Equal opposing changes"),
            Patch(
                facecolor=COLOR_LOSS_HEATMAP, edgecolor=COLOR_RED,
                label="Net decrease or loss",
            ),
            Line2D(
                [0], [0], marker="D", linestyle="none", markersize=5.5,
                markerfacecolor=COLOR_PANEL, markeredgecolor=COLOR_BLUE,
                label="No changes",
            ),
        ],
        loc="upper center", ncol=4, bbox_to_anchor=(0.58, 0.946),
        fontsize=13.0,
    )
    fig.suptitle(
        "Effects of steering on trace mechanics",
        x=0.58, y=0.975, color=COLOR_TITLE, fontsize=15.5,
    )
    fig.supxlabel(
        "Model component",
        x=0.58,
        y=0.035,
        fontsize=13.5,
        color=COLOR_BLUE,
    )
    fig.supylabel(
        "Intervention position and method",
        x=0.025,
        y=0.50,
        fontsize=13.5,
        color=COLOR_BLUE,
    )
    fig.subplots_adjust(
        left=0.15, right=0.985, bottom=0.19, top=0.84, wspace=0.12,
    )
    return save_figure(fig, "steering_side_effects", args)


def plot_checkpoint_summary(data: dict[str, Any], args: argparse.Namespace) -> list[Path]:
    rows = unique_condition_rows(positive_scaled_rows([
        row for row in data["test_condition_rows"] if row["label_family"] == "final_answer"
    ]))
    if not rows:
        return []
    methods = available_methods(rows)
    checkpoints = [
        checkpoint for checkpoint in CHECKPOINT_ORDER
        if any(row["checkpoint"] == checkpoint for row in rows)
    ]
    representations = available_representations(rows)
    row_specs = [
        (method, checkpoint)
        for checkpoint in checkpoints
        for method in methods
    ]
    matrix = np.full((len(row_specs), len(representations)), np.nan, dtype=float)
    lookup: dict[tuple[int, int], dict[str, Any]] = {}
    nets: list[int] = []
    for row_index, (method, checkpoint) in enumerate(row_specs):
        for col_index, representation in enumerate(representations):
            matches = rows_where(
                rows,
                method=method,
                checkpoint=checkpoint,
                representation_type=representation,
            )
            if len(matches) > 1:
                raise RuntimeError(
                    f"Expected at most one selected {method}/{checkpoint}/{representation} condition; "
                    f"found {len(matches)}."
                )
            if matches:
                gain, loss, net = paired_transition_counts(matches[0], "final_answer")
                matrix[row_index, col_index] = net
                lookup[(row_index, col_index)] = matches[0]
                nets.append(net)

    bound = max(1, max((abs(value) for value in nets), default=1))
    colour_norm = matplotlib.colors.TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    colour_map = matplotlib.colors.LinearSegmentedColormap.from_list(
        "checkpoint_loss_neutral_gain",
        (COLOR_LOSS_FILL, COLOR_PANEL, COLOR_GAIN_FILL),
    )
    colour_map.set_bad(COLOR_BG)
    fig, ax = plt.subplots(figsize=(9.4, 8.4))
    image = ax.imshow(matrix, aspect="auto", cmap=colour_map, norm=colour_norm)
    for (row_index, col_index), row in lookup.items():
        gain, loss, _net = paired_transition_counts(row, "final_answer")
        location = short_location(str(row["location_name"]))
        if not gain and not loss:
            ax.plot(
                col_index, row_index - 0.10, marker="D", markersize=6.0,
                markerfacecolor=COLOR_PANEL, markeredgecolor=COLOR_BLUE,
                markeredgewidth=1.0, linestyle="none",
            )
            ax.text(
                col_index, row_index + 0.24, location,
                ha="center", va="center", fontsize=8.3, color=COLOR_TEXT,
            )
        else:
            ax.text(
                col_index, row_index,
                f"+{gain} / −{loss}\n{location}",
                ha="center", va="center", fontsize=9.0, color=COLOR_TEXT,
            )
    ax.set_xticks(
        range(len(representations)),
        [REPRESENTATION_DISPLAY[representation] for representation in representations],
    )
    ax.set_yticks(
        range(len(row_specs)),
        [
            f"{CHECKPOINT_DISPLAY[checkpoint]}   •   {METHOD_DISPLAY[method]}"
            for method, checkpoint in row_specs
        ],
    )
    ax.tick_params(axis="x", labelrotation=16)
    style_paper_axis(ax, grid_axis="both")
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=11.3)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.74, pad=0.025)
    colorbar.set_label("Changed test problems", color=COLOR_BLUE, fontsize=12.0)
    colorbar.ax.tick_params(colors=COLOR_BLUE, labelsize=11.0)
    colorbar.outline.set_edgecolor(COLOR_BLUE)
    fig.suptitle(
        "Overview of final-answer steering",
        x=0.54, y=0.960, color=COLOR_TITLE, fontsize=13.5,
    )
    ax.set_xlabel(
        "Model component",
        fontsize=12.0,
        color=COLOR_BLUE,
        labelpad=7,
    )
    ax.set_ylabel(
        "Token position and steering method",
        fontsize=12.0,
        color=COLOR_BLUE,
        labelpad=10,
    )
    fig.subplots_adjust(left=0.27, right=0.90, bottom=0.11, top=0.91)
    return save_figure(fig, "steering_checkpoint_summary", args)


def plot_head_breadth(data: dict[str, Any], args: argparse.Namespace) -> list[Path]:
    """Show exact outcome changes under head-set and coefficient-scaling controls."""
    rows = unique_condition_rows(positive_scaled_rows(data["test_condition_rows"]))
    if not rows:
        return []
    families = [
        family for family in ("reachability", "final_answer")
        if any(row["label_family"] == family for row in rows)
    ]
    fig, axes = plt.subplots(
        2, len(families),
        figsize=(6.4 * len(families), 6.4),
        sharex="col", squeeze=False,
    )
    for col, family in enumerate(families):
        subset = rows_where(rows, label_family=family)
        methods = available_methods(subset)
        categories = [
            variant for variant in HEAD_VARIANT_ORDER
            if any(row["selection_variant"] == variant for row in subset)
        ]
        lookup: dict[tuple[str, str], dict[str, Any]] = {}
        for variant in categories:
            for method in methods:
                matches = rows_where(
                    subset,
                    selection_variant=variant,
                    method=method,
                )
                if len(matches) > 1:
                    raise RuntimeError(
                        f"Expected at most one selected {family}/{variant}/{method} condition; "
                        f"found {len(matches)}."
                    )
                if matches:
                    lookup[(variant, method)] = matches[0]
        draw_grouped_transition_bars(
            axes[0, col], categories, methods, lookup,
            "final_answer" if family == "final_answer" else "next_state_viable",
            show_direction_labels=False,
        )
        draw_grouped_transition_bars(
            axes[1, col], categories, methods, lookup, "valid_trace",
            show_direction_labels=False,
        ) 
        axes[0, col].set_title(
            (
                "Final-answer correctness"
                if family == "final_answer"
                else "Next-state reachability"
            ),
            fontsize=13.5,
        )
        axes[1, col].set_title("Whole-trace validity", fontsize=13.5)
        tick_labels = [HEAD_VARIANT_DISPLAY[variant] for variant in categories]
        for ax in axes[:, col]:
            ax.set_xticks(range(len(categories)), tick_labels)
            ax.tick_params(axis="both", labelsize=11.5, colors=COLOR_BLUE)
    
    axes[0, 0].set_ylabel("Changed test cases", fontsize=12.5)
    axes[1, 0].set_ylabel("Changed test cases", fontsize=12.5)
    
    methods = available_methods(rows)
    legend_handles = [
        Patch(facecolor=METHOD_COLOR[method], edgecolor=METHOD_COLOR[method], label=METHOD_DISPLAY[method])
        for method in methods
    ] + [
        Line2D([0], [0], marker="D", linestyle="none", markersize=5.5,
            markerfacecolor=COLOR_PANEL, markeredgecolor=COLOR_BLUE, label="No changes",),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=len(legend_handles), 
        bbox_to_anchor=(0.535, 0.945), fontsize=10.5)
    fig.suptitle(
        "Attention-head selection and intervention scaling",
        x=0.535,
        y=0.985, color=COLOR_TITLE,
        fontsize=15,
    ) 
    fig.subplots_adjust(
        left=0.09, right=0.98, bottom=0.15, top=0.83,
        hspace=0.30, wspace=0.12,
    )
    return save_figure(fig, "head_set_and_scaling_controls", args)


def plot_intervention_frequency_compact(data: dict[str, Any], args: argparse.Namespace) -> list[Path]:
    """Paper-facing paired counts for single, repeated, and multipoint steering."""
    rows = unique_condition_rows(positive_scaled_rows(data["test_condition_rows"]))
    if not rows:
        return []
    families = [family for family in FAMILY_ORDER if any(row["label_family"] == family for row in rows)]
    fig, axes = plt.subplots(
        2,
        len(families),
        figsize=(6.7 * len(families), 6.8),
        sharex="col",
        squeeze=False,
    )
    for col, family in enumerate(families):
        subset = rows_where(rows, label_family=family)
        methods = available_methods(subset)
        categories = [
            schedule
            for schedule in SCHEDULE_ORDER
            if any(row.get("schedule") == schedule for row in subset)
        ]
        lookup: dict[tuple[str, str], dict[str, Any]] = {}
        for schedule in categories:
            for method in methods:
                matches = rows_where(subset, schedule=schedule, method=method)
                if len(matches) != 1:
                    raise RuntimeError(
                        f"Expected one selected {family}/{schedule}/{method} condition; "
                        f"found {len(matches)}."
                    )
                lookup[(schedule, method)] = matches[0]
        primary_outcome = (
            "final_answer" if family == "final_answer" else "future_all_reachable"
        )
        draw_grouped_transition_bars(
            axes[0, col], categories, methods, lookup, primary_outcome,
            show_direction_labels=False,
        )
        draw_grouped_transition_bars(
            axes[1, col], categories, methods, lookup, "valid_trace",
            show_direction_labels=False,
        )
        axes[0, col].set_title(
            "Final-answer correctness"
            if family == "final_answer"
            else "Entire observed future remains reachable",
            fontsize=13.5, pad=7,
        )
        axes[1, col].set_title("Trace validity", fontsize=13.5, pad=7)
        axes[0, col].set_ylabel("")
        axes[1, col].set_ylabel("")
        axes[1, col].set_xticks(
            range(len(categories)),
            [SCHEDULE_DISPLAY[schedule] for schedule in categories],
        )
        axes[0, col].set_xticks(
            range(len(categories)),
            [SCHEDULE_DISPLAY[schedule] for schedule in categories],
        )
        for ax in axes[:, col]:
            ax.tick_params(axis="both", labelsize=11.5, colors=COLOR_BLUE)
    axes[0, 0].set_ylabel("Changed test problems", fontsize=12.5, labelpad=7)
    axes[1, 0].set_ylabel("Changed test problems", fontsize=12.5, labelpad=7)
    all_methods = available_methods(rows)
    fig.legend(
        handles=transition_legend_handles(all_methods),
        loc="upper center", ncol=len(all_methods) + 1,
        bbox_to_anchor=(0.525, 0.95), fontsize=10.5,
    )
    fig.suptitle(
        "Effects of single, repeated, and multipoint steering",
        x=0.525, y=0.985, 
        color=COLOR_TITLE, fontsize=15,
    )
    fig.supxlabel(
        "Intervention schedule", 
        x=0.525, y=0.040,
        fontsize=12, color=COLOR_BLUE,
    )
    fig.subplots_adjust(
        left=0.09, right=0.98, bottom=0.13, top=0.84,
        hspace=0.24, wspace=0.12,
    )
    return save_figure(fig, "intervention_frequency_heldout_effects_compact", args)


def plot_native_direct_pairs(data: dict[str, Any], args: argparse.Namespace) -> list[Path]:
    rows = unique_condition_rows([
        row for row in rows_where(data["test_condition_rows"], method="optimised")
        if row["label_family"] == "final_answer"
    ])
    native = [row for row in rows if bool(row.get("native_direct", False))]
    normalised = positive_scaled_rows(rows)
    if not native:
        return []
    key_fields = ("task", "checkpoint", "representation_type", "location_name")
    checkpoint_index = {name: index for index, name in enumerate(CHECKPOINT_ORDER)}
    representation_index = {name: index for index, name in enumerate(REPRESENTATION_ORDER)}
    native.sort(key=lambda row: (
        checkpoint_index.get(row["checkpoint"], len(checkpoint_index)),
        representation_index.get(row["representation_type"], len(representation_index)),
        row["location_name"],
    ))
    pairs = []
    for native_row in native:
        key = tuple(native_row[field] for field in key_fields)
        matches = [row for row in normalised if tuple(row[field] for field in key_fields) == key]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one normalised Direct match for {key}, found {len(matches)}.")
        pairs.append((matches[0], native_row))

    entries: list[tuple[dict[str, Any], str, str]] = []
    y_positions: list[float] = []
    y_labels: list[str] = []
    for pair_index, (normalised_row, native_row) in enumerate(pairs):
        base_y = 2.35 * pair_index
        checkpoint = CHECKPOINT_DISPLAY.get(
            str(native_row["checkpoint"]), str(native_row["checkpoint"]),
        )
        location = display_location(str(native_row["location_name"]))
        entries.extend([
            (normalised_row, "Normalised", COLOR_RED),
            (native_row, "Raw", COLOR_SLATE),
        ])
        y_positions.extend([base_y, base_y + 0.78])
        y_labels.extend([
            f"{checkpoint}   •   {location}\nNormalised",
            "Raw",
        ])

    fig, axes = plt.subplots(
        1, 2, figsize=(12.0, max(7.3, 1.05 * len(pairs) + 2.2)),
        sharey=True,
    )
    for ax, outcome, title, xlabel in (
        (axes[0], "final_answer", "Final-answer correctness", "Changed final-answer cases"),
        (axes[1], "valid_trace", "Trace validity", "Changed valid-trace cases"),
    ):
        maximum = max(
            max(paired_transition_counts(row, outcome)[:2])
            for row, _label, _color in entries
        )
        maximum = max(1, maximum)
        label_pad = max(0.16, 0.025 * maximum)
        for y, (row, _label, color) in zip(y_positions, entries):
            gain, loss, _net = paired_transition_counts(row, outcome)
            ax.barh(
                y, gain, height=0.58, color=color,
                edgecolor=color, linewidth=0.7, zorder=3,
            )
            ax.barh(
                y, -loss, height=0.58, color=color,
                edgecolor=color, alpha=0.34, hatch="//", linewidth=0.7, zorder=3,
            )
            if gain:
                ax.text(
                    gain + label_pad, y, f"+{gain}",
                    ha="left", va="center", fontsize=12.0,
                )
            if loss:
                ax.text(
                    -loss - label_pad, y, f"−{loss}",
                    ha="right", va="center", fontsize=12.0,
                )
            if not gain and not loss:
                ax.plot(
                    0, y, marker="D", markersize=5.0,
                    markerfacecolor=COLOR_PANEL, markeredgecolor=COLOR_BLUE,
                    markeredgewidth=1.0, linestyle="none", zorder=4,
                )
        for pair_index in range(1, len(pairs)):
            ax.axhline(2.35 * pair_index - 0.78, color=COLOR_GRID, linewidth=0.7)
        ax.axvline(0.0, color=COLOR_GREY, linestyle="--", linewidth=1.0, zorder=2)
        ax.set_xlim(-1.22 * maximum, 1.22 * maximum)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda value, _position: f"−{abs(value):g}" if value < 0 else f"{value:g}"
        ))
        integer_count_axis(ax, horizontal=True)
        style_paper_axis(ax, grid_axis="x")
        ax.tick_params(axis="x", labelsize=13.0, colors=COLOR_BLUE)
        ax.set_xlabel(xlabel, fontsize=14.0)
        ax.set_title(title, fontsize=14.0, pad=8)
    axes[0].set_yticks(y_positions, y_labels)
    axes[0].tick_params(
        axis="y",
        labelsize=13.0,
        colors=COLOR_BLUE,
        pad=2,
    )
    axes[0].invert_yaxis()
    fig.legend(
        handles=[
            Patch(facecolor=COLOR_RED, edgecolor=COLOR_RED, label="Normalised"),
            Patch(facecolor=COLOR_SLATE, edgecolor=COLOR_SLATE, label="Raw"),
            Line2D(
                [0], [0], marker="D", linestyle="none", markersize=5.5,
                markerfacecolor=COLOR_PANEL, markeredgecolor=COLOR_BLUE,
                label="No changes",
            ),
        ],
        loc="upper center", ncol=3, bbox_to_anchor=(0.63, 0.94),
        fontsize=11.5,
    )
    fig.suptitle(
        "Effects of vector normalisation on Direct Optimisation steering",
        x=0.63, y=0.958, color=COLOR_TITLE, fontsize=15.5,
    )
    fig.subplots_adjust(
        left=0.27, right=0.985, bottom=0.08, top=0.87, wspace=0.10,
    )
    return save_figure(fig, "steering_native_direct_pairs", args)


PLOTS: dict[str, Callable[[dict[str, Any], argparse.Namespace], list[Path]]] = {
    "dose_response": plot_dose_response,
    "validation_stability": plot_validation_stability,
    "test_comparison": plot_test_comparison,
    "rescue_damage": plot_rescue_damage,
    "validity_tradeoff": plot_validity_tradeoff,
    "validation_heatmap": plot_validation_heatmap,
    "side_effects": plot_side_effects,
    "checkpoint_summary": plot_checkpoint_summary,
    "head_breadth": plot_head_breadth,
    "intervention_frequency_compact": plot_intervention_frequency_compact,
    "native_direct_pairs": plot_native_direct_pairs,
}
PLOT_EXPERIMENT = {
    "dose_response": "steering-atlas",
    "validation_stability": "steering-atlas",
    "test_comparison": "steering-atlas",
    "rescue_damage": "steering-atlas",
    "validity_tradeoff": "steering-atlas",
    "validation_heatmap": "steering-atlas",
    "side_effects": "steering-atlas",
    "checkpoint_summary": "steering-atlas",
    "native_direct_pairs": "steering-atlas",
    "head_breadth": "headcount",
    "intervention_frequency_compact": "staying-power",
}
SUITE_PLOTS: dict[str, Callable[[dict[str, dict[str, Any]], argparse.Namespace], list[Path]]] = {
    "directionality": plot_directionality_summary,
}
CORE_PLOTS = (
    "dose_response", "validation_stability", "test_comparison",
    "rescue_damage", "validity_tradeoff", "validation_heatmap",
    "side_effects", "checkpoint_summary", "head_breadth",
    "intervention_frequency_compact", "native_direct_pairs",
    "directionality",
)


def selected_plots(values: list[str]) -> list[str]:
    if not values:
        raise ValueError("--plotlist requires at least one name.")
    if values == ["all"]:
        return [*PLOTS, *SUITE_PLOTS]
    if values == ["core"]:
        return list(CORE_PLOTS)
    if "all" in values or "core" in values:
        raise ValueError("Use --plotlist all or --plotlist core by itself.")
    unknown = sorted(set(values) - (set(PLOTS) | set(SUITE_PLOTS)))
    if unknown:
        raise ValueError(f"Unknown plots: {unknown}")
    return list(dict.fromkeys(values))


def main() -> None:
    args = parse_args()
    if args.list_plots:
        for name in (*PLOTS, *SUITE_PLOTS):
            print(name)
        return
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    allowed_formats = {"png", "pdf", "svg"}
    unknown_formats = set(args.formats) - allowed_formats
    if unknown_formats:
        raise ValueError(f"Unsupported formats: {sorted(unknown_formats)}")
    manifest_data, artifacts = load_suite(args.input)
    names = selected_plots(args.plotlist)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_count_plots = {
        "dose_response", "validation_stability", "validation_heatmap",
    }
    data_outputs = prepare_paired_counts(
        artifacts,
        args.output_dir,
        include_validation=bool(set(names) & validation_count_plots),
    )
    outputs: list[Path] = []
    for name in names:
        if name in SUITE_PLOTS:
            outputs.extend(SUITE_PLOTS[name](artifacts, args))
        else:
            outputs.extend(PLOTS[name](artifacts[PLOT_EXPERIMENT[name]], args))
    manifest = args.output_dir / EVIDENCE_DIR / "plot_steering_results_manifest.txt"
    manifest_lines = [
        f"Input suite  {args.input}",
        f"Run tag  {manifest_data['run_tag']}",
        f"Suite elapsed minutes  {manifest_data.get('elapsed_minutes', '')}",
        "",
        "COUNT DATA",
        *[str(path) for path in data_outputs],
        "",
        "OUTPUTS",
        *[str(path) for path in outputs],
    ]
    manifest.write_text("\n".join(manifest_lines) + "\n")
    print(f"Input suite: {args.input}")
    print(f"Output: {args.output_dir}")
    print(f"Plots:  {len(outputs)} files")
    print(f"Count data: {len(data_outputs)} files")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
