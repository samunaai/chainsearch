#!/usr/bin/env python3
"""
analyze_astar_reachability_by_backtrack_count.py

Analyze how often probe-eligible after-operation states are forward-unreachable
as a function of the total number of parsed BACK transitions in their generated
trace.

The script reports three complementary quantities for every BACK-count group:

  1. Occurrence-weighted unreachable share
     Long traces contribute once for every eligible after-operation state.

  2. Trace-weighted unreachable share
     Each trace contributes its own unreachable-state fraction once.

  3. Fraction of traces that ever enter a forward-unreachable valid state

Only the pure reachability-probe candidates are used. States where the target is
already present or the generated prefix is invalid are excluded, exactly as in
``label_generated_states_for_probing_samples.py``.

No model is loaded, no inference is run, and no reachability labels are
recomputed.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Iterable

import torch


# =========================
# DEFAULT CONFIG
# =========================

RUN_TAG = "exp34"

ASTAR_DIR = "/workspace/sam/minireason_archives/data_pts/astar"
PROBING_DIR = os.path.join(ASTAR_DIR, "probing")

LABEL_PATH = os.path.join(
    PROBING_DIR,
    f"{RUN_TAG}_generated_state_solvability_for_probing_samples.pt",
)

OUTPUT_PATH = os.path.join(
    PROBING_DIR,
    f"{RUN_TAG}_astar_reachability_by_backtrack_count.pt",
)

EXACT_CSV_PATH = os.path.join(
    PROBING_DIR,
    f"{RUN_TAG}_astar_reachability_by_exact_backtrack_count.csv",
)

GROUPED_CSV_PATH = os.path.join(
    PROBING_DIR,
    f"{RUN_TAG}_astar_reachability_by_backtrack_bin.csv",
)

TAIL_BIN_START = 7
OVERWRITE = True


# =========================
# BASIC HELPERS
# =========================


def safe_torch_save(obj: Any, path: str) -> None:
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, final_path)


def safe_write_csv(
    rows: list[dict[str, Any]],
    path: str,
    fieldnames: Iterable[str],
) -> None:
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")

    with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)

    os.replace(tmp_path, final_path)


def percent(numerator: int | float, denominator: int | float) -> float:
    return 100.0 * float(numerator) / float(denominator) if denominator else float("nan")


def finite_mean(values: torch.Tensor) -> float:
    values = values[torch.isfinite(values)]
    return float(values.mean().item()) if values.numel() else float("nan")


def finite_median(values: torch.Tensor) -> float:
    values = values[torch.isfinite(values)].to(torch.float64)
    if not values.numel():
        return float("nan")
    return float(torch.quantile(values, 0.5).item())


def pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    mask = torch.isfinite(x) & torch.isfinite(y)
    x = x[mask].to(torch.float64)
    y = y[mask].to(torch.float64)

    if x.numel() < 2:
        return float("nan")

    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = torch.sqrt(
        (x_centered.square().sum())
        * (y_centered.square().sum())
    )
    if float(denominator.item()) == 0.0:
        return float("nan")

    return float(
        (x_centered * y_centered).sum().div(denominator).item()
    )


def require_fields(
    table: dict[str, Any],
    fields: set[str],
    table_name: str,
) -> None:
    missing = fields - set(table)
    if missing:
        raise KeyError(
            f"{table_name} is missing required fields: {sorted(missing)}"
        )


# =========================
# ANALYSIS HELPERS
# =========================


def summarize_trace_group(
    *,
    label: str,
    trace_mask: torch.Tensor,
    eligible_per_trace: torch.Tensor,
    reachable_per_trace: torch.Tensor,
    unreachable_per_trace: torch.Tensor,
    unreachable_fraction_per_trace: torch.Tensor,
) -> dict[str, Any]:
    traces = int(trace_mask.sum().item())
    labeled_trace_mask = trace_mask & (eligible_per_trace > 0)
    traces_with_labels = int(labeled_trace_mask.sum().item())

    eligible = int(eligible_per_trace[trace_mask].sum().item())
    reachable = int(reachable_per_trace[trace_mask].sum().item())
    unreachable = int(unreachable_per_trace[trace_mask].sum().item())

    per_trace_fractions = unreachable_fraction_per_trace[labeled_trace_mask]
    ever_unreachable = int(
        (unreachable_per_trace[labeled_trace_mask] > 0).sum().item()
    )

    return {
        "backtrack_group": label,
        "traces": traces,
        "traces_with_probe_labels": traces_with_labels,
        "traces_without_probe_labels": traces - traces_with_labels,
        "eligible_state_occurrences": eligible,
        "reachable_state_occurrences": reachable,
        "unreachable_state_occurrences": unreachable,
        "occurrence_weighted_unreachable_pct": percent(unreachable, eligible),
        "mean_trace_unreachable_pct": 100.0 * finite_mean(per_trace_fractions),
        "median_trace_unreachable_pct": 100.0 * finite_median(per_trace_fractions),
        "traces_ever_unreachable": ever_unreachable,
        "traces_ever_unreachable_pct": percent(
            ever_unreachable,
            traces_with_labels,
        ),
        "mean_eligible_states_per_labeled_trace": (
            float(eligible) / traces_with_labels
            if traces_with_labels
            else float("nan")
        ),
    }


def build_exact_rows(
    *,
    back_counts: torch.Tensor,
    eligible_per_trace: torch.Tensor,
    reachable_per_trace: torch.Tensor,
    unreachable_per_trace: torch.Tensor,
    unreachable_fraction_per_trace: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for count in torch.unique(back_counts, sorted=True).tolist():
        count = int(count)
        rows.append(
            summarize_trace_group(
                label=str(count),
                trace_mask=back_counts == count,
                eligible_per_trace=eligible_per_trace,
                reachable_per_trace=reachable_per_trace,
                unreachable_per_trace=unreachable_per_trace,
                unreachable_fraction_per_trace=unreachable_fraction_per_trace,
            )
        )
    return rows


def build_grouped_rows(
    *,
    back_counts: torch.Tensor,
    eligible_per_trace: torch.Tensor,
    reachable_per_trace: torch.Tensor,
    unreachable_per_trace: torch.Tensor,
    unreachable_fraction_per_trace: torch.Tensor,
    tail_bin_start: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for count in range(tail_bin_start):
        rows.append(
            summarize_trace_group(
                label=str(count),
                trace_mask=back_counts == count,
                eligible_per_trace=eligible_per_trace,
                reachable_per_trace=reachable_per_trace,
                unreachable_per_trace=unreachable_per_trace,
                unreachable_fraction_per_trace=unreachable_fraction_per_trace,
            )
        )

    rows.append(
        summarize_trace_group(
            label=f"{tail_bin_start}+",
            trace_mask=back_counts >= tail_bin_start,
            eligible_per_trace=eligible_per_trace,
            reachable_per_trace=reachable_per_trace,
            unreachable_per_trace=unreachable_per_trace,
            unreachable_fraction_per_trace=unreachable_fraction_per_trace,
        )
    )

    return rows


def print_group_table(
    title: str,
    rows: list[dict[str, Any]],
) -> None:
    print()
    print(title)
    print(
        f"{'BACKs':>7} "
        f"{'traces':>10} "
        f"{'labeled':>10} "
        f"{'eligible':>12} "
        f"{'unreach':>12} "
        f"{'occ-unr%':>10} "
        f"{'mean-tr%':>10} "
        f"{'ever-unr%':>11}"
    )
    print("-" * 92)

    for row in rows:
        print(
            f"{str(row['backtrack_group']):>7} "
            f"{int(row['traces']):>10,} "
            f"{int(row['traces_with_probe_labels']):>10,} "
            f"{int(row['eligible_state_occurrences']):>12,} "
            f"{int(row['unreachable_state_occurrences']):>12,} "
            f"{float(row['occurrence_weighted_unreachable_pct']):>9.2f}% "
            f"{float(row['mean_trace_unreachable_pct']):>9.2f}% "
            f"{float(row['traces_ever_unreachable_pct']):>10.2f}%"
        )


# =========================
# MAIN
# =========================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze probe-eligible forward-unreachable states by the total "
            "number of parsed BACK transitions in each generated trace."
        )
    )
    parser.add_argument(
        "--label-path",
        default=LABEL_PATH,
        help="Path to the generated-state reachability label artifact.",
    )
    parser.add_argument(
        "--tail-bin-start",
        type=int,
        default=TAIL_BIN_START,
        help=(
            "Start of the final grouped BACK-count bin. "
            f"The default produces 0..{TAIL_BIN_START - 1} and {TAIL_BIN_START}+."
        ),
    )
    args = parser.parse_args()

    if args.tail_bin_start < 1:
        raise ValueError("--tail-bin-start must be at least 1.")

    label_path = os.path.realpath(args.label_path)
    if not os.path.isfile(label_path):
        raise FileNotFoundError(label_path)

    for output_path in (OUTPUT_PATH, EXACT_CSV_PATH, GROUPED_CSV_PATH):
        if os.path.exists(output_path) and not OVERWRITE:
            raise FileExistsError(output_path)

    width = 104
    print("=" * width)
    print("A* REACHABILITY BY PARSED BACKTRACK COUNT".center(width))
    print("=" * width)
    print(f"Label artifact:        {label_path}")
    print(f"Structured output:     {OUTPUT_PATH}")
    print(f"Exact-count CSV:       {EXACT_CSV_PATH}")
    print(f"Grouped-count CSV:     {GROUPED_CSV_PATH}")
    print(
        "Denominator:           probe-eligible after-operation states only "
        "(valid prefix, target not already present)"
    )
    print("=" * width)
    print()

    print("Loading reachability labels...", flush=True)
    data = torch.load(
        label_path,
        map_location="cpu",
        weights_only=False,
    )

    if not bool(data.get("complete", False)):
        raise RuntimeError("Label artifact is not marked complete.")
    if int(data.get("format_version", -1)) < 4:
        raise RuntimeError(
            f"Expected label format version at least 4, got {data.get('format_version')!r}."
        )

    generated_traces = data.get("generated_traces")
    state_occurrences = data.get("state_occurrences")
    if not isinstance(generated_traces, dict):
        raise TypeError("Label artifact has no generated_traces table.")
    if not isinstance(state_occurrences, dict):
        raise TypeError("Label artifact has no state_occurrences table.")

    require_fields(
        generated_traces,
        {"cache_idx", "n_backtracks"},
        "generated_traces",
    )
    require_fields(
        state_occurrences,
        {
            "record_pos",
            "position_type",
            "primary_probe_candidate",
            "binary_reachability_label",
            "min_steps_remaining",
        },
        "state_occurrences",
    )

    position_type_to_id = data.get("state_position_type_to_id") or {}
    if "after_operation" not in position_type_to_id:
        raise KeyError("Label artifact does not define after_operation position type.")
    after_operation_id = int(position_type_to_id["after_operation"])

    n_traces = int(generated_traces["cache_idx"].numel())
    back_counts = generated_traces["n_backtracks"].to(torch.int64)
    if back_counts.numel() != n_traces:
        raise RuntimeError("Generated-trace BACK counts are misaligned.")
    if bool((back_counts < 0).any()):
        raise RuntimeError("Generated-trace BACK counts contain a negative value.")

    eligible = state_occurrences["primary_probe_candidate"].to(torch.bool)
    labels = state_occurrences["binary_reachability_label"].to(torch.bool)
    record_pos = state_occurrences["record_pos"].to(torch.int64)
    position_type = state_occurrences["position_type"].to(torch.int64)
    min_steps = state_occurrences["min_steps_remaining"].to(torch.int16)

    if not (
        eligible.numel()
        == labels.numel()
        == record_pos.numel()
        == position_type.numel()
        == min_steps.numel()
    ):
        raise RuntimeError("State-occurrence fields are misaligned.")

    if bool((record_pos < 0).any()) or bool((record_pos >= n_traces).any()):
        raise RuntimeError("State-occurrence record_pos is outside the trace table.")

    if bool((position_type[eligible] != after_operation_id).any()):
        raise RuntimeError(
            "primary_probe_candidate contains a non-after-operation position."
        )
    if bool((min_steps[eligible] == 0).any()) or bool((min_steps[eligible] == -2).any()):
        raise RuntimeError(
            "Probe-eligible states unexpectedly include target-present or unknown labels."
        )
    if not torch.equal(labels[eligible], min_steps[eligible] >= 1):
        raise RuntimeError("Binary reachability labels do not match minimum steps.")

    eligible_trace = record_pos[eligible]
    eligible_label = labels[eligible]
    unreachable_event = ~eligible_label

    eligible_per_trace = torch.bincount(
        eligible_trace,
        minlength=n_traces,
    ).to(torch.int64)
    reachable_per_trace = torch.bincount(
        eligible_trace[eligible_label],
        minlength=n_traces,
    ).to(torch.int64)
    unreachable_per_trace = torch.bincount(
        eligible_trace[unreachable_event],
        minlength=n_traces,
    ).to(torch.int64)

    if not torch.equal(
        eligible_per_trace,
        reachable_per_trace + unreachable_per_trace,
    ):
        raise RuntimeError("Per-trace reachable and unreachable counts do not sum correctly.")

    has_eligible = eligible_per_trace > 0
    unreachable_fraction_per_trace = torch.full(
        (n_traces,),
        float("nan"),
        dtype=torch.float32,
    )
    unreachable_fraction_per_trace[has_eligible] = (
        unreachable_per_trace[has_eligible].to(torch.float32)
        / eligible_per_trace[has_eligible].to(torch.float32)
    )

    total_eligible = int(eligible_per_trace.sum().item())
    total_reachable = int(reachable_per_trace.sum().item())
    total_unreachable = int(unreachable_per_trace.sum().item())

    source_summary = data.get("summary") or {}
    expected_reachable = source_summary.get("reachable_binary_labels")
    expected_unreachable = source_summary.get("unreachable_binary_labels")
    if expected_reachable is not None and int(expected_reachable) != total_reachable:
        raise RuntimeError(
            "Reachable count does not match the source label summary."
        )
    if expected_unreachable is not None and int(expected_unreachable) != total_unreachable:
        raise RuntimeError(
            "Unreachable count does not match the source label summary."
        )

    exact_rows = build_exact_rows(
        back_counts=back_counts,
        eligible_per_trace=eligible_per_trace,
        reachable_per_trace=reachable_per_trace,
        unreachable_per_trace=unreachable_per_trace,
        unreachable_fraction_per_trace=unreachable_fraction_per_trace,
    )
    grouped_rows = build_grouped_rows(
        back_counts=back_counts,
        eligible_per_trace=eligible_per_trace,
        reachable_per_trace=reachable_per_trace,
        unreachable_per_trace=unreachable_per_trace,
        unreachable_fraction_per_trace=unreachable_fraction_per_trace,
        tail_bin_start=int(args.tail_bin_start),
    )

    labeled_back_counts = back_counts[has_eligible].to(torch.float32)
    labeled_unreachable_fraction = unreachable_fraction_per_trace[has_eligible]
    ever_unreachable = (unreachable_per_trace[has_eligible] > 0).to(torch.float32)

    summary = {
        "generated_traces": n_traces,
        "traces_with_probe_labels": int(has_eligible.sum().item()),
        "traces_without_probe_labels": int((~has_eligible).sum().item()),
        "eligible_state_occurrences": total_eligible,
        "reachable_state_occurrences": total_reachable,
        "unreachable_state_occurrences": total_unreachable,
        "overall_occurrence_weighted_unreachable_pct": percent(
            total_unreachable,
            total_eligible,
        ),
        "overall_mean_trace_unreachable_pct": (
            100.0 * finite_mean(unreachable_fraction_per_trace[has_eligible])
        ),
        "overall_median_trace_unreachable_pct": (
            100.0 * finite_median(unreachable_fraction_per_trace[has_eligible])
        ),
        "traces_ever_unreachable": int(
            (unreachable_per_trace[has_eligible] > 0).sum().item()
        ),
        "traces_ever_unreachable_pct": percent(
            int((unreachable_per_trace[has_eligible] > 0).sum().item()),
            int(has_eligible.sum().item()),
        ),
        "mean_parsed_back_count": float(back_counts.to(torch.float32).mean().item()),
        "maximum_parsed_back_count": int(back_counts.max().item()),
        "pearson_back_count_vs_trace_unreachable_fraction": pearson_correlation(
            labeled_back_counts,
            labeled_unreachable_fraction,
        ),
        "pearson_back_count_vs_ever_unreachable": pearson_correlation(
            labeled_back_counts,
            ever_unreachable,
        ),
    }

    csv_fields = [
        "backtrack_group",
        "traces",
        "traces_with_probe_labels",
        "traces_without_probe_labels",
        "eligible_state_occurrences",
        "reachable_state_occurrences",
        "unreachable_state_occurrences",
        "occurrence_weighted_unreachable_pct",
        "mean_trace_unreachable_pct",
        "median_trace_unreachable_pct",
        "traces_ever_unreachable",
        "traces_ever_unreachable_pct",
        "mean_eligible_states_per_labeled_trace",
    ]

    safe_write_csv(exact_rows, EXACT_CSV_PATH, csv_fields)
    safe_write_csv(grouped_rows, GROUPED_CSV_PATH, csv_fields)

    analysis_payload = {
        "format_version": 1,
        "complete": True,
        "run_tag": data.get("run_tag", RUN_TAG),
        "source_label_path": label_path,
        "definitions": {
            "eligible_state": (
                "state_occurrences.primary_probe_candidate is true; the state is "
                "after an operation, legally reached, and the target is not already present"
            ),
            "occurrence_weighted_unreachable_share": (
                "unreachable eligible state occurrences divided by all eligible state occurrences"
            ),
            "trace_weighted_unreachable_share": (
                "mean of each trace's unreachable eligible-state fraction, giving each trace equal weight"
            ),
            "trace_ever_unreachable": (
                "trace contains at least one probe-eligible forward-unreachable state"
            ),
            "back_count": (
                "total parsed BACK transitions in the generated trace"
            ),
        },
        "summary": summary,
        "per_trace": {
            "record_pos": torch.arange(n_traces, dtype=torch.int32),
            "cache_idx": generated_traces["cache_idx"].to(torch.int64),
            "parsed_back_count": back_counts.to(torch.int16),
            "eligible_state_count": eligible_per_trace.to(torch.int16),
            "reachable_state_count": reachable_per_trace.to(torch.int16),
            "unreachable_state_count": unreachable_per_trace.to(torch.int16),
            "unreachable_fraction": unreachable_fraction_per_trace,
            "has_probe_labels": has_eligible,
            "ever_unreachable": unreachable_per_trace > 0,
        },
        "exact_by_back_count": exact_rows,
        "grouped_by_back_count": grouped_rows,
        "tail_bin_start": int(args.tail_bin_start),
        "exact_csv_path": EXACT_CSV_PATH,
        "grouped_csv_path": GROUPED_CSV_PATH,
    }
    safe_torch_save(analysis_payload, OUTPUT_PATH)

    print("ANALYSIS SUMMARY")
    print(f"  Generated traces:                       {summary['generated_traces']:>12,}")
    print(f"  Traces with probe labels:               {summary['traces_with_probe_labels']:>12,}")
    print(f"  Eligible state occurrences:             {summary['eligible_state_occurrences']:>12,}")
    print(f"  Reachable state occurrences:            {summary['reachable_state_occurrences']:>12,}")
    print(f"  Unreachable state occurrences:          {summary['unreachable_state_occurrences']:>12,}")
    print(
        f"  Overall occurrence-weighted unreachable:"
        f" {summary['overall_occurrence_weighted_unreachable_pct']:>10.2f}%"
    )
    print(
        f"  Overall mean trace-level unreachable:   "
        f"{summary['overall_mean_trace_unreachable_pct']:>10.2f}%"
    )
    print(
        f"  Traces that ever become unreachable:    "
        f"{summary['traces_ever_unreachable_pct']:>10.2f}%"
    )
    print(f"  Mean parsed BACK count:                 {summary['mean_parsed_back_count']:>12.2f}")
    print(f"  Maximum parsed BACK count:              {summary['maximum_parsed_back_count']:>12,}")
    print(
        f"  Pearson r, BACKs vs trace unreachable fraction: "
        f"{summary['pearson_back_count_vs_trace_unreachable_fraction']:>8.4f}"
    )
    print(
        f"  Pearson r, BACKs vs ever unreachable:         "
        f"{summary['pearson_back_count_vs_ever_unreachable']:>8.4f}"
    )

    print_group_table(
        f"GROUPED BY TOTAL PARSED BACK COUNT (0..{args.tail_bin_start - 1}, {args.tail_bin_start}+)",
        grouped_rows,
    )

    print()
    print("Column meanings")
    print("  occ-unr%   occurrence-weighted unreachable share")
    print("  mean-tr%   mean within-trace unreachable share")
    print("  ever-unr%  fraction of labeled traces with at least one unreachable state")
    print()
    print(f"Saved structured analysis: {OUTPUT_PATH}")
    print(f"Saved exact-count table:    {EXACT_CSV_PATH}")
    print(f"Saved grouped table:        {GROUPED_CSV_PATH}")
    print("=" * width)


if __name__ == "__main__":
    main()
