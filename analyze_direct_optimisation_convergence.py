#!/usr/bin/env python3
"""
analyze_direct_optimisation_convergence.py

Inspect the saved Direct Optimisation archive without rerunning the model.
The analysis focuses on runs that reached the optimisation horizon before
patience could establish a plateau, then asks whether their directions and
validation losses were nevertheless settling near the end of training.
"""

from __future__ import annotations

import argparse
import math
import os
from collections import Counter, defaultdict
from typing import Any

import torch

import compute_linear_probe_vectors as probe


ZERO_NORM_EPS = 1e-12
DEFAULT_DIRECTION_THRESHOLD = 0.999
DEFAULT_LOSS_GAIN_THRESHOLD = 1e-4
DEFAULT_RECENT_INTERVALS = 4
WIDTH = 104


def heading(text: str) -> None:
    print()
    print("=" * WIDTH)
    print(text.center(WIDTH))
    print("=" * WIDTH)


def print_rows(title: str, rows: list[tuple[str, Any]]) -> None:
    print(title)
    width = max(len(str(name)) for name, _ in rows) if rows else 0
    for name, value in rows:
        print(f"  {name:<{width}}  {value}")


def quantiles(values: list[float]) -> dict[str, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {name: float("nan") for name in ("min", "p10", "p25", "median", "p75", "p90", "max")}
    tensor = torch.tensor(finite, dtype=torch.float64)
    probs = torch.tensor([0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0], dtype=torch.float64)
    values_q = torch.quantile(tensor, probs).tolist()
    return dict(zip(("min", "p10", "p25", "median", "p75", "p90", "max"), values_q))


def format_quantiles(values: list[float], digits: int = 6) -> str:
    q = quantiles(values)
    if not math.isfinite(q["median"]):
        return "n/a"
    fmt = f"{{:.{digits}f}}"
    return " / ".join(fmt.format(q[name]) for name in ("min", "p10", "p25", "median", "p75", "p90", "max"))


def parse_optimisation_id(optimisation_id: str) -> tuple[str, str, str]:
    parts = optimisation_id.split(":")
    if len(parts) != 3:
        raise RuntimeError(f"Unexpected optimisation ID: {optimisation_id!r}")
    return parts[0], parts[1], parts[2]


def site_family(site: str) -> str:
    if site.startswith("attention_"):
        return "attention"
    if site.startswith("block_"):
        return "block"
    return "other"


def site_layer(site: str) -> int:
    marker = "_L"
    if marker not in site:
        return -1
    suffix = site.split(marker, 1)[1]
    digits = []
    for char in suffix:
        if not char.isdigit():
            break
        digits.append(char)
    return int("".join(digits)) if digits else -1


def recent_history(summary: dict[str, Any], intervals: int) -> list[dict[str, Any]]:
    history = sorted(summary.get("validation_history", []), key=lambda row: int(row["step"]))
    return history[-(intervals + 1):]


def budget_limited(summary: dict[str, Any], max_steps: int, patience: int) -> bool:
    if int(summary["steps_completed"]) != max_steps:
        return False
    best_step = int(summary["best_step"])
    checks_after_best = sum(int(row["step"]) > best_step for row in summary.get("validation_history", []))
    return checks_after_best < patience


def analyse_run(
    optimisation_id: str,
    summary: dict[str, Any],
    max_steps: int,
    patience: int,
    recent_intervals: int,
    direction_threshold: float,
    loss_gain_threshold: float,
) -> dict[str, Any]:
    task, checkpoint, site = parse_optimisation_id(optimisation_id)
    history = sorted(summary.get("validation_history", []), key=lambda row: int(row["step"]))
    recent = recent_history(summary, recent_intervals)

    recent_cosine_rows = recent[1:] if len(recent) > 1 else []
    recent_cosines = [
        float(row["direction_cosine_to_previous_validation"])
        for row in recent_cosine_rows
        if math.isfinite(float(row.get("direction_cosine_to_previous_validation", float("nan"))))
    ]
    final_cosine = recent_cosines[-1] if recent_cosines else float("nan")
    recent_median_cosine = quantiles(recent_cosines)["median"] if recent_cosines else float("nan")
    recent_min_cosine = min(recent_cosines) if recent_cosines else float("nan")
    cosine_gain = recent_cosines[-1] - recent_cosines[0] if len(recent_cosines) >= 2 else float("nan")

    recent_losses = [float(row["loss"]) for row in recent]
    recent_best_gain = float("nan")
    recent_final_drop = float("nan")
    if len(recent_losses) >= 2:
        recent_best_gain = recent_losses[0] - min(recent_losses[1:])
        recent_final_drop = recent_losses[0] - recent_losses[-1]

    recent_norm_change = float("nan")
    if len(recent) >= 2:
        first_norm = float(recent[0]["vector_norm"])
        last_norm = float(recent[-1]["vector_norm"])
        if abs(first_norm) > ZERO_NORM_EPS:
            recent_norm_change = (last_norm - first_norm) / abs(first_norm)

    zero = float(summary.get("vector_norm", 0.0)) <= ZERO_NORM_EPS
    stable = (
        not zero
        and math.isfinite(final_cosine)
        and math.isfinite(recent_median_cosine)
        and final_cosine >= direction_threshold
        and recent_median_cosine >= direction_threshold
    )
    improving = math.isfinite(recent_best_gain) and recent_best_gain > loss_gain_threshold

    limited = budget_limited(summary, max_steps, patience)
    if zero:
        category = "zero"
    elif stable and improving:
        category = "stable + improving"
    elif stable:
        category = "stable + plateaued"
    elif improving:
        category = "changing + improving"
    else:
        category = "changing + flat/noisy"

    return {
        "optimisation_id": optimisation_id,
        "task": task,
        "checkpoint": checkpoint,
        "site": site,
        "family": site_family(site),
        "layer": site_layer(site),
        "best_step": int(summary["best_step"]),
        "steps_completed": int(summary["steps_completed"]),
        "budget_limited": limited,
        "zero": zero,
        "final_cosine": final_cosine,
        "recent_median_cosine": recent_median_cosine,
        "recent_min_cosine": recent_min_cosine,
        "cosine_gain": cosine_gain,
        "recent_best_gain": recent_best_gain,
        "recent_final_drop": recent_final_drop,
        "recent_norm_change": recent_norm_change,
        "category": category,
        "history_points": len(history),
    }


def category_counts(rows: list[dict[str, Any]]) -> Counter:
    order = ["stable + plateaued", "stable + improving", "changing + improving", "changing + flat/noisy", "zero"]
    counts = Counter(row["category"] for row in rows)
    return Counter({name: counts[name] for name in order if counts[name]})


def print_group_table(title: str, groups: dict[str, list[dict[str, Any]]]) -> None:
    print()
    print(title)
    print("  group                           n    stable   changing    zero")
    print("  " + "-" * 64)
    for name, rows in groups.items():
        stable = sum(row["category"].startswith("stable") for row in rows)
        changing = sum(row["category"].startswith("changing") for row in rows)
        zero = sum(row["zero"] for row in rows)
        print(f"  {name:<30} {len(rows):>4} {stable:>9} {changing:>10} {zero:>7}")


def print_concerning(rows: list[dict[str, Any]], limit: int) -> None:
    candidates = [row for row in rows if not row["zero"] and math.isfinite(row["final_cosine"])]
    candidates.sort(key=lambda row: (row["final_cosine"], -row["recent_best_gain"] if math.isfinite(row["recent_best_gain"]) else 0.0))
    if not candidates:
        return

    print()
    print(f"LOWEST FINAL DIRECTION COSINES AMONG BUDGET-LIMITED RUNS (first {min(limit, len(candidates))})")
    print("  checkpoint                    site               best/steps  final cos  recent med   loss gain  category")
    print("  " + "-" * 100)
    for row in candidates[:limit]:
        loss_gain = row["recent_best_gain"]
        loss_text = f"{loss_gain:.6g}" if math.isfinite(loss_gain) else "n/a"
        print(
            f"  {row['checkpoint']:<29} {row['site']:<18} "
            f"{row['best_step']:>4}/{row['steps_completed']:<4} "
            f"{row['final_cosine']:>10.6f} {row['recent_median_cosine']:>11.6f} "
            f"{loss_text:>11}  {row['category']}"
        )


def main() -> None:
    default_path = os.path.join(
        probe.ASTAR_DIR,
        "direct_optimisation",
        f"{probe.RUN_TAG}_astar_direct_optimisation_vectors.pt",
    )

    parser = argparse.ArgumentParser(description="Analyse convergence of saved Direct Optimisation vectors.")
    parser.add_argument("--input", default=default_path)
    parser.add_argument("--recent-intervals", type=int, default=DEFAULT_RECENT_INTERVALS)
    parser.add_argument("--direction-threshold", type=float, default=DEFAULT_DIRECTION_THRESHOLD)
    parser.add_argument("--loss-gain-threshold", type=float, default=DEFAULT_LOSS_GAIN_THRESHOLD)
    parser.add_argument("--show-worst", type=int, default=20)
    args = parser.parse_args()

    if args.recent_intervals <= 0 or args.show_worst < 0:
        raise ValueError("--recent-intervals must be positive and --show-worst must be non-negative.")
    if not (0.0 < args.direction_threshold <= 1.0):
        raise ValueError("--direction-threshold must lie in (0, 1].")
    if args.loss_gain_threshold < 0:
        raise ValueError("--loss-gain-threshold must be non-negative.")
    if not os.path.isfile(args.input):
        raise FileNotFoundError(args.input)

    archive = torch.load(args.input, map_location="cpu", weights_only=False)
    if not bool(archive.get("complete", False)):
        raise RuntimeError("Direct Optimisation archive is incomplete.")
    if archive.get("method") != "direct_optimisation":
        raise RuntimeError(f"Unexpected method: {archive.get('method')!r}")

    summaries = archive.get("optimisation_summaries", {})
    if not summaries:
        raise RuntimeError("Archive contains no optimisation_summaries.")
    
    expected_runs = archive.get("audit", {}).get("unique_optimisation_runs")
    if expected_runs is not None and len(summaries) != int(expected_runs):
        raise RuntimeError(
            f"Archive audit expects {expected_runs} unique optimisations, "
            f"but optimisation_summaries contains {len(summaries)}."
        )
        
    run_config = archive.get("run_config", {})
    config = run_config.get("config", archive.get("optimisation_config", {}))
    max_steps = int(config["max_steps"])
    val_every = int(config["val_every"])
    patience = int(config["patience"])

    rows = [
        analyse_run(
            optimisation_id,
            summary,
            max_steps,
            patience,
            args.recent_intervals,
            args.direction_threshold,
            args.loss_gain_threshold,
        )
        for optimisation_id, summary in summaries.items()
    ]
    limited = [row for row in rows if row["budget_limited"]]
    nonzero_limited = [row for row in limited if not row["zero"]]

    heading("ANALYSE DIRECT OPTIMISATION CONVERGENCE")
    print_rows(
        "ARCHIVE",
        [
            ("Input", args.input),
            ("Run tag", archive.get("run_tag", "n/a")),
            ("Unique optimisation runs", len(rows)),
            ("Maximum steps", max_steps),
            ("Validation interval", val_every),
            ("Patience", patience),
            ("Recent window", f"{args.recent_intervals} intervals = {args.recent_intervals * val_every} steps"),
            ("Direction threshold", args.direction_threshold),
            ("Material recent loss gain", args.loss_gain_threshold),
        ],
    )

    best_counts = Counter(row["best_step"] for row in rows)
    reached_max = sum(row["steps_completed"] == max_steps for row in rows)
    best_at_max = sum(row["best_step"] == max_steps for row in rows)
    print_rows(
        "HORIZON",
        [
            ("Reached maximum steps", f"{reached_max}/{len(rows)}"),
            ("Best step = maximum", f"{best_at_max}/{len(rows)}"),
            ("Budget-limited", f"{len(limited)}/{len(rows)} ({100 * len(limited) / len(rows):.1f}%)"),
            ("Best-step counts", dict(sorted(best_counts.items()))),
        ],
    )

    final_cosines = [row["final_cosine"] for row in nonzero_limited]
    recent_medians = [row["recent_median_cosine"] for row in nonzero_limited]
    recent_mins = [row["recent_min_cosine"] for row in nonzero_limited]
    cosine_gains = [row["cosine_gain"] for row in nonzero_limited]
    recent_best_gains = [row["recent_best_gain"] for row in nonzero_limited]
    recent_norm_changes = [row["recent_norm_change"] for row in nonzero_limited]

    stable_final = sum(math.isfinite(value) and value >= args.direction_threshold for value in final_cosines)
    stable_recent = sum(
        math.isfinite(row["final_cosine"])
        and math.isfinite(row["recent_median_cosine"])
        and row["final_cosine"] >= args.direction_threshold
        and row["recent_median_cosine"] >= args.direction_threshold
        for row in nonzero_limited
    )

    print_rows(
        "DIRECTION STABILITY AMONG BUDGET-LIMITED RUNS",
        [
            ("Quantile order", "min / p10 / p25 / median / p75 / p90 / max"),
            ("Final successive-direction cosine", format_quantiles(final_cosines)),
            ("Recent median direction cosine", format_quantiles(recent_medians)),
            ("Recent minimum direction cosine", format_quantiles(recent_mins)),
            ("Recent cosine gain", format_quantiles(cosine_gains)),
            ("Final cosine above threshold", f"{stable_final}/{len(nonzero_limited)}"),
            ("Final + recent median stable", f"{stable_recent}/{len(nonzero_limited)}"),
        ],
    )

    bands = [
        (">= 0.9999", lambda x: x >= 0.9999),
        ("0.9990--0.9999", lambda x: 0.999 <= x < 0.9999),
        ("0.9950--0.9990", lambda x: 0.995 <= x < 0.999),
        ("0.9900--0.9950", lambda x: 0.990 <= x < 0.995),
        ("< 0.9900", lambda x: x < 0.990),
    ]
    print_rows(
        "FINAL COSINE BANDS",
        [(name, sum(math.isfinite(value) and predicate(value) for value in final_cosines)) for name, predicate in bands],
    )

    print_rows(
        "RECENT LOSS AND NORM MOVEMENT",
        [
            ("Recent best validation-loss gain", format_quantiles(recent_best_gains)),
            ("Recent relative norm change", format_quantiles(recent_norm_changes)),
            ("Runs with material recent loss gain", f"{sum(math.isfinite(x) and x > args.loss_gain_threshold for x in recent_best_gains)}/{len(nonzero_limited)}"),
            ("Loss gain > 1e-5", sum(math.isfinite(x) and x > 1e-5 for x in recent_best_gains)),
            ("Loss gain > 1e-4", sum(math.isfinite(x) and x > 1e-4 for x in recent_best_gains)),
            ("Loss gain > 1e-3", sum(math.isfinite(x) and x > 1e-3 for x in recent_best_gains)),
        ],
    )

    counts = category_counts(limited)
    print_rows(
        "BUDGET-LIMITED CLASSIFICATION",
        [(name, f"{count}/{len(limited)} ({100 * count / len(limited):.1f}%)") for name, count in counts.items()],
    )

    groups_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups_checkpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in limited:
        groups_family[row["family"]].append(row)
        groups_layer[f"layer {row['layer']}"].append(row)
        groups_checkpoint[row["checkpoint"]].append(row)

    print_group_table("BUDGET-LIMITED BY REPRESENTATION FAMILY", dict(sorted(groups_family.items())))
    print_group_table("BUDGET-LIMITED BY LAYER", dict(sorted(groups_layer.items())))
    print_group_table("BUDGET-LIMITED BY CHECKPOINT", dict(sorted(groups_checkpoint.items())))

    if args.show_worst:
        print_concerning(limited, args.show_worst)

    changing_improving = sum(row["category"] == "changing + improving" for row in limited)
    changing_flat = sum(row["category"] == "changing + flat/noisy" for row in limited)
    stable_any = sum(row["category"].startswith("stable") for row in limited)
    below_995 = sum(math.isfinite(row["final_cosine"]) and row["final_cosine"] < 0.995 for row in nonzero_limited)
    below_990 = sum(math.isfinite(row["final_cosine"]) and row["final_cosine"] < 0.990 for row in nonzero_limited)
    below_995_recent = sum(
        math.isfinite(row["recent_median_cosine"])
        and row["recent_median_cosine"] < 0.995
        for row in nonzero_limited
    )

    if not limited:
        assessment = "PASS: patience established a plateau for every run before the optimisation horizon."
    elif changing_improving == 0 and below_995 == 0 and below_995_recent == 0:
        assessment = "LIKELY SUFFICIENT FOR DIRECTION-BASED STEERING: remaining horizon-limited directions are already very stable."
    elif changing_improving > 0:
        assessment = "REVIEW / EXTENSION MAY HELP: some horizon-limited directions are still changing while validation loss is materially improving."
    else:
        assessment = "REVIEW: some directions are still changing, but longer optimisation is not clearly supported by recent validation-loss progress."

    heading("CONVERGENCE ASSESSMENT")
    print_rows(
        "SUMMARY",
        [
            ("Budget-limited runs", len(limited)),
            ("Stable direction categories", stable_any),
            ("Changing + improving", changing_improving),
            ("Changing + flat/noisy", changing_flat),
            ("Final cosine < 0.995", below_995),
            ("Recent median cosine < 0.995", below_995_recent),
            ("Final cosine < 0.990", below_990),
            ("Assessment", assessment),
        ],
    )
    print("=" * WIDTH)


if __name__ == "__main__":
    main()
