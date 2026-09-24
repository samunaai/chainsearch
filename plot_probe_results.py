#!/usr/bin/env python3
"""
Plot the saved A* probing results.

Examples

    python -u plot_probe_results.py --plotlist all
    python -u plot_probe_results.py --plotlist core
    python -u plot_probe_results.py --plotlist trace_progress decision_point_progress
    python -u plot_probe_results.py --list-plots
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm


# =============================================================================
# CONFIGURATION
# =============================================================================

RUN_TAG = "exp34"

ASTAR_DIR = Path("/workspace/user/chainsearch_archives/data_pts/astar")
PROBING_DIR = ASTAR_DIR / "probing"

RESULTS_PATH = PROBING_DIR / f"{RUN_TAG}_astar_probe_results.pt"
LABEL_PATH = PROBING_DIR / f"{RUN_TAG}_generated_state_solvability_for_probing_samples.pt"

DEFAULT_METRIC = "test_auroc"
DEFAULT_DPI = 220
HEATMAP_CELL_ASPECT = 0.58
TRACE_PROGRESS_BINS = 8
DECISION_PROGRESS_BINS = 10
TOP_ATTENTION_HEADS = 4
SAMPLE_POINTS_PER_CLASS = 500
MIN_AUROC_CLASS_COUNT = 20

# Independently control the compact Figure 3 and Figure 4 title heights.
# Larger values move the title upward; smaller values move it downward.
COMPACT_ALL_TRACES_TITLE_Y = 1.065
COMPACT_MAX_8_BACKTRACKS_TITLE_Y = 1.065

COLOR_BG = "#FCFCFD"
COLOR_CANVAS = "#FFFFFF"
COLOR_PANEL = "#F2F6FF"
COLOR_DEEP_BLUE = "#1C4587"
COLOR_PASTEL_BLUE = "#7CA6DE"
COLOR_SLATE_BLUE = "#5F7896"
COLOR_DARK_TEXT = "#26364F"
COLOR_FIGURE_TITLE = "#000000"

COLOR_DEEP_RED = "#D57758"
COLOR_RED = "#D98468"
COLOR_MID_RED = "#DE9177"
COLOR_SOFT_RED = "#E19D85"
COLOR_PASTEL_RED = "#E9B3A1"
COLOR_PALE_RED = "#FCF2EE"

COLOR_GRID = "#C9D6EE"
COLOR_REFLINE = "#9AA7BD"

PROBE_SCORE_CMAP = LinearSegmentedColormap.from_list(
    "probe_score",
    [
        (0.00, COLOR_DEEP_RED),
        (0.12, COLOR_RED),
        (0.23, COLOR_MID_RED),
        (0.34, COLOR_SOFT_RED),
        (0.43, COLOR_PASTEL_RED),
        (0.48, COLOR_PALE_RED),
        (0.50, COLOR_BG),
        (0.54, "#EEF3FB"),
        (0.68, "#C9D9F2"),
        (0.84, COLOR_PASTEL_BLUE),
        (1.00, COLOR_DEEP_BLUE),
    ],
)

REPRESENTATION_ORDER = ("attention_head", "residual_stream", "mlp_output", "final_hidden")

REPRESENTATION_DISPLAY = {
    "attention_head": "Attention heads",
    "residual_stream": "Residual stream",
    "mlp_output": "MLP output",
    "final_hidden": "Final hidden",
}

REPRESENTATION_STYLE = {
    "attention_head": (COLOR_PASTEL_BLUE, "o", "-"),
    "residual_stream": (COLOR_DEEP_BLUE, "s", "-"),
    "mlp_output": (COLOR_SLATE_BLUE, "^", "--"),
    "final_hidden": ("#6E7787", "D", "--"),
}

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "figure.titlesize": 15,
    "figure.facecolor": COLOR_CANVAS,
    "savefig.facecolor": COLOR_CANVAS,
    "axes.facecolor": COLOR_PANEL,
    "axes.edgecolor": COLOR_DEEP_BLUE,
    "axes.labelcolor": COLOR_DEEP_BLUE,
    "axes.titlecolor": COLOR_DEEP_BLUE,
    "text.color": COLOR_DEEP_BLUE,
    "xtick.color": COLOR_DEEP_BLUE,
    "ytick.color": COLOR_DEEP_BLUE,
    "grid.color": COLOR_GRID,
    "axes.prop_cycle": cycler(color=[REPRESENTATION_STYLE[name][0] for name in REPRESENTATION_ORDER]),
    "legend.facecolor": COLOR_BG,
    "legend.edgecolor": COLOR_DEEP_BLUE,
    "legend.framealpha": 0.9,
})

TASK_DISPLAY = {
    "operation_reachability": "Operation reachability",
    "back_reachability": "BACK reachability",
    "trace_outcome_trajectory": "Final-answer signal before BACK",
    "terminal_outcome_controls": "Terminal final-answer signal",
    "trace_outcome_operation_decisions": "Final-answer signal at operation decisions",
    "trace_outcome_back_decisions": "Final-answer signal at BACK decisions",
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

REACHABILITY_SECTIONS = (
    ("operation_reachability", "Operation decisions"),
    ("back_reachability", "BACK decisions"),
)

FINAL_ANSWER_SECTIONS = (
    ("trace_outcome_operation_decisions", "Operation decisions"),
    ("trace_outcome_trajectory", "Before BACK"),
    ("trace_outcome_back_decisions", "BACK decisions"),
    ("terminal_outcome_controls", "End tokens"),
)

SEMANTIC_GROUPS = {
    "reachability": {
        "title": "Reachability",
        "sample_title": "Reachable and unreachable samples: Degree of separation at the decision boundary",
        "negative_label": "Unreachable",
        "positive_label": "Reachable",
        "sections": REACHABILITY_SECTIONS,
    },
    "final_answer_signal": {
        "title": "Final-answer signal",
        "sample_title": "Correct and incorrect samples: Degree of separation at the decision boundary",
        "negative_label": "Incorrect final answer",
        "positive_label": "Correct final answer",
        "sections": FINAL_ANSWER_SECTIONS,
    },
}

FINAL_ANSWER_TASKS = tuple(task for task, _ in FINAL_ANSWER_SECTIONS)

SUPPORTED_METRICS = (
    "test_auroc",
    "test_balanced_accuracy",
    "test_macro_auprc",
    "test_mcc",
)


# =============================================================================
# DATA AND HELPERS
# =============================================================================

@dataclass(frozen=True)
class PlotContext:
    results: dict[str, Any]
    metrics: list[dict[str, Any]]
    output_dir: Path
    metric: str
    dpi: int

    @property
    def task_specs(self) -> dict[str, Any]:
        return self.results["task_specs"]


@dataclass(frozen=True)
class TraceProgressData:
    labels: np.ndarray
    record_pos: np.ndarray
    backtrack_index: np.ndarray
    n_backtracks: np.ndarray
    trace_weights: np.ndarray
    selected_rows: dict[str, dict[str, Any]]
    scores: dict[str, np.ndarray]


def load_results(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    results = torch.load(path, map_location="cpu", weights_only=False)

    if not bool(results.get("complete", False)):
        raise RuntimeError("Probe results are not marked complete.")
    if results.get("run_tag") != RUN_TAG:
        raise RuntimeError(f"Results run tag {results.get('run_tag')!r} != {RUN_TAG!r}.")

    for field in ("task_specs", "prediction_paths", "source_probe_dataset_path"):
        if field not in results:
            raise KeyError(f"Probe results are missing {field!r}.")

    metrics = results.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise RuntimeError("Probe results contain no metrics rows.")

    required = {
        "probe_id",
        "task",
        "checkpoint",
        "representation_type",
        "location_name",
        "layer",
        "head",
        "val_auroc",
        "selected_threshold_logit",
        "test_auroc",
        "test_balanced_accuracy",
        "test_macro_auprc",
        "test_mcc",
    }

    for index, row in enumerate(metrics):
        missing = required - set(row)
        if missing:
            raise KeyError(f"Metric row {index} is missing fields {sorted(missing)}.")

    if len({str(row["probe_id"]) for row in metrics}) != len(metrics):
        raise RuntimeError("Probe metric rows contain duplicate probe IDs.")

    return results


def load_prediction_cache(ctx: PlotContext, task: str) -> dict[str, Any]:
    path = Path(ctx.results["prediction_paths"][task])
    if not path.is_file():
        raise FileNotFoundError(path)

    prediction = torch.load(path, map_location="cpu", weights_only=False)

    if not bool(prediction.get("complete", False)):
        raise RuntimeError(f"Prediction cache for {task!r} is incomplete.")
    if prediction.get("task") != task:
        raise RuntimeError(f"Prediction cache task {prediction.get('task')!r} does not match {task!r}.")

    return prediction


def select_rows(
    metrics: list[dict[str, Any]],
    *,
    task: str | None = None,
    checkpoint: str | None = None,
    representation_type: str | None = None,
) -> list[dict[str, Any]]:
    return [
        row
        for row in metrics
        if (task is None or row["task"] == task)
        and (checkpoint is None or row["checkpoint"] == checkpoint)
        and (
            representation_type is None
            or row["representation_type"] == representation_type
        )
    ]


def metric_label(metric: str) -> str:
    return {
        "test_auroc": "Test AUROC",
        "test_balanced_accuracy": "Test balanced accuracy",
        "test_macro_auprc": "Test macro AUPRC",
        "test_mcc": "Test MCC",
    }[metric]


def metric_bounds(metric: str) -> tuple[float, float]:
    return (-1.0, 1.0) if metric == "test_mcc" else (0.5, 1.0)


def metric_color_norm(metric: str) -> TwoSlopeNorm:
    if metric == "test_mcc":
        return TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    return TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def save_figure(fig: plt.Figure, path: Path, dpi: int, *, tight: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        output_path = path.with_suffix(f".{extension}")
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight" if tight else None)
    plt.close(fig)
    return path


def save_csv(path: Path, headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    return path


def checkpoint_names(ctx: PlotContext, task: str) -> list[str]:
    return list(ctx.task_specs[task]["checkpoints"])


def display_checkpoint(checkpoint: str) -> str:
    return CHECKPOINT_DISPLAY.get(checkpoint, checkpoint)


def attention_rows(ctx: PlotContext, task: str, checkpoint: str) -> list[dict[str, Any]]:
    rows = select_rows(ctx.metrics, task=task, checkpoint=checkpoint, representation_type="attention_head")
    if len(rows) != 32:
        raise RuntimeError(f"{task}/{checkpoint} has {len(rows)} attention probes, expected 32.")
    return rows


def validation_selected_probe(
    ctx: PlotContext,
    task: str,
    checkpoint: str,
    representation_type: str,
) -> dict[str, Any]:
    rows = select_rows(ctx.metrics, task=task, checkpoint=checkpoint, representation_type=representation_type)
    if not rows:
        raise RuntimeError(f"No probes found for {task}/{checkpoint}/{representation_type}.")
    return max(rows, key=lambda row: float(row["val_auroc"]))


def prediction_location_index(
    prediction: dict[str, Any],
    checkpoint_index: int,
    probe_id: str,
) -> int:
    probe_ids = list(prediction["probe_ids"][checkpoint_index])
    try:
        return probe_ids.index(probe_id)
    except ValueError as exc:
        raise RuntimeError(f"Probe {probe_id!r} is missing from the prediction cache.") from exc


def selected_probe_scores(
    ctx: PlotContext,
    prediction: dict[str, Any],
    task: str,
    checkpoint: str,
    representation_type: str,
    split: str = "test",
) -> tuple[dict[str, Any], np.ndarray]:
    checkpoint_index = list(prediction["checkpoint_names"]).index(checkpoint)
    row = validation_selected_probe(ctx, task, checkpoint, representation_type)
    location_index = prediction_location_index(prediction, checkpoint_index, str(row["probe_id"]))
    scores = prediction[split]["logits"][:, checkpoint_index, location_index].float().numpy()
    return row, scores


def equal_trace_weights(record_pos: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(record_pos, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]

def normalized_token_position_progress(activation_positions: dict[str, torch.Tensor]) -> np.ndarray:
    record_pos = activation_positions["record_pos"].long().numpy()
    token_idx = activation_positions["generated_token_idx"].long().numpy()

    order = np.lexsort((token_idx, record_pos))
    sorted_records = record_pos[order]
    starts = np.r_[0, np.flatnonzero(sorted_records[1:] != sorted_records[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]

    sorted_progress = np.empty(len(order), dtype=np.float64)
    for start, end in zip(starts, ends, strict=True):
        count = end - start
        sorted_progress[start:end] = 0.0 if count == 1 else np.arange(count) / (count - 1)

    progress = np.empty(len(order), dtype=np.float64)
    progress[order] = sorted_progress
    return progress

def safe_auroc(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray | None = None,
    min_class_count: int = 1,
) -> float:
    if labels.size == 0:
        return float("nan")
    counts = np.bincount(labels.astype(np.int64), minlength=2)
    if int(counts.min()) < min_class_count:
        return float("nan")
    return float(roc_auc_score(labels, scores, sample_weight=weights))


def display_sample(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if values.size <= SAMPLE_POINTS_PER_CLASS:
        return values
    return values[rng.choice(values.size, SAMPLE_POINTS_PER_CLASS, replace=False)]


def short_attention_name(location_name: str) -> str:
    return location_name.removeprefix("attention_").replace("_", "")


def add_value_annotations(ax: plt.Axes, matrix: np.ndarray, metric: str) -> None:
    norm = metric_color_norm(metric)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix[row, column])
            red, green, blue, _ = PROBE_SCORE_CMAP(norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue

            ax.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color=COLOR_DARK_TEXT if luminance >= 0.60 else "white",
            )


def finish_axis(
    ax: plt.Axes,
    metric: str,
    *,
    title: str,
    title_fontsize: float | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    if title_fontsize is None:
        ax.set_title(title, color=COLOR_FIGURE_TITLE)
    else:
        ax.set_title(title, color=COLOR_FIGURE_TITLE, fontsize=title_fontsize)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.set_ylim(*metric_bounds(metric))
    null = 0.0 if metric == "test_mcc" else 0.5
    ax.axhline(
        null,
        color=COLOR_REFLINE,
        linestyle="--",
        linewidth=1,
        zorder=0,
        label="Null" if metric == "test_mcc" else "Null / chance",
    )


def plot_representation(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    representation_type: str,
    label: str,
    yerr: np.ndarray | None = None,
) -> None:
    color, marker, linestyle = REPRESENTATION_STYLE[representation_type]
    kwargs = {
        "color": color,
        "marker": marker,
        "linestyle": linestyle,
        "linewidth": 2,
        "markersize": 6,
        "markeredgecolor": COLOR_BG,
        "markeredgewidth": 0.8,
        "label": label,
    }

    if yerr is None:
        ax.plot(x, y, **kwargs)
    else:
        ax.errorbar(x, y, yerr=yerr, capsize=3, **kwargs)


# =============================================================================
# LAYER / HEAD PANELS
# =============================================================================

def attention_matrix(ctx: PlotContext, task: str, checkpoint: str) -> np.ndarray:
    matrix = np.full((4, 8), np.nan, dtype=np.float64)

    for row in attention_rows(ctx, task, checkpoint):
        layer = int(row["layer"])
        head = int(row["head"])

        if not (0 <= layer < 4 and 0 <= head < 8):
            raise RuntimeError(f"Invalid attention location {layer=}, {head=}.")

        matrix[layer, head] = float(row[ctx.metric])

    if not bool(np.isfinite(matrix).all()):
        raise RuntimeError(f"Incomplete attention matrix for {task}/{checkpoint}.")

    return matrix


def plot_layer_head_panels(ctx: PlotContext) -> list[Path]:
    """Appendix-style layer-by-head heatmaps, one multi-panel figure per task."""

    outputs: list[Path] = []
    norm = metric_color_norm(ctx.metric)

    for task in tqdm(ctx.task_specs, desc="  Rendering layer/head panels", unit="task", dynamic_ncols=True):
        checkpoints = checkpoint_names(ctx, task)
        ncols = 2 if len(checkpoints) > 1 else 1
        nrows = math.ceil(len(checkpoints) / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 3.2 * nrows), squeeze=False, constrained_layout=True)
        flat_axes = axes.reshape(-1)
        image = None

        for panel_index, checkpoint in enumerate(checkpoints):
            ax = flat_axes[panel_index]
            matrix = attention_matrix(ctx, task, checkpoint)

            image = ax.imshow(matrix, aspect=HEATMAP_CELL_ASPECT, norm=norm, cmap=PROBE_SCORE_CMAP)

            ax.set_xticks(np.arange(8), [f"H{head}" for head in range(8)])
            ax.set_yticks(np.arange(4), [f"L{layer}" for layer in range(4)])
            ax.set(xlabel="Attention head", ylabel="Layer")
            ax.set_title(display_checkpoint(checkpoint))
            add_value_annotations(ax, matrix, ctx.metric)

        for ax in flat_axes[len(checkpoints):]:
            ax.axis("off")

        if image is None:
            raise RuntimeError(f"Task {task!r} has no checkpoints.")

        colorbar = fig.colorbar(image, ax=list(flat_axes[:len(checkpoints)]), fraction=0.025, pad=0.02)
        colorbar.set_label(f"{metric_label(ctx.metric)}\n" "red < null   |   white = null   |   blue > null")
        colorbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0] if ctx.metric == "test_mcc" else [0.0, 0.25, 0.5, 0.75, 1.0])

        fig.suptitle(TASK_DISPLAY.get(task, task), color=COLOR_FIGURE_TITLE)
        outputs.append(save_figure(fig, ctx.output_dir / "layer_head_panels" / f"{safe_name(task)}.png", ctx.dpi))

    return outputs


# =============================================================================
# ATTENTION-HEAD PERFORMANCE DISTRIBUTIONS
# =============================================================================

def attention_distribution_legend() -> list[Any]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=6,
            markerfacecolor=COLOR_PASTEL_BLUE,
            markeredgecolor=COLOR_BG,
            label="Individual attention head",
        ),
        Patch(facecolor=COLOR_PASTEL_BLUE, edgecolor=COLOR_DEEP_BLUE, alpha=0.45, label="Box = IQR; whiskers = 1.5×IQR"),
        Line2D([0], [0], color=COLOR_DEEP_BLUE, linewidth=2, label="Median"),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markersize=7,
            markerfacecolor=COLOR_DARK_TEXT,
            markeredgecolor=COLOR_BG,
            label="Arithmetic mean",
        ),
        Line2D([0], [0], color=COLOR_REFLINE, linestyle="--", linewidth=1, label="Null / chance"),
    ]


def semantic_attention_data(
    ctx: PlotContext,
    sections: tuple[tuple[str, str], ...],
) -> tuple[list[str], list[np.ndarray], list[tuple[int, int, str]]]:
    labels: list[str] = []
    values: list[np.ndarray] = []
    section_ranges: list[tuple[int, int, str]] = []

    for task, section_name in sections:
        start = len(labels)

        for checkpoint in checkpoint_names(ctx, task):
            labels.append(display_checkpoint(checkpoint))
            values.append(np.asarray([float(row[ctx.metric]) for row in attention_rows(ctx, task, checkpoint)], dtype=np.float64))

        section_ranges.append((start, len(labels) - 1, section_name))

    return labels, values, section_ranges


def plot_attention_position_distributions(ctx: PlotContext) -> list[Path]:
    """Exactly two figures: one reachability figure and one final-answer figure."""

    outputs: list[Path] = []
    rng = np.random.default_rng(42)

    for group_name, group in tqdm(SEMANTIC_GROUPS.items(), desc="  Rendering position distributions", unit="group", dynamic_ncols=True):
        labels, values_by_position, section_ranges = semantic_attention_data(ctx, group["sections"])
        positions = np.arange(1, len(labels) + 1)

        fig, ax = plt.subplots(figsize=(10.0, 4.38))
        boxplot = ax.boxplot(values_by_position, positions=positions, widths=0.55, patch_artist=True, showfliers=False, manage_ticks=False)

        for box in boxplot["boxes"]:
            box.set_facecolor(COLOR_PASTEL_BLUE)
            box.set_edgecolor(COLOR_DEEP_BLUE)
            box.set_alpha(0.45)

        for median in boxplot["medians"]:
            median.set_color(COLOR_DEEP_BLUE)
            median.set_linewidth(2)

        for item in (*boxplot["whiskers"], *boxplot["caps"]):
            item.set_color(COLOR_DEEP_BLUE)

        for position, values in zip(positions, values_by_position, strict=True):
            ax.scatter(
                position + rng.uniform(-0.13, 0.13, values.size),
                values,
                s=18,
                alpha=0.55,
                color=COLOR_PASTEL_BLUE,
                edgecolors=COLOR_BG,
                linewidths=0.4,
            )
            ax.scatter(
                position,
                float(values.mean()),
                marker="D",
                s=58,
                color=COLOR_DARK_TEXT,
                edgecolors=COLOR_BG,
                linewidths=0.8,
                zorder=4,
            )

        for section_index, (start, end, section_name) in enumerate(section_ranges):
            center = 0.5 * ((start + 1) + (end + 1))
            ax.text(
                center,
                1.015,
                section_name,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=10,
                color=COLOR_DARK_TEXT,
                clip_on=False,
            )
            if section_index < len(section_ranges) - 1:
                ax.axvline(end + 1.5, color=COLOR_REFLINE, linewidth=1, alpha=0.55)

        ax.set_xticks(positions, labels, rotation=27, ha="right")
        finish_axis(
            ax,
            ctx.metric,
            title="",
            xlabel="Token position",
            ylabel=metric_label(ctx.metric),
        )

        figure_title = {
            "final_answer_signal": "Probe Final-answer performance across token positions",
            "reachability": "Probe Reachability performance across token positions",
        }[group_name]
        title_fontsize = 12.5
        fig.suptitle(
            figure_title,
            x=0.53,
            y=0.98,
            fontsize=title_fontsize,
            color=COLOR_FIGURE_TITLE,
        )

        ax.tick_params(axis="both", labelsize=9.5)
        ax.set_xlabel(
            "Token position",
            fontsize=11.2,
            color=COLOR_DEEP_BLUE,
            labelpad=7,
        )
        ax.set_ylabel(
            metric_label(ctx.metric),
            fontsize=10,
            color=COLOR_DEEP_BLUE,
        )

        ax.grid(axis="y", alpha=0.2)
        ax.legend(
            handles=attention_distribution_legend()[:-1],
            loc="upper left",
            fontsize=9,
        )
        fig.subplots_adjust(
            left=0.10,
            right=0.98,
            bottom=0.32,
            top=0.86,
        )

        outputs.append(save_figure(fig, ctx.output_dir / "attention_position_distributions" / f"{group_name}.png", ctx.dpi))

    return outputs


# =============================================================================
# HELD-OUT INDIVIDUAL-SAMPLE DISTRIBUTIONS
# =============================================================================

def style_sample_boxplot(
    boxplot: dict[str, list[Any]],
    facecolors: list[str],
    edgecolors: list[str],
) -> None:
    for index, box in enumerate(boxplot["boxes"]):
        box.set_facecolor(facecolors[index])
        box.set_edgecolor(edgecolors[index])
        box.set_alpha(0.55)

    for index, median in enumerate(boxplot["medians"]):
        median.set_color(edgecolors[index])
        median.set_linewidth(1.8)

    for key in ("whiskers", "caps"):
        for index, item in enumerate(boxplot[key]):
            item.set_color(edgecolors[index // 2])


def draw_attention_sample_panel(
    ax: plt.Axes,
    ctx: PlotContext,
    prediction: dict[str, Any],
    task: str,
    checkpoint: str,
    rng: np.random.Generator,
) -> None:
    checkpoint_index = list(prediction["checkpoint_names"]).index(checkpoint)
    top_rows = sorted(attention_rows(ctx, task, checkpoint), key=lambda row: float(row["val_auroc"]), reverse=True)[:TOP_ATTENTION_HEADS]

    validation_logits = prediction["validation"]["logits"].float()
    test_logits = prediction["test"]["logits"].float()
    labels = prediction["test"]["label"].bool().numpy()
    thresholds = prediction["thresholds_logit"].float()

    centers = np.arange(1, len(top_rows) + 1, dtype=np.float64)
    box_values: list[np.ndarray] = []
    box_positions: list[float] = []
    facecolors: list[str] = []
    edgecolors: list[str] = []

    for head_index, row in enumerate(top_rows):
        location_index = prediction_location_index(prediction, checkpoint_index, str(row["probe_id"]))

        validation_scores = validation_logits[:, checkpoint_index, location_index].numpy()
        test_scores = test_logits[:, checkpoint_index, location_index].numpy()
        threshold = float(thresholds[checkpoint_index, location_index])
        scale = max(float(validation_scores.std()), 1e-6)

        # Zero is the validation-selected decision threshold.
        margins = (test_scores - threshold) / scale
        negative = margins[~labels]
        positive = margins[labels]

        negative_x = centers[head_index] - 0.18
        positive_x = centers[head_index] + 0.18

        box_values.extend((negative, positive))
        box_positions.extend((negative_x, positive_x))
        facecolors.extend((COLOR_SOFT_RED, COLOR_PASTEL_BLUE))
        edgecolors.extend((COLOR_DEEP_RED, COLOR_DEEP_BLUE))

        for x_value, values, color in (
            (negative_x, negative, COLOR_RED),
            (positive_x, positive, COLOR_DEEP_BLUE),
        ):
            shown = display_sample(values, rng)
            ax.scatter(
                x_value + rng.uniform(-0.06, 0.06, shown.size),
                shown,
                s=8,
                alpha=0.18,
                color=color,
                edgecolors="none",
                rasterized=True,
            )

    boxplot = ax.boxplot(box_values, positions=box_positions, widths=0.26, patch_artist=True, showfliers=False, manage_ticks=False)
    style_sample_boxplot(boxplot, facecolors, edgecolors)

    ax.axhline(0.0, color=COLOR_REFLINE, linestyle="--", linewidth=1, zorder=0)
    ax.set_xticks(
        centers,
        [
            f"{short_attention_name(str(row['location_name']))}\n"
            f"val {float(row['val_auroc']):.3f}"
            for row in top_rows
        ],
    )
    ax.set_title(display_checkpoint(checkpoint))
    ax.set_xlabel("Top attention heads by validation AUROC")
    ax.set_ylabel("Distance to boundary")
    ax.tick_params(axis="x", labelsize=12)
    ax.grid(axis="y", alpha=0.2)


def sample_distribution_legend(negative_label: str, positive_label: str) -> list[Any]:
    return [
        Line2D([0], [0], marker="o", markersize=9, linestyle="none", markerfacecolor=COLOR_RED, markeredgecolor="none", label=negative_label),
        Line2D([0], [0], marker="o", markersize=9, linestyle="none", markerfacecolor=COLOR_DEEP_BLUE, markeredgecolor="none", label=positive_label),
        Patch(facecolor=COLOR_BG, edgecolor=COLOR_DARK_TEXT, label="Box = IQR; line = median"),
        Line2D([0], [0], color=COLOR_REFLINE, linestyle="--", linewidth=2, label="Validation-selected threshold"),
    ]


def plot_attention_sample_distributions(ctx: PlotContext) -> list[Path]:
    """
    Held-out example-level score distributions for the top 4 attention heads
    selected independently at each checkpoint by validation AUROC.
    """

    outputs: list[Path] = []
    rng = np.random.default_rng(42)

    for group_name, group in tqdm(SEMANTIC_GROUPS.items(), desc="  Rendering sample distributions", unit="group", dynamic_ncols=True):
        predictions = {
            task: load_prediction_cache(ctx, task)
            for task, _ in group["sections"]
        }
        jobs = [
            (task, section_name, checkpoint)
            for task, section_name in group["sections"]
            for checkpoint in checkpoint_names(ctx, task)
        ]

        ncols = 2
        nrows = math.ceil(len(jobs) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, 3.8 * nrows), squeeze=False)
        flat_axes = axes.reshape(-1)

        for panel_index, (task, section_name, checkpoint) in enumerate(jobs):
            ax = flat_axes[panel_index]
            draw_attention_sample_panel(ax, ctx, predictions[task], task, checkpoint, rng)
            if panel_index % ncols != 0:
                ax.set_ylabel("")

        for ax in flat_axes[len(jobs):]:
            ax.axis("off")

        fig.suptitle(group["sample_title"], fontsize=16, color=COLOR_FIGURE_TITLE)
        fig.legend(
            handles=sample_distribution_legend(str(group["negative_label"]), str(group["positive_label"])),
            loc="lower center",
            ncol=4,
            bbox_to_anchor=(0.5, 0.01),
            fontsize=15,
            markerscale=1.6,
            handlelength=2.4,
            handleheight=1.2,
        )
        fig.text(
            0.5,
            0.045,
            f"Dots show up to {SAMPLE_POINTS_PER_CLASS} held-out examples per class and head",
            ha="center",
            fontsize=9,
            color=COLOR_DARK_TEXT,
        )
        fig.tight_layout(rect=(0.0, 0.065, 1.0, 0.97))

        if len(jobs) % ncols:
            last_ax = flat_axes[len(jobs) - 1]
            position = last_ax.get_position()
            last_ax.set_position([
                0.5 - position.width / 2,
                position.y0,
                position.width,
                position.height,
            ])

        outputs.append(save_figure(fig, ctx.output_dir / "attention_sample_distributions" / f"{group_name}.png", ctx.dpi))

    return outputs


# =============================================================================
# PRE-BACK TRACE PROGRESS
# =============================================================================

def load_trace_progress_data(ctx: PlotContext) -> TraceProgressData:
    task = "trace_outcome_trajectory"
    checkpoint = "pre_back"

    prediction = load_prediction_cache(ctx, task)

    if not LABEL_PATH.is_file():
        raise FileNotFoundError(LABEL_PATH)

    label_artifact = torch.load(LABEL_PATH, map_location="cpu", weights_only=False)
    if not bool(label_artifact.get("complete", False)):
        raise RuntimeError("Reachability-label artifact is incomplete.")

    checkpoint_index = list(prediction["checkpoint_names"]).index(checkpoint)
    test = prediction["test"]

    source_rows = test["source_row"].long()
    record_pos_tensor = test["record_pos"].long()
    labels = test["label"].to(torch.int8).numpy()
    record_pos = record_pos_tensor.numpy()

    backtrack_index = (
        label_artifact["back_transitions"]["backtrack_idx"][source_rows].long().numpy()
    )
    n_backtracks = (
        label_artifact["generated_traces"]["n_backtracks"][record_pos_tensor].long().numpy()
    )

    if np.any(n_backtracks <= 0):
        raise RuntimeError("A trajectory example comes from a zero-BACK trace.")
    if np.any(backtrack_index < 0) or np.any(backtrack_index >= n_backtracks):
        raise RuntimeError("A trajectory BACK index is outside its trace.")

    selected_rows: dict[str, dict[str, Any]] = {}
    scores: dict[str, np.ndarray] = {}

    for representation_type in REPRESENTATION_ORDER:
        selected_rows[representation_type], scores[representation_type] = selected_probe_scores(ctx, prediction, task, checkpoint, representation_type)

    return TraceProgressData(
        labels=labels,
        record_pos=record_pos,
        backtrack_index=backtrack_index,
        n_backtracks=n_backtracks,
        trace_weights=equal_trace_weights(record_pos),
        selected_rows=selected_rows,
        scores=scores,
    )


def trace_progress_diagnostics(
    output_root: Path,
    data: TraceProgressData,
    bin_index: np.ndarray,
    edges: np.ndarray,
) -> list[Path]:
    outputs: list[Path] = []
    back_counts = sorted(int(value) for value in np.unique(data.n_backtracks))

    composition_rows: list[tuple[Any, ...]] = []
    for current_bin in range(TRACE_PROGRESS_BINS):
        for total_backtracks in back_counts:
            mask = (
                (bin_index == current_bin)
                & (data.n_backtracks == total_backtracks)
            )
            if not bool(mask.any()):
                continue

            labels = data.labels[mask]
            composition_rows.append(
                (
                    float(edges[current_bin]),
                    float(edges[current_bin + 1]),
                    total_backtracks,
                    int(mask.sum()),
                    int(np.unique(data.record_pos[mask]).size),
                    int((labels == 1).sum()),
                    int((labels == 0).sum()),
                )
            )

    outputs.append(
        save_csv(
            output_root / "normalized_bin_composition.csv",
            (
                "bin_start",
                "bin_end",
                "n_backtracks",
                "test_examples",
                "test_traces",
                "positive_examples",
                "negative_examples",
            ),
            composition_rows,
        )
    )

    print("\n  Exact BACK-count distribution in trajectory test examples:")
    count_rows: list[tuple[Any, ...]] = []

    for total_backtracks in back_counts:
        mask = data.n_backtracks == total_backtracks
        trace_ids = np.unique(data.record_pos[mask])
        example_records = ";".join(str(int(value)) for value in trace_ids[:10])

        print(f"    {total_backtracks} BACKs: {trace_ids.size} test traces")
        if total_backtracks > 10:
            print(f"      example record_pos values: {example_records}")

        count_rows.append((total_backtracks, int(trace_ids.size), int(mask.sum()), example_records))

    outputs.append(
        save_csv(
            output_root / "backtrack_count_distribution.csv",
            (
                "n_backtracks",
                "test_traces",
                "test_examples",
                "example_record_pos",
            ),
            count_rows,
        )
    )

    return outputs


def plot_trace_progress(ctx: PlotContext) -> list[Path]:
    """
    Final-answer correctness at repeated pre-BACK positions.

    One fixed validation-selected pre-BACK probe is used per representation.
    AUROC is then recomputed separately at different BACK stages.
    """

    if ctx.metric != "test_auroc":
        raise RuntimeError("trace_progress currently supports --metric test_auroc only.")

    data = load_trace_progress_data(ctx)
    output_root = ctx.output_dir / "trace_progress"
    outputs: list[Path] = []

    progress = (data.backtrack_index + 1.0) / data.n_backtracks
    edges = np.linspace(0.0, 1.0, TRACE_PROGRESS_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_index = np.digitize(progress, edges[1:-1], right=True)

    normalized_rows: list[tuple[Any, ...]] = []
    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    for representation_type in REPRESENTATION_ORDER:
        aurocs = np.full(TRACE_PROGRESS_BINS, np.nan, dtype=np.float64)
        selected = data.selected_rows[representation_type]
        scores = data.scores[representation_type]

        for current_bin in range(TRACE_PROGRESS_BINS):
            mask = bin_index == current_bin
            bin_labels = data.labels[mask]
            aurocs[current_bin] = safe_auroc(bin_labels, scores[mask], data.trace_weights[mask])

            normalized_rows.append(
                (
                    representation_type,
                    str(selected["location_name"]),
                    float(edges[current_bin]),
                    float(edges[current_bin + 1]),
                    float(centers[current_bin]),
                    int(mask.sum()),
                    int((bin_labels == 1).sum()),
                    int((bin_labels == 0).sum()),
                    float(aurocs[current_bin]),
                )
            )

        finite = np.isfinite(aurocs)
        plot_representation(
            ax,
            centers[finite],
            aurocs[finite],
            representation_type,
            f"{REPRESENTATION_DISPLAY[representation_type]} ({selected['location_name']})",
        )

    finish_axis(
        ax,
        "test_auroc",
        title=(
            "Final-answer signal over normalized trace progress\n"
            "measured at pre-BACK events across all trace lengths"
        ),
        xlabel="Normalized BACK-event progress",
        ylabel="Trace-weighted test AUROC",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 5))
    ax.legend()
    ax.grid(alpha=0.25)

    outputs.extend(
        [
            save_figure(fig, output_root / "final_answer_signal_normalized_all.png", ctx.dpi),
            save_csv(
                output_root / "final_answer_signal_normalized_all.csv",
                (
                    "representation_type",
                    "validation_selected_location",
                    "bin_start",
                    "bin_end",
                    "bin_center",
                    "test_examples",
                    "positive_examples",
                    "negative_examples",
                    "trace_weighted_test_auroc",
                ),
                normalized_rows,
            ),
        ]
    )

    outputs.extend(trace_progress_diagnostics(output_root, data, bin_index, edges))

    exact_root = output_root / "by_backtrack_count"
    exact_root.mkdir(parents=True, exist_ok=True)

    for pattern in ("backtracks_*.png", "backtracks_*.pdf", "backtracks_*.csv"):
        for stale_path in exact_root.glob(pattern):
            stale_path.unlink()

    for total_backtracks in sorted(int(value) for value in np.unique(data.n_backtracks)):
        group_mask = data.n_backtracks == total_backtracks
        trace_count = int(np.unique(data.record_pos[group_mask]).size)
        event_numbers = np.arange(1, total_backtracks + 1)

        fig, ax = plt.subplots(figsize=(max(8.0, 5.5 + 0.65 * total_backtracks), 5.8))
        exact_rows: list[tuple[Any, ...]] = []

        for representation_type in REPRESENTATION_ORDER:
            aurocs = np.full(total_backtracks, np.nan, dtype=np.float64)
            selected = data.selected_rows[representation_type]
            scores = data.scores[representation_type]

            for event_index in range(total_backtracks):
                mask = group_mask & (data.backtrack_index == event_index)
                event_labels = data.labels[mask]
                aurocs[event_index] = safe_auroc(event_labels, scores[mask])

                exact_rows.append(
                    (
                        total_backtracks,
                        event_index + 1,
                        representation_type,
                        str(selected["location_name"]),
                        int(mask.sum()),
                        int((event_labels == 1).sum()),
                        int((event_labels == 0).sum()),
                        float(aurocs[event_index]),
                    )
                )

            finite = np.isfinite(aurocs)
            plot_representation(
                ax,
                event_numbers[finite],
                aurocs[finite],
                representation_type,
                f"{REPRESENTATION_DISPLAY[representation_type]} ({selected['location_name']})",
            )

        finish_axis(
            ax,
            "test_auroc",
            title=(
                f"Final-answer signal for traces with {total_backtracks} BACKs\n"
                f"measured immediately before each BACK event "
                f"(n={trace_count} test traces)"
            ),
            xlabel="BACK event",
            ylabel="Test AUROC",
        )
        ax.set_xlim(0.5, total_backtracks + 0.5)
        ax.set_xticks(event_numbers)
        ax.legend()
        ax.grid(alpha=0.25)

        outputs.extend(
            [
                save_figure(fig, exact_root / f"backtracks_{total_backtracks}.png", ctx.dpi),
                save_csv(
                    exact_root / f"backtracks_{total_backtracks}.csv",
                    (
                        "n_backtracks",
                        "back_event",
                        "representation_type",
                        "validation_selected_location",
                        "test_examples",
                        "positive_examples",
                        "negative_examples",
                        "test_auroc",
                    ),
                    exact_rows,
                ),
            ]
        )

    return outputs


# =============================================================================
# ALL-DECISION-POINT TRACE PROGRESS
# =============================================================================

def plot_decision_point_progress(ctx: PlotContext) -> list[Path]:
    outputs = _plot_decision_point_progress(ctx)
    outputs.extend(_plot_decision_point_progress(ctx, max_backtracks=8))
    return outputs

def _plot_decision_point_progress(ctx: PlotContext, max_backtracks: int | None = None) -> list[Path]:
    """
    Final-answer correctness across probed token positions.

    Each token-position type uses its own validation-selected probe. AUROC is computed
    separately within token-position type and progress bin before AUROCs are
    summarized, so raw logits from different probes are never pooled together.
    """

    if ctx.metric != "test_auroc":
        raise RuntimeError("decision_point_progress currently supports --metric test_auroc only.")

    dataset_path = Path(ctx.results["source_probe_dataset_path"])
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if not LABEL_PATH.is_file():
        raise FileNotFoundError(LABEL_PATH)

    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    label_artifact = torch.load(LABEL_PATH, map_location="cpu", weights_only=False)

    if not bool(dataset.get("complete", False)):
        raise RuntimeError("Probe dataset metadata is incomplete.")
    if not bool(label_artifact.get("complete", False)):
        raise RuntimeError("Reachability-label artifact is incomplete.")

    activation_positions = dataset["activation_positions"]
    position_record_pos = activation_positions["record_pos"].long()
    generated_backtracks = label_artifact["generated_traces"]["n_backtracks"].long()
    token_progress = normalized_token_position_progress(activation_positions)

    terminal_prediction = load_prediction_cache(ctx, "terminal_outcome_controls")
    terminal_index = list(terminal_prediction["checkpoint_names"]).index("expr_final_parenthesis")
    terminal_test = terminal_prediction["test"]
    terminal_record_pos = terminal_test["record_pos"].long()
    terminal_activation = terminal_test["activation_index"][:, terminal_index].long()

    if not torch.equal(position_record_pos[terminal_activation], terminal_record_pos):
        raise RuntimeError("Final-expression activation-to-trace join failed.")

    terminal_progress = token_progress[terminal_activation.numpy()]
    if not np.allclose(terminal_progress, 1.0):
        bad = int((~np.isclose(terminal_progress, 1.0)).sum())
        raise RuntimeError(
            f"{bad} traces do not place the final-expression token position at progress 1.0."
        )

    edges = np.linspace(0.0, 1.0, DECISION_PROGRESS_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    plot_x = np.concatenate(([0.0], centers, [1.0]))
    progress_starts = np.concatenate(([0.0], edges[:-1], [1.0]))
    progress_ends = np.concatenate(([0.0], edges[1:], [1.0]))
    n_progress_groups = len(plot_x)

    values_by_representation = {
        representation_type: [[] for _ in range(n_progress_groups)]
        for representation_type in REPRESENTATION_ORDER
    }
    checkpoints_by_representation = {
        representation_type: [[] for _ in range(n_progress_groups)]
        for representation_type in REPRESENTATION_ORDER
    }
    detailed_rows: list[tuple[Any, ...]] = []

    for task in tqdm(FINAL_ANSWER_TASKS, desc="  Computing probed decision-point AUROCs", unit="task", leave=False, dynamic_ncols=True):
        prediction = load_prediction_cache(ctx, task)
        test = prediction["test"]

        record_pos_tensor = test["record_pos"].long()
        keep = torch.ones(record_pos_tensor.shape, dtype=torch.bool)
        if max_backtracks is not None:
            keep = generated_backtracks[record_pos_tensor] <= max_backtracks

        # This progress analysis is defined only for traces whose endpoint
        # is the probed final-expression position at progress 1.0.
        keep &= torch.isin(record_pos_tensor, terminal_record_pos)

        record_pos_tensor = record_pos_tensor[keep]
        record_pos = record_pos_tensor.numpy()
 
        labels = test["label"][keep].to(torch.int8).numpy()
        logits = test["logits"][keep].float()
        activation_index = test["activation_index"][keep].long()
        trace_weights = equal_trace_weights(record_pos)

        for checkpoint_index, checkpoint in enumerate(prediction["checkpoint_names"]):
            checkpoint_activation = activation_index[:, checkpoint_index]

            if not torch.equal(position_record_pos[checkpoint_activation], record_pos_tensor):
                raise RuntimeError(f"Activation-to-trace join failed for {task}/{checkpoint}.")
            progress = token_progress[checkpoint_activation.numpy()]
            progress_group = np.digitize(progress, edges[1:-1], right=True) + 1
            progress_group[np.isclose(progress, 0.0)] = 0
            progress_group[np.isclose(progress, 1.0)] = n_progress_groups - 1

            for representation_type in REPRESENTATION_ORDER:
                selected = validation_selected_probe(ctx, task, checkpoint, representation_type)
                location_index = prediction_location_index(prediction, checkpoint_index, str(selected["probe_id"]))
                scores = logits[:, checkpoint_index, location_index].numpy()

                for current_bin in range(n_progress_groups):
                    mask = progress_group == current_bin
                    bin_labels = labels[mask]
                    n_positive = int((bin_labels == 1).sum())
                    n_negative = int((bin_labels == 0).sum())

                    auroc = safe_auroc(bin_labels, scores[mask], trace_weights[mask], min_class_count=MIN_AUROC_CLASS_COUNT)

                    if math.isfinite(auroc):
                        values_by_representation[representation_type][current_bin].append(auroc)
                        checkpoints_by_representation[representation_type][current_bin].append(checkpoint)

                    detailed_rows.append(
                        (
                            task,
                            checkpoint,
                            representation_type,
                            str(selected["location_name"]),
                            float(progress_starts[current_bin]),
                            float(progress_ends[current_bin]),
                            float(plot_x[current_bin]),
                            int(mask.sum()),
                            int(np.unique(record_pos[mask]).size),
                            n_positive,
                            n_negative,
                            auroc,
                        )
                    )

    summary_rows: list[tuple[Any, ...]] = []
    fig, ax = plt.subplots(figsize=(10.0, 6.0))

    for representation_type in REPRESENTATION_ORDER:
        means = np.full(n_progress_groups, np.nan, dtype=np.float64)
        standard_deviations = np.full(n_progress_groups, np.nan, dtype=np.float64)

        for current_bin in range(n_progress_groups):
            values = np.asarray(values_by_representation[representation_type][current_bin], dtype=np.float64)
            checkpoints = checkpoints_by_representation[representation_type][current_bin]

            if values.size:
                means[current_bin] = float(values.mean())
                standard_deviations[current_bin] = float(values.std())

            summary_rows.append(
                (
                    representation_type,
                    float(progress_starts[current_bin]),
                    float(progress_ends[current_bin]),
                    float(plot_x[current_bin]),
                    int(values.size),
                    ";".join(checkpoints),
                    float(means[current_bin]),
                    float(standard_deviations[current_bin]),
                    float(values.min()) if values.size else float("nan"),
                    float(values.max()) if values.size else float("nan"),
                )
            )

        finite = np.isfinite(means)
        plot_representation(
            ax,
            plot_x[finite],
            means[finite],
            representation_type,
            REPRESENTATION_DISPLAY[representation_type],
            yerr=standard_deviations[finite],
        )

    restriction = "" if max_backtracks is None else f" with at most {max_backtracks} BACKs"
    suffix = "" if max_backtracks is None else f"_max_{max_backtracks}_backtracks"

    finish_axis(
        ax,
        "test_auroc",
        title=f"Final-answer decodability across trace progress{restriction}",
        title_fontsize=12,
        xlabel="Normalized decision point progress",
        ylabel="Test AUROC",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.grid(alpha=0.25)
    ax.legend()

    fig.text(
        0.5,
        0.015,
        (
            "Error bars show ±1 SD across token-position-specific AUROCs. "
            f"Each token position/bin requires at least {MIN_AUROC_CLASS_COUNT} "
            "examples from each class."
        ),
        ha="center",
        fontsize=9,
        color=COLOR_DARK_TEXT,
    )
    fig.subplots_adjust(bottom=0.15)

    output_root = ctx.output_dir / "decision_point_progress"

    return [
        save_figure(fig, output_root / f"final_answer_signal{suffix}.png", ctx.dpi),
        save_csv(
            output_root / f"checkpoint_aurocs{suffix}.csv",
            (
                "task",
                "checkpoint",
                "representation_type",
                "validation_selected_location",
                "bin_start",
                "bin_end",
                "bin_center",
                "test_examples",
                "test_traces",
                "positive_examples",
                "negative_examples",
                "trace_weighted_test_auroc",
            ),
            detailed_rows,
        ),
        save_csv(
            output_root / f"summary{suffix}.csv",
            (
                "representation_type",
                "bin_start",
                "bin_end",
                "bin_center",
                "checkpoint_aurocs",
                "contributing_checkpoints",
                "mean_test_auroc",
                "sd_test_auroc",
                "min_test_auroc",
                "max_test_auroc",
            ),
            summary_rows,
        ),
    ]

# =============================================================================
# COMPACT PAPER FIGURES
# =============================================================================
def load_decision_progress_summary(path: Path) -> dict[str, list[dict[str, float]]]:
    """Load one summary CSV written by plot_decision_point_progress."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing decision-progress summary {path}. "
            "Run --plotlist decision_point_progress before paper_compact_figures."
        )

    grouped: dict[str, list[dict[str, float]]] = {
        representation_type: [] for representation_type in REPRESENTATION_ORDER
    }
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            representation_type = str(row["representation_type"])
            if representation_type not in grouped:
                raise RuntimeError(
                    f"Unknown representation type {representation_type!r} in {path}."
                )
            grouped[representation_type].append(
                {
                    "bin_center": float(row["bin_center"]),
                    "mean_test_auroc": float(row["mean_test_auroc"]),
                    "sd_test_auroc": float(row["sd_test_auroc"]),
                }
            )

    for representation_type, rows in grouped.items():
        rows.sort(key=lambda row: row["bin_center"])
        if not rows:
            raise RuntimeError(
                f"No {representation_type} rows were found in {path}."
            )

    return grouped


def plot_compact_decision_progress_figure(
    ctx: PlotContext,
    *,
    max_backtracks: int | None,
) -> Path:
    """Render a smaller paper variant without changing the existing figure."""

    suffix = "" if max_backtracks is None else f"_max_{max_backtracks}_backtracks"
    summary_path = (
        ctx.output_dir
        / "decision_point_progress"
        / f"summary{suffix}.csv"
    )
    grouped = load_decision_progress_summary(summary_path)

    # This is exactly 75% of the original 10 x 6 dimensions, preserving
    # its aspect ratio while reducing its area to 56.25%.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for representation_type in REPRESENTATION_ORDER:
        rows = grouped[representation_type]
        x = np.asarray(
            [row["bin_center"] for row in rows],
            dtype=np.float64,
        )
        y = np.asarray(
            [row["mean_test_auroc"] for row in rows],
            dtype=np.float64,
        )
        yerr = np.asarray(
            [row["sd_test_auroc"] for row in rows],
            dtype=np.float64,
        )
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)

        plot_representation(
            ax,
            x[finite],
            y[finite],
            representation_type,
            REPRESENTATION_DISPLAY[representation_type],
            yerr=yerr[finite],
        )

    title = (
        "Final-answer decodability across trace progress"
        if max_backtracks is None
        else f"Final-answer decodability with at most {max_backtracks} BACKs"
    )
    title_y = (
        COMPACT_ALL_TRACES_TITLE_Y
        if max_backtracks is None
        else COMPACT_MAX_8_BACKTRACKS_TITLE_Y
    )
    finish_axis(
        ax,
        "test_auroc",
        title="",
        title_fontsize=17,
        xlabel="Normalised trace progress $\\rightarrow$",
        ylabel="Test AUROC",
    )
    ax.set_title(
        title,
        fontsize=17,
        y=title_y,
        pad=0,
        color=COLOR_FIGURE_TITLE,
    )
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax.tick_params(axis="both", labelsize=13)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.17, top=0.89)

    output_root = ctx.output_dir / "paper_compact_figures"
    return save_figure(
        fig,
        output_root
        / f"final_answer_decodability_trace_progress{suffix}.png",
        ctx.dpi,
        tight=True,
    )


def draw_compact_attention_position_panel(
    ax: plt.Axes,
    ctx: PlotContext,
    *,
    group_name: str,
    panel_title: str,
    rng: np.random.Generator,
) -> None:
    """Draw one attention-head boxplot panel for the combined paper figure."""

    group = SEMANTIC_GROUPS[group_name]
    labels, values_by_position, section_ranges = semantic_attention_data(
        ctx,
        group["sections"],
    )

    finite_values = np.concatenate(
        [values[np.isfinite(values)] for values in values_by_position]
    )
    if finite_values.size == 0:
        raise RuntimeError(f"No finite values are available for {group_name}.")
    if float(finite_values.min()) < 0.6:
        raise RuntimeError(
            f"The compact {group_name} panel would clip values below 0.6; "
            f"observed minimum is {float(finite_values.min()):.4f}."
        )

    positions = np.arange(1, len(labels) + 1)
    boxplot = ax.boxplot(
        values_by_position,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )

    for box in boxplot["boxes"]:
        box.set_facecolor(COLOR_PASTEL_BLUE)
        box.set_edgecolor(COLOR_DEEP_BLUE)
        box.set_alpha(0.45)

    for median in boxplot["medians"]:
        median.set_color(COLOR_DEEP_BLUE)
        median.set_linewidth(2)

    for item in (*boxplot["whiskers"], *boxplot["caps"]):
        item.set_color(COLOR_DEEP_BLUE)

    for position, values in zip(
        positions,
        values_by_position,
        strict=True,
    ):
        ax.scatter(
            position + rng.uniform(-0.13, 0.13, values.size),
            values,
            s=18,
            alpha=0.55,
            color=COLOR_PASTEL_BLUE,
            edgecolors=COLOR_BG,
            linewidths=0.4,
        )
        ax.scatter(
            position,
            float(values.mean()),
            marker="D",
            s=58,
            color=COLOR_DARK_TEXT,
            edgecolors=COLOR_BG,
            linewidths=0.8,
            zorder=4,
        )

    for section_index, (start, end, section_name) in enumerate(
        section_ranges
    ):
        center = 0.5 * ((start + 1) + (end + 1))
        ax.text(
            center,
            1.015,
            section_name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10,
            color=COLOR_DARK_TEXT,
            clip_on=False,
        )

        if section_index < len(section_ranges) - 1:
            ax.axvline(
                end + 1.5,
                color=COLOR_REFLINE,
                linewidth=1,
                alpha=0.55,
            )

    ax.set_title(
        panel_title,
        loc="center",
        fontsize=12.5,
        pad=24,
        color=COLOR_FIGURE_TITLE,
    )
    ax.set_xticks(
        positions,
        labels,
        rotation=27,
        ha="right",
    )
    ax.tick_params(axis="both", labelsize=9.5)
    ax.set_yticks(np.arange(0.6, 1.01, 0.1))
    ax.grid(axis="y", alpha=0.2)


def plot_compact_attention_position_figure(ctx: PlotContext) -> Path:
    """Stack reachability above final-answer attention-head distributions."""

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.0, 7.45),
        sharey=True,
    )
    rng = np.random.default_rng(42)

    draw_compact_attention_position_panel(
        axes[0],
        ctx,
        group_name="reachability",
        panel_title="Probe Reachability performance across token positions",
        rng=rng,
    )
    draw_compact_attention_position_panel(
        axes[1],
        ctx,
        group_name="final_answer_signal",
        panel_title="Probe Final-answer performance across token positions",
        rng=rng,
    )

    # Chance AUROC is 0.5 and therefore lies below this compact
    # figure's explicitly displayed range.
    axes[1].legend(
        handles=attention_distribution_legend()[:-1],
        loc="upper left",
    )

    fig.supylabel(
        metric_label(ctx.metric),
        x=0.017,
        fontsize=10,
        color=COLOR_DEEP_BLUE,
    )
    for ax in axes:
        ax.set_xlabel(
            r"Token position",
            fontsize=11.2,
            color=COLOR_DEEP_BLUE,
            labelpad=7,
        )
    fig.subplots_adjust(
        top=0.95,
        bottom=0.17,
        left=0.10,
        right=0.98,
        hspace=0.88,
    )

    output_root = ctx.output_dir / "paper_compact_figures"
    return save_figure(
        fig,
        output_root / "attention_head_probe_performance_combined.png",
        ctx.dpi,
    )


def plot_paper_compact_figures(ctx: PlotContext) -> list[Path]:
    """Create additive compact alternatives for the paper's Figures 3-6."""

    if ctx.metric != "test_auroc":
        raise RuntimeError(
            "paper_compact_figures supports --metric test_auroc only."
        )

    return [
        plot_compact_decision_progress_figure(
            ctx,
            max_backtracks=None,
        ),
        plot_compact_decision_progress_figure(
            ctx,
            max_backtracks=8,
        ),
        plot_compact_attention_position_figure(ctx),
    ]



# =============================================================================
# REGISTRY AND CLI
# =============================================================================

CORE_PLOTS = (
    "attention_position_distributions",
    "trace_progress",
    "decision_point_progress",
)

PLOT_REGISTRY: dict[str, tuple[str, Callable[[PlotContext], list[Path]]]] = {
    "attention_position_distributions": (
        "two semantic attention-head performance summaries",
        plot_attention_position_distributions,
    ),
    "trace_progress": (
        "pre-BACK signal pooled and stratified by exact BACK count",
        plot_trace_progress,
    ),
    "decision_point_progress": (
        "final-answer signal across all semantic decision points",
        plot_decision_point_progress,
    ),
    "paper_compact_figures": (
        "compact paper alternatives for Figures 3-6",
        plot_paper_compact_figures,
    ),
    "attention_sample_distributions": (
        "held-out sample scores for the top four attention heads",
        plot_attention_sample_distributions,
    ),
    "layer_head_panels": (
        "appendix layer-by-head maps grouped by trainer task",
        plot_layer_head_panels,
    ),
}


def resolve_plotlist(requested: list[str]) -> list[str]:
    if not requested:
        raise ValueError("--plotlist requires at least one plot name.")
    if requested == ["core"]:
        return list(CORE_PLOTS)

    if "all" in requested:
        if len(requested) != 1:
            raise ValueError("Use --plotlist all by itself, or provide explicit plot names.")
        return list(PLOT_REGISTRY)

    if "core" in requested:
        raise ValueError("Use --plotlist core by itself.")

    unknown = sorted(set(requested) - set(PLOT_REGISTRY))
    if unknown:
        raise ValueError(f"Unknown plot names {unknown}. Use --list-plots to see valid choices.")

    return list(dict.fromkeys(requested))


def print_plot_registry() -> None:
    print("Available plots")
    width = max(len(name) for name in PLOT_REGISTRY)

    for name, (description, _) in PLOT_REGISTRY.items():
        print(f"  {name:<{width}}  {description}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot saved A* probing results.")
    parser.add_argument(
        "--plotlist",
        nargs="+",
        default=["all"],
        help=(
            "Use 'core' for the main results or 'all' for core plus optional "
            "diagnostics. Example: --plotlist trace_progress decision_point_progress"
        ),
    )
    parser.add_argument("--metric", choices=SUPPORTED_METRICS, default=DEFAULT_METRIC)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--list-plots", action="store_true")
    args = parser.parse_args()

    if args.list_plots:
        print_plot_registry()
        return
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")

    plot_names = resolve_plotlist(args.plotlist)
    results = load_results(RESULTS_PATH)
    ctx = PlotContext(results, results["metrics"], args.output_dir, args.metric, args.dpi)

    print("=" * 88)
    print("PLOT A* PROBE RESULTS".center(88))
    print("=" * 88)
    print(f"Results:      {RESULTS_PATH}")
    print(f"Output dir:   {ctx.output_dir}")
    print(f"Metric:       {ctx.metric}")
    print(f"Plots:        {', '.join(plot_names)}")
    print("=" * 88)

    all_outputs: list[Path] = []

    for plot_number, plot_name in enumerate(plot_names, start=1):
        description, function = PLOT_REGISTRY[plot_name]
        print(f"\n[{plot_number}/{len(plot_names)}] " f"{plot_name.replace('_', ' ')} — {description}")
        outputs = function(ctx)
        all_outputs.extend(outputs)
        print(f"  saved {len(outputs)} file(s)")

    if len(set(all_outputs)) != len(all_outputs):
        raise RuntimeError("Two plotting functions wrote the same output path.")

    print("\n" + "=" * 88)
    print(f"Generated {len(all_outputs)} output artifact(s).")
    print(f"Saved under {ctx.output_dir}")
    print("STATUS: PASS")
    print("=" * 88)


if __name__ == "__main__":
    main()
