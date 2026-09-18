#!/usr/bin/env python3
"""
plot_steering_method_vectors.py

Plot the pre-steering geometry saved by compare_steering_method_vectors.py.
The comparison artifact is the only input. This file does not reopen the Probe,
CAA, or Direct Optimisation archives and does not recompute any geometry.

Examples

    python -u plot_steering_method_vectors.py \
      --output-dir exp34_steering_method_vector_plots

    python -u plot_steering_method_vectors.py \
      --plotlist cross_method_stability cross_position_stability direct_probe_caa_span

    python -u plot_steering_method_vectors.py --list-plots
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import torch


# =============================================================================
# CONFIGURATION
# =============================================================================

RUN_TAG = "exp34"
ASTAR_DIR = Path("/workspace/sam/minireason_archives/data_pts/astar")
DEFAULT_INPUT = (ASTAR_DIR / "vector_types_comparison" / f"{RUN_TAG}_vector_types_comparison.pt")
DEFAULT_OUTPUT_DIR = Path(f"{RUN_TAG}_steering_method_vector_plots")
DEFAULT_DPI = 220

COLOR_BG = "#FCFCFD"
COLOR_PANEL = "#F2F6FF"
COLOR_GRID = "#C9D6EE"
COLOR_TEXT = "#26364F"
COLOR_BLUE = "#1C4587"
COLOR_LIGHT_BLUE = "#7CA6DE"
COLOR_SLATE = "#5F7896"
COLOR_RED = "#D57758"
COLOR_SOFT_RED = "#E19D85"
COLOR_GREY = "#8A94A6"
COLOR_FIGURE_TITLE = "#000000"

COMMON_AXIS_CMAP = LinearSegmentedColormap.from_list(
    "common_axis",
    [
        (0.00, COLOR_SOFT_RED),
        (0.25, "#E9B3A1"),
        (0.50, COLOR_BG),
        (0.75, "#C9D9F2"),
        (1.00, COLOR_BLUE),
    ],
)

METHOD_ORDER = ("probe", "caa", "optimised")
METHOD_DISPLAY = {"probe": "Probe", "caa": "CAA", "optimised": "Direct"}
METHOD_COLOR = {"probe": COLOR_BLUE, "caa": COLOR_LIGHT_BLUE, "optimised": COLOR_RED}
METHOD_MARKER = {"probe": "o", "caa": "s", "optimised": "^"}
PAIR_ORDER = ("probe_caa", "probe_optimised", "caa_optimised")
PAIR_DISPLAY = {
    "probe_caa": "Probe–CAA",
    "probe_optimised": "Probe–Direct",
    "caa_optimised": "CAA–Direct",
}
PAIR_COLOR = {
    "probe_caa": COLOR_BLUE,
    "probe_optimised": COLOR_RED,
    "caa_optimised": COLOR_SLATE,
}
REPRESENTATION_ORDER = ("attention_head", "residual_stream", "mlp_output", "final_hidden")
THREE_WAY_REPRESENTATION_ORDER = ("attention_head", "residual_stream", "mlp_output")
REPRESENTATION_DISPLAY = {
    "attention_head": "Attention heads",
    "residual_stream": "Residual stream",
    "mlp_output": "MLP output",
    "final_hidden": "Final hidden",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "figure.titlesize": 15,
    "figure.facecolor": COLOR_BG,
    "savefig.facecolor": COLOR_BG,
    "axes.facecolor": COLOR_PANEL,
    "axes.edgecolor": COLOR_BLUE,
    "axes.labelcolor": COLOR_BLUE,
    "axes.titlecolor": COLOR_BLUE,
    "text.color": COLOR_BLUE,
    "xtick.color": COLOR_BLUE,
    "ytick.color": COLOR_BLUE,
    "grid.color": COLOR_GRID,
    "legend.facecolor": COLOR_BG,
    "legend.edgecolor": COLOR_BLUE,
    "legend.framealpha": 0.92,
})


# =============================================================================
# LOADING AND GENERAL HELPERS
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot saved steering-vector geometry comparisons.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plotlist", nargs="+", default=["all"])
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--list-plots", action="store_true")
    return parser.parse_args()


def load_results(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = torch.load(path, map_location="cpu", weights_only=False)
    if int(data.get("format_version", -1)) != 1:
        raise RuntimeError(
            f"Unsupported comparison format version: {data.get('format_version')!r}"
        )
    required = {
        "complete", "run_tag", "plot_contract", "matched_alignment_rows",
        "common_axis_rows", "reachability_final_rows", "direction_evolution_rows",
        "direct_span_rows", "summaries", "audit",
    }
    missing = required - set(data)
    if missing:
        raise KeyError(f"Comparison artifact is missing fields: {sorted(missing)}")
    if not bool(data["complete"]):
        raise RuntimeError("Comparison artifact is not marked complete.")
    if data["run_tag"] != RUN_TAG:
        raise RuntimeError(f"Comparison run tag {data['run_tag']!r} != {RUN_TAG!r}.")
    required_contract = {
        "operation_sequence",
        "back_reachability_sequence",
        "primary_common_axis_scope",
        "primary_common_axis_metric",
    }
    missing_contract = required_contract - set(data["plot_contract"])
    if missing_contract:
        raise KeyError(
            f"Plot contract is missing fields: {sorted(missing_contract)}"
        )
    return data


def rows_where(rows: Iterable[dict[str, Any]], **filters: Any) -> list[dict[str, Any]]:
    return [row for row in rows if all(row.get(name) == value for name, value in filters.items())]


def finite_values(rows: Iterable[dict[str, Any]], field: str) -> np.ndarray:
    values = [float(row[field]) for row in rows if bool(row.get("valid", True)) and math.isfinite(float(row[field]))]
    return np.asarray(values, dtype=np.float64)


def median_iqr(values: np.ndarray) -> tuple[float, float, float]:
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(values)), float(np.quantile(values, 0.25)), float(np.quantile(values, 0.75))


def style_axis(ax: plt.Axes, y_zero: bool = False) -> None:
    ax.grid(True, axis="y", alpha=0.75, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    if y_zero:
        ax.axhline(0.0, color=COLOR_GREY, linewidth=1.0, linestyle="--", zorder=0)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str], dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in formats:
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def boxplot_with_points(
    ax: plt.Axes,
    groups: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    seed: int = 42,
) -> None:
    positions = np.arange(1, len(groups) + 1)
    nonempty = [group if group.size else np.asarray([np.nan]) for group in groups]
    boxes = ax.boxplot(nonempty, positions=positions, widths=0.58, patch_artist=True, showfliers=False)
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.4)
    for median_line, color in zip(boxes["medians"], colors):
        median_line.set_color(color)
        median_line.set_linewidth(1.6)
    for items in ("whiskers", "caps"):
        for index, line in enumerate(boxes[items]):
            line.set_color(colors[index // 2])
            line.set_linewidth(1.0)

    rng = np.random.default_rng(seed)
    for position, values, color in zip(positions, groups, colors):
        if not values.size:
            continue
        jitter = rng.normal(0.0, 0.045, size=values.size)
        ax.scatter(position + jitter, values, s=12, alpha=0.36, color=color, edgecolors="none", zorder=3)
    ax.set_xticks(positions, labels)


def ordered_aggregate(
    rows: list[dict[str, Any]],
    x_field: str,
    x_values: tuple[str, ...] | list[str],
    method: str,
    value_field: str = "signed_cosine",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    medians, q1s, q3s = [], [], []
    for value in x_values:
        values = finite_values(rows_where(rows, method=method, **{x_field: value}), value_field)
        median, q1, q3 = median_iqr(values)
        medians.append(median)
        q1s.append(q1)
        q3s.append(q3)
    return np.asarray(medians), np.asarray(q1s), np.asarray(q3s)


def short_checkpoint(name: str) -> str:
    return {
        "pre_first_operand": "Before\noperand 1",
        "first_operand_end": "Operand 1",
        "operator": "Operator",
        "second_operand_end": "Operand 2",
        "result_end": "Result",
        "resulting_state_semicolon": "Resulting\nstate",
        "pre_back": "Before\nBACK",
        "back_token": "BACK",
        "restored_state_semicolon": "Restored\nstate",
        "success_token": "SUCCESS",
    }.get(name, name)


def token_position_pair_label(name: str) -> str:
    start, end = name.split("_to_", 1)
    start_label = short_checkpoint(start).replace("\n", " ")
    end_label = short_checkpoint(end).replace("\n", " ")
    return f"{start_label}\nvs {end_label}"

# =============================================================================
# PLOTS
# =============================================================================

def plot_cross_method_stability(data: dict[str, Any], args: argparse.Namespace) -> list[Path]:
    rows = rows_where(
        data["matched_alignment_rows"],
        comparison_scope="three_way_shared",
        label_family="final_answer",
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    for ax, representation in zip(axes, THREE_WAY_REPRESENTATION_ORDER):
        groups, labels, colors = [], [], []
        for pair in PAIR_ORDER:
            values = finite_values(rows_where(rows, method_pair=pair, representation_type=representation), "signed_cosine")
            groups.append(values)
            labels.append(f"{PAIR_DISPLAY[pair]}\n$n$={values.size}")
            colors.append(PAIR_COLOR[pair])
        boxplot_with_points(ax, groups, labels, colors)
        ax.set_title(REPRESENTATION_DISPLAY[representation])
        ax.set_ylim(-1.05, 1.05)
        ax.tick_params(axis="x", rotation=18)
        style_axis(ax, y_zero=True)
    axes[0].set_ylabel(r"Cross-method stability"+"\n"+r"(Type 1) $S^{\mathrm{M1}}$")
    fig.suptitle(
        "Cross-method stability of final-answer directions",
        x=0.510,
        y=1.019,
        fontsize=14,
        color=COLOR_FIGURE_TITLE,
    )
    return save_figure(fig, args.output_dir, "cross_method_stability", args.formats, args.dpi)


def plot_cross_position_stability(data: dict[str, Any], args: argparse.Namespace) -> list[Path]:
    rows = rows_where(
        data["common_axis_rows"],
        scope=data["plot_contract"]["primary_common_axis_scope"],
        label_family="final_answer",
    )
    value_field = data["plot_contract"]["primary_common_axis_metric"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    for ax, representation in zip(axes, THREE_WAY_REPRESENTATION_ORDER):
        groups, labels, colors = [], [], []
        for method in METHOD_ORDER:
            values = finite_values(
                rows_where(
                    rows,
                    method=method,
                    representation_type=representation,
                ),
                value_field,
            )
            groups.append(values)
            labels.append(f"{METHOD_DISPLAY[method]}\n$n$={values.size}")
            colors.append(METHOD_COLOR[method])
        boxplot_with_points(ax, groups, labels, colors)
        ax.set_title(REPRESENTATION_DISPLAY[representation])
        ax.set_ylim(-1.05, 1.05)
        style_axis(ax, y_zero=True)
    axes[0].set_ylabel(r"Cross-position stability $S^{\mathrm{P}}$")
    fig.suptitle(
        "Cross-position stability of final-answer directions",
        x=0.510,
        y=1.035,
        fontsize=14,
        color=COLOR_FIGURE_TITLE,
    )
    return save_figure(fig, args.output_dir, "cross_position_stability", args.formats, args.dpi)


def plot_cross_position_stability_heatmap(data: dict[str, Any], args: argparse.Namespace) -> list[Path]:
    rows = rows_where(
        data["common_axis_rows"],
        scope=data["plot_contract"]["primary_common_axis_scope"],
        label_family="final_answer",
        representation_type="attention_head",
    )
    value_field = data["plot_contract"]["primary_common_axis_metric"]

    layers = sorted({int(row["layer"]) for row in rows if int(row["layer"]) >= 0})
    heads = sorted({int(row["head"]) for row in rows if int(row["head"]) >= 0})

    if not layers or not heads:
        raise RuntimeError("No attention-head common-axis rows are available.")

    layer_index = {layer: i for i, layer in enumerate(layers)}
    head_index = {head: i for i, head in enumerate(heads)}

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.8), sharex=True, sharey=True)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    image = None
    for ax, method in zip(axes, METHOD_ORDER):
        matrix = np.full((len(layers), len(heads)), np.nan)
        for row in rows_where(rows, method=method):
            if bool(row["valid"]):
                matrix[
                    layer_index[int(row["layer"])],
                    head_index[int(row["head"])],
                ] = float(row[value_field])
        image = ax.imshow(matrix, cmap=COMMON_AXIS_CMAP, norm=norm, aspect="auto", origin="lower")
        ax.set_title(METHOD_DISPLAY[method])
        ax.set_xticks(range(len(heads)), [f"H{head}" for head in heads])
        ax.set_yticks(range(len(layers)), [f"L{layer}" for layer in layers])
        for layer_i in range(len(layers)):
            for head_i in range(len(heads)):
                value = matrix[layer_i, head_i]
                if math.isfinite(value):
                    red, green, blue, _ = COMMON_AXIS_CMAP(norm(value))
                    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                    ax.text(
                      head_i, layer_i, f"{value:.2f}", ha="center", va="center", 
                      fontsize=7.3, color="white" if luminance < 0.50 else COLOR_BLUE
                      )
    axes[0].set_ylabel("Layer")
    axes[1].set_xlabel(
        "Attention head",
        labelpad=6,
    )
    if image is not None:
        fig.colorbar(image, ax=axes, fraction=0.025, pad=0.025, label=r"Cross-position stability $S^{\mathrm{P}}$")
    fig.suptitle(
        "Cross-position stability of attention-head directions",
        y=1.04,
        color=COLOR_FIGURE_TITLE,
    )
    return save_figure(fig, args.output_dir, "cross_position_stability_attention_heatmap", args.formats, args.dpi)


def plot_ordered_checkpoint_alignment(
    rows: list[dict[str, Any]],
    methods: tuple[str, ...],
    checkpoints: tuple[str, ...],
    title: str,
    stem: str,
    args: argparse.Namespace,
) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.5), sharey=True)
    x = np.arange(len(checkpoints))
    for ax, representation in zip(axes, THREE_WAY_REPRESENTATION_ORDER):
        subset = rows_where(rows, representation_type=representation)
        for method in methods:
            median, q1, q3 = ordered_aggregate(subset, "checkpoint", checkpoints, method)
            if not np.isfinite(median).any():
                continue
            ax.plot(x, median, marker=METHOD_MARKER[method], linewidth=1.8, color=METHOD_COLOR[method], label=METHOD_DISPLAY[method])
            ax.fill_between(x, q1, q3, color=METHOD_COLOR[method], alpha=0.16)
        ax.set_title(REPRESENTATION_DISPLAY[representation])
        ax.set_xticks(x, [short_checkpoint(checkpoint) for checkpoint in checkpoints], rotation=30, ha="right")
        ax.set_ylim(-1.05, 1.05)
        style_axis(ax, y_zero=True)
    axes[0].set_ylabel(r"Cross-label stability $S^{\mathrm{L}}$")
    axes[1].set_xlabel(
        r"Token position $\rightarrow$",
        labelpad=5,
    )
    handles = [Line2D([0], [0], color=METHOD_COLOR[m], marker=METHOD_MARKER[m], label=METHOD_DISPLAY[m]) for m in methods]
    fig.subplots_adjust(top=0.80, bottom=0.24)
    fig.legend(handles=handles, loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5185, 0.955))
    fig.suptitle(title, x=0.5185, y=1.01, fontsize=14, color=COLOR_FIGURE_TITLE)
    return save_figure(fig, args.output_dir, stem, args.formats, args.dpi)


def plot_cross_label_stability(data: dict[str, Any], args: argparse.Namespace) -> list[Path]:
    rows = data["reachability_final_rows"]
    outputs = []
    outputs += plot_ordered_checkpoint_alignment(
        rows_where(rows, reachability_task="operation_reachability"),
        ("probe", "caa"),
        tuple(data["plot_contract"]["operation_sequence"]),
        "Cross-label stability across operation positions",
        "cross_label_stability_operation",
        args,
    )
    outputs += plot_ordered_checkpoint_alignment(
        rows_where(rows, reachability_task="back_reachability"),
        ("probe", "caa"),
        tuple(data["plot_contract"]["back_reachability_sequence"]),
        "Cross-label stability around BACK",
        "cross_label_stability_back",
        args,
    )
    return outputs


def plot_sparse_successive_position_stability(
    rows: list[dict[str, Any]],
    methods: tuple[str, ...],
    transitions: tuple[str, ...],
    representations: tuple[str, ...],
    title: str,
    stem: str,
    args: argparse.Namespace,
) -> list[Path]:
    fig, axes = plt.subplots(1, len(transitions), figsize=(5.4, 4.1), sharex=True, sharey=True, squeeze=False)
    axes = axes.reshape(-1)
    y = np.arange(len(representations))
    offsets = np.linspace(-0.12, 0.12, len(methods))

    for ax, transition in zip(axes, transitions):
        transition_rows = rows_where(rows, transition=transition)
        for offset, method in zip(offsets, methods):
            median, q1, q3 = ordered_aggregate(
                transition_rows,
                "representation_type",
                representations,
                method,
            )
            finite = np.isfinite(median)
            ax.errorbar(
                median[finite],
                y[finite] + offset,
                xerr=np.vstack((median[finite] - q1[finite], q3[finite] - median[finite])),
                fmt=METHOD_MARKER[method],
                color=METHOD_COLOR[method],
                capsize=3,
                linestyle="none",
                label=METHOD_DISPLAY[method],
            )
        ax.set_xlim(0.0, 1.05)
        ax.set_xticks(np.linspace(0.0, 1.0, 6))
        ax.set_yticks(y, [REPRESENTATION_DISPLAY[name] for name in representations])
        if len(transitions) > 1:
            ax.set_title(token_position_pair_label(transition).replace("\n", " "))
        ax.set_xlabel(r"Successive-position stability $S^{\mathrm{S}}$")
        ax.grid(True, axis="x", alpha=0.75, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].invert_yaxis()
    axes[0].set_ylabel("Model component")
    handles = [Line2D([0], [0], color=METHOD_COLOR[m], marker=METHOD_MARKER[m], linestyle="none", label=METHOD_DISPLAY[m]) for m in methods]
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.82))
    plot_left = min(ax.get_position().x0 for ax in axes)
    plot_right = max(ax.get_position().x1 for ax in axes)
    header_x = 0.5 * (plot_left + plot_right) + 0.015
    fig.legend(handles=handles, loc="upper center", ncol=len(methods), bbox_to_anchor=(header_x, 0.91))
    fig.suptitle(title, x=0.5 + 0.015, y=0.99, ha="center", fontsize=13.5, color=COLOR_FIGURE_TITLE)
    return save_figure(fig, args.output_dir, stem, args.formats, args.dpi)


def plot_successive_position_stability(data: dict[str, Any], args: argparse.Namespace) -> list[Path]:
    rows = data["direction_evolution_rows"]
    outputs = []
    combinations = (
        ("operation", "reachability", ("probe", "caa")),
        ("operation", "final_answer", METHOD_ORDER),
        ("back", "reachability", ("probe", "caa")),
        ("back", "final_answer", METHOD_ORDER),
    )
    for sequence, family, methods in combinations:
        comparison_scope = (
            "three_way_shared"
            if family == "final_answer"
            else "probe_caa_shared"
        )
        subset = rows_where(
            rows,
            sequence=sequence,
            label_family=family,
            comparison_scope=comparison_scope,
        )
        token_position_pairs = tuple(
            row["transition"]
            for row in sorted(
                {row["transition"]: row for row in subset}.values(),
                key=lambda row: int(row["transition_index"]),
            )
        )
        if not token_position_pairs:
            continue
        representations = (
            THREE_WAY_REPRESENTATION_ORDER
            if family == "final_answer"
            else REPRESENTATION_ORDER
        )
        title_family = "Reachability" if family == "reachability" else "Final-answer"
        context = "around BACK" if sequence == "back" else "during operations"

        # ADD THIS BLOCK HERE:
        if sequence == "operation" and family == "final_answer":
            outputs += plot_successive_position_stability_wide(
                subset, methods, token_position_pairs, representations,
                f"Successive-position stability of {title_family.lower()} directions {context}",
                f"successive_position_stability_{sequence}_{family}",
                args
            )
            
        if sequence == "back" and family == "reachability":
            outputs += plot_sparse_successive_position_stability(
                subset,
                methods,
                token_position_pairs,
                representations,
                "Reachability stability across BACK restoration\n",
                f"successive_position_stability_{sequence}_{family}",
                args,
            )
            continue

        ncols = 2
        nrows = math.ceil(len(representations) / ncols)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(14.5, 4.8 * nrows),
            sharey=True,
            squeeze=False,
        )
        axes = axes.reshape(-1)
        x = np.arange(len(token_position_pairs))
        for ax, representation in zip(axes, representations):
            rep_rows = rows_where(subset, representation_type=representation)
            for method in methods:
                median, q1, q3 = ordered_aggregate(
                    rep_rows,
                    "transition",
                    token_position_pairs,
                    method,
                )
                if not np.isfinite(median).any():
                    continue
                ax.plot(x, median, marker=METHOD_MARKER[method], linewidth=1.8, color=METHOD_COLOR[method], label=METHOD_DISPLAY[method])
                ax.fill_between(x, q1, q3, color=METHOD_COLOR[method], alpha=0.16)
            ax.set_title(REPRESENTATION_DISPLAY[representation])
            ax.set_xticks(
                x,
                [token_position_pair_label(name) for name in token_position_pairs],
            )
            ax.set_ylim(-1.05, 1.05)
            style_axis(ax, y_zero=True)
        for ax in axes[len(representations):]:
            ax.axis("off")
        fig.subplots_adjust(
            top=0.86,
            bottom=0.16,
            hspace=0.32,
            wspace=0.18,
        )
        if len(representations) % 2 == 1:
            last_ax = axes[len(representations) - 1]
            pos = last_ax.get_position()
            last_ax.set_position([
                0.5 - pos.width / 2,
                pos.y0,
                pos.width,
                pos.height,
            ])

        axis_label_size = 11.8
        axes[0].set_ylabel(
            r"Successive-position stability $S^{\mathrm{S}}$",
            fontsize=axis_label_size,
        )
        axes[2].set_ylabel(
            r"Successive-position stability $S^{\mathrm{S}}$",
            fontsize=axis_label_size,
        )

        handles = [Line2D([0], [0], color=METHOD_COLOR[m], marker=METHOD_MARKER[m], label=METHOD_DISPLAY[m]) for m in methods]
        fig.legend(handles=handles, loc="upper center", ncol=len(methods), bbox_to_anchor=(0.515, 0.94))
        fig.suptitle(
            f"Successive-position stability of {title_family.lower()} directions {context}",
            x=0.515,
            fontsize=17 if sequence == "operation" else 15,
            color=COLOR_FIGURE_TITLE,
        )
        fig.supxlabel(
            "Successive position pairs $\\rightarrow$",
            x=0.515,
            y=0.075,
            fontsize=11.8,
            color=COLOR_BLUE,
        )
        outputs += save_figure(
            fig,
            args.output_dir,
            f"successive_position_stability_{sequence}_{family}",
            args.formats,
            args.dpi,
        )
    return outputs




def plot_successive_position_stability_wide(subset: list[dict[str, Any]], methods: tuple[str, ...], token_position_pairs: tuple[str, ...], representations: tuple[str, ...], title: str, stem: str, args: argparse.Namespace) -> list[Path]:
    # --- FONT SIZE CONFIGURATION ---
    FS_SUPTITLE = 11.0
    FS_SUBTITLE = 10
    FS_AXIS_LABEL = 9
    FS_TICKS = 8
    FS_LEGEND = 9
    LINE_WIDTH = 1.0      # Thickness of the main lines
    MARKER_SIZE = 4.0     # Size of the dots/markers
    ERROR_ALPHA = 0.16    # Transparency of the shaded area
    # -------------------------------

    fig, axes = plt.subplots(1, len(representations), figsize=(len(representations) * 3.3, 3.2), sharey=True, squeeze=False)
    axes = axes.reshape(-1)
    x = np.arange(len(token_position_pairs))
    
    for ax, representation in zip(axes, representations):
        rep_rows = rows_where(subset, representation_type=representation)
        for method in methods:
            median, q1, q3 = ordered_aggregate(rep_rows, "transition", token_position_pairs, method)
            if not np.isfinite(median).any(): continue
            ax.plot(x, median, marker=METHOD_MARKER[method], linewidth=LINE_WIDTH, markersize=MARKER_SIZE, color=METHOD_COLOR[method], label=METHOD_DISPLAY[method])
            ax.fill_between(x, q1, q3, color=METHOD_COLOR[method], alpha=ERROR_ALPHA)
        
        # Subplot Titles (e.g. "Attention heads")
        ax.set_title(REPRESENTATION_DISPLAY[representation], fontsize=FS_SUBTITLE)
        
        # Tick Labels (the numbers on the Y axis and marks on X)
        ax.tick_params(axis='both', labelsize=FS_TICKS)
        
        indices = [0, len(token_position_pairs) - 1]
        ax.set_xticks(indices)
        ax.set_xticklabels([]) 
        
        ax.set_ylim(-0.3, 1.05)
        style_axis(ax, y_zero=True)

    # Y-Axis Label
    axes[0].set_ylabel("Successive-position\nstability $S^{\mathrm{S}}$", 
                   fontsize=FS_AXIS_LABEL, 
                   labelpad=0) # Adjust labelpad if the text is too close to the numbers
    
    # Centered Bottom Label
    fig.text(0.5, 0.145, r"Successive position pairs $\rightarrow$", 
             ha="center", fontsize=FS_AXIS_LABEL, color=COLOR_BLUE)

    # Legend
    handles = [Line2D([0], [0], color=METHOD_COLOR[m], marker=METHOD_MARKER[m], linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label=METHOD_DISPLAY[m]) for m in methods]
    fig.legend(handles=handles, loc="upper center", ncol=len(methods), 
               bbox_to_anchor=(0.5, 0.90), fontsize=FS_LEGEND)
    
    fig.subplots_adjust(top=0.70, bottom=0.22, wspace=0.12) 
    
    # Main Figure Title
    fig.suptitle(title, fontsize=FS_SUPTITLE, color=COLOR_FIGURE_TITLE, y=0.98)
    
    return save_figure(fig, args.output_dir, f"{stem}_wide", args.formats, args.dpi)



def plot_direct_probe_caa_span(data: dict[str, Any], args: argparse.Namespace) -> list[Path]:
    rows = data["direct_span_rows"]
    representations = ("attention_head", "residual_stream", "mlp_output")
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), sharex=True, sharey=True)
    for ax, representation in zip(axes, representations):
        subset = [
            row
            for row in rows_where(rows, representation_type=representation)
            if bool(row["valid"])
        ]
        for rank, marker, color in ((1, "o", COLOR_BLUE), (2, "^", COLOR_RED)):
            rank_rows = [row for row in subset if int(row["probe_caa_span_rank"]) == rank]
            if not rank_rows:
                continue
            x = np.asarray([row["probe_caa_cosine"] for row in rank_rows], dtype=float)
            y = np.asarray([row["direct_projection_r2"] for row in rank_rows], dtype=float)
            ax.scatter(x, y, s=28, alpha=0.58, marker=marker, color=color, edgecolors="none", label="Direct Optimisation directions")
            chance = finite_values(rank_rows, "chance_projection_r2")
            if chance.size:
                ax.axhline(
                    float(np.median(chance)),
                    color=color,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.65,
                    zorder=0,
                    label="Random unit-vector baseline",
                )

        ax.set_title(REPRESENTATION_DISPLAY[representation])
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-0.03, 1.03)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="upper right", fontsize=8)
        style_axis(ax, y_zero=False)
    axes[0].set_ylabel(r"Cross-method"+"\n"+r"stability (Type 2) $S^{\mathrm{M2}}$")
    for ax in axes:
        ax.set_xlabel(r"Probe–CAA cross-method"+"\n"+r"stability (Type 1) $S^{\mathrm{M1}}$")
    fig.suptitle(
        "Direct Optimisation in the Probe–CAA span",
        x=0.510,
        y=1.01,
        color=COLOR_FIGURE_TITLE,
    )
    return save_figure(fig, args.output_dir, "direct_optimisation_probe_caa_span", args.formats, args.dpi)


PLOTTERS: dict[str, Callable[[dict[str, Any], argparse.Namespace], list[Path]]] = {
    "cross_method_stability": plot_cross_method_stability,
    "cross_position_stability": plot_cross_position_stability,
    "cross_position_stability_heatmap": plot_cross_position_stability_heatmap,
    "cross_label_stability": plot_cross_label_stability,
    "successive_position_stability": plot_successive_position_stability,
    "direct_probe_caa_span": plot_direct_probe_caa_span,
}
CORE_PLOTS = (
    "cross_method_stability",
    "cross_position_stability",
    "cross_label_stability",
    "successive_position_stability",
    "direct_probe_caa_span",
)


# =============================================================================
# MAIN
# =============================================================================

def resolve_plotlist(values: list[str]) -> list[str]:
    if not values:
        raise ValueError("--plotlist requires at least one name.")
    if values == ["all"]:
        return list(PLOTTERS)
    if values == ["core"]:
        return list(CORE_PLOTS)
    if "all" in values or "core" in values:
        raise ValueError(
            "Use --plotlist all or --plotlist core by itself."
        )

    unknown = sorted(set(values) - set(PLOTTERS))
    if unknown:
        raise KeyError(f"Unknown plots {unknown}. Use --list-plots.")
    return list(dict.fromkeys(values))


def main() -> None:
    args = parse_args()
    if args.list_plots:
        print("Available plots:")
        print("  core")
        print("  all")
        for name in PLOTTERS:
            print(f"  {name}")
        return
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    allowed_formats = {"png", "pdf", "svg"}
    if not args.formats or any(extension not in allowed_formats for extension in args.formats):
        raise ValueError(f"--formats must be drawn from {sorted(allowed_formats)}.")

    data = load_results(args.input)
    plots = resolve_plotlist(args.plotlist)
    generated: list[Path] = []
    for name in plots:
        print(f"Plotting {name}...", flush=True)
        generated.extend(PLOTTERS[name](data, args))

    if len(generated) != len(set(generated)):
        raise RuntimeError("Two plotters produced the same output path.")
    if not generated:
        raise RuntimeError("No plot files were generated.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "plot_manifest.txt"
    manifest_path.write_text("\n".join(str(path) for path in generated) + "\n")

    print()
    print("STEERING METHOD VECTOR PLOTS")
    print(f"  Input           {args.input}")
    print(f"  Output dir      {args.output_dir}")
    print(f"  Plot groups     {len(plots)}")
    print(f"  Files written   {len(generated)}")
    print(f"  Manifest        {manifest_path}")
    print("  OVERALL STATUS  PASS")


if __name__ == "__main__":
    main()
