#!/usr/bin/env python3
"""
Generate polished benchmark visualizations from a vLLM results directory.

Expected directory structure:

results/
├── Model-A/
│   ├── generation.csv
│   └── summarization.csv
├── Model-B/
│   ├── generation.csv
│   └── summarization.csv
└── ...

Each CSV is expected to have (at minimum) the columns:
    prefill_time, decode_time, total_time,
    prompt_tokens, completion_tokens, total_tokens

Usage:
    python visualize_results.py ./results

Outputs:
    results/visualizations/
        *.png
        summary_statistics.csv
        request_level_data.csv

Only the CSV files are required. JSONL generation files are not needed for
the performance visualizations.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# ===========================================================================
# Design system
# ===========================================================================
# Everything visual (colors, fonts, spacing, title/footer treatment) lives
# here so every figure in the report looks like it belongs to the same
# document.

INK = "#1A2130"          # primary text / titles
SUBTEXT = "#6B7684"       # subtitles, footers, secondary labels
GRID = "#E4E7EC"          # gridlines
AXIS_LINE = "#B9C0CB"     # spines / tick lines
CARD_BG = "#FFFFFF"

# A muted, print-safe, colorblind-considerate categorical palette. Models
# keep the same color across every figure via `build_color_map`.
PALETTE = [
    "#2E6F9E",  # steel blue
    "#C4622D",  # burnt orange
    "#3F8E6D",  # forest green
    "#8656A6",  # plum
    "#B6A233",  # olive gold
    "#C14E5B",  # brick red
    "#4B7BA8",  # slate blue
    "#5B6B79",  # graphite
]

# Two-role palette used whenever prefill/decode (or two tasks) are shown
# side-by-side within a single model's bar/segment.
ROLE_COLORS = {
    "prefill": "#2E6F9E",
    "decode": "#C4622D",
}

TASK_SHORT = {
    "summarization": "Summarization",
    "generation": "Generation",
}

FONT_STACK = ["DejaVu Sans", "sans-serif"]

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "savefig.facecolor": CARD_BG,
    "figure.facecolor": CARD_BG,
    "axes.facecolor": CARD_BG,
    "font.family": FONT_STACK,
    "font.size": 11,
    "text.color": INK,
    "axes.edgecolor": AXIS_LINE,
    "axes.labelcolor": INK,
    "axes.titlesize": 13.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 10.5,
    "axes.labelweight": "medium",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.9,
    "grid.alpha": 1.0,
    "xtick.color": SUBTEXT,
    "ytick.color": SUBTEXT,
    "xtick.labelsize": 9.7,
    "ytick.labelsize": 9.7,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "legend.labelcolor": INK,
})

# Vertical layout budget (figure-fraction units) reserved for the
# title block and the footer, independent of figure size. Keeping this
# fixed everywhere means every chart gets identical breathing room.
TITLE_BLOCK = 0.22   # top margin reserved when there's a title+subtitle
FOOTER_BLOCK = 0.10  # bottom margin reserved when there's a footer


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate GitHub-ready visualizations from vLLM benchmark results."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing one subdirectory per model.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for generated plots. Defaults to <results_dir>/visualizations.",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "both"],
        default="png",
        help="Output format. PNG is recommended for embedding in Markdown / GitHub.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving them.",
    )
    return parser.parse_args()


# ===========================================================================
# Small helpers
# ===========================================================================

def clean_model_name(name):
    """Make model directory names more readable in plots."""
    return str(name).replace("_", "-")


def shorten_model_name(name, max_chars=24):
    """Shorten long labels without destroying model identity (for legends
    and other places where horizontal space is tight and wrapping isn't
    an option)."""
    name = clean_model_name(name)
    if len(name) <= max_chars:
        return name
    stripped = name.split("/")[-1]
    if len(stripped) <= max_chars:
        return stripped
    return stripped[: max_chars - 1] + "\u2026"


def wrap_model_label(name, width=13, max_lines=3):
    """Wrap a model name onto multiple horizontal lines at hyphen
    boundaries. Used for x-axis category labels so we never need to tilt
    text (tilted labels are the thing that kept colliding with footers)."""
    name = clean_model_name(name)
    parts = name.split("-")
    lines = []
    current = ""
    for part in parts:
        candidate = f"{current}-{part}" if current else part
        if len(candidate) > width and current:
            lines.append(current)
            current = part
        else:
            current = candidate
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + ["\u2026"]
    return "\n".join(lines)


def model_sort_key(name):
    """Put models in a stable, intuitive order (falls back to alphabetical)."""
    lower = name.lower()
    if "1b" in lower and ("0425" in lower or "instruct" in lower) and "moe" not in lower:
        return (0, lower)
    if "moe" in lower:
        return (1, lower)
    if "7b" in lower:
        return (2, lower)
    return (10, lower)


def mean_ci95(values):
    values = pd.Series(values).dropna().astype(float)
    if len(values) == 0:
        return np.nan, np.nan
    mean = values.mean()
    if len(values) == 1:
        return mean, 0.0
    sem = values.std(ddof=1) / np.sqrt(len(values))
    return mean, 1.96 * sem


def build_color_map(models):
    return {model: PALETTE[i % len(PALETTE)] for i, model in enumerate(models)}


def task_df(df, task):
    return df[df["task"] == task].copy()


def style_axis(ax, y_percent=False, y_thousands=False):
    """Apply the shared axis look: light spines, subtle horizontal grid."""
    ax.grid(axis="y", zorder=0)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="both", length=0)
    if y_percent:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    if y_thousands:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))


def set_categorical_xticks(ax, models):
    """Horizontal, wrapped model-name tick labels. Never tilts text, so
    labels can't run past the bottom of the figure or collide with a
    footer the way rotated labels do."""
    labels = [wrap_model_label(m) for m in models]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.tick_params(axis="x", pad=6)


def set_headroom(ax, tops, factor=1.28, min_top=None):
    """Push the y-axis ceiling well above the tallest bar (including its
    error bar), so value-label annotations and any legend placed in the
    top corner always land in empty space instead of on top of a bar."""
    finite = [t for t in tops if t is not None and np.isfinite(t)]
    if not finite:
        return
    top = max(finite) * factor
    if min_top is not None:
        top = max(top, min_top)
    if top > 0:
        ax.set_ylim(0, top)


def annotate_bars(ax, bars, values, fmt="{:.2f}", errors=None, pad_frac=0.02):
    """Value labels above bars. When errors are supplied, the label sits
    above the error-bar cap rather than the bare bar top, so it never
    overlaps the whisker."""
    y0, y1 = ax.get_ylim()
    pad = (y1 - y0) * pad_frac if y1 > y0 else 0
    for i, (bar, val) in enumerate(zip(bars, values)):
        if val is None or not np.isfinite(val):
            continue
        err = 0.0
        if errors is not None:
            e = errors[i] if i < len(errors) else 0.0
            err = e if (e is not None and np.isfinite(e)) else 0.0
        top = bar.get_height() + err
        ax.annotate(
            fmt.format(val),
            xy=(bar.get_x() + bar.get_width() / 2, top + pad),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color=INK,
        )


def new_figure(figsize):
    fig = plt.figure(figsize=figsize)
    return fig


def finalize(fig, output_dir, stem, args, title=None, subtitle=None, footer=None,
             rect_top=None, rect_bottom=None, left=0.08, right=0.97):
    """Reserve consistent space for title/subtitle (top) and footer (bottom),
    then lay out everything else via constrained-ish manual margins.

    rect_top / rect_bottom let a caller reserve extra room for legends etc.
    on top of the default title/footer budget. left/right let a caller
    widen the margin for long axis tick labels (e.g. a row-labeled heatmap).
    """
    top = 1 - TITLE_BLOCK if (title or subtitle) else 0.94
    if rect_top is not None:
        top = min(top, rect_top)
    bottom = FOOTER_BLOCK if footer else 0.08
    if rect_bottom is not None:
        bottom = max(bottom, rect_bottom)

    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right)

    if title:
        fig.text(0.045, 0.965, title, ha="left", va="top",
                  fontsize=16.5, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.045, 0.965 - (0.052 if title else 0), subtitle, ha="left", va="top",
                  fontsize=10.5, color=SUBTEXT)
    if footer:
        fig.text(0.045, 0.035, footer, ha="left", va="bottom",
                  fontsize=8.6, color=SUBTEXT)

    output_dir.mkdir(parents=True, exist_ok=True)
    formats = ["png", "pdf"] if args.format == "both" else [args.format]
    for ext in formats:
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, facecolor=CARD_BG, pad_inches=0.15)
        print(f"  saved {path}")

    if not args.show:
        plt.close(fig)


# ===========================================================================
# Data loading
# ===========================================================================

def load_results(results_dir):
    """Load all available generation/summarization CSVs."""
    records = []

    model_dirs = sorted(
        (p for p in results_dir.iterdir() if p.is_dir() and p.name != "visualizations"),
        key=lambda p: model_sort_key(p.name),
    )

    if not model_dirs:
        raise FileNotFoundError(f"No model subdirectories found in {results_dir}")

    required = {
        "prefill_time", "decode_time", "total_time",
        "prompt_tokens", "completion_tokens", "total_tokens",
    }
    numeric_cols = sorted(required)

    for model_dir in model_dirs:
        for task in ("summarization", "generation"):
            path = model_dir / f"{task}.csv"
            if not path.exists():
                print(f"WARNING: missing {path}; skipping.")
                continue

            df = pd.read_csv(path)
            missing = required - set(df.columns)
            if missing:
                print(f"WARNING: {path} is missing columns {sorted(missing)}; skipping.")
                continue

            df = df.copy()
            df["model"] = model_dir.name
            df["task"] = task

            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df["prefill_tok_s"] = df["prompt_tokens"] / df["prefill_time"].replace(0, np.nan)
            df["decode_tok_s"] = df["completion_tokens"] / df["decode_time"].replace(0, np.nan)
            df["prefill_fraction"] = df["prefill_time"] / df["total_time"].replace(0, np.nan)
            df["decode_fraction"] = df["decode_time"] / df["total_time"].replace(0, np.nan)

            records.append(df)
            print(f"Loaded {task:13s} | {model_dir.name:35s} | {len(df):5d} rows")

    if not records:
        raise RuntimeError("No valid result CSVs were found.")

    return pd.concat(records, ignore_index=True)


def aggregate(df):
    rows = []
    metrics = [
        "prefill_time", "decode_time", "total_time",
        "prompt_tokens", "completion_tokens", "total_tokens",
        "prefill_tok_s", "decode_tok_s", "prefill_fraction", "decode_fraction",
    ]
    for (model, task), group in df.groupby(["model", "task"], sort=False):
        row = {"model": model, "task": task, "n": len(group)}
        for metric in metrics:
            mean, ci = mean_ci95(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95"] = ci
            row[f"{metric}_median"] = group[metric].median()
            row[f"{metric}_std"] = group[metric].std()
        rows.append(row)
    return pd.DataFrame(rows)


# ===========================================================================
# Plots
# ===========================================================================

def plot_summary_dashboard(df, output_dir, color_map, args):
    """A compact, README-friendly dashboard with the headline numbers."""
    models = sorted(df["model"].unique(), key=model_sort_key)
    x = np.arange(len(models))
    width = 0.34

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

    # 1. Total latency
    ax = axes[0, 0]
    bar_groups, all_means = [], []
    for i, task in enumerate(("summarization", "generation")):
        means = [df[(df["model"] == m) & (df["task"] == task)]["total_time"].mean() for m in models]
        bars = ax.bar(x + (i - 0.5) * width, means, width=width,
                       label=TASK_SHORT[task], color=PALETTE[i], alpha=0.92,
                       edgecolor="white", linewidth=0.6, zorder=3)
        bar_groups.append((bars, means))
        all_means.extend(means)
    set_headroom(ax, all_means, factor=1.28)
    for bars, means in bar_groups:
        annotate_bars(ax, bars, means, fmt="{:.2f}s")
    set_categorical_xticks(ax, models)
    ax.set_ylabel("Seconds")
    ax.set_title("Total latency")
    ax.legend(loc="upper right", ncol=2)
    style_axis(ax)

    # 2. Phase composition (stacked share of time)
    ax = axes[0, 1]
    for i, task in enumerate(("summarization", "generation")):
        s = df[df["task"] == task].groupby("model")[["prefill_fraction", "decode_fraction"]].mean().reindex(models)
        pos = x + (i - 0.5) * width
        ax.bar(pos, s["prefill_fraction"], width=width,
               label="Prefill" if i == 0 else None, color=ROLE_COLORS["prefill"],
               alpha=0.92, edgecolor="white", linewidth=0.6, zorder=3)
        ax.bar(pos, s["decode_fraction"], width=width, bottom=s["prefill_fraction"],
               label="Decode" if i == 0 else None, color=ROLE_COLORS["decode"],
               alpha=0.92, edgecolor="white", linewidth=0.6, zorder=3)
    set_categorical_xticks(ax, models)
    ax.set_ylabel("Share of total time")
    ax.set_ylim(0, 1.2)
    ax.set_title("Phase composition")
    ax.legend(loc="upper right", ncol=2)
    style_axis(ax, y_percent=True)

    # 3. Workload length
    ax = axes[1, 0]
    all_means = []
    for i, task in enumerate(("summarization", "generation")):
        metric = "prompt_tokens" if task == "summarization" else "completion_tokens"
        means = [df[(df["model"] == m) & (df["task"] == task)][metric].mean() for m in models]
        ax.bar(x + (i - 0.5) * width, means, width=width, label=TASK_SHORT[task],
               color=PALETTE[i], alpha=0.92, edgecolor="white", linewidth=0.6, zorder=3)
        all_means.extend(means)
    set_headroom(ax, all_means, factor=1.22)
    set_categorical_xticks(ax, models)
    ax.set_ylabel("Tokens")
    ax.set_title("Workload length")
    ax.legend(loc="upper right", ncol=2)
    style_axis(ax, y_thousands=True)

    # 4. Effective throughput
    ax = axes[1, 1]
    prefill = [df[(df["model"] == m) & (df["task"] == "summarization")]["prefill_tok_s"].mean() for m in models]
    decode = [df[(df["model"] == m) & (df["task"] == "generation")]["decode_tok_s"].mean() for m in models]
    ax.bar(x - width / 2, prefill, width=width, label="Prefill", color=ROLE_COLORS["prefill"],
           alpha=0.92, edgecolor="white", linewidth=0.6, zorder=3)
    ax.bar(x + width / 2, decode, width=width, label="Decode", color=ROLE_COLORS["decode"],
           alpha=0.92, edgecolor="white", linewidth=0.6, zorder=3)
    set_headroom(ax, prefill + decode, factor=1.22)
    set_categorical_xticks(ax, models)
    ax.set_ylabel("Tokens / second")
    ax.set_title("Effective throughput")
    ax.legend(loc="upper right", ncol=2)
    style_axis(ax, y_thousands=True)

    fig.subplots_adjust(hspace=0.55, wspace=0.24)
    finalize(
        fig, output_dir, "00_benchmark_dashboard", args,
        title="vLLM Inference Benchmark",
        subtitle="Prefill vs. decode behavior across models and workloads, at a glance",
        footer="Values are means across completed requests. See individual charts for distributions and confidence intervals.",
        rect_bottom=0.09,
    )


def plot_phase_times(df, output_dir, color_map, args):
    """Two-panel comparison of mean prefill and decode time."""
    summary = (
        df.groupby(["task", "model"])
        .agg(prefill_mean=("prefill_time", "mean"),
             prefill_ci=("prefill_time", lambda x: mean_ci95(x)[1]),
             decode_mean=("decode_time", "mean"),
             decode_ci=("decode_time", lambda x: mean_ci95(x)[1]))
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))

    panels = [
        (axes[0], "summarization", "prefill_mean", "prefill_ci", "Prefill time (summarization)"),
        (axes[1], "generation", "decode_mean", "decode_ci", "Decode time (generation)"),
    ]
    for ax, task, metric, ci, title in panels:
        s = summary[summary["task"] == task].copy()
        s["order"] = s["model"].map(model_sort_key)
        s = s.sort_values("order")

        x = np.arange(len(s))
        bars = ax.bar(x, s[metric], yerr=s[ci], capsize=4, width=0.6,
                       color=[color_map[m] for m in s["model"]], alpha=0.92,
                       edgecolor="white", linewidth=0.6, zorder=3,
                       error_kw={"elinewidth": 1.1, "ecolor": SUBTEXT})
        set_headroom(ax, (s[metric] + s[ci].fillna(0)).tolist(), factor=1.22)
        set_categorical_xticks(ax, s["model"])
        ax.set_ylabel("Seconds")
        ax.set_title(title)
        annotate_bars(ax, bars, s[metric], fmt="{:.3f}s", errors=s[ci].tolist())
        style_axis(ax)

    finalize(
        fig, output_dir, "01_phase_timing", args,
        title="Inference Phase Timing",
        subtitle="Mean request time by model, with 95% confidence intervals",
        footer="Left: prefill-dominant summarization workload. Right: decode-dominant report-generation workload.",
        rect_bottom=0.16,
    )


def plot_total_time(df, output_dir, color_map, args):
    summary = (
        df.groupby(["task", "model"])
        .agg(mean=("total_time", "mean"), ci=("total_time", lambda x: mean_ci95(x)[1]))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    models = sorted(df["model"].unique(), key=model_sort_key)
    x = np.arange(len(models))
    width = 0.34

    bar_groups, tops = [], []
    for i, task in enumerate(("summarization", "generation")):
        s = summary[summary["task"] == task].set_index("model").reindex(models)
        pos = x + (i - 0.5) * width
        bars = ax.bar(pos, s["mean"], width=width, yerr=s["ci"], capsize=4,
                       label=TASK_SHORT[task], color=PALETTE[i], alpha=0.92,
                       edgecolor="white", linewidth=0.6, zorder=3,
                       error_kw={"elinewidth": 1.1, "ecolor": SUBTEXT})
        bar_groups.append((bars, s["mean"].tolist(), s["ci"].tolist()))
        tops.extend((s["mean"].fillna(0) + s["ci"].fillna(0)).tolist())

    set_headroom(ax, tops, factor=1.3)
    for bars, means, cis in bar_groups:
        annotate_bars(ax, bars, means, fmt="{:.3f}", errors=cis)

    set_categorical_xticks(ax, models)
    ax.set_ylabel("Total request time (s)")
    ax.set_title("Total inference time")
    ax.legend(loc="upper right", ncol=2)
    style_axis(ax)

    finalize(
        fig, output_dir, "02_total_time", args,
        title="Total Inference Time",
        subtitle="End-to-end prefill + decode time across both benchmark workloads",
        footer="Bars show means; error bars show 95% confidence intervals.",
        rect_bottom=0.15,
    )


def plot_token_counts(df, output_dir, color_map, args):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))

    panels = [
        (axes[0], "summarization", "prompt_tokens", "Input length (summarization)", "Prompt tokens"),
        (axes[1], "generation", "completion_tokens", "Output length (generation)", "Generated tokens"),
    ]
    for ax, task, metric, title, ylabel in panels:
        s = task_df(df, task)
        models = sorted(s["model"].unique(), key=model_sort_key)
        means = [s.loc[s["model"] == m, metric].mean() for m in models]
        cis = [mean_ci95(s.loc[s["model"] == m, metric])[1] for m in models]

        x = np.arange(len(models))
        bars = ax.bar(x, means, yerr=cis, capsize=4, color=[color_map[m] for m in models],
                       width=0.6, alpha=0.92, edgecolor="white", linewidth=0.6, zorder=3,
                       error_kw={"elinewidth": 1.1, "ecolor": SUBTEXT})
        set_headroom(ax, [m + (c or 0) for m, c in zip(means, cis)], factor=1.2)
        set_categorical_xticks(ax, models)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        annotate_bars(ax, bars, means, fmt="{:,.0f}", errors=cis)
        style_axis(ax, y_thousands=True)

    finalize(
        fig, output_dir, "03_token_lengths", args,
        title="Workload Token Lengths",
        subtitle="Input and generated token counts define the character of each benchmark phase",
        footer="Summarization: long reports in, short summaries out. Generation: short prompts in, long reports out.",
        rect_bottom=0.16,
    )


def plot_throughput(df, output_dir, color_map, args):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))

    panels = [
        (axes[0], "summarization", "prefill_tok_s", "Prefill throughput"),
        (axes[1], "generation", "decode_tok_s", "Decode throughput"),
    ]
    for ax, task, metric, title in panels:
        s = task_df(df, task)
        models = sorted(s["model"].unique(), key=model_sort_key)
        vals = [s.loc[s["model"] == m, metric].mean() for m in models]

        x = np.arange(len(models))
        bars = ax.bar(x, vals, color=[color_map[m] for m in models], width=0.6,
                       alpha=0.92, edgecolor="white", linewidth=0.6, zorder=3)
        set_headroom(ax, vals, factor=1.18)
        set_categorical_xticks(ax, models)
        ax.set_ylabel("Tokens / second")
        ax.set_title(title)
        annotate_bars(ax, bars, vals, fmt="{:,.0f}")
        style_axis(ax, y_thousands=True)

    finalize(
        fig, output_dir, "04_throughput", args,
        title="Inference Throughput",
        subtitle="Effective phase throughput, computed from measured tokens and phase time",
        footer="Prefill throughput uses summarization requests; decode throughput uses report-generation requests.",
        rect_bottom=0.16,
    )


def plot_phase_balance(df, output_dir, color_map, args):
    summary = (
        df.groupby(["task", "model"])
        .agg(prefill=("prefill_fraction", "mean"), decode=("decode_fraction", "mean"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(11, 6.0))
    models = sorted(df["model"].unique(), key=model_sort_key)
    x = np.arange(len(models))
    width = 0.34

    for i, task in enumerate(("summarization", "generation")):
        s = summary[summary["task"] == task].set_index("model").reindex(models)
        pos = x + (i - 0.5) * width
        ax.bar(pos, s["prefill"], width=width, label=f"Prefill" if i == 0 else None,
               color=ROLE_COLORS["prefill"], alpha=0.92, edgecolor="white", linewidth=0.6, zorder=3)
        ax.bar(pos, s["decode"], width=width, bottom=s["prefill"],
               label="Decode" if i == 0 else None, color=ROLE_COLORS["decode"],
               alpha=0.92, edgecolor="white", linewidth=0.6, zorder=3)

    set_categorical_xticks(ax, models)
    ax.set_ylabel("Fraction of measured inference time")
    ax.set_ylim(0, 1.18)
    ax.set_title("Where inference time is spent")
    ax.legend(loc="upper right", ncol=2)
    style_axis(ax, y_percent=True)

    finalize(
        fig, output_dir, "05_phase_balance", args,
        title="Prefill vs. Decode Time Share",
        subtitle="Relative contribution of each phase, by model and workload",
        footer="Fractions are computed per request, then averaged by model and workload. Left bar in each pair: summarization; right: generation.",
        rect_bottom=0.16,
    )


def plot_distributions(df, output_dir, color_map, args):
    """Violin + box distributions of total request time."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0))
    models = sorted(df["model"].unique(), key=model_sort_key)

    for ax, task in zip(axes, ("summarization", "generation")):
        s = task_df(df, task)
        data = [s.loc[s["model"] == m, "total_time"].dropna().to_numpy() for m in models]

        parts = ax.violinplot(data, positions=np.arange(len(models)), widths=0.75,
                               showmeans=False, showmedians=False, showextrema=False)
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(color_map[models[i]])
            body.set_edgecolor("none")
            body.set_alpha(0.35)

        ax.boxplot(data, positions=np.arange(len(models)), widths=0.16, patch_artist=True,
                   showfliers=False,
                   boxprops=dict(facecolor="white", edgecolor=INK, linewidth=1),
                   medianprops=dict(color=INK, linewidth=1.6),
                   whiskerprops=dict(color=SUBTEXT, linewidth=1),
                   capprops=dict(color=SUBTEXT, linewidth=1))

        rng = np.random.default_rng(42)
        for i, values in enumerate(data):
            if len(values) == 0:
                continue
            jitter = rng.normal(0, 0.05, size=len(values))
            ax.scatter(np.full(len(values), i) + jitter, values, s=8, alpha=0.25,
                       color=color_map[models[i]], linewidths=0, zorder=2)

        set_categorical_xticks(ax, models)
        ax.set_ylabel("Total request time (s)")
        ax.set_title(TASK_SHORT[task])
        style_axis(ax)

    finalize(
        fig, output_dir, "06_latency_distributions", args,
        title="Distribution of Request Latency",
        subtitle="Violin distributions with overlaid boxplots and individual requests",
        footer="Each point is one benchmark request. Boxes show the median and interquartile range.",
        rect_bottom=0.16,
    )


def plot_scaling_relationships(df, output_dir, color_map, args):
    """Prompt tokens vs prefill time, and completion tokens vs decode time,
    with a per-model least-squares trend line."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0))

    specs = [
        (axes[0], "summarization", "prompt_tokens", "prefill_time", "Prompt tokens", "Prefill time (s)", "Prefill scaling"),
        (axes[1], "generation", "completion_tokens", "decode_time", "Generated tokens", "Decode time (s)", "Decode scaling"),
    ]

    for ax, task, xcol, ycol, xlabel, ylabel, title in specs:
        s = task_df(df, task)
        models = sorted(s["model"].unique(), key=model_sort_key)

        for model in models:
            d = s[s["model"] == model][[xcol, ycol]].dropna()
            if len(d) == 0:
                continue
            ax.scatter(d[xcol], d[ycol], s=18, alpha=0.32, color=color_map[model],
                       edgecolors="none", label=shorten_model_name(model), zorder=2)
            if len(d) >= 2 and d[xcol].nunique() >= 2:
                coeffs = np.polyfit(d[xcol], d[ycol], 1)
                xs = np.linspace(d[xcol].min(), d[xcol].max(), 100)
                ax.plot(xs, coeffs[0] * xs + coeffs[1], linewidth=2.2,
                        color=color_map[model], alpha=0.95, zorder=3)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="both")
        ax.grid(axis="x", visible=True)
        ax.tick_params(length=0)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4),
               bbox_to_anchor=(0.5, -0.02), frameon=False)

    finalize(
        fig, output_dir, "07_scaling_relationships", args,
        title="Empirical Scaling of Inference Phases",
        subtitle="Relationship between workload length and measured phase time",
        footer="Points are individual requests; lines are per-model least-squares fits, not theoretical scaling laws.",
        rect_bottom=0.22,
    )


def plot_model_workload_comparison(df, output_dir, color_map, args):
    """Heatmap of median total time: one row per model, one column per workload."""
    summary = df.groupby(["model", "task"])["total_time"].median().unstack("task")
    models = sorted(summary.index, key=model_sort_key)
    columns = [c for c in ("summarization", "generation") if c in summary.columns]
    summary = summary.reindex(index=models, columns=columns)
    values = summary.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, max(3.8, 0.85 * len(models) + 1.6)))
    cmap = plt.get_cmap("Blues")
    image = ax.imshow(values, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels([TASK_SHORT[c] for c in columns])
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([shorten_model_name(m, 30) for m in models])
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    finite = values[np.isfinite(values)]
    vmid = (finite.min() + finite.max()) / 2 if finite.size else 0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if not np.isfinite(v):
                continue
            text_color = "white" if v > vmid else INK
            ax.text(j, i, f"{v:.3f}s", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=text_color)

    ax.set_title("Median latency by model and workload")
    cbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.04)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=9)
    cbar.set_label("Median total time (s)", fontsize=9.5, color=SUBTEXT)

    finalize(
        fig, output_dir, "08_latency_matrix", args,
        title="Latency Across the Benchmark Matrix",
        subtitle="Median request latency for every model \u00d7 workload combination",
        footer="Median is used to reduce sensitivity to occasional unusually long requests.",
        left=0.26, right=0.92, rect_bottom=0.16,
    )


def plot_prefill_decode_scatter(df, output_dir, color_map, args):
    """Scatter of prefill time vs decode time for all requests."""
    fig, ax = plt.subplots(figsize=(10, 6.4))

    markers = {"summarization": "o", "generation": "s"}
    for task in ("summarization", "generation"):
        s = task_df(df, task)
        for model in sorted(s["model"].unique(), key=model_sort_key):
            d = s[s["model"] == model]
            if d.empty:
                continue
            ax.scatter(d["prefill_time"], d["decode_time"], s=24, alpha=0.35,
                       color=color_map[model], marker=markers[task], edgecolors="none",
                       label=f"{shorten_model_name(model)} \u2014 {TASK_SHORT[task]}", zorder=2)

    ax.set_xlabel("Prefill time (s)")
    ax.set_ylabel("Decode time (s)")
    ax.grid(axis="both")
    ax.grid(axis="x", visible=True)
    ax.tick_params(length=0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8.6)

    finalize(
        fig, output_dir, "09_prefill_vs_decode", args,
        title="Prefill vs. Decode Behavior",
        subtitle="Each point is one request; shape marks the workload, color marks the model",
        footer="Points near the axes are dominated by a single inference phase; points along the diagonal are balanced.",
        right=0.74, rect_bottom=0.13,
    )


# ===========================================================================
# Summary statistics
# ===========================================================================

def save_summary_statistics(df, aggregate_df, output_dir):
    """Save both aggregate and request-level data in analysis-friendly CSVs."""
    preferred = [
        "model", "task", "n",
        "prefill_time_mean", "prefill_time_ci95", "prefill_time_median",
        "decode_time_mean", "decode_time_ci95", "decode_time_median",
        "total_time_mean", "total_time_ci95", "total_time_median",
        "prompt_tokens_mean", "completion_tokens_mean", "total_tokens_mean",
        "prefill_tok_s_mean", "decode_tok_s_mean",
        "prefill_fraction_mean", "decode_fraction_mean",
    ]
    columns = [c for c in preferred if c in aggregate_df.columns]
    aggregate_df[columns].to_csv(output_dir / "summary_statistics.csv", index=False, float_format="%.6f")
    df.to_csv(output_dir / "request_level_data.csv", index=False)

    print(f"  saved {output_dir / 'summary_statistics.csv'}")
    print(f"  saved {output_dir / 'request_level_data.csv'}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    if not results_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {results_dir}")

    output_dir = (args.output_dir.resolve() if args.output_dir is not None
                  else results_dir / "visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("vLLM Benchmark Visualization")
    print("=" * 72)
    print(f"Results: {results_dir}")
    print(f"Output:  {output_dir}")
    print()

    df = load_results(results_dir)
    models = sorted(df["model"].unique(), key=model_sort_key)
    color_map = build_color_map(models)

    print()
    print(f"Models: {len(models)}")
    for model in models:
        print(f"  - {model}")
    print(f"Total requests: {len(df)}")
    print()

    aggregate_df = aggregate(df)

    print("Generating plots...")
    plot_summary_dashboard(df, output_dir, color_map, args)
    plot_phase_times(df, output_dir, color_map, args)
    plot_total_time(df, output_dir, color_map, args)
    plot_token_counts(df, output_dir, color_map, args)
    plot_throughput(df, output_dir, color_map, args)
    plot_phase_balance(df, output_dir, color_map, args)
    plot_distributions(df, output_dir, color_map, args)
    plot_scaling_relationships(df, output_dir, color_map, args)
    plot_model_workload_comparison(df, output_dir, color_map, args)
    plot_prefill_decode_scatter(df, output_dir, color_map, args)

    print()
    print("Saving statistics...")
    save_summary_statistics(df, aggregate_df, output_dir)

    print()
    print("=" * 72)
    print("Done.")
    print(f"All visualizations are in: {output_dir}")
    print("=" * 72)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()