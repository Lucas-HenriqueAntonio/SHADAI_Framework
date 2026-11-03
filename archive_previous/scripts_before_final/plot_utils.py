# plot_utils.py
# Shared utilities for SHADAI figures.
# Designed for multi-run experiments with mean, SD and 95% confidence intervals.

from pathlib import Path
import math
import re

import matplotlib.pyplot as plt
import pandas as pd

try:
    from scipy.stats import t
except Exception:
    t = None


# ============================================================
# GLOBAL IEEE-LIKE STYLE
# ============================================================

plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# SHARED VISUAL IDENTITY
# ============================================================

COLORS = {
    "SHADAI": "#000000",
    "Random": "#B02E0C",
    "Greedy": "#1D3557",
    "No Reputation": "#1B4332",
    "REVS-based": "#6A4C93",
    "Static": "#444444",
    "Dynamic": "#1D3557",
    "Normal": "#000000",
    "Failure Attack": "#B02E0C",
    "Flood Attack": "#1D3557",
    "100 Vehicles": "#1B4332",
    "400 Vehicles": "#B02E0C",
    "800 Vehicles": "#1D3557",
}

LINESTYLES = {
    "SHADAI": "-",
    "Random": "--",
    "Greedy": ":",
    "No Reputation": "-.",
    "REVS-based": "--",
    "Static": "--",
    "Dynamic": "-",
    "Normal": "-",
    "Failure Attack": "--",
    "Flood Attack": ":",
    "100 Vehicles": "-",
    "400 Vehicles": "--",
    "800 Vehicles": ":",
}

LINEWIDTHS = {
    "SHADAI": 3.0,
    "Random": 2.0,
    "Greedy": 2.0,
    "No Reputation": 2.0,
    "REVS-based": 2.2,
    "Static": 2.0,
    "Dynamic": 2.6,
    "Normal": 3.0,
    "Failure Attack": 2.2,
    "Flood Attack": 2.2,
    "100 Vehicles": 2.2,
    "400 Vehicles": 2.2,
    "800 Vehicles": 2.2,
}


# ============================================================
# FILE / DATA HELPERS
# ============================================================

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def read_csv_if_exists(path, required=False):
    path = Path(path)

    if path.exists():
        return pd.read_csv(path)

    if required:
        raise FileNotFoundError(path)

    print(f"[SKIP] File not found: {path}")
    return None


def find_first_existing(paths):
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    return None


def to_bool(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


# ============================================================
# STATISTICS
# ============================================================

def ci95(values):
    s = pd.Series(values, dtype="float64").dropna()
    n = len(s)

    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "sd": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }

    mean = s.mean()

    if n == 1:
        return {
            "n": 1,
            "mean": mean,
            "sd": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }

    sd = s.std(ddof=1)
    se = sd / math.sqrt(n)

    crit = (
        t.ppf(0.975, n - 1)
        if t is not None
        else 1.96
    )

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - crit * se,
        "ci95_high": mean + crit * se,
    }


def aggregate_by_group(df, group_columns, metrics):
    rows = []

    grouped = df.groupby(group_columns, dropna=False)

    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)

        row = {
            col: value
            for col, value in zip(group_columns, key)
        }

        for metric in metrics:
            if metric not in sub.columns:
                continue

            stats = ci95(
                pd.to_numeric(
                    sub[metric],
                    errors="coerce",
                )
            )

            row[f"{metric}_n"] = stats["n"]
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_sd"] = stats["sd"]
            row[f"{metric}_ci95_low"] = stats["ci95_low"]
            row[f"{metric}_ci95_high"] = stats["ci95_high"]

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# FIGURE HELPERS
# ============================================================

def save_figure(fig, output_dir, filename, formats=("png", "pdf", "eps")):
    output_dir = ensure_dir(output_dir)

    for ext in formats:
        kwargs = {
            "bbox_inches": "tight",
        }

        if ext == "png":
            kwargs["dpi"] = 300

        if ext == "eps":
            kwargs["format"] = "eps"

        fig.savefig(
            output_dir / f"{filename}.{ext}",
            **kwargs,
        )

    plt.close(fig)


def _as_percent(value, percent):
    return value * 100.0 if percent else value


def plot_categorical_ci(
    df,
    group_col,
    mean_col,
    low_col,
    high_col,
    ylabel,
    title,
    output_dir,
    filename,
    order=None,
    percent=False,
    xlabel=None,
):
    data = df.copy()

    if order is not None:
        existing = list(data[group_col].astype(str))
        order = [x for x in order if x in existing]
        data[group_col] = pd.Categorical(
            data[group_col].astype(str),
            categories=order,
            ordered=True,
        )
        data = data.sort_values(group_col)

    labels = data[group_col].astype(str).tolist()

    means = [
        _as_percent(v, percent)
        for v in data[mean_col]
    ]

    lows = [
        _as_percent(v, percent)
        for v in data[low_col]
    ]

    highs = [
        _as_percent(v, percent)
        for v in data[high_col]
    ]

    lower_error = [
        m - l
        for m, l in zip(means, lows)
    ]

    upper_error = [
        h - m
        for m, h in zip(means, highs)
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    for i, (label, mean, lo, hi) in enumerate(
        zip(labels, means, lower_error, upper_error)
    ):
        ax.errorbar(
            [i],
            [mean],
            yerr=[[lo], [hi]],
            fmt="o",
            capsize=5,
            linewidth=1.8,
            markersize=7,
            color=COLORS.get(label),
            label=label,
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)

    if xlabel:
        ax.set_xlabel(xlabel)

    ax.set_ylabel(
        f"{ylabel} (%)"
        if percent
        else ylabel
    )

    ax.set_title(title, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    save_figure(fig, output_dir, filename)


def plot_numeric_ci(
    df,
    x_col,
    mean_col,
    low_col,
    high_col,
    ylabel,
    title,
    output_dir,
    filename,
    percent=False,
    xlabel=None,
    label=None,
    invert_x=False,
):
    data = df.copy()

    data[x_col] = pd.to_numeric(
        data[x_col],
        errors="coerce",
    )

    data = data.sort_values(x_col)

    x = data[x_col].tolist()

    means = [
        _as_percent(v, percent)
        for v in data[mean_col]
    ]

    lows = [
        _as_percent(v, percent)
        for v in data[low_col]
    ]

    highs = [
        _as_percent(v, percent)
        for v in data[high_col]
    ]

    lower_error = [
        m - l
        for m, l in zip(means, lows)
    ]

    upper_error = [
        h - m
        for m, h in zip(means, highs)
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    ax.errorbar(
        x,
        means,
        yerr=[lower_error, upper_error],
        fmt="o-",
        capsize=5,
        linewidth=2.0,
        markersize=6,
        label=label,
    )

    if label:
        ax.legend()

    if invert_x:
        ax.invert_xaxis()

    ax.set_xlabel(xlabel if xlabel else x_col)

    ax.set_ylabel(
        f"{ylabel} (%)"
        if percent
        else ylabel
    )

    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    save_figure(fig, output_dir, filename)


def plot_multiline_temporal(
    data_by_label,
    column,
    ylabel,
    title,
    output_dir,
    filename,
    window=20,
):
    fig, ax = plt.subplots(figsize=(12, 6))

    plotted = False

    for label, df in data_by_label.items():
        if df is None:
            continue

        if "step" not in df.columns or column not in df.columns:
            continue

        avg = (
            df.groupby("step")[column]
            .mean()
            .reset_index()
            .sort_values("step")
        )

        avg["smooth"] = (
            avg[column]
            .rolling(window=window, min_periods=1)
            .mean()
        )

        ax.plot(
            avg["step"],
            avg["smooth"],
            color=COLORS.get(label),
            linestyle=LINESTYLES.get(label, "-"),
            linewidth=LINEWIDTHS.get(label, 2.0),
            label=label,
        )

        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xlabel("Simulation Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)

    fig.tight_layout()
    save_figure(fig, output_dir, filename)
    return True
