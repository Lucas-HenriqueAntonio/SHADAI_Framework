# plot_utils.py
# Shared plotting/statistical utilities for the final SHADAI campaign.

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import t
except Exception:
    t = None


# IEEE-friendly defaults. These preserve the visual identity of the
# original project while improving readability of multi-run figures.
plt.rcParams.update({
    "figure.figsize": (8.8, 5.2),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLORS = {
    "SHADAI": "#000000",
    "Greedy": "#1D3557",
    "No Reputation": "#1B4332",
    "REVS-based": "#6A4C93",
    "Random": "#B02E0C",
    "Static": "#555555",
    "Dynamic": "#1D3557",
    "100 Vehicles": "#1B4332",
    "400 Vehicles": "#B02E0C",
    "800 Vehicles": "#1D3557",
    "Failure": "#B02E0C",
    "Flood": "#1D3557",
}

LINESTYLES = {
    "SHADAI": "-",
    "Greedy": "--",
    "No Reputation": "-.",
    "REVS-based": ":",
    "100 Vehicles": "-",
    "400 Vehicles": "--",
    "800 Vehicles": ":",
}

MARKERS = {
    "SHADAI": "o",
    "Greedy": "s",
    "No Reputation": "^",
    "REVS-based": "D",
    "100 Vehicles": "o",
    "400 Vehicles": "s",
    "800 Vehicles": "^",
}


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ci95(values):
    s = pd.Series(values, dtype="float64").dropna()
    n = len(s)

    if n == 0:
        return n, np.nan, np.nan, np.nan, np.nan

    mean = s.mean()

    if n == 1:
        return n, mean, np.nan, np.nan, np.nan

    sd = s.std(ddof=1)
    crit = t.ppf(0.975, n - 1) if t is not None else 1.96
    half = crit * sd / math.sqrt(n)

    return n, mean, sd, mean - half, mean + half


def save_figure(fig, output_dir, stem):
    out = ensure_dir(output_dir)

    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.eps", format="eps", bbox_inches="tight")

    plt.close(fig)


def add_grid(ax):
    ax.grid(axis="y", linestyle="--", alpha=0.30)
    ax.set_axisbelow(True)


def grouped_bar_ci(
    data,
    label_col,
    mean_col,
    low_col,
    high_col,
    ylabel,
    title,
    output_dir,
    stem,
    order=None,
    percent=False,
):
    df = data.copy()

    if order is not None:
        order = [x for x in order if x in set(df[label_col].astype(str))]
        df[label_col] = pd.Categorical(
            df[label_col].astype(str),
            categories=order,
            ordered=True,
        )
        df = df.sort_values(label_col)

    labels = df[label_col].astype(str).tolist()
    factor = 100.0 if percent else 1.0

    means = pd.to_numeric(df[mean_col], errors="coerce").to_numpy() * factor
    low = pd.to_numeric(df[low_col], errors="coerce").to_numpy() * factor
    high = pd.to_numeric(df[high_col], errors="coerce").to_numpy() * factor

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    colors = [COLORS.get(label, "#666666") for label in labels]

    bars = ax.bar(
        x,
        means,
        width=0.62,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        yerr=np.vstack([means - low, high - means]),
        capsize=5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"{ylabel} (%)" if percent else ylabel)
    ax.set_title(title, fontweight="bold")
    add_grid(ax)

    # Value labels improve readability without replacing uncertainty bars.
    span = np.nanmax(means) - np.nanmin(means) if len(means) else 0.0
    offset = 0.015 * (abs(np.nanmax(means)) + span + 1e-9)

    for bar, value in zip(bars, means):
        if np.isfinite(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                f"{value:.2f}" if abs(value) >= 1 else f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    fig.tight_layout()
    save_figure(fig, output_dir, stem)


def line_ci(
    data,
    x_col,
    mean_col,
    low_col,
    high_col,
    xlabel,
    ylabel,
    title,
    output_dir,
    stem,
    percent=False,
    label=None,
    invert_x=False,
):
    df = data.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df = df.sort_values(x_col)

    factor = 100.0 if percent else 1.0

    x = df[x_col].to_numpy(dtype=float)
    mean = pd.to_numeric(df[mean_col], errors="coerce").to_numpy() * factor
    low = pd.to_numeric(df[low_col], errors="coerce").to_numpy() * factor
    high = pd.to_numeric(df[high_col], errors="coerce").to_numpy() * factor

    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    ax.plot(
        x,
        mean,
        marker="o",
        linewidth=2.2,
        markersize=6,
        label=label,
    )
    ax.fill_between(x, low, high, alpha=0.18)

    if label:
        ax.legend(frameon=True)

    if invert_x:
        ax.invert_xaxis()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{ylabel} (%)" if percent else ylabel)
    ax.set_title(title, fontweight="bold")
    add_grid(ax)

    fig.tight_layout()
    save_figure(fig, output_dir, stem)


def temporal_multiline_ci(
    data,
    metric,
    series_col,
    x_col,
    output_dir,
    stem,
    ylabel,
    title,
    order,
    rolling_window=5,
):
    df = data[data["metric"] == metric].copy()

    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    plotted = False

    for series in order:
        sub = df[df[series_col] == series].copy()

        if sub.empty:
            continue

        sub = sub.sort_values(x_col)

        # Smooth the across-seed mean and CI bounds only for visual clarity.
        # Raw aggregated values remain saved in the CSV.
        for col in ["mean", "ci95_low", "ci95_high"]:
            sub[col] = (
                pd.to_numeric(sub[col], errors="coerce")
                .rolling(rolling_window, min_periods=1, center=True)
                .mean()
            )

        x = pd.to_numeric(sub[x_col], errors="coerce").to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        lo = sub["ci95_low"].to_numpy(dtype=float)
        hi = sub["ci95_high"].to_numpy(dtype=float)

        color = COLORS.get(series)

        ax.plot(
            x,
            y,
            color=color,
            linestyle=LINESTYLES.get(series, "-"),
            linewidth=2.2,
            label=series,
        )

        ax.fill_between(
            x,
            lo,
            hi,
            color=color,
            alpha=0.12,
        )

        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xlabel("Simulation Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.28)
    ax.legend(frameon=True, ncol=min(3, len(order)))

    fig.tight_layout()
    save_figure(fig, output_dir, stem)
    return True
