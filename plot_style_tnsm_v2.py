# plot_style_tnsm_v2.py
# Shared visual identity for the SHADAI final-paper figures.
# Produces publication-ready PNG/PDF/EPS while keeping the richer
# comparison-oriented aesthetic used in the earlier SHADAI figures.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Dark, print-friendly, colorblind-conscious palette.
PALETTE = {
    "SHADAI": "#173F5F",
    "Greedy": "#ED553B",
    "No Reputation": "#3CAEA3",
    "REVS-based": "#7B5EA7",
    "100": "#2A9D8F",
    "400": "#E76F51",
    "800": "#264653",
    "Static": "#6C757D",
    "Dynamic": "#1D5D9B",
    "Baseline": "#173F5F",
    "Accent": "#F4A261",
    "Danger": "#C44536",
    "Neutral": "#6B7280",
}

MARKERS = {
    "SHADAI": "o",
    "Greedy": "s",
    "No Reputation": "^",
    "REVS-based": "D",
    "100": "o",
    "400": "s",
    "800": "^",
    "Static": "s",
    "Dynamic": "o",
}

LINESTYLES = {
    "SHADAI": "-",
    "Greedy": "--",
    "No Reputation": "-.",
    "REVS-based": ":",
    "100": "-",
    "400": "--",
    "800": "-.",
    "Static": "--",
    "Dynamic": "-",
}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 400,
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.2,
    "axes.linewidth": 0.8,
    "lines.linewidth": 2.0,
    "lines.markersize": 5.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_all(fig, outdir, stem):
    outdir = ensure_dir(outdir)
    fig.savefig(outdir / f"{stem}.png", bbox_inches="tight", dpi=400)
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.eps", bbox_inches="tight", format="eps")
    plt.close(fig)

def soften_axis(ax, ygrid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        ax.grid(axis="y", linestyle="--", linewidth=0.65, alpha=0.28)
        ax.set_axisbelow(True)

def panel_label(ax, label):
    ax.text(-0.12, 1.05, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")

def percent_axis(ax, axis="y", xmax=1.0):
    if axis == "y":
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=xmax, decimals=0))
    else:
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=xmax, decimals=0))

def ci_arrays(df, metric, percent=False):
    mean = pd.to_numeric(df[f"{metric}_mean"], errors="coerce").to_numpy(float)
    low = pd.to_numeric(df[f"{metric}_ci95_low"], errors="coerce").to_numpy(float)
    high = pd.to_numeric(df[f"{metric}_ci95_high"], errors="coerce").to_numpy(float)
    if percent:
        mean, low, high = mean*100.0, low*100.0, high*100.0
    return mean, low, high

def line_ci(ax, x, mean, low, high, label=None, key=None, marker=None,
            linestyle=None, color=None, ci_alpha=0.13, zorder=3):
    key = key or label
    color = color or PALETTE.get(key)
    marker = marker or MARKERS.get(key, "o")
    linestyle = linestyle or LINESTYLES.get(key, "-")
    ax.plot(x, mean, marker=marker, linestyle=linestyle, color=color,
            label=label, linewidth=2.1, markersize=5.4, zorder=zorder)
    ax.fill_between(x, low, high, color=color, alpha=ci_alpha, linewidth=0, zorder=zorder-1)

def errorbar_series(ax, x, mean, low, high, label=None, key=None, color=None,
                    marker=None, linestyle="-"):
    key = key or label
    color = color or PALETTE.get(key)
    marker = marker or MARKERS.get(key, "o")
    err = np.vstack([mean-low, high-mean])
    ax.errorbar(x, mean, yerr=err, label=label, color=color, marker=marker,
                linestyle=linestyle, linewidth=1.9, capsize=3.5,
                markersize=5.2, elinewidth=1.0)

def grouped_bar_ci(ax, labels, means, lows, highs, colors=None, ylabel=None,
                   percent=False, value_labels=False, rotate=0):
    x = np.arange(len(labels))
    colors = colors or [PALETTE["SHADAI"]] * len(labels)
    yerr = np.vstack([means-lows, highs-means])
    bars = ax.bar(x, means, color=colors, edgecolor="white", linewidth=0.7,
                  yerr=yerr, capsize=3.5, error_kw={"elinewidth":1.0})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center")
    if ylabel:
        ax.set_ylabel(ylabel)
    soften_axis(ax)
    if value_labels:
        span = max(np.nanmax(means)-np.nanmin(means), abs(np.nanmax(means))*0.1, 1e-9)
        for b, v in zip(bars, means):
            txt = f"{v:.1f}" if percent or abs(v) >= 10 else f"{v:.3f}"
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.03*span,
                    txt, ha="center", va="bottom", fontsize=7.6)

def metric_exists(df, metric):
    return all(c in df.columns for c in (
        f"{metric}_mean", f"{metric}_ci95_low", f"{metric}_ci95_high"
    ))

def numeric(df, col):
    return pd.to_numeric(df[col], errors="coerce").to_numpy(float)
