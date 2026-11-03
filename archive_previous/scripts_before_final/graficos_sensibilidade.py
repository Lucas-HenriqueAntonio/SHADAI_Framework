# graficos_sensibilidade.py
# C25b + C26 sensitivity/stability figures.
# All primary figures use multi-run mean + 95% CI.

from pathlib import Path
import pandas as pd

from plot_utils import (
    ensure_dir,
    find_first_existing,
    plot_numeric_ci,
)

OUTPUT_DIR = ensure_dir("figures_sensitivity")

C25B_CANDIDATES = [
    Path("pricing_sensitivity/analysis/c25b_multiseed.csv"),
    Path("c25b_sensitivity/analysis/c25b_multiseed.csv"),
    Path("analysis/c25b_multiseed.csv"),
    Path("c25b_multiseed.csv"),
]

C26_CANDIDATES = [
    Path("pricing_stability/analysis/c26_stability_multiseed.csv"),
    Path("c26_stability/analysis/c26_stability_multiseed.csv"),
    Path("analysis/c26_stability_multiseed.csv"),
    Path("c26_stability_multiseed.csv"),
]

SENSITIVITY_METRICS = {
    "avg_offered_price": ("Average Offered Price", False),
    "avg_price_multiplier": ("Average Price Multiplier", False),
    "eligibility_rate": ("Eligibility Rate", True),
    "avg_utility": ("Average Utility", False),
    "allocation_rate": ("Allocation Rate", True),
    "conditional_allocation_rate": ("Conditional Allocation Rate", True),
    "unique_selected_vehicles": ("Unique Selected Vehicles", False),
    "jain_selected_vehicles": ("Jain Fairness Index", False),
    "task_success_rate": ("Task Success Rate", True),
}

STABILITY_METRICS = {
    "avg_multiplier": ("Average Price Multiplier", False),
    "avg_price_multiplier": ("Average Price Multiplier", False),
    "multiplier_cv": ("Multiplier Coefficient of Variation", False),
    "mean_abs_multiplier_delta": ("Mean Absolute Multiplier Change", False),
    "p95_abs_multiplier_delta": ("P95 Absolute Multiplier Change", False),
    "max_abs_multiplier_delta": ("Maximum Absolute Multiplier Change", False),
    "pmax_saturation_rate": ("Pmax Saturation Rate", True),
    "eligibility_rate": ("Eligibility Rate", True),
    "conditional_allocation_rate": ("Conditional Allocation Rate", True),
    "task_success_rate": ("Task Success Rate", True),
}

def plot_if_available(
    df,
    metric,
    ylabel,
    percent,
    x_col,
    xlabel,
    title,
    filename,
    invert_x=False,
):
    required = [
        f"{metric}_mean",
        f"{metric}_ci95_low",
        f"{metric}_ci95_high",
    ]

    if not all(col in df.columns for col in required):
        return False

    plot_numeric_ci(
        df,
        x_col=x_col,
        mean_col=f"{metric}_mean",
        low_col=f"{metric}_ci95_low",
        high_col=f"{metric}_ci95_high",
        ylabel=ylabel,
        title=title,
        output_dir=OUTPUT_DIR,
        filename=filename,
        percent=percent,
        xlabel=xlabel,
        invert_x=invert_x,
    )

    return True

# ============================================================
# C25b
# ============================================================

path = find_first_existing(C25B_CANDIDATES)

if path is not None:
    print(f"[OK] C25b: {path}")
    df = pd.read_csv(path).copy()

    if "block" not in df.columns:
        raise ValueError(
            "C25b summary must contain a 'block' column "
            "with alpha/beta."
        )

    for block, x_col, xlabel in [
        ("alpha", "alpha", r"$\alpha$"),
        ("beta", "beta", r"$\beta$"),
    ]:
        sub = df[
            df["block"].astype(str).str.lower() == block
        ].copy()

        if len(sub) == 0 or x_col not in sub.columns:
            continue

        for metric, (ylabel, percent) in SENSITIVITY_METRICS.items():
            plot_if_available(
                sub,
                metric,
                ylabel,
                percent,
                x_col,
                xlabel,
                f"Dynamic-Pricing Sensitivity to {xlabel}: {ylabel}",
                f"c25b_{block}_{metric}",
            )
else:
    print("[SKIP] C25b summary not found.")

# ============================================================
# C26
# ============================================================

path = find_first_existing(C26_CANDIDATES)

if path is not None:
    print(f"[OK] C26: {path}")
    df = pd.read_csv(path).copy()

    if "request_interval" not in df.columns:
        raise ValueError(
            "C26 summary must contain request_interval."
        )

    plotted_bases = set()

    for metric, (ylabel, percent) in STABILITY_METRICS.items():
        # Avoid plotting both aliases when both exist.
        conceptual = (
            "multiplier"
            if metric in ["avg_multiplier", "avg_price_multiplier"]
            else metric
        )

        if conceptual in plotted_bases:
            continue

        if plot_if_available(
            df,
            metric,
            ylabel,
            percent,
            "request_interval",
            "Request Interval (simulation steps)",
            f"Dynamic-Pricing Stability: {ylabel}",
            f"c26_{metric}",
            invert_x=True,
        ):
            plotted_bases.add(conceptual)

    print(
        "[NOTE] Request-interval axis is inverted: "
        "moving right represents increasing offered load."
    )
else:
    print("[SKIP] C26 summary not found.")

print(f"[DONE] Figures: {OUTPUT_DIR}")
