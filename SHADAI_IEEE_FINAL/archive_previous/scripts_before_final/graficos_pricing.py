# graficos_pricing.py
# C25/C27 - static vs dynamic pricing figures.
# Uses multi-run means and 95% confidence intervals.

from pathlib import Path
import pandas as pd

from plot_utils import (
    ensure_dir,
    find_first_existing,
    plot_categorical_ci,
)

OUTPUT_DIR = ensure_dir("figures_pricing")

C25_CANDIDATES = [
    Path("pricing_ablation/analysis/c25_pricing_multiseed.csv"),
    Path("c25_pricing/analysis/c25_pricing_multiseed.csv"),
    Path("analysis/c25_pricing_multiseed.csv"),
    Path("c25_pricing_multiseed.csv"),
]

ORDER = [
    "Static",
    "Dynamic",
]

METRICS = {
    "avg_offered_price": ("Average Offered Price", False),
    "avg_price_multiplier": ("Average Price Multiplier", False),
    "eligibility_rate": ("Eligibility Rate", True),
    "avg_utility": ("Average Utility", False),
    "allocation_rate": ("Allocation Rate", True),
    "conditional_allocation_rate": ("Conditional Allocation Rate", True),
    "avg_selected_reputation_norm": ("Selected Reputation", False),
    "avg_available_energy_at_selection": ("Available Energy at Selection", False),
    "unique_selected_vehicles": ("Unique Selected Vehicles", False),
    "jain_selected_vehicles": ("Jain Fairness Index", False),
    "task_success_rate": ("Task Success Rate", True),
}

path = find_first_existing(C25_CANDIDATES)

if path is None:
    raise FileNotFoundError(
        "C25 multi-seed summary not found. "
        "Adjust C25_CANDIDATES at the top of the script."
    )

print(f"[OK] {path}")
df = pd.read_csv(path).copy()

mode_col = None

for candidate in [
    "pricing_mode",
    "mode",
    "variant",
]:
    if candidate in df.columns:
        mode_col = candidate
        break

if mode_col is None:
    raise ValueError(
        "No pricing_mode/mode/variant column found."
    )

df["pricing_label"] = (
    df[mode_col]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "static": "Static",
        "dynamic": "Dynamic",
    })
    .fillna(df[mode_col].astype(str))
)

for metric, (ylabel, percent) in METRICS.items():
    required = [
        f"{metric}_mean",
        f"{metric}_ci95_low",
        f"{metric}_ci95_high",
    ]

    if not all(col in df.columns for col in required):
        continue

    subset = df[
        df[f"{metric}_mean"].notna()
    ].copy()

    plot_categorical_ci(
        subset,
        group_col="pricing_label",
        mean_col=f"{metric}_mean",
        low_col=f"{metric}_ci95_low",
        high_col=f"{metric}_ci95_high",
        ylabel=ylabel,
        title=f"Static vs. Dynamic Pricing: {ylabel}",
        output_dir=OUTPUT_DIR,
        filename=f"c25_{metric}",
        order=ORDER,
        percent=percent,
    )

df.to_csv(
    OUTPUT_DIR / "c25_pricing_source_used.csv",
    index=False,
)

print(f"[DONE] Figures: {OUTPUT_DIR}")
