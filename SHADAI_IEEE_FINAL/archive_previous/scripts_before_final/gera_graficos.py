# gera_graficos.py
# Selection-strategy comparison.
#
# Updated scope:
# SHADAI, Greedy, No Reputation, REVS-based and optionally Random.
# Primary figures use multi-run mean + 95% CI.
#
# IMPORTANT:
# - SHADAI raw score is NOT compared against REVS score.
# - SHADAI task-distance is NOT compared against REVS V2V distance.

from pathlib import Path
import pandas as pd

from plot_utils import (
    ensure_dir,
    find_first_existing,
    plot_categorical_ci,
)

OUTPUT_DIR = ensure_dir("figures_strategies")

ORDER = [
    "SHADAI",
    "Greedy",
    "No Reputation",
    "REVS-based",
    "Random",
]

# Canonical file can be supplied directly if desired.
CANONICAL = Path("analysis/strategy_comparison_multiseed.csv")

# Existing validated experiment summaries.
C24_CANDIDATES = [
    Path("c24_revs_benchmark/analysis/c24_shadai_vs_revs_multiseed.csv"),
]

C22_CANDIDATES = [
    Path("greedy_ablation/analysis/c22_shadai_vs_greedy_multiseed.csv"),
    Path("analysis/c22_shadai_vs_greedy_multiseed.csv"),
    Path("c22_shadai_vs_greedy_multiseed.csv"),
]

C23_CANDIDATES = [
    Path("norep_ablation/analysis/c23_shadai_vs_norep_multiseed.csv"),
    Path("analysis/c23_shadai_vs_norep_multiseed.csv"),
    Path("c23_shadai_vs_norep_multiseed.csv"),
]

METRICS = {
    "allocation_rate": ("Allocation Rate", True),
    "conditional_allocation_ge2": ("Conditional Allocation Rate", True),
    "conditional_allocation_rate": ("Conditional Allocation Rate", True),
    "avg_selected_reputation_norm": ("Selected Reputation", False),
    "avg_reputation_norm": ("Selected Reputation", False),
    "avg_available_energy_at_selection": ("Available Energy at Selection", False),
    "unique_selected_vehicles": ("Unique Selected Vehicles", False),
    "tasks_per_selected_vehicle": ("Tasks per Selected Vehicle", False),
    "jain_selected_vehicles": ("Jain Fairness Index", False),
    "task_success_rate": ("Task Success Rate", True),
}

ALIASES = {
    "shadai": "SHADAI",
    "greedy": "Greedy",
    "norep": "No Reputation",
    "no_rep": "No Reputation",
    "no_reputation": "No Reputation",
    "no reputation": "No Reputation",
    "revs": "REVS-based",
    "revs-based": "REVS-based",
    "revs_based": "REVS-based",
    "random": "Random",
}

def normalize_method(value):
    text = str(value).strip()
    return ALIASES.get(text.lower(), text)

def normalize_summary(df):
    df = df.copy()

    method_col = None

    for candidate in [
        "method",
        "mode",
        "policy",
        "strategy",
        "variant",
    ]:
        if candidate in df.columns:
            method_col = candidate
            break

    if method_col is None:
        raise ValueError("No method/mode/policy/strategy/variant column found.")

    df["method"] = df[method_col].map(normalize_method)
    return df

def load_candidate(paths):
    path = find_first_existing(paths)

    if path is None:
        return None

    print(f"[OK] {path}")
    return normalize_summary(pd.read_csv(path))

def merge_sources():
    if CANONICAL.exists():
        print(f"[OK] canonical: {CANONICAL}")
        return normalize_summary(pd.read_csv(CANONICAL))

    sources = []

    for paths in [
        C22_CANDIDATES,
        C23_CANDIDATES,
        C24_CANDIDATES,
    ]:
        df = load_candidate(paths)

        if df is not None:
            sources.append(df)

    if not sources:
        raise FileNotFoundError(
            "No strategy summary found. Put the C22/C23/C24 "
            "analysis outputs in their expected folders or create "
            "analysis/strategy_comparison_multiseed.csv."
        )

    # Concatenate, then retain one row per method.
    combined = pd.concat(sources, ignore_index=True, sort=False)

    priority = {
        "SHADAI": 0,
        "Greedy": 1,
        "No Reputation": 2,
        "REVS-based": 3,
        "Random": 4,
    }

    combined["_priority"] = combined["method"].map(
        lambda x: priority.get(x, 999)
    )

    # Multiple source files contain SHADAI. Keep the last complete row
    # for each method because all finalized comparisons use the same
    # baseline configuration and 30 seeds.
    combined = (
        combined.sort_values(["_priority"])
        .drop_duplicates("method", keep="last")
        .drop(columns="_priority")
    )

    return combined

summary = merge_sources()
summary.to_csv(
    OUTPUT_DIR / "strategy_comparison_consolidated.csv",
    index=False,
)

for metric, (ylabel, percent) in METRICS.items():
    mean_col = f"{metric}_mean"
    low_col = f"{metric}_ci95_low"
    high_col = f"{metric}_ci95_high"

    if not all(
        col in summary.columns
        for col in [mean_col, low_col, high_col]
    ):
        continue

    subset = summary[
        summary[mean_col].notna()
    ].copy()

    if len(subset) < 2:
        continue

    plot_categorical_ci(
        subset,
        group_col="method",
        mean_col=mean_col,
        low_col=low_col,
        high_col=high_col,
        ylabel=ylabel,
        title=f"{ylabel} Across Selection Strategies",
        output_dir=OUTPUT_DIR,
        filename=f"{metric}_strategy_comparison",
        order=ORDER,
        percent=percent,
    )

print("[NOTE] Raw SHADAI and REVS scores are intentionally excluded.")
print("[NOTE] SHADAI task-distance and REVS V2V-distance are intentionally excluded.")
print(f"[DONE] Figures: {OUTPUT_DIR}")
