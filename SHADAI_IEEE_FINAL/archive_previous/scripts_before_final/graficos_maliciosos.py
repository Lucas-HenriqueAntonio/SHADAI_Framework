# graficos_maliciosos.py
# Security/adversarial figures.
#
# Updated replacement for the original single-run attack script.
# C17: deliberate service-failure attack versus malicious ratio.
# C18: candidacy-flood attack versus flood intensity.
#
# Primary figures use multi-run mean + 95% CI.

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from plot_utils import (
    ensure_dir,
    find_first_existing,
    plot_numeric_ci,
    save_figure,
)

OUTPUT_DIR = ensure_dir("figures_security")

C17_CANDIDATES = [
    Path("malicious_ablation/analysis/c17_malicious_ratio_multiseed.csv"),
    Path("malicious_ablation/analysis/c17_malicious_multiseed.csv"),
    Path("analysis/c17_malicious_ratio_multiseed.csv"),
    Path("c17_malicious_ratio_multiseed.csv"),
]

C18_CANDIDATES = [
    Path("flood_ablation/analysis/c18_flood_multiseed.csv"),
    Path("flood_ablation/analysis/c18_flood_intensity_multiseed.csv"),
    Path("analysis/c18_flood_multiseed.csv"),
    Path("c18_flood_multiseed.csv"),
]

RATIO_COLUMNS = [
    "malicious_ratio",
    "configured_malicious_ratio",
    "ratio",
]

FLOOD_COLUMNS = [
    "flood_intensity",
    "intensity",
]

METRIC_ALIASES = {
    "task_success_rate": [
        "task_success_rate",
        "service_success_rate",
        "success_rate",
    ],
    "honest_success_rate": [
        "honest_success_rate",
        "honest_task_success_rate",
    ],
    "malicious_selection_rate": [
        "malicious_selection_rate",
        "malicious_selected_rate",
        "malicious_selection_share",
        "selected_malicious_rate",
    ],
    "exposure_ratio": [
        "exposure_ratio",
        "malicious_exposure_ratio",
        "exposure",
    ],
    "exploration_malicious_rate": [
        "exploration_malicious_rate",
        "exploration_selected_malicious_rate",
        "exploration_malicious_selection_rate",
    ],
    "exploitation_malicious_rate": [
        "exploitation_malicious_rate",
        "exploitation_selected_malicious_rate",
        "exploitation_malicious_selection_rate",
    ],
    "artificial_candidates": [
        "artificial_candidates",
        "avg_artificial_candidates",
        "artificial_candidate_count",
    ],
    "malicious_pool_share": [
        "malicious_pool_share",
        "malicious_candidate_pool_share",
    ],
    "allocation_rate": [
        "allocation_rate",
    ],
}

LABELS = {
    "task_success_rate": ("Task Success Rate", True),
    "honest_success_rate": ("Honest-Provider Success Rate", True),
    "malicious_selection_rate": ("Malicious Selection Rate", True),
    "exposure_ratio": ("Exposure Ratio", False),
    "exploration_malicious_rate": ("Exploration Malicious Selection", True),
    "exploitation_malicious_rate": ("Exploitation Malicious Selection", True),
    "artificial_candidates": ("Artificial Candidacies per Request", False),
    "malicious_pool_share": ("Malicious Candidate-Pool Share", True),
    "allocation_rate": ("Allocation Rate", True),
}

def first_col(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None

def metric_base(df, aliases):
    for base in aliases:
        required = [
            f"{base}_mean",
            f"{base}_ci95_low",
            f"{base}_ci95_high",
        ]

        if all(col in df.columns for col in required):
            return base

    return None

def load_first(candidates, label):
    path = find_first_existing(candidates)

    if path is None:
        print(f"[SKIP] {label}: no summary file found.")
        return None

    print(f"[OK] {label}: {path}")
    return pd.read_csv(path)

def plot_attack_metrics(df, x_col, x_label, prefix, title_prefix):
    if df is None:
        return

    for canonical, aliases in METRIC_ALIASES.items():
        base = metric_base(df, aliases)

        if base is None:
            continue

        ylabel, percent = LABELS[canonical]

        plot_numeric_ci(
            df,
            x_col=x_col,
            mean_col=f"{base}_mean",
            low_col=f"{base}_ci95_low",
            high_col=f"{base}_ci95_high",
            ylabel=ylabel,
            title=f"{title_prefix}: {ylabel}",
            output_dir=OUTPUT_DIR,
            filename=f"{prefix}_{canonical}",
            percent=percent,
            xlabel=x_label,
        )

def combined_explore_exploit(df, x_col):
    exp_base = metric_base(
        df,
        METRIC_ALIASES["exploration_malicious_rate"],
    )

    exploit_base = metric_base(
        df,
        METRIC_ALIASES["exploitation_malicious_rate"],
    )

    if exp_base is None or exploit_base is None:
        return

    data = df.copy()
    data[x_col] = pd.to_numeric(data[x_col], errors="coerce")
    data = data.sort_values(x_col)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    for base, label, style in [
        (exp_base, "Exploration", "-"),
        (exploit_base, "Exploitation", "--"),
    ]:
        x = data[x_col].to_numpy(dtype=float)
        mean = data[f"{base}_mean"].to_numpy(dtype=float) * 100.0
        low = data[f"{base}_ci95_low"].to_numpy(dtype=float) * 100.0
        high = data[f"{base}_ci95_high"].to_numpy(dtype=float) * 100.0

        ax.errorbar(
            x,
            mean,
            yerr=[mean - low, high - mean],
            fmt="o",
            linestyle=style,
            capsize=4,
            linewidth=2.0,
            label=label,
        )

    ax.set_xlabel("Flood intensity (additional candidacies)")
    ax.set_ylabel("Malicious Selection Rate (%)")
    ax.set_title(
        "Flood Attack: Exploration vs. Exploitation",
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    fig.tight_layout()
    save_figure(
        fig,
        OUTPUT_DIR,
        "c18_exploration_vs_exploitation",
    )

# ============================================================
# C17
# ============================================================

c17 = load_first(
    C17_CANDIDATES,
    "C17 malicious-ratio",
)

if c17 is not None:
    ratio_col = first_col(
        c17,
        RATIO_COLUMNS,
    )

    if ratio_col is None:
        print(
            "[SKIP] C17: malicious-ratio column not found."
        )
    else:
        # If ratios are expressed as fractions, plots remain in fraction
        # on x unless converted here.
        c17 = c17.copy()
        x = pd.to_numeric(
            c17[ratio_col],
            errors="coerce",
        )

        if x.max() <= 1.0:
            c17["_malicious_percent"] = x * 100.0
        else:
            c17["_malicious_percent"] = x

        plot_attack_metrics(
            c17,
            x_col="_malicious_percent",
            x_label="Malicious vehicles (%)",
            prefix="c17",
            title_prefix="Deliberate Service-Failure Attack",
        )

# ============================================================
# C18
# ============================================================

c18 = load_first(
    C18_CANDIDATES,
    "C18 flood-intensity",
)

if c18 is not None:
    flood_col = first_col(
        c18,
        FLOOD_COLUMNS,
    )

    if flood_col is None:
        print(
            "[SKIP] C18: flood-intensity column not found."
        )
    else:
        plot_attack_metrics(
            c18,
            x_col=flood_col,
            x_label="Flood intensity (additional candidacies)",
            prefix="c18",
            title_prefix="Candidacy Flood Attack",
        )

        combined_explore_exploit(
            c18,
            flood_col,
        )

print(f"[DONE] Figures: {OUTPUT_DIR}")
