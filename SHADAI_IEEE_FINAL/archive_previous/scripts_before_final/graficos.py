# graficos.py
# Scalability / traffic-density figures.
# Modernized version of the original script.
#
# Main result: multi-run mean + 95% CI when per-seed files exist.
# Compatibility: also preserves the original temporal curves.

from pathlib import Path
import re
import pandas as pd

from plot_utils import (
    aggregate_by_group,
    ensure_dir,
    plot_categorical_ci,
    plot_multiline_temporal,
)

OUTPUT_DIR = ensure_dir("figures_scalability")

SCENARIOS = {
    "100 Vehicles": {
        "vehicles": 100,
        "legacy": Path("metricas_shadai_100.csv"),
    },
    "400 Vehicles": {
        "vehicles": 400,
        "legacy": Path("metricas_shadai_400.csv"),
    },
    "800 Vehicles": {
        "vehicles": 800,
        "legacy": Path("metricas_shadai_800.csv"),
    },
}

WINDOW = 20

TEMPORAL_METRICS = {
    "score": (
        "Average Score",
        "Score Evolution Across Traffic Densities",
        "score_comparison",
    ),
    "utilidade": (
        "Average Utility",
        "Utility Evolution Across Traffic Densities",
        "utility_comparison",
    ),
    "reputacao": (
        "Average Reputation",
        "Reputation Evolution Across Traffic Densities",
        "reputation_comparison",
    ),
}

PER_RUN_METRICS = [
    "score",
    "utilidade",
    "reputacao",
]

LABELS = {
    "score": "Average Score",
    "utilidade": "Average Utility",
    "reputacao": "Average Reputation",
}

def extract_seed(path):
    match = re.search(r"seed[_-]?(\d+)", path.stem, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None

def find_multirun_files(vehicle_count):
    patterns = [
        f"metricas_shadai_{vehicle_count}_*seed*.csv",
        f"**/metricas_shadai_{vehicle_count}_*seed*.csv",
        f"**/metricas_shadai_{vehicle_count}_*.csv",
    ]

    found = []
    seen = set()

    for pattern in patterns:
        for path in Path(".").glob(pattern):
            if path.name == f"metricas_shadai_{vehicle_count}.csv":
                continue

            seed = extract_seed(path)

            if seed is None:
                continue

            key = str(path.resolve())

            if key not in seen:
                found.append(path)
                seen.add(key)

    return sorted(found)

def build_per_seed_summary():
    rows = []

    for scenario, cfg in SCENARIOS.items():
        files = find_multirun_files(cfg["vehicles"])

        for path in files:
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                print(f"[SKIP] {path}: {exc}")
                continue

            seed = extract_seed(path)

            row = {
                "scenario": scenario,
                "vehicles": cfg["vehicles"],
                "seed": seed,
                "source_file": str(path),
            }

            usable = False

            for metric in PER_RUN_METRICS:
                if metric in df.columns:
                    row[metric] = pd.to_numeric(
                        df[metric],
                        errors="coerce",
                    ).mean()
                    usable = True

            if usable:
                rows.append(row)

    return pd.DataFrame(rows)

# ============================================================
# 1) ORIGINAL TEMPORAL FIGURES (backward compatibility)
# ============================================================

legacy_data = {}

for scenario, cfg in SCENARIOS.items():
    if cfg["legacy"].exists():
        legacy_data[scenario] = pd.read_csv(cfg["legacy"])
        print(f"[OK] legacy: {cfg['legacy']}")
    else:
        legacy_data[scenario] = None

for column, (ylabel, title, filename) in TEMPORAL_METRICS.items():
    plot_multiline_temporal(
        legacy_data,
        column=column,
        ylabel=ylabel,
        title=title,
        output_dir=OUTPUT_DIR,
        filename=filename,
        window=WINDOW,
    )

# ============================================================
# 2) MULTI-RUN FIGURES WITH 95% CI
# ============================================================

per_seed = build_per_seed_summary()

if not per_seed.empty:
    per_seed.to_csv(
        OUTPUT_DIR / "scalability_per_seed.csv",
        index=False,
    )

    agg = aggregate_by_group(
        per_seed,
        ["scenario", "vehicles"],
        PER_RUN_METRICS,
    )

    agg.to_csv(
        OUTPUT_DIR / "scalability_multiseed_summary.csv",
        index=False,
    )

    order = [
        "100 Vehicles",
        "400 Vehicles",
        "800 Vehicles",
    ]

    for metric in PER_RUN_METRICS:
        mean_col = f"{metric}_mean"

        if mean_col not in agg.columns:
            continue

        plot_categorical_ci(
            agg,
            group_col="scenario",
            mean_col=mean_col,
            low_col=f"{metric}_ci95_low",
            high_col=f"{metric}_ci95_high",
            ylabel=LABELS[metric],
            title=f"{LABELS[metric]} Across Traffic Densities",
            output_dir=OUTPUT_DIR,
            filename=f"{metric}_comparison_multirun",
            order=order,
        )

    print(
        f"[OK] Multi-run scalability summary with "
        f"{per_seed['seed'].nunique()} detected seeds."
    )
else:
    print(
        "[INFO] No per-seed density files were detected. "
        "Legacy temporal figures were preserved."
    )

print(f"[DONE] Figures: {OUTPUT_DIR}")
