# analisar_scalability_final.py
# Multi-run analysis for the final 100/400/800 scalability campaign.

import math
import os
from pathlib import Path

import pandas as pd

try:
    from scipy.stats import t
except Exception:
    t = None

INPUT_DIR = Path("scalability_final")
ANALYSIS_DIR = INPUT_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

VEHICLE_CAPS = [100, 400, 800]
SEEDS = list(range(1, 31))

PER_SEED_FILE = ANALYSIS_DIR / "scalability_per_seed.csv"
AGG_FILE = ANALYSIS_DIR / "scalability_multiseed.csv"
TEMPORAL_FILE = ANALYSIS_DIR / "scalability_temporal_ci95.csv"


def ci95(values):
    s = pd.Series(values, dtype="float64").dropna()
    n = len(s)

    if n == 0:
        return n, float("nan"), float("nan"), float("nan"), float("nan")

    mean = s.mean()

    if n == 1:
        return n, mean, float("nan"), float("nan"), float("nan")

    sd = s.std(ddof=1)
    se = sd / math.sqrt(n)
    crit = t.ppf(0.975, n - 1) if t is not None else 1.96

    return n, mean, sd, mean - crit * se, mean + crit * se


def bool_series(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1"])


def exp(max_vehicles, seed):
    return f"scalability_v{max_vehicles}_seed_{seed:02d}"


def paths(max_vehicles, seed):
    e = exp(max_vehicles, seed)
    return {
        "metrics": INPUT_DIR / f"metricas_shadai_{max_vehicles}_{e}.csv",
        "requests": INPUT_DIR / f"requests_shadai_{max_vehicles}_{e}.csv",
        "outcomes": INPUT_DIR / f"outcomes_shadai_{max_vehicles}_{e}.csv",
    }


per_seed = []
temporal_seed = []
missing = []

for max_vehicles in VEHICLE_CAPS:
    for seed in SEEDS:
        p = paths(max_vehicles, seed)

        if not all(x.exists() for x in p.values()):
            missing.append((max_vehicles, seed))
            continue

        metrics = pd.read_csv(p["metrics"])
        requests = pd.read_csv(p["requests"])
        outcomes = pd.read_csv(p["outcomes"])

        allocated = bool_series(requests["allocated"])
        active = pd.to_numeric(requests["active_vehicles"], errors="coerce").fillna(0)

        row = {
            "max_vehicles": max_vehicles,
            "seed": seed,
            "n_requests": len(requests),
            "allocation_rate": allocated.mean() if len(requests) else float("nan"),
            "conditional_allocation_rate": (
                allocated[active > 0].mean()
                if (active > 0).any()
                else float("nan")
            ),
            "avg_score": pd.to_numeric(metrics["score"], errors="coerce").mean(),
            "avg_utility": pd.to_numeric(metrics["utilidade"], errors="coerce").mean(),
            "avg_reputation": pd.to_numeric(metrics["reputacao"], errors="coerce").mean(),
            "avg_selected_reputation_norm": pd.to_numeric(metrics["reputacao_norm"], errors="coerce").mean(),
            "avg_available_energy_at_selection": pd.to_numeric(metrics["energia"], errors="coerce").mean(),
            "unique_selected_vehicles": metrics["veiculo"].astype(str).nunique(),
            "task_success_rate": (
                bool_series(outcomes["success"]).mean()
                if len(outcomes)
                else float("nan")
            ),
        }

        counts = metrics["veiculo"].astype(str).value_counts()

        if len(counts):
            denom = len(counts) * (counts.astype(float) ** 2).sum()
            row["jain_selected_vehicles"] = (
                counts.sum() ** 2 / denom
                if denom > 0
                else float("nan")
            )
        else:
            row["jain_selected_vehicles"] = float("nan")

        per_seed.append(row)

        # Original temporal metrics, first averaged within seed+step,
        # then aggregated across independent seeds below.
        cols = ["step", "score", "utilidade", "reputacao"]

        if all(c in metrics.columns for c in cols):
            tmp = (
                metrics[cols]
                .assign(
                    step=lambda d: pd.to_numeric(d["step"], errors="coerce"),
                    score=lambda d: pd.to_numeric(d["score"], errors="coerce"),
                    utilidade=lambda d: pd.to_numeric(d["utilidade"], errors="coerce"),
                    reputacao=lambda d: pd.to_numeric(d["reputacao"], errors="coerce"),
                )
                .dropna(subset=["step"])
                .groupby("step", as_index=False)
                .mean(numeric_only=True)
            )

            tmp["max_vehicles"] = max_vehicles
            tmp["seed"] = seed
            temporal_seed.append(tmp)


per_seed_df = pd.DataFrame(per_seed)
per_seed_df.to_csv(PER_SEED_FILE, index=False)

metrics_to_aggregate = [
    "allocation_rate",
    "conditional_allocation_rate",
    "avg_score",
    "avg_utility",
    "avg_reputation",
    "avg_selected_reputation_norm",
    "avg_available_energy_at_selection",
    "unique_selected_vehicles",
    "jain_selected_vehicles",
    "task_success_rate",
]

agg_rows = []

for max_vehicles, sub in per_seed_df.groupby("max_vehicles"):
    row = {
        "max_vehicles": max_vehicles,
        "n_runs": len(sub),
    }

    for metric in metrics_to_aggregate:
        n, mean, sd, low, high = ci95(sub[metric])
        row[f"{metric}_mean"] = mean
        row[f"{metric}_sd"] = sd
        row[f"{metric}_ci95_low"] = low
        row[f"{metric}_ci95_high"] = high

    agg_rows.append(row)

agg = pd.DataFrame(agg_rows).sort_values("max_vehicles")
agg.to_csv(AGG_FILE, index=False)

temporal_rows = []

if temporal_seed:
    temporal_all = pd.concat(temporal_seed, ignore_index=True)

    for (max_vehicles, step), sub in temporal_all.groupby(["max_vehicles", "step"]):
        for metric in ["score", "utilidade", "reputacao"]:
            n, mean, sd, low, high = ci95(sub[metric])

            temporal_rows.append({
                "max_vehicles": max_vehicles,
                "step": step,
                "metric": metric,
                "n_runs": n,
                "mean": mean,
                "sd": sd,
                "ci95_low": low,
                "ci95_high": high,
            })

pd.DataFrame(temporal_rows).to_csv(TEMPORAL_FILE, index=False)

print("=" * 110)
print("FINAL SCALABILITY ANALYSIS")
print("=" * 110)
print(agg.to_string(index=False))
print(f"\nMissing runs: {len(missing)}")
if missing:
    print(missing[:20])
print(f"\n{PER_SEED_FILE}")
print(AGG_FILE)
print(TEMPORAL_FILE)
