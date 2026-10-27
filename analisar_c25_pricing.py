# analisar_c25_pricing.py

import math
import os

import pandas as pd

try:
    from scipy.stats import t
except ImportError:
    t = None


INPUT_DIR = "pricing_ablation"
OUTPUT_DIR = os.path.join(INPUT_DIR, "analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEEDS = list(range(1, 31))
MODES = ["static", "dynamic"]

PER_SEED_FILE = os.path.join(OUTPUT_DIR, "c25_pricing_por_seed.csv")
AGG_FILE = os.path.join(OUTPUT_DIR, "c25_pricing_multiseed.csv")
PAIRED_FILE = os.path.join(
    OUTPUT_DIR,
    "c25_pricing_paired_dynamic_minus_static.csv",
)
LATEX_FILE = os.path.join(
    OUTPUT_DIR,
    "tabela_c25_pricing_tnsm.tex",
)


def to_bool(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1"])
    )


def ci95(values):
    s = pd.Series(values, dtype="float64").dropna()
    n = len(s)

    if n == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )

    mean = s.mean()

    if n == 1:
        return (
            mean,
            float("nan"),
            float("nan"),
            float("nan"),
        )

    sd = s.std(ddof=1)
    se = sd / math.sqrt(n)

    critical = (
        t.ppf(0.975, n - 1)
        if t is not None
        else 1.96
    )

    margin = critical * se

    return (
        mean,
        sd,
        mean - margin,
        mean + margin,
    )


def jain_index(values):
    s = pd.Series(values, dtype="float64")
    s = s[s > 0]
    n = len(s)

    if n == 0:
        return float("nan")

    denominator = n * (s ** 2).sum()

    if denominator == 0:
        return float("nan")

    return (s.sum() ** 2) / denominator


def experiment_name(mode, seed):
    return f"c25_{mode}_seed_{seed:02d}"


def process_run(mode, seed):
    experiment = experiment_name(mode, seed)

    metrics_path = os.path.join(
        INPUT_DIR,
        f"metricas_shadai_400_{experiment}.csv",
    )

    requests_path = os.path.join(
        INPUT_DIR,
        f"requests_shadai_400_{experiment}.csv",
    )

    eligibility_path = os.path.join(
        INPUT_DIR,
        f"eligibilidade_shadai_400_{experiment}.csv",
    )

    outcomes_path = os.path.join(
        INPUT_DIR,
        f"outcomes_shadai_400_{experiment}.csv",
    )

    for path in [
        metrics_path,
        requests_path,
        eligibility_path,
        outcomes_path,
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    metrics = pd.read_csv(metrics_path)
    requests = pd.read_csv(requests_path)
    eligibility = pd.read_csv(eligibility_path)
    outcomes = pd.read_csv(outcomes_path)

    required_request_columns = [
        "pricing_mode",
        "base_price",
        "offered_price",
        "demand_pressure",
        "supply_scarcity",
        "price_multiplier",
        "allocated",
        "active_vehicles",
    ]

    for column in required_request_columns:
        if column not in requests.columns:
            raise RuntimeError(
                f"Missing C25 column '{column}' in {requests_path}"
            )

    observed_modes = set(
        requests["pricing_mode"]
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )

    if observed_modes != {mode}:
        raise RuntimeError(
            f"Unexpected pricing mode in {requests_path}: "
            f"{observed_modes}"
        )

    total_requests = len(requests)

    active = pd.to_numeric(
        requests["active_vehicles"],
        errors="coerce",
    ).fillna(0)

    active_requests = int((active > 0).sum())

    allocated = to_bool(requests["allocated"])
    allocations = int(allocated.sum())

    allocation_rate = (
        allocations / total_requests
        if total_requests > 0
        else float("nan")
    )

    conditional_allocation_rate = (
        allocations / active_requests
        if active_requests > 0
        else float("nan")
    )

    base_price = pd.to_numeric(
        requests["base_price"],
        errors="coerce",
    )

    offered_price = pd.to_numeric(
        requests["offered_price"],
        errors="coerce",
    )

    price_multiplier = pd.to_numeric(
        requests["price_multiplier"],
        errors="coerce",
    )

    demand_pressure = pd.to_numeric(
        requests["demand_pressure"],
        errors="coerce",
    )

    supply_scarcity = pd.to_numeric(
        requests["supply_scarcity"],
        errors="coerce",
    )

    price_increase_rate = (
        offered_price > base_price + 1e-12
    ).mean()

    max_offered_price = (
        offered_price.max()
        if len(offered_price) > 0
        else float("nan")
    )

    eligibility_rate = pd.to_numeric(
        eligibility["eligibility_rate"],
        errors="coerce",
    ).mean()

    avg_utility = pd.to_numeric(
        eligibility["avg_utility"],
        errors="coerce",
    ).mean()

    if len(metrics) > 0:
        avg_score = pd.to_numeric(
            metrics["score"],
            errors="coerce",
        ).mean()

        avg_rep = pd.to_numeric(
            metrics["reputacao_norm"],
            errors="coerce",
        ).mean()

        avg_energy_norm = pd.to_numeric(
            metrics["energia_norm"],
            errors="coerce",
        ).mean()

        avg_prox = pd.to_numeric(
            metrics["proximidade_norm"],
            errors="coerce",
        ).mean()

        avg_distance = pd.to_numeric(
            metrics["distancia"],
            errors="coerce",
        ).mean()

        avg_available_energy = pd.to_numeric(
            metrics["energia"],
            errors="coerce",
        ).mean()

        task_counts = (
            metrics["veiculo"]
            .astype(str)
            .value_counts()
        )

        unique_selected = len(task_counts)
        tasks_per_vehicle = task_counts.mean()
        jain = jain_index(task_counts.values)

    else:
        avg_score = float("nan")
        avg_rep = float("nan")
        avg_energy_norm = float("nan")
        avg_prox = float("nan")
        avg_distance = float("nan")
        avg_available_energy = float("nan")
        unique_selected = 0
        tasks_per_vehicle = float("nan")
        jain = float("nan")

    if len(outcomes) > 0:
        success = to_bool(outcomes["success"])
        task_success_rate = success.mean()
        task_failure_rate = 1.0 - task_success_rate
    else:
        task_success_rate = float("nan")
        task_failure_rate = float("nan")

    return {
        "pricing_mode": mode,
        "seed": seed,

        "total_requests": total_requests,
        "active_requests": active_requests,
        "allocations": allocations,

        "allocation_rate": allocation_rate,
        "conditional_allocation_rate":
            conditional_allocation_rate,

        "avg_base_price": base_price.mean(),
        "avg_offered_price": offered_price.mean(),
        "max_offered_price": max_offered_price,
        "avg_price_multiplier":
            price_multiplier.mean(),

        "avg_demand_pressure":
            demand_pressure.mean(),

        "avg_supply_scarcity":
            supply_scarcity.mean(),

        "price_increase_rate":
            price_increase_rate,

        "eligibility_rate":
            eligibility_rate,

        "avg_utility":
            avg_utility,

        "avg_final_score":
            avg_score,

        "avg_reputation_norm":
            avg_rep,

        "avg_energy_norm":
            avg_energy_norm,

        "avg_proximity_norm":
            avg_prox,

        "avg_distance":
            avg_distance,

        "avg_available_energy_at_selection":
            avg_available_energy,

        "unique_selected_vehicles":
            unique_selected,

        "tasks_per_selected_vehicle":
            tasks_per_vehicle,

        "jain_selected_vehicles":
            jain,

        "task_success_rate":
            task_success_rate,

        "task_failure_rate":
            task_failure_rate,
    }


print()
print("=" * 180)
print("C25 - STATIC VS DYNAMIC PRICING ANALYSIS")
print("=" * 180)
print(
    "Dynamic pricing uses the same RNG streams, eligibility, "
    "score weights, epsilon and cooldown as the static baseline."
)
print(
    "Price is computed before eligibility from request-rate "
    "pressure and active-vehicle scarcity only."
)
print("=" * 180)

rows = []
missing = []

for mode in MODES:
    for seed in SEEDS:
        try:
            rows.append(
                process_run(mode, seed)
            )
        except FileNotFoundError as exc:
            missing.append(
                (
                    mode,
                    seed,
                    str(exc),
                )
            )

df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError("No C25 results found.")

df.to_csv(PER_SEED_FILE, index=False)

print()
print("DATASET COMPLETENESS")
print("-" * 180)

for mode in MODES:
    n = len(
        df[df["pricing_mode"] == mode]
    )
    print(f"{mode:<10}: {n}/30")

if missing:
    print(
        f"\n[WARNING] Missing runs: "
        f"{len(missing)}"
    )

print()
print("C25 SANITY CHECK")
print("=" * 180)

static = df[
    df["pricing_mode"] == "static"
].copy()

dynamic = df[
    df["pricing_mode"] == "dynamic"
].copy()

if (
    static["avg_offered_price"]
    .sub(static["avg_base_price"])
    .abs()
    < 1e-12
).all():
    print(
        "[PASS] Static pricing reproduces base service prices."
    )
else:
    print(
        "[FAIL] Static pricing differs from base service prices."
    )

if (
    dynamic["avg_offered_price"]
    >= dynamic["avg_base_price"] - 1e-12
).all():
    print(
        "[PASS] Dynamic offered prices never fall below base prices."
    )
else:
    print(
        "[FAIL] Dynamic price below base price detected."
    )

if (
    dynamic["max_offered_price"]
    <= 60.0 + 1e-12
).all():
    print(
        "[PASS] Every dynamic price respects Pmax = 60."
    )
else:
    print(
        "[FAIL] Dynamic price exceeds Pmax."
    )

if (
    dynamic["avg_price_multiplier"] > 1.0
).any():
    print(
        "[PASS] Dynamic pricing changes offered prices."
    )
else:
    print(
        "[FAIL] Dynamic pricing never changes prices."
    )

if (
    static["avg_demand_pressure"].abs()
    < 1e-12
).all():
    print(
        "[PASS] Static mode records zero demand pressure."
    )
else:
    print(
        "[FAIL] Static mode has nonzero demand pressure."
    )

if (
    static["avg_supply_scarcity"].abs()
    < 1e-12
).all():
    print(
        "[PASS] Static mode records zero supply scarcity."
    )
else:
    print(
        "[FAIL] Static mode has nonzero supply scarcity."
    )

if (
    dynamic["avg_demand_pressure"].abs()
    < 1e-12
).all():
    print(
        "[PASS] Dynamic C25 baseline isolates supply scarcity "
        "(demand pressure = 0 at interval 20)."
    )
else:
    print(
        "[CHECK] Dynamic demand pressure is nonzero in the "
        "interval-20 baseline."
    )


METRICS = [
    "allocation_rate",
    "conditional_allocation_rate",

    "avg_base_price",
    "avg_offered_price",
    "max_offered_price",
    "avg_price_multiplier",

    "avg_demand_pressure",
    "avg_supply_scarcity",
    "price_increase_rate",

    "eligibility_rate",
    "avg_utility",

    "avg_final_score",
    "avg_reputation_norm",
    "avg_energy_norm",
    "avg_proximity_norm",
    "avg_distance",
    "avg_available_energy_at_selection",

    "unique_selected_vehicles",
    "tasks_per_selected_vehicle",
    "jain_selected_vehicles",

    "task_success_rate",
    "task_failure_rate",
]

agg_rows = []

for mode in MODES:
    subset = df[
        df["pricing_mode"] == mode
    ]

    row = {
        "pricing_mode": mode,
        "n_runs": len(subset),
    }

    for metric in METRICS:
        mean, sd, low, high = ci95(
            subset[metric]
        )

        row[f"{metric}_mean"] = mean
        row[f"{metric}_sd"] = sd
        row[f"{metric}_ci95_low"] = low
        row[f"{metric}_ci95_high"] = high

    agg_rows.append(row)

agg = pd.DataFrame(agg_rows)

agg.to_csv(AGG_FILE, index=False)

print()
print("C25 MULTI-SEED RESULTS")
print("=" * 180)

print(
    f"{'MODE':<10}"
    f"{'PRICE':>10}"
    f"{'MULT':>10}"
    f"{'ELIG%':>10}"
    f"{'UTILITY':>10}"
    f"{'ALLOC%':>10}"
    f"{'COND%':>10}"
    f"{'SCORE':>10}"
    f"{'DIST':>10}"
    f"{'ENERGY':>10}"
    f"{'JAIN':>10}"
)

print("-" * 180)

for _, row in agg.iterrows():
    print(
        f"{row['pricing_mode']:<10}"
        f"{row['avg_offered_price_mean']:>10.2f}"
        f"{row['avg_price_multiplier_mean']:>10.4f}"
        f"{row['eligibility_rate_mean'] * 100:>10.2f}"
        f"{row['avg_utility_mean']:>10.4f}"
        f"{row['allocation_rate_mean'] * 100:>10.2f}"
        f"{row['conditional_allocation_rate_mean'] * 100:>10.2f}"
        f"{row['avg_final_score_mean']:>10.4f}"
        f"{row['avg_distance_mean']:>10.2f}"
        f"{row['avg_available_energy_at_selection_mean']:>10.2f}"
        f"{row['jain_selected_vehicles_mean']:>10.4f}"
    )

paired = pd.merge(
    dynamic,
    static,
    on="seed",
    suffixes=(
        "_dynamic",
        "_static",
    ),
)

PAIRED_METRICS = [
    "avg_offered_price",
    "avg_price_multiplier",

    "eligibility_rate",
    "avg_utility",

    "allocation_rate",
    "conditional_allocation_rate",

    "avg_final_score",
    "avg_reputation_norm",
    "avg_energy_norm",
    "avg_proximity_norm",
    "avg_distance",
    "avg_available_energy_at_selection",

    "unique_selected_vehicles",
    "tasks_per_selected_vehicle",
    "jain_selected_vehicles",

    "task_success_rate",
]

paired_rows = []

print()
print("PAIRED DIFFERENCES: DYNAMIC - STATIC")
print("=" * 180)

for metric in PAIRED_METRICS:
    diff = (
        paired[f"{metric}_dynamic"]
        - paired[f"{metric}_static"]
    )

    mean, sd, low, high = ci95(diff)

    excludes_zero = (
        (low > 0 or high < 0)
        if (
            not pd.isna(low)
            and not pd.isna(high)
        )
        else False
    )

    paired_rows.append(
        {
            "metric": metric,
            "n_pairs": len(diff.dropna()),
            "mean_difference_dynamic_minus_static":
                mean,
            "difference_sd": sd,
            "ci95_low": low,
            "ci95_high": high,
            "ci95_excludes_zero":
                excludes_zero,
        }
    )

    print(
        f"{metric:<42}"
        f" diff={mean:+.5f}"
        f" | CI95=[{low:+.5f}, {high:+.5f}]"
        f" | excludes zero={excludes_zero}"
    )

paired_df = pd.DataFrame(
    paired_rows
)

paired_df.to_csv(
    PAIRED_FILE,
    index=False,
)

static_row = agg[
    agg["pricing_mode"] == "static"
].iloc[0]

dynamic_row = agg[
    agg["pricing_mode"] == "dynamic"
].iloc[0]

print()
print("C25 CORE SUMMARY")
print("=" * 180)

print("Static:")
print(
    f"  average base price           : "
    f"{static_row['avg_base_price_mean']:.2f}"
)
print(
    f"  average offered price        : "
    f"{static_row['avg_offered_price_mean']:.2f}"
)
print(
    f"  eligibility rate             : "
    f"{static_row['eligibility_rate_mean'] * 100:.2f}%"
)
print(
    f"  conditional allocation       : "
    f"{static_row['conditional_allocation_rate_mean'] * 100:.2f}%"
)
print(
    f"  average utility              : "
    f"{static_row['avg_utility_mean']:.4f}"
)

print()
print("Dynamic:")
print(
    f"  average base price           : "
    f"{dynamic_row['avg_base_price_mean']:.2f}"
)
print(
    f"  average offered price        : "
    f"{dynamic_row['avg_offered_price_mean']:.2f}"
)
print(
    f"  average multiplier           : "
    f"{dynamic_row['avg_price_multiplier_mean']:.4f}"
)
print(
    f"  demand pressure              : "
    f"{dynamic_row['avg_demand_pressure_mean']:.4f}"
)
print(
    f"  supply scarcity              : "
    f"{dynamic_row['avg_supply_scarcity_mean']:.4f}"
)
print(
    f"  requests receiving premium   : "
    f"{dynamic_row['price_increase_rate_mean'] * 100:.2f}%"
)
print(
    f"  eligibility rate             : "
    f"{dynamic_row['eligibility_rate_mean'] * 100:.2f}%"
)
print(
    f"  conditional allocation       : "
    f"{dynamic_row['conditional_allocation_rate_mean'] * 100:.2f}%"
)
print(
    f"  average utility              : "
    f"{dynamic_row['avg_utility_mean']:.4f}"
)
print(
    f"  Jain fairness                : "
    f"{dynamic_row['jain_selected_vehicles_mean']:.4f}"
)


latex = []

latex.append(r"\begin{table}[t]")
latex.append(r"\centering")
latex.append(
    r"\caption{Static and dynamic pricing comparison. "
    r"The dynamic mechanism adjusts the base service price "
    r"according to exogenous request pressure and active-vehicle "
    r"supply scarcity.}"
)
latex.append(r"\label{tab:c25_pricing}")
latex.append(r"\begin{tabular}{lccccc}")
latex.append(r"\hline")
latex.append(
    r"Mode & Price & Eligibility & Cond. Alloc. & Utility & Jain \\"
)
latex.append(r"\hline")

latex.append(
    f"Static & "
    f"{static_row['avg_offered_price_mean']:.2f} & "
    f"{static_row['eligibility_rate_mean'] * 100:.2f}\\% & "
    f"{static_row['conditional_allocation_rate_mean'] * 100:.2f}\\% & "
    f"{static_row['avg_utility_mean']:.4f} & "
    f"{static_row['jain_selected_vehicles_mean']:.4f} "
    r"\\"
)

latex.append(
    f"Dynamic & "
    f"{dynamic_row['avg_offered_price_mean']:.2f} & "
    f"{dynamic_row['eligibility_rate_mean'] * 100:.2f}\\% & "
    f"{dynamic_row['conditional_allocation_rate_mean'] * 100:.2f}\\% & "
    f"{dynamic_row['avg_utility_mean']:.4f} & "
    f"{dynamic_row['jain_selected_vehicles_mean']:.4f} "
    r"\\"
)

latex.append(r"\hline")
latex.append(r"\end{tabular}")
latex.append(r"\end{table}")

with open(
    LATEX_FILE,
    "w",
    encoding="utf-8",
) as f:
    f.write("\n".join(latex))

print()
print("OUTPUT FILES")
print("=" * 180)
print(f"Per seed   : {PER_SEED_FILE}")
print(f"Aggregated : {AGG_FILE}")
print(f"Paired     : {PAIRED_FILE}")
print(f"LaTeX      : {LATEX_FILE}")

print()
print("=" * 180)
print("C25 STATIC VS DYNAMIC PRICING ANALYSIS COMPLETED")
print("=" * 180)
print()
