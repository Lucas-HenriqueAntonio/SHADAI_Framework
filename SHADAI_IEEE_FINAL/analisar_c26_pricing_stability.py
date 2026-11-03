# analisar_c26_pricing_stability.py

import math
import os

import numpy as np
import pandas as pd

try:
    from scipy.stats import t
except ImportError:
    t = None


INPUT_DIR = "pricing_stability"
OUTPUT_DIR = os.path.join(INPUT_DIR, "analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEEDS = list(range(1, 31))
REQUEST_INTERVALS = [60, 40, 20, 10]

ALPHA = 0.20
BETA = 0.20
REFERENCE_INTERVAL = 20

PMAX = 60.0
NMAX = 400.0

PER_SEED_FILE = os.path.join(
    OUTPUT_DIR,
    "c26_stability_por_seed.csv",
)
AGG_FILE = os.path.join(
    OUTPUT_DIR,
    "c26_stability_multiseed.csv",
)
PAIRED_FILE = os.path.join(
    OUTPUT_DIR,
    "c26_stability_paired_loads.csv",
)
LATEX_FILE = os.path.join(
    OUTPUT_DIR,
    "tabela_c26_stability.tex",
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


def to_bool(series):
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1"])
    )


def clip(value, low, high):
    return max(low, min(value, high))


def experiment_name(interval, seed):
    return (
        f"c26_ri_{interval:02d}_"
        f"seed_{seed:02d}"
    )


def expected_pricing_row(row):
    base_price = float(row["base_price"])
    active = float(row["active_vehicles"])
    interval = float(row["request_interval"])

    demand = clip(
        REFERENCE_INTERVAL / interval - 1.0,
        0.0,
        1.0,
    )

    active_ratio = clip(
        active / NMAX,
        0.0,
        1.0,
    )

    scarcity = 1.0 - active_ratio

    raw_multiplier = (
        1.0
        + ALPHA * demand
        + BETA * scarcity
    )

    raw_price = (
        base_price
        * raw_multiplier
    )

    offered = clip(
        raw_price,
        base_price,
        PMAX,
    )

    effective_multiplier = (
        offered / base_price
        if base_price > 0
        else 1.0
    )

    return (
        demand,
        scarcity,
        offered,
        effective_multiplier,
    )


def sign_reversal_rate(values):
    x = np.asarray(values, dtype=float)

    if len(x) < 3:
        return float("nan")

    d = np.diff(x)

    d = d[
        np.abs(d) > 1e-12
    ]

    if len(d) < 2:
        return 0.0

    signs = np.sign(d)

    reversals = (
        signs[1:]
        * signs[:-1]
        < 0
    )

    return float(
        reversals.mean()
    )


def process_run(interval, seed):
    experiment = experiment_name(
        interval,
        seed,
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

    metrics_path = os.path.join(
        INPUT_DIR,
        f"metricas_shadai_400_{experiment}.csv",
    )

    for path in [
        requests_path,
        eligibility_path,
        outcomes_path,
        metrics_path,
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    requests = pd.read_csv(
        requests_path
    )

    eligibility = pd.read_csv(
        eligibility_path
    )

    outcomes = pd.read_csv(
        outcomes_path
    )

    metrics = pd.read_csv(
        metrics_path
    )

    requests = requests.sort_values(
        "step"
    ).reset_index(drop=True)

    expected = requests.apply(
        expected_pricing_row,
        axis=1,
        result_type="expand",
    )

    expected.columns = [
        "expected_demand",
        "expected_scarcity",
        "expected_price",
        "expected_multiplier",
    ]

    requests = pd.concat(
        [
            requests,
            expected,
        ],
        axis=1,
    )

    observed_price = pd.to_numeric(
        requests["offered_price"],
        errors="coerce",
    )

    observed_multiplier = pd.to_numeric(
        requests["price_multiplier"],
        errors="coerce",
    )

    base_price = pd.to_numeric(
        requests["base_price"],
        errors="coerce",
    )

    observed_demand = pd.to_numeric(
        requests["demand_pressure"],
        errors="coerce",
    )

    observed_scarcity = pd.to_numeric(
        requests["supply_scarcity"],
        errors="coerce",
    )

    price_abs_error = (
        observed_price
        - requests["expected_price"]
    ).abs()

    multiplier_abs_error = (
        observed_multiplier
        - requests["expected_multiplier"]
    ).abs()

    demand_abs_error = (
        observed_demand
        - requests["expected_demand"]
    ).abs()

    scarcity_abs_error = (
        observed_scarcity
        - requests["expected_scarcity"]
    ).abs()

    multipliers = (
        observed_multiplier
        .dropna()
        .to_numpy(dtype=float)
    )

    if len(multipliers) >= 2:
        abs_delta = np.abs(
            np.diff(multipliers)
        )

        mean_abs_delta = float(
            abs_delta.mean()
        )

        p95_abs_delta = float(
            np.quantile(
                abs_delta,
                0.95,
            )
        )

        max_abs_delta = float(
            abs_delta.max()
        )

    else:
        mean_abs_delta = float("nan")
        p95_abs_delta = float("nan")
        max_abs_delta = float("nan")

    multiplier_mean = float(
        np.mean(multipliers)
    )

    multiplier_sd = float(
        np.std(
            multipliers,
            ddof=1,
        )
        if len(multipliers) > 1
        else 0.0
    )

    multiplier_cv = (
        multiplier_sd / multiplier_mean
        if multiplier_mean > 0
        else float("nan")
    )

    reversal_rate = sign_reversal_rate(
        multipliers
    )

    demand_value = float(
        requests["expected_demand"].iloc[0]
    )

    theoretical_run_span = (
        ALPHA * demand_value
        + BETA
    )

    observed_span = (
        float(
            np.max(multipliers)
            - np.min(multipliers)
        )
        if len(multipliers) > 0
        else float("nan")
    )

    below_base_count = int(
        (
            observed_price
            < base_price - 1e-9
        ).sum()
    )

    above_pmax_count = int(
        (
            observed_price
            > PMAX + 1e-9
        ).sum()
    )

    multiplier_below_one_count = int(
        (
            observed_multiplier
            < 1.0 - 1e-9
        ).sum()
    )

    multiplier_above_global_bound_count = int(
        (
            observed_multiplier
            >
            1.0 + ALPHA + BETA + 1e-9
        ).sum()
    )

    saturation_rate = (
        (
            observed_price
            >= PMAX - 1e-9
        ).mean()
    )

    price_increase_rate = (
        (
            observed_price
            > base_price + 1e-12
        ).mean()
    )

    total_requests = len(requests)

    active = pd.to_numeric(
        requests["active_vehicles"],
        errors="coerce",
    ).fillna(0)

    active_requests = int(
        (active > 0).sum()
    )

    allocated = to_bool(
        requests["allocated"]
    )

    allocations = int(
        allocated.sum()
    )

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

    eligibility_rate = pd.to_numeric(
        eligibility["eligibility_rate"],
        errors="coerce",
    ).mean()

    avg_utility = pd.to_numeric(
        eligibility["avg_utility"],
        errors="coerce",
    ).mean()

    task_success_rate = (
        to_bool(
            outcomes["success"]
        ).mean()
        if len(outcomes) > 0
        else float("nan")
    )

    avg_score = (
        pd.to_numeric(
            metrics["score"],
            errors="coerce",
        ).mean()
        if len(metrics) > 0
        else float("nan")
    )

    return {
        "request_interval":
            interval,

        "seed":
            seed,

        "requests":
            total_requests,

        "active_requests":
            active_requests,

        "avg_active_vehicles":
            active.mean(),

        "avg_base_price":
            base_price.mean(),

        "avg_offered_price":
            observed_price.mean(),

        "avg_multiplier":
            multiplier_mean,

        "multiplier_sd":
            multiplier_sd,

        "multiplier_cv":
            multiplier_cv,

        "mean_abs_multiplier_delta":
            mean_abs_delta,

        "p95_abs_multiplier_delta":
            p95_abs_delta,

        "max_abs_multiplier_delta":
            max_abs_delta,

        "multiplier_span":
            observed_span,

        "theoretical_run_span_bound":
            theoretical_run_span,

        "sign_reversal_rate":
            reversal_rate,

        "avg_demand_pressure":
            observed_demand.mean(),

        "avg_supply_scarcity":
            observed_scarcity.mean(),

        "price_increase_rate":
            price_increase_rate,

        "pmax_saturation_rate":
            saturation_rate,

        "max_price_formula_error":
            price_abs_error.max(),

        "max_multiplier_formula_error":
            multiplier_abs_error.max(),

        "max_demand_formula_error":
            demand_abs_error.max(),

        "max_scarcity_formula_error":
            scarcity_abs_error.max(),

        "below_base_count":
            below_base_count,

        "above_pmax_count":
            above_pmax_count,

        "multiplier_below_one_count":
            multiplier_below_one_count,

        "multiplier_above_global_bound_count":
            multiplier_above_global_bound_count,

        "allocation_rate":
            allocation_rate,

        "conditional_allocation_rate":
            conditional_allocation_rate,

        "eligibility_rate":
            eligibility_rate,

        "avg_utility":
            avg_utility,

        "avg_final_score":
            avg_score,

        "task_success_rate":
            task_success_rate,
    }


print()
print("=" * 200)
print("C26 - DYNAMIC PRICING STABILITY ANALYSIS")
print("=" * 200)
print(
    "Stability is evaluated on the normalized price multiplier, "
    "so service-specific base-price changes are not mistaken for "
    "pricing oscillation."
)
print(
    "Checks include analytical rule reproduction, hard bounds, "
    "temporal variation, saturation, and operational behavior "
    "across request loads."
)
print("=" * 200)

rows = []
missing = []

for interval in REQUEST_INTERVALS:
    for seed in SEEDS:
        try:
            rows.append(
                process_run(
                    interval,
                    seed,
                )
            )
        except FileNotFoundError as exc:
            missing.append(
                (
                    interval,
                    seed,
                    str(exc),
                )
            )

df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError(
        "No C26 results found."
    )

df.to_csv(
    PER_SEED_FILE,
    index=False,
)

print()
print("DATASET COMPLETENESS")
print("-" * 200)

for interval in REQUEST_INTERVALS:
    n = len(
        df[
            df["request_interval"]
            == interval
        ]
    )

    print(
        f"interval={interval:<2}: "
        f"{n}/30"
    )

if missing:
    print(
        f"\n[WARNING] Missing runs: "
        f"{len(missing)}"
    )

print()
print("C26 SANITY AND STABILITY CHECK")
print("=" * 200)

if df["below_base_count"].sum() == 0:
    print(
        "[PASS] No offered price falls below its service base price."
    )
else:
    print(
        "[FAIL] Offered price below base price detected."
    )

if df["above_pmax_count"].sum() == 0:
    print(
        "[PASS] No offered price exceeds Pmax = 60."
    )
else:
    print(
        "[FAIL] Offered price above Pmax detected."
    )

if df["multiplier_below_one_count"].sum() == 0:
    print(
        "[PASS] Price multiplier is never below 1."
    )
else:
    print(
        "[FAIL] Price multiplier below 1 detected."
    )

if (
    df[
        "multiplier_above_global_bound_count"
    ].sum()
    ==
    0
):
    print(
        "[PASS] Price multiplier never exceeds "
        "the analytical global bound 1 + alpha + beta = 1.4."
    )
else:
    print(
        "[FAIL] Multiplier exceeds analytical global bound."
    )

if (
    df[
        "max_price_formula_error"
    ].max()
    <
    1e-5
):
    print(
        "[PASS] Logged prices reproduce the dynamic-pricing "
        "rule within numerical tolerance."
    )
else:
    print(
        "[FAIL] Logged price differs from the analytical rule."
    )

if (
    df[
        "max_demand_formula_error"
    ].max()
    <
    1e-8
    and
    df[
        "max_scarcity_formula_error"
    ].max()
    <
    1e-8
):
    print(
        "[PASS] Demand pressure and supply scarcity telemetry "
        "reproduce the analytical state variables."
    )
else:
    print(
        "[FAIL] State-variable telemetry mismatch detected."
    )

if (
    (
        df["multiplier_span"]
        <=
        df["theoretical_run_span_bound"]
        + 1e-8
    ).all()
):
    print(
        "[PASS] Within-run multiplier variation remains "
        "inside its analytical bound."
    )
else:
    print(
        "[FAIL] Within-run multiplier variation exceeds "
        "the analytical bound."
    )


METRICS = [
    "avg_active_vehicles",
    "avg_base_price",
    "avg_offered_price",

    "avg_multiplier",
    "multiplier_sd",
    "multiplier_cv",

    "mean_abs_multiplier_delta",
    "p95_abs_multiplier_delta",
    "max_abs_multiplier_delta",
    "multiplier_span",
    "sign_reversal_rate",

    "avg_demand_pressure",
    "avg_supply_scarcity",

    "price_increase_rate",
    "pmax_saturation_rate",

    "allocation_rate",
    "conditional_allocation_rate",
    "eligibility_rate",
    "avg_utility",
    "avg_final_score",
    "task_success_rate",
]

agg_rows = []

for interval in REQUEST_INTERVALS:
    subset = df[
        df["request_interval"]
        == interval
    ]

    row = {
        "request_interval":
            interval,

        "n_runs":
            len(subset),
    }

    for metric in METRICS:
        (
            mean,
            sd,
            low,
            high,
        ) = ci95(
            subset[metric]
        )

        row[
            f"{metric}_mean"
        ] = mean

        row[
            f"{metric}_sd"
        ] = sd

        row[
            f"{metric}_ci95_low"
        ] = low

        row[
            f"{metric}_ci95_high"
        ] = high

    agg_rows.append(row)

agg = pd.DataFrame(
    agg_rows
)

agg.to_csv(
    AGG_FILE,
    index=False,
)

print()
print("C26 MULTI-SEED STABILITY RESULTS")
print("=" * 200)

print(
    f"{'INT':>5}"
    f"{'ACTIVE':>10}"
    f"{'DEMAND':>10}"
    f"{'SCARC':>10}"
    f"{'PRICE':>10}"
    f"{'MULT':>10}"
    f"{'CV':>10}"
    f"{'MADΔ':>10}"
    f"{'P95Δ':>10}"
    f"{'MAXΔ':>10}"
    f"{'SAT%':>9}"
    f"{'ELIG%':>9}"
    f"{'COND%':>9}"
    f"{'SUCCESS%':>11}"
)

print("-" * 200)

for _, row in agg.iterrows():
    print(
        f"{int(row['request_interval']):>5}"
        f"{row['avg_active_vehicles_mean']:>10.2f}"
        f"{row['avg_demand_pressure_mean']:>10.4f}"
        f"{row['avg_supply_scarcity_mean']:>10.4f}"
        f"{row['avg_offered_price_mean']:>10.2f}"
        f"{row['avg_multiplier_mean']:>10.4f}"
        f"{row['multiplier_cv_mean']:>10.4f}"
        f"{row['mean_abs_multiplier_delta_mean']:>10.4f}"
        f"{row['p95_abs_multiplier_delta_mean']:>10.4f}"
        f"{row['max_abs_multiplier_delta_mean']:>10.4f}"
        f"{row['pmax_saturation_rate_mean'] * 100:>9.2f}"
        f"{row['eligibility_rate_mean'] * 100:>9.2f}"
        f"{row['conditional_allocation_rate_mean'] * 100:>9.2f}"
        f"{row['task_success_rate_mean'] * 100:>11.2f}"
    )


REFERENCE_INTERVAL = 20

reference = df[
    df["request_interval"]
    ==
    REFERENCE_INTERVAL
]

PAIRED_METRICS = [
    "avg_multiplier",
    "multiplier_cv",
    "mean_abs_multiplier_delta",
    "p95_abs_multiplier_delta",
    "max_abs_multiplier_delta",
    "pmax_saturation_rate",
    "eligibility_rate",
    "conditional_allocation_rate",
    "task_success_rate",
]

paired_rows = []

print()
print(
    "PAIRED DIFFERENCES VS REQUEST INTERVAL 20"
)
print("=" * 200)

for interval in [
    60,
    40,
    10,
]:
    current = df[
        df["request_interval"]
        ==
        interval
    ]

    paired = pd.merge(
        current,
        reference,
        on="seed",
        suffixes=(
            "_current",
            "_reference",
        ),
    )

    print()
    print(
        f"interval {interval} - interval 20"
    )

    for metric in PAIRED_METRICS:
        diff = (
            paired[
                f"{metric}_current"
            ]
            -
            paired[
                f"{metric}_reference"
            ]
        )

        (
            mean,
            sd,
            low,
            high,
        ) = ci95(
            diff
        )

        excludes_zero = (
            (
                low > 0
                or
                high < 0
            )
            if (
                not pd.isna(low)
                and
                not pd.isna(high)
            )
            else False
        )

        paired_rows.append(
            {
                "comparison":
                    (
                        f"interval_{interval}"
                        f"_minus_20"
                    ),

                "metric":
                    metric,

                "n_pairs":
                    len(
                        diff.dropna()
                    ),

                "mean_difference":
                    mean,

                "difference_sd":
                    sd,

                "ci95_low":
                    low,

                "ci95_high":
                    high,

                "ci95_excludes_zero":
                    excludes_zero,
            }
        )

        print(
            f"{metric:<38}"
            f" diff={mean:+.5f}"
            f" | CI95=["
            f"{low:+.5f}, "
            f"{high:+.5f}]"
            f" | excludes zero="
            f"{excludes_zero}"
        )

paired_df = pd.DataFrame(
    paired_rows
)

paired_df.to_csv(
    PAIRED_FILE,
    index=False,
)


latex = []

latex.append(
    r"\begin{table}[t]"
)

latex.append(
    r"\centering"
)

latex.append(
    r"\caption{Dynamic-pricing stability across workload intensities. "
    r"Temporal variation is measured on the normalized price multiplier "
    r"to avoid conflating service-specific base-price changes with "
    r"pricing oscillation.}"
)

latex.append(
    r"\label{tab:c26_pricing_stability}"
)

latex.append(
    r"\begin{tabular}{lcccccc}"
)

latex.append(
    r"\hline"
)

latex.append(
    r"Interval & Mult. & CV & Mean $|\Delta|$ & P95 $|\Delta|$ & Elig. & Cond. Alloc. \\"
)

latex.append(
    r"\hline"
)

for _, row in agg.iterrows():
    latex.append(
        f"{int(row['request_interval'])} & "
        f"{row['avg_multiplier_mean']:.4f} & "
        f"{row['multiplier_cv_mean']:.4f} & "
        f"{row['mean_abs_multiplier_delta_mean']:.4f} & "
        f"{row['p95_abs_multiplier_delta_mean']:.4f} & "
        f"{row['eligibility_rate_mean'] * 100:.2f}\\% & "
        f"{row['conditional_allocation_rate_mean'] * 100:.2f}\\% "
        r"\\"
    )

latex.append(
    r"\hline"
)

latex.append(
    r"\end{tabular}"
)

latex.append(
    r"\end{table}"
)

with open(
    LATEX_FILE,
    "w",
    encoding="utf-8",
) as f:
    f.write(
        "\n".join(latex)
    )

print()
print("OUTPUT FILES")
print("=" * 200)

print(
    f"Per seed   : "
    f"{PER_SEED_FILE}"
)

print(
    f"Aggregated : "
    f"{AGG_FILE}"
)

print(
    f"Paired     : "
    f"{PAIRED_FILE}"
)

print(
    f"LaTeX      : "
    f"{LATEX_FILE}"
)

print()
print("=" * 200)
print("C26 DYNAMIC PRICING STABILITY ANALYSIS COMPLETED")
print("=" * 200)
print()
