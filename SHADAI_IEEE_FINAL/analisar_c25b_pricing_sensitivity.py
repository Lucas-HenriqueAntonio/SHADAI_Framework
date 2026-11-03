# analisar_c25b_pricing_sensitivity.py

import math
import os

import pandas as pd


try:
    from scipy.stats import t
except ImportError:
    t = None


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = "pricing_sensitivity"

OUTPUT_DIR = os.path.join(
    INPUT_DIR,
    "analysis",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

SEEDS = list(range(1, 31))

CONFIGS = [
    {
        "block": "beta",
        "label": "beta_010",
        "alpha": 0.20,
        "beta": 0.10,
        "request_interval": 20,
    },
    {
        "block": "beta",
        "label": "beta_020",
        "alpha": 0.20,
        "beta": 0.20,
        "request_interval": 20,
    },
    {
        "block": "beta",
        "label": "beta_030",
        "alpha": 0.20,
        "beta": 0.30,
        "request_interval": 20,
    },

    {
        "block": "alpha",
        "label": "alpha_010",
        "alpha": 0.10,
        "beta": 0.20,
        "request_interval": 10,
    },
    {
        "block": "alpha",
        "label": "alpha_020",
        "alpha": 0.20,
        "beta": 0.20,
        "request_interval": 10,
    },
    {
        "block": "alpha",
        "label": "alpha_030",
        "alpha": 0.30,
        "beta": 0.20,
        "request_interval": 10,
    },
]


PER_SEED_FILE = os.path.join(
    OUTPUT_DIR,
    "c25b_por_seed.csv",
)

AGG_FILE = os.path.join(
    OUTPUT_DIR,
    "c25b_multiseed.csv",
)

PAIRED_FILE = os.path.join(
    OUTPUT_DIR,
    "c25b_paired_vs_020.csv",
)

LATEX_FILE = os.path.join(
    OUTPUT_DIR,
    "tabela_c25b_sensitivity.tex",
)


# ============================================================
# HELPERS
# ============================================================

def to_bool(series):
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1"])
    )


def ci95(values):
    s = pd.Series(
        values,
        dtype="float64",
    ).dropna()

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

    se = (
        sd
        /
        math.sqrt(n)
    )

    critical = (
        t.ppf(
            0.975,
            n - 1,
        )
        if t is not None
        else 1.96
    )

    margin = (
        critical
        *
        se
    )

    return (
        mean,
        sd,
        mean - margin,
        mean + margin,
    )


def jain_index(values):
    s = pd.Series(
        values,
        dtype="float64",
    )

    s = s[
        s > 0
    ]

    n = len(s)

    if n == 0:
        return float("nan")

    denominator = (
        n
        *
        (s ** 2).sum()
    )

    if denominator == 0:
        return float("nan")

    return (
        s.sum() ** 2
    ) / denominator


def experiment_name(
    config,
    seed,
):
    return (
        f"c25b_{config['label']}_"
        f"seed_{seed:02d}"
    )


# ============================================================
# PROCESS RUN
# ============================================================

def process_run(
    config,
    seed,
):
    experiment = (
        experiment_name(
            config,
            seed,
        )
    )

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
            raise FileNotFoundError(
                path
            )

    metrics = pd.read_csv(
        metrics_path
    )

    requests = pd.read_csv(
        requests_path
    )

    eligibility = pd.read_csv(
        eligibility_path
    )

    outcomes = pd.read_csv(
        outcomes_path
    )

    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    modes = set(
        requests[
            "pricing_mode"
        ]
        .astype(str)
        .str.lower()
        .unique()
    )

    if modes != {"dynamic"}:
        raise RuntimeError(
            f"Non-dynamic run detected: "
            f"{experiment}"
        )

    observed_alpha = float(
        requests[
            "pricing_alpha"
        ].iloc[0]
    )

    observed_beta = float(
        requests[
            "pricing_beta"
        ].iloc[0]
    )

    observed_interval = int(
        float(
            requests[
                "request_interval"
            ].iloc[0]
        )
    )

    if abs(
        observed_alpha
        -
        config["alpha"]
    ) > 1e-12:
        raise RuntimeError(
            f"Alpha mismatch: {experiment}"
        )

    if abs(
        observed_beta
        -
        config["beta"]
    ) > 1e-12:
        raise RuntimeError(
            f"Beta mismatch: {experiment}"
        )

    if (
        observed_interval
        !=
        config["request_interval"]
    ):
        raise RuntimeError(
            f"Request interval mismatch: "
            f"{experiment}"
        )

    # ========================================================
    # REQUEST METRICS
    # ========================================================

    total_requests = len(
        requests
    )

    active = pd.to_numeric(
        requests[
            "active_vehicles"
        ],
        errors="coerce",
    ).fillna(0)

    active_requests = int(
        (
            active > 0
        ).sum()
    )

    allocated = to_bool(
        requests[
            "allocated"
        ]
    )

    allocations = int(
        allocated.sum()
    )

    allocation_rate = (
        allocations
        /
        total_requests
        if total_requests > 0
        else float("nan")
    )

    conditional_allocation_rate = (
        allocations
        /
        active_requests
        if active_requests > 0
        else float("nan")
    )

    base_price = pd.to_numeric(
        requests[
            "base_price"
        ],
        errors="coerce",
    )

    offered_price = pd.to_numeric(
        requests[
            "offered_price"
        ],
        errors="coerce",
    )

    price_multiplier = pd.to_numeric(
        requests[
            "price_multiplier"
        ],
        errors="coerce",
    )

    demand_pressure = pd.to_numeric(
        requests[
            "demand_pressure"
        ],
        errors="coerce",
    )

    supply_scarcity = pd.to_numeric(
        requests[
            "supply_scarcity"
        ],
        errors="coerce",
    )

    price_increase_rate = (
        offered_price
        >
        base_price
        +
        1e-12
    ).mean()

    max_offered_price = (
        offered_price.max()
        if len(
            offered_price
        ) > 0
        else float("nan")
    )

    # ========================================================
    # ELIGIBILITY
    # ========================================================

    eligibility_rate = pd.to_numeric(
        eligibility[
            "eligibility_rate"
        ],
        errors="coerce",
    ).mean()

    avg_utility = pd.to_numeric(
        eligibility[
            "avg_utility"
        ],
        errors="coerce",
    ).mean()

    # ========================================================
    # SELECTED VEHICLES
    # ========================================================

    if len(
        metrics
    ) > 0:
        avg_score = pd.to_numeric(
            metrics[
                "score"
            ],
            errors="coerce",
        ).mean()

        avg_rep = pd.to_numeric(
            metrics[
                "reputacao_norm"
            ],
            errors="coerce",
        ).mean()

        avg_energy_norm = pd.to_numeric(
            metrics[
                "energia_norm"
            ],
            errors="coerce",
        ).mean()

        avg_prox = pd.to_numeric(
            metrics[
                "proximidade_norm"
            ],
            errors="coerce",
        ).mean()

        avg_distance = pd.to_numeric(
            metrics[
                "distancia"
            ],
            errors="coerce",
        ).mean()

        avg_available_energy = pd.to_numeric(
            metrics[
                "energia"
            ],
            errors="coerce",
        ).mean()

        task_counts = (
            metrics[
                "veiculo"
            ]
            .astype(str)
            .value_counts()
        )

        unique_selected = len(
            task_counts
        )

        tasks_per_vehicle = (
            task_counts.mean()
        )

        jain = jain_index(
            task_counts.values
        )

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

    # ========================================================
    # TASK RELIABILITY
    # ========================================================

    if len(
        outcomes
    ) > 0:
        success = to_bool(
            outcomes[
                "success"
            ]
        )

        task_success_rate = (
            success.mean()
        )
    else:
        task_success_rate = float(
            "nan"
        )

    return {
        "block":
            config["block"],

        "label":
            config["label"],

        "seed":
            seed,

        "alpha":
            config["alpha"],

        "beta":
            config["beta"],

        "request_interval":
            config[
                "request_interval"
            ],

        "total_requests":
            total_requests,

        "active_requests":
            active_requests,

        "allocations":
            allocations,

        "allocation_rate":
            allocation_rate,

        "conditional_allocation_rate":
            conditional_allocation_rate,

        "avg_base_price":
            base_price.mean(),

        "avg_offered_price":
            offered_price.mean(),

        "max_offered_price":
            max_offered_price,

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
    }


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 190)
print("C25b - DYNAMIC PRICING PARAMETER SENSITIVITY ANALYSIS")
print("=" * 190)

print(
    "Beta block isolates supply-scarcity sensitivity "
    "at request interval 20."
)

print(
    "Alpha block isolates demand-pressure sensitivity "
    "at request interval 10."
)

print("=" * 190)


rows = []
missing = []

for config in CONFIGS:
    for seed in SEEDS:
        try:
            rows.append(
                process_run(
                    config,
                    seed,
                )
            )

        except FileNotFoundError as exc:
            missing.append(
                (
                    config["label"],
                    seed,
                    str(exc),
                )
            )


df = pd.DataFrame(
    rows
)

if df.empty:
    raise RuntimeError(
        "No C25b results found."
    )


df.to_csv(
    PER_SEED_FILE,
    index=False,
)


# ============================================================
# COMPLETENESS
# ============================================================

print()
print("DATASET COMPLETENESS")
print("-" * 190)

for config in CONFIGS:
    n = len(
        df[
            df[
                "label"
            ]
            ==
            config[
                "label"
            ]
        ]
    )

    print(
        f"{config['label']:<12}: "
        f"{n}/30"
    )

if missing:
    print(
        f"\n[WARNING] Missing runs: "
        f"{len(missing)}"
    )


# ============================================================
# SANITY CHECKS
# ============================================================

print()
print("C25b SANITY CHECK")
print("=" * 190)


beta_block = df[
    df[
        "block"
    ]
    ==
    "beta"
]


alpha_block = df[
    df[
        "block"
    ]
    ==
    "alpha"
]


if (
    beta_block[
        "avg_demand_pressure"
    ].abs()
    <
    1e-12
).all():
    print(
        "[PASS] Beta block has zero demand pressure."
    )
else:
    print(
        "[FAIL] Beta block unexpectedly has demand pressure."
    )


if (
    alpha_block[
        "avg_demand_pressure"
    ]
    .sub(1.0)
    .abs()
    <
    1e-12
).all():
    print(
        "[PASS] Alpha block has demand pressure = 1."
    )
else:
    print(
        "[FAIL] Alpha block does not isolate active demand pressure."
    )


if (
    df[
        "max_offered_price"
    ]
    <=
    60.0
    +
    1e-12
).all():
    print(
        "[PASS] All dynamic prices respect Pmax = 60."
    )
else:
    print(
        "[FAIL] Price above Pmax detected."
    )


# ============================================================
# AGGREGATION
# ============================================================

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
]


agg_rows = []

for config in CONFIGS:
    subset = df[
        df[
            "label"
        ]
        ==
        config[
            "label"
        ]
    ]

    row = {
        "block":
            config["block"],

        "label":
            config["label"],

        "n_runs":
            len(subset),

        "alpha":
            config["alpha"],

        "beta":
            config["beta"],

        "request_interval":
            config[
                "request_interval"
            ],
    }

    for metric in METRICS:
        (
            mean,
            sd,
            low,
            high,
        ) = ci95(
            subset[
                metric
            ]
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

    agg_rows.append(
        row
    )


agg = pd.DataFrame(
    agg_rows
)


agg.to_csv(
    AGG_FILE,
    index=False,
)


# ============================================================
# DISPLAY
# ============================================================

for block in [
    "beta",
    "alpha",
]:
    print()
    print(
        f"C25b {block.upper()} SENSITIVITY"
    )
    print("=" * 190)

    print(
        f"{'CONFIG':<12}"
        f"{'ALPHA':>8}"
        f"{'BETA':>8}"
        f"{'PRICE':>10}"
        f"{'MULT':>10}"
        f"{'ELIG%':>10}"
        f"{'UTILITY':>10}"
        f"{'ALLOC%':>10}"
        f"{'COND%':>10}"
        f"{'UNIQUE':>10}"
        f"{'JAIN':>10}"
        f"{'SUCCESS%':>11}"
    )

    print("-" * 190)

    sub = agg[
        agg[
            "block"
        ]
        ==
        block
    ]

    for _, row in sub.iterrows():
        print(
            f"{row['label']:<12}"
            f"{row['alpha']:>8.2f}"
            f"{row['beta']:>8.2f}"
            f"{row['avg_offered_price_mean']:>10.2f}"
            f"{row['avg_price_multiplier_mean']:>10.4f}"
            f"{row['eligibility_rate_mean'] * 100:>10.2f}"
            f"{row['avg_utility_mean']:>10.4f}"
            f"{row['allocation_rate_mean'] * 100:>10.2f}"
            f"{row['conditional_allocation_rate_mean'] * 100:>10.2f}"
            f"{row['unique_selected_vehicles_mean']:>10.2f}"
            f"{row['jain_selected_vehicles_mean']:>10.4f}"
            f"{row['task_success_rate_mean'] * 100:>11.2f}"
        )


# ============================================================
# PAIRED DIFFERENCES AGAINST 0.20
# ============================================================

PAIRED_METRICS = [
    "avg_offered_price",
    "avg_price_multiplier",
    "eligibility_rate",
    "avg_utility",
    "allocation_rate",
    "conditional_allocation_rate",
    "avg_final_score",
    "unique_selected_vehicles",
    "jain_selected_vehicles",
    "task_success_rate",
]


paired_rows = []


for block in [
    "beta",
    "alpha",
]:
    reference_label = (
        "beta_020"
        if block == "beta"
        else "alpha_020"
    )

    reference = df[
        df[
            "label"
        ]
        ==
        reference_label
    ]


    print()
    print(
        f"PAIRED DIFFERENCES VS {reference_label.upper()}"
    )
    print("=" * 190)


    labels = (
        ["beta_010", "beta_030"]
        if block == "beta"
        else ["alpha_010", "alpha_030"]
    )


    for label in labels:
        current = df[
            df[
                "label"
            ]
            ==
            label
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
            f"{label} - {reference_label}"
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
                    "block":
                        block,

                    "comparison":
                        (
                            f"{label}_minus_"
                            f"{reference_label}"
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
                f"{metric:<42}"
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


# ============================================================
# LATEX
# ============================================================

latex = []

latex.append(
    r"\begin{table}[t]"
)

latex.append(
    r"\centering"
)

latex.append(
    r"\caption{Sensitivity of the dynamic-pricing coefficients. "
    r"The supply coefficient $\beta$ is varied under the baseline "
    r"request interval, whereas the demand coefficient $\alpha$ is "
    r"varied under a high-load setting in which demand pressure is active.}"
)

latex.append(
    r"\label{tab:c25b_pricing_sensitivity}"
)

latex.append(
    r"\begin{tabular}{lcccccc}"
)

latex.append(
    r"\hline"
)

latex.append(
    r"Config. & $\alpha$ & $\beta$ & Price & Elig. & Cond. Alloc. & Utility \\"
)

latex.append(
    r"\hline"
)


for _, row in agg.iterrows():
    latex.append(
        f"{row['label']} & "
        f"{row['alpha']:.2f} & "
        f"{row['beta']:.2f} & "
        f"{row['avg_offered_price_mean']:.2f} & "
        f"{row['eligibility_rate_mean'] * 100:.2f}\\% & "
        f"{row['conditional_allocation_rate_mean'] * 100:.2f}\\% & "
        f"{row['avg_utility_mean']:.4f} "
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
        "\n".join(
            latex
        )
    )


# ============================================================
# OUTPUT
# ============================================================

print()
print("OUTPUT FILES")
print("=" * 190)

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
print("=" * 190)
print("C25b PRICING SENSITIVITY ANALYSIS COMPLETED")
print("=" * 190)
print()
