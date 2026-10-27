# analisar_c22_shadai_vs_greedy.py

import math
import os

import pandas as pd


try:
    from scipy.stats import t
except ImportError:
    t = None


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = "greedy_alignment"

OUTPUT_DIR = os.path.join(
    INPUT_DIR,
    "analysis"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


SEEDS = list(
    range(1, 31)
)


POLICIES = {

    "shadai": 0.2,

    "greedy": 0.0
}


# ============================================================
# OUTPUT
# ============================================================

PER_SEED_FILE = os.path.join(
    OUTPUT_DIR,
    "c22_por_seed.csv"
)

AGG_FILE = os.path.join(
    OUTPUT_DIR,
    "c22_multiseed.csv"
)

PAIRED_FILE = os.path.join(
    OUTPUT_DIR,
    "c22_paired_shadai_minus_greedy.csv"
)

LATEX_FILE = os.path.join(
    OUTPUT_DIR,
    "tabela_c22_shadai_vs_greedy.tex"
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
        .isin([
            "true",
            "1"
        ])
    )


def ci95(values):

    s = pd.Series(
        values,
        dtype="float64"
    ).dropna()


    n = len(s)


    if n == 0:

        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan")
        )


    mean = s.mean()


    if n == 1:

        return (
            mean,
            float("nan"),
            float("nan"),
            float("nan")
        )


    sd = s.std(
        ddof=1
    )


    se = (
        sd
        /
        math.sqrt(n)
    )


    critical = (
        t.ppf(
            0.975,
            n - 1
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
        mean + margin
    )


def jain_index(values):

    values = pd.Series(
        values,
        dtype="float64"
    )


    values = values[
        values > 0
    ]


    n = len(values)


    if n == 0:

        return float("nan")


    denominator = (
        n
        *
        (
            values
            ** 2
        ).sum()
    )


    if denominator == 0:

        return float("nan")


    return (
        values.sum()
        ** 2
    ) / denominator


def experiment_name(
    policy,
    seed
):

    return (
        f"c22_{policy}_"
        f"seed_{seed:02d}"
    )


# ============================================================
# PROCESS RUN
# ============================================================

def process_run(
    policy,
    seed
):

    experiment = (
        experiment_name(
            policy,
            seed
        )
    )


    metrics_path = os.path.join(

        INPUT_DIR,

        (
            f"metricas_shadai_400_"
            f"{experiment}.csv"
        )
    )


    requests_path = os.path.join(

        INPUT_DIR,

        (
            f"requests_shadai_400_"
            f"{experiment}.csv"
        )
    )


    outcomes_path = os.path.join(

        INPUT_DIR,

        (
            f"outcomes_shadai_400_"
            f"{experiment}.csv"
        )
    )


    required = [
        metrics_path,
        requests_path,
        outcomes_path
    ]


    for path in required:

        if not os.path.exists(
            path
        ):

            raise FileNotFoundError(
                path
            )


    metrics = pd.read_csv(
        metrics_path
    )


    requests = pd.read_csv(
        requests_path
    )


    outcomes = pd.read_csv(
        outcomes_path
    )


    # ========================================================
    # REQUESTS
    # ========================================================

    total_requests = len(
        requests
    )


    active_vehicles = pd.to_numeric(
        requests[
            "active_vehicles"
        ],
        errors="coerce"
    ).fillna(
        0
    )


    active_mask = (
        active_vehicles
        >
        0
    )


    active_requests = int(
        active_mask.sum()
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


    # ========================================================
    # SELECTED METRICS
    # ========================================================

    selected_count = len(
        metrics
    )


    if selected_count > 0:

        avg_score = pd.to_numeric(
            metrics[
                "score"
            ],
            errors="coerce"
        ).mean()


        avg_reputation_norm = pd.to_numeric(
            metrics[
                "reputacao_norm"
            ],
            errors="coerce"
        ).mean()


        avg_energy_norm = pd.to_numeric(
            metrics[
                "energia_norm"
            ],
            errors="coerce"
        ).mean()


        avg_proximity_norm = pd.to_numeric(
            metrics[
                "proximidade_norm"
            ],
            errors="coerce"
        ).mean()


        avg_distance = pd.to_numeric(
            metrics[
                "distancia"
            ],
            errors="coerce"
        ).mean()


        # IMPORTANT:
        # 'energia' is energy available at selection time,
        # before service execution.
        avg_selected_energy = pd.to_numeric(
            metrics[
                "energia"
            ],
            errors="coerce"
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


        tasks_per_selected_vehicle = (
            task_counts.mean()
        )


        jain = jain_index(
            task_counts.values
        )


        modes = (
            metrics[
                "selection_mode"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
        )


        exploration_mask = modes.isin([
            "exploration",
            "explore"
        ])


        exploitation_mask = modes.isin([
            "exploitation",
            "exploit",
            "greedy"
        ])


        unknown_modes = (
            ~(
                exploration_mask
                |
                exploitation_mask
            )
        ).sum()


        exploration_rate = (
            exploration_mask.mean()
        )


    else:

        avg_score = float("nan")
        avg_reputation_norm = float("nan")
        avg_energy_norm = float("nan")
        avg_proximity_norm = float("nan")
        avg_distance = float("nan")
        avg_selected_energy = float("nan")

        unique_selected = 0

        tasks_per_selected_vehicle = float(
            "nan"
        )

        jain = float("nan")

        exploration_rate = float("nan")

        unknown_modes = 0


    # ========================================================
    # OUTCOMES
    # ========================================================

    completed_tasks = len(
        outcomes
    )


    if (
        completed_tasks > 0
        and
        "success" in outcomes.columns
    ):

        success_bool = to_bool(
            outcomes[
                "success"
            ]
        )


        task_success_rate = (
            success_bool.mean()
        )


        task_failure_rate = (
            1.0
            -
            task_success_rate
        )


    else:

        task_success_rate = float(
            "nan"
        )

        task_failure_rate = float(
            "nan"
        )


    # ========================================================
    # ROW
    # ========================================================

    return {

        "policy":
            policy,

        "seed":
            seed,

        "epsilon":
            POLICIES[
                policy
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

        "avg_final_score":
            avg_score,

        "avg_reputation_norm":
            avg_reputation_norm,

        "avg_energy_norm":
            avg_energy_norm,

        "avg_proximity_norm":
            avg_proximity_norm,

        "avg_distance":
            avg_distance,

        "avg_selected_energy":
            avg_selected_energy,

        "unique_selected_vehicles":
            unique_selected,

        "tasks_per_selected_vehicle":
            tasks_per_selected_vehicle,

        "jain_selected_vehicles":
            jain,

        "exploration_rate":
            exploration_rate,

        "unknown_selection_modes":
            unknown_modes,

        "completed_tasks":
            completed_tasks,

        "task_success_rate":
            task_success_rate,

        "task_failure_rate":
            task_failure_rate
    }


# ============================================================
# HEADER
# ============================================================

print("\n")

print("=" * 180)

print(
    "C22 - SHADAI VS CORRECTED GREEDY ANALYSIS"
)

print("=" * 180)


print(
    "Greedy definition:"
)

print(
    "  deterministic argmax of the same SHADAI final score"
)

print(
    "  epsilon = 0"
)

print(
    "  same eligibility, score weights, normalization and cooldown"
)

print("=" * 180)


# ============================================================
# LOAD
# ============================================================

rows = []

missing = []


for policy in POLICIES:

    for seed in SEEDS:

        try:

            rows.append(
                process_run(
                    policy,
                    seed
                )
            )

        except FileNotFoundError as e:

            missing.append(
                (
                    policy,
                    seed,
                    str(e)
                )
            )


df = pd.DataFrame(
    rows
)


if df.empty:

    raise RuntimeError(
        "No C22 results found."
    )


df.to_csv(
    PER_SEED_FILE,
    index=False
)


# ============================================================
# COMPLETENESS
# ============================================================

print("\n")

print(
    "DATASET COMPLETENESS"
)

print("-" * 180)


for policy in POLICIES:

    n = len(
        df[
            df[
                "policy"
            ]
            ==
            policy
        ]
    )


    print(
        f"{policy:<10}: "
        f"{n}/30"
    )


if missing:

    print(
        f"\n[WARNING] Missing runs: "
        f"{len(missing)}"
    )


# ============================================================
# SANITY
# ============================================================

print("\n")

print(
    "C22 SANITY CHECK"
)

print("=" * 180)


greedy = df[
    df[
        "policy"
    ]
    ==
    "greedy"
]


shadai = df[
    df[
        "policy"
    ]
    ==
    "shadai"
]


if (
    greedy[
        "exploration_rate"
    ]
    .fillna(0)
    .abs()
    <
    1e-12
).all():

    print(
        "[PASS] Greedy performs zero exploration."
    )

else:

    print(
        "[FAIL] Greedy contains exploratory selections."
    )


if (
    greedy[
        "unknown_selection_modes"
    ]
    ==
    0
).all():

    print(
        "[PASS] All Greedy selection modes are recognized."
    )

else:

    print(
        "[FAIL] Unknown Greedy selection modes detected."
    )


shadai_exploration = (
    shadai[
        "exploration_rate"
    ].mean()
)


print(
    f"[INFO] Mean SHADAI exploration rate: "
    f"{shadai_exploration * 100:.2f}%"
)


if abs(
    shadai_exploration
    -
    0.20
) < 0.05:

    print(
        "[PASS] SHADAI exploration rate is consistent with epsilon=0.2."
    )

else:

    print(
        "[CHECK] SHADAI exploration rate is unexpectedly far from epsilon=0.2."
    )


# ============================================================
# AGGREGATE
# ============================================================

METRICS = [

    "allocation_rate",

    "conditional_allocation_rate",

    "avg_final_score",

    "avg_reputation_norm",

    "avg_energy_norm",

    "avg_proximity_norm",

    "avg_distance",

    "avg_selected_energy",

    "unique_selected_vehicles",

    "tasks_per_selected_vehicle",

    "jain_selected_vehicles",

    "exploration_rate",

    "task_success_rate",

    "task_failure_rate"
]


agg_rows = []


for policy in POLICIES:

    subset = df[
        df[
            "policy"
        ]
        ==
        policy
    ]


    row = {

        "policy":
            policy,

        "epsilon":
            POLICIES[
                policy
            ],

        "n_runs":
            len(
                subset
            )
    }


    for metric in METRICS:

        (
            mean,
            sd,
            low,
            high
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
    index=False
)


# ============================================================
# MAIN TABLE
# ============================================================

print("\n")

print(
    "C22 MULTI-SEED RESULTS"
)

print("=" * 180)


print(

    f"{'POLICY':<10}"

    f"{'ALLOC%':>10}"

    f"{'COND%':>10}"

    f"{'SCORE':>10}"

    f"{'DIST':>10}"

    f"{'ENERGY':>10}"

    f"{'UNIQUE':>10}"

    f"{'JAIN':>10}"

    f"{'EXPLORE%':>11}"

    f"{'SUCCESS%':>11}"
)


print("-" * 180)


for _, row in agg.iterrows():

    print(

        f"{row['policy']:<10}"

        f"{row['allocation_rate_mean'] * 100:>10.2f}"

        f"{row['conditional_allocation_rate_mean'] * 100:>10.2f}"

        f"{row['avg_final_score_mean']:>10.4f}"

        f"{row['avg_distance_mean']:>10.2f}"

        f"{row['avg_selected_energy_mean']:>10.2f}"

        f"{row['unique_selected_vehicles_mean']:>10.2f}"

        f"{row['jain_selected_vehicles_mean']:>10.4f}"

        f"{row['exploration_rate_mean'] * 100:>11.2f}"

        f"{row['task_success_rate_mean'] * 100:>11.2f}"
    )


# ============================================================
# PAIRED ANALYSIS
# ============================================================

paired = pd.merge(

    shadai,

    greedy,

    on="seed",

    suffixes=(
        "_shadai",
        "_greedy"
    )
)


PAIRED_METRICS = [

    "allocation_rate",

    "conditional_allocation_rate",

    "avg_final_score",

    "avg_reputation_norm",

    "avg_energy_norm",

    "avg_proximity_norm",

    "avg_distance",

    "avg_selected_energy",

    "unique_selected_vehicles",

    "tasks_per_selected_vehicle",

    "jain_selected_vehicles",

    "task_success_rate"
]


paired_rows = []


print("\n")

print(
    "PAIRED DIFFERENCES: SHADAI - GREEDY"
)

print("=" * 180)


for metric in PAIRED_METRICS:

    difference = (

        paired[
            f"{metric}_shadai"
        ]

        -

        paired[
            f"{metric}_greedy"
        ]
    )


    (
        mean,
        sd,
        low,
        high
    ) = ci95(
        difference
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


    paired_rows.append({

        "metric":
            metric,

        "n_pairs":
            len(
                difference.dropna()
            ),

        "mean_difference_shadai_minus_greedy":
            mean,

        "difference_sd":
            sd,

        "ci95_low":
            low,

        "ci95_high":
            high,

        "ci95_excludes_zero":
            excludes_zero
    })


    print(

        f"{metric:<36}"

        f" diff="
        f"{mean:+.5f}"

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
    index=False
)


# ============================================================
# CORE SUMMARY
# ============================================================

shadai_row = agg[
    agg[
        "policy"
    ]
    ==
    "shadai"
].iloc[
    0
]


greedy_row = agg[
    agg[
        "policy"
    ]
    ==
    "greedy"
].iloc[
    0
]


print("\n")

print(
    "C22 CORE SUMMARY"
)

print("=" * 180)


print(
    "SHADAI:"
)

print(
    f"  epsilon                         : 0.20"
)

print(
    f"  final score                     : "
    f"{shadai_row['avg_final_score_mean']:.4f}"
)

print(
    f"  distance                        : "
    f"{shadai_row['avg_distance_mean']:.2f}"
)

print(
    f"  available energy at selection   : "
    f"{shadai_row['avg_selected_energy_mean']:.2f}"
)

print(
    f"  Jain fairness                   : "
    f"{shadai_row['jain_selected_vehicles_mean']:.4f}"
)


print(
    "\nGreedy:"
)

print(
    f"  epsilon                         : 0.00"
)

print(
    f"  final score                     : "
    f"{greedy_row['avg_final_score_mean']:.4f}"
)

print(
    f"  distance                        : "
    f"{greedy_row['avg_distance_mean']:.2f}"
)

print(
    f"  available energy at selection   : "
    f"{greedy_row['avg_selected_energy_mean']:.2f}"
)

print(
    f"  Jain fairness                   : "
    f"{greedy_row['jain_selected_vehicles_mean']:.4f}"
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
    r"\caption{Comparison between SHADAI and the corrected Greedy baseline. Greedy uses the same eligibility, scoring, normalization, and cooldown mechanisms as SHADAI, but performs pure exploitation ($\epsilon=0$).}"
)

latex.append(
    r"\label{tab:c22_greedy}"
)

latex.append(
    r"\begin{tabular}{lccccc}"
)

latex.append(
    r"\hline"
)

latex.append(
    r"Policy & Alloc. & Score & Distance & Energy & Jain \\"
)

latex.append(
    r"\hline"
)


latex.append(

    f"SHADAI & "

    f"{shadai_row['conditional_allocation_rate_mean'] * 100:.2f}\\% & "

    f"{shadai_row['avg_final_score_mean']:.4f} & "

    f"{shadai_row['avg_distance_mean']:.2f} & "

    f"{shadai_row['avg_selected_energy_mean']:.2f} & "

    f"{shadai_row['jain_selected_vehicles_mean']:.4f} "

    r"\\"
)


latex.append(

    f"Greedy & "

    f"{greedy_row['conditional_allocation_rate_mean'] * 100:.2f}\\% & "

    f"{greedy_row['avg_final_score_mean']:.4f} & "

    f"{greedy_row['avg_distance_mean']:.2f} & "

    f"{greedy_row['avg_selected_energy_mean']:.2f} & "

    f"{greedy_row['jain_selected_vehicles_mean']:.4f} "

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
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(
            latex
        )
    )


# ============================================================
# OUTPUT
# ============================================================

print("\n")

print(
    "OUTPUT FILES"
)

print("=" * 180)


print(
    f"Per seed   : {PER_SEED_FILE}"
)

print(
    f"Aggregated : {AGG_FILE}"
)

print(
    f"Paired     : {PAIRED_FILE}"
)

print(
    f"LaTeX      : {LATEX_FILE}"
)


print("\n")

print("=" * 180)

print(
    "C22 SHADAI VS CORRECTED GREEDY ANALYSIS COMPLETED"
)

print("=" * 180)

print()