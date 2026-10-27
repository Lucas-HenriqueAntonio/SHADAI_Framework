# analisar_c23_shadai_vs_norep.py

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

INPUT_DIR = "norep_alignment"

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

    "shadai": {

        "w1": 0.4,

        "w2": 0.3,

        "w3": 0.3
    },

    "norep": {

        "w1": 0.0,

        "w2": 0.5,

        "w3": 0.5
    }
}


# ============================================================
# OUTPUT
# ============================================================

PER_SEED_FILE = os.path.join(
    OUTPUT_DIR,
    "c23_por_seed.csv"
)

AGG_FILE = os.path.join(
    OUTPUT_DIR,
    "c23_multiseed.csv"
)

PAIRED_FILE = os.path.join(
    OUTPUT_DIR,
    "c23_paired_shadai_minus_norep.csv"
)

LATEX_FILE = os.path.join(
    OUTPUT_DIR,
    "tabela_c23_shadai_vs_norep.tex"
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

    s = pd.Series(
        values,
        dtype="float64"
    )


    s = s[
        s > 0
    ]


    n = len(s)


    if n == 0:

        return float(
            "nan"
        )


    denominator = (

        n

        *

        (
            s
            **
            2
        ).sum()
    )


    if denominator == 0:

        return float(
            "nan"
        )


    return (

        s.sum()
        **
        2

    ) / denominator


def experiment_name(
    policy,
    seed
):

    return (
        f"c23_{policy}_"
        f"seed_{seed:02d}"
    )


# ============================================================
# PROCESS ONE RUN
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


    for path in [
        metrics_path,
        requests_path,
        outcomes_path
    ]:

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
    # VERIFY WEIGHTS
    # ========================================================

    expected = POLICIES[
        policy
    ]


    if len(metrics) > 0:

        first = metrics.iloc[
            0
        ]


        actual_w1 = float(
            first[
                "w1"
            ]
        )

        actual_w2 = float(
            first[
                "w2"
            ]
        )

        actual_w3 = float(
            first[
                "w3"
            ]
        )


        if not (

            abs(
                actual_w1
                -
                expected[
                    "w1"
                ]
            )
            <
            1e-12

            and

            abs(
                actual_w2
                -
                expected[
                    "w2"
                ]
            )
            <
            1e-12

            and

            abs(
                actual_w3
                -
                expected[
                    "w3"
                ]
            )
            <
            1e-12
        ):

            raise RuntimeError(
                f"Unexpected weights in {metrics_path}: "
                f"({actual_w1}, {actual_w2}, {actual_w3})"
            )


    # ========================================================
    # REQUEST STATISTICS
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


    active_requests = int(
        (
            active_vehicles
            >
            0
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


    # ========================================================
    # SELECTION METRICS
    # ========================================================

    if len(metrics) > 0:

        score = pd.to_numeric(
            metrics[
                "score"
            ],
            errors="coerce"
        )


        reputation = pd.to_numeric(
            metrics[
                "reputacao_norm"
            ],
            errors="coerce"
        )


        energy_norm = pd.to_numeric(
            metrics[
                "energia_norm"
            ],
            errors="coerce"
        )


        proximity = pd.to_numeric(
            metrics[
                "proximidade_norm"
            ],
            errors="coerce"
        )


        distance = pd.to_numeric(
            metrics[
                "distancia"
            ],
            errors="coerce"
        )


        selected_energy = pd.to_numeric(
            metrics[
                "energia"
            ],
            errors="coerce"
        )


        avg_score = score.mean()

        avg_reputation = reputation.mean()

        avg_energy_norm = energy_norm.mean()

        avg_proximity = proximity.mean()

        avg_distance = distance.mean()

        avg_selected_energy = (
            selected_energy.mean()
        )


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


        selection_modes = (
            metrics[
                "selection_mode"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
        )


        exploration = (
            selection_modes.isin([
                "exploration",
                "explore"
            ])
        )


        exploration_rate = (
            exploration.mean()
        )


    else:

        avg_score = float("nan")

        avg_reputation = float("nan")

        avg_energy_norm = float("nan")

        avg_proximity = float("nan")

        avg_distance = float("nan")

        avg_selected_energy = float(
            "nan"
        )

        unique_selected = 0

        tasks_per_vehicle = float(
            "nan"
        )

        jain = float(
            "nan"
        )

        exploration_rate = float(
            "nan"
        )


    # ========================================================
    # TASK RELIABILITY
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


    return {

        "policy":
            policy,

        "seed":
            seed,

        "w1":
            expected[
                "w1"
            ],

        "w2":
            expected[
                "w2"
            ],

        "w3":
            expected[
                "w3"
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
            avg_reputation,

        "avg_energy_norm":
            avg_energy_norm,

        "avg_proximity_norm":
            avg_proximity,

        "avg_distance":
            avg_distance,

        # This is available energy at selection,
        # not post-service residual energy.
        "avg_available_energy_at_selection":
            avg_selected_energy,

        "unique_selected_vehicles":
            unique_selected,

        "tasks_per_selected_vehicle":
            tasks_per_vehicle,

        "jain_selected_vehicles":
            jain,

        "exploration_rate":
            exploration_rate,

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
    "C23 - SHADAI VS RENORMALIZED NO-REPUTATION ANALYSIS"
)

print("=" * 180)


print(
    "SHADAI:"
)

print(
    "  w = (0.4, 0.3, 0.3)"
)


print(
    "No-Reputation:"
)

print(
    "  w = (0.0, 0.5, 0.5)"
)


print(
    "Important:"
)

print(
    "  Raw final-score values are not directly comparable "
    "as identical objective functions because the score "
    "composition changes between policies."
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
        "No C23 results found."
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
# SANITY CHECK
# ============================================================

print("\n")

print(
    "C23 SANITY CHECK"
)

print("=" * 180)


shadai = df[
    df[
        "policy"
    ]
    ==
    "shadai"
]


norep = df[
    df[
        "policy"
    ]
    ==
    "norep"
]


if (
    (
        norep[
            "w1"
        ]
        ==
        0.0
    )
    &
    (
        norep[
            "w2"
        ]
        ==
        0.5
    )
    &
    (
        norep[
            "w3"
        ]
        ==
        0.5
    )
).all():

    print(
        "[PASS] No-Reputation uses weights "
        "(0.0, 0.5, 0.5)."
    )

else:

    print(
        "[FAIL] Incorrect No-Reputation weights detected."
    )


if (
    (
        norep[
            "w1"
        ]
        +
        norep[
            "w2"
        ]
        +
        norep[
            "w3"
        ]
    )
    .sub(
        1.0
    )
    .abs()
    <
    1e-12
).all():

    print(
        "[PASS] No-Reputation weights sum to 1."
    )

else:

    print(
        "[FAIL] No-Reputation weights are not normalized."
    )


if (
    (
        shadai[
            "w1"
        ]
        +
        shadai[
            "w2"
        ]
        +
        shadai[
            "w3"
        ]
    )
    .sub(
        1.0
    )
    .abs()
    <
    1e-12
).all():

    print(
        "[PASS] SHADAI weights sum to 1."
    )

else:

    print(
        "[FAIL] SHADAI weights are not normalized."
    )


mean_shadai_exploration = (
    shadai[
        "exploration_rate"
    ].mean()
)


mean_norep_exploration = (
    norep[
        "exploration_rate"
    ].mean()
)


print(
    f"[INFO] SHADAI exploration rate : "
    f"{mean_shadai_exploration * 100:.2f}%"
)


print(
    f"[INFO] NoRep exploration rate  : "
    f"{mean_norep_exploration * 100:.2f}%"
)


if abs(
    mean_shadai_exploration
    -
    mean_norep_exploration
) < 0.01:

    print(
        "[PASS] Exploration behavior is aligned "
        "between SHADAI and No-Reputation."
    )

else:

    print(
        "[CHECK] Exploration rates differ unexpectedly."
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

    "avg_available_energy_at_selection",

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

        "n_runs":
            len(
                subset
            ),

        "w1":
            POLICIES[
                policy
            ][
                "w1"
            ],

        "w2":
            POLICIES[
                policy
            ][
                "w2"
            ],

        "w3":
            POLICIES[
                policy
            ][
                "w3"
            ]
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
# RESULTS TABLE
# ============================================================

print("\n")

print(
    "C23 MULTI-SEED RESULTS"
)

print("=" * 180)


print(

    f"{'POLICY':<10}"

    f"{'ALLOC%':>10}"

    f"{'COND%':>10}"

    f"{'REP':>10}"

    f"{'ENORM':>10}"

    f"{'PROX':>10}"

    f"{'DIST':>10}"

    f"{'ENERGY':>10}"

    f"{'UNIQUE':>10}"

    f"{'JAIN':>10}"

    f"{'SUCCESS%':>11}"
)


print("-" * 180)


for _, row in agg.iterrows():

    print(

        f"{row['policy']:<10}"

        f"{row['allocation_rate_mean'] * 100:>10.2f}"

        f"{row['conditional_allocation_rate_mean'] * 100:>10.2f}"

        f"{row['avg_reputation_norm_mean']:>10.4f}"

        f"{row['avg_energy_norm_mean']:>10.4f}"

        f"{row['avg_proximity_norm_mean']:>10.4f}"

        f"{row['avg_distance_mean']:>10.2f}"

        f"{row['avg_available_energy_at_selection_mean']:>10.2f}"

        f"{row['unique_selected_vehicles_mean']:>10.2f}"

        f"{row['jain_selected_vehicles_mean']:>10.4f}"

        f"{row['task_success_rate_mean'] * 100:>11.2f}"
    )


# ============================================================
# PAIRED ANALYSIS
# ============================================================

paired = pd.merge(

    shadai,

    norep,

    on="seed",

    suffixes=(
        "_shadai",
        "_norep"
    )
)


# IMPORTANT:
# avg_final_score is intentionally omitted from inferential
# paired comparisons because the score definition changes.
PAIRED_METRICS = [

    "allocation_rate",

    "conditional_allocation_rate",

    "avg_reputation_norm",

    "avg_energy_norm",

    "avg_proximity_norm",

    "avg_distance",

    "avg_available_energy_at_selection",

    "unique_selected_vehicles",

    "tasks_per_selected_vehicle",

    "jain_selected_vehicles",

    "task_success_rate"
]


paired_rows = []


print("\n")

print(
    "PAIRED DIFFERENCES: SHADAI - NO-REPUTATION"
)

print("=" * 180)


for metric in PAIRED_METRICS:

    difference = (

        paired[
            f"{metric}_shadai"
        ]

        -

        paired[
            f"{metric}_norep"
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

        "mean_difference_shadai_minus_norep":
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

        f"{metric:<40}"

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


norep_row = agg[
    agg[
        "policy"
    ]
    ==
    "norep"
].iloc[
    0
]


print("\n")

print(
    "C23 CORE SUMMARY"
)

print("=" * 180)


print(
    "SHADAI:"
)

print(
    "  weights                          : "
    "(0.4, 0.3, 0.3)"
)

print(
    f"  selected reputation              : "
    f"{shadai_row['avg_reputation_norm_mean']:.4f}"
)

print(
    f"  normalized selected energy       : "
    f"{shadai_row['avg_energy_norm_mean']:.4f}"
)

print(
    f"  normalized proximity             : "
    f"{shadai_row['avg_proximity_norm_mean']:.4f}"
)

print(
    f"  distance                         : "
    f"{shadai_row['avg_distance_mean']:.2f}"
)

print(
    f"  available energy at selection    : "
    f"{shadai_row['avg_available_energy_at_selection_mean']:.2f}"
)

print(
    f"  Jain fairness                    : "
    f"{shadai_row['jain_selected_vehicles_mean']:.4f}"
)


print(
    "\nNo-Reputation:"
)

print(
    "  weights                          : "
    "(0.0, 0.5, 0.5)"
)

print(
    f"  selected reputation              : "
    f"{norep_row['avg_reputation_norm_mean']:.4f}"
)

print(
    f"  normalized selected energy       : "
    f"{norep_row['avg_energy_norm_mean']:.4f}"
)

print(
    f"  normalized proximity             : "
    f"{norep_row['avg_proximity_norm_mean']:.4f}"
)

print(
    f"  distance                         : "
    f"{norep_row['avg_distance_mean']:.2f}"
)

print(
    f"  available energy at selection    : "
    f"{norep_row['avg_available_energy_at_selection_mean']:.2f}"
)

print(
    f"  Jain fairness                    : "
    f"{norep_row['jain_selected_vehicles_mean']:.4f}"
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
    r"\caption{Comparison between SHADAI and the renormalized No-Reputation baseline. The reputation term is removed in No-Reputation and the remaining energy and proximity weights are set to 0.5 each.}"
)

latex.append(
    r"\label{tab:c23_norep}"
)

latex.append(
    r"\begin{tabular}{lcccccc}"
)

latex.append(
    r"\hline"
)

latex.append(
    r"Policy & Rep. & Energy & Prox. & Distance & Jain & Success \\"
)

latex.append(
    r"\hline"
)


latex.append(

    f"SHADAI & "

    f"{shadai_row['avg_reputation_norm_mean']:.4f} & "

    f"{shadai_row['avg_energy_norm_mean']:.4f} & "

    f"{shadai_row['avg_proximity_norm_mean']:.4f} & "

    f"{shadai_row['avg_distance_mean']:.2f} & "

    f"{shadai_row['jain_selected_vehicles_mean']:.4f} & "

    f"{shadai_row['task_success_rate_mean'] * 100:.2f}\\% "

    r"\\"
)


latex.append(

    f"No Reputation & "

    f"{norep_row['avg_reputation_norm_mean']:.4f} & "

    f"{norep_row['avg_energy_norm_mean']:.4f} & "

    f"{norep_row['avg_proximity_norm_mean']:.4f} & "

    f"{norep_row['avg_distance_mean']:.2f} & "

    f"{norep_row['jain_selected_vehicles_mean']:.4f} & "

    f"{norep_row['task_success_rate_mean'] * 100:.2f}\\% "

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


print("\n")

print("=" * 180)

print(
    "C23 SHADAI VS NO-REPUTATION ANALYSIS COMPLETED"
)

print("=" * 180)

print()