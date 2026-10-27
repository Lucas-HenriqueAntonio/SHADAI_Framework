# analisar_task_success_failure.py

import os
import math

import pandas as pd


try:

    from scipy.stats import t

except ImportError:

    t = None


# ============================================================
# DIRETÓRIOS
# ============================================================

MALICIOUS_DIR = "malicious_ablation"

FLOOD_DIR = "flood_ablation"

OUTPUT_DIR = "task_reliability_analysis"


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURAÇÕES C17
# ============================================================

MALICIOUS_CONFIGS = {

    "malicious_00": 0.00,

    "malicious_10": 0.10,

    "malicious_20": 0.20,

    "malicious_30": 0.30,

    "malicious_40": 0.40
}


# ============================================================
# CONFIGURAÇÕES C18
# ============================================================

FLOOD_CONFIGS = {

    "flood_00": 0,

    "flood_01": 1,

    "flood_02": 2,

    "flood_03": 3,

    "flood_05": 5
}


SEEDS = list(
    range(
        1,
        31
    )
)


# ============================================================
# OUTPUT FILES
# ============================================================

PER_SEED_FILE = os.path.join(
    OUTPUT_DIR,
    "task_reliability_por_seed.csv"
)

AGG_FILE = os.path.join(
    OUTPUT_DIR,
    "task_reliability_multiseed.csv"
)

PAIRED_FILE = os.path.join(
    OUTPUT_DIR,
    "task_reliability_paired.csv"
)

LATEX_FILE = os.path.join(
    OUTPUT_DIR,
    "tabela_task_reliability_tnsm.tex"
)


# ============================================================
# HELPERS
# ============================================================

def to_bool(
    series
):

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


def ci95(
    values
):

    s = pd.Series(
        values
    ).dropna()


    n = len(
        s
    )


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
        math.sqrt(
            n
        )
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


def reliability_from_outcomes(
    outcomes
):

    outcomes = outcomes.copy()


    outcomes[
        "success_bool"
    ] = to_bool(
        outcomes[
            "success"
        ]
    )


    completed = len(
        outcomes
    )


    successful = int(
        outcomes[
            "success_bool"
        ].sum()
    )


    failed = (
        completed
        -
        successful
    )


    success_rate = (

        successful
        /
        completed

        if completed > 0

        else float("nan")
    )


    failure_rate = (

        failed
        /
        completed

        if completed > 0

        else float("nan")
    )


    # ========================================================
    # HONEST / MALICIOUS SPLIT
    # ========================================================

    malicious_success_rate = float(
        "nan"
    )

    malicious_failure_rate = float(
        "nan"
    )

    honest_success_rate = float(
        "nan"
    )

    honest_failure_rate = float(
        "nan"
    )


    malicious_completed = 0
    honest_completed = 0


    if "malicious" in outcomes.columns:

        outcomes[
            "malicious_bool"
        ] = to_bool(
            outcomes[
                "malicious"
            ]
        )


        malicious = outcomes[
            outcomes[
                "malicious_bool"
            ]
        ]


        honest = outcomes[
            ~outcomes[
                "malicious_bool"
            ]
        ]


        malicious_completed = len(
            malicious
        )


        honest_completed = len(
            honest
        )


        if malicious_completed > 0:

            malicious_success_rate = (
                malicious[
                    "success_bool"
                ].mean()
            )


            malicious_failure_rate = (
                1.0
                -
                malicious_success_rate
            )


        if honest_completed > 0:

            honest_success_rate = (
                honest[
                    "success_bool"
                ].mean()
            )


            honest_failure_rate = (
                1.0
                -
                honest_success_rate
            )


    return {

        "completed_tasks":
            completed,

        "successful_tasks":
            successful,

        "failed_tasks":
            failed,

        "task_success_rate":
            success_rate,

        "task_failure_rate":
            failure_rate,

        "malicious_completed_tasks":
            malicious_completed,

        "honest_completed_tasks":
            honest_completed,

        "malicious_task_success_rate":
            malicious_success_rate,

        "malicious_task_failure_rate":
            malicious_failure_rate,

        "honest_task_success_rate":
            honest_success_rate,

        "honest_task_failure_rate":
            honest_failure_rate
    }


# ============================================================
# HEADER
# ============================================================

print("\n")

print("=" * 180)

print(
    "SHADAI - C19 TASK SUCCESS / FAILURE RATE ANALYSIS"
)

print("=" * 180)


print(
    "Definition:"
)

print(
    "Task Success Rate = successful completed tasks / completed tasks"
)

print(
    "Task Failure Rate = failed completed tasks / completed tasks"
)

print(
    "Allocation/admission failures are excluded from these task-execution rates."
)

print("=" * 180)


# ============================================================
# LOAD C17
# ============================================================

rows = []

missing = []


for config, ratio in MALICIOUS_CONFIGS.items():

    for seed in SEEDS:

        experiment = (

            f"{config}_"
            f"ratio_{int(ratio * 100):02d}_"
            f"seed_{seed:02d}"
        )


        outcome_file = os.path.join(

            MALICIOUS_DIR,

            (
                f"outcomes_shadai_400_"
                f"{experiment}.csv"
            )
        )


        if not os.path.exists(
            outcome_file
        ):

            missing.append(
                (
                    "C17",
                    config,
                    seed
                )
            )

            continue


        outcomes = pd.read_csv(
            outcome_file
        )


        reliability = (
            reliability_from_outcomes(
                outcomes
            )
        )


        row = {

            "experiment_group":
                "malicious_ratio",

            "configuration":
                config,

            "parameter_value":
                ratio,

            "seed":
                seed
        }


        row.update(
            reliability
        )


        rows.append(
            row
        )


# ============================================================
# LOAD C18
# ============================================================

for config, intensity in FLOOD_CONFIGS.items():

    for seed in SEEDS:

        experiment = (

            f"{config}_"
            f"intensity_{intensity:02d}_"
            f"seed_{seed:02d}"
        )


        outcome_file = os.path.join(

            FLOOD_DIR,

            (
                f"outcomes_shadai_400_"
                f"{experiment}.csv"
            )
        )


        if not os.path.exists(
            outcome_file
        ):

            missing.append(
                (
                    "C18",
                    config,
                    seed
                )
            )

            continue


        outcomes = pd.read_csv(
            outcome_file
        )


        reliability = (
            reliability_from_outcomes(
                outcomes
            )
        )


        row = {

            "experiment_group":
                "flood_intensity",

            "configuration":
                config,

            "parameter_value":
                intensity,

            "seed":
                seed
        }


        row.update(
            reliability
        )


        rows.append(
            row
        )


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    rows
)


if df.empty:

    raise RuntimeError(
        "No C17/C18 outcomes files were found."
    )


df.to_csv(
    PER_SEED_FILE,
    index=False
)


# ============================================================
# COMPLETUDE
# ============================================================

print("\n")

print(
    "DATASET COMPLETENESS"
)

print("-" * 180)


for config in MALICIOUS_CONFIGS:

    n = len(

        df[
            (
                df[
                    "experiment_group"
                ]
                ==
                "malicious_ratio"
            )
            &
            (
                df[
                    "configuration"
                ]
                ==
                config
            )
        ]
    )


    print(
        f"C17 {config:<15}: "
        f"{n}/30"
    )


for config in FLOOD_CONFIGS:

    n = len(

        df[
            (
                df[
                    "experiment_group"
                ]
                ==
                "flood_intensity"
            )
            &
            (
                df[
                    "configuration"
                ]
                ==
                config
            )
        ]
    )


    print(
        f"C18 {config:<15}: "
        f"{n}/30"
    )


if missing:

    print(
        f"\n[WARNING] Missing runs: "
        f"{len(missing)}"
    )


# ============================================================
# SANITY CHECK: SUCCESS + FAILURE = 1
# ============================================================

deviation = (

    df[
        "task_success_rate"
    ]

    +

    df[
        "task_failure_rate"
    ]

    -

    1.0
).abs()


print("\n")

print(
    "TASK RELIABILITY SANITY CHECK"
)

print("=" * 180)


if (
    deviation.dropna()
    <=
    1e-12
).all():

    print(
        "[PASS] task_success_rate + task_failure_rate = 1 "
        "for every completed run."
    )

else:

    print(
        "[FAIL] Success/failure rates do not sum to 1."
    )


counts_ok = (

    df[
        "completed_tasks"
    ]

    ==

    (
        df[
            "successful_tasks"
        ]

        +

        df[
            "failed_tasks"
        ]
    )
)


if counts_ok.all():

    print(
        "[PASS] completed_tasks = successful_tasks + failed_tasks "
        "for every run."
    )

else:

    print(
        "[FAIL] Task counts are inconsistent."
    )


# ============================================================
# AGGREGATE
# ============================================================

METRICS = [

    "completed_tasks",

    "successful_tasks",

    "failed_tasks",

    "task_success_rate",

    "task_failure_rate",

    "malicious_task_success_rate",

    "malicious_task_failure_rate",

    "honest_task_success_rate",

    "honest_task_failure_rate"
]


aggregate_rows = []


for group in [
    "malicious_ratio",
    "flood_intensity"
]:

    subset_group = df[
        df[
            "experiment_group"
        ]
        ==
        group
    ]


    configurations = (

        MALICIOUS_CONFIGS.keys()

        if group == "malicious_ratio"

        else FLOOD_CONFIGS.keys()
    )


    for config in configurations:

        subset = subset_group[
            subset_group[
                "configuration"
            ]
            ==
            config
        ]


        if subset.empty:

            continue


        row = {

            "experiment_group":
                group,

            "configuration":
                config,

            "parameter_value":
                subset[
                    "parameter_value"
                ].iloc[
                    0
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


        aggregate_rows.append(
            row
        )


agg = pd.DataFrame(
    aggregate_rows
)


agg.to_csv(
    AGG_FILE,
    index=False
)


# ============================================================
# C17 RESULTS
# ============================================================

print("\n")

print(
    "C17 - TASK RELIABILITY VS MALICIOUS-VEHICLE RATIO"
)

print("=" * 180)


print(

    f"{'MAL%':>6}"

    f"{'COMPLETED':>12}"

    f"{'SUCCESS%':>12}"

    f"{'FAILURE%':>12}"

    f"{'HON SUC%':>12}"

    f"{'MAL SUC%':>12}"
)


print("-" * 180)


c17_agg = agg[
    agg[
        "experiment_group"
    ]
    ==
    "malicious_ratio"
].sort_values(
    "parameter_value"
)


for _, row in c17_agg.iterrows():

    mal_success = row[
        "malicious_task_success_rate_mean"
    ]


    mal_success_text = (

        f"{mal_success * 100:.2f}"

        if not pd.isna(
            mal_success
        )

        else "NA"
    )


    print(

        f"{row['parameter_value'] * 100:>5.0f}%"

        f"{row['completed_tasks_mean']:>12.2f}"

        f"{row['task_success_rate_mean'] * 100:>12.2f}"

        f"{row['task_failure_rate_mean'] * 100:>12.2f}"

        f"{row['honest_task_success_rate_mean'] * 100:>12.2f}"

        f"{mal_success_text:>12}"
    )


# ============================================================
# C18 RESULTS
# ============================================================

print("\n")

print(
    "C18 - TASK RELIABILITY VS FLOOD INTENSITY"
)

print("=" * 180)


print(

    f"{'INT':>6}"

    f"{'COMPLETED':>12}"

    f"{'SUCCESS%':>12}"

    f"{'FAILURE%':>12}"

    f"{'HON SUC%':>12}"

    f"{'MAL SUC%':>12}"
)


print("-" * 180)


c18_agg = agg[
    agg[
        "experiment_group"
    ]
    ==
    "flood_intensity"
].sort_values(
    "parameter_value"
)


for _, row in c18_agg.iterrows():

    print(

        f"{int(row['parameter_value']):>6}"

        f"{row['completed_tasks_mean']:>12.2f}"

        f"{row['task_success_rate_mean'] * 100:>12.2f}"

        f"{row['task_failure_rate_mean'] * 100:>12.2f}"

        f"{row['honest_task_success_rate_mean'] * 100:>12.2f}"

        f"{row['malicious_task_success_rate_mean'] * 100:>12.2f}"
    )


# ============================================================
# PAIRED COMPARISONS
# ============================================================

paired_rows = []


PAIRED_METRICS = [

    "task_success_rate",

    "task_failure_rate",

    "honest_task_success_rate"
]


# ============================================================
# C17 VS 0%
# ============================================================

baseline_c17 = df[
    (
        df[
            "experiment_group"
        ]
        ==
        "malicious_ratio"
    )
    &
    (
        df[
            "configuration"
        ]
        ==
        "malicious_00"
    )
].copy()


for config, ratio in MALICIOUS_CONFIGS.items():

    if config == "malicious_00":

        continue


    alt = df[
        (
            df[
                "experiment_group"
            ]
            ==
            "malicious_ratio"
        )
        &
        (
            df[
                "configuration"
            ]
            ==
            config
        )
    ].copy()


    paired = pd.merge(

        baseline_c17,

        alt,

        on="seed",

        suffixes=(
            "_baseline",
            "_alternative"
        )
    )


    for metric in PAIRED_METRICS:

        difference = (

            paired[
                f"{metric}_alternative"
            ]

            -

            paired[
                f"{metric}_baseline"
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


        paired_rows.append({

            "experiment_group":
                "malicious_ratio",

            "configuration":
                config,

            "parameter_value":
                ratio,

            "metric":
                metric,

            "n_pairs":
                len(
                    paired
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

                (
                    low > 0
                    or
                    high < 0
                )

                if (
                    not pd.isna(
                        low
                    )
                    and
                    not pd.isna(
                        high
                    )
                )

                else False
        })


# ============================================================
# C18 VS INTENSITY 0
# ============================================================

baseline_c18 = df[
    (
        df[
            "experiment_group"
        ]
        ==
        "flood_intensity"
    )
    &
    (
        df[
            "configuration"
        ]
        ==
        "flood_00"
    )
].copy()


for config, intensity in FLOOD_CONFIGS.items():

    if config == "flood_00":

        continue


    alt = df[
        (
            df[
                "experiment_group"
            ]
            ==
            "flood_intensity"
        )
        &
        (
            df[
                "configuration"
            ]
            ==
            config
        )
    ].copy()


    paired = pd.merge(

        baseline_c18,

        alt,

        on="seed",

        suffixes=(
            "_baseline",
            "_alternative"
        )
    )


    for metric in PAIRED_METRICS:

        difference = (

            paired[
                f"{metric}_alternative"
            ]

            -

            paired[
                f"{metric}_baseline"
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


        paired_rows.append({

            "experiment_group":
                "flood_intensity",

            "configuration":
                config,

            "parameter_value":
                intensity,

            "metric":
                metric,

            "n_pairs":
                len(
                    paired
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

                (
                    low > 0
                    or
                    high < 0
                )

                if (
                    not pd.isna(
                        low
                    )
                    and
                    not pd.isna(
                        high
                    )
                )

                else False
        })


paired_df = pd.DataFrame(
    paired_rows
)


paired_df.to_csv(
    PAIRED_FILE,
    index=False
)


# ============================================================
# PAIRED PRINT C17
# ============================================================

print("\n")

print(
    "C17 PAIRED TASK-RELIABILITY COMPARISONS VS 0% MALICIOUS"
)

print("=" * 180)


for config, ratio in MALICIOUS_CONFIGS.items():

    if config == "malicious_00":

        continue


    print(
        f"\nMalicious ratio = "
        f"{ratio * 100:.0f}%"
    )


    subset = paired_df[
        (
            paired_df[
                "experiment_group"
            ]
            ==
            "malicious_ratio"
        )
        &
        (
            paired_df[
                "configuration"
            ]
            ==
            config
        )
    ]


    for _, row in subset.iterrows():

        print(

            f"  "
            f"{row['metric']:<30} "

            f"diff="
            f"{row['mean_difference']:+.5f} | "

            f"CI95=["
            f"{row['ci95_low']:+.5f}, "
            f"{row['ci95_high']:+.5f}] | "

            f"excludes zero="
            f"{row['ci95_excludes_zero']}"
        )


# ============================================================
# PAIRED PRINT C18
# ============================================================

print("\n")

print(
    "C18 PAIRED TASK-RELIABILITY COMPARISONS VS FLOOD INTENSITY 0"
)

print("=" * 180)


for config, intensity in FLOOD_CONFIGS.items():

    if config == "flood_00":

        continue


    print(
        f"\nFlood intensity = "
        f"{intensity}"
    )


    subset = paired_df[
        (
            paired_df[
                "experiment_group"
            ]
            ==
            "flood_intensity"
        )
        &
        (
            paired_df[
                "configuration"
            ]
            ==
            config
        )
    ]


    for _, row in subset.iterrows():

        print(

            f"  "
            f"{row['metric']:<30} "

            f"diff="
            f"{row['mean_difference']:+.5f} | "

            f"CI95=["
            f"{row['ci95_low']:+.5f}, "
            f"{row['ci95_high']:+.5f}] | "

            f"excludes zero="
            f"{row['ci95_excludes_zero']}"
        )


# ============================================================
# C19 CORE SUMMARY
# ============================================================

baseline = c17_agg[
    c17_agg[
        "parameter_value"
    ]
    ==
    0.0
].iloc[
    0
]


mal40 = c17_agg[
    c17_agg[
        "parameter_value"
    ]
    ==
    0.40
].iloc[
    0
]


flood3 = c18_agg[
    c18_agg[
        "parameter_value"
    ]
    ==
    3
].iloc[
    0
]


print("\n")

print(
    "C19 CORE SUMMARY"
)

print("=" * 180)


print(
    "No deliberate-failure attack:"
)

print(
    f"  Task success rate : "
    f"{baseline['task_success_rate_mean'] * 100:.2f}%"
)

print(
    f"  Task failure rate : "
    f"{baseline['task_failure_rate_mean'] * 100:.2f}%"
)


print(
    "\n40% deliberate-failure malicious vehicles:"
)

print(
    f"  Task success rate : "
    f"{mal40['task_success_rate_mean'] * 100:.2f}%"
)

print(
    f"  Task failure rate : "
    f"{mal40['task_failure_rate_mean'] * 100:.2f}%"
)


print(
    "\nTable II Flood Attack intensity = 3:"
)

print(
    f"  Task success rate : "
    f"{flood3['task_success_rate_mean'] * 100:.2f}%"
)

print(
    f"  Task failure rate : "
    f"{flood3['task_failure_rate_mean'] * 100:.2f}%"
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
    r"\caption{Task execution reliability under the evaluated adversarial conditions. Task success and failure rates are computed only over completed task executions; unallocated service requests are reported separately as allocation failures.}"
)

latex.append(
    r"\label{tab:task_reliability}"
)

latex.append(
    r"\begin{tabular}{lcc}"
)

latex.append(
    r"\hline"
)

latex.append(
    r"Scenario & Task Success & Task Failure \\"
)

latex.append(
    r"\hline"
)


latex.append(

    f"No malicious failure & "

    f"{baseline['task_success_rate_mean'] * 100:.2f}\\% & "

    f"{baseline['task_failure_rate_mean'] * 100:.2f}\\% "

    r"\\"
)


latex.append(

    f"Failure attack (20\\%) & "

    f"{c17_agg[c17_agg['parameter_value'] == 0.20].iloc[0]['task_success_rate_mean'] * 100:.2f}\\% & "

    f"{c17_agg[c17_agg['parameter_value'] == 0.20].iloc[0]['task_failure_rate_mean'] * 100:.2f}\\% "

    r"\\"
)


latex.append(

    f"Failure attack (40\\%) & "

    f"{mal40['task_success_rate_mean'] * 100:.2f}\\% & "

    f"{mal40['task_failure_rate_mean'] * 100:.2f}\\% "

    r"\\"
)


latex.append(

    f"Flood attack (intensity 3) & "

    f"{flood3['task_success_rate_mean'] * 100:.2f}\\% & "

    f"{flood3['task_failure_rate_mean'] * 100:.2f}\\% "

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
    "C19 TASK SUCCESS / FAILURE ANALYSIS COMPLETED"
)

print("=" * 180)

print()