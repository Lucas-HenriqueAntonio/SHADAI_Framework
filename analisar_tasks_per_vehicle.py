# analisar_tasks_per_vehicle.py

import os
import math

import pandas as pd


try:
    from scipy.stats import t
except ImportError:
    t = None


# ============================================================
# CONFIGURATION
# ============================================================

MALICIOUS_DIR = "malicious_ablation"
FLOOD_DIR = "flood_ablation"

OUTPUT_DIR = "task_distribution_analysis"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


SEEDS = list(
    range(1, 31)
)


# ============================================================
# C17 CONFIGURATIONS
# ============================================================

MALICIOUS_CONFIGS = {

    "malicious_00": 0.00,

    "malicious_10": 0.10,

    "malicious_20": 0.20,

    "malicious_30": 0.30,

    "malicious_40": 0.40
}


# ============================================================
# C18 CONFIGURATIONS
# ============================================================

FLOOD_CONFIGS = {

    "flood_00": 0,

    "flood_01": 1,

    "flood_02": 2,

    "flood_03": 3,

    "flood_05": 5
}


# ============================================================
# OUTPUT FILES
# ============================================================

RAW_FILE = os.path.join(
    OUTPUT_DIR,
    "tasks_per_vehicle_raw.csv"
)

PER_SEED_FILE = os.path.join(
    OUTPUT_DIR,
    "tasks_per_vehicle_por_seed.csv"
)

AGG_FILE = os.path.join(
    OUTPUT_DIR,
    "tasks_per_vehicle_multiseed.csv"
)

PAIRED_FILE = os.path.join(
    OUTPUT_DIR,
    "tasks_per_vehicle_paired.csv"
)

LATEX_FILE = os.path.join(
    OUTPUT_DIR,
    "tabela_tasks_per_vehicle_tnsm.tex"
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
    ).dropna()

    values = values[
        values > 0
    ]

    n = len(values)

    if n == 0:

        return float("nan")

    numerator = (
        values.sum()
        ** 2
    )

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
        numerator
        /
        denominator
    )


def safe_share(
    numerator,
    denominator
):

    if denominator <= 0:

        return float("nan")

    return (
        numerator
        /
        denominator
    )


# ============================================================
# PROCESS ONE RUN
# ============================================================

def process_run(

    metrics_path,

    outcomes_path,

    experiment_group,

    configuration,

    parameter_value,

    seed
):

    metrics = pd.read_csv(
        metrics_path
    )

    outcomes = pd.read_csv(
        outcomes_path
    )


    # ========================================================
    # VALIDATE METRICS
    # ========================================================

    if "veiculo" not in metrics.columns:

        raise RuntimeError(
            f"Missing column 'veiculo' in {metrics_path}"
        )


    # One metrics row = one allocated task.
    metrics[
        "vehicle_id"
    ] = (
        metrics[
            "veiculo"
        ]
        .astype(str)
        .str.strip()
    )


    metrics = metrics[
        metrics[
            "vehicle_id"
        ]
        !=
        ""
    ].copy()


    # ========================================================
    # ALLOCATED TASKS PER VEHICLE
    # ========================================================

    allocated_counts = (

        metrics
        .groupby(
            "vehicle_id"
        )
        .size()
        .rename(
            "tasks_allocated"
        )
    )


    # ========================================================
    # OUTCOMES / COMPLETED TASKS
    # ========================================================

    if "vehicle" not in outcomes.columns:

        raise RuntimeError(
            f"Missing column 'vehicle' in {outcomes_path}"
        )


    outcomes[
        "vehicle_id"
    ] = (
        outcomes[
            "vehicle"
        ]
        .astype(str)
        .str.strip()
    )


    completed_counts = (

        outcomes
        .groupby(
            "vehicle_id"
        )
        .size()
        .rename(
            "tasks_completed"
        )
    )


    # ========================================================
    # SUCCESS / FAILURE COUNTS
    # ========================================================

    if "success" in outcomes.columns:

        outcomes[
            "success_bool"
        ] = to_bool(
            outcomes[
                "success"
            ]
        )

    else:

        outcomes[
            "success_bool"
        ] = False


    success_counts = (

        outcomes[
            outcomes[
                "success_bool"
            ]
        ]
        .groupby(
            "vehicle_id"
        )
        .size()
        .rename(
            "tasks_successful"
        )
    )


    failure_counts = (

        outcomes[
            ~outcomes[
                "success_bool"
            ]
        ]
        .groupby(
            "vehicle_id"
        )
        .size()
        .rename(
            "tasks_failed"
        )
    )


    # ========================================================
    # VEHICLE TABLE
    # ========================================================

    vehicle_table = pd.concat(

        [
            allocated_counts,
            completed_counts,
            success_counts,
            failure_counts
        ],

        axis=1

    ).fillna(
        0
    )


    for column in [
        "tasks_allocated",
        "tasks_completed",
        "tasks_successful",
        "tasks_failed"
    ]:

        vehicle_table[
            column
        ] = (
            vehicle_table[
                column
            ]
            .astype(int)
        )


    vehicle_table[
        "experiment_group"
    ] = experiment_group


    vehicle_table[
        "configuration"
    ] = configuration


    vehicle_table[
        "parameter_value"
    ] = parameter_value


    vehicle_table[
        "seed"
    ] = seed


    vehicle_table[
        "vehicle_id"
    ] = vehicle_table.index


    # ========================================================
    # MALICIOUS FLAG
    # ========================================================

    malicious_map = {}


    if (
        "selected_malicious"
        in metrics.columns
    ):

        temp = metrics[[
            "vehicle_id",
            "selected_malicious"
        ]].copy()


        temp[
            "malicious_bool"
        ] = to_bool(
            temp[
                "selected_malicious"
            ]
        )


        malicious_map = (

            temp
            .groupby(
                "vehicle_id"
            )[
                "malicious_bool"
            ]
            .max()
            .to_dict()
        )


    vehicle_table[
        "malicious"
    ] = (

        vehicle_table[
            "vehicle_id"
        ]
        .map(
            malicious_map
        )
        .fillna(
            False
        )
        .astype(bool)
    )


    # ========================================================
    # PER-VEHICLE SHARES
    # ========================================================

    total_allocated = int(
        vehicle_table[
            "tasks_allocated"
        ].sum()
    )


    if total_allocated > 0:

        vehicle_table[
            "allocation_share"
        ] = (

            vehicle_table[
                "tasks_allocated"
            ]

            /

            total_allocated
        )

    else:

        vehicle_table[
            "allocation_share"
        ] = float(
            "nan"
        )


    # ========================================================
    # SELECTED VEHICLES ONLY
    # ========================================================

    selected = vehicle_table[
        vehicle_table[
            "tasks_allocated"
        ]
        >
        0
    ].copy()


    tasks = (
        selected[
            "tasks_allocated"
        ]
    )


    unique_selected = len(
        selected
    )


    total_completed = int(
        vehicle_table[
            "tasks_completed"
        ].sum()
    )


    total_successful = int(
        vehicle_table[
            "tasks_successful"
        ].sum()
    )


    total_failed = int(
        vehicle_table[
            "tasks_failed"
        ].sum()
    )


    # ========================================================
    # DISTRIBUTION
    # ========================================================

    if unique_selected > 0:

        mean_tasks = (
            tasks.mean()
        )


        median_tasks = (
            tasks.median()
        )


        sd_tasks = (
            tasks.std(
                ddof=1
            )
            if unique_selected > 1
            else 0.0
        )


        min_tasks = (
            tasks.min()
        )


        max_tasks = (
            tasks.max()
        )


        p25_tasks = (
            tasks.quantile(
                0.25
            )
        )


        p75_tasks = (
            tasks.quantile(
                0.75
            )
        )


        jain = jain_index(
            tasks
        )


        sorted_tasks = (
            tasks
            .sort_values(
                ascending=False
            )
        )


        top1_tasks = (
            sorted_tasks
            .head(1)
            .sum()
        )


        top5_tasks = (
            sorted_tasks
            .head(5)
            .sum()
        )


        top1_share = safe_share(
            top1_tasks,
            total_allocated
        )


        top5_share = safe_share(
            top5_tasks,
            total_allocated
        )


        repeated_vehicle_count = int(
            (
                tasks > 1
            ).sum()
        )


        repeated_vehicle_share = (

            repeated_vehicle_count
            /
            unique_selected
        )


        single_task_vehicle_share = (

            (
                tasks == 1
            ).sum()

            /
            unique_selected
        )


        # Fraction of all allocated tasks that are beyond
        # each selected vehicle's first allocation.
        repeated_allocations = (

            total_allocated
            -
            unique_selected
        )


        repeated_allocation_share = safe_share(
            repeated_allocations,
            total_allocated
        )


    else:

        mean_tasks = float("nan")
        median_tasks = float("nan")
        sd_tasks = float("nan")

        min_tasks = float("nan")
        max_tasks = float("nan")

        p25_tasks = float("nan")
        p75_tasks = float("nan")

        jain = float("nan")

        top1_share = float("nan")
        top5_share = float("nan")

        repeated_vehicle_count = 0

        repeated_vehicle_share = float(
            "nan"
        )

        single_task_vehicle_share = float(
            "nan"
        )

        repeated_allocations = 0

        repeated_allocation_share = float(
            "nan"
        )


    # ========================================================
    # MALICIOUS / HONEST DISTRIBUTION
    # ========================================================

    malicious_selected = selected[
        selected[
            "malicious"
        ]
    ]


    honest_selected = selected[
        ~selected[
            "malicious"
        ]
    ]


    malicious_tasks = int(
        malicious_selected[
            "tasks_allocated"
        ].sum()
    )


    honest_tasks = int(
        honest_selected[
            "tasks_allocated"
        ].sum()
    )


    malicious_task_share = safe_share(
        malicious_tasks,
        total_allocated
    )


    # ========================================================
    # COMPLETION CONSISTENCY
    # ========================================================

    allocation_completion_gap = (

        total_allocated
        -
        total_completed
    )


    completion_ratio = safe_share(
        total_completed,
        total_allocated
    )


    # ========================================================
    # PER-SEED OUTPUT
    # ========================================================

    row = {

        "experiment_group":
            experiment_group,

        "configuration":
            configuration,

        "parameter_value":
            parameter_value,

        "seed":
            seed,

        "total_allocated_tasks":
            total_allocated,

        "total_completed_tasks":
            total_completed,

        "total_successful_tasks":
            total_successful,

        "total_failed_tasks":
            total_failed,

        "allocation_completion_gap":
            allocation_completion_gap,

        "completion_ratio":
            completion_ratio,

        "unique_selected_vehicles":
            unique_selected,

        "mean_tasks_per_selected_vehicle":
            mean_tasks,

        "median_tasks_per_selected_vehicle":
            median_tasks,

        "sd_tasks_per_selected_vehicle":
            sd_tasks,

        "min_tasks_per_selected_vehicle":
            min_tasks,

        "max_tasks_per_selected_vehicle":
            max_tasks,

        "p25_tasks_per_selected_vehicle":
            p25_tasks,

        "p75_tasks_per_selected_vehicle":
            p75_tasks,

        "jain_tasks_selected_only":
            jain,

        "top1_task_share":
            top1_share,

        "top5_task_share":
            top5_share,

        "repeated_vehicle_count":
            repeated_vehicle_count,

        "repeated_vehicle_share":
            repeated_vehicle_share,

        "single_task_vehicle_share":
            single_task_vehicle_share,

        "repeated_allocations":
            repeated_allocations,

        "repeated_allocation_share":
            repeated_allocation_share,

        "malicious_selected_vehicles":
            len(
                malicious_selected
            ),

        "honest_selected_vehicles":
            len(
                honest_selected
            ),

        "malicious_allocated_tasks":
            malicious_tasks,

        "honest_allocated_tasks":
            honest_tasks,

        "malicious_task_share":
            malicious_task_share
    }


    vehicle_table = (
        vehicle_table
        .reset_index(
            drop=True
        )
    )


    return (
        row,
        vehicle_table
    )


# ============================================================
# HEADER
# ============================================================

print("\n")

print("=" * 190)

print(
    "SHADAI - C21 TASKS PER VEHICLE / CONCENTRATION ANALYSIS"
)

print("=" * 190)


print(
    "Primary unit:"
)

print(
    "  number of allocated tasks per selected physical vehicle"
)


print(
    "Fairness definition:"
)

print(
    "  Jain index is computed over vehicles receiving at least "
    "one task, consistent with the existing SHADAI fairness metric."
)


print(
    "Concentration metrics:"
)

print(
    "  top-1 share, top-5 share, repeated-selection share, "
    "maximum tasks per selected vehicle"
)

print("=" * 190)


# ============================================================
# LOAD ALL RUNS
# ============================================================

rows = []

raw_frames = []

missing = []


# ============================================================
# C17
# ============================================================

for config, ratio in MALICIOUS_CONFIGS.items():

    for seed in SEEDS:

        experiment = (
            f"{config}_"
            f"ratio_{int(ratio * 100):02d}_"
            f"seed_{seed:02d}"
        )


        metrics_path = os.path.join(

            MALICIOUS_DIR,

            (
                f"metricas_shadai_400_"
                f"{experiment}.csv"
            )
        )


        outcomes_path = os.path.join(

            MALICIOUS_DIR,

            (
                f"outcomes_shadai_400_"
                f"{experiment}.csv"
            )
        )


        if not (
            os.path.exists(
                metrics_path
            )
            and
            os.path.exists(
                outcomes_path
            )
        ):

            missing.append(
                (
                    "C17",
                    config,
                    seed
                )
            )

            continue


        row, raw = process_run(

            metrics_path,

            outcomes_path,

            "malicious_ratio",

            config,

            ratio,

            seed
        )


        rows.append(
            row
        )


        raw_frames.append(
            raw
        )


# ============================================================
# C18
# ============================================================

for config, intensity in FLOOD_CONFIGS.items():

    for seed in SEEDS:

        experiment = (
            f"{config}_"
            f"intensity_{intensity:02d}_"
            f"seed_{seed:02d}"
        )


        metrics_path = os.path.join(

            FLOOD_DIR,

            (
                f"metricas_shadai_400_"
                f"{experiment}.csv"
            )
        )


        outcomes_path = os.path.join(

            FLOOD_DIR,

            (
                f"outcomes_shadai_400_"
                f"{experiment}.csv"
            )
        )


        if not (
            os.path.exists(
                metrics_path
            )
            and
            os.path.exists(
                outcomes_path
            )
        ):

            missing.append(
                (
                    "C18",
                    config,
                    seed
                )
            )

            continue


        row, raw = process_run(

            metrics_path,

            outcomes_path,

            "flood_intensity",

            config,

            intensity,

            seed
        )


        rows.append(
            row
        )


        raw_frames.append(
            raw
        )


# ============================================================
# DATAFRAMES
# ============================================================

df = pd.DataFrame(
    rows
)


if df.empty:

    raise RuntimeError(
        "No C17/C18 task-allocation files found."
    )


raw_df = pd.concat(
    raw_frames,
    ignore_index=True
)


df.to_csv(
    PER_SEED_FILE,
    index=False
)


raw_df.to_csv(
    RAW_FILE,
    index=False
)


# ============================================================
# COMPLETENESS
# ============================================================

print("\n")

print(
    "DATASET COMPLETENESS"
)

print("-" * 190)


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
# SANITY CHECK
# ============================================================

print("\n")

print(
    "C21 SANITY CHECK"
)

print("=" * 190)


if (
    df[
        "total_allocated_tasks"
    ]
    >=
    df[
        "unique_selected_vehicles"
    ]
).all():

    print(
        "[PASS] Allocated task count is never lower than the "
        "number of selected vehicles."
    )

else:

    print(
        "[FAIL] Invalid task/vehicle count detected."
    )


if (
    raw_df[
        "tasks_allocated"
    ]
    >=
    0
).all():

    print(
        "[PASS] Per-vehicle task counts are non-negative."
    )

else:

    print(
        "[FAIL] Negative task counts detected."
    )


if (
    df[
        "jain_tasks_selected_only"
    ]
    .dropna()
    .between(
        0,
        1
    )
).all():

    print(
        "[PASS] Jain fairness values are within [0,1]."
    )

else:

    print(
        "[FAIL] Jain fairness outside [0,1]."
    )


if (
    df[
        "top1_task_share"
    ]
    <=
    df[
        "top5_task_share"
    ]
).all():

    print(
        "[PASS] Top-1 concentration never exceeds Top-5 concentration."
    )

else:

    print(
        "[FAIL] Top-1/Top-5 concentration inconsistency."
    )


if (
    (
        df[
            "total_successful_tasks"
        ]
        +
        df[
            "total_failed_tasks"
        ]
    )
    ==
    df[
        "total_completed_tasks"
    ]
).all():

    print(
        "[PASS] Completed tasks = successful + failed tasks."
    )

else:

    print(
        "[FAIL] Completion outcome counts are inconsistent."
    )


# ============================================================
# AGGREGATE
# ============================================================

METRICS = [

    "total_allocated_tasks",

    "total_completed_tasks",

    "unique_selected_vehicles",

    "mean_tasks_per_selected_vehicle",

    "median_tasks_per_selected_vehicle",

    "sd_tasks_per_selected_vehicle",

    "max_tasks_per_selected_vehicle",

    "jain_tasks_selected_only",

    "top1_task_share",

    "top5_task_share",

    "repeated_vehicle_share",

    "single_task_vehicle_share",

    "repeated_allocation_share",

    "completion_ratio",

    "malicious_task_share"
]


aggregate_rows = []


for group in [
    "malicious_ratio",
    "flood_intensity"
]:

    configurations = (

        MALICIOUS_CONFIGS.keys()

        if group == "malicious_ratio"

        else FLOOD_CONFIGS.keys()
    )


    for config in configurations:

        subset = df[
            (
                df[
                    "experiment_group"
                ]
                ==
                group
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
    "C17 - TASK DISTRIBUTION VS MALICIOUS RATIO"
)

print("=" * 190)


print(

    f"{'MAL%':>6}"

    f"{'TASKS':>8}"

    f"{'UNIQUE':>9}"

    f"{'TASK/VEH':>11}"

    f"{'MAX':>7}"

    f"{'JAIN':>9}"

    f"{'TOP1%':>9}"

    f"{'TOP5%':>9}"

    f"{'REPEAT%':>10}"

    f"{'MAL TASK%':>11}"
)


print("-" * 190)


c17_agg = (

    agg[
        agg[
            "experiment_group"
        ]
        ==
        "malicious_ratio"
    ]
    .sort_values(
        "parameter_value"
    )
)


for _, row in c17_agg.iterrows():

    print(

        f"{row['parameter_value'] * 100:>5.0f}%"

        f"{row['total_allocated_tasks_mean']:>8.2f}"

        f"{row['unique_selected_vehicles_mean']:>9.2f}"

        f"{row['mean_tasks_per_selected_vehicle_mean']:>11.3f}"

        f"{row['max_tasks_per_selected_vehicle_mean']:>7.2f}"

        f"{row['jain_tasks_selected_only_mean']:>9.4f}"

        f"{row['top1_task_share_mean'] * 100:>9.2f}"

        f"{row['top5_task_share_mean'] * 100:>9.2f}"

        f"{row['repeated_allocation_share_mean'] * 100:>10.2f}"

        f"{row['malicious_task_share_mean'] * 100:>11.2f}"
    )


# ============================================================
# C18 RESULTS
# ============================================================

print("\n")

print(
    "C18 - TASK DISTRIBUTION VS FLOOD INTENSITY"
)

print("=" * 190)


print(

    f"{'INT':>5}"

    f"{'TASKS':>8}"

    f"{'UNIQUE':>9}"

    f"{'TASK/VEH':>11}"

    f"{'MAX':>7}"

    f"{'JAIN':>9}"

    f"{'TOP1%':>9}"

    f"{'TOP5%':>9}"

    f"{'REPEAT%':>10}"

    f"{'MAL TASK%':>11}"
)


print("-" * 190)


c18_agg = (

    agg[
        agg[
            "experiment_group"
        ]
        ==
        "flood_intensity"
    ]
    .sort_values(
        "parameter_value"
    )
)


for _, row in c18_agg.iterrows():

    print(

        f"{int(row['parameter_value']):>5}"

        f"{row['total_allocated_tasks_mean']:>8.2f}"

        f"{row['unique_selected_vehicles_mean']:>9.2f}"

        f"{row['mean_tasks_per_selected_vehicle_mean']:>11.3f}"

        f"{row['max_tasks_per_selected_vehicle_mean']:>7.2f}"

        f"{row['jain_tasks_selected_only_mean']:>9.4f}"

        f"{row['top1_task_share_mean'] * 100:>9.2f}"

        f"{row['top5_task_share_mean'] * 100:>9.2f}"

        f"{row['repeated_allocation_share_mean'] * 100:>10.2f}"

        f"{row['malicious_task_share_mean'] * 100:>11.2f}"
    )


# ============================================================
# PAIRED COMPARISONS
# ============================================================

paired_rows = []


PAIRED_METRICS = [

    "unique_selected_vehicles",

    "mean_tasks_per_selected_vehicle",

    "max_tasks_per_selected_vehicle",

    "jain_tasks_selected_only",

    "top1_task_share",

    "top5_task_share",

    "repeated_allocation_share",

    "malicious_task_share"
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
                    not pd.isna(low)
                    and
                    not pd.isna(high)
                )
                else False
        })


# ============================================================
# C18 VS FLOOD 0
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
                    not pd.isna(low)
                    and
                    not pd.isna(high)
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
# C17 PAIRED PRINT
# ============================================================

print("\n")

print(
    "C17 PAIRED TASK-DISTRIBUTION COMPARISONS VS 0% MALICIOUS"
)

print("=" * 190)


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
            f"{row['metric']:<35}"

            f" diff="
            f"{row['mean_difference']:+.5f}"

            f" | CI95=["
            f"{row['ci95_low']:+.5f}, "
            f"{row['ci95_high']:+.5f}]"

            f" | excludes zero="
            f"{row['ci95_excludes_zero']}"
        )


# ============================================================
# C18 PAIRED PRINT
# ============================================================

print("\n")

print(
    "C18 PAIRED TASK-DISTRIBUTION COMPARISONS VS FLOOD INTENSITY 0"
)

print("=" * 190)


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
            f"{row['metric']:<35}"

            f" diff="
            f"{row['mean_difference']:+.5f}"

            f" | CI95=["
            f"{row['ci95_low']:+.5f}, "
            f"{row['ci95_high']:+.5f}]"

            f" | excludes zero="
            f"{row['ci95_excludes_zero']}"
        )


# ============================================================
# CORE SUMMARY
# ============================================================

baseline = c18_agg[
    c18_agg[
        "parameter_value"
    ]
    ==
    0
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
    "C21 CORE SUMMARY"
)

print("=" * 190)


print(
    "Baseline / no flood duplication:"
)


print(
    f"  Allocated tasks / run                 : "
    f"{baseline['total_allocated_tasks_mean']:.2f}"
)


print(
    f"  Unique selected vehicles / run        : "
    f"{baseline['unique_selected_vehicles_mean']:.2f}"
)


print(
    f"  Tasks / selected vehicle              : "
    f"{baseline['mean_tasks_per_selected_vehicle_mean']:.3f}"
)


print(
    f"  Maximum tasks / selected vehicle      : "
    f"{baseline['max_tasks_per_selected_vehicle_mean']:.2f}"
)


print(
    f"  Jain fairness (selected vehicles)     : "
    f"{baseline['jain_tasks_selected_only_mean']:.4f}"
)


print(
    f"  Top-1 task share                      : "
    f"{baseline['top1_task_share_mean'] * 100:.2f}%"
)


print(
    f"  Top-5 task share                      : "
    f"{baseline['top5_task_share_mean'] * 100:.2f}%"
)


print(
    f"  Repeated-allocation share             : "
    f"{baseline['repeated_allocation_share_mean'] * 100:.2f}%"
)


print(
    "\nTable II Flood Attack, intensity = 3:"
)


print(
    f"  Allocated tasks / run                 : "
    f"{flood3['total_allocated_tasks_mean']:.2f}"
)


print(
    f"  Unique selected vehicles / run        : "
    f"{flood3['unique_selected_vehicles_mean']:.2f}"
)


print(
    f"  Tasks / selected vehicle              : "
    f"{flood3['mean_tasks_per_selected_vehicle_mean']:.3f}"
)


print(
    f"  Jain fairness (selected vehicles)     : "
    f"{flood3['jain_tasks_selected_only_mean']:.4f}"
)


print(
    f"  Top-1 task share                      : "
    f"{flood3['top1_task_share_mean'] * 100:.2f}%"
)


print(
    f"  Top-5 task share                      : "
    f"{flood3['top5_task_share_mean'] * 100:.2f}%"
)


print(
    f"  Malicious task share                  : "
    f"{flood3['malicious_task_share_mean'] * 100:.2f}%"
)


# ============================================================
# CONCENTRATION SANITY
# ============================================================

print("\n")

print(
    "TASK-CONCENTRATION SANITY CHECK"
)

print("=" * 190)


if (
    baseline[
        "mean_tasks_per_selected_vehicle_mean"
    ]
    >=
    1.0
):

    print(
        "[PASS] Mean tasks per selected vehicle is >= 1."
    )

else:

    print(
        "[FAIL] Invalid mean task count."
    )


if (
    0
    <=
    baseline[
        "jain_tasks_selected_only_mean"
    ]
    <=
    1
):

    print(
        "[PASS] Baseline Jain fairness is valid."
    )

else:

    print(
        "[FAIL] Invalid baseline Jain fairness."
    )


# ============================================================
# LATEX TABLE
# ============================================================

latex = []


latex.append(
    r"\begin{table*}[t]"
)

latex.append(
    r"\centering"
)

latex.append(
    r"\caption{Distribution and concentration of allocated tasks across selected vehicles. Jain's fairness index is computed over vehicles receiving at least one task.}"
)

latex.append(
    r"\label{tab:task_distribution}"
)

latex.append(
    r"\begin{tabular}{lccccccc}"
)

latex.append(
    r"\hline"
)

latex.append(
    r"Scenario & Tasks & Selected Veh. & Tasks/Veh. & Jain & Top-1 & Top-5 & Repeat Alloc. \\"
)

latex.append(
    r"\hline"
)


latex.append(

    f"Baseline & "

    f"{baseline['total_allocated_tasks_mean']:.2f} & "

    f"{baseline['unique_selected_vehicles_mean']:.2f} & "

    f"{baseline['mean_tasks_per_selected_vehicle_mean']:.3f} & "

    f"{baseline['jain_tasks_selected_only_mean']:.4f} & "

    f"{baseline['top1_task_share_mean'] * 100:.2f}\\% & "

    f"{baseline['top5_task_share_mean'] * 100:.2f}\\% & "

    f"{baseline['repeated_allocation_share_mean'] * 100:.2f}\\% "

    r"\\"
)


for intensity in [
    1,
    2,
    3,
    5
]:

    row = c18_agg[
        c18_agg[
            "parameter_value"
        ]
        ==
        intensity
    ].iloc[
        0
    ]


    latex.append(

        f"Flood ($f={intensity}$) & "

        f"{row['total_allocated_tasks_mean']:.2f} & "

        f"{row['unique_selected_vehicles_mean']:.2f} & "

        f"{row['mean_tasks_per_selected_vehicle_mean']:.3f} & "

        f"{row['jain_tasks_selected_only_mean']:.4f} & "

        f"{row['top1_task_share_mean'] * 100:.2f}\\% & "

        f"{row['top5_task_share_mean'] * 100:.2f}\\% & "

        f"{row['repeated_allocation_share_mean'] * 100:.2f}\\% "

        r"\\"
    )


latex.append(
    r"\hline"
)

latex.append(
    r"\end{tabular}"
)

latex.append(
    r"\end{table*}"
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

print("=" * 190)


print(
    f"Per vehicle : "
    f"{RAW_FILE}"
)

print(
    f"Per seed    : "
    f"{PER_SEED_FILE}"
)

print(
    f"Aggregated  : "
    f"{AGG_FILE}"
)

print(
    f"Paired      : "
    f"{PAIRED_FILE}"
)

print(
    f"LaTeX       : "
    f"{LATEX_FILE}"
)


print("\n")

print("=" * 190)

print(
    "C21 TASKS PER VEHICLE ANALYSIS COMPLETED"
)

print("=" * 190)

print()