# analisar_candidates_per_request.py

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

OUTPUT_DIR = "candidates_analysis"

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

PER_REQUEST_FILE = os.path.join(
    OUTPUT_DIR,
    "candidates_per_request_raw.csv"
)

PER_SEED_FILE = os.path.join(
    OUTPUT_DIR,
    "candidates_per_request_por_seed.csv"
)

AGG_FILE = os.path.join(
    OUTPUT_DIR,
    "candidates_per_request_multiseed.csv"
)

PAIRED_FILE = os.path.join(
    OUTPUT_DIR,
    "candidates_per_request_paired.csv"
)

LATEX_FILE = os.path.join(
    OUTPUT_DIR,
    "tabela_candidates_per_request_tnsm.tex"
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


def numeric_column(
    df,
    column,
    default=0.0
):

    if column not in df.columns:

        return pd.Series(
            [default] * len(df),
            index=df.index,
            dtype="float64"
        )

    return pd.to_numeric(
        df[column],
        errors="coerce"
    ).fillna(
        default
    )


# ============================================================
# LOAD ELIGIBILITY FALLBACK
# ============================================================

def load_eligibility_fallback(
    eligibility_path
):

    """
    Legacy C17 fallback.

    Old requests CSVs may not contain
    unique_eligible_candidates.

    The corresponding eligibility CSV already stores:
        step
        active_vehicles
        eligible

    Therefore:
        eligible -> unique eligible physical vehicles
    """

    if not os.path.exists(
        eligibility_path
    ):

        raise RuntimeError(
            "Legacy request file does not contain "
            "'unique_eligible_candidates' and the corresponding "
            f"eligibility file was not found:\n{eligibility_path}"
        )


    eligibility = pd.read_csv(
        eligibility_path
    )


    required = [
        "step",
        "active_vehicles",
        "eligible"
    ]


    missing = [
        column
        for column in required
        if column not in eligibility.columns
    ]


    if missing:

        raise RuntimeError(
            f"Eligibility file {eligibility_path} "
            f"is missing columns: {missing}"
        )


    fallback = eligibility[[
        "step",
        "active_vehicles",
        "eligible"
    ]].copy()


    fallback[
        "step"
    ] = pd.to_numeric(
        fallback["step"],
        errors="coerce"
    )


    fallback[
        "active_vehicles"
    ] = pd.to_numeric(
        fallback["active_vehicles"],
        errors="coerce"
    ).fillna(
        0
    )


    fallback[
        "eligible"
    ] = pd.to_numeric(
        fallback["eligible"],
        errors="coerce"
    ).fillna(
        0
    )


    # Normally one eligibility row per request.
    # Grouping makes the fallback robust against accidental
    # duplicate rows.
    fallback = (
        fallback
        .groupby(
            "step",
            as_index=False
        )
        .agg({
            "active_vehicles": "max",
            "eligible": "max"
        })
    )


    return fallback


# ============================================================
# PREPARE REQUEST TELEMETRY
# ============================================================

def prepare_request_telemetry(
    request_path,
    eligibility_path
):

    requests = pd.read_csv(
        request_path
    )


    if "step" not in requests.columns:

        raise RuntimeError(
            f"Missing required column 'step' in {request_path}"
        )


    requests[
        "step"
    ] = pd.to_numeric(
        requests["step"],
        errors="coerce"
    )


    # ========================================================
    # ACTIVE VEHICLES
    # ========================================================

    if "active_vehicles" in requests.columns:

        requests[
            "active_vehicles_num"
        ] = pd.to_numeric(
            requests[
                "active_vehicles"
            ],
            errors="coerce"
        ).fillna(
            0
        )

    else:

        fallback = (
            load_eligibility_fallback(
                eligibility_path
            )
        )


        requests = requests.merge(
            fallback[[
                "step",
                "active_vehicles"
            ]],
            on="step",
            how="left"
        )


        requests[
            "active_vehicles_num"
        ] = pd.to_numeric(
            requests[
                "active_vehicles"
            ],
            errors="coerce"
        ).fillna(
            0
        )


    # ========================================================
    # UNIQUE ELIGIBLE CANDIDATES
    # ========================================================

    if (
        "unique_eligible_candidates"
        in requests.columns
    ):

        requests[
            "eligible_candidates"
        ] = pd.to_numeric(
            requests[
                "unique_eligible_candidates"
            ],
            errors="coerce"
        ).fillna(
            0
        )


        candidate_source = (
            "requests.unique_eligible_candidates"
        )


    else:

        fallback = (
            load_eligibility_fallback(
                eligibility_path
            )
        )


        # Avoid collision if active_vehicles already exists.
        fallback_for_merge = fallback[[
            "step",
            "eligible"
        ]].copy()


        requests = requests.merge(
            fallback_for_merge,
            on="step",
            how="left"
        )


        requests[
            "eligible_candidates"
        ] = pd.to_numeric(
            requests[
                "eligible"
            ],
            errors="coerce"
        ).fillna(
            0
        )


        candidate_source = (
            "eligibility.eligible"
        )


    # ========================================================
    # C18 FLOOD FIELDS
    # ========================================================

    if (
        "candidate_entries_after_flood"
        in requests.columns
    ):

        requests[
            "candidate_entries"
        ] = pd.to_numeric(
            requests[
                "candidate_entries_after_flood"
            ],
            errors="coerce"
        ).fillna(
            0
        )

    else:

        # Legacy / non-flood experiment:
        # one entry per unique eligible vehicle.
        requests[
            "candidate_entries"
        ] = requests[
            "eligible_candidates"
        ].copy()


    if (
        "artificial_candidate_count"
        in requests.columns
    ):

        requests[
            "artificial_candidates"
        ] = pd.to_numeric(
            requests[
                "artificial_candidate_count"
            ],
            errors="coerce"
        ).fillna(
            0
        )

    else:

        requests[
            "artificial_candidates"
        ] = 0.0


    # ========================================================
    # ALLOCATION
    # ========================================================

    if "allocated" in requests.columns:

        requests[
            "allocated_bool"
        ] = to_bool(
            requests[
                "allocated"
            ]
        )

    else:

        requests[
            "allocated_bool"
        ] = False


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    requests[
        "active_request"
    ] = (
        requests[
            "active_vehicles_num"
        ]
        >
        0
    )


    requests[
        "has_eligible_candidate"
    ] = (
        requests[
            "eligible_candidates"
        ]
        >
        0
    )


    return (
        requests,
        candidate_source
    )


# ============================================================
# PROCESS ONE RUN
# ============================================================

def process_request_file(

    request_path,

    eligibility_path,

    experiment_group,

    configuration,

    parameter_value,

    seed
):

    (
        requests,
        candidate_source

    ) = prepare_request_telemetry(

        request_path,
        eligibility_path
    )


    # ========================================================
    # SANITY CHECKS
    # ========================================================

    negative = (
        requests[
            "eligible_candidates"
        ]
        <
        0
    )


    if negative.any():

        raise RuntimeError(
            f"Negative eligible-candidate count detected in "
            f"{request_path}"
        )


    invalid = (
        requests[
            "eligible_candidates"
        ]
        >
        requests[
            "active_vehicles_num"
        ]
    )


    if invalid.any():

        bad = int(
            invalid.sum()
        )


        bad_rows = requests.loc[
            invalid,
            [
                "step",
                "active_vehicles_num",
                "eligible_candidates"
            ]
        ].head(
            10
        )


        raise RuntimeError(
            f"{bad} rows in {request_path} have "
            "eligible_candidates > active_vehicles.\n"
            f"{bad_rows.to_string(index=False)}"
        )


    # ========================================================
    # RAW DATA
    # ========================================================

    raw = pd.DataFrame({

        "experiment_group":
            experiment_group,

        "configuration":
            configuration,

        "parameter_value":
            parameter_value,

        "seed":
            seed,

        "candidate_source":
            candidate_source,

        "step":
            requests[
                "step"
            ],

        "active_vehicles":
            requests[
                "active_vehicles_num"
            ],

        "eligible_candidates":
            requests[
                "eligible_candidates"
            ],

        "candidate_entries":
            requests[
                "candidate_entries"
            ],

        "artificial_candidates":
            requests[
                "artificial_candidates"
            ],

        "active_request":
            requests[
                "active_request"
            ],

        "has_eligible_candidate":
            requests[
                "has_eligible_candidate"
            ],

        "allocated":
            requests[
                "allocated_bool"
            ]
    })


    # ========================================================
    # REQUEST SETS
    # ========================================================

    total_requests = len(
        requests
    )


    active = requests[
        requests[
            "active_request"
        ]
    ].copy()


    active_requests = len(
        active
    )


    active_with_candidates = active[
        active[
            "has_eligible_candidate"
        ]
    ].copy()


    # ========================================================
    # ALL REQUESTS
    # ========================================================

    mean_all = (
        requests[
            "eligible_candidates"
        ].mean()
    )


    median_all = (
        requests[
            "eligible_candidates"
        ].median()
    )


    # ========================================================
    # ACTIVE REQUESTS
    # ========================================================

    if active_requests > 0:

        active_eligible = (
            active[
                "eligible_candidates"
            ]
        )


        mean_active = (
            active_eligible.mean()
        )


        median_active = (
            active_eligible.median()
        )


        sd_active = (
            active_eligible.std(
                ddof=1
            )
        )


        min_active = (
            active_eligible.min()
        )


        max_active = (
            active_eligible.max()
        )


        p25_active = (
            active_eligible.quantile(
                0.25
            )
        )


        p75_active = (
            active_eligible.quantile(
                0.75
            )
        )


        zero_candidate_rate_active = (
            (
                active_eligible
                ==
                0
            ).mean()
        )


        mean_active_vehicles = (
            active[
                "active_vehicles_num"
            ].mean()
        )


        total_active_vehicle_opportunities = (
            active[
                "active_vehicles_num"
            ].sum()
        )


        eligible_fraction = (

            active_eligible.sum()

            /

            total_active_vehicle_opportunities

            if total_active_vehicle_opportunities > 0

            else float("nan")
        )


        mean_candidate_entries_active = (
            active[
                "candidate_entries"
            ].mean()
        )


        mean_artificial_active = (
            active[
                "artificial_candidates"
            ].mean()
        )


    else:

        mean_active = float("nan")

        median_active = float("nan")

        sd_active = float("nan")

        min_active = float("nan")

        max_active = float("nan")

        p25_active = float("nan")

        p75_active = float("nan")

        zero_candidate_rate_active = float(
            "nan"
        )

        mean_active_vehicles = float(
            "nan"
        )

        eligible_fraction = float(
            "nan"
        )

        mean_candidate_entries_active = float(
            "nan"
        )

        mean_artificial_active = float(
            "nan"
        )


    # ========================================================
    # REQUESTS WITH >=1 ELIGIBLE VEHICLE
    # ========================================================

    mean_when_available = (

        active_with_candidates[
            "eligible_candidates"
        ].mean()

        if len(
            active_with_candidates
        ) > 0

        else float("nan")
    )


    # ========================================================
    # PER-SEED ROW
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

        "candidate_source":
            candidate_source,

        "total_requests":
            total_requests,

        "active_requests":
            active_requests,

        "requests_with_eligible_candidates":
            len(
                active_with_candidates
            ),

        "mean_eligible_all_requests":
            mean_all,

        "median_eligible_all_requests":
            median_all,

        "mean_eligible_active_requests":
            mean_active,

        "median_eligible_active_requests":
            median_active,

        "sd_eligible_active_requests":
            sd_active,

        "min_eligible_active_requests":
            min_active,

        "max_eligible_active_requests":
            max_active,

        "p25_eligible_active_requests":
            p25_active,

        "p75_eligible_active_requests":
            p75_active,

        "mean_eligible_when_available":
            mean_when_available,

        "zero_candidate_rate_active":
            zero_candidate_rate_active,

        "mean_active_vehicles":
            mean_active_vehicles,

        "eligible_fraction_of_active_vehicles":
            eligible_fraction,

        "mean_candidate_entries_active":
            mean_candidate_entries_active,

        "mean_artificial_candidates_active":
            mean_artificial_active
    }


    return (
        row,
        raw
    )


# ============================================================
# HEADER
# ============================================================

print("\n")

print("=" * 190)

print(
    "SHADAI - C20 ELIGIBLE CANDIDATES PER REQUEST ANALYSIS"
)

print("=" * 190)


print(
    "Primary metric:"
)

print(
    "  unique eligible physical vehicles per request "
    "with at least one active vehicle"
)


print(
    "Eligibility condition:"
)

print(
    "  positive utility AND sufficient energy"
)


print(
    "Legacy compatibility:"
)

print(
    "  if requests.unique_eligible_candidates is absent, "
    "eligibility.eligible is used for the same request step"
)


print(
    "Flood duplicate entries are NOT counted as additional "
    "eligible physical vehicles."
)

print("=" * 190)


# ============================================================
# LOAD ALL RUNS
# ============================================================

rows = []

raw_frames = []

missing = []

source_counts = {}


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


        request_path = os.path.join(

            MALICIOUS_DIR,

            (
                f"requests_shadai_400_"
                f"{experiment}.csv"
            )
        )


        eligibility_path = os.path.join(

            MALICIOUS_DIR,

            (
                f"eligibilidade_shadai_400_"
                f"{experiment}.csv"
            )
        )


        if not os.path.exists(
            request_path
        ):

            missing.append(
                (
                    "C17",
                    config,
                    seed,
                    "requests"
                )
            )

            continue


        row, raw = process_request_file(

            request_path,

            eligibility_path,

            "malicious_ratio",

            config,

            ratio,

            seed
        )


        source = row[
            "candidate_source"
        ]


        source_counts[
            source
        ] = (
            source_counts.get(
                source,
                0
            )
            +
            1
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


        request_path = os.path.join(

            FLOOD_DIR,

            (
                f"requests_shadai_400_"
                f"{experiment}.csv"
            )
        )


        eligibility_path = os.path.join(

            FLOOD_DIR,

            (
                f"eligibilidade_shadai_400_"
                f"{experiment}.csv"
            )
        )


        if not os.path.exists(
            request_path
        ):

            missing.append(
                (
                    "C18",
                    config,
                    seed,
                    "requests"
                )
            )

            continue


        row, raw = process_request_file(

            request_path,

            eligibility_path,

            "flood_intensity",

            config,

            intensity,

            seed
        )


        source = row[
            "candidate_source"
        ]


        source_counts[
            source
        ] = (
            source_counts.get(
                source,
                0
            )
            +
            1
        )


        rows.append(
            row
        )


        raw_frames.append(
            raw
        )


# ============================================================
# BUILD DATAFRAMES
# ============================================================

df = pd.DataFrame(
    rows
)


if df.empty:

    raise RuntimeError(
        "No request telemetry files were found."
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
    PER_REQUEST_FILE,
    index=False
)


# ============================================================
# TELEMETRY SOURCE CHECK
# ============================================================

print("\n")

print(
    "CANDIDATE TELEMETRY SOURCE"
)

print("=" * 190)


for source, count in sorted(
    source_counts.items()
):

    print(
        f"{source:<45}: "
        f"{count} runs"
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
# GLOBAL SANITY CHECKS
# ============================================================

print("\n")

print(
    "C20 SANITY CHECK"
)

print("=" * 190)


if (
    raw_df[
        "eligible_candidates"
    ]
    >=
    0
).all():

    print(
        "[PASS] Eligible-candidate counts are non-negative."
    )

else:

    print(
        "[FAIL] Negative eligible-candidate counts detected."
    )


eligible_le_active = (

    raw_df[
        "eligible_candidates"
    ]

    <=

    raw_df[
        "active_vehicles"
    ]
)


if eligible_le_active.all():

    print(
        "[PASS] Eligible candidates never exceed active vehicles."
    )

else:

    print(
        "[FAIL] Eligible candidates exceed active vehicles."
    )


no_active_ok = (

    raw_df.loc[
        raw_df[
            "active_vehicles"
        ]
        ==
        0,
        "eligible_candidates"
    ]
    ==
    0
).all()


if no_active_ok:

    print(
        "[PASS] Requests with no active vehicles have zero "
        "eligible candidates."
    )

else:

    print(
        "[FAIL] A no-active-vehicle request has eligible candidates."
    )


# ============================================================
# AGGREGATION
# ============================================================

METRICS = [

    "total_requests",

    "active_requests",

    "requests_with_eligible_candidates",

    "mean_eligible_all_requests",

    "mean_eligible_active_requests",

    "median_eligible_active_requests",

    "mean_eligible_when_available",

    "zero_candidate_rate_active",

    "mean_active_vehicles",

    "eligible_fraction_of_active_vehicles",

    "mean_candidate_entries_active",

    "mean_artificial_candidates_active"
]


aggregate_rows = []


for group in [
    "malicious_ratio",
    "flood_intensity"
]:

    if group == "malicious_ratio":

        configurations = (
            MALICIOUS_CONFIGS.keys()
        )

    else:

        configurations = (
            FLOOD_CONFIGS.keys()
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
    "C17 - ELIGIBLE CANDIDATES VS MALICIOUS RATIO"
)

print("=" * 190)


print(

    f"{'MAL%':>6}"

    f"{'REQ':>8}"

    f"{'ACTIVE REQ':>12}"

    f"{'ACTIVE VEH':>12}"

    f"{'ELIG/ALL':>12}"

    f"{'ELIG/ACTIVE':>14}"

    f"{'ELIGIBLE%':>12}"

    f"{'ZERO-CAND%':>13}"
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

        f"{row['total_requests_mean']:>8.2f}"

        f"{row['active_requests_mean']:>12.2f}"

        f"{row['mean_active_vehicles_mean']:>12.2f}"

        f"{row['mean_eligible_all_requests_mean']:>12.2f}"

        f"{row['mean_eligible_active_requests_mean']:>14.2f}"

        f"{row['eligible_fraction_of_active_vehicles_mean'] * 100:>12.2f}"

        f"{row['zero_candidate_rate_active_mean'] * 100:>13.2f}"
    )


# ============================================================
# C18 RESULTS
# ============================================================

print("\n")

print(
    "C18 - UNIQUE ELIGIBLE CANDIDATES VS FLOOD INTENSITY"
)

print("=" * 190)


print(

    f"{'INT':>5}"

    f"{'ELIG/ACTIVE':>14}"

    f"{'POOL ENTRIES':>14}"

    f"{'ARTIFICIAL':>13}"

    f"{'ELIGIBLE%':>12}"

    f"{'ZERO-CAND%':>13}"
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

        f"{row['mean_eligible_active_requests_mean']:>14.2f}"

        f"{row['mean_candidate_entries_active_mean']:>14.2f}"

        f"{row['mean_artificial_candidates_active_mean']:>13.2f}"

        f"{row['eligible_fraction_of_active_vehicles_mean'] * 100:>12.2f}"

        f"{row['zero_candidate_rate_active_mean'] * 100:>13.2f}"
    )


# ============================================================
# PAIRED COMPARISONS
# ============================================================

paired_rows = []


PAIRED_METRICS = [

    "mean_eligible_all_requests",

    "mean_eligible_active_requests",

    "eligible_fraction_of_active_vehicles",

    "zero_candidate_rate_active",

    "mean_candidate_entries_active"
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
# C18 PAIRED PRINT
# ============================================================

print("\n")

print(
    "C18 PAIRED CHECK VS FLOOD INTENSITY 0"
)

print("=" * 190)


for config, intensity in FLOOD_CONFIGS.items():

    if config == "flood_00":

        continue


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


    print(
        f"\nFlood intensity = "
        f"{intensity}"
    )


    for _, row in subset.iterrows():

        print(

            f"  "
            f"{row['metric']:<38}"

            f" diff="
            f"{row['mean_difference']:+.5f}"

            f" | CI95=["
            f"{row['ci95_low']:+.5f}, "
            f"{row['ci95_high']:+.5f}]"

            f" | excludes zero="
            f"{row['ci95_excludes_zero']}"
        )


# ============================================================
# C20 CORE SUMMARY
# ============================================================

nominal = c18_agg[
    c18_agg[
        "parameter_value"
    ]
    ==
    3
].iloc[
    0
]


baseline = c18_agg[
    c18_agg[
        "parameter_value"
    ]
    ==
    0
].iloc[
    0
]


print("\n")

print(
    "C20 CORE SUMMARY"
)

print("=" * 190)


print(
    "Baseline / no candidate duplication:"
)


print(
    f"  Eligible candidates / all request     : "
    f"{baseline['mean_eligible_all_requests_mean']:.2f}"
)


print(
    f"  Eligible candidates / active request  : "
    f"{baseline['mean_eligible_active_requests_mean']:.2f}"
)


print(
    f"  Active vehicles / active request      : "
    f"{baseline['mean_active_vehicles_mean']:.2f}"
)


print(
    f"  Eligible fraction of active vehicles  : "
    f"{baseline['eligible_fraction_of_active_vehicles_mean'] * 100:.2f}%"
)


print(
    f"  Zero-candidate active-request rate    : "
    f"{baseline['zero_candidate_rate_active_mean'] * 100:.2f}%"
)


print(
    "\nTable II Flood Attack, intensity = 3:"
)


print(
    f"  Unique eligible vehicles / active req : "
    f"{nominal['mean_eligible_active_requests_mean']:.2f}"
)


print(
    f"  Candidate entries / active request    : "
    f"{nominal['mean_candidate_entries_active_mean']:.2f}"
)


print(
    f"  Artificial entries / active request   : "
    f"{nominal['mean_artificial_candidates_active_mean']:.2f}"
)


# ============================================================
# FLOOD STRUCTURAL SANITY
# ============================================================

print("\n")

print(
    "FLOOD CANDIDATE-POOL SANITY CHECK"
)

print("=" * 190)


ordered_c18 = c18_agg.sort_values(
    "parameter_value"
)


unique_values = (
    ordered_c18[
        "mean_eligible_active_requests_mean"
    ]
    .tolist()
)


unique_range = (
    max(unique_values)
    -
    min(unique_values)
)


if unique_range < 0.01:

    print(
        "[PASS] Flood intensity does not materially change "
        "the number of unique eligible physical vehicles."
    )

else:

    print(
        "[CHECK] Unique eligible-candidate count differs "
        "materially across flood intensities."
    )

    print(
        f"        Range across intensities: "
        f"{unique_range:.10f}"
    )


pool_values = (
    ordered_c18[
        "mean_candidate_entries_active_mean"
    ]
    .tolist()
)


pool_monotonic = all(

    pool_values[i]
    <=
    pool_values[i + 1]
    +
    1e-12

    for i in range(
        len(pool_values) - 1
    )
)


if pool_monotonic:

    print(
        "[PASS] Total candidate entries increase monotonically "
        "with flood intensity."
    )

else:

    print(
        "[CHECK] Candidate entries are not monotonic."
    )


# ============================================================
# LATEX TABLE
# ============================================================

latex = []


latex.append(
    r"\begin{table}[t]"
)

latex.append(
    r"\centering"
)

latex.append(
    r"\caption{Eligible candidates per service request. Eligible candidates are distinct physical vehicles satisfying both the positive-utility and sufficient-energy conditions. Requests with no active vehicles are excluded from the primary candidate-pool statistic.}"
)

latex.append(
    r"\label{tab:eligible_candidates}"
)

latex.append(
    r"\begin{tabular}{lccc}"
)

latex.append(
    r"\hline"
)

latex.append(
    r"Scenario & Eligible Vehicles & Candidate Entries & Zero-Candidate Req. \\"
)

latex.append(
    r"\hline"
)


latex.append(

    f"Baseline & "

    f"{baseline['mean_eligible_active_requests_mean']:.2f} & "

    f"{baseline['mean_candidate_entries_active_mean']:.2f} & "

    f"{baseline['zero_candidate_rate_active_mean'] * 100:.2f}\\% "

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

        f"{row['mean_eligible_active_requests_mean']:.2f} & "

        f"{row['mean_candidate_entries_active_mean']:.2f} & "

        f"{row['zero_candidate_rate_active_mean'] * 100:.2f}\\% "

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
# OUTPUT FILES
# ============================================================

print("\n")

print(
    "OUTPUT FILES"
)

print("=" * 190)


print(
    f"Per request : "
    f"{PER_REQUEST_FILE}"
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
    "C20 ELIGIBLE CANDIDATES PER REQUEST ANALYSIS COMPLETED"
)

print("=" * 190)

print()