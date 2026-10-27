# analisar_flood_multiseed.py

import os
import math

import pandas as pd


try:

    from scipy.stats import t

except ImportError:

    t = None


# ============================================================
# CONFIG
# ============================================================

DIR = "flood_ablation"


CONFIGS = {

    "flood_00": 0,

    "flood_01": 1,

    "flood_02": 2,

    "flood_03": 3,

    "flood_05": 5
}


ORDER = [

    "flood_00",

    "flood_01",

    "flood_02",

    "flood_03",

    "flood_05"
]


BASELINE = "flood_00"

TABLE_II = "flood_03"


SEEDS = list(
    range(
        1,
        31
    )
)


# ============================================================
# OUTPUTS
# ============================================================

PER_SEED = os.path.join(
    DIR,
    "flood_por_seed.csv"
)

AGG_FILE = os.path.join(
    DIR,
    "flood_multiseed.csv"
)

PAIRED_FILE = os.path.join(
    DIR,
    "flood_pareado_vs_zero.csv"
)

LATEX_FILE = os.path.join(
    DIR,
    "tabela_flood_intensity_tnsm.tex"
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
        .str.lower()
        .isin([
            "true",
            "1"
        ])
    )


def normalize_selection_mode(
    series
):

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
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


# ============================================================
# HEADER
# ============================================================

print("\n")

print("=" * 190)

print(
    "SHADAI - C18 MULTI-SEED FLOOD INTENSITY ANALYSIS"
)

print("=" * 190)


print(
    f"Directory       : {DIR}"
)

print(
    f"Seeds           : {len(SEEDS)}"
)

print(
    "Malicious ratio : 20%"
)

print(
    "Table II nominal intensity : 3"
)

print("=" * 190)


# ============================================================
# LOAD
# ============================================================

rows = []

missing = []

selection_modes_seen = set()


for config in ORDER:

    intensity = CONFIGS[
        config
    ]


    for seed in SEEDS:

        experiment = (

            f"{config}_"
            f"intensity_{intensity:02d}_"
            f"seed_{seed:02d}"
        )


        flood_file = os.path.join(

            DIR,

            (
                f"flood_shadai_400_"
                f"{experiment}.csv"
            )
        )


        metrics_file = os.path.join(

            DIR,

            (
                f"metricas_shadai_400_"
                f"{experiment}.csv"
            )
        )


        vehicles_file = os.path.join(

            DIR,

            (
                f"vehicles_shadai_400_"
                f"{experiment}.csv"
            )
        )


        outcomes_file = os.path.join(

            DIR,

            (
                f"outcomes_shadai_400_"
                f"{experiment}.csv"
            )
        )


        requests_file = os.path.join(

            DIR,

            (
                f"requests_shadai_400_"
                f"{experiment}.csv"
            )
        )


        files = [

            flood_file,

            metrics_file,

            vehicles_file,

            outcomes_file,

            requests_file
        ]


        if not all(
            os.path.exists(
                x
            )
            for x in files
        ):

            missing.append(
                (
                    config,
                    seed
                )
            )

            continue


        flood = pd.read_csv(
            flood_file
        )

        metrics = pd.read_csv(
            metrics_file
        )

        vehicles = pd.read_csv(
            vehicles_file
        )

        outcomes = pd.read_csv(
            outcomes_file
        )

        requests = pd.read_csv(
            requests_file
        )


        # ====================================================
        # VEHICLE POPULATION
        # ====================================================

        vehicles[
            "malicious_bool"
        ] = to_bool(
            vehicles[
                "malicious"
            ]
        )


        vehicle_count = len(
            vehicles
        )


        malicious_count = int(
            vehicles[
                "malicious_bool"
            ].sum()
        )


        realized_ratio = (

            malicious_count
            /
            vehicle_count

            if vehicle_count > 0

            else float("nan")
        )


        # ====================================================
        # SELECTIONS
        # ====================================================

        metrics[
            "selected_malicious_bool"
        ] = to_bool(
            metrics[
                "selected_malicious"
            ]
        )


        allocation_count = len(
            metrics
        )


        malicious_selection_share = (

            metrics[
                "selected_malicious_bool"
            ].mean()

            if allocation_count > 0

            else float("nan")
        )


        exposure_ratio = (

            malicious_selection_share
            /
            realized_ratio

            if (
                realized_ratio > 0

                and

                not pd.isna(
                    malicious_selection_share
                )
            )

            else float("nan")
        )


        # ====================================================
        # EXPLORATION / EXPLOITATION
        #
        # Aceita:
        # exploration / exploitation
        # explore / exploit
        # ====================================================

        mode = normalize_selection_mode(
            metrics[
                "selection_mode"
            ]
        )


        for value in mode.dropna().unique():

            if (
                value
                and
                value != "nan"
            ):

                selection_modes_seen.add(
                    value
                )


        explore_mask = (
            mode.isin([
                "exploration",
                "explore"
            ])
        )


        exploit_mask = (
            mode.isin([
                "exploitation",
                "exploit"
            ])
        )


        exploration_count = int(
            explore_mask.sum()
        )


        exploitation_count = int(
            exploit_mask.sum()
        )


        unknown_mode_count = int(

            (
                ~(
                    explore_mask
                    |
                    exploit_mask
                )
            ).sum()
        )


        explore_malicious_share = (

            metrics.loc[
                explore_mask,
                "selected_malicious_bool"
            ].mean()

            if exploration_count > 0

            else float("nan")
        )


        exploit_malicious_share = (

            metrics.loc[
                exploit_mask,
                "selected_malicious_bool"
            ].mean()

            if exploitation_count > 0

            else float("nan")
        )


        exploration_share = (

            exploration_count
            /
            allocation_count

            if allocation_count > 0

            else float("nan")
        )


        # ====================================================
        # FLOOD TELEMETRY
        # ====================================================

        unique_candidates = pd.to_numeric(
            flood[
                "unique_candidate_count"
            ],
            errors="coerce"
        )


        artificial = pd.to_numeric(
            flood[
                "artificial_candidate_count"
            ],
            errors="coerce"
        )


        total_entries = pd.to_numeric(
            flood[
                "candidate_entries_after_flood"
            ],
            errors="coerce"
        )


        malicious_entries = pd.to_numeric(
            flood[
                "malicious_candidate_entries"
            ],
            errors="coerce"
        )


        malicious_entry_share = pd.to_numeric(
            flood[
                "malicious_entry_share"
            ],
            errors="coerce"
        )


        unique_malicious = pd.to_numeric(
            flood[
                "unique_malicious_candidates"
            ],
            errors="coerce"
        )


        # ====================================================
        # REQUEST ALLOCATION
        # ====================================================

        requests[
            "allocated_bool"
        ] = to_bool(
            requests[
                "allocated"
            ]
        )


        active_requests = (

            pd.to_numeric(
                requests[
                    "active_vehicles"
                ],
                errors="coerce"
            )
            >
            0
        )


        conditional_allocation = (

            requests[
                "allocated_bool"
            ].sum()
            /
            active_requests.sum()

            if active_requests.sum() > 0

            else float("nan")
        )


        # ====================================================
        # SERVICE SUCCESS
        # ====================================================

        outcomes[
            "success_bool"
        ] = to_bool(
            outcomes[
                "success"
            ]
        )


        service_success = (

            outcomes[
                "success_bool"
            ].mean()

            if len(
                outcomes
            ) > 0

            else float("nan")
        )


        # ====================================================
        # SCORE
        # ====================================================

        avg_score = pd.to_numeric(
            metrics[
                "score"
            ],
            errors="coerce"
        ).mean()


        # ====================================================
        # STORE
        # ====================================================

        rows.append({

            "configuration":
                config,

            "intensity":
                intensity,

            "seed":
                seed,

            "vehicle_count":
                vehicle_count,

            "realized_malicious_ratio":
                realized_ratio,

            "allocation_count":
                allocation_count,

            "conditional_allocation_rate":
                conditional_allocation,

            "malicious_selection_share":
                malicious_selection_share,

            "exposure_ratio":
                exposure_ratio,

            "exploration_count":
                exploration_count,

            "exploitation_count":
                exploitation_count,

            "unknown_selection_mode_count":
                unknown_mode_count,

            "exploration_share":
                exploration_share,

            "explore_malicious_share":
                explore_malicious_share,

            "exploit_malicious_share":
                exploit_malicious_share,

            "mean_unique_candidates":
                unique_candidates.mean(),

            "mean_unique_malicious_candidates":
                unique_malicious.mean(),

            "mean_artificial_candidates":
                artificial.mean(),

            "total_artificial_candidates":
                artificial.sum(),

            "mean_candidate_entries_after_flood":
                total_entries.mean(),

            "mean_malicious_candidate_entries":
                malicious_entries.mean(),

            "mean_malicious_entry_share":
                malicious_entry_share.mean(),

            "avg_score":
                avg_score,

            "service_success_rate":
                service_success
        })


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    rows
)


if df.empty:

    raise RuntimeError(
        "No valid C18 experiments found."
    )


df.to_csv(
    PER_SEED,
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


for config in ORDER:

    n = len(
        df[
            df[
                "configuration"
            ]
            ==
            config
        ]
    )


    print(
        f"{config:<12}: "
        f"{n}/30 runs"
    )


if missing:

    print(
        f"\nMissing runs: "
        f"{len(missing)}"
    )


# ============================================================
# SELECTION MODE SANITY CHECK
# ============================================================

print("\n")

print(
    "SELECTION-MODE SANITY CHECK"
)

print("=" * 190)


print(
    "Observed selection_mode values:"
)


for value in sorted(
    selection_modes_seen
):

    print(
        f"  - {value}"
    )


total_exploration = int(
    df[
        "exploration_count"
    ].sum()
)


total_exploitation = int(
    df[
        "exploitation_count"
    ].sum()
)


total_unknown = int(
    df[
        "unknown_selection_mode_count"
    ].sum()
)


print(
    f"\nExploration selections : "
    f"{total_exploration}"
)

print(
    f"Exploitation selections: "
    f"{total_exploitation}"
)

print(
    f"Unknown-mode selections: "
    f"{total_unknown}"
)


if (
    total_exploration > 0
    and
    total_exploitation > 0
    and
    total_unknown == 0
):

    print(
        "[PASS] Exploration/exploitation modes recognized correctly."
    )

elif (
    total_exploration > 0
    and
    total_exploitation > 0
):

    print(
        "[CHECK] Modes recognized, but some unknown labels remain."
    )

else:

    print(
        "[FAIL] Exploration/exploitation labels were not recognized."
    )


# ============================================================
# AGGREGATE
# ============================================================

METRICS = [

    "realized_malicious_ratio",

    "conditional_allocation_rate",

    "malicious_selection_share",

    "exposure_ratio",

    "exploration_share",

    "explore_malicious_share",

    "exploit_malicious_share",

    "mean_unique_candidates",

    "mean_unique_malicious_candidates",

    "mean_artificial_candidates",

    "total_artificial_candidates",

    "mean_candidate_entries_after_flood",

    "mean_malicious_candidate_entries",

    "mean_malicious_entry_share",

    "avg_score",

    "service_success_rate"
]


agg_rows = []


for config in ORDER:

    subset = df[
        df[
            "configuration"
        ]
        ==
        config
    ]


    row = {

        "configuration":
            config,

        "intensity":
            CONFIGS[
                config
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
            metric
            +
            "_mean"
        ] = mean

        row[
            metric
            +
            "_sd"
        ] = sd

        row[
            metric
            +
            "_ci95_low"
        ] = low

        row[
            metric
            +
            "_ci95_high"
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
# MAIN RESULTS
# ============================================================

print("\n")

print(
    "MAIN FLOOD-INTENSITY RESULTS"
)

print("=" * 190)


print(

    f"{'INT':>4}"

    f"{'REAL%':>9}"

    f"{'ALLOC%':>10}"

    f"{'UNIQUE':>10}"

    f"{'ARTIF':>10}"

    f"{'POOL':>10}"

    f"{'MAL POOL%':>11}"

    f"{'MAL SEL%':>11}"

    f"{'EXPOSURE':>10}"

    f"{'EXP MAL%':>11}"

    f"{'EXPL MAL%':>12}"

    f"{'SUCCESS%':>11}"
)


print("-" * 190)


for _, row in agg.iterrows():

    print(

        f"{int(row['intensity']):>4}"

        f"{row['realized_malicious_ratio_mean'] * 100:>9.2f}"

        f"{row['conditional_allocation_rate_mean'] * 100:>10.2f}"

        f"{row['mean_unique_candidates_mean']:>10.2f}"

        f"{row['mean_artificial_candidates_mean']:>10.2f}"

        f"{row['mean_candidate_entries_after_flood_mean']:>10.2f}"

        f"{row['mean_malicious_entry_share_mean'] * 100:>11.2f}"

        f"{row['malicious_selection_share_mean'] * 100:>11.2f}"

        f"{row['exposure_ratio_mean']:>10.3f}"

        f"{row['explore_malicious_share_mean'] * 100:>11.2f}"

        f"{row['exploit_malicious_share_mean'] * 100:>12.2f}"

        f"{row['service_success_rate_mean'] * 100:>11.2f}"
    )


# ============================================================
# EXPLORATION SHARE
# ============================================================

print("\n")

print(
    "EXPLORATION / EXPLOITATION SUMMARY"
)

print("=" * 190)


print(

    f"{'INT':>4}"

    f"{'EXPLORATION%':>16}"

    f"{'MAL IN EXPLORATION%':>22}"

    f"{'MAL IN EXPLOITATION%':>23}"
)


print("-" * 190)


for _, row in agg.iterrows():

    print(

        f"{int(row['intensity']):>4}"

        f"{row['exploration_share_mean'] * 100:>16.2f}"

        f"{row['explore_malicious_share_mean'] * 100:>22.2f}"

        f"{row['exploit_malicious_share_mean'] * 100:>23.2f}"
    )


# ============================================================
# PAIRED VS INTENSITY 0
# ============================================================

baseline = df[
    df[
        "configuration"
    ]
    ==
    BASELINE
].copy()


PAIRED_METRICS = [

    "malicious_selection_share",

    "exposure_ratio",

    "explore_malicious_share",

    "exploit_malicious_share",

    "mean_candidate_entries_after_flood",

    "mean_malicious_entry_share",

    "conditional_allocation_rate",

    "avg_score",

    "service_success_rate"
]


paired_rows = []


for config in ORDER:

    if config == BASELINE:

        continue


    alternative = df[
        df[
            "configuration"
        ]
        ==
        config
    ].copy()


    paired = pd.merge(

        baseline,

        alternative,

        on="seed",

        suffixes=(
            "_baseline",
            "_alternative"
        )
    )


    for metric in PAIRED_METRICS:

        difference = (

            paired[
                metric
                +
                "_alternative"
            ]

            -

            paired[
                metric
                +
                "_baseline"
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

            "configuration":
                config,

            "intensity":
                CONFIGS[
                    config
                ],

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
# PAIRED PRINT
# ============================================================

print("\n")

print(
    "PAIRED COMPARISONS VS FLOOD INTENSITY 0"
)

print("=" * 190)


for config in ORDER:

    if config == BASELINE:

        continue


    intensity = CONFIGS[
        config
    ]


    print(
        f"\nFlood intensity = "
        f"{intensity}"
    )


    subset = paired_df[
        paired_df[
            "configuration"
        ]
        ==
        config
    ]


    for _, row in subset.iterrows():

        print(

            f"  "
            f"{row['metric']:<38} "

            f"diff="
            f"{row['mean_difference']:+.5f} | "

            f"CI95=["
            f"{row['ci95_low']:+.5f}, "
            f"{row['ci95_high']:+.5f}] | "

            f"excludes zero="
            f"{row['ci95_excludes_zero']}"
        )


# ============================================================
# TABLE II SCENARIO
# ============================================================

print("\n")

print(
    "TABLE II NOMINAL FLOOD SCENARIO (INTENSITY = 3)"
)

print("=" * 190)


row = agg[
    agg[
        "configuration"
    ]
    ==
    TABLE_II
].iloc[
    0
]


print(
    f"Realized malicious ratio       : "
    f"{row['realized_malicious_ratio_mean'] * 100:.2f}%"
)

print(
    f"Unique candidates/request      : "
    f"{row['mean_unique_candidates_mean']:.2f}"
)

print(
    f"Artificial candidacies/request : "
    f"{row['mean_artificial_candidates_mean']:.2f}"
)

print(
    f"Candidate entries/request      : "
    f"{row['mean_candidate_entries_after_flood_mean']:.2f}"
)

print(
    f"Malicious candidate pool share : "
    f"{row['mean_malicious_entry_share_mean'] * 100:.2f}%"
)

print(
    f"Malicious selection share      : "
    f"{row['malicious_selection_share_mean'] * 100:.2f}%"
)

print(
    f"Malicious exposure ratio       : "
    f"{row['exposure_ratio_mean']:.3f}"
)

print(
    f"Exploration rate               : "
    f"{row['exploration_share_mean'] * 100:.2f}%"
)

print(
    f"Explore malicious selection    : "
    f"{row['explore_malicious_share_mean'] * 100:.2f}%"
)

print(
    f"Exploit malicious selection    : "
    f"{row['exploit_malicious_share_mean'] * 100:.2f}%"
)

print(
    f"Conditional allocation rate    : "
    f"{row['conditional_allocation_rate_mean'] * 100:.2f}%"
)


# ============================================================
# FLOOD SANITY
# ============================================================

print("\n")

print(
    "FLOOD IMPLEMENTATION SANITY CHECK"
)

print("=" * 190)


zero = df[
    df[
        "intensity"
    ]
    ==
    0
]


if (
    zero[
        "total_artificial_candidates"
    ]
    ==
    0
).all():

    print(
        "[PASS] Intensity 0 generated no artificial candidacies."
    )

else:

    print(
        "[FAIL] Artificial candidacies detected at intensity 0."
    )


sanity_ok = True


for config in ORDER:

    intensity = CONFIGS[
        config
    ]


    subset = df[
        df[
            "configuration"
        ]
        ==
        config
    ]


    if intensity == 0:

        continue


    if (
        subset[
            "mean_artificial_candidates"
        ].mean()
        <=
        0
    ):

        sanity_ok = False


if sanity_ok:

    print(
        "[PASS] Positive flood intensities generated "
        "artificial candidacies."
    )

else:

    print(
        "[FAIL] A positive intensity generated no "
        "artificial candidacies."
    )


# ============================================================
# MONOTONIC POOL CHECK
# ============================================================

ordered_pool = (

    agg
    .sort_values(
        "intensity"
    )[
        "mean_candidate_entries_after_flood_mean"
    ]
    .tolist()
)


pool_monotonic = all(

    ordered_pool[i]
    <=
    ordered_pool[i + 1]

    for i in range(
        len(
            ordered_pool
        )
        -
        1
    )
)


if pool_monotonic:

    print(
        "[PASS] Candidate-pool size increases monotonically "
        "with flood intensity."
    )

else:

    print(
        "[CHECK] Candidate-pool size is not monotonic."
    )


# ============================================================
# MALICIOUS POOL SHARE CHECK
# ============================================================

ordered_mal_pool = (

    agg
    .sort_values(
        "intensity"
    )[
        "mean_malicious_entry_share_mean"
    ]
    .tolist()
)


mal_pool_monotonic = all(

    ordered_mal_pool[i]
    <=
    ordered_mal_pool[i + 1]

    for i in range(
        len(
            ordered_mal_pool
        )
        -
        1
    )
)


if mal_pool_monotonic:

    print(
        "[PASS] Malicious candidate-pool share increases "
        "monotonically with flood intensity."
    )

else:

    print(
        "[CHECK] Malicious candidate-pool share is not monotonic."
    )


# ============================================================
# LATEX
# ============================================================

latex = []


latex.append(
    r"\begin{table*}[t]"
)

latex.append(
    r"\centering"
)

latex.append(
    r"\caption{Sensitivity of SHADAI to Flood Attack intensity. Flood intensity denotes the number of additional artificial candidacies submitted by each eligible malicious vehicle for the same service request. The malicious-vehicle ratio is fixed at 20\%, and values are averaged over 30 simulation seeds.}"
)

latex.append(
    r"\label{tab:flood_intensity}"
)

latex.append(
    r"\resizebox{\textwidth}{!}{%"
)

latex.append(
    r"\begin{tabular}{c|ccccccccc}"
)

latex.append(
    r"\hline"
)

latex.append(
    r"Intensity & Unique Cand. & Artificial Cand. & Candidate Entries & Malicious Pool & Malicious Sel. & Exposure Ratio & Explore Malicious & Exploit Malicious & Conditional Alloc. \\"
)

latex.append(
    r"\hline"
)


for _, row in agg.iterrows():

    intensity_text = (

        r"\textbf{3}"

        if int(
            row[
                "intensity"
            ]
        ) == 3

        else str(
            int(
                row[
                    "intensity"
                ]
            )
        )
    )


    latex.append(

        f"{intensity_text} & "

        f"{row['mean_unique_candidates_mean']:.2f} & "

        f"{row['mean_artificial_candidates_mean']:.2f} & "

        f"{row['mean_candidate_entries_after_flood_mean']:.2f} & "

        f"{row['mean_malicious_entry_share_mean'] * 100:.2f}\\% & "

        f"{row['malicious_selection_share_mean'] * 100:.2f}\\% & "

        f"{row['exposure_ratio_mean']:.3f} & "

        f"{row['explore_malicious_share_mean'] * 100:.2f}\\% & "

        f"{row['exploit_malicious_share_mean'] * 100:.2f}\\% & "

        f"{row['conditional_allocation_rate_mean'] * 100:.2f}\\% "

        r"\\"
    )


latex.append(
    r"\hline"
)

latex.append(
    r"\end{tabular}%"
)

latex.append(
    r"}"
)

latex.append(
    r"\vspace{1mm}"
)

latex.append(
    r"\parbox{0.98\textwidth}{\footnotesize Flood intensity counts additional candidacy entries rather than additional service requests. An intensity of three therefore represents one legitimate candidacy plus three artificial duplicates for each eligible malicious vehicle. Duplicate entries refer to the same physical vehicle and do not cause multiple task executions or multiple energy deductions.}"
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
    f"Per seed   : {PER_SEED}"
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

print("=" * 190)

print(
    "C18 FLOOD-INTENSITY ANALYSIS COMPLETED"
)

print("=" * 190)

print()