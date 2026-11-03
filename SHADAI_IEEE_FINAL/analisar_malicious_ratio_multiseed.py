# analisar_malicious_ratio_multiseed.py

import os
import math

import pandas as pd


try:
    from scipy.stats import t
except ImportError:
    t = None


# ============================================================
# DIRETÓRIO
# ============================================================

DIR = "malicious_ablation"


# ============================================================
# CONFIGURAÇÕES
# ============================================================

CONFIGS = {

    "malicious_00": {
        "label": "0%",
        "ratio": 0.00
    },

    "malicious_10": {
        "label": "10%",
        "ratio": 0.10
    },

    "malicious_20": {
        "label": "20%",
        "ratio": 0.20
    },

    "malicious_30": {
        "label": "30%",
        "ratio": 0.30
    },

    "malicious_40": {
        "label": "40%",
        "ratio": 0.40
    }
}


ORDER = [
    "malicious_00",
    "malicious_10",
    "malicious_20",
    "malicious_30",
    "malicious_40"
]


BASELINE = "malicious_00"


SEEDS = list(
    range(
        1,
        31
    )
)


# ============================================================
# OUTPUTS
# ============================================================

PER_SEED_FILE = os.path.join(
    DIR,
    "malicious_ratio_por_seed.csv"
)


AGG_FILE = os.path.join(
    DIR,
    "malicious_ratio_multiseed.csv"
)


PAIRED_FILE = os.path.join(
    DIR,
    "malicious_ratio_pareado_vs_zero.csv"
)


ADAPTATION_FILE = os.path.join(
    DIR,
    "malicious_adaptation_analysis.csv"
)


LATEX_FILE = os.path.join(
    DIR,
    "tabela_malicious_ratio_tnsm.tex"
)


# ============================================================
# HELPERS
# ============================================================

def to_bool(series):

    return (
        series
        .astype(str)
        .str.lower()
        .isin([
            "true",
            "1"
        ])
    )


def ic95(values):

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


    if t is not None:

        critical = t.ppf(
            0.975,
            n - 1
        )

    else:

        critical = 1.96


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
    "SHADAI - MULTI-SEED MALICIOUS RATIO ANALYSIS"
)

print("=" * 190)


print(
    f"Directory : {DIR}"
)

print(
    f"Seeds     : {len(SEEDS)}"
)

print(
    "Baseline  : 0% malicious vehicles"
)

print(
    "Attack    : deliberate service failure"
)

print(
    "Adaptation: first half vs second half of actual selections"
)

print(
    "Exposure  : malicious selection share / realized malicious ratio"
)

print("=" * 190)


# ============================================================
# LEITURA
# ============================================================

rows = []

missing = []


for config_name in ORDER:

    cfg = CONFIGS[
        config_name
    ]


    ratio = cfg[
        "ratio"
    ]


    for seed in SEEDS:

        experiment_id = (
            f"{config_name}_"
            f"ratio_{int(ratio * 100):02d}_"
            f"seed_{seed:02d}"
        )


        vehicle_file = os.path.join(
            DIR,
            (
                f"vehicles_shadai_400_"
                f"{experiment_id}.csv"
            )
        )


        metric_file = os.path.join(
            DIR,
            (
                f"metricas_shadai_400_"
                f"{experiment_id}.csv"
            )
        )


        outcome_file = os.path.join(
            DIR,
            (
                f"outcomes_shadai_400_"
                f"{experiment_id}.csv"
            )
        )


        request_file = os.path.join(
            DIR,
            (
                f"requests_shadai_400_"
                f"{experiment_id}.csv"
            )
        )


        files = [
            vehicle_file,
            metric_file,
            outcome_file,
            request_file
        ]


        if not all(
            os.path.exists(f)
            for f in files
        ):

            missing.append(
                (
                    config_name,
                    seed
                )
            )

            continue


        vehicles = pd.read_csv(
            vehicle_file
        )

        metrics = pd.read_csv(
            metric_file
        )

        outcomes = pd.read_csv(
            outcome_file
        )

        requests = pd.read_csv(
            request_file
        )


        # ====================================================
        # POPULAÇÃO
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


        malicious_vehicle_count = int(
            vehicles[
                "malicious_bool"
            ].sum()
        )


        realized_ratio = (

            malicious_vehicle_count
            /
            vehicle_count

            if vehicle_count > 0

            else float("nan")
        )


        # ====================================================
        # REQUESTS
        # ====================================================

        requests[
            "allocated_bool"
        ] = to_bool(
            requests[
                "allocated"
            ]
        )


        total_requests = len(
            requests
        )


        allocations = int(
            requests[
                "allocated_bool"
            ].sum()
        )


        active_requests = int(

            (
                pd.to_numeric(
                    requests[
                        "active_vehicles"
                    ],
                    errors="coerce"
                )
                >
                0
            ).sum()
        )


        end_to_end_allocation = (

            allocations
            /
            total_requests

            if total_requests > 0

            else float("nan")
        )


        conditional_allocation = (

            allocations
            /
            active_requests

            if active_requests > 0

            else float("nan")
        )


        # ====================================================
        # SELEÇÕES
        # ====================================================

        metrics = metrics.copy()


        metrics[
            "selected_malicious_bool"
        ] = to_bool(
            metrics[
                "selected_malicious"
            ]
        )


        metrics[
            "step_numeric"
        ] = pd.to_numeric(
            metrics[
                "step"
            ],
            errors="coerce"
        )


        metrics = (
            metrics
            .sort_values(
                by="step_numeric"
            )
            .reset_index(
                drop=True
            )
        )


        selected_count = len(
            metrics
        )


        malicious_selections = int(
            metrics[
                "selected_malicious_bool"
            ].sum()
        )


        honest_selections = (
            selected_count
            -
            malicious_selections
        )


        malicious_selection_share = (

            malicious_selections
            /
            selected_count

            if selected_count > 0

            else float("nan")
        )


        # ====================================================
        # EXPOSURE RATIO
        #
        # < 1 -> malicious vehicles selected less often
        #        than their population prevalence.
        #
        # = 1 -> proportional exposure.
        #
        # > 1 -> over-exposure.
        # ====================================================

        if (
            realized_ratio > 0
            and
            not pd.isna(
                malicious_selection_share
            )
        ):

            exposure_ratio = (
                malicious_selection_share
                /
                realized_ratio
            )

        else:

            exposure_ratio = float(
                "nan"
            )


        # ====================================================
        # SCORE / REPUTATION
        # ====================================================

        avg_score = pd.to_numeric(
            metrics[
                "score"
            ],
            errors="coerce"
        ).mean()


        avg_selected_reputation = pd.to_numeric(
            metrics[
                "reputacao"
            ],
            errors="coerce"
        ).mean()


        mal_metrics = metrics[
            metrics[
                "selected_malicious_bool"
            ]
        ]


        honest_metrics = metrics[
            ~metrics[
                "selected_malicious_bool"
            ]
        ]


        avg_malicious_selected_rep = pd.to_numeric(
            mal_metrics[
                "reputacao"
            ],
            errors="coerce"
        ).mean()


        avg_honest_selected_rep = pd.to_numeric(
            honest_metrics[
                "reputacao"
            ],
            errors="coerce"
        ).mean()


        # ====================================================
        # SERVICE OUTCOMES
        # ====================================================

        outcomes = outcomes.copy()


        outcomes[
            "malicious_bool"
        ] = to_bool(
            outcomes[
                "malicious"
            ]
        )


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


        service_success_rate = (

            successful
            /
            completed

            if completed > 0

            else float("nan")
        )


        malicious_outcomes = outcomes[
            outcomes[
                "malicious_bool"
            ]
        ]


        honest_outcomes = outcomes[
            ~outcomes[
                "malicious_bool"
            ]
        ]


        malicious_success_rate = (

            malicious_outcomes[
                "success_bool"
            ].mean()

            if len(
                malicious_outcomes
            ) > 0

            else float("nan")
        )


        honest_success_rate = (

            honest_outcomes[
                "success_bool"
            ].mean()

            if len(
                honest_outcomes
            ) > 0

            else float("nan")
        )


        malicious_rep_delta = pd.to_numeric(
            malicious_outcomes[
                "reputation_delta"
            ],
            errors="coerce"
        ).mean()


        honest_rep_delta = pd.to_numeric(
            honest_outcomes[
                "reputation_delta"
            ],
            errors="coerce"
        ).mean()


        # ====================================================
        # NOVA ANÁLISE ADAPTATIVA:
        # PRIMEIRA METADE VS SEGUNDA METADE DAS SELEÇÕES
        # ====================================================

        split_index = (
            selected_count
            //
            2
        )


        if (
            selected_count >= 2
            and
            split_index > 0
        ):

            early = metrics.iloc[
                :split_index
            ]


            late = metrics.iloc[
                split_index:
            ]


            early_malicious_share = (
                early[
                    "selected_malicious_bool"
                ].mean()
            )


            late_malicious_share = (
                late[
                    "selected_malicious_bool"
                ].mean()
            )


            malicious_share_change = (
                late_malicious_share
                -
                early_malicious_share
            )


        else:

            early_malicious_share = float(
                "nan"
            )

            late_malicious_share = float(
                "nan"
            )

            malicious_share_change = float(
                "nan"
            )


        # ====================================================
        # EARLY / LATE EXPOSURE RATIO
        # ====================================================

        if (
            realized_ratio > 0
            and
            not pd.isna(
                early_malicious_share
            )
        ):

            early_exposure_ratio = (
                early_malicious_share
                /
                realized_ratio
            )

        else:

            early_exposure_ratio = float(
                "nan"
            )


        if (
            realized_ratio > 0
            and
            not pd.isna(
                late_malicious_share
            )
        ):

            late_exposure_ratio = (
                late_malicious_share
                /
                realized_ratio
            )

        else:

            late_exposure_ratio = float(
                "nan"
            )


        if (
            not pd.isna(
                early_exposure_ratio
            )
            and
            not pd.isna(
                late_exposure_ratio
            )
        ):

            exposure_ratio_change = (
                late_exposure_ratio
                -
                early_exposure_ratio
            )

        else:

            exposure_ratio_change = float(
                "nan"
            )


        # ====================================================
        # STORE
        # ====================================================

        rows.append({

            "configuration":
                config_name,

            "configured_ratio":
                ratio,

            "seed":
                seed,

            "vehicle_count":
                vehicle_count,

            "malicious_vehicle_count":
                malicious_vehicle_count,

            "realized_malicious_ratio":
                realized_ratio,

            "total_requests":
                total_requests,

            "active_requests":
                active_requests,

            "allocations":
                allocations,

            "end_to_end_allocation_rate":
                end_to_end_allocation,

            "conditional_allocation_rate":
                conditional_allocation,

            "selection_count":
                selected_count,

            "malicious_selections":
                malicious_selections,

            "honest_selections":
                honest_selections,

            "malicious_selection_share":
                malicious_selection_share,

            "exposure_ratio":
                exposure_ratio,

            "avg_score":
                avg_score,

            "avg_selected_reputation":
                avg_selected_reputation,

            "avg_malicious_selected_reputation":
                avg_malicious_selected_rep,

            "avg_honest_selected_reputation":
                avg_honest_selected_rep,

            "completed_services":
                completed,

            "service_success_rate":
                service_success_rate,

            "malicious_success_rate":
                malicious_success_rate,

            "honest_success_rate":
                honest_success_rate,

            "malicious_reputation_delta":
                malicious_rep_delta,

            "honest_reputation_delta":
                honest_rep_delta,

            "early_malicious_selection_share":
                early_malicious_share,

            "late_malicious_selection_share":
                late_malicious_share,

            "malicious_selection_share_change":
                malicious_share_change,

            "early_exposure_ratio":
                early_exposure_ratio,

            "late_exposure_ratio":
                late_exposure_ratio,

            "exposure_ratio_change":
                exposure_ratio_change
        })


# ============================================================
# DATAFRAME POR SEED
# ============================================================

df = pd.DataFrame(
    rows
)


if df.empty:

    raise RuntimeError(
        "No valid C17 experiment found."
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
        f"{config:<18}: "
        f"{n}/30 runs"
    )


if missing:

    print(
        f"\n[WARNING] Missing runs: "
        f"{len(missing)}"
    )


# ============================================================
# AGREGAÇÃO
# ============================================================

METRICS = [

    "vehicle_count",

    "malicious_vehicle_count",

    "realized_malicious_ratio",

    "allocations",

    "conditional_allocation_rate",

    "malicious_selection_share",

    "exposure_ratio",

    "avg_score",

    "avg_selected_reputation",

    "avg_malicious_selected_reputation",

    "avg_honest_selected_reputation",

    "service_success_rate",

    "malicious_success_rate",

    "honest_success_rate",

    "malicious_reputation_delta",

    "honest_reputation_delta",

    "early_malicious_selection_share",

    "late_malicious_selection_share",

    "malicious_selection_share_change",

    "early_exposure_ratio",

    "late_exposure_ratio",

    "exposure_ratio_change"
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

        "configured_ratio":
            CONFIGS[
                config
            ][
                "ratio"
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
        ) = ic95(
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
# RESULTADOS PRINCIPAIS
# ============================================================

print("\n")

print(
    "MAIN MALICIOUS-RATIO RESULTS"
)

print("=" * 190)


header = (

    f"{'RATIO':<8}"

    f"{'REAL%':>9}"

    f"{'ALLOC%':>10}"

    f"{'MAL SEL%':>11}"

    f"{'EXPOSURE':>10}"

    f"{'SUCCESS%':>11}"

    f"{'HON SUC%':>11}"

    f"{'MAL SUC%':>11}"

    f"{'REP ALL':>10}"

    f"{'REP MAL':>10}"

    f"{'REP HON':>10}"
)


print(
    header
)

print("-" * 190)


for _, row in agg.iterrows():

    mal_success = row[
        "malicious_success_rate_mean"
    ]


    exposure = row[
        "exposure_ratio_mean"
    ]


    print(

        f"{row['configured_ratio'] * 100:>6.0f}%"

        f"{row['realized_malicious_ratio_mean'] * 100:>9.2f}"

        f"{row['conditional_allocation_rate_mean'] * 100:>10.2f}"

        f"{row['malicious_selection_share_mean'] * 100:>11.2f}"

        f"{exposure:>10.3f}"

        f"{row['service_success_rate_mean'] * 100:>11.2f}"

        f"{row['honest_success_rate_mean'] * 100:>11.2f}"

        f"{mal_success * 100:>11.2f}"

        f"{row['avg_selected_reputation_mean']:>10.3f}"

        f"{row['avg_malicious_selected_reputation_mean']:>10.3f}"

        f"{row['avg_honest_selected_reputation_mean']:>10.3f}"
    )


# ============================================================
# ADAPTATION RESULTS
# ============================================================

print("\n")

print(
    "MALICIOUS-SELECTION ADAPTATION RESULTS"
)

print("=" * 190)


header = (

    f"{'RATIO':<8}"

    f"{'EARLY MAL%':>13}"

    f"{'LATE MAL%':>13}"

    f"{'DELTA MAL':>12}"

    f"{'EARLY EXP':>12}"

    f"{'LATE EXP':>12}"

    f"{'DELTA EXP':>12}"
)


print(
    header
)

print("-" * 190)


adaptation_rows = []


for _, row in agg.iterrows():

    if row[
        "configured_ratio"
    ] == 0:

        continue


    adaptation_rows.append({

        "configuration":
            row[
                "configuration"
            ],

        "configured_ratio":
            row[
                "configured_ratio"
            ],

        "early_malicious_selection_share":
            row[
                "early_malicious_selection_share_mean"
            ],

        "late_malicious_selection_share":
            row[
                "late_malicious_selection_share_mean"
            ],

        "malicious_selection_share_change":
            row[
                "malicious_selection_share_change_mean"
            ],

        "malicious_selection_share_change_ci95_low":
            row[
                "malicious_selection_share_change_ci95_low"
            ],

        "malicious_selection_share_change_ci95_high":
            row[
                "malicious_selection_share_change_ci95_high"
            ],

        "early_exposure_ratio":
            row[
                "early_exposure_ratio_mean"
            ],

        "late_exposure_ratio":
            row[
                "late_exposure_ratio_mean"
            ],

        "exposure_ratio_change":
            row[
                "exposure_ratio_change_mean"
            ],

        "exposure_ratio_change_ci95_low":
            row[
                "exposure_ratio_change_ci95_low"
            ],

        "exposure_ratio_change_ci95_high":
            row[
                "exposure_ratio_change_ci95_high"
            ]
    })


    print(

        f"{row['configured_ratio'] * 100:>6.0f}%"

        f"{row['early_malicious_selection_share_mean'] * 100:>13.2f}"

        f"{row['late_malicious_selection_share_mean'] * 100:>13.2f}"

        f"{row['malicious_selection_share_change_mean'] * 100:>+12.2f}"

        f"{row['early_exposure_ratio_mean']:>12.3f}"

        f"{row['late_exposure_ratio_mean']:>12.3f}"

        f"{row['exposure_ratio_change_mean']:>+12.3f}"
    )


pd.DataFrame(
    adaptation_rows
).to_csv(
    ADAPTATION_FILE,
    index=False
)


# ============================================================
# EXPOSURE RATIO CI
# ============================================================

print("\n")

print(
    "EXPOSURE-RATIO CONFIDENCE INTERVALS"
)

print("=" * 190)


for _, row in agg.iterrows():

    if row[
        "configured_ratio"
    ] == 0:

        continue


    low = row[
        "exposure_ratio_ci95_low"
    ]

    high = row[
        "exposure_ratio_ci95_high"
    ]


    below_one = (
        high < 1.0
    )


    print(

        f"{row['configured_ratio'] * 100:>4.0f}% | "

        f"Exposure="
        f"{row['exposure_ratio_mean']:.3f} | "

        f"CI95=["
        f"{low:.3f}, "
        f"{high:.3f}] | "

        f"CI entirely below 1="
        f"{below_one}"
    )


# ============================================================
# PAIRED VS 0%
# ============================================================

baseline = df[
    df[
        "configuration"
    ]
    ==
    BASELINE
].copy()


PAIRED_METRICS = [

    "conditional_allocation_rate",

    "avg_score",

    "avg_selected_reputation",

    "service_success_rate",

    "honest_success_rate"
]


paired_rows = []


for config in ORDER:

    if config == BASELINE:

        continue


    alt = df[
        df[
            "configuration"
        ]
        ==
        config
    ].copy()


    paired = pd.merge(

        baseline,

        alt,

        on="seed",

        suffixes=(
            "_baseline",
            "_alternative"
        )
    )


    for metric in PAIRED_METRICS:

        diffs = (

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
        ) = ic95(
            diffs
        )


        paired_rows.append({

            "configuration":
                config,

            "configured_ratio":
                CONFIGS[
                    config
                ][
                    "ratio"
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
# ADAPTATION CI CHECK
# ============================================================

print("\n")

print(
    "ADAPTATION CONFIDENCE-INTERVAL CHECK"
)

print("=" * 190)


for _, row in agg.iterrows():

    ratio = row[
        "configured_ratio"
    ]


    if ratio == 0:

        continue


    low = row[
        "malicious_selection_share_change_ci95_low"
    ]

    high = row[
        "malicious_selection_share_change_ci95_high"
    ]


    reduction_supported = (
        high < 0
    )


    print(

        f"{ratio * 100:>4.0f}% | "

        f"Late-Early malicious share = "
        f"{row['malicious_selection_share_change_mean'] * 100:+.2f} p.p. | "

        f"CI95=["
        f"{low * 100:+.2f}, "
        f"{high * 100:+.2f}] p.p. | "

        f"clear reduction="
        f"{reduction_supported}"
    )


# ============================================================
# NESTED MALICIOUS SET CHECK
# ============================================================

print("\n")

print(
    "NESTED MALICIOUS-SET SANITY CHECK"
)

print("=" * 190)


nested_ok = True


for seed in SEEDS:

    previous_set = set()


    for config in ORDER:

        ratio = CONFIGS[
            config
        ][
            "ratio"
        ]


        experiment_id = (

            f"{config}_"
            f"ratio_{int(ratio * 100):02d}_"
            f"seed_{seed:02d}"
        )


        filename = os.path.join(

            DIR,

            (
                f"vehicles_shadai_400_"
                f"{experiment_id}.csv"
            )
        )


        vehicles = pd.read_csv(
            filename
        )


        malicious = set(

            vehicles.loc[

                to_bool(
                    vehicles[
                        "malicious"
                    ]
                ),

                "vehicle"

            ].astype(str)
        )


        if not previous_set.issubset(
            malicious
        ):

            nested_ok = False


        previous_set = malicious


if nested_ok:

    print(
        "[PASS] Malicious sets are nested across ratios "
        "for every seed."
    )

else:

    print(
        "[FAIL] Nested malicious-set property violated."
    )


# ============================================================
# ATTACK MODEL CHECK
# ============================================================

nonzero = df[
    df[
        "configured_ratio"
    ]
    >
    0
]


malicious_success_values = (

    nonzero[
        "malicious_success_rate"
    ]
    .dropna()
)


if (
    len(
        malicious_success_values
    ) > 0

    and

    (
        malicious_success_values
        ==
        0
    ).all()
):

    print(
        "[PASS] All completed malicious services failed "
        "as specified by the C17 attack model."
    )

else:

    print(
        "[CHECK] Unexpected successful malicious service "
        "detected."
    )


# ============================================================
# PAIRED RESULTS
# ============================================================

print("\n")

print(
    "PAIRED COMPARISONS VS 0% MALICIOUS"
)

print("=" * 190)


for config in ORDER:

    if config == BASELINE:

        continue


    ratio = CONFIGS[
        config
    ][
        "ratio"
    ]


    print(
        f"\nMalicious ratio = "
        f"{ratio * 100:.0f}%"
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
            f"{row['metric']:<32} "

            f"diff="
            f"{row['mean_difference']:+.5f} | "

            f"CI95=["
            f"{row['ci95_low']:+.5f}, "
            f"{row['ci95_high']:+.5f}] | "

            f"excludes zero="
            f"{row['ci95_excludes_zero']}"
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
    r"\caption{Sensitivity of SHADAI to the malicious-vehicle ratio. Malicious vehicles participate normally in resource allocation but deliberately fail assigned services. Values are means over 30 simulation seeds.}"
)

latex.append(
    r"\label{tab:malicious_ratio}"
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
    r"Configured & Realized & Conditional Alloc. & Malicious Sel. & Exposure Ratio & Service Success & Honest Success & Selected Rep. & Malicious Rep. & Honest Rep. \\"
)

latex.append(
    r"\hline"
)


for _, row in agg.iterrows():

    malicious_rep = row[
        "avg_malicious_selected_reputation_mean"
    ]


    malicious_rep_text = (

        f"{malicious_rep:.3f}"

        if not pd.isna(
            malicious_rep
        )

        else "--"
    )


    exposure = row[
        "exposure_ratio_mean"
    ]


    exposure_text = (

        f"{exposure:.3f}"

        if not pd.isna(
            exposure
        )

        else "--"
    )


    latex.append(

        f"{row['configured_ratio'] * 100:.0f}\\% & "

        f"{row['realized_malicious_ratio_mean'] * 100:.2f}\\% & "

        f"{row['conditional_allocation_rate_mean'] * 100:.2f}\\% & "

        f"{row['malicious_selection_share_mean'] * 100:.2f}\\% & "

        f"{exposure_text} & "

        f"{row['service_success_rate_mean'] * 100:.2f}\\% & "

        f"{row['honest_success_rate_mean'] * 100:.2f}\\% & "

        f"{row['avg_selected_reputation_mean']:.3f} & "

        f"{malicious_rep_text} & "

        f"{row['avg_honest_selected_reputation_mean']:.3f} "

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
    r"\parbox{0.98\textwidth}{\footnotesize The exposure ratio is defined as the malicious-selection share divided by the realized malicious fraction. Values below one indicate that malicious vehicles are selected less frequently than expected from their population prevalence. Adaptation is evaluated by comparing the first and second halves of the actual allocation sequence in each run rather than by fixed simulation-time windows.}"
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
# OUTPUTS
# ============================================================

print("\n")

print(
    "OUTPUT FILES"
)

print("=" * 190)


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
    f"Adaptation : {ADAPTATION_FILE}"
)

print(
    f"LaTeX      : {LATEX_FILE}"
)


print("\n")

print("=" * 190)

print(
    "C17 MALICIOUS-RATIO ANALYSIS COMPLETED"
)

print("=" * 190)

print()