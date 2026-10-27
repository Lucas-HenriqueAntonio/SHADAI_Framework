# analisar_runtime_multiseed.py

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

RUNTIME_DIR = "runtime_ablation"


CONFIGURACOES = {

    "runtime_low": {
        "label": "Low",
        "interval": 60
    },

    "runtime_moderate": {
        "label": "Moderate",
        "interval": 40
    },

    "runtime_high": {
        "label": "High",
        "interval": 20
    },

    "runtime_very_high": {
        "label": "Very High",
        "interval": 10
    }
}


ORDEM = [

    "runtime_low",

    "runtime_moderate",

    "runtime_high",

    "runtime_very_high"
]


BASELINE = "runtime_high"


SEEDS = list(
    range(
        1,
        31
    )
)


RNG_SCHEME_REQUIRED = "split_v1"

TELEMETRY_SCHEME_REQUIRED = "c15_runtime_v1"


# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_PER_SEED = os.path.join(

    RUNTIME_DIR,

    "runtime_por_seed.csv"
)


OUTPUT_AGG = os.path.join(

    RUNTIME_DIR,

    "runtime_multiseed.csv"
)


OUTPUT_PAIRED = os.path.join(

    RUNTIME_DIR,

    "runtime_pareado_vs_baseline.csv"
)


OUTPUT_LATEX = os.path.join(

    RUNTIME_DIR,

    "tabela_runtime_tnsm.tex"
)


# ============================================================
# IC95
# ============================================================

def calcular_ic95(
    valores
):

    serie = pd.Series(
        valores
    ).dropna()


    n = len(
        serie
    )


    if n == 0:

        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan")
        )


    media = serie.mean()


    if n == 1:

        return (
            media,
            float("nan"),
            float("nan"),
            float("nan")
        )


    sd = serie.std(
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

        crit = t.ppf(
            0.975,
            df=n - 1
        )

    else:

        crit = 1.96


    margem = (
        crit
        *
        se
    )


    return (
        media,
        sd,
        media - margem,
        media + margem
    )


# ============================================================
# BOOL
# ============================================================

def bool_series(
    serie
):

    return (

        serie
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1"
            ]
        )
    )


# ============================================================
# VALIDATION
# ============================================================

def validar(
    df,
    intervalo
):

    obrigatorias = [

        "rng_scheme",

        "telemetry_scheme",

        "request_interval",

        "active_vehicles",

        "candidate_count",

        "offer_executed",

        "selection_executed",

        "offer_runtime_ms",

        "selection_runtime_ms",

        "decision_runtime_ms"
    ]


    for coluna in obrigatorias:

        if coluna not in df.columns:

            return False


    if not (

        df[
            "rng_scheme"
        ]

        .astype(str)

        .eq(
            RNG_SCHEME_REQUIRED
        )

        .all()
    ):

        return False


    if not (

        df[
            "telemetry_scheme"
        ]

        .astype(str)

        .eq(
            TELEMETRY_SCHEME_REQUIRED
        )

        .all()
    ):

        return False


    intervalos = (

        pd.to_numeric(
            df[
                "request_interval"
            ],
            errors="coerce"
        )

        .dropna()

        .unique()
    )


    if len(
        intervalos
    ) != 1:

        return False


    return (
        int(
            intervalos[0]
        )
        ==
        intervalo
    )


# ============================================================
# HEADER
# ============================================================

print("\n")

print("=" * 180)

print(
    "SHADAI - MULTI-SEED COMPUTATIONAL RUNTIME ANALYSIS"
)

print("=" * 180)


print(
    f"Directory        : {RUNTIME_DIR}"
)

print(
    f"Seeds            : {len(SEEDS)}"
)

print(
    f"RNG scheme       : {RNG_SCHEME_REQUIRED}"
)

print(
    f"Telemetry scheme : {TELEMETRY_SCHEME_REQUIRED}"
)

print(
    "Runtime unit     : milliseconds"
)

print(
    "Replication unit : simulation seed"
)

print("=" * 180)


# ============================================================
# LOAD DATA
# ============================================================

resultados = []

faltantes = []

invalidos = []


for configuracao in ORDEM:

    config = CONFIGURACOES[
        configuracao
    ]


    label = config[
        "label"
    ]


    interval = config[
        "interval"
    ]


    for seed in SEEDS:

        experiment_id = (

            f"{configuracao}_"
            f"interval_"
            f"{interval:03d}_"
            f"seed_"
            f"{seed:02d}"
        )


        arquivo = os.path.join(

            RUNTIME_DIR,

            (
                f"runtime_shadai_400_"
                f"{experiment_id}.csv"
            )
        )


        if not os.path.exists(
            arquivo
        ):

            faltantes.append(
                (
                    configuracao,
                    seed
                )
            )

            continue


        df = pd.read_csv(
            arquivo
        )


        if (
            df.empty
            or
            not validar(
                df,
                interval
            )
        ):

            invalidos.append(
                (
                    configuracao,
                    seed
                )
            )

            continue


        # ====================================================
        # CONVERSÕES
        # ====================================================

        offer_executed = bool_series(
            df[
                "offer_executed"
            ]
        )


        selection_executed = bool_series(
            df[
                "selection_executed"
            ]
        )


        active_vehicles = pd.to_numeric(

            df[
                "active_vehicles"
            ],

            errors="coerce"
        )


        candidate_count = pd.to_numeric(

            df[
                "candidate_count"
            ],

            errors="coerce"
        )


        offer = pd.to_numeric(

            df[
                "offer_runtime_ms"
            ],

            errors="coerce"
        )


        selection = pd.to_numeric(

            df[
                "selection_runtime_ms"
            ],

            errors="coerce"
        )


        decision = pd.to_numeric(

            df[
                "decision_runtime_ms"
            ],

            errors="coerce"
        )


        # ====================================================
        # OFFER SAMPLE
        # ====================================================

        offer_values = offer[
            offer_executed
            &
            offer.notna()
        ]


        # ====================================================
        # SELECTION SAMPLE
        # ====================================================

        selection_mask = (

            selection_executed

            &

            selection.notna()

            &

            candidate_count.notna()
        )


        selection_values = selection[
            selection_mask
        ]


        selection_candidates = candidate_count[
            selection_mask
        ]


        selection_decision = decision[
            selection_mask
        ]


        if len(
            selection_values
        ) == 0:

            invalidos.append(
                (
                    configuracao,
                    seed
                )
            )

            continue


        # ====================================================
        # SANITY ACCOUNTING
        # ====================================================

        timing_error = (

            decision

            -

            (
                offer
                +
                selection
            )

        ).abs().max()


        # ====================================================
        # RUNTIME
        # ====================================================

        mean_offer = (
            offer_values.mean()
        )


        median_offer = (
            offer_values.median()
        )


        p95_offer = (
            offer_values.quantile(
                0.95
            )
        )


        mean_selection = (
            selection_values.mean()
        )


        median_selection = (
            selection_values.median()
        )


        p95_selection = (
            selection_values.quantile(
                0.95
            )
        )


        p99_selection = (
            selection_values.quantile(
                0.99
            )
        )


        max_selection = (
            selection_values.max()
        )


        mean_decision = (
            selection_decision.mean()
        )


        p95_decision = (
            selection_decision.quantile(
                0.95
            )
        )


        # ====================================================
        # CUSTO ACUMULADO
        # ====================================================

        total_offer_ms = (
            offer_values.sum()
        )


        total_selection_ms = (
            selection_values.sum()
        )


        total_algorithm_runtime_ms = (

            total_offer_ms
            +
            total_selection_ms
        )


        # ====================================================
        # CANDIDATOS
        # ====================================================

        mean_candidates = (
            selection_candidates.mean()
        )


        median_candidates = (
            selection_candidates.median()
        )


        p95_candidates = (
            selection_candidates.quantile(
                0.95
            )
        )


        mean_active = (

            active_vehicles[
                active_vehicles > 0
            ]
            .mean()
        )


        # ====================================================
        # CORRELAÇÃO
        # ====================================================

        pair_df = pd.DataFrame({

            "candidates":
                selection_candidates,

            "runtime":
                selection_values
        }).dropna()


        if (
            len(
                pair_df
            ) >= 2

            and

            pair_df[
                "candidates"
            ].nunique() > 1

            and

            pair_df[
                "runtime"
            ].nunique() > 1
        ):

            corr = (

                pair_df[
                    [
                        "candidates",
                        "runtime"
                    ]
                ]

                .corr()

                .iloc[
                    0,
                    1
                ]
            )

        else:

            corr = float(
                "nan"
            )


        resultados.append({

            "configuration":
                configuracao,

            "workload":
                label,

            "request_interval":
                interval,

            "seed":
                seed,

            "n_offered_requests":
                len(
                    df
                ),

            "n_offer_operations":
                int(
                    offer_executed.sum()
                ),

            "n_selection_operations":
                int(
                    selection_executed.sum()
                ),

            "mean_active_vehicles_when_available":
                mean_active,

            "mean_candidate_count":
                mean_candidates,

            "median_candidate_count":
                median_candidates,

            "p95_candidate_count":
                p95_candidates,

            "mean_offer_runtime_ms":
                mean_offer,

            "median_offer_runtime_ms":
                median_offer,

            "p95_offer_runtime_ms":
                p95_offer,

            "mean_selection_runtime_ms":
                mean_selection,

            "median_selection_runtime_ms":
                median_selection,

            "p95_selection_runtime_ms":
                p95_selection,

            "p99_selection_runtime_ms":
                p99_selection,

            "max_selection_runtime_ms":
                max_selection,

            "mean_decision_runtime_ms":
                mean_decision,

            "p95_decision_runtime_ms":
                p95_decision,

            "total_offer_runtime_ms":
                total_offer_ms,

            "total_selection_runtime_ms":
                total_selection_ms,

            "total_algorithm_runtime_ms":
                total_algorithm_runtime_ms,

            "corr_candidates_selection_runtime":
                corr,

            "max_timing_accounting_error_ms":
                timing_error
        })


# ============================================================
# DATAFRAME POR SEED
# ============================================================

df_seed = pd.DataFrame(
    resultados
)


if df_seed.empty:

    raise RuntimeError(
        "Nenhum experimento C15 válido encontrado."
    )


df_seed.to_csv(
    OUTPUT_PER_SEED,
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


for config in ORDEM:

    n = len(

        df_seed[
            df_seed[
                "configuration"
            ]
            ==
            config
        ]
    )


    print(

        f"{config:<20}: "

        f"{n:>2}/30 runs"
    )


if faltantes:

    print(

        f"\n[WARNING] Missing runs: "
        f"{len(faltantes)}"
    )


if invalidos:

    print(

        f"\n[WARNING] Invalid runs: "
        f"{len(invalidos)}"
    )


# ============================================================
# TIMING SANITY
# ============================================================

max_error = (

    df_seed[
        "max_timing_accounting_error_ms"
    ]

    .max()
)


if (
    pd.notna(
        max_error
    )

    and

    max_error < 1e-6
):

    print(

        "\n[PASS] "
        "decision_runtime = "
        "offer_runtime + selection_runtime."
    )

else:

    print(

        "\n[CHECK] Maximum timing accounting error: "
        f"{max_error:.9f} ms"
    )


# ============================================================
# AGREGAÇÃO
# ============================================================

METRICAS = [

    "n_offered_requests",

    "n_offer_operations",

    "n_selection_operations",

    "mean_active_vehicles_when_available",

    "mean_candidate_count",

    "p95_candidate_count",

    "mean_offer_runtime_ms",

    "p95_offer_runtime_ms",

    "mean_selection_runtime_ms",

    "median_selection_runtime_ms",

    "p95_selection_runtime_ms",

    "p99_selection_runtime_ms",

    "max_selection_runtime_ms",

    "mean_decision_runtime_ms",

    "p95_decision_runtime_ms",

    "total_offer_runtime_ms",

    "total_selection_runtime_ms",

    "total_algorithm_runtime_ms",

    "corr_candidates_selection_runtime"
]


resumo = []


for config in ORDEM:

    dados = df_seed[
        df_seed[
            "configuration"
        ]
        ==
        config
    ]


    metadata = CONFIGURACOES[
        config
    ]


    linha = {

        "configuration":
            config,

        "workload":
            metadata[
                "label"
            ],

        "request_interval":
            metadata[
                "interval"
            ],

        "n_runs":
            len(
                dados
            )
    }


    for metrica in METRICAS:

        (
            media,
            sd,
            low,
            high
        ) = calcular_ic95(

            dados[
                metrica
            ]
        )


        linha[
            f"{metrica}_mean"
        ] = media

        linha[
            f"{metrica}_sd"
        ] = sd

        linha[
            f"{metrica}_ci95_low"
        ] = low

        linha[
            f"{metrica}_ci95_high"
        ] = high


    resumo.append(
        linha
    )


df_agg = pd.DataFrame(
    resumo
)


df_agg.to_csv(
    OUTPUT_AGG,
    index=False
)


# ============================================================
# RESULTADOS PRINCIPAIS
# ============================================================

print("\n")

print(
    "MAIN COMPUTATIONAL RUNTIME RESULTS"
)

print("=" * 180)


header = (

    f"{'LOAD':<12}"

    f"{'INT':>6}"

    f"{'SEL N':>8}"

    f"{'CAND':>9}"

    f"{'OFFER':>11}"

    f"{'SEL AVG':>11}"

    f"{'SEL P95':>11}"

    f"{'SEL P99':>11}"

    f"{'DEC AVG':>11}"

    f"{'TOTAL ALG':>12}"

    f"{'CORR':>9}"
)


print(
    header
)

print("-" * 180)


for _, row in df_agg.iterrows():

    print(

        f"{row['workload']:<12}"

        f"{int(row['request_interval']):>6}"

        f"{row['n_selection_operations_mean']:>8.2f}"

        f"{row['mean_candidate_count_mean']:>9.2f}"

        f"{row['mean_offer_runtime_ms_mean']:>11.4f}"

        f"{row['mean_selection_runtime_ms_mean']:>11.4f}"

        f"{row['p95_selection_runtime_ms_mean']:>11.4f}"

        f"{row['p99_selection_runtime_ms_mean']:>11.4f}"

        f"{row['mean_decision_runtime_ms_mean']:>11.4f}"

        f"{row['total_algorithm_runtime_ms_mean']:>12.3f}"

        f"{row['corr_candidates_selection_runtime_mean']:>9.3f}"
    )


# ============================================================
# COMPARAÇÃO PAREADA
# ============================================================

baseline = (

    df_seed[
        df_seed[
            "configuration"
        ]
        ==
        BASELINE
    ]

    .copy()
)


METRICAS_PAREADAS = [

    "mean_candidate_count",

    "mean_offer_runtime_ms",

    "mean_selection_runtime_ms",

    "p95_selection_runtime_ms",

    "mean_decision_runtime_ms",

    "total_algorithm_runtime_ms"
]


comparacoes = []


for config in ORDEM:

    if config == BASELINE:

        continue


    alternativa = (

        df_seed[
            df_seed[
                "configuration"
            ]
            ==
            config
        ]

        .copy()
    )


    pareado = pd.merge(

        baseline,

        alternativa,

        on="seed",

        suffixes=(
            "_baseline",
            "_alternative"
        )
    )


    for metrica in METRICAS_PAREADAS:

        diferencas = (

            pareado[
                f"{metrica}_alternative"
            ]

            -

            pareado[
                f"{metrica}_baseline"
            ]
        )


        (
            mean_diff,
            sd_diff,
            low,
            high
        ) = calcular_ic95(
            diferencas
        )


        comparacoes.append({

            "configuration":
                config,

            "workload":
                CONFIGURACOES[
                    config
                ][
                    "label"
                ],

            "metric":
                metrica,

            "n_pairs":
                len(
                    pareado
                ),

            "mean_difference":
                mean_diff,

            "difference_sd":
                sd_diff,

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


df_paired = pd.DataFrame(
    comparacoes
)


df_paired.to_csv(
    OUTPUT_PAIRED,
    index=False
)


# ============================================================
# BASELINE DETAIL
# ============================================================

print("\n")

print(
    "BASELINE COMPUTATIONAL COST (INTERVAL=20)"
)

print("=" * 180)


row = df_agg[
    df_agg[
        "configuration"
    ]
    ==
    BASELINE
].iloc[
    0
]


print(

    f"Mean candidate count          : "
    f"{row['mean_candidate_count_mean']:.2f}"
)

print(

    f"Mean offer runtime            : "
    f"{row['mean_offer_runtime_ms_mean']:.4f} ms"
)

print(

    f"Mean selection runtime        : "
    f"{row['mean_selection_runtime_ms_mean']:.4f} ms "
    f"["
    f"{row['mean_selection_runtime_ms_ci95_low']:.4f}, "
    f"{row['mean_selection_runtime_ms_ci95_high']:.4f}"
    f"]"
)

print(

    f"Median selection runtime      : "
    f"{row['median_selection_runtime_ms_mean']:.4f} ms"
)

print(

    f"P95 selection runtime         : "
    f"{row['p95_selection_runtime_ms_mean']:.4f} ms"
)

print(

    f"P99 selection runtime         : "
    f"{row['p99_selection_runtime_ms_mean']:.4f} ms"
)

print(

    f"Mean full decision runtime    : "
    f"{row['mean_decision_runtime_ms_mean']:.4f} ms"
)

print(

    f"P95 full decision runtime     : "
    f"{row['p95_decision_runtime_ms_mean']:.4f} ms"
)

print(

    f"Total algorithm runtime / run : "
    f"{row['total_algorithm_runtime_ms_mean']:.3f} ms"
)

print(

    f"Candidate/runtime correlation : "
    f"{row['corr_candidates_selection_runtime_mean']:.3f}"
)


# ============================================================
# SANITY / SCALING
# ============================================================

print("\n")

print(
    "AUTOMATIC RUNTIME SANITY CHECK"
)

print("=" * 180)


very_high = df_agg[
    df_agg[
        "configuration"
    ]
    ==
    "runtime_very_high"
].iloc[
    0
]


baseline_row = row


selection_ratio = (

    very_high[
        "mean_selection_runtime_ms_mean"
    ]

    /

    baseline_row[
        "mean_selection_runtime_ms_mean"
    ]
)


total_ratio = (

    very_high[
        "total_algorithm_runtime_ms_mean"
    ]

    /

    baseline_row[
        "total_algorithm_runtime_ms_mean"
    ]
)


print(

    f"Very-high / baseline "
    f"mean selection runtime ratio = "
    f"{selection_ratio:.3f}"
)


print(

    f"Very-high / baseline "
    f"total algorithm runtime ratio = "
    f"{total_ratio:.3f}"
)


if selection_ratio < 1.5:

    print(

        "[PASS] Per-selection runtime remains "
        "relatively stable under very-high workload."
    )

else:

    print(

        "[CHECK] Per-selection runtime increased "
        "substantially under very-high workload."
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
    r"\caption{Computational overhead of SHADAI under increasing offered workloads. Runtime was measured using a high-resolution monotonic wall-clock timer in headless execution. Values are aggregated over 30 simulation seeds.}"
)

latex.append(
    r"\label{tab:shadai_runtime}"
)

latex.append(
    r"\resizebox{\textwidth}{!}{%"
)

latex.append(
    r"\begin{tabular}{l|cccccccc}"
)

latex.append(
    r"\hline"
)

latex.append(
    r"Workload & Interval & Selections & Candidates & Offer (ms) & Selection (ms) & Selection P95 (ms) & Decision (ms) & Total Algorithm Time (ms) \\"
)

latex.append(
    r"\hline"
)


for _, row in df_agg.iterrows():

    nome = row[
        "workload"
    ]


    if row[
        "configuration"
    ] == BASELINE:

        nome = (
            r"\textbf{High (baseline)}"
        )


    latex.append(

        f"{nome} & "

        f"{int(row['request_interval'])} & "

        f"{row['n_selection_operations_mean']:.1f} & "

        f"{row['mean_candidate_count_mean']:.1f} & "

        f"{row['mean_offer_runtime_ms_mean']:.4f} & "

        f"{row['mean_selection_runtime_ms_mean']:.4f} & "

        f"{row['p95_selection_runtime_ms_mean']:.4f} & "

        f"{row['mean_decision_runtime_ms_mean']:.4f} & "

        f"{row['total_algorithm_runtime_ms_mean']:.2f} "

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
    r"\parbox{0.98\textwidth}{\footnotesize Offer time measures candidate construction and eligibility filtering performed by the platform. Selection time measures score computation, cooldown adjustment, candidate ordering, and the $\epsilon$-greedy choice. Decision time is their sum for requests reaching the selection stage. Runtime values are hardware-dependent and represent computational overhead rather than communication, task-execution, or end-to-end service latency.}"
)

latex.append(
    r"\end{table*}"
)


with open(
    OUTPUT_LATEX,
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
    f"{OUTPUT_PER_SEED}"
)

print(
    f"Aggregated : "
    f"{OUTPUT_AGG}"
)

print(
    f"Paired     : "
    f"{OUTPUT_PAIRED}"
)

print(
    f"LaTeX      : "
    f"{OUTPUT_LATEX}"
)


print("\n")

print("=" * 180)

print(
    "C15 RUNTIME ANALYSIS COMPLETED"
)

print("=" * 180)

print()