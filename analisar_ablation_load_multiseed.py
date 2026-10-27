# analisar_ablation_load_multiseed.py

import os
import math

import pandas as pd


try:

    from scipy.stats import t

except ImportError:

    t = None


# ============================================================
# CONFIGURAÇÃO
# ============================================================

LOAD_DIR = "load_ablation"


CONFIGURACOES = {

    "load_low": {
        "label": "Low",
        "interval": 60
    },

    "load_moderate": {
        "label": "Moderate",
        "interval": 40
    },

    "load_high": {
        "label": "High",
        "interval": 20
    },

    "load_very_high": {
        "label": "Very High",
        "interval": 10
    }
}


ORDEM = [

    "load_low",

    "load_moderate",

    "load_high",

    "load_very_high"
]


BASELINE = "load_high"


SEEDS = list(
    range(
        1,
        31
    )
)


RNG_SCHEME_REQUIRED = "split_v1"

TELEMETRY_SCHEME_REQUIRED = "c9_regret_v1"


# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_POR_SEED = os.path.join(

    LOAD_DIR,

    "resumo_ablation_load_por_seed.csv"
)


OUTPUT_AGREGADO = os.path.join(

    LOAD_DIR,

    "resumo_ablation_load_multiseed.csv"
)


OUTPUT_PAREADO = os.path.join(

    LOAD_DIR,

    "comparacao_pareada_load_vs_original.csv"
)


OUTPUT_LATEX = os.path.join(

    LOAD_DIR,

    "tabela_ablation_load_tnsm.tex"
)


# ============================================================
# AUXILIARES
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


    erro = (
        sd
        /
        math.sqrt(
            n
        )
    )


    if t is not None:

        tcrit = t.ppf(
            0.975,
            df=n - 1
        )

    else:

        tcrit = 1.96


    margem = (
        tcrit
        *
        erro
    )


    return (
        media,
        sd,
        media - margem,
        media + margem
    )


def calcular_jain(
    valores
):

    valores = list(
        valores
    )


    n = len(
        valores
    )


    if n == 0:

        return float("nan")


    soma = sum(
        valores
    )


    soma_quad = sum(
        x ** 2
        for x in valores
    )


    if soma_quad == 0:

        return float("nan")


    return (
        soma ** 2
        /
        (
            n
            *
            soma_quad
        )
    )


def validar_dataframe(
    df,
    intervalo_esperado
):

    obrigatorias = [

        "rng_scheme",

        "telemetry_scheme",

        "request_interval",

        "cooldown_steps"
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


    if int(
        intervalos[0]
    ) != intervalo_esperado:

        return False


    return True


def percentual_diferenca(
    valor,
    referencia
):

    if (
        pd.isna(
            referencia
        )
        or
        referencia == 0
    ):

        return float("nan")


    return (
        (
            valor
            - referencia
        )
        /
        referencia
        *
        100.0
    )


# ============================================================
# CABEÇALHO
# ============================================================

print("\n")

print("=" * 165)

print(
    "SHADAI - MULTI-SEED WORKLOAD SENSITIVITY ANALYSIS"
)

print("=" * 165)


print(
    f"Directory : {LOAD_DIR}"
)

print(
    f"Seeds     : {len(SEEDS)}"
)

print(
    f"Baseline  : "
    f"{CONFIGURACOES[BASELINE]['label']} "
    f"(interval="
    f"{CONFIGURACOES[BASELINE]['interval']})"
)

print("=" * 165)


# ============================================================
# LEITURA
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


    load_factor = (
        CONFIGURACOES[
            BASELINE
        ][
            "interval"
        ]
        /
        interval
    )


    for seed in SEEDS:

        experiment_id = (

            f"{configuracao}_"
            f"interval_"
            f"{interval:03d}_"
            f"seed_"
            f"{seed:02d}"
        )


        metrics_file = os.path.join(

            LOAD_DIR,

            (
                f"metricas_shadai_400_"
                f"{experiment_id}.csv"
            )
        )


        eligibility_file = os.path.join(

            LOAD_DIR,

            (
                f"eligibilidade_shadai_400_"
                f"{experiment_id}.csv"
            )
        )


        if (
            not os.path.exists(
                metrics_file
            )
            or
            not os.path.exists(
                eligibility_file
            )
        ):

            faltantes.append(
                (
                    configuracao,
                    seed
                )
            )

            continue


        df_metrics = pd.read_csv(
            metrics_file
        )


        df_elig = pd.read_csv(
            eligibility_file
        )


        if (
            df_metrics.empty
            or
            df_elig.empty
        ):

            invalidos.append(
                (
                    configuracao,
                    seed
                )
            )

            continue


        if (
            not validar_dataframe(
                df_metrics,
                interval
            )
            or
            not validar_dataframe(
                df_elig,
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
        # REQUESTS
        #
        # Cada linha de eligibility corresponde a uma request
        # gerada quando havia veículos ativos.
        # ====================================================

        total_requests = len(
            df_elig
        )


        successful_allocations = len(
            df_metrics
        )


        allocation_rate = (

            successful_allocations
            /
            total_requests

            if total_requests > 0

            else float("nan")
        )


        # ====================================================
        # ELEGIBILITY GLOBAL
        # ====================================================

        total_vehicle_evaluations = (

            pd.to_numeric(

                df_elig[
                    "active_vehicles"
                ],

                errors="coerce"
            )

            .sum()
        )


        total_eligible = (

            pd.to_numeric(

                df_elig[
                    "eligible"
                ],

                errors="coerce"
            )

            .sum()
        )


        overall_eligibility_rate = (

            total_eligible
            /
            total_vehicle_evaluations

            if total_vehicle_evaluations > 0

            else float("nan")
        )


        # ====================================================
        # MÉTRICAS DOS SELECIONADOS
        # ====================================================

        avg_score = (

            pd.to_numeric(
                df_metrics[
                    "score"
                ],
                errors="coerce"
            )
            .mean()
        )


        avg_distance = (

            pd.to_numeric(
                df_metrics[
                    "distancia"
                ],
                errors="coerce"
            )
            .mean()
        )


        # IMPORTANTE:
        # esta é a energia disponível no instante da seleção,
        # antes de executar o serviço.
        avg_selected_energy = (

            pd.to_numeric(
                df_metrics[
                    "energia"
                ],
                errors="coerce"
            )
            .mean()
        )


        avg_candidate_count = (

            pd.to_numeric(
                df_metrics[
                    "candidate_count"
                ],
                errors="coerce"
            )
            .mean()
        )


        # ====================================================
        # EXPLORATION
        # ====================================================

        exploration_rate = (

            df_metrics[
                "selection_mode"
            ]
            .astype(str)
            .str.lower()
            .eq(
                "exploration"
            )
            .mean()
        )


        # ====================================================
        # COOLDOWN
        # ====================================================

        cooldown_bool = (

            df_metrics[
                "cooldown_penalty"
            ]
            .astype(str)
            .str.lower()
            .isin(
                [
                    "true",
                    "1"
                ]
            )
        )


        cooldown_rate = (
            cooldown_bool.mean()
        )


        # ====================================================
        # UNIQUE VEHICLES
        # ====================================================

        unique_vehicles = (

            df_metrics[
                "veiculo"
            ]
            .nunique()
        )


        # ====================================================
        # JAIN ENTRE VEÍCULOS SELECIONADOS
        # ====================================================

        allocation_counts = (

            df_metrics[
                "veiculo"
            ]
            .value_counts()
        )


        jain_fairness = (
            calcular_jain(
                allocation_counts.values
            )
        )


        # ====================================================
        # REQUEST RATE NORMALIZADA
        #
        # Relação com o baseline:
        # interval=20 -> 1.0
        # interval=10 -> 2.0
        # interval=40 -> 0.5
        # interval=60 -> 0.333...
        # ====================================================

        resultados.append({

            "configuration":
                configuracao,

            "workload":
                label,

            "request_interval":
                interval,

            "load_factor_vs_baseline":
                load_factor,

            "seed":
                seed,

            "total_requests":
                total_requests,

            "successful_allocations":
                successful_allocations,

            "request_allocation_rate":
                allocation_rate,

            "overall_eligibility_rate":
                overall_eligibility_rate,

            "avg_score":
                avg_score,

            "avg_distance":
                avg_distance,

            "avg_selected_energy":
                avg_selected_energy,

            "avg_candidate_count":
                avg_candidate_count,

            "exploration_rate":
                exploration_rate,

            "cooldown_rate":
                cooldown_rate,

            "unique_selected_vehicles":
                unique_vehicles,

            "jain_selected_vehicles":
                jain_fairness
        })


# ============================================================
# DATAFRAME POR SEED
# ============================================================

df_seed = pd.DataFrame(
    resultados
)


if df_seed.empty:

    raise RuntimeError(
        "Nenhum experimento de workload válido encontrado."
    )


df_seed.to_csv(
    OUTPUT_POR_SEED,
    index=False
)


# ============================================================
# COMPLETUDE
# ============================================================

print("\n")

print(
    "DATASET COMPLETENESS"
)

print("-" * 165)


for configuracao in ORDEM:

    n = len(

        df_seed[
            df_seed[
                "configuration"
            ]
            ==
            configuracao
        ]
    )


    print(

        f"{configuracao:<17}: "
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
# MÉTRICAS PARA AGREGAÇÃO
# ============================================================

METRICAS = [

    "total_requests",

    "successful_allocations",

    "request_allocation_rate",

    "overall_eligibility_rate",

    "avg_score",

    "avg_distance",

    "avg_selected_energy",

    "avg_candidate_count",

    "exploration_rate",

    "cooldown_rate",

    "unique_selected_vehicles",

    "jain_selected_vehicles"
]


# ============================================================
# AGREGAÇÃO
# ============================================================

resumo = []


for configuracao in ORDEM:

    dados = df_seed[
        df_seed[
            "configuration"
        ]
        ==
        configuracao
    ]


    if dados.empty:

        continue


    config = CONFIGURACOES[
        configuracao
    ]


    linha = {

        "configuration":
            configuracao,

        "workload":
            config[
                "label"
            ],

        "request_interval":
            config[
                "interval"
            ],

        "load_factor_vs_baseline":

            CONFIGURACOES[
                BASELINE
            ][
                "interval"
            ]
            /
            config[
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
            ci_low,
            ci_high
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
        ] = ci_low


        linha[
            f"{metrica}_ci95_high"
        ] = ci_high


    resumo.append(
        linha
    )


df_agregado = pd.DataFrame(
    resumo
)


df_agregado.to_csv(
    OUTPUT_AGREGADO,
    index=False
)


# ============================================================
# CONSOLE PRINCIPAL
# ============================================================

print("\n")

print(
    "MAIN WORKLOAD RESULTS"
)

print("=" * 165)


header = (

    f"{'LOAD':<12}"

    f"{'INT':>6}"

    f"{'FACTOR':>9}"

    f"{'REQ':>9}"

    f"{'ALLOC':>9}"

    f"{'ALLOC%':>10}"

    f"{'ELIG%':>10}"

    f"{'SCORE':>10}"

    f"{'DIST':>10}"

    f"{'ENERGY':>10}"

    f"{'CAND':>9}"

    f"{'JAIN':>9}"
)


print(
    header
)

print("-" * 165)


for _, row in df_agregado.iterrows():

    print(

        f"{row['workload']:<12}"

        f"{int(row['request_interval']):>6}"

        f"{row['load_factor_vs_baseline']:>9.2f}"

        f"{row['total_requests_mean']:>9.2f}"

        f"{row['successful_allocations_mean']:>9.2f}"

        f"{row['request_allocation_rate_mean'] * 100:>9.2f}%"

        f"{row['overall_eligibility_rate_mean'] * 100:>9.2f}%"

        f"{row['avg_score_mean']:>10.4f}"

        f"{row['avg_distance_mean']:>10.2f}"

        f"{row['avg_selected_energy_mean']:>10.2f}"

        f"{row['avg_candidate_count_mean']:>9.2f}"

        f"{row['jain_selected_vehicles_mean']:>9.4f}"
    )


# ============================================================
# COMPARAÇÕES PAREADAS COM BASELINE
# ============================================================

baseline_df = (

    df_seed[
        df_seed[
            "configuration"
        ]
        ==
        BASELINE
    ]

    .copy()
)


comparacoes = []


for configuracao in ORDEM:

    if configuracao == BASELINE:

        continue


    alt_df = (

        df_seed[
            df_seed[
                "configuration"
            ]
            ==
            configuracao
        ]

        .copy()
    )


    pareado = pd.merge(

        baseline_df,

        alt_df,

        on="seed",

        suffixes=(
            "_baseline",
            "_alternative"
        )
    )


    for metrica in METRICAS:

        base_col = (
            f"{metrica}_baseline"
        )

        alt_col = (
            f"{metrica}_alternative"
        )


        diferencas = (

            pareado[
                alt_col
            ]

            -

            pareado[
                base_col
            ]
        )


        (
            media_diff,
            sd_diff,
            ci_low,
            ci_high
        ) = calcular_ic95(
            diferencas
        )


        baseline_mean = (

            pareado[
                base_col
            ]
            .mean()
        )


        alternative_mean = (

            pareado[
                alt_col
            ]
            .mean()
        )


        comparacoes.append({

            "configuration":
                configuracao,

            "workload":
                CONFIGURACOES[
                    configuracao
                ][
                    "label"
                ],

            "request_interval":
                CONFIGURACOES[
                    configuracao
                ][
                    "interval"
                ],

            "metric":
                metrica,

            "n_pairs":
                len(
                    pareado
                ),

            "baseline_mean":
                baseline_mean,

            "alternative_mean":
                alternative_mean,

            "mean_difference":
                media_diff,

            "difference_sd":
                sd_diff,

            "difference_ci95_low":
                ci_low,

            "difference_ci95_high":
                ci_high,

            "difference_pct":
                percentual_diferenca(

                    alternative_mean,

                    baseline_mean
                ),

            "ci95_excludes_zero":

                (
                    (
                        ci_low > 0
                    )

                    or

                    (
                        ci_high < 0
                    )
                )

                if (
                    not pd.isna(
                        ci_low
                    )

                    and

                    not pd.isna(
                        ci_high
                    )
                )

                else False
        })


df_pareado = pd.DataFrame(
    comparacoes
)


df_pareado.to_csv(
    OUTPUT_PAREADO,
    index=False
)


# ============================================================
# PRINCIPAIS COMPARAÇÕES
# ============================================================

print("\n")

print(
    "PAIRED COMPARISONS VS ORIGINAL LOAD (INTERVAL=20)"
)

print("=" * 165)


metricas_console = [

    "request_allocation_rate",

    "overall_eligibility_rate",

    "avg_score",

    "avg_distance",

    "avg_selected_energy",

    "jain_selected_vehicles"
]


for configuracao in ORDEM:

    if configuracao == BASELINE:

        continue


    label = CONFIGURACOES[
        configuracao
    ][
        "label"
    ]


    interval = CONFIGURACOES[
        configuracao
    ][
        "interval"
    ]


    print(
        f"\n{label} "
        f"(interval={interval}):"
    )


    subset = df_pareado[
        df_pareado[
            "configuration"
        ]
        ==
        configuracao
    ]


    for metrica in metricas_console:

        row = subset[
            subset[
                "metric"
            ]
            ==
            metrica
        ]


        if row.empty:

            continue


        row = row.iloc[
            0
        ]


        print(

            f"  "
            f"{metrica:<29} "

            f"diff="
            f"{row['mean_difference']:+.5f} | "

            f"CI95=["
            f"{row['difference_ci95_low']:+.5f}, "
            f"{row['difference_ci95_high']:+.5f}] | "

            f"Δ="
            f"{row['difference_pct']:+.2f}% | "

            f"excludes zero="
            f"{row['ci95_excludes_zero']}"
        )


# ============================================================
# SANITY CHECKS
# ============================================================

print("\n")

print(
    "AUTOMATIC WORKLOAD SANITY CHECK"
)

print("=" * 165)


ordenado = df_agregado.sort_values(
    "request_interval",
    ascending=False
)


requests = ordenado[
    "total_requests_mean"
].tolist()


if all(

    requests[i]
    <=
    requests[i + 1]

    for i in range(
        len(requests) - 1
    )
):

    print(

        "[PASS] Request count increases "
        "as request interval decreases."
    )

else:

    print(

        "[CHECK] Request count is not "
        "monotonic with workload."
    )


very_high = df_agregado[
    df_agregado[
        "configuration"
    ]
    ==
    "load_very_high"
]


baseline = df_agregado[
    df_agregado[
        "configuration"
    ]
    ==
    BASELINE
]


if (
    not very_high.empty
    and
    not baseline.empty
):

    ratio = (

        very_high.iloc[
            0
        ][
            "total_requests_mean"
        ]

        /

        baseline.iloc[
            0
        ][
            "total_requests_mean"
        ]
    )


    print(

        f"Very-high / baseline request ratio = "
        f"{ratio:.3f}"
    )


# ============================================================
# LATEX
# ============================================================

latex_lines = []


latex_lines.append(
    r"\begin{table*}[t]"
)

latex_lines.append(
    r"\centering"
)

latex_lines.append(
    r"\caption{SHADAI performance under different request workloads. The original workload uses a request interval of 20 simulation steps. Values are means over 30 simulation seeds.}"
)

latex_lines.append(
    r"\label{tab:workload_sensitivity}"
)

latex_lines.append(
    r"\resizebox{\textwidth}{!}{%"
)

latex_lines.append(
    r"\begin{tabular}{l|cccccccc}"
)

latex_lines.append(
    r"\hline"
)

latex_lines.append(
    r"Workload & Interval & Requests & Alloc. Rate & Eligibility & Score & Distance & Selected Energy & Jain \\"
)

latex_lines.append(
    r"\hline"
)


for _, row in df_agregado.iterrows():

    workload = row[
        "workload"
    ]


    if row[
        "configuration"
    ] == BASELINE:

        workload = (
            r"\textbf{High (baseline)}"
        )


    latex_lines.append(

        f"{workload} & "

        f"{int(row['request_interval'])} & "

        f"{row['total_requests_mean']:.1f} & "

        f"{row['request_allocation_rate_mean'] * 100:.2f}\\% & "

        f"{row['overall_eligibility_rate_mean'] * 100:.2f}\\% & "

        f"{row['avg_score_mean']:.4f} & "

        f"{row['avg_distance_mean']:.2f} & "

        f"{row['avg_selected_energy_mean']:.2f} & "

        f"{row['jain_selected_vehicles_mean']:.4f} "

        r"\\"
    )


latex_lines.append(
    r"\hline"
)

latex_lines.append(
    r"\end{tabular}%"
)

latex_lines.append(
    r"}"
)

latex_lines.append(
    r"\vspace{1mm}"
)

latex_lines.append(
    r"\parbox{0.98\textwidth}{\footnotesize Lower request intervals represent higher offered workloads. Selected Energy denotes the available energy of the selected vehicle at decision time, before service execution. Jain's index is computed over vehicles receiving at least one allocation in each run.}"
)

latex_lines.append(
    r"\end{table*}"
)


with open(
    OUTPUT_LATEX,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(
            latex_lines
        )
    )


# ============================================================
# OUTPUTS
# ============================================================

print("\n")

print(
    "OUTPUT FILES"
)

print("=" * 165)


print(
    f"Per seed   : {OUTPUT_POR_SEED}"
)

print(
    f"Aggregated : {OUTPUT_AGREGADO}"
)

print(
    f"Paired     : {OUTPUT_PAREADO}"
)

print(
    f"LaTeX      : {OUTPUT_LATEX}"
)


print("\n")

print("=" * 165)

print(
    "WORKLOAD ANALYSIS COMPLETED"
)

print("=" * 165)

print()