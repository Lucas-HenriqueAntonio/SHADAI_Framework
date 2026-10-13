# analisar_ablation_epsilon_multiseed.py

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

EPSILON_DIR = "epsilon_ablation"


# ============================================================
# RNG
# ============================================================

RNG_SCHEME_REQUIRED = "split_v1"


# ============================================================
# CONFIGURAÇÕES
# ============================================================

CONFIGURACOES = {

    "epsilon_000": 0.0,

    "epsilon_010": 0.1,

    "epsilon_020": 0.2,

    "epsilon_030": 0.3,

    "epsilon_050": 0.5,

    "epsilon_100": 1.0
}


ORDEM = list(
    CONFIGURACOES.keys()
)


BASELINE = "epsilon_020"


SEEDS = list(
    range(
        1,
        31
    )
)


# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_POR_SEED = os.path.join(

    EPSILON_DIR,

    "resumo_ablation_epsilon_por_seed.csv"
)


OUTPUT_AGREGADO = os.path.join(

    EPSILON_DIR,

    "resumo_ablation_epsilon_multiseed.csv"
)


OUTPUT_PAREADO = os.path.join(

    EPSILON_DIR,

    "comparacao_pareada_epsilon_vs_020.csv"
)


OUTPUT_MODOS = os.path.join(

    EPSILON_DIR,

    "resumo_selection_modes_epsilon_multiseed.csv"
)


# ============================================================
# MÉTRICAS
# ============================================================

METRICAS = [

    "request_allocation_rate",

    "overall_eligibility_rate",

    "exploration_rate",

    "avg_score_base",

    "avg_score_final",

    "avg_reputation_norm",

    "avg_energy_norm",

    "avg_proximity_norm",

    "avg_distance",

    "avg_energy",

    "avg_reputation",

    "avg_utility",

    "unique_vehicles",

    "allocations_per_vehicle",

    "cooldown_penalty_rate"
]


# ============================================================
# FUNÇÕES
# ============================================================

def safe_div(
    numerador,
    denominador
):

    if denominador == 0:

        return float(
            "nan"
        )

    return (
        numerador
        /
        denominador
    )


def bool_mean(
    serie
):

    if serie.empty:

        return float(
            "nan"
        )


    valores = (

        serie.astype(
            str
        )

        .str.lower()

        .map({

            "true": 1,

            "false": 0,

            "1": 1,

            "0": 0
        })
    )


    return valores.mean()


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

        t_critico = t.ppf(
            0.975,
            df=n - 1
        )

    else:

        t_critico = 1.96


    margem = (
        t_critico
        *
        erro
    )


    return (

        media,

        sd,

        media - margem,

        media + margem
    )


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

        return float(
            "nan"
        )


    return (

        (
            valor
            - referencia
        )

        /
        referencia

    ) * 100.0


def validar_rng(
    dataframe
):

    if "rng_scheme" not in dataframe.columns:

        return False


    valores = (

        dataframe[
            "rng_scheme"
        ]

        .dropna()

        .astype(str)

        .unique()
    )


    if len(
        valores
    ) == 0:

        return False


    return all(

        valor
        ==
        RNG_SCHEME_REQUIRED

        for valor in valores
    )


# ============================================================
# CABEÇALHO
# ============================================================

print("\n")
print("=" * 155)

print(
    "SHADAI - MULTI-SEED EPSILON SENSITIVITY ANALYSIS"
)

print("=" * 155)

print(
    f"Directory   : {EPSILON_DIR}"
)

print(
    f"RNG scheme : {RNG_SCHEME_REQUIRED}"
)

print(
    f"Seeds       : {len(SEEDS)}"
)

print(
    f"Expected    : "
    f"{len(CONFIGURACOES) * len(SEEDS)} runs"
)

print(
    f"Baseline    : "
    f"{BASELINE} "
    f"(epsilon="
    f"{CONFIGURACOES[BASELINE]:.2f})"
)

print("=" * 155)


# ============================================================
# RESULTADOS
# ============================================================

resultados = []

resultados_modos = []

faltantes = []

rng_invalidos = []


# ============================================================
# LEITURA DOS 180 EXPERIMENTOS
# ============================================================

for configuracao in ORDEM:

    epsilon = (
        CONFIGURACOES[
            configuracao
        ]
    )


    for seed in SEEDS:

        experiment_id = (

            f"{configuracao}_"
            f"seed_"
            f"{seed:02d}"
        )


        metrics_file = os.path.join(

            EPSILON_DIR,

            (
                f"metricas_shadai_400_"
                f"{experiment_id}.csv"
            )
        )


        eligibility_file = os.path.join(

            EPSILON_DIR,

            (
                f"eligibilidade_shadai_400_"
                f"{experiment_id}.csv"
            )
        )


        # ====================================================
        # EXISTÊNCIA
        # ====================================================

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


        # ====================================================
        # LEITURA
        # ====================================================

        df_metricas = pd.read_csv(
            metrics_file
        )


        df_elig = pd.read_csv(
            eligibility_file
        )


        if df_elig.empty:

            faltantes.append(
                (
                    configuracao,
                    seed
                )
            )

            continue


        # ====================================================
        # VALIDA RNG
        # ====================================================

        if (
            not validar_rng(
                df_elig
            )
            or
            (
                not df_metricas.empty
                and
                not validar_rng(
                    df_metricas
                )
            )
        ):

            rng_invalidos.append(
                (
                    configuracao,
                    seed
                )
            )

            continue


        # ====================================================
        # REQUESTS
        # ====================================================

        requests = len(
            df_elig
        )


        vehicle_evaluations = (

            df_elig[
                "active_vehicles"
            ].sum()
        )


        total_eligible = (

            df_elig[
                "eligible"
            ].sum()
        )


        overall_eligibility_rate = (
            safe_div(

                total_eligible,

                vehicle_evaluations
            )
        )


        mean_eligibility_rate = (

            df_elig[
                "eligibility_rate"
            ].mean()
        )


        # ====================================================
        # ALOCAÇÕES
        # ====================================================

        allocations = len(
            df_metricas
        )


        request_allocation_rate = (
            safe_div(

                allocations,

                requests
            )
        )


        # ====================================================
        # SEM ALOCAÇÃO
        # ====================================================

        if df_metricas.empty:

            resultados.append({

                "configuration":
                    configuracao,

                "epsilon":
                    epsilon,

                "seed":
                    seed,

                "requests":
                    requests,

                "vehicle_evaluations":
                    vehicle_evaluations,

                "allocations":
                    allocations,

                "request_allocation_rate":
                    request_allocation_rate,

                "overall_eligibility_rate":
                    overall_eligibility_rate,

                "mean_eligibility_rate":
                    mean_eligibility_rate,

                "exploration_rate":
                    float("nan"),

                "avg_score_base":
                    float("nan"),

                "avg_score_final":
                    float("nan"),

                "avg_reputation_norm":
                    float("nan"),

                "avg_energy_norm":
                    float("nan"),

                "avg_proximity_norm":
                    float("nan"),

                "avg_distance":
                    float("nan"),

                "avg_energy":
                    float("nan"),

                "avg_reputation":
                    float("nan"),

                "avg_utility":
                    float("nan"),

                "unique_vehicles":
                    0,

                "allocations_per_vehicle":
                    float("nan"),

                "cooldown_penalty_rate":
                    float("nan")
            })

            continue


        # ====================================================
        # SELECTION MODE
        # ====================================================

        selection_mode = (

            df_metricas[
                "selection_mode"
            ]

            .astype(str)

            .str.lower()
        )


        exploration_count = (

            selection_mode

            .eq(
                "exploration"
            )

            .sum()
        )


        exploitation_count = (

            selection_mode

            .eq(
                "exploitation"
            )

            .sum()
        )


        exploration_rate = (
            safe_div(

                exploration_count,

                allocations
            )
        )


        # ====================================================
        # MÉTRICAS
        # ====================================================

        avg_score_base = (

            df_metricas[
                "score_base"
            ].mean()
        )


        avg_score_final = (

            df_metricas[
                "score"
            ].mean()
        )


        avg_reputation_norm = (

            df_metricas[
                "reputacao_norm"
            ].mean()
        )


        avg_energy_norm = (

            df_metricas[
                "energia_norm"
            ].mean()
        )


        avg_proximity_norm = (

            df_metricas[
                "proximidade_norm"
            ].mean()
        )


        avg_distance = (

            df_metricas[
                "distancia"
            ].mean()
        )


        avg_energy = (

            df_metricas[
                "energia"
            ].mean()
        )


        avg_reputation = (

            df_metricas[
                "reputacao"
            ].mean()
        )


        avg_utility = (

            df_metricas[
                "utilidade"
            ].mean()
        )


        unique_vehicles = (

            df_metricas[
                "veiculo"
            ].nunique()
        )


        allocations_per_vehicle = (
            safe_div(

                allocations,

                unique_vehicles
            )
        )


        cooldown_penalty_rate = (
            bool_mean(

                df_metricas[
                    "cooldown_penalty"
                ]
            )
        )


        # ====================================================
        # POR SEED
        # ====================================================

        resultados.append({

            "configuration":
                configuracao,

            "epsilon":
                epsilon,

            "seed":
                seed,

            "requests":
                requests,

            "vehicle_evaluations":
                vehicle_evaluations,

            "allocations":
                allocations,

            "request_allocation_rate":
                request_allocation_rate,

            "overall_eligibility_rate":
                overall_eligibility_rate,

            "mean_eligibility_rate":
                mean_eligibility_rate,

            "exploration_count":
                exploration_count,

            "exploitation_count":
                exploitation_count,

            "exploration_rate":
                exploration_rate,

            "avg_score_base":
                avg_score_base,

            "avg_score_final":
                avg_score_final,

            "avg_reputation_norm":
                avg_reputation_norm,

            "avg_energy_norm":
                avg_energy_norm,

            "avg_proximity_norm":
                avg_proximity_norm,

            "avg_distance":
                avg_distance,

            "avg_energy":
                avg_energy,

            "avg_reputation":
                avg_reputation,

            "avg_utility":
                avg_utility,

            "unique_vehicles":
                unique_vehicles,

            "allocations_per_vehicle":
                allocations_per_vehicle,

            "cooldown_penalty_rate":
                cooldown_penalty_rate
        })


        # ====================================================
        # EXPLORATION / EXPLOITATION POR SEED
        # ====================================================

        for modo in [
            "exploration",
            "exploitation"
        ]:

            subset = df_metricas[
                selection_mode
                ==
                modo
            ]


            if subset.empty:

                resultados_modos.append({

                    "configuration":
                        configuracao,

                    "epsilon":
                        epsilon,

                    "seed":
                        seed,

                    "selection_mode":
                        modo,

                    "n":
                        0,

                    "avg_score":
                        float("nan"),

                    "avg_reputation_norm":
                        float("nan"),

                    "avg_energy_norm":
                        float("nan"),

                    "avg_proximity_norm":
                        float("nan"),

                    "avg_distance":
                        float("nan"),

                    "avg_energy":
                        float("nan"),

                    "avg_utility":
                        float("nan")
                })

                continue


            resultados_modos.append({

                "configuration":
                    configuracao,

                "epsilon":
                    epsilon,

                "seed":
                    seed,

                "selection_mode":
                    modo,

                "n":
                    len(
                        subset
                    ),

                "avg_score":
                    subset[
                        "score"
                    ].mean(),

                "avg_reputation_norm":
                    subset[
                        "reputacao_norm"
                    ].mean(),

                "avg_energy_norm":
                    subset[
                        "energia_norm"
                    ].mean(),

                "avg_proximity_norm":
                    subset[
                        "proximidade_norm"
                    ].mean(),

                "avg_distance":
                    subset[
                        "distancia"
                    ].mean(),

                "avg_energy":
                    subset[
                        "energia"
                    ].mean(),

                "avg_utility":
                    subset[
                        "utilidade"
                    ].mean()
            })


# ============================================================
# DATAFRAMES
# ============================================================

df_seed = pd.DataFrame(
    resultados
)


df_modos_seed = pd.DataFrame(
    resultados_modos
)


if df_seed.empty:

    raise RuntimeError(
        "Nenhum experimento multi-seed válido "
        "de epsilon foi encontrado."
    )


# ============================================================
# SALVA POR SEED
# ============================================================

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

print("-" * 155)


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

        f"{configuracao:<14}: "

        f"{n:>2}/"
        f"{len(SEEDS)} runs"
    )


if faltantes:

    print(

        f"\n[WARNING] "
        f"{len(faltantes)} "
        f"runs are missing."
    )


if rng_invalidos:

    print(

        f"\n[WARNING] "
        f"{len(rng_invalidos)} "
        f"runs have incompatible RNG scheme."
    )


if (
    not faltantes
    and
    not rng_invalidos
):

    print(

        "\nAll expected 180 runs were found "
        "with rng_scheme=split_v1."
    )


# ============================================================
# AGREGAÇÃO DAS MÉTRICAS
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


    linha = {

        "configuration":
            configuracao,

        "epsilon":
            CONFIGURACOES[
                configuracao
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
# COMPARAÇÃO PAREADA CONTRA EPSILON=0.2
# ============================================================

df_baseline = (

    df_seed[
        df_seed[
            "configuration"
        ]
        ==
        BASELINE
    ]

    .copy()
)


resultados_pareados = []


for configuracao in ORDEM:

    if configuracao == BASELINE:

        continue


    df_alt = (

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

        df_baseline,

        df_alt,

        on="seed",

        suffixes=(
            "_baseline",
            "_alternative"
        )
    )


    for metrica in METRICAS:

        baseline_col = (
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
                baseline_col
            ]
        )


        (
            mean_diff,
            sd_diff,
            ci_low,
            ci_high
        ) = calcular_ic95(
            diferencas
        )


        baseline_mean = (

            pareado[
                baseline_col
            ].mean()
        )


        alternative_mean = (

            pareado[
                alt_col
            ].mean()
        )


        percentual = (
            percentual_diferenca(

                alternative_mean,

                baseline_mean
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
        ):

            ci_excludes_zero = (

                ci_low > 0

                or

                ci_high < 0
            )

        else:

            ci_excludes_zero = False


        resultados_pareados.append({

            "configuration":
                configuracao,

            "epsilon":
                CONFIGURACOES[
                    configuracao
                ],

            "baseline_epsilon":
                CONFIGURACOES[
                    BASELINE
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
                mean_diff,

            "difference_sd":
                sd_diff,

            "difference_ci95_low":
                ci_low,

            "difference_ci95_high":
                ci_high,

            "difference_pct":
                percentual,

            "ci95_excludes_zero":
                ci_excludes_zero
        })


df_pareado = pd.DataFrame(
    resultados_pareados
)


df_pareado.to_csv(
    OUTPUT_PAREADO,
    index=False
)


# ============================================================
# AGREGA SELECTION MODES
# ============================================================

resumo_modos = []


METRICAS_MODOS = [

    "avg_score",

    "avg_reputation_norm",

    "avg_energy_norm",

    "avg_proximity_norm",

    "avg_distance",

    "avg_energy",

    "avg_utility"
]


for configuracao in ORDEM:

    for modo in [
        "exploration",
        "exploitation"
    ]:

        dados = df_modos_seed[
            (
                df_modos_seed[
                    "configuration"
                ]
                ==
                configuracao
            )
            &
            (
                df_modos_seed[
                    "selection_mode"
                ]
                ==
                modo
            )
        ]


        if dados.empty:

            continue


        linha = {

            "configuration":
                configuracao,

            "epsilon":
                CONFIGURACOES[
                    configuracao
                ],

            "selection_mode":
                modo,

            "runs_with_data":
                int(
                    (
                        dados[
                            "n"
                        ]
                        >
                        0
                    ).sum()
                ),

            "total_selections":
                int(
                    dados[
                        "n"
                    ].sum()
                )
        }


        for metrica in METRICAS_MODOS:

            validos = dados.loc[

                dados[
                    "n"
                ]
                >
                0,

                metrica
            ]


            (
                media,
                sd,
                ci_low,
                ci_high
            ) = calcular_ic95(
                validos
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


        resumo_modos.append(
            linha
        )


df_modos = pd.DataFrame(
    resumo_modos
)


df_modos.to_csv(
    OUTPUT_MODOS,
    index=False
)


# ============================================================
# CONSOLE
# ============================================================

pd.set_option(
    "display.width",
    300
)


pd.set_option(
    "display.max_columns",
    None
)


pd.set_option(
    "display.precision",
    4
)


# ============================================================
# RESULTADOS PRINCIPAIS
# ============================================================

print("\n")

print(
    "MAIN MULTI-SEED EPSILON RESULTS"
)

print("=" * 155)


cabecalho = (

    f"{'CONFIG':<14}"

    f"{'EPS':>7}"

    f"{'N':>5}"

    f"{'EXP%':>10}"

    f"{'ALLOC%':>10}"

    f"{'SCORE':>11}"

    f"{'REP_N':>10}"

    f"{'ENER_N':>10}"

    f"{'PROX_N':>10}"

    f"{'DIST':>11}"

    f"{'ENERGY':>11}"

    f"{'UNIQUE':>9}"
)


print(
    cabecalho
)

print("-" * 155)


for _, row in df_agregado.sort_values(
    "epsilon"
).iterrows():

    print(

        f"{row['configuration']:<14}"

        f"{row['epsilon']:>7.2f}"

        f"{int(row['n_runs']):>5}"

        f"{row['exploration_rate_mean'] * 100:>9.2f}%"

        f"{row['request_allocation_rate_mean'] * 100:>9.2f}%"

        f"{row['avg_score_final_mean']:>11.4f}"

        f"{row['avg_reputation_norm_mean']:>10.4f}"

        f"{row['avg_energy_norm_mean']:>10.4f}"

        f"{row['avg_proximity_norm_mean']:>10.4f}"

        f"{row['avg_distance_mean']:>11.2f}"

        f"{row['avg_energy_mean']:>11.2f}"

        f"{row['unique_vehicles_mean']:>9.2f}"
    )


# ============================================================
# EPSILON VS EXPLORAÇÃO OBSERVADA
# ============================================================

print("\n")

print(
    "CONFIGURED EPSILON VS OBSERVED EXPLORATION"
)

print("=" * 155)


for _, row in df_agregado.sort_values(
    "epsilon"
).iterrows():

    esperado = row[
        "epsilon"
    ]


    observado = row[
        "exploration_rate_mean"
    ]


    print(

        f"epsilon="
        f"{esperado:.2f} | "

        f"observed="
        f"{observado * 100:.2f}% | "

        f"CI95=["
        f"{row['exploration_rate_ci95_low'] * 100:.2f}%, "
        f"{row['exploration_rate_ci95_high'] * 100:.2f}%] | "

        f"difference="
        f"{(observado - esperado) * 100:+.2f} pp"
    )


# ============================================================
# CONTROLES DE CENÁRIO
# ============================================================

print("\n")

print(
    "ELIGIBILITY / ALLOCATION CONTROL"
)

print("=" * 155)


for _, row in df_agregado.sort_values(
    "epsilon"
).iterrows():

    print(

        f"epsilon="
        f"{row['epsilon']:.2f} | "

        f"eligibility="
        f"{row['overall_eligibility_rate_mean'] * 100:.2f}% | "

        f"allocation="
        f"{row['request_allocation_rate_mean'] * 100:.2f}% | "

        f"cooldown="
        f"{row['cooldown_penalty_rate_mean'] * 100:.2f}%"
    )


# ============================================================
# COMPARAÇÃO PAREADA - SCORE
# ============================================================

print("\n")

print(
    "PAIRED SCORE COMPARISON VS EPSILON=0.20"
)

print("=" * 155)


score_rows = df_pareado[
    df_pareado[
        "metric"
    ]
    ==
    "avg_score_final"
].sort_values(
    "epsilon"
)


for _, row in score_rows.iterrows():

    print(

        f"epsilon="
        f"{row['epsilon']:.2f} | "

        f"Δscore="
        f"{row['mean_difference']:+.5f} | "

        f"CI95=["
        f"{row['difference_ci95_low']:+.5f}, "
        f"{row['difference_ci95_high']:+.5f}] | "

        f"Δ%="
        f"{row['difference_pct']:+.2f}% | "

        f"excludes zero="
        f"{row['ci95_excludes_zero']}"
    )


# ============================================================
# COMPARAÇÃO PAREADA - DISTÂNCIA
# ============================================================

print("\n")

print(
    "PAIRED DISTANCE COMPARISON VS EPSILON=0.20"
)

print("=" * 155)


distance_rows = df_pareado[
    df_pareado[
        "metric"
    ]
    ==
    "avg_distance"
].sort_values(
    "epsilon"
)


for _, row in distance_rows.iterrows():

    print(

        f"epsilon="
        f"{row['epsilon']:.2f} | "

        f"Δdistance="
        f"{row['mean_difference']:+.2f} | "

        f"CI95=["
        f"{row['difference_ci95_low']:+.2f}, "
        f"{row['difference_ci95_high']:+.2f}] | "

        f"excludes zero="
        f"{row['ci95_excludes_zero']}"
    )


# ============================================================
# EXTREMOS: EPSILON 0 E 1 VS BASELINE
# ============================================================

print("\n")

print(
    "KEY CONTRASTS AGAINST EPSILON=0.20"
)

print("=" * 155)


for configuracao in [
    "epsilon_000",
    "epsilon_100"
]:

    epsilon = (
        CONFIGURACOES[
            configuracao
        ]
    )


    print(
        f"\nepsilon={epsilon:.2f}"
    )


    for metrica in [

        "exploration_rate",

        "avg_score_final",

        "avg_reputation_norm",

        "avg_energy_norm",

        "avg_proximity_norm",

        "avg_distance",

        "avg_energy",

        "unique_vehicles"
    ]:

        linha = df_pareado[
            (
                df_pareado[
                    "configuration"
                ]
                ==
                configuracao
            )
            &
            (
                df_pareado[
                    "metric"
                ]
                ==
                metrica
            )
        ]


        if linha.empty:

            continue


        row = linha.iloc[
            0
        ]


        print(

            f"  "
            f"{metrica:<24} "

            f"Δ="
            f"{row['mean_difference']:+.5f} | "

            f"CI95=["
            f"{row['difference_ci95_low']:+.5f}, "
            f"{row['difference_ci95_high']:+.5f}]"
        )


# ============================================================
# EXPLORATION VS EXPLOITATION
# ============================================================

print("\n")

print(
    "EXPLORATION VS EXPLOITATION QUALITY"
)

print("=" * 155)


for configuracao in ORDEM:

    epsilon = (
        CONFIGURACOES[
            configuracao
        ]
    )


    print(
        f"\nepsilon={epsilon:.2f}"
    )


    subset = df_modos[
        df_modos[
            "configuration"
        ]
        ==
        configuracao
    ]


    for modo in [
        "exploration",
        "exploitation"
    ]:

        linha = subset[
            subset[
                "selection_mode"
            ]
            ==
            modo
        ]


        if linha.empty:

            continue


        row = linha.iloc[
            0
        ]


        print(

            f"  {modo:<12} | "

            f"runs="
            f"{int(row['runs_with_data']):>2} | "

            f"n="
            f"{int(row['total_selections']):>4} | "

            f"score="
            f"{row['avg_score_mean']:.4f} | "

            f"rep="
            f"{row['avg_reputation_norm_mean']:.4f} | "

            f"energy="
            f"{row['avg_energy_norm_mean']:.4f} | "

            f"proximity="
            f"{row['avg_proximity_norm_mean']:.4f} | "

            f"distance="
            f"{row['avg_distance_mean']:.2f}"
        )


# ============================================================
# CORRELAÇÕES
# ============================================================

print("\n")

print(
    "EPSILON TREND DIAGNOSTICS"
)

print("=" * 155)


df_corr = df_agregado.sort_values(
    "epsilon"
)


cor_exp = (

    df_corr[
        [
            "epsilon",
            "exploration_rate_mean"
        ]
    ]

    .corr()

    .iloc[
        0,
        1
    ]
)


cor_score = (

    df_corr[
        [
            "epsilon",
            "avg_score_final_mean"
        ]
    ]

    .corr()

    .iloc[
        0,
        1
    ]
)


cor_distance = (

    df_corr[
        [
            "epsilon",
            "avg_distance_mean"
        ]
    ]

    .corr()

    .iloc[
        0,
        1
    ]
)


print(

    f"corr(epsilon, exploration) = "
    f"{cor_exp:.4f}"
)


print(

    f"corr(epsilon, score)       = "
    f"{cor_score:.4f}"
)


print(

    f"corr(epsilon, distance)    = "
    f"{cor_distance:.4f}"
)


# ============================================================
# SANITY CHECK
# ============================================================

print("\n")

print(
    "AUTOMATIC MULTI-SEED SANITY CHECK"
)

print("=" * 155)


linha_zero = df_agregado[
    df_agregado[
        "configuration"
    ]
    ==
    "epsilon_000"
]


linha_um = df_agregado[
    df_agregado[
        "configuration"
    ]
    ==
    "epsilon_100"
]


if not linha_zero.empty:

    valor = linha_zero.iloc[
        0
    ][
        "exploration_rate_mean"
    ]


    if abs(
        valor
    ) < 1e-12:

        print(

            "[PASS] epsilon=0.0 produced "
            "0% exploration across the experiment."
        )

    else:

        print(

            "[CHECK] epsilon=0.0 produced "
            "non-zero exploration."
        )


if not linha_um.empty:

    valor = linha_um.iloc[
        0
    ][
        "exploration_rate_mean"
    ]


    if abs(
        valor - 1.0
    ) < 1e-12:

        print(

            "[PASS] epsilon=1.0 produced "
            "100% exploration across the experiment."
        )

    else:

        print(

            "[CHECK] epsilon=1.0 did not produce "
            "100% exploration."
        )


if cor_exp > 0.98:

    print(

        "[PASS] Observed exploration "
        "strongly follows configured epsilon."
    )

else:

    print(

        "[CHECK] Exploration does not closely "
        "follow configured epsilon."
    )


if cor_score < 0:

    print(

        "[PASS] Increasing epsilon is associated "
        "with lower selected score."
    )

else:

    print(

        "[CHECK] Score did not decrease "
        "with increasing epsilon."
    )


if cor_distance > 0:

    print(

        "[PASS] Increasing epsilon is associated "
        "with greater selected distance."
    )

else:

    print(

        "[CHECK] Distance did not increase "
        "with increasing epsilon."
    )


# ============================================================
# OUTPUTS
# ============================================================

print("\n")

print(
    "OUTPUT FILES"
)

print("=" * 155)


print(

    f"Per seed        : "
    f"{OUTPUT_POR_SEED}"
)


print(

    f"Aggregated      : "
    f"{OUTPUT_AGREGADO}"
)


print(

    f"Paired baseline : "
    f"{OUTPUT_PAREADO}"
)


print(

    f"Selection modes : "
    f"{OUTPUT_MODOS}"
)


# ============================================================
# NOTA FINAL
# ============================================================

print("\n")

print(
    "IMPORTANT"
)

print("=" * 155)


print(

    "The seed/run is the experimental replication unit."
)


print(

    "All accepted runs must use rng_scheme=split_v1."
)


print(

    "Because lambda and score weights remain fixed, "
    "score values are directly comparable across epsilon "
    "configurations."
)


print(

    "Paired confidence intervals compare each epsilon "
    "configuration against the SHADAI baseline epsilon=0.20 "
    "using matching simulation seeds."
)


print(

    "Changes in vehicle state caused by different policy "
    "decisions are legitimate downstream policy effects; "
    "exogenous request and RNG streams remain controlled."
)


print("\n")
print("=" * 155)

print(
    "MULTI-SEED EPSILON ANALYSIS COMPLETED"
)

print("=" * 155)
print()