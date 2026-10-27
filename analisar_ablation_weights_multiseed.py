# analisar_ablation_weights_multiseed.py

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

WEIGHT_DIR = "weight_ablation"


os.makedirs(
    WEIGHT_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

CONFIGURACOES = [

    "original",

    "equal",

    "reputation_high",

    "energy_high",

    "proximity_high",

    "no_reputation",

    "no_energy",

    "no_proximity"
]


PESOS = {

    "original":
        (0.4, 0.3, 0.3),

    "equal":
        (0.3333, 0.3333, 0.3334),

    "reputation_high":
        (0.7, 0.15, 0.15),

    "energy_high":
        (0.15, 0.7, 0.15),

    "proximity_high":
        (0.15, 0.15, 0.7),

    "no_reputation":
        (0.0, 0.5, 0.5),

    "no_energy":
        (0.5714, 0.0, 0.4286),

    "no_proximity":
        (0.5714, 0.4286, 0.0)
}


SEEDS = list(
    range(
        1,
        31
    )
)


# ============================================================
# SAÍDAS
# ============================================================

ARQUIVO_POR_SEED = os.path.join(

    WEIGHT_DIR,

    "resumo_ablation_weights_por_seed.csv"
)


ARQUIVO_AGREGADO = os.path.join(

    WEIGHT_DIR,

    "resumo_ablation_weights_multiseed.csv"
)


ARQUIVO_PAREADO = os.path.join(

    WEIGHT_DIR,

    "comparacao_pareada_weights_vs_original.csv"
)


# ============================================================
# MÉTRICAS PRINCIPAIS
# ============================================================

METRICAS_PRINCIPAIS = [

    "request_allocation_rate",

    "overall_eligibility_rate",

    "avg_score_base",

    "avg_score_final",

    "avg_reputation_norm",

    "avg_energy_norm",

    "avg_proximity_norm",

    "avg_distance",

    "avg_reputation",

    "avg_energy",

    "avg_utility",

    "unique_vehicles",

    "allocations_per_vehicle",

    "exploration_rate",

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
        / denominador
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
        / referencia
    ) * 100.0


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
        / math.sqrt(
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
        * erro
    )


    return (
        media,
        sd,
        media - margem,
        media + margem
    )


def bool_mean(
    serie
):

    if serie.empty:

        return float(
            "nan"
        )


    normalizada = (

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


    return normalizada.mean()


# ============================================================
# CABEÇALHO
# ============================================================

print("\n")
print("=" * 145)

print(
    "SHADAI - MULTI-SEED WEIGHT SENSITIVITY / ABLATION ANALYSIS"
)

print("=" * 145)

print(
    f"Directory : {WEIGHT_DIR}"
)

print(
    f"Seeds     : {len(SEEDS)}"
)

print(
    f"Expected  : "
    f"{len(CONFIGURACOES) * len(SEEDS)} runs"
)

print("=" * 145)


# ============================================================
# LEITURA POR SEED
# ============================================================

resultados = []

faltantes = []


for configuracao in CONFIGURACOES:

    w1, w2, w3 = (
        PESOS[
            configuracao
        ]
    )


    for seed in SEEDS:

        experiment_id = (

            f"weights_"
            f"{configuracao}_"
            f"seed_"
            f"{seed:02d}"
        )


        metrics_file = os.path.join(

            WEIGHT_DIR,

            (
                f"metricas_shadai_400_"
                f"{experiment_id}.csv"
            )
        )


        eligibility_file = os.path.join(

            WEIGHT_DIR,

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
        # REQUESTS / ELEGIBILIDADE
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


        overall_eligibility_rate = safe_div(

            total_eligible,

            vehicle_evaluations
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


        request_allocation_rate = safe_div(

            allocations,

            requests
        )


        # ====================================================
        # MÉTRICAS DOS SELECIONADOS
        # ====================================================

        if not df_metricas.empty:

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


            avg_reputation = (

                df_metricas[
                    "reputacao"
                ].mean()
            )


            avg_energy = (

                df_metricas[
                    "energia"
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


            allocations_per_vehicle = safe_div(

                allocations,

                unique_vehicles
            )


            exploration_count = (

                df_metricas[
                    "selection_mode"
                ]

                .astype(
                    str
                )

                .str.lower()

                .eq(
                    "exploration"
                )

                .sum()
            )


            exploration_rate = safe_div(

                exploration_count,

                allocations
            )


            cooldown_penalty_rate = bool_mean(

                df_metricas[
                    "cooldown_penalty"
                ]
            )


        else:

            avg_score_base = float("nan")

            avg_score_final = float("nan")

            avg_reputation_norm = float("nan")

            avg_energy_norm = float("nan")

            avg_proximity_norm = float("nan")

            avg_distance = float("nan")

            avg_reputation = float("nan")

            avg_energy = float("nan")

            avg_utility = float("nan")

            unique_vehicles = 0

            allocations_per_vehicle = float("nan")

            exploration_rate = float("nan")

            cooldown_penalty_rate = float("nan")


        # ====================================================
        # RESULTADO POR SEED
        # ====================================================

        resultados.append({

            "configuration":
                configuracao,

            "seed":
                seed,

            "w1":
                w1,

            "w2":
                w2,

            "w3":
                w3,

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

            "avg_reputation":
                avg_reputation,

            "avg_energy":
                avg_energy,

            "avg_utility":
                avg_utility,

            "unique_vehicles":
                unique_vehicles,

            "allocations_per_vehicle":
                allocations_per_vehicle,

            "exploration_rate":
                exploration_rate,

            "cooldown_penalty_rate":
                cooldown_penalty_rate
        })


# ============================================================
# DATAFRAME POR SEED
# ============================================================

df_seed = pd.DataFrame(
    resultados
)


if df_seed.empty:

    raise RuntimeError(
        "No valid weight experiment files were found."
    )


df_seed.to_csv(
    ARQUIVO_POR_SEED,
    index=False
)


# ============================================================
# COMPLETUDE
# ============================================================

print("\n")

print(
    "DATASET COMPLETENESS"
)

print("-" * 145)


for config in CONFIGURACOES:

    n = len(

        df_seed[
            df_seed[
                "configuration"
            ] == config
        ]
    )


    print(

        f"{config:<19}: "

        f"{n:>2}/"
        f"{len(SEEDS)} runs"
    )


if faltantes:

    print(

        f"\n[WARNING] "

        f"{len(faltantes)} "

        f"experiment pairs are missing."
    )

else:

    print(
        "\nAll expected 240 experiment pairs were found."
    )


# ============================================================
# AGREGAÇÃO
# ============================================================

resumo = []


for config in CONFIGURACOES:

    dados = df_seed[
        df_seed[
            "configuration"
        ] == config
    ]


    if dados.empty:

        continue


    linha = {

        "configuration":
            config,

        "n_runs":
            len(
                dados
            ),

        "w1":
            PESOS[
                config
            ][0],

        "w2":
            PESOS[
                config
            ][1],

        "w3":
            PESOS[
                config
            ][2]
    }


    for metrica in METRICAS_PRINCIPAIS:

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


# ============================================================
# DIFERENÇAS DAS MÉDIAS VS ORIGINAL
# ============================================================

original_agregado = (

    df_agregado[
        df_agregado[
            "configuration"
        ] == "original"
    ]

    .iloc[
        0
    ]
)


for metrica in METRICAS_PRINCIPAIS:

    coluna_media = (
        f"{metrica}_mean"
    )


    nova_coluna = (
        f"{metrica}_diff_pct_vs_original"
    )


    df_agregado[
        nova_coluna
    ] = (

        df_agregado[
            coluna_media
        ]

        .apply(

            lambda valor:

                percentual_diferenca(

                    valor,

                    original_agregado[
                        coluna_media
                    ]
                )
        )
    )


df_agregado.to_csv(
    ARQUIVO_AGREGADO,
    index=False
)


# ============================================================
# COMPARAÇÃO PAREADA POR SEED
# ============================================================

df_original = (

    df_seed[
        df_seed[
            "configuration"
        ] == "original"
    ]

    .copy()
)


resultados_pareados = []


for config in CONFIGURACOES:

    if config == "original":

        continue


    df_config = (

        df_seed[
            df_seed[
                "configuration"
            ] == config
        ]

        .copy()
    )


    pareado = pd.merge(

        df_original,

        df_config,

        on="seed",

        suffixes=(
            "_original",
            "_alternative"
        )
    )


    for metrica in METRICAS_PRINCIPAIS:

        coluna_original = (
            f"{metrica}_original"
        )


        coluna_alternativa = (
            f"{metrica}_alternative"
        )


        diferencas = (

            pareado[
                coluna_alternativa
            ]

            -

            pareado[
                coluna_original
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


        mean_original = (

            pareado[
                coluna_original
            ].mean()
        )


        mean_alternative = (

            pareado[
                coluna_alternativa
            ].mean()
        )


        pct_diff = percentual_diferenca(

            mean_alternative,

            mean_original
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
                config,

            "metric":
                metrica,

            "n_pairs":
                len(
                    pareado
                ),

            "original_mean":
                mean_original,

            "alternative_mean":
                mean_alternative,

            "mean_difference":
                mean_diff,

            "difference_sd":
                sd_diff,

            "difference_ci95_low":
                ci_low,

            "difference_ci95_high":
                ci_high,

            "difference_pct":
                pct_diff,

            "ci95_excludes_zero":
                ci_excludes_zero
        })


df_pareado = pd.DataFrame(
    resultados_pareados
)


df_pareado.to_csv(
    ARQUIVO_PAREADO,
    index=False
)


# ============================================================
# CONSOLE
# ============================================================

pd.set_option(
    "display.width",
    280
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
    "MAIN MULTI-SEED WEIGHT RESULTS"
)

print("=" * 145)


cabecalho = (

    f"{'CONFIGURATION':<19}"

    f"{'N':>4}"

    f"{'REP_N':>11}"

    f"{'ENER_N':>11}"

    f"{'PROX_N':>11}"

    f"{'DIST':>11}"

    f"{'ENERGY':>11}"

    f"{'UNIQUE':>9}"
)


print(
    cabecalho
)

print("-" * 145)


for _, row in df_agregado.iterrows():

    print(

        f"{row['configuration']:<19}"

        f"{int(row['n_runs']):>4}"

        f"{row['avg_reputation_norm_mean']:>11.4f}"

        f"{row['avg_energy_norm_mean']:>11.4f}"

        f"{row['avg_proximity_norm_mean']:>11.4f}"

        f"{row['avg_distance_mean']:>11.2f}"

        f"{row['avg_energy_mean']:>11.2f}"

        f"{row['unique_vehicles_mean']:>9.2f}"
    )


# ============================================================
# SENSITIVITY PRINCIPAL
# ============================================================

print("\n")

print(
    "DIRECT MULTI-SEED SENSITIVITY CHECK"
)

print("=" * 145)


checks = {

    "reputation_high":
        "avg_reputation_norm",

    "energy_high":
        "avg_energy_norm",

    "proximity_high":
        "avg_proximity_norm"
}


for config, metrica in checks.items():

    linha = df_pareado[
        (
            df_pareado[
                "configuration"
            ] == config
        )
        &
        (
            df_pareado[
                "metric"
            ] == metrica
        )
    ]


    if linha.empty:

        continue


    row = linha.iloc[0]


    print(

        f"{config:<19} "

        f"{metrica:<24} "

        f"Δ="
        f"{row['mean_difference']:+.5f} | "

        f"CI95=["
        f"{row['difference_ci95_low']:+.5f}, "
        f"{row['difference_ci95_high']:+.5f}] | "

        f"excludes zero="
        f"{row['ci95_excludes_zero']}"
    )


# ============================================================
# DISTÂNCIA PARA PROXIMITY_HIGH
# ============================================================

linha_dist = df_pareado[
    (
        df_pareado[
            "configuration"
        ] == "proximity_high"
    )
    &
    (
        df_pareado[
            "metric"
        ] == "avg_distance"
    )
]


if not linha_dist.empty:

    row = linha_dist.iloc[0]


    print(

        f"{'proximity_high':<19} "

        f"{'avg_distance':<24} "

        f"Δ="
        f"{row['mean_difference']:+.2f} | "

        f"CI95=["
        f"{row['difference_ci95_low']:+.2f}, "
        f"{row['difference_ci95_high']:+.2f}] | "

        f"excludes zero="
        f"{row['ci95_excludes_zero']}"
    )


# ============================================================
# ABLATION
# ============================================================

print("\n")

print(
    "COMPONENT ABLATION - PAIRED SUMMARY"
)

print("=" * 145)


ablations = {

    "no_reputation":
        "avg_reputation_norm",

    "no_energy":
        "avg_energy_norm",

    "no_proximity":
        "avg_proximity_norm"
}


for config, metrica in ablations.items():

    linha = df_pareado[
        (
            df_pareado[
                "configuration"
            ] == config
        )
        &
        (
            df_pareado[
                "metric"
            ] == metrica
        )
    ]


    if linha.empty:

        continue


    row = linha.iloc[0]


    print(

        f"{config:<19} "

        f"{metrica:<24} "

        f"Δ="
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
# ELIGIBILITY / ALLOCATION CONTROL
# ============================================================

print("\n")

print(
    "ELIGIBILITY / ALLOCATION CONTROL"
)

print("=" * 145)


for _, row in df_agregado.iterrows():

    print(

        f"{row['configuration']:<19} "

        f"eligibility="
        f"{row['overall_eligibility_rate_mean'] * 100:.2f}% | "

        f"allocation="
        f"{row['request_allocation_rate_mean'] * 100:.2f}% | "

        f"exploration="
        f"{row['exploration_rate_mean'] * 100:.2f}% | "

        f"cooldown="
        f"{row['cooldown_penalty_rate_mean'] * 100:.2f}%"
    )


# ============================================================
# SANITY CHECK AUTOMÁTICO
# ============================================================

print("\n")

print(
    "AUTOMATIC MULTI-SEED SANITY CHECK"
)

print("=" * 145)


sanity_checks = [

    (
        "reputation_high",
        "avg_reputation_norm",
        "positive",
        "Higher w1 increases selected reputation"
    ),

    (
        "energy_high",
        "avg_energy_norm",
        "positive",
        "Higher w2 increases selected energy"
    ),

    (
        "proximity_high",
        "avg_proximity_norm",
        "positive",
        "Higher w3 increases selected proximity"
    ),

    (
        "proximity_high",
        "avg_distance",
        "negative",
        "Higher w3 reduces selected distance"
    ),

    (
        "no_reputation",
        "avg_reputation_norm",
        "negative",
        "Removing reputation reduces selected reputation"
    ),

    (
        "no_energy",
        "avg_energy_norm",
        "negative",
        "Removing energy reduces selected energy"
    ),

    (
        "no_proximity",
        "avg_proximity_norm",
        "negative",
        "Removing proximity reduces selected proximity"
    )
]


for (
    config,
    metrica,
    direcao,
    descricao
) in sanity_checks:

    linha = df_pareado[
        (
            df_pareado[
                "configuration"
            ] == config
        )
        &
        (
            df_pareado[
                "metric"
            ] == metrica
        )
    ]


    if linha.empty:

        print(
            f"[MISSING] {descricao}"
        )

        continue


    row = linha.iloc[0]


    diff = row[
        "mean_difference"
    ]


    if direcao == "positive":

        direcao_ok = (
            diff > 0
        )

    else:

        direcao_ok = (
            diff < 0
        )


    estatisticamente_estavel = (
        bool(
            row[
                "ci95_excludes_zero"
            ]
        )
    )


    if (
        direcao_ok
        and
        estatisticamente_estavel
    ):

        status = "PASS"

    elif direcao_ok:

        status = "DIRECTION-ONLY"

    else:

        status = "CHECK"


    print(

        f"[{status}] "

        f"{descricao} | "

        f"Δ="
        f"{diff:+.5f} | "

        f"CI95=["
        f"{row['difference_ci95_low']:+.5f}, "
        f"{row['difference_ci95_high']:+.5f}]"
    )


# ============================================================
# OUTPUTS
# ============================================================

print("\n")

print(
    "OUTPUT FILES"
)

print("=" * 145)


print(
    f"Per seed     : "
    f"{ARQUIVO_POR_SEED}"
)


print(
    f"Aggregated   : "
    f"{ARQUIVO_AGREGADO}"
)


print(
    f"Paired       : "
    f"{ARQUIVO_PAREADO}"
)


print("\n")

print(
    "IMPORTANT:"
)

print(
    "The seed/run is the replication unit."
)

print(
    "Do not treat individual requests or selected vehicles "
    "as independent experimental replicates."
)

print(
    "Average score values should be interpreted cautiously "
    "because changing w1/w2/w3 changes the score definition."
)

print(
    "The normalized selected components and their paired "
    "differences are the primary evidence for weight sensitivity."
)


print("\n")
print("=" * 145)

print(
    "MULTI-SEED WEIGHT ANALYSIS COMPLETED"
)

print("=" * 145)
print()