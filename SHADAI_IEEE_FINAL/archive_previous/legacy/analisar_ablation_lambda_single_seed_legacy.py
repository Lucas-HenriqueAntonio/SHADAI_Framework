# analisar_ablation_lambda.py

import os
import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

CONFIGURACOES = [
    "original",
    "equal",
    "energy_high",
    "distance_high",
    "time_high",
    "no_energy",
    "no_distance",
    "no_time"
]


PREFIXO_METRICAS = (
    "metricas_shadai_400_lambda_"
)

PREFIXO_ELEGIBILIDADE = (
    "eligibilidade_shadai_400_lambda_"
)


ARQUIVO_SAIDA = (
    "resumo_ablation_lambda_normalized.csv"
)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def percentual_diferenca(
    valor,
    referencia
):

    if referencia == 0:

        return float("nan")


    return (
        (valor - referencia)
        / referencia
    ) * 100.0


def classificar_variacao(
    valor
):

    if pd.isna(
        valor
    ):

        return "N/A"


    valor_abs = abs(
        valor
    )


    if valor_abs < 1.0:

        return "VERY SMALL"

    elif valor_abs < 3.0:

        return "SMALL"

    elif valor_abs < 5.0:

        return "MODERATE"

    else:

        return "LARGE"


def imprimir_linha(
    tamanho=145
):

    print(
        "=" * tamanho
    )


# ============================================================
# CABEÇALHO
# ============================================================

print("\n")

imprimir_linha()

print(
    "SHADAI - NORMALIZED LAMBDA ABLATION AND SENSITIVITY ANALYSIS"
)

imprimir_linha()


# ============================================================
# RESULTADOS
# ============================================================

resultados = []


# ============================================================
# LEITURA DAS CONFIGURAÇÕES
# ============================================================

for configuracao in CONFIGURACOES:

    arquivo_metricas = (
        f"{PREFIXO_METRICAS}"
        f"{configuracao}.csv"
    )


    arquivo_elegibilidade = (
        f"{PREFIXO_ELEGIBILIDADE}"
        f"{configuracao}.csv"
    )


    print(
        f"\nLoading configuration: "
        f"{configuracao}"
    )


    print(
        f"  Metrics     : "
        f"{arquivo_metricas}"
    )


    print(
        f"  Eligibility : "
        f"{arquivo_elegibilidade}"
    )


    # ========================================================
    # VERIFICA ARQUIVOS
    # ========================================================

    if not os.path.exists(
        arquivo_metricas
    ):

        print(
            f"[WARNING] Metrics file not found: "
            f"{arquivo_metricas}"
        )

        continue


    if not os.path.exists(
        arquivo_elegibilidade
    ):

        print(
            f"[WARNING] Eligibility file not found: "
            f"{arquivo_elegibilidade}"
        )

        continue


    # ========================================================
    # LEITURA
    # ========================================================

    df_metricas = pd.read_csv(
        arquivo_metricas
    )


    df_elegibilidade = pd.read_csv(
        arquivo_elegibilidade
    )


    if df_elegibilidade.empty:

        print(
            f"[WARNING] Eligibility file is empty: "
            f"{arquivo_elegibilidade}"
        )

        continue


    # ========================================================
    # PARÂMETROS
    # ========================================================

    if not df_metricas.empty:

        lambda1 = float(
            df_metricas[
                "lambda1"
            ].iloc[0]
        )


        lambda2 = float(
            df_metricas[
                "lambda2"
            ].iloc[0]
        )


        lambda3 = float(
            df_metricas[
                "lambda3"
            ].iloc[0]
        )


        seed = (

            int(
                df_metricas[
                    "seed"
                ].iloc[0]
            )

            if "seed"
            in df_metricas.columns

            else None
        )


    else:

        # ====================================================
        # CASO EXTREMO:
        # nenhuma alocação ocorreu.
        #
        # Ainda conseguimos analisar eligibility.
        # Os lambdas são inferidos pelo nome.
        # ====================================================

        pesos_fallback = {

            "original":
                (0.5, 0.3, 0.2),

            "equal":
                (
                    0.3333,
                    0.3333,
                    0.3334
                ),

            "energy_high":
                (0.7, 0.2, 0.1),

            "distance_high":
                (0.2, 0.7, 0.1),

            "time_high":
                (0.2, 0.1, 0.7),

            "no_energy":
                (0.0, 0.6, 0.4),

            "no_distance":
                (
                    0.7143,
                    0.0,
                    0.2857
                ),

            "no_time":
                (
                    0.625,
                    0.375,
                    0.0
                )
        }


        (
            lambda1,
            lambda2,
            lambda3
        ) = pesos_fallback[
            configuracao
        ]


        seed = (

            int(
                df_elegibilidade[
                    "seed"
                ].iloc[0]
            )

            if "seed"
            in df_elegibilidade.columns

            else None
        )


    # ========================================================
    # MÉTRICAS DAS ALOCAÇÕES
    # ========================================================

    allocations = len(
        df_metricas
    )


    if not df_metricas.empty:

        avg_utility_selected = (
            df_metricas[
                "utilidade"
            ].mean()
        )


        std_utility_selected = (
            df_metricas[
                "utilidade"
            ].std()
        )


        min_utility_selected = (
            df_metricas[
                "utilidade"
            ].min()
        )


        max_utility_selected = (
            df_metricas[
                "utilidade"
            ].max()
        )


        avg_score = (
            df_metricas[
                "score"
            ].mean()
        )


        std_score = (
            df_metricas[
                "score"
            ].std()
        )


        avg_energy = (
            df_metricas[
                "energia"
            ].mean()
        )


        std_energy = (
            df_metricas[
                "energia"
            ].std()
        )


        avg_reputation = (
            df_metricas[
                "reputacao"
            ].mean()
        )


        std_reputation = (
            df_metricas[
                "reputacao"
            ].std()
        )


        unique_vehicles = (
            df_metricas[
                "veiculo"
            ].nunique()
        )


    else:

        avg_utility_selected = float(
            "nan"
        )

        std_utility_selected = float(
            "nan"
        )

        min_utility_selected = float(
            "nan"
        )

        max_utility_selected = float(
            "nan"
        )

        avg_score = float(
            "nan"
        )

        std_score = float(
            "nan"
        )

        avg_energy = float(
            "nan"
        )

        std_energy = float(
            "nan"
        )

        avg_reputation = float(
            "nan"
        )

        std_reputation = float(
            "nan"
        )

        unique_vehicles = 0


    # ========================================================
    # MÉTRICAS DE ELEGIBILIDADE
    # ========================================================

    total_requests = len(
        df_elegibilidade
    )


    total_vehicle_evaluations = (
        df_elegibilidade[
            "active_vehicles"
        ].sum()
    )


    total_positive_utility = (
        df_elegibilidade[
            "positive_utility"
        ].sum()
    )


    total_sufficient_energy = (
        df_elegibilidade[
            "sufficient_energy"
        ].sum()
    )


    total_eligible = (
        df_elegibilidade[
            "eligible"
        ].sum()
    )


    total_rejected_utility = (
        df_elegibilidade[
            "rejected_utility"
        ].sum()
    )


    total_rejected_energy = (
        df_elegibilidade[
            "rejected_energy"
        ].sum()
    )


    # ========================================================
    # TAXAS
    # ========================================================

    mean_eligibility_rate = (
        df_elegibilidade[
            "eligibility_rate"
        ].mean()
    )


    overall_eligibility_rate = (

        total_eligible
        / total_vehicle_evaluations

        if total_vehicle_evaluations > 0

        else float("nan")
    )


    utility_rejection_rate = (

        total_rejected_utility
        / total_vehicle_evaluations

        if total_vehicle_evaluations > 0

        else float("nan")
    )


    energy_rejection_rate = (

        total_rejected_energy
        / total_vehicle_evaluations

        if total_vehicle_evaluations > 0

        else float("nan")
    )


    positive_utility_rate = (

        total_positive_utility
        / total_vehicle_evaluations

        if total_vehicle_evaluations > 0

        else float("nan")
    )


    sufficient_energy_rate = (

        total_sufficient_energy
        / total_vehicle_evaluations

        if total_vehicle_evaluations > 0

        else float("nan")
    )


    request_allocation_rate = (

        allocations
        / total_requests

        if total_requests > 0

        else float("nan")
    )


    allocations_per_vehicle = (

        allocations
        / unique_vehicles

        if unique_vehicles > 0

        else float("nan")
    )


    # ========================================================
    # UTILITY DE TODOS OS VEÍCULOS AVALIADOS
    # ========================================================

    avg_candidate_utility = (
        df_elegibilidade[
            "avg_utility"
        ].mean()
    )


    min_candidate_utility = (
        df_elegibilidade[
            "min_utility"
        ].min()
    )


    max_candidate_utility = (
        df_elegibilidade[
            "max_utility"
        ].max()
    )


    # ========================================================
    # SALVA RESULTADO
    # ========================================================

    resultados.append({

        "configuration":
            configuracao,

        "seed":
            seed,

        "lambda1":
            lambda1,

        "lambda2":
            lambda2,

        "lambda3":
            lambda3,

        "requests":
            total_requests,

        "vehicle_evaluations":
            total_vehicle_evaluations,

        "allocations":
            allocations,

        "request_allocation_rate":
            request_allocation_rate,

        "unique_vehicles":
            unique_vehicles,

        "allocations_per_vehicle":
            allocations_per_vehicle,

        "mean_eligibility_rate":
            mean_eligibility_rate,

        "overall_eligibility_rate":
            overall_eligibility_rate,

        "positive_utility_rate":
            positive_utility_rate,

        "utility_rejection_rate":
            utility_rejection_rate,

        "sufficient_energy_rate":
            sufficient_energy_rate,

        "energy_rejection_rate":
            energy_rejection_rate,

        "avg_candidate_utility":
            avg_candidate_utility,

        "min_candidate_utility":
            min_candidate_utility,

        "max_candidate_utility":
            max_candidate_utility,

        "avg_selected_utility":
            avg_utility_selected,

        "std_selected_utility":
            std_utility_selected,

        "min_selected_utility":
            min_utility_selected,

        "max_selected_utility":
            max_utility_selected,

        "avg_score":
            avg_score,

        "std_score":
            std_score,

        "avg_energy":
            avg_energy,

        "std_energy":
            std_energy,

        "avg_reputation":
            avg_reputation,

        "std_reputation":
            std_reputation
    })


# ============================================================
# DATAFRAME FINAL
# ============================================================

resumo = pd.DataFrame(
    resultados
)


if resumo.empty:

    raise RuntimeError(
        "No valid lambda experiments were found."
    )


# ============================================================
# CONFIGURAÇÃO ORIGINAL
# ============================================================

original_df = resumo[
    resumo[
        "configuration"
    ]
    == "original"
]


if original_df.empty:

    raise RuntimeError(
        "Original configuration was not found."
    )


original = original_df.iloc[
    0
]


# ============================================================
# DIFERENÇAS PERCENTUAIS VS ORIGINAL
# ============================================================

metricas_comparativas = [

    (
        "avg_selected_utility",
        "selected_utility_diff_pct"
    ),

    (
        "avg_score",
        "score_diff_pct"
    ),

    (
        "avg_energy",
        "energy_diff_pct"
    ),

    (
        "avg_reputation",
        "reputation_diff_pct"
    ),

    (
        "allocations",
        "allocation_diff_pct"
    ),

    (
        "unique_vehicles",
        "vehicle_diff_pct"
    ),

    (
        "mean_eligibility_rate",
        "mean_eligibility_diff_pct"
    ),

    (
        "overall_eligibility_rate",
        "overall_eligibility_diff_pct"
    ),

    (
        "utility_rejection_rate",
        "utility_rejection_diff_pct"
    ),

    (
        "request_allocation_rate",
        "request_allocation_diff_pct"
    )
]


for (
    coluna,
    nova_coluna
) in metricas_comparativas:

    resumo[
        nova_coluna
    ] = resumo[
        coluna
    ].apply(

        lambda x:
            percentual_diferenca(
                x,
                original[
                    coluna
                ]
            )
    )


# ============================================================
# CLASSIFICAÇÕES
# ============================================================

resumo[
    "eligibility_change"
] = resumo[
    "overall_eligibility_diff_pct"
].apply(
    classificar_variacao
)


resumo[
    "allocation_change"
] = resumo[
    "request_allocation_diff_pct"
].apply(
    classificar_variacao
)


resumo[
    "utility_change"
] = resumo[
    "selected_utility_diff_pct"
].apply(
    classificar_variacao
)


# ============================================================
# ORDENAÇÃO
# ============================================================

ordem = {

    nome: indice

    for indice, nome
    in enumerate(
        CONFIGURACOES
    )
}


resumo[
    "_ordem"
] = resumo[
    "configuration"
].map(
    ordem
)


resumo = (

    resumo

    .sort_values(
        "_ordem"
    )

    .drop(
        columns=[
            "_ordem"
        ]
    )
)


# ============================================================
# SALVA CSV
# ============================================================

resumo.to_csv(
    ARQUIVO_SAIDA,
    index=False
)


# ============================================================
# CONFIGURAÇÃO DE EXIBIÇÃO
# ============================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    260
)

pd.set_option(
    "display.precision",
    4
)


# ============================================================
# VISÃO GERAL
# ============================================================

print("\n")

print(
    "EXPERIMENT OVERVIEW"
)

imprimir_linha()


print(
    f"Configurations analyzed : "
    f"{len(resumo)}"
)


print(
    f"Reference configuration : "
    f"original "
    f"({original['lambda1']:.4f}, "
    f"{original['lambda2']:.4f}, "
    f"{original['lambda3']:.4f})"
)


print(
    f"Output CSV              : "
    f"{ARQUIVO_SAIDA}"
)


# ============================================================
# TABELA PRINCIPAL
# ============================================================

print("\n")

print(
    "MAIN COMPARISON"
)

imprimir_linha()


cabecalho = (

    f"{'CONFIGURATION':<18}"

    f"{'L1':>7}"

    f"{'L2':>7}"

    f"{'L3':>7}"

    f"{'REQ':>7}"

    f"{'ALLOC':>8}"

    f"{'ALLOC%':>9}"

    f"{'ELIG%':>9}"

    f"{'REJ-U%':>9}"

    f"{'UTILITY':>11}"

    f"{'SCORE':>10}"

    f"{'ENERGY':>11}"

    f"{'VEH':>7}"
)


print(
    cabecalho
)

print(
    "-" * 145
)


for _, row in resumo.iterrows():

    print(

        f"{row['configuration']:<18}"

        f"{row['lambda1']:>7.4f}"

        f"{row['lambda2']:>7.4f}"

        f"{row['lambda3']:>7.4f}"

        f"{int(row['requests']):>7}"

        f"{int(row['allocations']):>8}"

        f"{row['request_allocation_rate'] * 100:>8.2f}%"

        f"{row['overall_eligibility_rate'] * 100:>8.2f}%"

        f"{row['utility_rejection_rate'] * 100:>8.2f}%"

        f"{row['avg_selected_utility']:>11.4f}"

        f"{row['avg_score']:>10.4f}"

        f"{row['avg_energy']:>11.4f}"

        f"{int(row['unique_vehicles']):>7}"
    )


# ============================================================
# DIFERENÇAS VS ORIGINAL
# ============================================================

print("\n")

print(
    "DIFFERENCES RELATIVE TO ORIGINAL"
)

imprimir_linha()


print(

    resumo[[
        "configuration",

        "selected_utility_diff_pct",

        "score_diff_pct",

        "energy_diff_pct",

        "mean_eligibility_diff_pct",

        "overall_eligibility_diff_pct",

        "request_allocation_diff_pct",

        "allocation_diff_pct",

        "vehicle_diff_pct"
    ]]

    .to_string(
        index=False
    )
)


# ============================================================
# ELEGIBILIDADE
# ============================================================

print("\n")

print(
    "ELIGIBILITY ANALYSIS"
)

imprimir_linha()


print(

    resumo[[
        "configuration",

        "vehicle_evaluations",

        "positive_utility_rate",

        "utility_rejection_rate",

        "sufficient_energy_rate",

        "energy_rejection_rate",

        "mean_eligibility_rate",

        "overall_eligibility_rate"
    ]]

    .to_string(
        index=False
    )
)


# ============================================================
# ALOCAÇÃO
# ============================================================

print("\n")

print(
    "ALLOCATION ANALYSIS"
)

imprimir_linha()


print(

    resumo[[
        "configuration",

        "requests",

        "allocations",

        "request_allocation_rate",

        "unique_vehicles",

        "allocations_per_vehicle"
    ]]

    .to_string(
        index=False
    )
)


# ============================================================
# UTILITY
# ============================================================

print("\n")

print(
    "UTILITY ANALYSIS"
)

imprimir_linha()


print(

    resumo[[
        "configuration",

        "avg_candidate_utility",

        "min_candidate_utility",

        "max_candidate_utility",

        "avg_selected_utility",

        "std_selected_utility",

        "min_selected_utility",

        "max_selected_utility"
    ]]

    .to_string(
        index=False
    )
)


# ============================================================
# SCORE / ENERGY / REPUTATION
# ============================================================

print("\n")

print(
    "SELECTED VEHICLE METRICS"
)

imprimir_linha()


print(

    resumo[[
        "configuration",

        "avg_score",

        "std_score",

        "avg_energy",

        "std_energy",

        "avg_reputation",

        "std_reputation"
    ]]

    .to_string(
        index=False
    )
)


# ============================================================
# RANKINGS
# ============================================================

print("\n")

print(
    "RANKING - OVERALL ELIGIBILITY RATE"
)

imprimir_linha()


print(

    resumo[[
        "configuration",

        "overall_eligibility_rate",

        "overall_eligibility_diff_pct",

        "eligibility_change"
    ]]

    .sort_values(
        "overall_eligibility_rate",
        ascending=False
    )

    .to_string(
        index=False
    )
)


print("\n")

print(
    "RANKING - REQUEST ALLOCATION RATE"
)

imprimir_linha()


print(

    resumo[[
        "configuration",

        "request_allocation_rate",

        "request_allocation_diff_pct",

        "allocation_change"
    ]]

    .sort_values(
        "request_allocation_rate",
        ascending=False
    )

    .to_string(
        index=False
    )
)


print("\n")

print(
    "RANKING - SELECTED UTILITY"
)

imprimir_linha()


print(

    resumo[[
        "configuration",

        "avg_selected_utility",

        "selected_utility_diff_pct",

        "utility_change"
    ]]

    .sort_values(
        "avg_selected_utility",
        ascending=False
    )

    .to_string(
        index=False
    )
)


# ============================================================
# MELHORES CONFIGURAÇÕES
# ============================================================

best_eligibility = resumo.loc[
    resumo[
        "overall_eligibility_rate"
    ].idxmax()
]


best_allocation = resumo.loc[
    resumo[
        "request_allocation_rate"
    ].idxmax()
]


best_utility = resumo.loc[
    resumo[
        "avg_selected_utility"
    ].idxmax()
]


best_score = resumo.loc[
    resumo[
        "avg_score"
    ].idxmax()
]


best_energy = resumo.loc[
    resumo[
        "avg_energy"
    ].idxmax()
]


best_diversity = resumo.loc[
    resumo[
        "unique_vehicles"
    ].idxmax()
]


print("\n")

print(
    "BEST OBSERVED RESULTS"
)

imprimir_linha()


print(
    f"Highest eligibility rate : "
    f"{best_eligibility['configuration']} "
    f"({best_eligibility['overall_eligibility_rate'] * 100:.2f}%)"
)


print(
    f"Highest allocation rate  : "
    f"{best_allocation['configuration']} "
    f"({best_allocation['request_allocation_rate'] * 100:.2f}%)"
)


print(
    f"Highest selected utility : "
    f"{best_utility['configuration']} "
    f"({best_utility['avg_selected_utility']:.4f})"
)


print(
    f"Highest average score    : "
    f"{best_score['configuration']} "
    f"({best_score['avg_score']:.4f})"
)


print(
    f"Highest residual energy  : "
    f"{best_energy['configuration']} "
    f"({best_energy['avg_energy']:.4f})"
)


print(
    f"Most unique vehicles     : "
    f"{best_diversity['configuration']} "
    f"({int(best_diversity['unique_vehicles'])})"
)


# ============================================================
# ABLATION DOS COMPONENTES
# ============================================================

print("\n")

print(
    "COMPONENT REMOVAL ANALYSIS"
)

imprimir_linha()


for config in [
    "no_energy",
    "no_distance",
    "no_time"
]:

    row = resumo[
        resumo[
            "configuration"
        ]
        == config
    ]


    if row.empty:

        continue


    row = row.iloc[
        0
    ]


    print(
        f"\n{config}:"
    )


    print(
        f"  Eligibility difference : "
        f"{row['overall_eligibility_diff_pct']:+.2f}%"
    )


    print(
        f"  Allocation difference  : "
        f"{row['request_allocation_diff_pct']:+.2f}%"
    )


    print(
        f"  Utility difference     : "
        f"{row['selected_utility_diff_pct']:+.2f}%"
    )


    print(
        f"  Score difference       : "
        f"{row['score_diff_pct']:+.2f}%"
    )


    print(
        f"  Energy difference      : "
        f"{row['energy_diff_pct']:+.2f}%"
    )


# ============================================================
# OBSERVAÇÕES AUTOMÁTICAS
# ============================================================

print("\n")

print(
    "MAIN OBSERVATIONS"
)

imprimir_linha()


alternativas = resumo[
    resumo[
        "configuration"
    ]
    != "original"
]


max_eligibility_change = (

    alternativas[
        "overall_eligibility_diff_pct"
    ]

    .abs()

    .max()
)


max_allocation_change = (

    alternativas[
        "request_allocation_diff_pct"
    ]

    .abs()

    .max()
)


max_utility_change = (

    alternativas[
        "selected_utility_diff_pct"
    ]

    .abs()

    .max()
)


print(
    f"Maximum absolute eligibility variation : "
    f"{max_eligibility_change:.2f}%"
)


print(
    f"Maximum absolute allocation variation  : "
    f"{max_allocation_change:.2f}%"
)


print(
    f"Maximum absolute utility variation     : "
    f"{max_utility_change:.2f}%"
)


# ============================================================
# INTERPRETAÇÃO EXPLORATÓRIA
# ============================================================

if max_eligibility_change < 1.0:

    print(
        "\n[ELIGIBILITY] "
        "Very low sensitivity to lambda variation."
    )

elif max_eligibility_change < 5.0:

    print(
        "\n[ELIGIBILITY] "
        "Low-to-moderate sensitivity to lambda variation."
    )

else:

    print(
        "\n[ELIGIBILITY] "
        "Lambda variation meaningfully affects "
        "the feasible candidate set."
    )


if max_allocation_change < 1.0:

    print(
        "[ALLOCATION] "
        "Allocation outcomes remain almost unchanged."
    )

elif max_allocation_change < 5.0:

    print(
        "[ALLOCATION] "
        "Lambda variation produces limited changes "
        "in allocation outcomes."
    )

else:

    print(
        "[ALLOCATION] "
        "Lambda variation materially affects "
        "request allocation success."
    )


# ============================================================
# CHECAGEM DA RESTRIÇÃO ENERGÉTICA
# ============================================================

max_energy_rejection = (
    resumo[
        "energy_rejection_rate"
    ].max()
)


if max_energy_rejection == 0:

    print(
        "\n[ENERGY WARNING] "
        "No candidate was rejected by the hard energy "
        "constraint in any configuration."
    )


# ============================================================
# ALERTA SOBRE SEED
# ============================================================

print("\n")

print(
    "IMPORTANT NOTES"
)

imprimir_linha()


print(
    "1. These results currently represent one seed only."
)


print(
    "2. Percentage-difference labels are exploratory."
)


print(
    "3. Statistical significance must not be inferred "
    "from this table."
)


print(
    "4. Multiple independent seeds and 95% confidence "
    "intervals will be required before final paper claims."
)


print(
    "5. The most informative ablation outcomes are changes "
    "in eligibility and allocation behavior, not only changes "
    "in the numerical utility value."
)


# ============================================================
# FINAL
# ============================================================

print("\n")

imprimir_linha()


print(
    f"FULL SUMMARY SAVED TO: "
    f"{ARQUIVO_SAIDA}"
)


imprimir_linha()

print("\n")