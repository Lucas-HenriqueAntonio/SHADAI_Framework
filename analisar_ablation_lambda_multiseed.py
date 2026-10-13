# analisar_ablation_lambda_multiseed.py

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

LAMBDA_DIR = (
    "lambda_ablation"
)


os.makedirs(
    LAMBDA_DIR,
    exist_ok=True
)


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


PESOS = {

    "original":
        (0.5, 0.3, 0.2),

    "equal":
        (0.3333, 0.3333, 0.3334),

    "energy_high":
        (0.7, 0.2, 0.1),

    "distance_high":
        (0.2, 0.7, 0.1),

    "time_high":
        (0.2, 0.1, 0.7),

    "no_energy":
        (0.0, 0.6, 0.4),

    "no_distance":
        (0.7143, 0.0, 0.2857),

    "no_time":
        (0.625, 0.375, 0.0)
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
    LAMBDA_DIR,
    "resumo_ablation_lambda_por_seed.csv"
)


ARQUIVO_AGREGADO = os.path.join(
    LAMBDA_DIR,
    "resumo_ablation_lambda_multiseed.csv"
)


ARQUIVO_PAREADO = os.path.join(
    LAMBDA_DIR,
    "comparacao_pareada_lambda_vs_original.csv"
)


# ============================================================
# MÉTRICAS
# ============================================================

METRICAS_PRINCIPAIS = [

    "request_allocation_rate",

    "overall_eligibility_rate",

    "mean_eligibility_rate",

    "utility_rejection_rate",

    "energy_rejection_rate",

    "avg_selected_utility",

    "avg_score",

    "avg_energy",

    "avg_reputation",

    "allocations",

    "unique_vehicles"
]


# ============================================================
# FUNÇÕES
# ============================================================

def safe_div(
    numerador,
    denominador
):

    if denominador == 0:
        return float("nan")

    return (
        numerador
        / denominador
    )


def percentual_diferenca(
    valor,
    referencia
):

    if (
        pd.isna(referencia)
        or referencia == 0
    ):
        return float("nan")

    return (
        (valor - referencia)
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


    media = (
        serie.mean()
    )


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
        / math.sqrt(n)
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


# ============================================================
# LEITURA
# ============================================================

resultados = []

faltantes = []


print("\n")
print("=" * 120)

print(
    "SHADAI - MULTI-SEED LAMBDA ANALYSIS"
)

print("=" * 120)

print(
    f"Input/output directory: "
    f"{LAMBDA_DIR}"
)


for configuracao in CONFIGURACOES:

    lambda1, lambda2, lambda3 = (
        PESOS[
            configuracao
        ]
    )


    for seed in SEEDS:

        experiment_id = (
            f"lambda_"
            f"{configuracao}_"
            f"seed_"
            f"{seed:02d}"
        )


        arquivo_metricas = os.path.join(
            LAMBDA_DIR,
            (
                f"metricas_shadai_400_"
                f"{experiment_id}.csv"
            )
        )


        arquivo_elig = os.path.join(
            LAMBDA_DIR,
            (
                f"eligibilidade_shadai_400_"
                f"{experiment_id}.csv"
            )
        )


        if (
            not os.path.exists(
                arquivo_metricas
            )
            or
            not os.path.exists(
                arquivo_elig
            )
        ):

            faltantes.append(
                (
                    configuracao,
                    seed
                )
            )

            continue


        df_metricas = pd.read_csv(
            arquivo_metricas
        )


        df_elig = pd.read_csv(
            arquivo_elig
        )


        if df_elig.empty:

            faltantes.append(
                (
                    configuracao,
                    seed
                )
            )

            continue


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


        total_positive_utility = (
            df_elig[
                "positive_utility"
            ].sum()
        )


        total_rejected_utility = (
            df_elig[
                "rejected_utility"
            ].sum()
        )


        total_rejected_energy = (
            df_elig[
                "rejected_energy"
            ].sum()
        )


        overall_eligibility_rate = safe_div(
            total_eligible,
            vehicle_evaluations
        )


        utility_rejection_rate = safe_div(
            total_rejected_utility,
            vehicle_evaluations
        )


        energy_rejection_rate = safe_div(
            total_rejected_energy,
            vehicle_evaluations
        )


        positive_utility_rate = safe_div(
            total_positive_utility,
            vehicle_evaluations
        )


        mean_eligibility_rate = (
            df_elig[
                "eligibility_rate"
            ].mean()
        )


        if vehicle_evaluations > 0:

            avg_candidate_utility = (

                (
                    df_elig[
                        "avg_utility"
                    ]
                    *
                    df_elig[
                        "active_vehicles"
                    ]
                ).sum()

                / vehicle_evaluations
            )

        else:

            avg_candidate_utility = (
                float("nan")
            )


        allocations = len(
            df_metricas
        )


        request_allocation_rate = safe_div(
            allocations,
            requests
        )


        if not df_metricas.empty:

            avg_selected_utility = (
                df_metricas[
                    "utilidade"
                ].mean()
            )

            avg_score = (
                df_metricas[
                    "score"
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

            unique_vehicles = (
                df_metricas[
                    "veiculo"
                ].nunique()
            )

        else:

            avg_selected_utility = float("nan")
            avg_score = float("nan")
            avg_energy = float("nan")
            avg_reputation = float("nan")
            unique_vehicles = 0


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
                requests,

            "vehicle_evaluations":
                vehicle_evaluations,

            "allocations":
                allocations,

            "request_allocation_rate":
                request_allocation_rate,

            "unique_vehicles":
                unique_vehicles,

            "mean_eligibility_rate":
                mean_eligibility_rate,

            "overall_eligibility_rate":
                overall_eligibility_rate,

            "positive_utility_rate":
                positive_utility_rate,

            "utility_rejection_rate":
                utility_rejection_rate,

            "energy_rejection_rate":
                energy_rejection_rate,

            "avg_candidate_utility":
                avg_candidate_utility,

            "avg_selected_utility":
                avg_selected_utility,

            "avg_score":
                avg_score,

            "avg_energy":
                avg_energy,

            "avg_reputation":
                avg_reputation
        })


# ============================================================
# DATAFRAME POR SEED
# ============================================================

df_seed = pd.DataFrame(
    resultados
)


if df_seed.empty:

    raise RuntimeError(
        "No valid lambda experiment files found."
    )


df_seed.to_csv(
    ARQUIVO_POR_SEED,
    index=False
)


# ============================================================
# COMPLETUDE
# ============================================================

print("\nDATASET COMPLETENESS")
print("-" * 120)


for config in CONFIGURACOES:

    n = len(
        df_seed[
            df_seed[
                "configuration"
            ] == config
        ]
    )

    print(
        f"{config:<16}: "
        f"{n}/30"
    )


if faltantes:

    print(
        f"\nWARNING: "
        f"{len(faltantes)} runs missing."
    )

else:

    print(
        "\nAll 240 runs found."
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
            len(dados),

        "lambda1":
            PESOS[config][0],

        "lambda2":
            PESOS[config][1],

        "lambda3":
            PESOS[config][2]
    }


    for metrica in METRICAS_PRINCIPAIS:

        (
            media,
            sd,
            ci_low,
            ci_high
        ) = calcular_ic95(
            dados[metrica]
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
# DIFERENÇAS VS ORIGINAL
# ============================================================

original = df_agregado[
    df_agregado[
        "configuration"
    ] == "original"
].iloc[0]


for metrica in METRICAS_PRINCIPAIS:

    mean_col = (
        f"{metrica}_mean"
    )


    diff_col = (
        f"{metrica}_diff_pct_vs_original"
    )


    df_agregado[
        diff_col
    ] = df_agregado[
        mean_col
    ].apply(

        lambda x:
            percentual_diferenca(
                x,
                original[
                    mean_col
                ]
            )
    )


df_agregado.to_csv(
    ARQUIVO_AGREGADO,
    index=False
)


# ============================================================
# COMPARAÇÃO PAREADA
# ============================================================

df_original = df_seed[
    df_seed[
        "configuration"
    ] == "original"
].copy()


pareados_resultados = []


for config in CONFIGURACOES:

    if config == "original":
        continue


    df_config = df_seed[
        df_seed[
            "configuration"
        ] == config
    ].copy()


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

        original_col = (
            f"{metrica}_original"
        )

        alternative_col = (
            f"{metrica}_alternative"
        )


        diferencas = (
            pareado[
                alternative_col
            ]
            -
            pareado[
                original_col
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


        original_mean = (
            pareado[
                original_col
            ].mean()
        )


        alternative_mean = (
            pareado[
                alternative_col
            ].mean()
        )


        pct_diff = percentual_diferenca(
            alternative_mean,
            original_mean
        )


        ci_excludes_zero = (
            (
                ci_low > 0
                or
                ci_high < 0
            )
            if not pd.isna(ci_low)
            else False
        )


        pareados_resultados.append({

            "configuration":
                config,

            "metric":
                metrica,

            "n_pairs":
                len(pareado),

            "original_mean":
                original_mean,

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
                pct_diff,

            "ci95_excludes_zero":
                ci_excludes_zero
        })


df_pareado = pd.DataFrame(
    pareados_resultados
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
    250
)


print("\n")
print("=" * 120)

print(
    "MAIN MULTI-SEED RESULTS"
)

print("=" * 120)


for _, row in df_agregado.iterrows():

    print(
        f"{row['configuration']:<16} | "
        f"N={int(row['n_runs']):02d} | "
        f"Allocation="
        f"{row['request_allocation_rate_mean'] * 100:.2f}% "
        f"[{row['request_allocation_rate_ci95_low'] * 100:.2f}, "
        f"{row['request_allocation_rate_ci95_high'] * 100:.2f}] | "
        f"Eligibility="
        f"{row['overall_eligibility_rate_mean'] * 100:.2f}% "
        f"[{row['overall_eligibility_rate_ci95_low'] * 100:.2f}, "
        f"{row['overall_eligibility_rate_ci95_high'] * 100:.2f}] | "
        f"Score="
        f"{row['avg_score_mean']:.4f} | "
        f"Energy="
        f"{row['avg_energy_mean']:.2f}"
    )


# ============================================================
# PAREADO
# ============================================================

print("\n")
print("=" * 120)

print(
    "PAIRED COMPARISONS AGAINST ORIGINAL"
)

print("=" * 120)


metricas_console = [
    "request_allocation_rate",
    "overall_eligibility_rate",
    "avg_score",
    "avg_energy"
]


for config in CONFIGURACOES:

    if config == "original":
        continue


    print(
        f"\n{config}:"
    )


    subset = df_pareado[
        (
            df_pareado[
                "configuration"
            ] == config
        )
        &
        (
            df_pareado[
                "metric"
            ].isin(
                metricas_console
            )
        )
    ]


    for _, row in subset.iterrows():

        print(
            f"  {row['metric']:<27} "
            f"diff={row['mean_difference']:+.5f} | "
            f"CI95=["
            f"{row['difference_ci95_low']:+.5f}, "
            f"{row['difference_ci95_high']:+.5f}] | "
            f"Δ={row['difference_pct']:+.2f}% | "
            f"excludes zero="
            f"{row['ci95_excludes_zero']}"
        )


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 120)

print(
    "OUTPUT FILES"
)

print("=" * 120)

print(
    ARQUIVO_POR_SEED
)

print(
    ARQUIVO_AGREGADO
)

print(
    ARQUIVO_PAREADO
)

print("\nANALYSIS COMPLETED\n")