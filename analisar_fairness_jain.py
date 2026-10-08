# analisar_fairness_jain.py

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


RNG_SCHEME_REQUIRED = "split_v1"


# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_POR_SEED = os.path.join(
    EPSILON_DIR,
    "fairness_jain_por_seed.csv"
)


OUTPUT_AGREGADO = os.path.join(
    EPSILON_DIR,
    "fairness_jain_multiseed.csv"
)


OUTPUT_PAREADO = os.path.join(
    EPSILON_DIR,
    "fairness_jain_pareado_vs_epsilon020.csv"
)


OUTPUT_LATEX = os.path.join(
    EPSILON_DIR,
    "tabela_fairness_jain_epsilon_tnsm.tex"
)


# ============================================================
# FUNÇÕES
# ============================================================

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


    soma_quadrados = sum(
        x ** 2
        for x in valores
    )


    if soma_quadrados == 0:
        return float("nan")


    return (
        soma ** 2
        /
        (
            n
            *
            soma_quadrados
        )
    )


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
        valor == RNG_SCHEME_REQUIRED
        for valor in valores
    )


# ============================================================
# CABEÇALHO
# ============================================================

print("\n")

print("=" * 145)

print(
    "SHADAI - JAIN FAIRNESS INDEX ANALYSIS"
)

print("=" * 145)

print(
    f"Directory   : {EPSILON_DIR}"
)

print(
    f"Seeds       : {len(SEEDS)}"
)

print(
    f"RNG scheme  : {RNG_SCHEME_REQUIRED}"
)

print(
    f"Baseline    : {BASELINE} "
    f"(epsilon={CONFIGURACOES[BASELINE]:.2f})"
)

print("=" * 145)


# ============================================================
# RESULTADOS POR SEED
# ============================================================

resultados = []

faltantes = []

rng_invalidos = []


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


        if not os.path.exists(
            metrics_file
        ):

            faltantes.append(
                (
                    configuracao,
                    seed
                )
            )

            continue


        df = pd.read_csv(
            metrics_file
        )


        if df.empty:

            faltantes.append(
                (
                    configuracao,
                    seed
                )
            )

            continue


        if not validar_rng(
            df
        ):

            rng_invalidos.append(
                (
                    configuracao,
                    seed
                )
            )

            continue


        # ====================================================
        # CONTA QUANTAS ALOCAÇÕES CADA VEÍCULO RECEBEU
        # ====================================================

        contagens = (

            df[
                "veiculo"
            ]

            .value_counts()

            .sort_index()
        )


        valores = (
            contagens.values
        )


        # ====================================================
        # JAIN FAIRNESS
        # ====================================================

        jain = calcular_jain(
            valores
        )


        total_allocations = len(
            df
        )


        unique_vehicles = len(
            contagens
        )


        allocations_per_vehicle = (

            total_allocations
            /
            unique_vehicles

            if unique_vehicles > 0

            else float("nan")
        )


        # ====================================================
        # MAIOR PARTICIPAÇÃO INDIVIDUAL
        # ====================================================

        max_allocations = (

            contagens.max()

            if unique_vehicles > 0

            else 0
        )


        max_vehicle_share = (

            max_allocations
            /
            total_allocations

            if total_allocations > 0

            else float("nan")
        )


        # ====================================================
        # VARIAÇÃO DAS ALOCAÇÕES
        # ====================================================

        allocation_sd = (

            pd.Series(
                valores
            ).std(
                ddof=1
            )

            if len(
                valores
            ) > 1

            else 0.0
        )


        allocation_cv = (

            allocation_sd
            /
            allocations_per_vehicle

            if (
                allocations_per_vehicle
                and
                allocations_per_vehicle != 0
            )

            else float("nan")
        )


        resultados.append({

            "configuration":
                configuracao,

            "epsilon":
                epsilon,

            "seed":
                seed,

            "total_allocations":
                total_allocations,

            "unique_selected_vehicles":
                unique_vehicles,

            "allocations_per_vehicle":
                allocations_per_vehicle,

            "jain_fairness_index":
                jain,

            "max_allocations_single_vehicle":
                max_allocations,

            "max_vehicle_share":
                max_vehicle_share,

            "allocation_sd":
                allocation_sd,

            "allocation_cv":
                allocation_cv
        })


# ============================================================
# DATAFRAME POR SEED
# ============================================================

df_seed = pd.DataFrame(
    resultados
)


if df_seed.empty:

    raise RuntimeError(
        "Nenhum resultado válido foi encontrado."
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

print("-" * 145)


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

        f"{config:<14}: "

        f"{n:>2}/"
        f"{len(SEEDS)} runs"
    )


if faltantes:

    print(

        f"\n[WARNING] "
        f"{len(faltantes)} runs missing."
    )


if rng_invalidos:

    print(

        f"\n[WARNING] "
        f"{len(rng_invalidos)} runs with invalid RNG scheme."
    )


# ============================================================
# AGREGAÇÃO
# ============================================================

METRICAS = [

    "jain_fairness_index",

    "unique_selected_vehicles",

    "allocations_per_vehicle",

    "max_vehicle_share",

    "allocation_cv"
]


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
# COMPARAÇÃO PAREADA VS EPSILON=0.20
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


comparacoes = []


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
            mean_diff,
            sd_diff,
            ci_low,
            ci_high
        ) = calcular_ic95(
            diferencas
        )


        baseline_mean = (

            pareado[
                base_col
            ].mean()
        )


        alternative_mean = (

            pareado[
                alt_col
            ].mean()
        )


        diff_pct = percentual_diferenca(

            alternative_mean,

            baseline_mean
        )


        ci_excludes_zero = (

            (
                ci_low > 0
            )
            or
            (
                ci_high < 0
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
        )


        comparacoes.append({

            "configuration":
                configuracao,

            "epsilon":
                CONFIGURACOES[
                    configuracao
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
                diff_pct,

            "ci95_excludes_zero":
                ci_excludes_zero
        })


df_pareado = pd.DataFrame(
    comparacoes
)


df_pareado.to_csv(
    OUTPUT_PAREADO,
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


print("\n")

print(
    "MAIN JAIN FAIRNESS RESULTS"
)

print("=" * 145)


cabecalho = (

    f"{'CONFIG':<14}"

    f"{'EPS':>7}"

    f"{'N':>5}"

    f"{'JAIN':>11}"

    f"{'UNIQUE':>11}"

    f"{'ALLOC/V':>11}"

    f"{'MAX SHARE':>12}"

    f"{'CV':>11}"
)


print(
    cabecalho
)

print("-" * 145)


for _, row in df_agregado.sort_values(
    "epsilon"
).iterrows():

    print(

        f"{row['configuration']:<14}"

        f"{row['epsilon']:>7.2f}"

        f"{int(row['n_runs']):>5}"

        f"{row['jain_fairness_index_mean']:>11.4f}"

        f"{row['unique_selected_vehicles_mean']:>11.2f}"

        f"{row['allocations_per_vehicle_mean']:>11.3f}"

        f"{row['max_vehicle_share_mean'] * 100:>11.2f}%"

        f"{row['allocation_cv_mean']:>11.4f}"
    )


# ============================================================
# PAIRED JAIN VS BASELINE
# ============================================================

print("\n")

print(
    "PAIRED JAIN FAIRNESS COMPARISON VS EPSILON=0.20"
)

print("=" * 145)


jain_rows = df_pareado[
    df_pareado[
        "metric"
    ]
    ==
    "jain_fairness_index"
].sort_values(
    "epsilon"
)


for _, row in jain_rows.iterrows():

    print(

        f"epsilon="
        f"{row['epsilon']:.2f} | "

        f"ΔJain="
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
# TREND
# ============================================================

print("\n")

print(
    "FAIRNESS TREND DIAGNOSTICS"
)

print("=" * 145)


df_corr = df_agregado.sort_values(
    "epsilon"
)


cor_jain = (

    df_corr[
        [
            "epsilon",
            "jain_fairness_index_mean"
        ]
    ]

    .corr()

    .iloc[
        0,
        1
    ]
)


cor_unique = (

    df_corr[
        [
            "epsilon",
            "unique_selected_vehicles_mean"
        ]
    ]

    .corr()

    .iloc[
        0,
        1
    ]
)


print(

    f"corr(epsilon, Jain fairness) = "
    f"{cor_jain:.4f}"
)


print(

    f"corr(epsilon, unique vehicles) = "
    f"{cor_unique:.4f}"
)


# ============================================================
# INTERPRETAÇÃO AUTOMÁTICA
# ============================================================

print("\n")

print(
    "AUTOMATIC FAIRNESS CHECK"
)

print("=" * 145)


if cor_jain > 0:

    print(

        "[INFO] Higher epsilon is associated "
        "with higher allocation-distribution fairness."
    )

elif cor_jain < 0:

    print(

        "[INFO] Higher epsilon is associated "
        "with lower allocation-distribution fairness."
    )

else:

    print(

        "[INFO] No monotonic relation between "
        "epsilon and Jain fairness was observed."
    )


# ============================================================
# LATEX TABLE
# ============================================================

latex_lines = []


latex_lines.append(
    r"\begin{table}[t]"
)

latex_lines.append(
    r"\centering"
)

latex_lines.append(
    r"\caption{Allocation-distribution fairness under different $\epsilon$ values. Values are reported as mean and 95\% confidence interval over 30 simulation seeds.}"
)

latex_lines.append(
    r"\label{tab:jain_fairness_epsilon}"
)

latex_lines.append(
    r"\begin{tabular}{c|cc}"
)

latex_lines.append(
    r"\hline"
)

latex_lines.append(
    r"$\epsilon$ & Jain's Fairness Index & Unique Vehicles \\"
)

latex_lines.append(
    r"\hline"
)


for _, row in df_agregado.sort_values(
    "epsilon"
).iterrows():

    epsilon_label = (
        f"{row['epsilon']:.2f}"
    )


    if abs(
        row[
            "epsilon"
        ]
        -
        0.2
    ) < 1e-9:

        epsilon_label = (
            r"\textbf{0.20}"
        )


    jain_text = (

        f"{row['jain_fairness_index_mean']:.4f} "
        f"["
        f"{row['jain_fairness_index_ci95_low']:.4f}, "
        f"{row['jain_fairness_index_ci95_high']:.4f}"
        f"]"
    )


    unique_text = (

        f"{row['unique_selected_vehicles_mean']:.2f} "
        f"["
        f"{row['unique_selected_vehicles_ci95_low']:.2f}, "
        f"{row['unique_selected_vehicles_ci95_high']:.2f}"
        f"]"
    )


    latex_lines.append(

        f"{epsilon_label} & "
        f"{jain_text} & "
        f"{unique_text} "
        r"\\"
    )


latex_lines.append(
    r"\hline"
)

latex_lines.append(
    r"\end{tabular}"
)

latex_lines.append(
    r"\vspace{1mm}"
)

latex_lines.append(
    r"\parbox{0.96\columnwidth}{\footnotesize Jain's index is computed from the number of task allocations received by each vehicle that was selected at least once during a simulation run. Therefore, it quantifies the uniformity of allocation distribution among selected vehicles, rather than fairness over the complete eligible-vehicle population.}"
)

latex_lines.append(
    r"\end{table}"
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

print("=" * 145)


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

print(
    "IMPORTANT"
)

print("=" * 145)


print(
    "Jain's index is computed over vehicles that received "
    "at least one allocation in each run."
)

print(
    "This measures allocation-distribution fairness among "
    "selected vehicles."
)

print(
    "It must not be described as fairness over every eligible "
    "vehicle unless zero-allocation eligible vehicles are also tracked."
)


print("\n")

print("=" * 145)

print(
    "JAIN FAIRNESS ANALYSIS COMPLETED"
)

print("=" * 145)

print()