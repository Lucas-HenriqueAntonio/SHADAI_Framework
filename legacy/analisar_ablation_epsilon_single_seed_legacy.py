# analisar_ablation_epsilon.py

import os
import pandas as pd


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


BASELINE = "epsilon_020"

SEED = 42


# ============================================================
# ARQUIVOS DE SAÍDA
# ============================================================

OUTPUT_RESUMO = os.path.join(

    EPSILON_DIR,

    "resumo_ablation_epsilon_seed42.csv"
)


OUTPUT_COMPARACAO = os.path.join(

    EPSILON_DIR,

    "comparacao_ablation_epsilon_vs_baseline_seed42.csv"
)


OUTPUT_MODOS = os.path.join(

    EPSILON_DIR,

    "resumo_exploration_exploitation_seed42.csv"
)


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


def diff_percentual(
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
# VERIFICA DIRETÓRIO
# ============================================================

if not os.path.exists(
    EPSILON_DIR
):

    raise FileNotFoundError(
        f"Diretório não encontrado: "
        f"{EPSILON_DIR}"
    )


# ============================================================
# CABEÇALHO
# ============================================================

print("\n")
print("=" * 150)

print(
    "SHADAI - EPSILON SENSITIVITY ANALYSIS"
)

print("=" * 150)

print(
    f"Directory : {EPSILON_DIR}"
)

print(
    f"Seed      : {SEED}"
)

print(
    f"Baseline  : "
    f"{BASELINE} "
    f"(epsilon="
    f"{CONFIGURACOES[BASELINE]:.2f})"
)

print("=" * 150)


# ============================================================
# RESULTADOS
# ============================================================

resultados = []

resultados_modos = []

faltantes = []


# ============================================================
# LEITURA
# ============================================================

for configuracao, epsilon in CONFIGURACOES.items():

    experiment_id = (
        f"{configuracao}_"
        f"seed_"
        f"{SEED:02d}"
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


    # ========================================================
    # EXISTÊNCIA
    # ========================================================

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
            configuracao
        )

        continue


    # ========================================================
    # CARREGA
    # ========================================================

    df_metricas = pd.read_csv(
        metrics_file
    )


    df_elig = pd.read_csv(
        eligibility_file
    )


    if df_elig.empty:

        faltantes.append(
            configuracao
        )

        continue


    # ========================================================
    # REQUESTS / ELIGIBILITY
    # ========================================================

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


    # ========================================================
    # ALOCAÇÕES
    # ========================================================

    allocations = len(
        df_metricas
    )


    request_allocation_rate = safe_div(

        allocations,

        requests
    )


    # ========================================================
    # SEM ALOCAÇÕES
    # ========================================================

    if df_metricas.empty:

        resultados.append({

            "configuration":
                configuracao,

            "epsilon":
                epsilon,

            "requests":
                requests,

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


    # ========================================================
    # EXPLORATION
    # ========================================================

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


    exploration_rate = safe_div(

        exploration_count,

        allocations
    )


    # ========================================================
    # MÉTRICAS GERAIS
    # ========================================================

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


    # ========================================================
    # DIVERSIDADE
    # ========================================================

    unique_vehicles = (

        df_metricas[
            "veiculo"
        ].nunique()
    )


    allocations_per_vehicle = safe_div(

        allocations,

        unique_vehicles
    )


    # ========================================================
    # COOLDOWN
    # ========================================================

    cooldown_penalty_rate = bool_mean(

        df_metricas[
            "cooldown_penalty"
        ]
    )


    # ========================================================
    # RESULTADO GERAL
    # ========================================================

    resultados.append({

        "configuration":
            configuracao,

        "epsilon":
            epsilon,

        "requests":
            requests,

        "allocations":
            allocations,

        "request_allocation_rate":
            request_allocation_rate,

        "vehicle_evaluations":
            vehicle_evaluations,

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


    # ========================================================
    # EXPLORATION VS EXPLOITATION
    # ========================================================

    for modo in [
        "exploration",
        "exploitation"
    ]:

        subset = df_metricas[
            selection_mode
            == modo
        ]


        if subset.empty:

            resultados_modos.append({

                "configuration":
                    configuracao,

                "epsilon":
                    epsilon,

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

df = pd.DataFrame(
    resultados
)


df_modos = pd.DataFrame(
    resultados_modos
)


if df.empty:

    raise RuntimeError(
        "Nenhum experimento válido de epsilon foi encontrado."
    )


# ============================================================
# SALVA
# ============================================================

df.to_csv(
    OUTPUT_RESUMO,
    index=False
)


df_modos.to_csv(
    OUTPUT_MODOS,
    index=False
)


# ============================================================
# BASELINE
# ============================================================

baseline = df[
    df[
        "configuration"
    ] == BASELINE
]


if baseline.empty:

    raise RuntimeError(
        "Configuração baseline epsilon_020 "
        "não foi encontrada."
    )


baseline = baseline.iloc[
    0
]


# ============================================================
# COMPARAÇÃO CONTRA EPSILON 0.2
# ============================================================

METRICAS_COMPARACAO = [

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


comparacoes = []


for _, row in df.iterrows():

    for metrica in METRICAS_COMPARACAO:

        valor = row[
            metrica
        ]

        referencia = baseline[
            metrica
        ]


        comparacoes.append({

            "configuration":
                row[
                    "configuration"
                ],

            "epsilon":
                row[
                    "epsilon"
                ],

            "metric":
                metrica,

            "baseline":
                referencia,

            "value":
                valor,

            "difference":
                valor
                - referencia,

            "difference_pct":
                diff_percentual(
                    valor,
                    referencia
                )
        })


df_comparacao = pd.DataFrame(
    comparacoes
)


df_comparacao.to_csv(
    OUTPUT_COMPARACAO,
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
    "MAIN EPSILON SENSITIVITY RESULTS"
)

print("=" * 150)


cabecalho = (

    f"{'CONFIG':<14}"

    f"{'EPS':>7}"

    f"{'EXP%':>9}"

    f"{'ALLOC%':>10}"

    f"{'SCORE':>10}"

    f"{'REP_N':>10}"

    f"{'ENER_N':>10}"

    f"{'PROX_N':>10}"

    f"{'DIST':>10}"

    f"{'ENERGY':>10}"

    f"{'UNIQUE':>9}"
)


print(
    cabecalho
)

print("-" * 150)


for _, row in df.sort_values(
    "epsilon"
).iterrows():

    print(

        f"{row['configuration']:<14}"

        f"{row['epsilon']:>7.2f}"

        f"{row['exploration_rate'] * 100:>8.2f}%"

        f"{row['request_allocation_rate'] * 100:>9.2f}%"

        f"{row['avg_score_final']:>10.4f}"

        f"{row['avg_reputation_norm']:>10.4f}"

        f"{row['avg_energy_norm']:>10.4f}"

        f"{row['avg_proximity_norm']:>10.4f}"

        f"{row['avg_distance']:>10.2f}"

        f"{row['avg_energy']:>10.2f}"

        f"{int(row['unique_vehicles']):>9}"
    )


# ============================================================
# EPSILON VS EXPLORATION OBSERVADA
# ============================================================

print("\n")

print(
    "CONFIGURED EPSILON VS OBSERVED EXPLORATION"
)

print("=" * 150)


for _, row in df.sort_values(
    "epsilon"
).iterrows():

    esperado = (
        row[
            "epsilon"
        ]
    )


    observado = (
        row[
            "exploration_rate"
        ]
    )


    print(

        f"epsilon="
        f"{esperado:.2f} | "

        f"observed exploration="
        f"{observado * 100:.2f}% | "

        f"difference="
        f"{(observado - esperado) * 100:+.2f} pp"
    )


# ============================================================
# DIVERSIDADE
# ============================================================

print("\n")

print(
    "SELECTION DIVERSITY"
)

print("=" * 150)


for _, row in df.sort_values(
    "epsilon"
).iterrows():

    print(

        f"epsilon="
        f"{row['epsilon']:.2f} | "

        f"unique vehicles="
        f"{int(row['unique_vehicles'])} | "

        f"allocations/vehicle="
        f"{row['allocations_per_vehicle']:.3f}"
    )


# ============================================================
# ELIGIBILITY CONTROL
# ============================================================

print("\n")

print(
    "ELIGIBILITY / ALLOCATION CONTROL"
)

print("=" * 150)


for _, row in df.sort_values(
    "epsilon"
).iterrows():

    print(

        f"epsilon="
        f"{row['epsilon']:.2f} | "

        f"eligibility="
        f"{row['overall_eligibility_rate'] * 100:.2f}% | "

        f"allocation="
        f"{row['request_allocation_rate'] * 100:.2f}% | "

        f"cooldown="
        f"{row['cooldown_penalty_rate'] * 100:.2f}%"
    )


# ============================================================
# EXPLORATION VS EXPLOITATION
# ============================================================

print("\n")

print(
    "EXPLORATION VS EXPLOITATION QUALITY"
)

print("=" * 150)


for configuracao, epsilon in CONFIGURACOES.items():

    subset = df_modos[
        df_modos[
            "configuration"
        ] == configuracao
    ]


    exploracao = subset[
        subset[
            "selection_mode"
        ] == "exploration"
    ]


    exploitation = subset[
        subset[
            "selection_mode"
        ] == "exploitation"
    ]


    print(
        f"\nepsilon={epsilon:.2f}"
    )


    if not exploracao.empty:

        exp = exploracao.iloc[
            0
        ]

        print(

            f"  exploration  | "

            f"n={int(exp['n']):>3} | "

            f"score="
            f"{exp['avg_score']:.4f} | "

            f"rep="
            f"{exp['avg_reputation_norm']:.4f} | "

            f"energy="
            f"{exp['avg_energy_norm']:.4f} | "

            f"proximity="
            f"{exp['avg_proximity_norm']:.4f} | "

            f"distance="
            f"{exp['avg_distance']:.2f}"
        )


    if not exploitation.empty:

        greedy = exploitation.iloc[
            0
        ]

        print(

            f"  exploitation | "

            f"n={int(greedy['n']):>3} | "

            f"score="
            f"{greedy['avg_score']:.4f} | "

            f"rep="
            f"{greedy['avg_reputation_norm']:.4f} | "

            f"energy="
            f"{greedy['avg_energy_norm']:.4f} | "

            f"proximity="
            f"{greedy['avg_proximity_norm']:.4f} | "

            f"distance="
            f"{greedy['avg_distance']:.2f}"
        )


# ============================================================
# SANITY CHECK
# ============================================================

print("\n")

print(
    "AUTOMATIC SANITY CHECK"
)

print("=" * 150)


df_ordenado = df.sort_values(
    "epsilon"
)


linha_zero = df[
    df[
        "configuration"
    ] == "epsilon_000"
]


linha_um = df[
    df[
        "configuration"
    ] == "epsilon_100"
]


if not linha_zero.empty:

    exploracao_zero = (

        linha_zero.iloc[
            0
        ][
            "exploration_rate"
        ]
    )


    if abs(
        exploracao_zero
    ) < 1e-12:

        print(
            "[PASS] epsilon=0.0 produced 0% exploration."
        )

    else:

        print(
            "[CHECK] epsilon=0.0 produced non-zero exploration."
        )


if not linha_um.empty:

    exploracao_um = (

        linha_um.iloc[
            0
        ][
            "exploration_rate"
        ]
    )


    if abs(
        exploracao_um - 1.0
    ) < 1e-12:

        print(
            "[PASS] epsilon=1.0 produced 100% exploration."
        )

    else:

        print(
            "[CHECK] epsilon=1.0 did not produce "
            "100% exploration."
        )


correlacao = (

    df[
        [
            "epsilon",
            "exploration_rate"
        ]
    ]

    .corr()

    .iloc[
        0,
        1
    ]
)


print(
    f"[INFO] Correlation between configured epsilon "
    f"and observed exploration rate: "
    f"{correlacao:.4f}"
)


if correlacao > 0.95:

    print(
        "[PASS] Exploration rate strongly follows epsilon."
    )

else:

    print(
        "[CHECK] Exploration rate does not strongly "
        "follow epsilon."
    )


# ============================================================
# OUTPUTS
# ============================================================

print("\n")

print(
    "OUTPUT FILES"
)

print("=" * 150)


print(
    f"Summary          : "
    f"{OUTPUT_RESUMO}"
)


print(
    f"Vs baseline      : "
    f"{OUTPUT_COMPARACAO}"
)


print(
    f"Selection modes  : "
    f"{OUTPUT_MODOS}"
)


# ============================================================
# NOTA
# ============================================================

print("\n")

print(
    "IMPORTANT"
)

print("=" * 150)


print(
    "This is a single-seed exploratory sensitivity analysis."
)


print(
    "Do not interpret single-seed differences as final "
    "statistical evidence."
)


print(
    "The primary validation at this stage is whether the "
    "observed exploration rate follows configured epsilon "
    "and whether increasing exploration produces plausible "
    "quality/diversity trade-offs."
)


print(
    "If the directional behavior is coherent, the next step "
    "is a 30-seed experiment with mean, SD and 95% confidence "
    "intervals."
)


print("\n")
print("=" * 150)

print(
    "EPSILON ANALYSIS COMPLETED"
)

print("=" * 150)
print()