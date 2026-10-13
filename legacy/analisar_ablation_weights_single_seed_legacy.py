# analisar_ablation_weights.py

import os
import pandas as pd
import numpy as np


# ============================================================
# DIRETÓRIO
# ============================================================

WEIGHT_DIR = "weight_ablation"


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


# ============================================================
# SEED
# ============================================================

SEED = 42


# ============================================================
# ARQUIVOS DE SAÍDA
# ============================================================

OUTPUT_RESUMO = os.path.join(
    WEIGHT_DIR,
    "resumo_ablation_weights_seed42.csv"
)


OUTPUT_COMPARACAO = os.path.join(
    WEIGHT_DIR,
    "comparacao_ablation_weights_vs_original_seed42.csv"
)


# ============================================================
# FUNÇÕES AUXILIARES
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
        or referencia == 0
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

    # ========================================================
    # ACEITA TRUE/FALSE COMO STRING OU BOOLEAN
    # ========================================================

    normalizada = serie.astype(
        str
    ).str.lower().map({

        "true": 1,

        "false": 0,

        "1": 1,

        "0": 0
    })


    return normalizada.mean()


# ============================================================
# VERIFICA DIRETÓRIO
# ============================================================

if not os.path.exists(
    WEIGHT_DIR
):

    raise FileNotFoundError(
        f"Diretório não encontrado: "
        f"{WEIGHT_DIR}"
    )


# ============================================================
# CABEÇALHO
# ============================================================

print("\n")

print(
    "=" * 130
)

print(
    "SHADAI - WEIGHT SENSITIVITY / ABLATION ANALYSIS"
)

print(
    "=" * 130
)

print(
    f"Directory : "
    f"{WEIGHT_DIR}"
)

print(
    f"Seed      : "
    f"{SEED}"
)

print(
    "=" * 130
)


# ============================================================
# RESULTADOS
# ============================================================

resultados = []

faltantes = []


# ============================================================
# LEITURA DOS EXPERIMENTOS
# ============================================================

for configuracao in CONFIGURACOES:

    w1, w2, w3 = (
        PESOS[
            configuracao
        ]
    )


    experiment_id = (

        f"weights_"
        f"{configuracao}_"
        f"seed_"
        f"{SEED:02d}"
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
    # LEITURA
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
    # REQUESTS
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


    # ========================================================
    # ALOCAÇÕES
    # ========================================================

    allocations = len(
        df_metricas
    )


    request_allocation_rate = (
        safe_div(
            allocations,
            requests
        )
    )


    # ========================================================
    # SE NÃO HOUVER ALOCAÇÃO
    # ========================================================

    if df_metricas.empty:

        resultados.append({

            "configuration":
                configuracao,

            "w1":
                w1,

            "w2":
                w2,

            "w3":
                w3,

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

            "avg_reputation":
                float("nan"),

            "avg_energy":
                float("nan"),

            "avg_utility":
                float("nan"),

            "unique_vehicles":
                0,

            "allocations_per_vehicle":
                float("nan"),

            "exploration_rate":
                float("nan"),

            "exploitation_rate":
                float("nan"),

            "cooldown_penalty_rate":
                float("nan")
        })

        continue


    # ========================================================
    # SCORE
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


    # ========================================================
    # COMPONENTES NORMALIZADOS
    # ========================================================

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


    # ========================================================
    # DISTÂNCIA
    # ========================================================

    avg_distance = (
        df_metricas[
            "distancia"
        ].mean()
    )


    # ========================================================
    # VALORES REAIS
    # ========================================================

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


    # ========================================================
    # DIVERSIDADE DE SELEÇÃO
    # ========================================================

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


    # ========================================================
    # EXPLORATION / EXPLOITATION
    # ========================================================

    exploration_count = (

        df_metricas[
            "selection_mode"
        ]
        .astype(str)
        .str.lower()
        .eq(
            "exploration"
        )
        .sum()
    )


    exploitation_count = (

        df_metricas[
            "selection_mode"
        ]
        .astype(str)
        .str.lower()
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


    exploitation_rate = (
        safe_div(
            exploitation_count,
            allocations
        )
    )


    # ========================================================
    # COOLDOWN
    # ========================================================

    cooldown_penalty_rate = (
        bool_mean(
            df_metricas[
                "cooldown_penalty"
            ]
        )
    )


    # ========================================================
    # RESULTADO
    # ========================================================

    resultados.append({

        "configuration":
            configuracao,

        "w1":
            w1,

        "w2":
            w2,

        "w3":
            w3,

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

        "exploitation_rate":
            exploitation_rate,

        "cooldown_penalty_rate":
            cooldown_penalty_rate
    })


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    resultados
)


if df.empty:

    raise RuntimeError(
        "Nenhum experimento válido de weights foi encontrado."
    )


# ============================================================
# SALVA RESUMO
# ============================================================

df.to_csv(
    OUTPUT_RESUMO,
    index=False
)


# ============================================================
# BASELINE
# ============================================================

baseline = df[
    df[
        "configuration"
    ] == "original"
]


if baseline.empty:

    raise RuntimeError(
        "Configuração original não encontrada."
    )


baseline = (
    baseline.iloc[
        0
    ]
)


# ============================================================
# COMPARAÇÃO COM ORIGINAL
# ============================================================

metricas_comparacao = [

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


comparacoes = []


for _, row in df.iterrows():

    config = row[
        "configuration"
    ]


    for metrica in metricas_comparacao:

        valor = row[
            metrica
        ]

        referencia = baseline[
            metrica
        ]


        diferenca = (
            valor
            - referencia
        )


        diferenca_pct = (
            diff_percentual(
                valor,
                referencia
            )
        )


        comparacoes.append({

            "configuration":
                config,

            "metric":
                metrica,

            "original":
                referencia,

            "value":
                valor,

            "difference":
                diferenca,

            "difference_pct":
                diferenca_pct
        })


df_comparacao = pd.DataFrame(
    comparacoes
)


df_comparacao.to_csv(
    OUTPUT_COMPARACAO,
    index=False
)


# ============================================================
# CONSOLE - TABELA PRINCIPAL
# ============================================================

pd.set_option(
    "display.width",
    250
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
    "MAIN WEIGHT SENSITIVITY RESULTS"
)

print(
    "=" * 130
)


cabecalho = (

    f"{'CONFIGURATION':<19}"

    f"{'ALLOC%':>9}"

    f"{'REP_N':>9}"

    f"{'ENER_N':>9}"

    f"{'PROX_N':>9}"

    f"{'DIST':>10}"

    f"{'SCORE':>10}"

    f"{'ENERGY':>10}"

    f"{'UNIQUE':>9}"
)


print(
    cabecalho
)


print(
    "-" * 130
)


for _, row in df.iterrows():

    print(

        f"{row['configuration']:<19}"

        f"{row['request_allocation_rate'] * 100:>8.2f}%"

        f"{row['avg_reputation_norm']:>9.4f}"

        f"{row['avg_energy_norm']:>9.4f}"

        f"{row['avg_proximity_norm']:>9.4f}"

        f"{row['avg_distance']:>10.2f}"

        f"{row['avg_score_final']:>10.4f}"

        f"{row['avg_energy']:>10.2f}"

        f"{int(row['unique_vehicles']):>9}"
    )


# ============================================================
# SENSITIVITY CHECK
# ============================================================

print("\n")

print(
    "DIRECT SENSITIVITY CHECK"
)

print(
    "=" * 130
)


def get_config(
    nome
):

    linha = df[
        df[
            "configuration"
        ] == nome
    ]

    if linha.empty:

        return None

    return (
        linha.iloc[
            0
        ]
    )


original = get_config(
    "original"
)


reputation_high = get_config(
    "reputation_high"
)


energy_high = get_config(
    "energy_high"
)


proximity_high = get_config(
    "proximity_high"
)


# ============================================================
# REPUTAÇÃO
# ============================================================

if (
    original is not None
    and
    reputation_high is not None
):

    delta_rep = (

        reputation_high[
            "avg_reputation_norm"
        ]

        - original[
            "avg_reputation_norm"
        ]
    )


    print(
        f"reputation_high -> "
        f"Δ selected reputation_norm = "
        f"{delta_rep:+.4f}"
    )


# ============================================================
# ENERGIA
# ============================================================

if (
    original is not None
    and
    energy_high is not None
):

    delta_energy = (

        energy_high[
            "avg_energy_norm"
        ]

        - original[
            "avg_energy_norm"
        ]
    )


    print(
        f"energy_high     -> "
        f"Δ selected energy_norm     = "
        f"{delta_energy:+.4f}"
    )


# ============================================================
# PROXIMIDADE
# ============================================================

if (
    original is not None
    and
    proximity_high is not None
):

    delta_prox = (

        proximity_high[
            "avg_proximity_norm"
        ]

        - original[
            "avg_proximity_norm"
        ]
    )


    delta_dist = (

        proximity_high[
            "avg_distance"
        ]

        - original[
            "avg_distance"
        ]
    )


    print(
        f"proximity_high  -> "
        f"Δ selected proximity_norm  = "
        f"{delta_prox:+.4f}"
    )


    print(
        f"proximity_high  -> "
        f"Δ selected distance        = "
        f"{delta_dist:+.2f}"
    )


# ============================================================
# ABLATION CHECK
# ============================================================

print("\n")

print(
    "COMPONENT ABLATION CHECK"
)

print(
    "=" * 130
)


for config in [

    "no_reputation",

    "no_energy",

    "no_proximity"
]:

    row = get_config(
        config
    )


    if row is None:

        continue


    print(
        f"\n{config}"
    )


    print(
        f"  Reputation norm : "
        f"{row['avg_reputation_norm']:.4f} "
        f"(Δ={row['avg_reputation_norm'] - original['avg_reputation_norm']:+.4f})"
    )


    print(
        f"  Energy norm     : "
        f"{row['avg_energy_norm']:.4f} "
        f"(Δ={row['avg_energy_norm'] - original['avg_energy_norm']:+.4f})"
    )


    print(
        f"  Proximity norm  : "
        f"{row['avg_proximity_norm']:.4f} "
        f"(Δ={row['avg_proximity_norm'] - original['avg_proximity_norm']:+.4f})"
    )


    print(
        f"  Distance        : "
        f"{row['avg_distance']:.2f} "
        f"(Δ={row['avg_distance'] - original['avg_distance']:+.2f})"
    )


    print(
        f"  Energy          : "
        f"{row['avg_energy']:.2f} "
        f"(Δ={row['avg_energy'] - original['avg_energy']:+.2f})"
    )


    print(
        f"  Unique vehicles : "
        f"{int(row['unique_vehicles'])}"
    )


# ============================================================
# ELIGIBILITY CONTROL
# ============================================================

print("\n")

print(
    "ELIGIBILITY CONTROL"
)

print(
    "=" * 130
)


for _, row in df.iterrows():

    print(

        f"{row['configuration']:<19} "

        f"overall eligibility = "
        f"{row['overall_eligibility_rate'] * 100:.2f}% | "

        f"allocation = "
        f"{row['request_allocation_rate'] * 100:.2f}%"
    )


# ============================================================
# EXPLORATION / COOLDOWN
# ============================================================

print("\n")

print(
    "EXPLORATION AND COOLDOWN DIAGNOSTIC"
)

print(
    "=" * 130
)


for _, row in df.iterrows():

    print(

        f"{row['configuration']:<19} "

        f"exploration="
        f"{row['exploration_rate'] * 100:.2f}% | "

        f"cooldown penalty="
        f"{row['cooldown_penalty_rate'] * 100:.2f}% | "

        f"alloc/vehicle="
        f"{row['allocations_per_vehicle']:.3f}"
    )


# ============================================================
# AUTOMATIC INTERPRETATION
# ============================================================

print("\n")

print(
    "AUTOMATIC SANITY CHECK"
)

print(
    "=" * 130
)


checks = []


if (
    reputation_high is not None
):

    cond = (

        reputation_high[
            "avg_reputation_norm"
        ]

        >
        original[
            "avg_reputation_norm"
        ]
    )


    checks.append(
        (
            "Higher w1 increases selected reputation",
            cond
        )
    )


if (
    energy_high is not None
):

    cond = (

        energy_high[
            "avg_energy_norm"
        ]

        >
        original[
            "avg_energy_norm"
        ]
    )


    checks.append(
        (
            "Higher w2 increases selected energy",
            cond
        )
    )


if (
    proximity_high is not None
):

    cond_prox = (

        proximity_high[
            "avg_proximity_norm"
        ]

        >
        original[
            "avg_proximity_norm"
        ]
    )


    cond_dist = (

        proximity_high[
            "avg_distance"
        ]

        <
        original[
            "avg_distance"
        ]
    )


    checks.append(
        (
            "Higher w3 increases selected proximity",
            cond_prox
        )
    )


    checks.append(
        (
            "Higher w3 reduces selected distance",
            cond_dist
        )
    )


for descricao, resultado in checks:

    status = (
        "PASS"
        if resultado
        else "CHECK"
    )


    print(
        f"[{status}] "
        f"{descricao}"
    )


# ============================================================
# ARQUIVOS GERADOS
# ============================================================

print("\n")

print(
    "OUTPUT FILES"
)

print(
    "=" * 130
)


print(
    f"Summary     : "
    f"{OUTPUT_RESUMO}"
)


print(
    f"Comparison  : "
    f"{OUTPUT_COMPARACAO}"
)


# ============================================================
# AVISO METODOLÓGICO
# ============================================================

print("\n")

print(
    "IMPORTANT"
)

print(
    "=" * 130
)


print(
    "This is a single-seed exploratory analysis."
)


print(
    "Do not use these differences as final statistical evidence."
)


print(
    "If the expected directional effects are observed, "
    "the next stage should repeat all weight configurations "
    "over multiple independent seeds."
)


print(
    "The average score itself must be interpreted carefully, "
    "because changing w1/w2/w3 also changes the score definition."
)


print("\n")

print(
    "=" * 130
)

print(
    "WEIGHT ANALYSIS COMPLETED"
)

print(
    "=" * 130
)

print()