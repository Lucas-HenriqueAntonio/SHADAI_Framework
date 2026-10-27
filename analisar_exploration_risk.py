# analisar_exploration_risk.py

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

EPSILON_DIR = "epsilon_ablation"

RNG_SCHEME_REQUIRED = "split_v1"

TELEMETRY_SCHEME_REQUIRED = "c9_regret_v1"


CONFIGURACOES = {

    "epsilon_000": 0.0,

    "epsilon_010": 0.1,

    "epsilon_020": 0.2,

    "epsilon_030": 0.3,

    "epsilon_050": 0.5,

    "epsilon_100": 1.0
}


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

    "exploration_risk_por_seed.csv"
)


OUTPUT_AGREGADO = os.path.join(

    EPSILON_DIR,

    "exploration_risk_multiseed.csv"
)


OUTPUT_LATEX = os.path.join(

    EPSILON_DIR,

    "tabela_exploration_risk_tnsm.tex"
)


# ============================================================
# FUNÇÕES
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


def validar_dataframe(
    df
):

    colunas = [

        "rng_scheme",

        "telemetry_scheme",

        "selection_mode",

        "score",

        "candidate_count",

        "chosen_rank",

        "greedy_best_score",

        "greedy_best_reputation_norm",

        "greedy_best_energy_norm",

        "greedy_best_proximity_norm",

        "greedy_best_distance",

        "exploration_regret",

        "exploration_regret_relative",

        "selected_greedy_best"
    ]

    for coluna in colunas:

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

    return True


def bool_series(
    serie
):

    return (

        serie.astype(str)
        .str.lower()
        .map({

            "true": True,

            "false": False,

            "1": True,

            "0": False
        })
    )


# ============================================================
# LEITURA
# ============================================================

print("\n")
print("=" * 160)

print(
    "SHADAI - EXPLORATION RISK / COUNTERFACTUAL GREEDY ANALYSIS"
)

print("=" * 160)

print(
    f"Directory         : {EPSILON_DIR}"
)

print(
    f"RNG scheme        : {RNG_SCHEME_REQUIRED}"
)

print(
    f"Telemetry scheme  : {TELEMETRY_SCHEME_REQUIRED}"
)

print(
    f"Seeds             : {len(SEEDS)}"
)

print("=" * 160)


resultados = []

faltantes = []

invalidos = []


for config, epsilon in CONFIGURACOES.items():

    for seed in SEEDS:

        experiment_id = (

            f"{config}_"
            f"seed_"
            f"{seed:02d}"
        )

        arquivo = os.path.join(

            EPSILON_DIR,

            (
                f"metricas_shadai_400_"
                f"{experiment_id}.csv"
            )
        )

        if not os.path.exists(
            arquivo
        ):

            faltantes.append(
                (
                    config,
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
            not validar_dataframe(
                df
            )
        ):

            invalidos.append(
                (
                    config,
                    seed
                )
            )

            continue

        df[
            "selected_greedy_best_bool"
        ] = bool_series(

            df[
                "selected_greedy_best"
            ]
        )

        exploration = df[
            df[
                "selection_mode"
            ]
            .astype(str)
            .str.lower()
            .eq(
                "exploration"
            )
        ].copy()

        exploitation = df[
            df[
                "selection_mode"
            ]
            .astype(str)
            .str.lower()
            .eq(
                "exploitation"
            )
        ].copy()

        total = len(
            df
        )

        n_exploration = len(
            exploration
        )

        n_exploitation = len(
            exploitation
        )

        exploration_rate = (
            n_exploration
            / total
            if total > 0
            else float("nan")
        )

        # ====================================================
        # SANITY: EXPLOITATION DEVE TER REGRET ~ 0
        # ====================================================

        exploitation_regret = (

            exploitation[
                "exploration_regret"
            ].mean()

            if not exploitation.empty

            else float("nan")
        )

        # ====================================================
        # MÉTRICAS DE RISCO CONDICIONAIS À EXPLORAÇÃO
        # ====================================================

        if not exploration.empty:

            mean_regret = (

                exploration[
                    "exploration_regret"
                ].mean()
            )

            mean_relative_regret = (

                exploration[
                    "exploration_regret_relative"
                ].mean()
            )

            median_relative_regret = (

                exploration[
                    "exploration_regret_relative"
                ].median()
            )

            p90_relative_regret = (

                exploration[
                    "exploration_regret_relative"
                ].quantile(
                    0.90
                )
            )

            probability_nonzero_regret = (

                exploration[
                    "exploration_regret"
                ]
                .gt(
                    1e-12
                )
                .mean()
            )

            risk_10pct = (

                exploration[
                    "exploration_regret_relative"
                ]
                .ge(
                    0.10
                )
                .mean()
            )

            risk_20pct = (

                exploration[
                    "exploration_regret_relative"
                ]
                .ge(
                    0.20
                )
                .mean()
            )

            mean_chosen_rank = (

                exploration[
                    "chosen_rank"
                ].mean()
            )

            selected_best_rate = (

                exploration[
                    "selected_greedy_best_bool"
                ].mean()
            )

            mean_candidate_count = (

                exploration[
                    "candidate_count"
                ].mean()
            )

            chosen_to_greedy_ratio = (

                exploration[
                    "score"
                ]
                /
                exploration[
                    "greedy_best_score"
                ]
            ).mean()

            distance_penalty = (

                exploration[
                    "distancia"
                ]
                -
                exploration[
                    "greedy_best_distance"
                ]
            ).mean()

            reputation_gap = (

                exploration[
                    "reputacao_norm"
                ]
                -
                exploration[
                    "greedy_best_reputation_norm"
                ]
            ).mean()

            energy_gap = (

                exploration[
                    "energia_norm"
                ]
                -
                exploration[
                    "greedy_best_energy_norm"
                ]
            ).mean()

            proximity_gap = (

                exploration[
                    "proximidade_norm"
                ]
                -
                exploration[
                    "greedy_best_proximity_norm"
                ]
            ).mean()

        else:

            mean_regret = float("nan")

            mean_relative_regret = float("nan")

            median_relative_regret = float("nan")

            p90_relative_regret = float("nan")

            probability_nonzero_regret = float("nan")

            risk_10pct = float("nan")

            risk_20pct = float("nan")

            mean_chosen_rank = float("nan")

            selected_best_rate = float("nan")

            mean_candidate_count = float("nan")

            chosen_to_greedy_ratio = float("nan")

            distance_penalty = float("nan")

            reputation_gap = float("nan")

            energy_gap = float("nan")

            proximity_gap = float("nan")

        # ====================================================
        # REGRET GLOBAL DA POLÍTICA
        #
        # exploitation possui regret 0;
        # portanto mede o custo médio efetivo de epsilon.
        # ====================================================

        overall_mean_regret = (

            df[
                "exploration_regret"
            ].mean()
        )

        overall_relative_regret = (

            df[
                "exploration_regret_relative"
            ].mean()
        )

        resultados.append({

            "configuration":
                config,

            "epsilon":
                epsilon,

            "seed":
                seed,

            "total_allocations":
                total,

            "exploration_count":
                n_exploration,

            "exploitation_count":
                n_exploitation,

            "exploration_rate":
                exploration_rate,

            "mean_exploration_regret":
                mean_regret,

            "mean_exploration_relative_regret":
                mean_relative_regret,

            "median_exploration_relative_regret":
                median_relative_regret,

            "p90_exploration_relative_regret":
                p90_relative_regret,

            "probability_nonzero_regret":
                probability_nonzero_regret,

            "risk_regret_ge_10pct":
                risk_10pct,

            "risk_regret_ge_20pct":
                risk_20pct,

            "mean_exploration_chosen_rank":
                mean_chosen_rank,

            "exploration_selected_best_rate":
                selected_best_rate,

            "mean_exploration_candidate_count":
                mean_candidate_count,

            "chosen_to_greedy_score_ratio":
                chosen_to_greedy_ratio,

            "exploration_distance_penalty":
                distance_penalty,

            "exploration_reputation_gap":
                reputation_gap,

            "exploration_energy_gap":
                energy_gap,

            "exploration_proximity_gap":
                proximity_gap,

            "overall_mean_regret":
                overall_mean_regret,

            "overall_relative_regret":
                overall_relative_regret,

            "exploitation_mean_regret":
                exploitation_regret
        })


# ============================================================
# DATAFRAME
# ============================================================

df_seed = pd.DataFrame(
    resultados
)


if df_seed.empty:

    raise RuntimeError(
        "Nenhum experimento válido de C9 foi encontrado."
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

print("-" * 160)


for config in CONFIGURACOES:

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
        f"{n:>2}/30 runs"
    )


if faltantes:

    print(
        f"\n[WARNING] Missing runs: "
        f"{len(faltantes)}"
    )


if invalidos:

    print(
        f"\n[WARNING] Invalid telemetry runs: "
        f"{len(invalidos)}"
    )


# ============================================================
# AGREGAÇÃO
# ============================================================

METRICAS = [

    "exploration_rate",

    "mean_exploration_regret",

    "mean_exploration_relative_regret",

    "median_exploration_relative_regret",

    "p90_exploration_relative_regret",

    "probability_nonzero_regret",

    "risk_regret_ge_10pct",

    "risk_regret_ge_20pct",

    "mean_exploration_chosen_rank",

    "exploration_selected_best_rate",

    "mean_exploration_candidate_count",

    "chosen_to_greedy_score_ratio",

    "exploration_distance_penalty",

    "exploration_reputation_gap",

    "exploration_energy_gap",

    "exploration_proximity_gap",

    "overall_mean_regret",

    "overall_relative_regret",

    "exploitation_mean_regret"
]


agregados = []


for config, epsilon in CONFIGURACOES.items():

    dados = df_seed[
        df_seed[
            "configuration"
        ]
        ==
        config
    ]

    linha = {

        "configuration":
            config,

        "epsilon":
            epsilon,

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

    agregados.append(
        linha
    )


df_agregado = pd.DataFrame(
    agregados
)


df_agregado.to_csv(
    OUTPUT_AGREGADO,
    index=False
)


# ============================================================
# RESULTADOS PRINCIPAIS
# ============================================================

print("\n")

print(
    "MAIN EXPLORATION RISK RESULTS"
)

print("=" * 160)


header = (

    f"{'EPS':>6}"

    f"{'EXP%':>9}"

    f"{'REGRET':>11}"

    f"{'REL.REG':>11}"

    f"{'RISK>=10%':>12}"

    f"{'RISK>=20%':>12}"

    f"{'BEST%':>10}"

    f"{'RANK':>9}"

    f"{'DIST.PEN':>11}"
)


print(
    header
)

print("-" * 160)


for _, row in df_agregado.sort_values(
    "epsilon"
).iterrows():

    print(

        f"{row['epsilon']:>6.2f}"

        f"{row['exploration_rate_mean'] * 100:>8.2f}%"

        f"{row['mean_exploration_regret_mean']:>11.4f}"

        f"{row['mean_exploration_relative_regret_mean'] * 100:>10.2f}%"

        f"{row['risk_regret_ge_10pct_mean'] * 100:>11.2f}%"

        f"{row['risk_regret_ge_20pct_mean'] * 100:>11.2f}%"

        f"{row['exploration_selected_best_rate_mean'] * 100:>9.2f}%"

        f"{row['mean_exploration_chosen_rank_mean']:>9.2f}"

        f"{row['exploration_distance_penalty_mean']:>11.2f}"
    )


# ============================================================
# BASELINE EPSILON=0.2
# ============================================================

baseline = df_agregado[
    df_agregado[
        "configuration"
    ]
    ==
    "epsilon_020"
]


print("\n")

print(
    "SHADAI BASELINE EXPLORATION RISK (EPSILON=0.20)"
)

print("=" * 160)


if not baseline.empty:

    row = baseline.iloc[
        0
    ]

    print(

        f"Observed exploration rate     : "
        f"{row['exploration_rate_mean'] * 100:.2f}%"
    )

    print(

        f"Mean exploration regret       : "
        f"{row['mean_exploration_regret_mean']:.4f} "
        f"["
        f"{row['mean_exploration_regret_ci95_low']:.4f}, "
        f"{row['mean_exploration_regret_ci95_high']:.4f}"
        f"]"
    )

    print(

        f"Mean relative regret          : "
        f"{row['mean_exploration_relative_regret_mean'] * 100:.2f}% "
        f"["
        f"{row['mean_exploration_relative_regret_ci95_low'] * 100:.2f}%, "
        f"{row['mean_exploration_relative_regret_ci95_high'] * 100:.2f}%"
        f"]"
    )

    print(

        f"Exploration risk >=10%        : "
        f"{row['risk_regret_ge_10pct_mean'] * 100:.2f}%"
    )

    print(

        f"Exploration risk >=20%        : "
        f"{row['risk_regret_ge_20pct_mean'] * 100:.2f}%"
    )

    print(

        f"Exploration picked greedy-best: "
        f"{row['exploration_selected_best_rate_mean'] * 100:.2f}%"
    )

    print(

        f"Mean chosen rank              : "
        f"{row['mean_exploration_chosen_rank_mean']:.2f}"
    )

    print(

        f"Mean distance penalty         : "
        f"{row['exploration_distance_penalty_mean']:.2f}"
    )

    print(

        f"Overall policy relative regret: "
        f"{row['overall_relative_regret_mean'] * 100:.2f}%"
    )


# ============================================================
# SANITY CHECK
# ============================================================

print("\n")

print(
    "AUTOMATIC C9 SANITY CHECK"
)

print("=" * 160)


exploit = df_seed[
    "exploitation_mean_regret"
].dropna()


if (
    len(
        exploit
    ) == 0
    or
    exploit.abs().max() < 1e-9
):

    print(

        "[PASS] Exploitation decisions have zero "
        "counterfactual greedy regret."
    )

else:

    print(

        "[CHECK] Non-zero regret detected in "
        "exploitation decisions."
    )


epsilon_zero = df_agregado[
    df_agregado[
        "configuration"
    ]
    ==
    "epsilon_000"
]


if not epsilon_zero.empty:

    exp_rate = (
        epsilon_zero.iloc[
            0
        ][
            "exploration_rate_mean"
        ]
    )

    if abs(
        exp_rate
    ) < 1e-12:

        print(
            "[PASS] epsilon=0 produced no exploration."
        )

    else:

        print(
            "[CHECK] epsilon=0 produced exploration."
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
    r"\caption{Risk of exploratory selections under the SHADAI $\epsilon$-greedy policy. Regret is computed against the counterfactual greedy-best candidate available in the same request. Values are means over 30 simulation seeds.}"
)

latex_lines.append(
    r"\label{tab:exploration_risk}"
)

latex_lines.append(
    r"\resizebox{\textwidth}{!}{%"
)

latex_lines.append(
    r"\begin{tabular}{c|ccccccc}"
)

latex_lines.append(
    r"\hline"
)

latex_lines.append(
    r"$\epsilon$ & Exploration & Rel. Regret & Risk $\geq10\%$ & Risk $\geq20\%$ & Greedy-Best Selected & Mean Rank & Distance Penalty \\"
)

latex_lines.append(
    r"\hline"
)


for _, row in df_agregado.sort_values(
    "epsilon"
).iterrows():

    epsilon = row[
        "epsilon"
    ]

    epsilon_text = (
        f"{epsilon:.2f}"
    )

    if abs(
        epsilon - 0.2
    ) < 1e-9:

        epsilon_text = (
            r"\textbf{0.20}"
        )

    if pd.isna(
        row[
            "mean_exploration_relative_regret_mean"
        ]
    ):

        relative = "--"

        risk10 = "--"

        risk20 = "--"

        best = "--"

        rank = "--"

        distance = "--"

    else:

        relative = (
            f"{row['mean_exploration_relative_regret_mean'] * 100:.2f}\\%"
        )

        risk10 = (
            f"{row['risk_regret_ge_10pct_mean'] * 100:.2f}\\%"
        )

        risk20 = (
            f"{row['risk_regret_ge_20pct_mean'] * 100:.2f}\\%"
        )

        best = (
            f"{row['exploration_selected_best_rate_mean'] * 100:.2f}\\%"
        )

        rank = (
            f"{row['mean_exploration_chosen_rank_mean']:.2f}"
        )

        distance = (
            f"{row['exploration_distance_penalty_mean']:.2f}"
        )

    exploration = (
        f"{row['exploration_rate_mean'] * 100:.2f}\\%"
    )

    latex_lines.append(

        f"{epsilon_text} & "
        f"{exploration} & "
        f"{relative} & "
        f"{risk10} & "
        f"{risk20} & "
        f"{best} & "
        f"{rank} & "
        f"{distance} "
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
    r"\parbox{0.98\textwidth}{\footnotesize The counterfactual greedy reference is the highest cooldown-adjusted platform score among the eligible candidates available at the same request. A regret of zero means that exploration happened to select the same candidate that exploitation would have selected.}"
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

print("=" * 160)

print(
    f"Per seed   : "
    f"{OUTPUT_POR_SEED}"
)

print(
    f"Aggregated : "
    f"{OUTPUT_AGREGADO}"
)

print(
    f"LaTeX      : "
    f"{OUTPUT_LATEX}"
)


print("\n")
print("=" * 160)

print(
    "EXPLORATION RISK ANALYSIS COMPLETED"
)

print("=" * 160)
print()