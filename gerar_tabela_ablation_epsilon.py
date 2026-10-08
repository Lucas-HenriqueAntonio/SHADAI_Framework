# gerar_tabela_ablation_epsilon.py

import os
import pandas as pd


# ============================================================
# DIRETÓRIO
# ============================================================

EPSILON_DIR = "epsilon_ablation"

os.makedirs(
    EPSILON_DIR,
    exist_ok=True
)


# ============================================================
# ARQUIVOS
# ============================================================

INPUT_FILE = os.path.join(
    EPSILON_DIR,
    "resumo_ablation_epsilon_multiseed.csv"
)

OUTPUT_CSV = os.path.join(
    EPSILON_DIR,
    "tabela_ablation_epsilon_tnsm.csv"
)

OUTPUT_LATEX = os.path.join(
    EPSILON_DIR,
    "tabela_ablation_epsilon_tnsm.tex"
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ORDEM = [

    "epsilon_000",

    "epsilon_010",

    "epsilon_020",

    "epsilon_030",

    "epsilon_050",

    "epsilon_100"
]


LABELS = {

    "epsilon_000":
        r"$\epsilon=0.00$",

    "epsilon_010":
        r"$\epsilon=0.10$",

    "epsilon_020":
        r"$\epsilon=0.20$",

    "epsilon_030":
        r"$\epsilon=0.30$",

    "epsilon_050":
        r"$\epsilon=0.50$",

    "epsilon_100":
        r"$\epsilon=1.00$"
}


# ============================================================
# VERIFICA INPUT
# ============================================================

if not os.path.exists(
    INPUT_FILE
):

    raise FileNotFoundError(
        f"Arquivo não encontrado: {INPUT_FILE}"
    )


# ============================================================
# LEITURA
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)


# ============================================================
# COLUNAS OBRIGATÓRIAS
# ============================================================

COLUNAS_NECESSARIAS = [

    "configuration",

    "epsilon",

    "n_runs",

    "exploration_rate_mean",
    "exploration_rate_ci95_low",
    "exploration_rate_ci95_high",

    "request_allocation_rate_mean",
    "request_allocation_rate_ci95_low",
    "request_allocation_rate_ci95_high",

    "avg_score_final_mean",
    "avg_score_final_ci95_low",
    "avg_score_final_ci95_high",

    "avg_reputation_norm_mean",
    "avg_reputation_norm_ci95_low",
    "avg_reputation_norm_ci95_high",

    "avg_energy_norm_mean",
    "avg_energy_norm_ci95_low",
    "avg_energy_norm_ci95_high",

    "avg_proximity_norm_mean",
    "avg_proximity_norm_ci95_low",
    "avg_proximity_norm_ci95_high",

    "avg_distance_mean",
    "avg_distance_ci95_low",
    "avg_distance_ci95_high",

    "avg_energy_mean",
    "avg_energy_ci95_low",
    "avg_energy_ci95_high",

    "unique_vehicles_mean",
    "unique_vehicles_ci95_low",
    "unique_vehicles_ci95_high"
]


faltantes = [

    coluna

    for coluna in COLUNAS_NECESSARIAS

    if coluna not in df.columns
]


if faltantes:

    raise ValueError(
        "As seguintes colunas necessárias não existem "
        f"no arquivo de entrada: {faltantes}"
    )


# ============================================================
# FUNÇÕES DE FORMATAÇÃO
# ============================================================

def format_mean_ci(
    mean,
    low,
    high,
    decimals=4
):

    return (
        f"{mean:.{decimals}f} "
        f"[{low:.{decimals}f}, "
        f"{high:.{decimals}f}]"
    )


def format_percent_ci(
    mean,
    low,
    high,
    decimals=2
):

    return (
        f"{mean * 100:.{decimals}f}\\% "
        f"[{low * 100:.{decimals}f}\\%, "
        f"{high * 100:.{decimals}f}\\%]"
    )


def format_float(
    value,
    decimals=2
):

    return f"{value:.{decimals}f}"


# ============================================================
# MONTA TABELA
# ============================================================

rows = []


for config in ORDEM:

    resultado = df[
        df[
            "configuration"
        ]
        ==
        config
    ]


    if resultado.empty:

        print(
            f"[WARNING] "
            f"Configuration not found: "
            f"{config}"
        )

        continue


    row = resultado.iloc[
        0
    ]


    # ========================================================
    # EXPLORATION
    # ========================================================

    exploration = format_percent_ci(

        row[
            "exploration_rate_mean"
        ],

        row[
            "exploration_rate_ci95_low"
        ],

        row[
            "exploration_rate_ci95_high"
        ],

        decimals=2
    )


    # ========================================================
    # ALLOCATION
    # ========================================================

    allocation = format_percent_ci(

        row[
            "request_allocation_rate_mean"
        ],

        row[
            "request_allocation_rate_ci95_low"
        ],

        row[
            "request_allocation_rate_ci95_high"
        ],

        decimals=2
    )


    # ========================================================
    # SCORE
    # ========================================================

    score = format_mean_ci(

        row[
            "avg_score_final_mean"
        ],

        row[
            "avg_score_final_ci95_low"
        ],

        row[
            "avg_score_final_ci95_high"
        ],

        decimals=4
    )


    # ========================================================
    # REPUTAÇÃO NORMALIZADA
    # ========================================================

    reputation = format_mean_ci(

        row[
            "avg_reputation_norm_mean"
        ],

        row[
            "avg_reputation_norm_ci95_low"
        ],

        row[
            "avg_reputation_norm_ci95_high"
        ],

        decimals=4
    )


    # ========================================================
    # ENERGIA NORMALIZADA
    # ========================================================

    energy_norm = format_mean_ci(

        row[
            "avg_energy_norm_mean"
        ],

        row[
            "avg_energy_norm_ci95_low"
        ],

        row[
            "avg_energy_norm_ci95_high"
        ],

        decimals=4
    )


    # ========================================================
    # PROXIMIDADE NORMALIZADA
    # ========================================================

    proximity = format_mean_ci(

        row[
            "avg_proximity_norm_mean"
        ],

        row[
            "avg_proximity_norm_ci95_low"
        ],

        row[
            "avg_proximity_norm_ci95_high"
        ],

        decimals=4
    )


    # ========================================================
    # DISTÂNCIA
    # ========================================================

    distance = format_mean_ci(

        row[
            "avg_distance_mean"
        ],

        row[
            "avg_distance_ci95_low"
        ],

        row[
            "avg_distance_ci95_high"
        ],

        decimals=2
    )


    # ========================================================
    # ENERGIA RESIDUAL
    # ========================================================

    residual_energy = format_mean_ci(

        row[
            "avg_energy_mean"
        ],

        row[
            "avg_energy_ci95_low"
        ],

        row[
            "avg_energy_ci95_high"
        ],

        decimals=2
    )


    # ========================================================
    # VEÍCULOS ÚNICOS
    # ========================================================

    unique_vehicles = format_mean_ci(

        row[
            "unique_vehicles_mean"
        ],

        row[
            "unique_vehicles_ci95_low"
        ],

        row[
            "unique_vehicles_ci95_high"
        ],

        decimals=2
    )


    # ========================================================
    # ADICIONA LINHA
    # ========================================================

    rows.append({

        "Configuration":
            config,

        "Epsilon":
            row[
                "epsilon"
            ],

        "Runs":
            int(
                row[
                    "n_runs"
                ]
            ),

        "Exploration":
            exploration,

        "Allocation":
            allocation,

        "Score":
            score,

        "Reputation Norm":
            reputation,

        "Energy Norm":
            energy_norm,

        "Proximity Norm":
            proximity,

        "Distance":
            distance,

        "Residual Energy":
            residual_energy,

        "Unique Vehicles":
            unique_vehicles
    })


table = pd.DataFrame(
    rows
)


# ============================================================
# SALVA CSV
# ============================================================

table.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# EXIBE NO TERMINAL
# ============================================================

pd.set_option(
    "display.width",
    400
)

pd.set_option(
    "display.max_columns",
    None
)


print("\n")

print("=" * 220)

print(
    "SHADAI - FINAL EPSILON SENSITIVITY TABLE"
)

print("=" * 220)


print(
    table.to_string(
        index=False
    )
)


# ============================================================
# MONTA LATEX
# ============================================================

latex_lines = []


latex_lines.append(
    r"\begin{table*}[t]"
)


latex_lines.append(
    r"\centering"
)


latex_lines.append(
    r"\caption{Sensitivity analysis of the SHADAI $\epsilon$-greedy exploration parameter over 30 simulation seeds. Values are reported as mean and 95\% confidence interval.}"
)


latex_lines.append(
    r"\label{tab:epsilon_sensitivity}"
)


latex_lines.append(
    r"\resizebox{\textwidth}{!}{%"
)


latex_lines.append(
    r"\begin{tabular}{c|cccccccc}"
)


latex_lines.append(
    r"\hline"
)


latex_lines.append(
    r"$\epsilon$ & Exploration & Score & $\hat{R}$ & $\hat{E}$ & $\hat{P}$ & Distance & Residual Energy & Unique Vehicles \\"
)


latex_lines.append(
    r"\hline"
)


for _, row in table.iterrows():

    config = row[
        "Configuration"
    ]


    epsilon_label = LABELS[
        config
    ]


    # Destaca baseline epsilon=0.20
    if config == "epsilon_020":

        epsilon_label = (
            r"\textbf{$\epsilon=0.20$}"
        )


    line = (

        f"{epsilon_label} & "

        f"{row['Exploration']} & "

        f"{row['Score']} & "

        f"{row['Reputation Norm']} & "

        f"{row['Energy Norm']} & "

        f"{row['Proximity Norm']} & "

        f"{row['Distance']} & "

        f"{row['Residual Energy']} & "

        f"{row['Unique Vehicles']} "

        r"\\"
    )


    latex_lines.append(
        line
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
    r"\begin{minipage}{0.98\textwidth}"
)


latex_lines.append(
    r"\footnotesize"
)


latex_lines.append(
    r"$\hat{R}$, $\hat{E}$, and $\hat{P}$ denote the normalized reputation, residual-energy, and proximity attributes of the selected vehicles, respectively. The baseline SHADAI configuration uses $\epsilon=0.20$. Increasing $\epsilon$ increases random exploration among eligible candidates, whereas $\epsilon=0$ corresponds to fully greedy exploitation and $\epsilon=1$ corresponds to fully exploratory selection. All experiments use the same $\lambda=(0.5,0.3,0.2)$ and score weights $(w_1,w_2,w_3)=(0.4,0.3,0.3)$."
)


latex_lines.append(
    r"\end{minipage}"
)


latex_lines.append(
    r"\end{table*}"
)


# ============================================================
# SALVA LATEX
# ============================================================

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
# VALIDAÇÃO
# ============================================================

print("\n")

print("=" * 220)

print(
    "TABLE VALIDATION"
)

print("=" * 220)


print(
    f"Configurations expected : "
    f"{len(ORDEM)}"
)


print(
    f"Configurations included : "
    f"{len(table)}"
)


if len(
    table
) == len(
    ORDEM
):

    print(
        "[OK] All epsilon configurations are present."
    )

else:

    print(
        "[WARNING] Some epsilon configurations are missing."
    )


# ============================================================
# BASELINE
# ============================================================

baseline = table[
    table[
        "Configuration"
    ]
    ==
    "epsilon_020"
]


if not baseline.empty:

    print(
        "[OK] Baseline epsilon=0.20 found."
    )

else:

    print(
        "[WARNING] Baseline epsilon=0.20 not found."
    )


# ============================================================
# EXTREMOS
# ============================================================

epsilon_zero = table[
    table[
        "Configuration"
    ]
    ==
    "epsilon_000"
]


epsilon_one = table[
    table[
        "Configuration"
    ]
    ==
    "epsilon_100"
]


if (
    not epsilon_zero.empty
    and
    not epsilon_one.empty
):

    print(
        "[OK] Extreme configurations epsilon=0 "
        "and epsilon=1 are present."
    )

else:

    print(
        "[WARNING] Extreme epsilon configurations "
        "are incomplete."
    )


# ============================================================
# NÚMERO DE RUNS
# ============================================================

if (
    table[
        "Runs"
    ]
    ==
    30
).all():

    print(
        "[OK] All configurations contain 30 runs."
    )

else:

    print(
        "[WARNING] At least one configuration "
        "does not contain 30 runs."
    )


# ============================================================
# OUTPUTS
# ============================================================

print("\n")

print("=" * 220)

print(
    "OUTPUT FILES"
)

print("=" * 220)


print(
    f"CSV   : "
    f"{OUTPUT_CSV}"
)


print(
    f"LaTeX : "
    f"{OUTPUT_LATEX}"
)


print("\n")

print(
    "Table generation completed successfully."
)

print()