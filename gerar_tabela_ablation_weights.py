# gerar_tabela_ablation_weights.py

import os
import pandas as pd


# ============================================================
# DIRETÓRIO
# ============================================================

WEIGHT_DIR = "weight_ablation"


os.makedirs(
    WEIGHT_DIR,
    exist_ok=True
)


# ============================================================
# ARQUIVOS
# ============================================================

INPUT_FILE = os.path.join(
    WEIGHT_DIR,
    "resumo_ablation_weights_multiseed.csv"
)


OUTPUT_CSV = os.path.join(
    WEIGHT_DIR,
    "tabela_ablation_weights_tnsm.csv"
)


OUTPUT_LATEX = os.path.join(
    WEIGHT_DIR,
    "tabela_ablation_weights_tnsm.tex"
)


# ============================================================
# VERIFICAÇÃO
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
# CONFIGURAÇÕES
# ============================================================

ORDEM = [

    "original",

    "equal",

    "reputation_high",

    "energy_high",

    "proximity_high",

    "no_reputation",

    "no_energy",

    "no_proximity"
]


LABELS = {

    "original":
        "Baseline",

    "equal":
        "Equal",

    "reputation_high":
        "Reputation-high",

    "energy_high":
        "Energy-high",

    "proximity_high":
        "Proximity-high",

    "no_reputation":
        "No Reputation",

    "no_energy":
        "No Energy",

    "no_proximity":
        "No Proximity"
}


# ============================================================
# FUNÇÕES
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


def format_weight(
    value
):

    return f"{value:.4f}"


# ============================================================
# VALIDAÇÃO DAS COLUNAS
# ============================================================

COLUNAS_NECESSARIAS = [

    "configuration",

    "w1",
    "w2",
    "w3",

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
    "avg_energy_ci95_high"
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
# MONTA TABELA
# ============================================================

rows = []


for config in ORDEM:

    resultado = df[
        df[
            "configuration"
        ] == config
    ]


    if resultado.empty:

        print(
            f"[WARNING] "
            f"Configuration not found: {config}"
        )

        continue


    row = resultado.iloc[0]


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
    # DISTÂNCIA REAL
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
    # ENERGIA RESIDUAL REAL
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
    # LINHA
    # ========================================================

    rows.append({

        "Configuration":
            LABELS[
                config
            ],

        "w1":
            row[
                "w1"
            ],

        "w2":
            row[
                "w2"
            ],

        "w3":
            row[
                "w3"
            ],

        "Reputation Norm":
            reputation,

        "Energy Norm":
            energy_norm,

        "Proximity Norm":
            proximity,

        "Distance":
            distance,

        "Residual Energy":
            residual_energy
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


print("\n")

print(
    "=" * 180
)


print(
    "SHADAI - FINAL WEIGHT SENSITIVITY / ABLATION TABLE"
)


print(
    "=" * 180
)


print(
    table.to_string(
        index=False
    )
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
    r"\caption{Sensitivity and ablation analysis of the SHADAI platform score weights over 30 independent simulation seeds. Values are reported as mean and 95\% confidence interval.}"
)


latex_lines.append(
    r"\label{tab:weight_ablation}"
)


latex_lines.append(
    r"\resizebox{\textwidth}{!}{%"
)


latex_lines.append(
    r"\begin{tabular}{lccc|ccccc}"
)


latex_lines.append(
    r"\hline"
)


latex_lines.append(
    r"Configuration & $w_1$ & $w_2$ & $w_3$ & $\hat{R}$ & $\hat{E}$ & $\hat{P}$ & Distance & Residual Energy \\"
)


latex_lines.append(
    r"\hline"
)


for _, row in table.iterrows():

    line = (

        f"{row['Configuration']} & "

        f"{format_weight(row['w1'])} & "

        f"{format_weight(row['w2'])} & "

        f"{format_weight(row['w3'])} & "

        f"{row['Reputation Norm']} & "

        f"{row['Energy Norm']} & "

        f"{row['Proximity Norm']} & "

        f"{row['Distance']} & "

        f"{row['Residual Energy']} "

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
    r"$w_1$, $w_2$, and $w_3$ respectively weight reputation, residual energy, and proximity. $\hat{R}$, $\hat{E}$, and $\hat{P}$ denote the normalized attributes of the selected vehicles. Eligibility and request allocation rates are omitted because they remained unchanged across the evaluated score-weight configurations; the score weights operate only after candidate eligibility is established."
)


latex_lines.append(
    r"\end{minipage}"
)


latex_lines.append(
    r"\end{table*}"
)


# ============================================================
# ESCREVE LATEX
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
# VERIFICAÇÃO DA TABELA
# ============================================================

print("\n")

print(
    "=" * 180
)


print(
    "TABLE VALIDATION"
)


print(
    "=" * 180
)


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
        "[OK] All weight configurations are present."
    )

else:

    print(
        "[WARNING] Some weight configurations are missing."
    )


# ============================================================
# RESULTADOS DE REFERÊNCIA
# ============================================================

baseline = table[
    table[
        "Configuration"
    ] == "Baseline"
]


if not baseline.empty:

    print(
        "[OK] Baseline configuration found."
    )

else:

    print(
        "[WARNING] Baseline configuration not found."
    )


# ============================================================
# ARQUIVOS
# ============================================================

print("\n")

print(
    "=" * 180
)


print(
    "OUTPUT FILES"
)


print(
    "=" * 180
)


print(
    f"CSV   : {OUTPUT_CSV}"
)


print(
    f"LaTeX : {OUTPUT_LATEX}"
)


print("\n")

print(
    "Table generation completed."
)

print()