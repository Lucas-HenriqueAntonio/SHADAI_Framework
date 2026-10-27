# gerar_tabela_ablation_lambda.py

import os
import pandas as pd


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
# ARQUIVOS
# ============================================================

INPUT_FILE = os.path.join(
    LAMBDA_DIR,
    "resumo_ablation_lambda_multiseed.csv"
)


OUTPUT_CSV = os.path.join(
    LAMBDA_DIR,
    "tabela_ablation_lambda_tnsm.csv"
)


OUTPUT_LATEX = os.path.join(
    LAMBDA_DIR,
    "tabela_ablation_lambda_tnsm.tex"
)


# ============================================================
# VERIFICAÇÃO
# ============================================================

if not os.path.exists(
    INPUT_FILE
):

    raise FileNotFoundError(
        f"Arquivo não encontrado: "
        f"{INPUT_FILE}"
    )


# ============================================================
# CARREGA
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)


# ============================================================
# ORDEM
# ============================================================

ORDEM = [

    "original",

    "equal",

    "energy_high",

    "distance_high",

    "time_high",

    "no_energy",

    "no_distance",

    "no_time"
]


LABELS = {

    "original":
        "Baseline",

    "equal":
        "Equal",

    "energy_high":
        "Energy-high",

    "distance_high":
        "Distance-high",

    "time_high":
        "Time-high",

    "no_energy":
        "No Energy",

    "no_distance":
        "No Distance",

    "no_time":
        "No Time"
}


# ============================================================
# FORMATADOR
# ============================================================

def format_mean_ci(
    mean,
    low,
    high,
    percent=False,
    decimals=2
):

    if percent:

        mean *= 100
        low *= 100
        high *= 100


    return (
        f"{mean:.{decimals}f} "
        f"[{low:.{decimals}f}, "
        f"{high:.{decimals}f}]"
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
            f"Missing: {config}"
        )

        continue


    row = resultado.iloc[0]


    allocation = format_mean_ci(

        row[
            "request_allocation_rate_mean"
        ],

        row[
            "request_allocation_rate_ci95_low"
        ],

        row[
            "request_allocation_rate_ci95_high"
        ],

        percent=True
    )


    eligibility = format_mean_ci(

        row[
            "overall_eligibility_rate_mean"
        ],

        row[
            "overall_eligibility_rate_ci95_low"
        ],

        row[
            "overall_eligibility_rate_ci95_high"
        ],

        percent=True
    )


    score = format_mean_ci(

        row[
            "avg_score_mean"
        ],

        row[
            "avg_score_ci95_low"
        ],

        row[
            "avg_score_ci95_high"
        ],

        decimals=4
    )


    energy = format_mean_ci(

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


    rows.append({

        "Configuration":
            LABELS[
                config
            ],

        "lambda1":
            row[
                "lambda1"
            ],

        "lambda2":
            row[
                "lambda2"
            ],

        "lambda3":
            row[
                "lambda3"
            ],

        "Allocation Rate (%)":
            allocation,

        "Eligibility Rate (%)":
            eligibility,

        "Platform Score":
            score,

        "Residual Energy":
            energy
    })


table = pd.DataFrame(
    rows
)


# ============================================================
# CSV
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
    250
)


print("\n")
print("=" * 155)

print(
    "SHADAI - FINAL LAMBDA ABLATION TABLE"
)

print("=" * 155)


print(
    table.to_string(
        index=False
    )
)


# ============================================================
# LATEX
# ============================================================

latex_lines = [

    r"\begin{table*}[t]",

    r"\centering",

    r"\caption{Sensitivity and ablation analysis of the vehicle utility weights over 30 independent simulation seeds. Values are reported as mean and 95\% confidence interval.}",

    r"\label{tab:lambda_ablation}",

    r"\resizebox{\textwidth}{!}{%",

    r"\begin{tabular}{lccc|cccc}",

    r"\hline",

    r"Configuration & $\lambda_1$ & $\lambda_2$ & $\lambda_3$ & Allocation Rate (\%) & Eligibility Rate (\%) & Platform Score & Residual Energy \\",

    r"\hline"
]


for _, row in table.iterrows():

    latex_lines.append(

        f"{row['Configuration']} & "
        f"{row['lambda1']:.4f} & "
        f"{row['lambda2']:.4f} & "
        f"{row['lambda3']:.4f} & "
        f"{row['Allocation Rate (%)']} & "
        f"{row['Eligibility Rate (%)']} & "
        f"{row['Platform Score']} & "
        f"{row['Residual Energy']} "
        r"\\"
    )


latex_lines.extend([

    r"\hline",

    r"\end{tabular}%",

    r"}",

    r"\end{table*}"
])


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
# FINAL
# ============================================================

print("\n")

print(
    f"CSV   : {OUTPUT_CSV}"
)

print(
    f"LaTeX : {OUTPUT_LATEX}"
)

print()