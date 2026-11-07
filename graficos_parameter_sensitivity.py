# -*- coding: utf-8 -*-

"""
graficos_parameter_sensitivity.py

SHADAI — Parameter Sensitivity and Ablation
Visualização baseada em diferenças PAREADAS em relação ao baseline.

===============================================================================
FIGURA
===============================================================================

(a) Utility Weights lambda
    Mudança em relação ao baseline:
        - Request Allocation Rate
        - Overall Eligibility Rate

    Unidade:
        pontos percentuais (p.p.)

(b) Platform Score Weights w
    Heatmap da mudança em relação ao baseline:
        - Selected Normalized Reputation
        - Selected Normalized Energy
        - Selected Normalized Proximity

    Unidade:
        diferença absoluta na escala normalizada

(c) Exploration Sensitivity epsilon
    Dois gráficos empilhados:
        - mudança no Average Selected Score
        - mudança em Unique Selected Providers

===============================================================================
BASELINES
===============================================================================

lambda  = (0.5, 0.3, 0.2)
w       = (0.4, 0.3, 0.3)
epsilon = 0.20

===============================================================================
METODOLOGIA
===============================================================================

As mesmas seeds são utilizadas entre configurações.

Portanto, para cada seed:

    delta(seed) = metric(configuration, seed)
                  - metric(baseline, seed)

A figura apresenta:

    mean(delta)

e intervalo de confiança de 95% calculado sobre as diferenças pareadas.

Isso é preferível a comparar apenas médias independentes porque preserva o
pareamento experimental entre configurações.

===============================================================================
IMPORTANTE
===============================================================================

O raw score NÃO é comparado entre diferentes configurações de w, pois mudar
w1, w2 e w3 altera a própria definição do score.

No estudo de epsilon, o score permanece diretamente comparável porque lambda
e w permanecem fixos.

Nenhum CSV experimental é modificado.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import (
    LinearSegmentedColormap,
    TwoSlopeNorm,
)


# =============================================================================
# CONFIGURAÇÃO GERAL
# =============================================================================

EXPECTED_SEEDS = 30

OUTPUT_DIR = (
    Path("figures_final")
    /
    "sensitivity"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# INPUT FILES
# =============================================================================

LAMBDA_FILE = (
    Path("lambda_ablation")
    /
    "resumo_ablation_lambda_por_seed.csv"
)

WEIGHT_FILE = (
    Path("weight_ablation")
    /
    "resumo_ablation_weights_por_seed.csv"
)

EPSILON_FILE = (
    Path("epsilon_ablation")
    /
    "resumo_ablation_epsilon_por_seed.csv"
)


# =============================================================================
# BASELINES
# =============================================================================

BASELINE_LAMBDA_CONFIG = "original"

BASELINE_WEIGHT_CONFIG = "original"

BASELINE_EPSILON = 0.20


# =============================================================================
# CONFIGURAÇÕES — LAMBDA
# =============================================================================

LAMBDA_ORDER = [

    "original",

    "equal",

    "energy_high",

    "distance_high",

    "time_high",

    "no_energy",

    "no_distance",

    "no_time",

]


LAMBDA_ALTERNATIVES = [

    item

    for item
    in LAMBDA_ORDER

    if item != BASELINE_LAMBDA_CONFIG

]


LAMBDA_LABELS = {

    "original":
        "Original",

    "equal":
        "Equal",

    "energy_high":
        "Energy-high",

    "distance_high":
        "Distance-high",

    "time_high":
        "Time-high",

    "no_energy":
        "No energy",

    "no_distance":
        "No distance",

    "no_time":
        "No time",

}


# =============================================================================
# CONFIGURAÇÕES — SCORE WEIGHTS
# =============================================================================

WEIGHT_ORDER = [

    "original",

    "equal",

    "reputation_high",

    "energy_high",

    "proximity_high",

    "no_reputation",

    "no_energy",

    "no_proximity",

]


WEIGHT_ALTERNATIVES = [

    item

    for item
    in WEIGHT_ORDER

    if item != BASELINE_WEIGHT_CONFIG

]


WEIGHT_LABELS = {

    "original":
        "Original",

    "equal":
        "Equal",

    "reputation_high":
        "Reputation-high",

    "energy_high":
        "Energy-high",

    "proximity_high":
        "Proximity-high",

    "no_reputation":
        "No reputation",

    "no_energy":
        "No energy",

    "no_proximity":
        "No proximity",

}


# =============================================================================
# CORES
# =============================================================================

# Paleta visual utilizada nas demais figuras do artigo:
#
# Preto      -> métrica principal
# Azul       -> segunda métrica / efeitos negativos no heatmap
# Terracota  -> métrica contrastante / efeitos positivos no heatmap
# Cinza      -> referências e elementos auxiliares

BLACK = "#000000"

DARK_BLUE = "#243B5A"

TERRACOTTA = "#B23A1B"

# Mantemos o nome da variável para preservar exatamente
# a estrutura do código original, mas a cor agora é o
# mesmo azul escuro da paleta oficial.
DARK_GREEN = "#243B5A"

GRAY = "#666666"

LIGHT_GRID = "#DEDEDE"

WHITE = "#FFFFFF"


# Heatmap:
# negativo -> azul escuro
# zero     -> branco
# positivo -> terracota

DIVERGING_CMAP = LinearSegmentedColormap.from_list(

    "shadai_delta",

    [

        DARK_BLUE,

        "#F7F7F7",

        TERRACOTTA,

    ],

)


# =============================================================================
# COLUMN ALIASES
# =============================================================================

CONFIG_ALIASES = [

    "configuration",

    "config",

    "scenario",

    "label",

]


SEED_ALIASES = [

    "seed",

    "run",

    "replication",

]


# Lambda
ALLOCATION_ALIASES = [

    "request_allocation_rate",

    "allocation_rate",

    "alloc_rate",

]


ELIGIBILITY_ALIASES = [

    "overall_eligibility_rate",

    "eligibility_rate",

    "mean_eligibility_rate",

]


# Score weights
REPUTATION_NORM_ALIASES = [

    "avg_reputation_norm",

    "avg_selected_reputation_norm",

    "mean_reputation_norm",

    "reputation_norm",

]


ENERGY_NORM_ALIASES = [

    "avg_energy_norm",

    "avg_selected_energy_norm",

    "mean_energy_norm",

    "energy_norm",

]


PROXIMITY_NORM_ALIASES = [

    "avg_proximity_norm",

    "avg_selected_proximity_norm",

    "mean_proximity_norm",

    "proximity_norm",

]


# Epsilon
EPSILON_ALIASES = [

    "epsilon",

    "epsilon_value",

    "eps",

]


SCORE_ALIASES = [

    "avg_score_final",

    "avg_final_score",

    "avg_selected_score",

    "avg_score",

    "mean_score",

    "score",

]


UNIQUE_ALIASES = [

    "unique_vehicles",

    "unique_selected_vehicles",

    "unique_providers",

    "unique_selected_providers",

    "n_unique_vehicles",

]


# =============================================================================
# MATPLOTLIB
# =============================================================================

plt.rcParams.update({

    "font.family":
        "DejaVu Sans",

    "font.size":
        10.5,

    "axes.labelsize":
        11.5,

    "axes.titlesize":
        13.0,

    "xtick.labelsize":
        9.8,

    "ytick.labelsize":
        9.8,

    "legend.fontsize":
        9.5,

    "pdf.fonttype":
        42,

    "ps.fonttype":
        42,

})


# =============================================================================
# HELPERS
# =============================================================================

def require_file(path):

    if not path.exists():

        raise FileNotFoundError(

            "\nArquivo não encontrado:\n"
            f"  {path}\n"

        )


def normalize_column_name(value):

    return (

        str(value)

        .strip()

        .lower()

        .replace(" ", "_")

        .replace("-", "_")

    )


def find_column(
    df,
    aliases,
    description,
):

    lookup = {

        normalize_column_name(column):
            column

        for column
        in df.columns

    }


    for alias in aliases:

        key = normalize_column_name(
            alias
        )


        if key in lookup:

            return lookup[key]


    raise RuntimeError(

        f"\nNão encontrei a coluna para '{description}'.\n\n"

        f"Aliases testados:\n"
        f"  {aliases}\n\n"

        f"Colunas disponíveis:\n"
        f"  {list(df.columns)}\n"

    )


def numeric(series):

    return pd.to_numeric(

        series,

        errors="coerce",

    )


# =============================================================================
# RATE -> PERCENT
# =============================================================================

def convert_rate_to_percent(series):

    values = numeric(
        series
    )


    valid = values.dropna()


    if valid.empty:

        return values


    if valid.abs().max() <= 1.5:

        return (
            values
            *
            100.0
        )


    return values


# =============================================================================
# T CRITICAL
# =============================================================================

def t_critical_95(n):

    if n <= 1:

        return np.nan


    try:

        from scipy.stats import t

        return float(

            t.ppf(

                0.975,

                n - 1,

            )

        )


    except Exception:

        return 1.96


# =============================================================================
# SUMMARY
# =============================================================================

def mean_ci95(values):

    values = (

        pd.Series(
            values,
            dtype="float64",
        )

        .replace(

            [
                np.inf,
                -np.inf,
            ],

            np.nan,

        )

        .dropna()

    )


    n = len(
        values
    )


    if n == 0:

        return {

            "n":
                0,

            "mean":
                np.nan,

            "sd":
                np.nan,

            "ci95":
                np.nan,

            "ci95_low":
                np.nan,

            "ci95_high":
                np.nan,

        }


    mean = float(
        values.mean()
    )


    if n == 1:

        return {

            "n":
                n,

            "mean":
                mean,

            "sd":
                np.nan,

            "ci95":
                np.nan,

            "ci95_low":
                np.nan,

            "ci95_high":
                np.nan,

        }


    sd = float(

        values.std(
            ddof=1
        )

    )


    sem = (

        sd

        /

        math.sqrt(
            n
        )

    )


    critical = t_critical_95(
        n
    )


    ci95 = (

        critical
        *
        sem

    )


    return {

        "n":
            n,

        "mean":
            mean,

        "sd":
            sd,

        "ci95":
            ci95,

        "ci95_low":
            mean - ci95,

        "ci95_high":
            mean + ci95,

    }


# =============================================================================
# CONFIG VALIDATION
# =============================================================================

def validate_configuration_set(
    df,
    configuration_column,
    expected,
    study_name,
):

    observed = set(

        df[
            configuration_column
        ]

        .astype(str)

        .str.strip()

    )


    missing = [

        config

        for config
        in expected

        if config not in observed

    ]


    if missing:

        raise RuntimeError(

            f"\n{study_name}: "
            f"configurações ausentes:\n"
            f"  {missing}\n\n"

            f"Encontradas:\n"
            f"  {sorted(observed)}\n"

        )


# =============================================================================
# COLLAPSE ONE VALUE PER SEED
# =============================================================================

def collapse_seed_metric(
    df,
    seed_column,
    metric_column,
):

    work = pd.DataFrame({

        "seed":

            numeric(
                df[
                    seed_column
                ]
            ),


        "value":

            numeric(
                df[
                    metric_column
                ]
            ),

    })


    work = (

        work

        .replace(

            [
                np.inf,
                -np.inf,
            ],

            np.nan,

        )

        .dropna()

    )


    work[
        "seed"
    ] = (

        work[
            "seed"
        ]

        .astype(int)

    )


    result = (

        work

        .groupby(

            "seed",

            as_index=False,

        )

        .agg(

            value=(

                "value",

                "mean",

            ),

        )

    )


    return result


# =============================================================================
# PAIRED CATEGORY DELTA
# =============================================================================

def paired_category_delta(
    df,
    config_column,
    seed_column,
    metric_column,
    baseline_config,
    configurations,
    metric_name,
):

    baseline_df = df[

        df[
            config_column
        ]

        .astype(str)

        .str.strip()

        ==
        baseline_config

    ]


    baseline = collapse_seed_metric(

        baseline_df,

        seed_column,

        metric_column,

    )


    baseline = baseline.rename(

        columns={

            "value":
                "baseline_value",

        }

    )


    if len(baseline) != EXPECTED_SEEDS:

        raise RuntimeError(

            f"\nBaseline '{baseline_config}' "
            f"de {metric_name}: "
            f"{len(baseline)} seeds, "
            f"esperadas {EXPECTED_SEEDS}.\n"

        )


    rows = []


    for configuration in configurations:

        current_df = df[

            df[
                config_column
            ]

            .astype(str)

            .str.strip()

            ==
            configuration

        ]


        current = collapse_seed_metric(

            current_df,

            seed_column,

            metric_column,

        )


        current = current.rename(

            columns={

                "value":
                    "configuration_value",

            }

        )


        paired = pd.merge(

            baseline,

            current,

            on="seed",

            how="inner",

            validate="one_to_one",

        )


        if len(paired) != EXPECTED_SEEDS:

            raise RuntimeError(

                f"\n{metric_name}: "
                f"{configuration} possui "
                f"{len(paired)} seeds pareadas; "
                f"esperadas {EXPECTED_SEEDS}.\n"

            )


        paired[
            "delta"
        ] = (

            paired[
                "configuration_value"
            ]

            -
            paired[
                "baseline_value"
            ]

        )


        stats = mean_ci95(

            paired[
                "delta"
            ]

        )


        rows.append({

            "configuration":
                configuration,

            "metric":
                metric_name,

            **stats,

        })


    return pd.DataFrame(
        rows
    )


# =============================================================================
# EPSILON PAIRED DELTA
# =============================================================================

def paired_epsilon_delta(
    df,
    epsilon_column,
    seed_column,
    metric_column,
    baseline_epsilon,
    epsilon_values,
    metric_name,
):

    baseline_df = df[

        np.isclose(

            numeric(
                df[
                    epsilon_column
                ]
            ),

            baseline_epsilon,

        )

    ]


    baseline = collapse_seed_metric(

        baseline_df,

        seed_column,

        metric_column,

    )


    baseline = baseline.rename(

        columns={

            "value":
                "baseline_value",

        }

    )


    if len(baseline) != EXPECTED_SEEDS:

        raise RuntimeError(

            f"\nBaseline epsilon={baseline_epsilon}: "
            f"{len(baseline)} seeds encontradas, "
            f"esperadas {EXPECTED_SEEDS}.\n"

        )


    rows = []


    for epsilon in epsilon_values:

        if np.isclose(
            epsilon,
            baseline_epsilon,
        ):

            rows.append({

                "epsilon":
                    float(
                        epsilon
                    ),

                "metric":
                    metric_name,

                "n":
                    EXPECTED_SEEDS,

                "mean":
                    0.0,

                "sd":
                    0.0,

                "ci95":
                    0.0,

                "ci95_low":
                    0.0,

                "ci95_high":
                    0.0,

            })

            continue


        current_df = df[

            np.isclose(

                numeric(
                    df[
                        epsilon_column
                    ]
                ),

                epsilon,

            )

        ]


        current = collapse_seed_metric(

            current_df,

            seed_column,

            metric_column,

        )


        current = current.rename(

            columns={

                "value":
                    "configuration_value",

            }

        )


        paired = pd.merge(

            baseline,

            current,

            on="seed",

            how="inner",

            validate="one_to_one",

        )


        if len(paired) != EXPECTED_SEEDS:

            raise RuntimeError(

                f"\n{metric_name}, epsilon={epsilon}: "
                f"{len(paired)} seeds pareadas; "
                f"esperadas {EXPECTED_SEEDS}.\n"

            )


        paired[
            "delta"
        ] = (

            paired[
                "configuration_value"
            ]

            -
            paired[
                "baseline_value"
            ]

        )


        stats = mean_ci95(

            paired[
                "delta"
            ]

        )


        rows.append({

            "epsilon":
                float(
                    epsilon
                ),

            "metric":
                metric_name,

            **stats,

        })


    return pd.DataFrame(
        rows
    )


# =============================================================================
# LOAD LAMBDA
# =============================================================================

def load_lambda_results():

    require_file(
        LAMBDA_FILE
    )


    df = pd.read_csv(
        LAMBDA_FILE
    )


    config_column = find_column(

        df,

        CONFIG_ALIASES,

        "lambda configuration",

    )


    seed_column = find_column(

        df,

        SEED_ALIASES,

        "seed",

    )


    allocation_column = find_column(

        df,

        ALLOCATION_ALIASES,

        "request allocation rate",

    )


    eligibility_column = find_column(

        df,

        ELIGIBILITY_ALIASES,

        "overall eligibility rate",

    )


    df[
        config_column
    ] = (

        df[
            config_column
        ]

        .astype(str)

        .str.strip()

    )


    df[
        allocation_column
    ] = convert_rate_to_percent(

        df[
            allocation_column
        ]

    )


    df[
        eligibility_column
    ] = convert_rate_to_percent(

        df[
            eligibility_column
        ]

    )


    validate_configuration_set(

        df,

        config_column,

        LAMBDA_ORDER,

        "Lambda sensitivity",

    )


    allocation = paired_category_delta(

        df=
            df,

        config_column=
            config_column,

        seed_column=
            seed_column,

        metric_column=
            allocation_column,

        baseline_config=
            BASELINE_LAMBDA_CONFIG,

        configurations=
            LAMBDA_ALTERNATIVES,

        metric_name=
            "Allocation Rate",

    )


    eligibility = paired_category_delta(

        df=
            df,

        config_column=
            config_column,

        seed_column=
            seed_column,

        metric_column=
            eligibility_column,

        baseline_config=
            BASELINE_LAMBDA_CONFIG,

        configurations=
            LAMBDA_ALTERNATIVES,

        metric_name=
            "Eligibility Rate",

    )


    return {

        "allocation":
            allocation,

        "eligibility":
            eligibility,

    }


# =============================================================================
# LOAD WEIGHTS
# =============================================================================

def load_weight_results():

    require_file(
        WEIGHT_FILE
    )


    df = pd.read_csv(
        WEIGHT_FILE
    )


    config_column = find_column(

        df,

        CONFIG_ALIASES,

        "weight configuration",

    )


    seed_column = find_column(

        df,

        SEED_ALIASES,

        "seed",

    )


    reputation_column = find_column(

        df,

        REPUTATION_NORM_ALIASES,

        "normalized reputation",

    )


    energy_column = find_column(

        df,

        ENERGY_NORM_ALIASES,

        "normalized energy",

    )


    proximity_column = find_column(

        df,

        PROXIMITY_NORM_ALIASES,

        "normalized proximity",

    )


    df[
        config_column
    ] = (

        df[
            config_column
        ]

        .astype(str)

        .str.strip()

    )


    validate_configuration_set(

        df,

        config_column,

        WEIGHT_ORDER,

        "Score-weight sensitivity",

    )


    reputation = paired_category_delta(

        df=
            df,

        config_column=
            config_column,

        seed_column=
            seed_column,

        metric_column=
            reputation_column,

        baseline_config=
            BASELINE_WEIGHT_CONFIG,

        configurations=
            WEIGHT_ALTERNATIVES,

        metric_name=
            "Reputation",

    )


    energy = paired_category_delta(

        df=
            df,

        config_column=
            config_column,

        seed_column=
            seed_column,

        metric_column=
            energy_column,

        baseline_config=
            BASELINE_WEIGHT_CONFIG,

        configurations=
            WEIGHT_ALTERNATIVES,

        metric_name=
            "Energy",

    )


    proximity = paired_category_delta(

        df=
            df,

        config_column=
            config_column,

        seed_column=
            seed_column,

        metric_column=
            proximity_column,

        baseline_config=
            BASELINE_WEIGHT_CONFIG,

        configurations=
            WEIGHT_ALTERNATIVES,

        metric_name=
            "Proximity",

    )


    return {

        "reputation":
            reputation,

        "energy":
            energy,

        "proximity":
            proximity,

    }


# =============================================================================
# LOAD EPSILON
# =============================================================================

def load_epsilon_results():

    require_file(
        EPSILON_FILE
    )


    df = pd.read_csv(
        EPSILON_FILE
    )


    seed_column = find_column(

        df,

        SEED_ALIASES,

        "seed",

    )


    epsilon_column = find_column(

        df,

        EPSILON_ALIASES,

        "epsilon",

    )


    score_column = find_column(

        df,

        SCORE_ALIASES,

        "average selected score",

    )


    unique_column = find_column(

        df,

        UNIQUE_ALIASES,

        "unique selected providers",

    )


    df[
        epsilon_column
    ] = numeric(

        df[
            epsilon_column
        ]

    )


    df[
        score_column
    ] = numeric(

        df[
            score_column
        ]

    )


    df[
        unique_column
    ] = numeric(

        df[
            unique_column
        ]

    )


    epsilon_values = sorted(

        df[
            epsilon_column
        ]

        .dropna()

        .unique()

        .tolist()

    )


    if not any(

        np.isclose(
            epsilon_values,
            BASELINE_EPSILON,
        )

    ):

        raise RuntimeError(

            f"\nBaseline epsilon={BASELINE_EPSILON} "
            "não encontrado.\n"

        )


    score = paired_epsilon_delta(

        df=
            df,

        epsilon_column=
            epsilon_column,

        seed_column=
            seed_column,

        metric_column=
            score_column,

        baseline_epsilon=
            BASELINE_EPSILON,

        epsilon_values=
            epsilon_values,

        metric_name=
            "Selected Score",

    )


    unique = paired_epsilon_delta(

        df=
            df,

        epsilon_column=
            epsilon_column,

        seed_column=
            seed_column,

        metric_column=
            unique_column,

        baseline_epsilon=
            BASELINE_EPSILON,

        epsilon_values=
            epsilon_values,

        metric_name=
            "Unique Providers",

    )


    return {

        "score":
            score,

        "unique":
            unique,

    }


# =============================================================================
# AXIS STYLE
# =============================================================================

def style_axis(
    ax,
    grid_axis="both",
):

    ax.grid(

        True,

        axis=
            grid_axis,

        linestyle=
            "--",

        linewidth=
            0.7,

        color=
            LIGHT_GRID,

    )


    ax.set_axisbelow(
        True
    )


    for spine in ax.spines.values():

        spine.set_color(
            "#555555"
        )

        spine.set_linewidth(
            0.8
        )


# =============================================================================
# AUTO LIMIT INCLUDING ZERO
# =============================================================================

def set_delta_ylim(
    ax,
    means,
    cis,
    padding_fraction=0.12,
):

    means = np.asarray(
        means,
        dtype=float,
    )


    cis = np.asarray(
        cis,
        dtype=float,
    )


    lower = means - cis

    upper = means + cis


    minimum = min(
        0.0,
        float(
            np.nanmin(
                lower
            )
        ),
    )


    maximum = max(
        0.0,
        float(
            np.nanmax(
                upper
            )
        ),
    )


    span = maximum - minimum


    if span <= 0:

        span = 1.0


    padding = (
        span
        *
        padding_fraction
    )


    ax.set_ylim(

        minimum - padding,

        maximum + padding,

    )


# =============================================================================
# PANEL A — LAMBDA
# =============================================================================

def plot_lambda_panel(
    ax,
    results,
):

    allocation = results[
        "allocation"
    ]


    eligibility = results[
        "eligibility"
    ]


    y = np.arange(
        len(
            LAMBDA_ALTERNATIVES
        )
    )


    offset = 0.13


    ax.axvline(

        0.0,

        color=
            GRAY,

        linestyle=
            "--",

        linewidth=
            1.25,

        zorder=
            1,

    )


    # Allocation
    ax.errorbar(

        allocation[
            "mean"
        ],

        y - offset,

        xerr=
            allocation[
                "ci95"
            ],

        fmt=
            "o",

        markersize=
            6.2,

        color=
            BLACK,

        ecolor=
            BLACK,

        elinewidth=
            1.4,

        capsize=
            3,

        label=
            "Allocation Rate",

        zorder=
            3,

    )


    # Eligibility
    ax.errorbar(

        eligibility[
            "mean"
        ],

        y + offset,

        xerr=
            eligibility[
                "ci95"
            ],

        fmt=
            "s",

        markersize=
            5.8,

        color=
            TERRACOTTA,

        ecolor=
            TERRACOTTA,

        elinewidth=
            1.4,

        capsize=
            3,

        label=
            "Eligibility Rate",

        zorder=
            3,

    )


    ax.set_yticks(
        y
    )


    ax.set_yticklabels(

        [

            LAMBDA_LABELS[
                item
            ]

            for item
            in LAMBDA_ALTERNATIVES

        ]

    )


    ax.invert_yaxis()


    ax.set_xlabel(
        "Change from Baseline (percentage points)"
    )


    ax.set_title(
        r"Utility Weights $\lambda$"
    )


    ax.legend(

        loc=
            "lower right",

        frameon=
            True,

        framealpha=
            1.0,

        facecolor=
            "white",

        edgecolor=
            "#D0D0D0",

    )


    style_axis(
        ax,
        grid_axis="x",
    )


    all_lower = np.concatenate([

        (
            allocation[
                "mean"
            ]
            -
            allocation[
                "ci95"
            ]
        ).to_numpy(),

        (
            eligibility[
                "mean"
            ]
            -
            eligibility[
                "ci95"
            ]
        ).to_numpy(),

    ])


    all_upper = np.concatenate([

        (
            allocation[
                "mean"
            ]
            +
            allocation[
                "ci95"
            ]
        ).to_numpy(),

        (
            eligibility[
                "mean"
            ]
            +
            eligibility[
                "ci95"
            ]
        ).to_numpy(),

    ])


    maximum_abs = max(

        abs(
            np.nanmin(
                all_lower
            )
        ),

        abs(
            np.nanmax(
                all_upper
            )
        ),

    )


    maximum_abs *= 1.18


    ax.set_xlim(

        -maximum_abs,

        maximum_abs,

    )


    ax.text(

        0.015,

        0.985,

        "(a)",

        transform=
            ax.transAxes,

        ha=
            "left",

        va=
            "top",

        fontsize=
            12.5,

        fontweight=
            "bold",

    )


    ax.text(

        0.985,

        0.02,

        r"Reference: $\lambda=(0.5,0.3,0.2)$",

        transform=
            ax.transAxes,

        ha=
            "right",

        va=
            "bottom",

        fontsize=
            8.7,

        color=
            GRAY,

    )


# =============================================================================
# PANEL B — WEIGHT HEATMAP
# =============================================================================

def plot_weight_panel(
    fig,
    ax,
    results,
):

    metrics = [

        (
            "reputation",
            "Reputation",
        ),

        (
            "energy",
            "Energy",
        ),

        (
            "proximity",
            "Proximity",
        ),

    ]


    matrix = np.zeros(

        (

            len(
                WEIGHT_ALTERNATIVES
            ),

            len(
                metrics
            ),

        ),

        dtype=float,

    )


    for column_index, (
        result_key,
        metric_label,

    ) in enumerate(
        metrics
    ):

        frame = results[
            result_key
        ]


        lookup = {

            row[
                "configuration"
            ]:
                float(
                    row[
                        "mean"
                    ]
                )

            for _, row
            in frame.iterrows()

        }


        for row_index, configuration in enumerate(
            WEIGHT_ALTERNATIVES
        ):

            matrix[
                row_index,
                column_index
            ] = lookup[
                configuration
            ]


    maximum_abs = float(

        np.nanmax(

            np.abs(
                matrix
            )

        )

    )


    if maximum_abs <= 0:

        maximum_abs = 1.0


    norm = TwoSlopeNorm(

        vmin=
            -maximum_abs,

        vcenter=
            0.0,

        vmax=
            maximum_abs,

    )


    image = ax.imshow(

        matrix,

        cmap=
            DIVERGING_CMAP,

        norm=
            norm,

        aspect=
            "auto",

        interpolation=
            "nearest",

    )


    ax.set_xticks(

        np.arange(
            len(
                metrics
            )
        )

    )


    ax.set_xticklabels(

        [

            label

            for _,
            label
            in metrics

        ]

    )


    ax.set_yticks(

        np.arange(
            len(
                WEIGHT_ALTERNATIVES
            )
        )

    )


    ax.set_yticklabels(

        [

            WEIGHT_LABELS[
                item
            ]

            for item
            in WEIGHT_ALTERNATIVES

        ]

    )


    ax.set_xticks(

        np.arange(
            -0.5,
            len(metrics),
            1,
        ),

        minor=True,

    )


    ax.set_yticks(

        np.arange(
            -0.5,
            len(
                WEIGHT_ALTERNATIVES
            ),
            1,
        ),

        minor=True,

    )


    ax.grid(

        which=
            "minor",

        color=
            "white",

        linestyle=
            "-",

        linewidth=
            1.5,

    )


    ax.tick_params(

        which=
            "minor",

        bottom=
            False,

        left=
            False,

    )


    for row_index in range(
        matrix.shape[0]
    ):

        for column_index in range(
            matrix.shape[1]
        ):

            value = matrix[
                row_index,
                column_index
            ]


            relative_intensity = (

                abs(value)

                /
                maximum_abs

            )


            text_color = (

                WHITE

                if relative_intensity >= 0.58

                else BLACK

            )


            ax.text(

                column_index,

                row_index,

                f"{value:+.3f}",

                ha=
                    "center",

                va=
                    "center",

                color=
                    text_color,

                fontsize=
                    9.4,

                fontweight=
                    "bold",

            )


    ax.set_title(
        r"Platform Score Weights $w$"
    )


    ax.text(

        0.015,

        0.985,

        "(b)",

        transform=
            ax.transAxes,

        ha=
            "left",

        va=
            "top",

        fontsize=
            12.5,

        fontweight=
            "bold",

    )


    ax.text(

        0.985,

        -0.105,

        r"Reference: $w=(0.4,0.3,0.3)$",

        transform=
            ax.transAxes,

        ha=
            "right",

        va=
            "top",

        fontsize=
            8.7,

        color=
            GRAY,

    )


    colorbar = fig.colorbar(

        image,

        ax=
            ax,

        fraction=
            0.046,

        pad=
            0.035,

    )


    colorbar.set_label(

        "Mean Paired Change\nfrom Baseline",

        fontsize=
            9.5,

    )


# =============================================================================
# PANEL C1 — EPSILON SCORE
# =============================================================================

def plot_epsilon_score(
    ax,
    score,
):

    epsilon = (

        score[
            "epsilon"
        ]

        .to_numpy(
            dtype=float
        )

    )


    mean = (

        score[
            "mean"
        ]

        .to_numpy(
            dtype=float
        )

    )


    ci = (

        score[
            "ci95"
        ]

        .to_numpy(
            dtype=float
        )

    )


    ax.axhline(

        0.0,

        color=
            GRAY,

        linestyle=
            "--",

        linewidth=
            1.1,

        zorder=
            1,

    )


    ax.axvline(

        BASELINE_EPSILON,

        color=
            GRAY,

        linestyle=
            ":",

        linewidth=
            1.15,

        zorder=
            1,

    )


    ax.errorbar(

        epsilon,

        mean,

        yerr=
            ci,

        color=
            BLACK,

        marker=
            "o",

        markersize=
            5.5,

        linewidth=
            1.9,

        capsize=
            2.8,

        zorder=
            3,

    )


    ax.set_ylabel(
        r"$\Delta$ Selected Score"
    )


    ax.tick_params(

        axis=
            "x",

        labelbottom=
            False,

    )


    style_axis(
        ax,
        grid_axis="both",
    )


    set_delta_ylim(

        ax,

        mean,

        ci,

    )


# =============================================================================
# PANEL C2 — EPSILON UNIQUE PROVIDERS
# =============================================================================

def plot_epsilon_unique(
    ax,
    unique,
):

    epsilon = (

        unique[
            "epsilon"
        ]

        .to_numpy(
            dtype=float
        )

    )


    mean = (

        unique[
            "mean"
        ]

        .to_numpy(
            dtype=float
        )

    )


    ci = (

        unique[
            "ci95"
        ]

        .to_numpy(
            dtype=float
        )

    )


    ax.axhline(

        0.0,

        color=
            GRAY,

        linestyle=
            "--",

        linewidth=
            1.1,

        zorder=
            1,

    )


    ax.axvline(

        BASELINE_EPSILON,

        color=
            GRAY,

        linestyle=
            ":",

        linewidth=
            1.15,

        zorder=
            1,

    )


    ax.errorbar(

        epsilon,

        mean,

        yerr=
            ci,

        color=
            DARK_BLUE,

        marker=
            "s",

        markersize=
            5.3,

        linewidth=
            1.9,

        capsize=
            2.8,

        zorder=
            3,

    )


    ax.set_ylabel(
        r"$\Delta$ Unique Providers"
    )


    ax.set_xlabel(
        r"Exploration Factor $\epsilon$"
    )


    ax.set_xticks(
        epsilon
    )


    ax.set_xticklabels(

        [

            f"{value:g}"

            for value
            in epsilon

        ]

    )


    style_axis(
        ax,
        grid_axis="both",
    )


    set_delta_ylim(

        ax,

        mean,

        ci,

    )


# =============================================================================
# PANEL C — WRAPPER
# =============================================================================

def plot_epsilon_panel(
    ax_top,
    ax_bottom,
    results,
):

    plot_epsilon_score(

        ax_top,

        results[
            "score"
        ],

    )


    plot_epsilon_unique(

        ax_bottom,

        results[
            "unique"
        ],

    )


    ax_top.set_title(
        r"Exploration Sensitivity $\epsilon$"
    )


    ax_top.text(

        0.015,

        0.965,

        "(c)",

        transform=
            ax_top.transAxes,

        ha=
            "left",

        va=
            "top",

        fontsize=
            12.5,

        fontweight=
            "bold",

    )


    ax_top.text(

        BASELINE_EPSILON,

        0.95,

        " ref.",

        transform=
            ax_top.get_xaxis_transform(),

        ha=
            "left",

        va=
            "top",

        fontsize=
            8.5,

        color=
            GRAY,

    )


# =============================================================================
# EXPORT
# =============================================================================

def export_delta_data(
    lambda_results,
    weight_results,
    epsilon_results,
):

    rows = []


    for key in [
        "allocation",
        "eligibility",
    ]:

        frame = lambda_results[
            key
        ]


        for _, row in frame.iterrows():

            rows.append({

                "study":
                    "lambda",

                "configuration":
                    row[
                        "configuration"
                    ],

                "metric":
                    row[
                        "metric"
                    ],

                "baseline":
                    "original",

                "n_pairs":
                    row[
                        "n"
                    ],

                "mean_delta":
                    row[
                        "mean"
                    ],

                "ci95":
                    row[
                        "ci95"
                    ],

                "ci95_low":
                    row[
                        "ci95_low"
                    ],

                "ci95_high":
                    row[
                        "ci95_high"
                    ],

            })


    for key in [
        "reputation",
        "energy",
        "proximity",
    ]:

        frame = weight_results[
            key
        ]


        for _, row in frame.iterrows():

            rows.append({

                "study":
                    "weights",

                "configuration":
                    row[
                        "configuration"
                    ],

                "metric":
                    row[
                        "metric"
                    ],

                "baseline":
                    "original",

                "n_pairs":
                    row[
                        "n"
                    ],

                "mean_delta":
                    row[
                        "mean"
                    ],

                "ci95":
                    row[
                        "ci95"
                    ],

                "ci95_low":
                    row[
                        "ci95_low"
                    ],

                "ci95_high":
                    row[
                        "ci95_high"
                    ],

            })


    for key in [
        "score",
        "unique",
    ]:

        frame = epsilon_results[
            key
        ]


        for _, row in frame.iterrows():

            rows.append({

                "study":
                    "epsilon",

                "configuration":
                    float(
                        row[
                            "epsilon"
                        ]
                    ),

                "metric":
                    row[
                        "metric"
                    ],

                "baseline":
                    BASELINE_EPSILON,

                "n_pairs":
                    row[
                        "n"
                    ],

                "mean_delta":
                    row[
                        "mean"
                    ],

                "ci95":
                    row[
                        "ci95"
                    ],

                "ci95_low":
                    row[
                        "ci95_low"
                    ],

                "ci95_high":
                    row[
                        "ci95_high"
                    ],

            })


    output = pd.DataFrame(
        rows
    )


    path = (

        OUTPUT_DIR
        /
        "parameter_sensitivity_paired_deltas.csv"

    )


    output.to_csv(

        path,

        index=False,

    )


    print(

        f"[SAVE] {path}"

    )


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def print_diagnostics(
    lambda_results,
    weight_results,
    epsilon_results,
):

    print(
        "\n"
        +
        "=" * 110
    )

    print(
        "LAMBDA — PAIRED CHANGE FROM BASELINE"
    )

    print(
        "=" * 110
    )


    lambda_table = pd.DataFrame({

        "configuration":
            LAMBDA_ALTERNATIVES,

        "allocation_delta_pp":

            lambda_results[
                "allocation"
            ][
                "mean"
            ],

        "allocation_ci95":

            lambda_results[
                "allocation"
            ][
                "ci95"
            ],

        "eligibility_delta_pp":

            lambda_results[
                "eligibility"
            ][
                "mean"
            ],

        "eligibility_ci95":

            lambda_results[
                "eligibility"
            ][
                "ci95"
            ],

    })


    print(

        lambda_table

        .round(4)

        .to_string(
            index=False
        )

    )


    print(
        "\n"
        +
        "=" * 110
    )

    print(
        "SCORE WEIGHTS — PAIRED CHANGE FROM BASELINE"
    )

    print(
        "=" * 110
    )


    weight_table = pd.DataFrame({

        "configuration":
            WEIGHT_ALTERNATIVES,

        "delta_reputation":

            weight_results[
                "reputation"
            ][
                "mean"
            ],

        "delta_energy":

            weight_results[
                "energy"
            ][
                "mean"
            ],

        "delta_proximity":

            weight_results[
                "proximity"
            ][
                "mean"
            ],

    })


    print(

        weight_table

        .round(4)

        .to_string(
            index=False
        )

    )


    print(
        "\n"
        +
        "=" * 110
    )

    print(
        "EPSILON — PAIRED CHANGE FROM epsilon=0.20"
    )

    print(
        "=" * 110
    )


    epsilon_table = pd.DataFrame({

        "epsilon":

            epsilon_results[
                "score"
            ][
                "epsilon"
            ],

        "delta_score":

            epsilon_results[
                "score"
            ][
                "mean"
            ],

        "score_ci95":

            epsilon_results[
                "score"
            ][
                "ci95"
            ],

        "delta_unique_providers":

            epsilon_results[
                "unique"
            ][
                "mean"
            ],

        "unique_ci95":

            epsilon_results[
                "unique"
            ][
                "ci95"
            ],

    })


    print(

        epsilon_table

        .round(4)

        .to_string(
            index=False
        )

    )


# =============================================================================
# SAVE FIGURE
# =============================================================================

def save_figure(fig):

    base = (

        OUTPUT_DIR
        /
        "parameter_sensitivity_and_ablation"

    )


    png_path = base.with_suffix(
        ".png"
    )

    pdf_path = base.with_suffix(
        ".pdf"
    )

    eps_path = base.with_suffix(
        ".eps"
    )


    fig.savefig(

        png_path,

        dpi=400,

        bbox_inches="tight",

    )


    fig.savefig(

        pdf_path,

        bbox_inches="tight",

    )


    fig.savefig(

        eps_path,

        format="eps",

        bbox_inches="tight",

    )


    print(

        f"[SAVE] {png_path}"

    )

    print(

        f"[SAVE] {pdf_path}"

    )

    print(

        f"[SAVE] {eps_path}"

    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "\n"
        +
        "#" * 110
    )

    print(
        "SHADAI — PARAMETER SENSITIVITY AND ABLATION"
    )

    print(
        "PAIRED BASELINE-DELTA VISUALIZATION"
    )

    print(
        "#" * 110
    )


    print(
        "\nInput files:"
    )

    print(
        f"  Lambda  : {LAMBDA_FILE}"
    )

    print(
        f"  Weights : {WEIGHT_FILE}"
    )

    print(
        f"  Epsilon : {EPSILON_FILE}"
    )


    lambda_results = load_lambda_results()

    weight_results = load_weight_results()

    epsilon_results = load_epsilon_results()


    fig = plt.figure(

        figsize=(
            16.2,
            5.8,
        ),

        constrained_layout=True,

    )


    outer = fig.add_gridspec(

        nrows=1,

        ncols=3,

        width_ratios=[

            1.12,

            1.10,

            1.05,

        ],

    )


    ax_lambda = fig.add_subplot(
        outer[0, 0]
    )


    ax_weights = fig.add_subplot(
        outer[0, 1]
    )


    epsilon_grid = outer[
        0,
        2
    ].subgridspec(

        nrows=2,

        ncols=1,

        hspace=0.08,

    )


    ax_epsilon_score = fig.add_subplot(

        epsilon_grid[
            0,
            0
        ]

    )


    ax_epsilon_unique = fig.add_subplot(

        epsilon_grid[
            1,
            0
        ],

        sharex=
            ax_epsilon_score,

    )


    plot_lambda_panel(

        ax_lambda,

        lambda_results,

    )


    plot_weight_panel(

        fig,

        ax_weights,

        weight_results,

    )


    plot_epsilon_panel(

        ax_epsilon_score,

        ax_epsilon_unique,

        epsilon_results,

    )


    save_figure(
        fig
    )


    plt.close(
        fig
    )


    export_delta_data(

        lambda_results,

        weight_results,

        epsilon_results,

    )


    print_diagnostics(

        lambda_results,

        weight_results,

        epsilon_results,

    )


    print(
        "\n"
        +
        "#" * 110
    )

    print(
        "DONE"
    )

    print(
        "#" * 110
    )


    print(
        "\nGenerated files:"
    )


    print(

        "  figures_final/sensitivity/"
        "parameter_sensitivity_and_ablation.png"

    )


    print(

        "  figures_final/sensitivity/"
        "parameter_sensitivity_and_ablation.pdf"

    )


    print(

        "  figures_final/sensitivity/"
        "parameter_sensitivity_and_ablation.eps"

    )


    print(

        "  figures_final/sensitivity/"
        "parameter_sensitivity_paired_deltas.csv"

    )


if __name__ == "__main__":

    main()