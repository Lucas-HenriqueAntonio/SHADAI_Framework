# -*- coding: utf-8 -*-

"""
graficos_estrategias.py

Comparação interna entre estratégias/ablations da campanha final SHADAI.

FIGURAS
===============================================================================

1. Reputation Comparison
   - SHADAI
   - Greedy
   - No Reputation

2. Score Comparison
   - SHADAI
   - Greedy
   - No Reputation

3. Utility Comparison
   - SHADAI
   - Greedy
   - No Reputation

4. Fairness and Provider Diversity Comparison
   Painel esquerdo:
   - Cumulative Jain Fairness
   Painel direito:
   - Cumulative Unique Selected Providers

   Estratégias:
   - SHADAI
   - Greedy
   - No Reputation


METODOLOGIA
===============================================================================

Cenário:
- 400 Providers
- 30 seeds
- rng_scheme = split_v2
- ataques desabilitados

Métricas temporais:
- janela base: 50 simulation steps
- centered moving average: 3 janelas = 150 steps
- suavização separada por seed
- agregação entre seeds somente depois da suavização
- warm-up e drain-out definidos por active_vehicles
- comparação restrita ao intervalo operacional comum

Fairness:
- utiliza a coluna "veiculo" dos CSVs metricas_*.csv
- para cada seed, mantém a contagem acumulada de tarefas por provedor
- em cada janela temporal calcula:

                  (sum x_i)^2
      J = ------------------------------
            n * sum(x_i^2)

  onde x_i é o número acumulado de seleções do provedor i.

Provider Diversity:
- utiliza exatamente as mesmas seleções empregadas no cálculo de fairness
- mantém o conjunto acumulado de provedores selecionados por seed
- em cada janela temporal contabiliza o número acumulado de provedores únicos
- aplica o mesmo smoothing temporal de 150 steps

Nenhum CSV, analyzer ou resultado de simulação é alterado.
"""

from pathlib import Path
import glob
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# OUTPUT
# =============================================================================

OUTPUT_DIR = Path(
    "figures_estrategias"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# EXPERIMENTO
# =============================================================================

EXPECTED_SEEDS = set(
    range(1, 31)
)

EXPECTED_RNG = (
    "split_v2"
)

PROVIDER_SCENARIO = 400


# =============================================================================
# TEMPORAL
# =============================================================================

WINDOW_STEPS = 50

SMOOTHING_WINDOWS = 3


# =============================================================================
# COBERTURA
# =============================================================================

MIN_SEED_COVERAGE = 0.80

MIN_OBSERVATION_SUPPORT_RATIO = 0.50


# =============================================================================
# INTERVALO OPERACIONAL
# =============================================================================

ACTIVE_MIN_SEED_COVERAGE = 0.80

ACTIVE_REFERENCE_QUANTILE = 0.75

ACTIVE_OPERATIONAL_FRACTION = 0.50

ACTIVE_CONSECUTIVE_WINDOWS = 3


# =============================================================================
# IC95%
# =============================================================================

SHOW_REPUTATION_CI = False

SHOW_SCORE_CI = False

SHOW_UTILITY_CI = False

SHOW_FAIRNESS_CI = False

SHOW_DIVERSITY_CI = False


# =============================================================================
# LIMITES Y
# =============================================================================

REPUTATION_Y_MIN = 0.73
REPUTATION_Y_MAX = 0.95

SCORE_Y_MIN = 0.84
SCORE_Y_MAX = 0.93

UTILITY_Y_MIN = 0.125
UTILITY_Y_MAX = 0.182

FAIRNESS_Y_MIN = 0.875
FAIRNESS_Y_MAX = 1.005


# =============================================================================
# PALETA OFICIAL
# =============================================================================

BLACK = "#000000"

TERRACOTTA = "#B23A1B"

DARK_BLUE = "#243B5A"

DARK_GREEN = "#2F5D50"

GRAY = "#6B6B6B"

LIGHT_GRID = "#DEDEDE"


# =============================================================================
# ESPESSURAS
# =============================================================================

LW_MAIN = 2.4

LW_SECONDARY = 2.0

LW_DOTTED = 2.2


# =============================================================================
# ESTRATÉGIAS
# =============================================================================

STRATEGIES = {

    "SHADAI": {

        "label":
            "SHADAI",

        "metric_pattern":
            "scalability_final/"
            "metricas_shadai_400_scalability_v400_seed_*.csv",

        "request_pattern":
            "scalability_final/"
            "requests_shadai_400_scalability_v400_seed_*.csv",

        "color":
            BLACK,

        "linestyle":
            "-",

        "linewidth":
            LW_MAIN,

    },


    "GREEDY": {

        "label":
            "Greedy",

        "metric_pattern":
            "greedy_alignment/"
            "metricas_shadai_400_c22_greedy_seed_*.csv",

        "request_pattern":
            "greedy_alignment/"
            "requests_shadai_400_c22_greedy_seed_*.csv",

        "color":
            DARK_BLUE,

        "linestyle":
            ":",

        "linewidth":
            LW_DOTTED,

    },


    "NOREP": {

        "label":
            "No Reputation",

        "metric_pattern":
            "norep_alignment/"
            "metricas_shadai_400_c23_norep_seed_*.csv",

        "request_pattern":
            "norep_alignment/"
            "requests_shadai_400_c23_norep_seed_*.csv",

        "color":
            DARK_GREEN,

        "linestyle":
            "-.",

        "linewidth":
            LW_SECONDARY,

    },

}


STRATEGY_ORDER = [

    "SHADAI",

    "GREEDY",

    "NOREP",

]


# =============================================================================
# ALIASES
# =============================================================================

REPUTATION_NORMALIZED_ALIASES = [

    "reputacao_norm",

    "reputation_norm",

    "normalized_reputation",

    "reputation_normalized",

]


REPUTATION_RAW_ALIASES = [

    "reputacao",

    "reputation",

    "provider_reputation",

    "selected_reputation",

]


SCORE_ALIASES = [

    "score",

    "provider_score",

    "selected_score",

]


UTILITY_ALIASES = [

    "utilidade",

    "utility",

    "utility_norm",

    "normalized_utility",

    "utilidade_norm",

]


PROVIDER_ALIASES = [

    "veiculo",

    "vehicle",

    "vehicle_id",

    "provider",

    "provider_id",

    "selected_vehicle",

    "selected_vehicle_id",

    "selected_provider",

    "selected_provider_id",

]


# =============================================================================
# ESTILO GLOBAL
# =============================================================================

plt.rcParams.update({

    "font.family":
        "DejaVu Sans",

    "font.size":
        14,

    "axes.labelsize":
        17,

    "xtick.labelsize":
        15,

    "ytick.labelsize":
        15,

    "legend.fontsize":
        13,

    "pdf.fonttype":
        42,

    "ps.fonttype":
        42,

})


# =============================================================================
# SAVE
# =============================================================================

def save_figure(
    fig,
    filename,
):

    png_path = OUTPUT_DIR / f"{filename}.png"

    pdf_path = OUTPUT_DIR / f"{filename}.pdf"

    eps_path = OUTPUT_DIR / f"{filename}.eps"


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

        f"[SAVE] "
        f"{png_path}"

    )


# =============================================================================
# AXIS STYLE
# =============================================================================

def style_axis(
    ax,
):

    ax.grid(

        True,

        linestyle="--",

        linewidth=0.75,

        color=LIGHT_GRID,

    )


    ax.set_axisbelow(
        True
    )


    for spine in ax.spines.values():

        spine.set_color(
            "#505050"
        )

        spine.set_linewidth(
            0.8
        )


# =============================================================================
# T CRÍTICO 95%
# =============================================================================

def t_critical_95(
    n,
):

    if n <= 1:

        return np.nan


    table = {

        23: 2.068658,

        24: 2.063899,

        25: 2.059539,

        26: 2.055529,

        27: 2.051831,

        28: 2.048407,

        29: 2.045230,

    }


    df = int(
        n - 1
    )


    if df in table:

        return table[df]


    if df > 29:

        return 1.96


    return 2.069


# =============================================================================
# FIND COLUMN
# =============================================================================

def find_column(
    df,
    candidates,
    required=True,
):

    lookup = {

        str(column)
        .strip()
        .lower():
            column

        for column
        in df.columns

    }


    for candidate in candidates:

        key = (

            str(candidate)
            .strip()
            .lower()

        )


        if key in lookup:

            return lookup[key]


    if required:

        raise RuntimeError(

            "Nenhuma coluna encontrada entre:\n"
            f"  {candidates}\n\n"
            "Colunas disponíveis:\n"
            f"  {list(df.columns)}"

        )


    return None


# =============================================================================
# FILES
# =============================================================================

def get_files(
    strategy_key,
    kind,
):

    config = STRATEGIES[
        strategy_key
    ]


    if kind == "metric":

        pattern = config[
            "metric_pattern"
        ]


    elif kind == "request":

        pattern = config[
            "request_pattern"
        ]


    else:

        raise ValueError(

            f"kind inválido: "
            f"{kind}"

        )


    files = [

        Path(item)

        for item
        in sorted(

            glob.glob(
                pattern
            )

        )

    ]


    print(

        f"[FILES] "
        f"{config['label']:<15} "
        f"{kind:<8}: "
        f"{len(files)} selected"

    )


    if files:

        print(

            f"        first: "
            f"{files[0].name}"

        )

        print(

            f"        last : "
            f"{files[-1].name}"

        )


    if len(files) != 30:

        raise RuntimeError(

            f"{config['label']}: "
            f"esperados 30 arquivos '{kind}', "
            f"encontrados {len(files)}.\n\n"
            f"Pattern:\n"
            f"  {pattern}"

        )


    return files


# =============================================================================
# METADATA
# =============================================================================

def validate_metadata(
    df,
    path,
):

    if "rng_scheme" in df.columns:

        values = (

            df[
                "rng_scheme"
            ]

            .dropna()

            .astype(str)

            .str.strip()

        )


        if not values.empty:

            observed = values.iloc[0]


            if observed != EXPECTED_RNG:

                raise RuntimeError(

                    f"{path.name}: "
                    f"rng_scheme={observed}, "
                    f"esperado={EXPECTED_RNG}"

                )


    if "max_vehicles" in df.columns:

        values = (

            pd.to_numeric(

                df[
                    "max_vehicles"
                ],

                errors="coerce",

            )

            .dropna()

        )


        if not values.empty:

            observed = int(
                values.iloc[0]
            )


            if observed != PROVIDER_SCENARIO:

                raise RuntimeError(

                    f"{path.name}: "
                    f"max_vehicles={observed}, "
                    f"esperado={PROVIDER_SCENARIO}"

                )


    if "attack_type" in df.columns:

        values = (

            df[
                "attack_type"
            ]

            .dropna()

            .astype(str)

            .str.strip()

            .str.lower()

        )


        if not values.empty:

            observed = values.iloc[0]


            allowed = {

                "none",

                "no_attack",

                "disabled",

                "off",

            }


            if observed not in allowed:

                raise RuntimeError(

                    f"{path.name}: "
                    f"attack_type={observed}"

                )


    if "malicious_ratio" in df.columns:

        values = (

            pd.to_numeric(

                df[
                    "malicious_ratio"
                ],

                errors="coerce",

            )

            .dropna()

        )


        if not values.empty:

            observed = float(
                values.iloc[0]
            )


            if abs(observed) > 1e-12:

                raise RuntimeError(

                    f"{path.name}: "
                    f"malicious_ratio={observed}"

                )


# =============================================================================
# LOAD ACTIVITY
# =============================================================================

def load_activity(
    strategy_key,
):

    files = get_files(

        strategy_key,

        "request",

    )


    frames = []

    observed_seeds = set()


    for path in files:

        df = pd.read_csv(
            path
        )


        validate_metadata(
            df,
            path,
        )


        required = {

            "seed",

            "step",

            "active_vehicles",

        }


        missing = (

            required
            -
            set(df.columns)

        )


        if missing:

            raise RuntimeError(

                f"{path.name}: "
                f"colunas ausentes: "
                f"{sorted(missing)}"

            )


        seed = int(

            float(

                df[
                    "seed"
                ].iloc[0]

            )

        )


        if seed in observed_seeds:

            raise RuntimeError(

                f"{strategy_key}: "
                f"seed duplicada {seed}"

            )


        observed_seeds.add(
            seed
        )


        work = pd.DataFrame({

            "step":

                pd.to_numeric(

                    df[
                        "step"
                    ],

                    errors="coerce",

                ),


            "active_vehicles":

                pd.to_numeric(

                    df[
                        "active_vehicles"
                    ],

                    errors="coerce",

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

            .dropna(

                subset=[

                    "step",

                    "active_vehicles",

                ]

            )

        )


        work[
            "seed"
        ] = seed


        frames.append(
            work
        )


    missing_seeds = sorted(

        EXPECTED_SEEDS
        -
        observed_seeds

    )


    if missing_seeds:

        raise RuntimeError(

            f"{strategy_key}: "
            f"seeds ausentes nos requests: "
            f"{missing_seeds}"

        )


    return pd.concat(

        frames,

        ignore_index=True,

    )


# =============================================================================
# ACTIVITY SUMMARY
# =============================================================================

def build_activity_summary(
    activity,
):

    work = activity.copy()


    work[
        "window_start"
    ] = (

        np.floor(

            work[
                "step"
            ]

            /
            WINDOW_STEPS

        )

        *
        WINDOW_STEPS

    )


    per_seed = (

        work

        .groupby(

            [
                "seed",
                "window_start",
            ],

            as_index=False,

        )

        .agg(

            mean_active_vehicles=(

                "active_vehicles",

                "mean",

            ),

        )

    )


    per_seed[
        "simulation_step"
    ] = (

        per_seed[
            "window_start"
        ]

        +
        WINDOW_STEPS
        /
        2

    )


    summary = (

        per_seed

        .groupby(

            "simulation_step",

            as_index=False,

        )

        .agg(

            median_active_vehicles=(

                "mean_active_vehicles",

                "median",

            ),

            mean_active_vehicles=(

                "mean_active_vehicles",

                "mean",

            ),

            n_seeds=(

                "seed",

                "nunique",

            ),

        )

        .sort_values(
            "simulation_step"
        )

        .reset_index(
            drop=True
        )

    )


    summary[
        "seed_coverage"
    ] = (

        summary[
            "n_seeds"
        ]

        /
        len(
            EXPECTED_SEEDS
        )

    )


    return summary


# =============================================================================
# OPERATIONAL INTERVAL
# =============================================================================

def detect_operational_interval(
    strategy_key,
):

    activity = load_activity(
        strategy_key
    )


    summary = build_activity_summary(
        activity
    )


    work = (

        summary[

            summary[
                "seed_coverage"
            ]

            >=

            ACTIVE_MIN_SEED_COVERAGE

        ]

        .copy()

        .reset_index(
            drop=True
        )

    )


    if work.empty:

        raise RuntimeError(

            f"{strategy_key}: "
            "atividade insuficiente."

        )


    reference_active = float(

        work[
            "median_active_vehicles"
        ]

        .quantile(
            ACTIVE_REFERENCE_QUANTILE
        )

    )


    threshold = (

        reference_active

        *
        ACTIVE_OPERATIONAL_FRACTION

    )


    values = (

        work[
            "median_active_vehicles"
        ]

        .to_numpy(
            dtype=float
        )

    )


    steps = (

        work[
            "simulation_step"
        ]

        .to_numpy(
            dtype=float
        )

    )


    peak_position = int(
        values.argmax()
    )


    needed = ACTIVE_CONSECUTIVE_WINDOWS


    operational_start = None


    for index in range(

        0,

        peak_position + 1,

    ):

        block = values[

            index:
            index + needed

        ]


        if (

            len(block)
            ==
            needed

            and

            np.all(
                block >= threshold
            )

        ):

            operational_start = float(
                steps[index]
            )

            break


    operational_end = None


    for index in range(

        peak_position + 1,

        len(work)
        -
        needed
        +
        1,

    ):

        block = values[

            index:
            index + needed

        ]


        if np.all(
            block < threshold
        ):

            operational_end = float(
                steps[index]
            )

            break


    if operational_start is None:

        raise RuntimeError(

            f"{strategy_key}: "
            "início operacional não detectado."

        )


    if operational_end is None:

        raise RuntimeError(

            f"{strategy_key}: "
            "fim operacional não detectado."

        )


    return {

        "strategy":
            strategy_key,

        "label":
            STRATEGIES[
                strategy_key
            ][
                "label"
            ],

        "reference_active":
            reference_active,

        "threshold":
            threshold,

        "peak_step":
            float(
                steps[
                    peak_position
                ]
            ),

        "peak_active":
            float(
                values[
                    peak_position
                ]
            ),

        "operational_start":
            operational_start,

        "operational_end":
            operational_end,

    }


# =============================================================================
# COMMON OPERATIONAL CONTEXT
# =============================================================================

def prepare_operational_context():

    records = []


    for strategy_key in STRATEGY_ORDER:

        records.append(

            detect_operational_interval(
                strategy_key
            )

        )


    table = pd.DataFrame(
        records
    )


    common_start = float(

        table[
            "operational_start"
        ].max()

    )


    common_end = float(

        table[
            "operational_end"
        ].min()

    )


    if common_start >= common_end:

        raise RuntimeError(

            "Não existe intervalo operacional "
            "comum entre as estratégias."

        )


    table.to_csv(

        OUTPUT_DIR
        /
        "operational_intervals_internal_strategies.csv",

        index=False,

    )


    print(
        "\n"
        +
        "=" * 100
    )


    print(
        "COMMON OPERATIONAL INTERVAL — INTERNAL STRATEGIES"
    )


    print(
        "=" * 100
    )


    print(

        table[

            [

                "label",

                "reference_active",

                "threshold",

                "peak_step",

                "peak_active",

                "operational_start",

                "operational_end",

            ]

        ]

        .round(2)

        .to_string(
            index=False
        )

    )


    print(

        f"\nCommon interval: "
        f"{common_start:.0f} - "
        f"{common_end:.0f}"

    )


    return {

        "table":
            table,

        "common_start":
            common_start,

        "common_end":
            common_end,

    }


# =============================================================================
# EXTRACT STANDARD TEMPORAL METRIC
# =============================================================================

def extract_metric_values(
    df,
    path,
    metric_name,
):

    if metric_name == "reputation":

        normalized_col = find_column(

            df,

            REPUTATION_NORMALIZED_ALIASES,

            required=False,

        )


        if normalized_col is not None:

            return (

                pd.to_numeric(

                    df[
                        normalized_col
                    ],

                    errors="coerce",

                ),

                str(
                    normalized_col
                ),

                "normalized",

            )


        raw_col = find_column(

            df,

            REPUTATION_RAW_ALIASES,

            required=True,

        )


        values = pd.to_numeric(

            df[
                raw_col
            ],

            errors="coerce",

        )


        finite = (

            values

            .replace(

                [
                    np.inf,
                    -np.inf,
                ],

                np.nan,

            )

            .dropna()

        )


        if finite.empty:

            raise RuntimeError(

                f"{path.name}: "
                "reputation vazia."

            )


        q99 = float(

            finite.quantile(
                0.99
            )

        )


        if q99 <= 1.05:

            return (

                values,

                str(
                    raw_col
                ),

                "already_0_1",

            )


        if q99 <= 10.5:

            return (

                values / 10.0,

                str(
                    raw_col
                ),

                "raw_0_10_div10",

            )


        raise RuntimeError(

            f"{path.name}: "
            "escala de reputation "
            "não reconhecida."

        )


    if metric_name == "score":

        column = find_column(

            df,

            SCORE_ALIASES,

            required=True,

        )


        return (

            pd.to_numeric(

                df[
                    column
                ],

                errors="coerce",

            ),

            str(
                column
            ),

            "direct",

        )


    if metric_name == "utility":

        column = find_column(

            df,

            UTILITY_ALIASES,

            required=True,

        )


        return (

            pd.to_numeric(

                df[
                    column
                ],

                errors="coerce",

            ),

            str(
                column
            ),

            "direct",

        )


    raise ValueError(

        f"Métrica inválida: "
        f"{metric_name}"

    )


# =============================================================================
# LOAD STANDARD TEMPORAL METRIC
# =============================================================================

def load_metric(
    strategy_key,
    metric_name,
):

    files = get_files(

        strategy_key,

        "metric",

    )


    datasets = {}

    audit_records = []


    for path in files:

        df = pd.read_csv(
            path
        )


        validate_metadata(
            df,
            path,
        )


        required = {

            "seed",

            "step",

        }


        missing = (

            required
            -
            set(df.columns)

        )


        if missing:

            raise RuntimeError(

                f"{path.name}: "
                f"colunas ausentes: "
                f"{sorted(missing)}"

            )


        seed = int(

            float(

                df[
                    "seed"
                ].iloc[0]

            )

        )


        if seed in datasets:

            raise RuntimeError(

                f"{strategy_key}: "
                f"seed duplicada {seed}"

            )


        (
            metric_values,
            source_column,
            extraction_mode,

        ) = extract_metric_values(

            df,

            path,

            metric_name,

        )


        work = pd.DataFrame({

            "step":

                pd.to_numeric(

                    df[
                        "step"
                    ],

                    errors="coerce",

                ),


            "metric":

                metric_values,

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

            .dropna(

                subset=[

                    "step",

                    "metric",

                ]

            )

        )


        if work.empty:

            raise RuntimeError(

                f"{path.name}: "
                f"nenhuma observação válida "
                f"para {metric_name}."

            )


        datasets[
            seed
        ] = work


        audit_records.append({

            "strategy":
                strategy_key,

            "label":
                STRATEGIES[
                    strategy_key
                ][
                    "label"
                ],

            "seed":
                seed,

            "file":
                path.name,

            "metric":
                metric_name,

            "source_column":
                source_column,

            "extraction_mode":
                extraction_mode,

            "minimum":
                float(
                    work[
                        "metric"
                    ].min()
                ),

            "maximum":
                float(
                    work[
                        "metric"
                    ].max()
                ),

            "mean":
                float(
                    work[
                        "metric"
                    ].mean()
                ),

        })


    missing_seeds = sorted(

        EXPECTED_SEEDS
        -
        set(
            datasets
        )

    )


    if missing_seeds:

        raise RuntimeError(

            f"{strategy_key}: "
            f"seeds ausentes para "
            f"{metric_name}: "
            f"{missing_seeds}"

        )


    print(

        f"[OK] "
        f"{STRATEGIES[strategy_key]['label']}: "
        f"{len(datasets)} metric seeds "
        f"for {metric_name}"

    )


    return (

        datasets,

        pd.DataFrame(
            audit_records
        ),

    )


# =============================================================================
# STANDARD SMOOTHING
# =============================================================================

def smooth_seed_series(
    per_seed,
):

    per_seed = (

        per_seed

        .sort_values(
            "window_start"
        )

        .copy()

    )


    start = int(

        per_seed[
            "window_start"
        ].min()

    )


    end = int(

        per_seed[
            "window_start"
        ].max()

    )


    complete_index = np.arange(

        start,

        end
        +
        WINDOW_STEPS,

        WINDOW_STEPS,

    )


    work = (

        per_seed

        .set_index(
            "window_start"
        )

        .reindex(
            complete_index
        )

    )


    work.index.name = (
        "window_start"
    )


    work[
        "metric_smooth"
    ] = (

        work[
            "metric_mean"
        ]

        .rolling(

            window=
                SMOOTHING_WINDOWS,

            center=True,

            min_periods=
                SMOOTHING_WINDOWS,

        )

        .mean()

    )


    work = work.reset_index()


    work[
        "simulation_step"
    ] = (

        work[
            "window_start"
        ]

        +
        WINDOW_STEPS
        /
        2

    )


    return work


# =============================================================================
# BUILD STANDARD METRIC SUMMARY
# =============================================================================

def build_metric_summary(
    strategy_key,
    metric_name,
):

    (
        datasets,
        audit,

    ) = load_metric(

        strategy_key,

        metric_name,

    )


    frames = []


    for seed in sorted(
        datasets
    ):

        work = datasets[
            seed
        ].copy()


        work[
            "window_start"
        ] = (

            np.floor(

                work[
                    "step"
                ]

                /
                WINDOW_STEPS

            )

            *
            WINDOW_STEPS

        )


        per_seed = (

            work

            .groupby(

                "window_start",

                as_index=False,

            )

            .agg(

                metric_mean=(

                    "metric",

                    "mean",

                ),

                n_observations=(

                    "metric",

                    "size",

                ),

            )

        )


        per_seed = smooth_seed_series(
            per_seed
        )


        per_seed[
            "seed"
        ] = seed


        per_seed[
            "strategy"
        ] = strategy_key


        frames.append(
            per_seed
        )


    per_seed_df = pd.concat(

        frames,

        ignore_index=True,

    )


    return finalize_summary(

        strategy_key=
            strategy_key,

        metric_name=
            metric_name,

        per_seed_df=
            per_seed_df,

        audit=
            audit,

    )


# =============================================================================
# JAIN INDEX
# =============================================================================

def jain_index(
    values,
):

    x = np.asarray(
        values,
        dtype=float,
    )


    x = x[
        np.isfinite(x)
    ]


    x = x[
        x > 0
    ]


    if len(x) == 0:

        return np.nan


    denominator = (

        len(x)

        *
        np.sum(
            x ** 2
        )

    )


    if denominator <= 0:

        return np.nan


    numerator = (

        np.sum(x)
        **
        2

    )


    return float(

        numerator
        /
        denominator

    )


# =============================================================================
# BUILD CUMULATIVE JAIN FOR ONE SEED
# =============================================================================

def build_cumulative_jain_seed(
    df,
    provider_column,
):

    work = pd.DataFrame({

        "step":

            pd.to_numeric(

                df[
                    "step"
                ],

                errors="coerce",

            ),


        "provider":

            df[
                provider_column
            ]

            .astype(str)

            .str.strip(),

    })


    work = (

        work

        .replace(

            {

                "provider": {

                    "":
                        np.nan,

                    "nan":
                        np.nan,

                    "None":
                        np.nan,

                    "none":
                        np.nan,

                }

            }

        )

        .replace(

            [
                np.inf,
                -np.inf,
            ],

            np.nan,

        )

        .dropna(

            subset=[

                "step",

                "provider",

            ]

        )

    )


    if work.empty:

        raise RuntimeError(

            "Nenhuma seleção válida encontrada "
            "para calcular Jain fairness."

        )


    work[
        "window_start"
    ] = (

        np.floor(

            work[
                "step"
            ]

            /
            WINDOW_STEPS

        )

        *
        WINDOW_STEPS

    )


    first_window = int(

        work[
            "window_start"
        ].min()

    )


    last_window = int(

        work[
            "window_start"
        ].max()

    )


    all_windows = np.arange(

        first_window,

        last_window
        +
        WINDOW_STEPS,

        WINDOW_STEPS,

    )


    provider_counts = {}


    rows = []


    grouped = {

        int(window):
            group

        for window, group
        in work.groupby(
            "window_start"
        )

    }


    for window in all_windows:

        current = grouped.get(
            int(window)
        )


        selections_in_window = 0


        if current is not None:

            providers = (

                current[
                    "provider"
                ]

                .astype(str)

            )


            selections_in_window = int(
                len(
                    providers
                )
            )


            for provider in providers:

                provider_counts[
                    provider
                ] = (

                    provider_counts.get(
                        provider,
                        0,
                    )

                    +
                    1

                )


        fairness = jain_index(

            list(
                provider_counts.values()
            )

        )


        rows.append({

            "window_start":
                float(
                    window
                ),

            "metric_mean":
                fairness,

            "n_observations":
                selections_in_window,

            "cumulative_selections":
                int(

                    sum(
                        provider_counts.values()
                    )

                ),

            "cumulative_unique_providers":
                int(

                    len(
                        provider_counts
                    )

                ),

        })


    result = pd.DataFrame(
        rows
    )


    result[
        "metric_mean"
    ] = (

        result[
            "metric_mean"
        ]

        .ffill()

    )


    result[
        "metric_smooth"
    ] = (

        result[
            "metric_mean"
        ]

        .rolling(

            window=
                SMOOTHING_WINDOWS,

            center=True,

            min_periods=
                SMOOTHING_WINDOWS,

        )

        .mean()

    )


    result[
        "simulation_step"
    ] = (

        result[
            "window_start"
        ]

        +
        WINDOW_STEPS
        /
        2

    )


    return result


# =============================================================================
# BUILD FAIRNESS SUMMARY
# =============================================================================

def build_fairness_summary(
    strategy_key,
):

    files = get_files(

        strategy_key,

        "metric",

    )


    frames = []

    audit_records = []

    observed_seeds = set()


    for path in files:

        df = pd.read_csv(
            path
        )


        validate_metadata(
            df,
            path,
        )


        required = {

            "seed",

            "step",

        }


        missing = (

            required
            -
            set(
                df.columns
            )

        )


        if missing:

            raise RuntimeError(

                f"{path.name}: "
                f"colunas ausentes: "
                f"{sorted(missing)}"

            )


        provider_column = find_column(

            df,

            PROVIDER_ALIASES,

            required=True,

        )


        seed = int(

            float(

                df[
                    "seed"
                ].iloc[0]

            )

        )


        if seed in observed_seeds:

            raise RuntimeError(

                f"{strategy_key}: "
                f"seed duplicada {seed}"

            )


        observed_seeds.add(
            seed
        )


        per_seed = build_cumulative_jain_seed(

            df=
                df,

            provider_column=
                provider_column,

        )


        per_seed[
            "seed"
        ] = seed


        per_seed[
            "strategy"
        ] = strategy_key


        frames.append(
            per_seed
        )


        final_fairness = float(

            per_seed[
                "metric_mean"
            ]

            .dropna()

            .iloc[-1]

        )


        final_unique = int(

            per_seed[
                "cumulative_unique_providers"
            ]

            .iloc[-1]

        )


        final_selections = int(

            per_seed[
                "cumulative_selections"
            ]

            .iloc[-1]

        )


        audit_records.append({

            "strategy":
                strategy_key,

            "label":
                STRATEGIES[
                    strategy_key
                ][
                    "label"
                ],

            "seed":
                seed,

            "file":
                path.name,

            "provider_column":
                str(
                    provider_column
                ),

            "final_cumulative_jain":
                final_fairness,

            "final_unique_providers":
                final_unique,

            "final_selections":
                final_selections,

        })


    missing_seeds = sorted(

        EXPECTED_SEEDS
        -
        observed_seeds

    )


    if missing_seeds:

        raise RuntimeError(

            f"{strategy_key}: "
            "seeds ausentes para fairness: "
            f"{missing_seeds}"

        )


    per_seed_df = pd.concat(

        frames,

        ignore_index=True,

    )


    audit = pd.DataFrame(
        audit_records
    )


    print(

        f"[OK] "
        f"{STRATEGIES[strategy_key]['label']}: "
        f"{len(observed_seeds)} fairness seeds"

    )


    result = finalize_summary(

        strategy_key=
            strategy_key,

        metric_name=
            "fairness",

        per_seed_df=
            per_seed_df,

        audit=
            audit,

    )


    print(

        f"     final mean cumulative Jain: "
        f"{audit['final_cumulative_jain'].mean():.4f}"

    )


    print(

        f"     final mean unique providers: "
        f"{audit['final_unique_providers'].mean():.2f}"

    )


    return result


# =============================================================================
# BUILD CUMULATIVE PROVIDER DIVERSITY SUMMARY
# =============================================================================

def build_diversity_summary(
    strategy_key,
):

    files = get_files(

        strategy_key,

        "metric",

    )


    frames = []

    audit_records = []

    observed_seeds = set()


    for path in files:

        df = pd.read_csv(
            path
        )


        validate_metadata(
            df,
            path,
        )


        required = {

            "seed",

            "step",

        }


        missing = (

            required
            -
            set(
                df.columns
            )

        )


        if missing:

            raise RuntimeError(

                f"{path.name}: "
                f"colunas ausentes: "
                f"{sorted(missing)}"

            )


        provider_column = find_column(

            df,

            PROVIDER_ALIASES,

            required=True,

        )


        seed = int(

            float(

                df[
                    "seed"
                ].iloc[0]

            )

        )


        if seed in observed_seeds:

            raise RuntimeError(

                f"{strategy_key}: "
                f"seed duplicada {seed}"

            )


        observed_seeds.add(
            seed
        )


        cumulative = build_cumulative_jain_seed(

            df=
                df,

            provider_column=
                provider_column,

        )


        per_seed = pd.DataFrame({

            "window_start":

                cumulative[
                    "window_start"
                ],


            "simulation_step":

                cumulative[
                    "simulation_step"
                ],


            "metric_mean":

                pd.to_numeric(

                    cumulative[
                        "cumulative_unique_providers"
                    ],

                    errors="coerce",

                ),


            "n_observations":

                cumulative[
                    "n_observations"
                ],

        })


        per_seed[
            "metric_smooth"
        ] = (

            per_seed[
                "metric_mean"
            ]

            .rolling(

                window=
                    SMOOTHING_WINDOWS,

                center=True,

                min_periods=
                    SMOOTHING_WINDOWS,

            )

            .mean()

        )


        per_seed[
            "seed"
        ] = seed


        per_seed[
            "strategy"
        ] = strategy_key


        frames.append(
            per_seed
        )


        final_unique = float(

            cumulative[
                "cumulative_unique_providers"
            ]

            .iloc[-1]

        )


        final_selections = int(

            cumulative[
                "cumulative_selections"
            ]

            .iloc[-1]

        )


        audit_records.append({

            "strategy":
                strategy_key,

            "label":
                STRATEGIES[
                    strategy_key
                ][
                    "label"
                ],

            "seed":
                seed,

            "file":
                path.name,

            "provider_column":
                str(
                    provider_column
                ),

            "final_unique_providers":
                final_unique,

            "final_selections":
                final_selections,

        })


    missing_seeds = sorted(

        EXPECTED_SEEDS
        -
        observed_seeds

    )


    if missing_seeds:

        raise RuntimeError(

            f"{strategy_key}: "
            "seeds ausentes para diversity: "
            f"{missing_seeds}"

        )


    per_seed_df = pd.concat(

        frames,

        ignore_index=True,

    )


    audit = pd.DataFrame(
        audit_records
    )


    print(

        f"[OK] "
        f"{STRATEGIES[strategy_key]['label']}: "
        f"{len(observed_seeds)} diversity seeds"

    )


    result = finalize_summary(

        strategy_key=
            strategy_key,

        metric_name=
            "diversity",

        per_seed_df=
            per_seed_df,

        audit=
            audit,

    )


    print(

        f"     final mean unique providers: "
        f"{audit['final_unique_providers'].mean():.2f}"

    )


    return result


# =============================================================================
# FINALIZE SUMMARY
# =============================================================================

def finalize_summary(
    strategy_key,
    metric_name,
    per_seed_df,
    audit,
):

    summary = (

        per_seed_df

        .groupby(

            "simulation_step",

            as_index=False,

        )

        .agg(

            mean_raw=(

                "metric_mean",

                "mean",

            ),

            mean_smooth=(

                "metric_smooth",

                "mean",

            ),

            std_smooth=(

                "metric_smooth",

                "std",

            ),

            n_seeds=(

                "metric_smooth",

                "count",

            ),

            total_observations=(

                "n_observations",

                "sum",

            ),

        )

        .sort_values(
            "simulation_step"
        )

        .reset_index(
            drop=True
        )

    )


    summary[
        "strategy"
    ] = strategy_key


    minimum_seed_count = int(

        math.ceil(

            len(
                EXPECTED_SEEDS
            )

            *
            MIN_SEED_COVERAGE

        )

    )


    summary[
        "seed_coverage"
    ] = (

        summary[
            "n_seeds"
        ]

        /
        len(
            EXPECTED_SEEDS
        )

    )


    summary[
        "passes_seed_coverage"
    ] = (

        summary[
            "n_seeds"
        ]

        >=

        minimum_seed_count

    )


    reference = (

        summary[

            summary[
                "passes_seed_coverage"
            ]

        ]

        .copy()

    )


    if reference.empty:

        raise RuntimeError(

            f"{strategy_key}: "
            f"sem cobertura suficiente "
            f"para {metric_name}."

        )


    typical_observations = float(

        reference[
            "total_observations"
        ]

        .median()

    )


    if metric_name in {

        "fairness",

        "diversity",

    }:

        minimum_observations = 0.0


        summary[
            "passes_observation_support"
        ] = True


    else:

        minimum_observations = (

            typical_observations

            *
            MIN_OBSERVATION_SUPPORT_RATIO

        )


        summary[
            "passes_observation_support"
        ] = (

            summary[
                "total_observations"
            ]

            >=

            minimum_observations

        )


    summary[
        "sem"
    ] = (

        summary[
            "std_smooth"
        ]

        /

        np.sqrt(

            summary[
                "n_seeds"
            ]

        )

    )


    summary[
        "t_critical"
    ] = (

        summary[
            "n_seeds"
        ]

        .apply(
            t_critical_95
        )

    )


    summary[
        "ci95"
    ] = (

        summary[
            "t_critical"
        ]

        *
        summary[
            "sem"
        ]

    )


    summary[
        "ci95_low"
    ] = (

        summary[
            "mean_smooth"
        ]

        -
        summary[
            "ci95"
        ]

    )


    summary[
        "ci95_high"
    ] = (

        summary[
            "mean_smooth"
        ]

        +
        summary[
            "ci95"
        ]

    )


    return {

        "per_seed":
            per_seed_df,

        "summary":
            summary,

        "audit":
            audit,

        "typical_observations":
            typical_observations,

        "minimum_observations":
            minimum_observations,

    }


# =============================================================================
# PREPARE COMPARISON
# =============================================================================

def prepare_metric_comparison(
    metric_name,
    operational_context,
):

    summaries = {}

    per_seed_frames = []

    audit_frames = []

    diagnostic_records = []


    common_start = operational_context[
        "common_start"
    ]


    common_end = operational_context[
        "common_end"
    ]


    for strategy_key in STRATEGY_ORDER:

        if metric_name == "fairness":

            result = build_fairness_summary(
                strategy_key
            )


        elif metric_name == "diversity":

            result = build_diversity_summary(
                strategy_key
            )


        else:

            result = build_metric_summary(

                strategy_key,

                metric_name,

            )


        summary = result[
            "summary"
        ].copy()


        summary[
            "inside_common_operational_interval"
        ] = (

            (
                summary[
                    "simulation_step"
                ]

                >=

                common_start

            )

            &

            (
                summary[
                    "simulation_step"
                ]

                <

                common_end

            )

        )


        summary[
            "used_in_plot"
        ] = (

            summary[
                "passes_seed_coverage"
            ]

            &

            summary[
                "passes_observation_support"
            ]

            &

            summary[
                "inside_common_operational_interval"
            ]

            &

            summary[
                "mean_smooth"
            ].notna()

        )


        valid = summary[
            "used_in_plot"
        ]


        if not valid.any():

            raise RuntimeError(

                f"{strategy_key}: "
                f"nenhum ponto válido "
                f"para {metric_name}."

            )


        summaries[
            strategy_key
        ] = summary


        per_seed_frames.append(

            result[
                "per_seed"
            ]

        )


        audit_frames.append(

            result[
                "audit"
            ]

        )


        diagnostic_records.append({

            "strategy":
                strategy_key,

            "label":
                STRATEGIES[
                    strategy_key
                ][
                    "label"
                ],

            "metric":
                metric_name,

            "plot_windows":
                int(
                    valid.sum()
                ),

            "min_plot_seeds":
                int(

                    summary.loc[
                        valid,
                        "n_seeds",
                    ].min()

                ),

            "mean_metric":
                float(

                    summary.loc[
                        valid,
                        "mean_smooth",
                    ].mean()

                ),

            "plot_min":
                float(

                    summary.loc[
                        valid,
                        "mean_smooth",
                    ].min()

                ),

            "plot_max":
                float(

                    summary.loc[
                        valid,
                        "mean_smooth",
                    ].max()

                ),

        })


    return {

        "summaries":
            summaries,

        "all_summary":

            pd.concat(

                [

                    summaries[key]

                    for key
                    in STRATEGY_ORDER

                ],

                ignore_index=True,

            ),

        "all_per_seed":

            pd.concat(

                per_seed_frames,

                ignore_index=True,

            ),

        "audit":

            pd.concat(

                audit_frames,

                ignore_index=True,

            ),

        "diagnostics":

            pd.DataFrame(
                diagnostic_records
            ),

        "common_start":
            common_start,

        "common_end":
            common_end,

    }


# =============================================================================
# DRAW COMPARISON
# =============================================================================

def draw_metric_comparison(
    data,
    ylabel,
    y_min,
    y_max,
    show_ci,
    legend_location,
):

    fig, ax = plt.subplots(

        figsize=(
            10.2,
            5.8,
        )

    )


    for strategy_key in STRATEGY_ORDER:

        summary = data[
            "summaries"
        ][
            strategy_key
        ]


        config = STRATEGIES[
            strategy_key
        ]


        plot_y = np.where(

            summary[
                "used_in_plot"
            ],

            summary[
                "mean_smooth"
            ],

            np.nan,

        )


        ax.plot(

            summary[
                "simulation_step"
            ],

            plot_y,

            label=
                config[
                    "label"
                ],

            color=
                config[
                    "color"
                ],

            linestyle=
                config[
                    "linestyle"
                ],

            linewidth=
                config[
                    "linewidth"
                ],

            zorder=3,

        )


        if show_ci:

            valid = summary[
                "used_in_plot"
            ]


            ax.fill_between(

                summary.loc[
                    valid,
                    "simulation_step",
                ],

                summary.loc[
                    valid,
                    "ci95_low",
                ],

                summary.loc[
                    valid,
                    "ci95_high",
                ],

                color=
                    config[
                        "color"
                    ],

                alpha=0.10,

                linewidth=0,

                zorder=1,

            )


    ax.set_xlabel(
        "Simulation Step"
    )


    ax.set_ylabel(
        ylabel
    )


    ax.set_ylim(

        y_min,

        y_max,

    )


    ax.set_xlim(

        data[
            "common_start"
        ]

        -
        WINDOW_STEPS
        *
        0.20,

        data[
            "common_end"
        ]

        -
        WINDOW_STEPS
        *
        0.30,

    )


    style_axis(
        ax
    )


    ax.legend(

        loc=
            legend_location,

        frameon=True,

        framealpha=1.0,

        facecolor="white",

        edgecolor="#D0D0D0",

    )


    fig.tight_layout()


    return fig


# =============================================================================
# SAVE TABLES
# =============================================================================

def save_metric_tables(
    data,
    prefix,
):

    data[
        "all_per_seed"
    ].to_csv(

        OUTPUT_DIR
        /
        f"{prefix}_strategies_per_seed.csv",

        index=False,

    )


    data[
        "all_summary"
    ].to_csv(

        OUTPUT_DIR
        /
        f"{prefix}_strategies_all_windows.csv",

        index=False,

    )


    data[
        "all_summary"
    ][

        data[
            "all_summary"
        ][
            "used_in_plot"
        ]

    ].to_csv(

        OUTPUT_DIR
        /
        f"{prefix}_strategies_plot_data.csv",

        index=False,

    )


    data[
        "audit"
    ].to_csv(

        OUTPUT_DIR
        /
        f"{prefix}_strategies_audit.csv",

        index=False,

    )


    data[
        "diagnostics"
    ].to_csv(

        OUTPUT_DIR
        /
        f"{prefix}_strategies_diagnostics.csv",

        index=False,

    )


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def print_temporal_diagnostics(
    title,
    data,
    y_min,
    y_max,
):

    print(
        "\n"
        +
        "=" * 100
    )


    print(
        title
    )


    print(
        "=" * 100
    )


    print(

        f"Base temporal window      : "
        f"{WINDOW_STEPS} steps"

    )


    print(

        f"Smoothing span            : "
        f"{WINDOW_STEPS * SMOOTHING_WINDOWS} steps"

    )


    print(

        f"Common operational start  : "
        f"{data['common_start']:.0f}"

    )


    print(

        f"Common operational end    : "
        f"{data['common_end']:.0f}"

    )


    print(

        f"Y-axis range              : "
        f"{y_min:.4f} - "
        f"{y_max:.4f}"

    )


    print(
        "\nDiagnostics:"
    )


    print(

        data[
            "diagnostics"
        ]

        .round(4)

        .to_string(
            index=False
        )

    )


# =============================================================================
# GENERIC TEMPORAL FIGURE
# =============================================================================

def plot_temporal_metric(
    operational_context,
    metric_name,
    ylabel,
    y_min,
    y_max,
    show_ci,
    legend_location,
    filename,
    diagnostic_title,
):

    data = prepare_metric_comparison(

        metric_name=
            metric_name,

        operational_context=
            operational_context,

    )


    fig = draw_metric_comparison(

        data=
            data,

        ylabel=
            ylabel,

        y_min=
            y_min,

        y_max=
            y_max,

        show_ci=
            show_ci,

        legend_location=
            legend_location,

    )


    save_figure(

        fig,

        filename,

    )


    plt.close(
        fig
    )


    save_metric_tables(

        data,

        metric_name,

    )


    print_temporal_diagnostics(

        diagnostic_title,

        data,

        y_min,

        y_max,

    )


# =============================================================================
# FIGURE 1 — REPUTATION
# =============================================================================

def plot_reputation_comparison(
    operational_context,
):

    plot_temporal_metric(

        operational_context=
            operational_context,

        metric_name=
            "reputation",

        ylabel=
            "Average Normalized Reputation",

        y_min=
            REPUTATION_Y_MIN,

        y_max=
            REPUTATION_Y_MAX,

        show_ci=
            SHOW_REPUTATION_CI,

        legend_location=
            "lower right",

        filename=
            "reputation_comparison_strategies",

        diagnostic_title=
            "REPUTATION COMPARISON BETWEEN INTERNAL STRATEGIES",

    )


# =============================================================================
# FIGURE 2 — SCORE
# =============================================================================

def plot_score_comparison(
    operational_context,
):

    plot_temporal_metric(

        operational_context=
            operational_context,

        metric_name=
            "score",

        ylabel=
            "Average Score",

        y_min=
            SCORE_Y_MIN,

        y_max=
            SCORE_Y_MAX,

        show_ci=
            SHOW_SCORE_CI,

        legend_location=
            "lower right",

        filename=
            "score_comparison_strategies",

        diagnostic_title=
            "SCORE COMPARISON BETWEEN INTERNAL STRATEGIES",

    )


# =============================================================================
# FIGURE 3 — UTILITY
# =============================================================================

def plot_utility_comparison(
    operational_context,
):

    plot_temporal_metric(

        operational_context=
            operational_context,

        metric_name=
            "utility",

        ylabel=
            "Average Utility",

        y_min=
            UTILITY_Y_MIN,

        y_max=
            UTILITY_Y_MAX,

        show_ci=
            SHOW_UTILITY_CI,

        legend_location=
            "lower right",

        filename=
            "utility_comparison_strategies",

        diagnostic_title=
            "UTILITY COMPARISON BETWEEN INTERNAL STRATEGIES",

    )


# =============================================================================
# FIGURE 4 — FAIRNESS + PROVIDER DIVERSITY
# =============================================================================

def plot_fairness_diversity_comparison(
    operational_context,
):

    fairness_data = prepare_metric_comparison(

        metric_name=
            "fairness",

        operational_context=
            operational_context,

    )


    diversity_data = prepare_metric_comparison(

        metric_name=
            "diversity",

        operational_context=
            operational_context,

    )


    fig, axes = plt.subplots(

        1,

        2,

        figsize=(
            13.0,
            5.3,
        ),

        sharex=True,

    )


    ax_fairness = axes[0]

    ax_diversity = axes[1]


    # =========================================================================
    # LEFT — CUMULATIVE JAIN FAIRNESS
    # =========================================================================

    for strategy_key in STRATEGY_ORDER:

        summary = fairness_data[
            "summaries"
        ][
            strategy_key
        ]


        config = STRATEGIES[
            strategy_key
        ]


        plot_y = np.where(

            summary[
                "used_in_plot"
            ],

            summary[
                "mean_smooth"
            ],

            np.nan,

        )


        ax_fairness.plot(

            summary[
                "simulation_step"
            ],

            plot_y,

            label=
                config[
                    "label"
                ],

            color=
                config[
                    "color"
                ],

            linestyle=
                config[
                    "linestyle"
                ],

            linewidth=
                config[
                    "linewidth"
                ],

            zorder=3,

        )


        if SHOW_FAIRNESS_CI:

            valid = summary[
                "used_in_plot"
            ]


            ax_fairness.fill_between(

                summary.loc[
                    valid,
                    "simulation_step",
                ],

                summary.loc[
                    valid,
                    "ci95_low",
                ],

                summary.loc[
                    valid,
                    "ci95_high",
                ],

                color=
                    config[
                        "color"
                    ],

                alpha=0.10,

                linewidth=0,

                zorder=1,

            )


    ax_fairness.set_xlabel(
        "Simulation Step"
    )


    ax_fairness.set_ylabel(
        "Cumulative Jain Fairness Index"
    )


    ax_fairness.set_ylim(

        FAIRNESS_Y_MIN,

        FAIRNESS_Y_MAX,

    )


    ax_fairness.set_xlim(

        fairness_data[
            "common_start"
        ]

        -
        WINDOW_STEPS
        *
        0.20,

        fairness_data[
            "common_end"
        ]

        -
        WINDOW_STEPS
        *
        0.30,

    )


    style_axis(
        ax_fairness
    )


    ax_fairness.legend(

        loc=
            "lower left",

        frameon=True,

        framealpha=1.0,

        facecolor="white",

        edgecolor="#D0D0D0",

    )


    # =========================================================================
    # RIGHT — CUMULATIVE UNIQUE SELECTED PROVIDERS
    # =========================================================================

    diversity_values = []


    for strategy_key in STRATEGY_ORDER:

        summary = diversity_data[
            "summaries"
        ][
            strategy_key
        ]


        config = STRATEGIES[
            strategy_key
        ]


        plot_y = np.where(

            summary[
                "used_in_plot"
            ],

            summary[
                "mean_smooth"
            ],

            np.nan,

        )


        valid_values = (

            summary.loc[

                summary[
                    "used_in_plot"
                ],

                "mean_smooth",

            ]

            .dropna()

            .to_numpy(
                dtype=float
            )

        )


        if len(valid_values) > 0:

            diversity_values.extend(

                valid_values.tolist()

            )


        ax_diversity.plot(

            summary[
                "simulation_step"
            ],

            plot_y,

            label=
                config[
                    "label"
                ],

            color=
                config[
                    "color"
                ],

            linestyle=
                config[
                    "linestyle"
                ],

            linewidth=
                config[
                    "linewidth"
                ],

            zorder=3,

        )


        if SHOW_DIVERSITY_CI:

            valid = summary[
                "used_in_plot"
            ]


            ax_diversity.fill_between(

                summary.loc[
                    valid,
                    "simulation_step",
                ],

                summary.loc[
                    valid,
                    "ci95_low",
                ],

                summary.loc[
                    valid,
                    "ci95_high",
                ],

                color=
                    config[
                        "color"
                    ],

                alpha=0.10,

                linewidth=0,

                zorder=1,

            )


    if not diversity_values:

        raise RuntimeError(

            "Nenhum valor válido encontrado "
            "para cumulative provider diversity."

        )


    diversity_min = float(

        np.nanmin(
            diversity_values
        )

    )


    diversity_max = float(

        np.nanmax(
            diversity_values
        )

    )


    diversity_range = (

        diversity_max
        -
        diversity_min

    )


    padding = max(

        2.0,

        diversity_range
        *
        0.08,

    )


    diversity_y_min = max(

        0.0,

        diversity_min
        -
        padding,

    )


    diversity_y_max = (

        diversity_max
        +
        padding

    )


    ax_diversity.set_xlabel(
        "Simulation Step"
    )


    ax_diversity.set_ylabel(
        "Cumulative Unique Selected Providers"
    )


    ax_diversity.set_ylim(

        diversity_y_min,

        diversity_y_max,

    )


    ax_diversity.set_xlim(

        diversity_data[
            "common_start"
        ]

        -
        WINDOW_STEPS
        *
        0.20,

        diversity_data[
            "common_end"
        ]

        -
        WINDOW_STEPS
        *
        0.30,

    )


    style_axis(
        ax_diversity
    )


    fig.tight_layout()


    save_figure(

        fig,

        "fairness_diversity_comparison_strategies",

    )


    plt.close(
        fig
    )


    save_metric_tables(

        fairness_data,

        "fairness",

    )


    save_metric_tables(

        diversity_data,

        "diversity",

    )


    print_temporal_diagnostics(

        "CUMULATIVE JAIN FAIRNESS COMPARISON BETWEEN INTERNAL STRATEGIES",

        fairness_data,

        FAIRNESS_Y_MIN,

        FAIRNESS_Y_MAX,

    )


    print_temporal_diagnostics(

        "CUMULATIVE UNIQUE PROVIDER COMPARISON BETWEEN INTERNAL STRATEGIES",

        diversity_data,

        diversity_y_min,

        diversity_y_max,

    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "\n"
        +
        "#" * 100
    )


    print(
        "SHADAI - INTERNAL STRATEGY COMPARISON"
    )


    print(
        "#" * 100
    )


    operational_context = prepare_operational_context()


    # =========================================================================
    # 1 — REPUTATION
    # =========================================================================

    plot_reputation_comparison(

        operational_context

    )


    # =========================================================================
    # 2 — SCORE
    # =========================================================================

    plot_score_comparison(

        operational_context

    )


    # =========================================================================
    # 3 — UTILITY
    # =========================================================================

    plot_utility_comparison(

        operational_context

    )


    # =========================================================================
    # 4 — FAIRNESS + PROVIDER DIVERSITY
    # =========================================================================

    plot_fairness_diversity_comparison(

        operational_context

    )


    # =========================================================================
    # FINAL
    # =========================================================================

    print(
        "\n"
        +
        "#" * 100
    )


    print(
        "ALL INTERNAL STRATEGY-COMPARISON FIGURES GENERATED"
    )


    print(
        "#" * 100
    )


    print(
        "\nFigures:"
    )


    figures = [

        "reputation_comparison_strategies.png",

        "score_comparison_strategies.png",

        "utility_comparison_strategies.png",

        "fairness_diversity_comparison_strategies.png",

    ]


    for figure in figures:

        print(

            f"  "
            f"{OUTPUT_DIR / figure}"

        )


if __name__ == "__main__":

    main()