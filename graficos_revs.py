# -*- coding: utf-8 -*-

"""
graficos_revs.py

Comparação entre:

    SHADAI
    REVS-based Provider Selection Benchmark

Campanha final C24.

FIGURAS
===============================================================================

1. Normalized Reputation
   - SHADAI
   - REVS-based

2. Available Energy at Selection
   - SHADAI
   - REVS-based

3. Cumulative Jain Fairness Index
   - SHADAI
   - REVS-based


IMPORTANTE
===============================================================================

Somente métricas diretamente comparáveis entre SHADAI e REVS-based
são utilizadas.

NÃO comparar diretamente:

- SHADAI score vs. REVS trust score
- SHADAI vehicle-to-task distance vs. REVS requester-to-provider distance


METODOLOGIA
===============================================================================

Cenário:
- 400 providers
- 30 paired seeds
- rng_scheme = split_v2
- dynamic pricing
- ataques desabilitados

Temporal:
- janela base = 50 simulation steps
- centered moving average = 3 janelas = 150 steps
- suavização realizada individualmente por seed
- agregação entre seeds somente após suavização
- warm-up e drain-out definidos por active_vehicles
- comparação restrita ao intervalo operacional comum

Fairness:
- utiliza a coluna "veiculo"
- mantém contagem acumulada de seleções por provedor
- Jain acumulado:

                  (sum x_i)^2
      J = ------------------------------
            n * sum(x_i^2)

  onde x_i representa o número acumulado de seleções do provedor i.

Nenhum CSV, analyzer ou resultado experimental é alterado.
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
    "figures_revs"
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

EXPECTED_RNG = "split_v2"

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

SHOW_ENERGY_CI = False

SHOW_FAIRNESS_CI = False


# =============================================================================
# LIMITES Y
# =============================================================================

REPUTATION_Y_MIN = 0.73
REPUTATION_Y_MAX = 0.92

FAIRNESS_Y_MIN = 0.84
FAIRNESS_Y_MAX = 1.005


# =============================================================================
# PALETA OFICIAL
# =============================================================================

BLACK = "#000000"

DARK_BLUE = "#243B5A"

LIGHT_GRID = "#DEDEDE"


# =============================================================================
# ESPESSURAS
# =============================================================================

LW_MAIN = 2.4

LW_DOTTED = 2.2


# =============================================================================
# MÉTODOS
# =============================================================================

METHODS = {

    "SHADAI": {

        "label":
            "SHADAI",

        "metric_pattern":
            "c24_revs_benchmark/"
            "metricas_shadai_400_c24_shadai_seed_*.csv",

        "request_pattern":
            "c24_revs_benchmark/"
            "requests_shadai_400_c24_shadai_seed_*.csv",

        "color":
            BLACK,

        "linestyle":
            "-",

        "linewidth":
            LW_MAIN,

    },


    "REVS": {

        "label":
            "REVS-based",

        "metric_pattern":
            "c24_revs_benchmark/"
            "metricas_revs_400_c24_revs_seed_*.csv",

        "request_pattern":
            "c24_revs_benchmark/"
            "requests_revs_400_c24_revs_seed_*.csv",

        "color":
            DARK_BLUE,

        "linestyle":
            ":",

        "linewidth":
            LW_DOTTED,

    },

}


METHOD_ORDER = [

    "SHADAI",

    "REVS",

]


# =============================================================================
# ALIASES
# =============================================================================

REPUTATION_ALIASES = [

    "reputacao_norm",

    "reputation_norm",

    "normalized_reputation",

    "reputation_normalized",

]


ENERGY_ALIASES = [

    "energia",

    "energy",

    "available_energy",

    "available_energy_at_selection",

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
# SAVE FIGURE
# =============================================================================

def save_figure(
    fig,
    filename,
):

    png_path = (

        OUTPUT_DIR
        /
        f"{filename}.png"

    )

    pdf_path = (

        OUTPUT_DIR
        /
        f"{filename}.pdf"

    )

    eps_path = (

        OUTPUT_DIR
        /
        f"{filename}.eps"

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
    method_key,
    kind,
):

    config = METHODS[
        method_key
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
        f"{config['label']:<12} "
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


# =============================================================================
# LOAD REQUEST ACTIVITY
# =============================================================================

def load_activity(
    method_key,
):

    files = get_files(

        method_key,

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


        seed = int(

            float(

                df[
                    "seed"
                ].iloc[0]

            )

        )


        if seed in observed_seeds:

            raise RuntimeError(

                f"{method_key}: "
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

            f"{method_key}: "
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
    method_key,
):

    activity = load_activity(
        method_key
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

            f"{method_key}: "
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


    # =========================================================================
    # START
    # =========================================================================

    operational_start = None


    for index in range(

        0,

        peak_position + 1,

    ):

        block = values[

            index:
            index
            +
            needed

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


    # =========================================================================
    # END
    # =========================================================================

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
            index
            +
            needed

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

            f"{method_key}: "
            "início operacional não detectado."

        )


    if operational_end is None:

        raise RuntimeError(

            f"{method_key}: "
            "fim operacional não detectado."

        )


    return {

        "method":
            method_key,

        "label":
            METHODS[
                method_key
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


    for method_key in METHOD_ORDER:

        records.append(

            detect_operational_interval(
                method_key
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
            "comum entre SHADAI e REVS-based."

        )


    table.to_csv(

        OUTPUT_DIR
        /
        "operational_intervals_shadai_vs_revs.csv",

        index=False,

    )


    print(
        "\n"
        +
        "=" * 100
    )


    print(
        "COMMON OPERATIONAL INTERVAL — SHADAI VS REVS-BASED"
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
# LOAD TEMPORAL METRIC
# =============================================================================

def load_temporal_metric(
    method_key,
    aliases,
    metric_name,
):

    files = get_files(

        method_key,

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


        column = find_column(

            df,

            aliases,

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

                f"{method_key}: "
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


            "metric":

                pd.to_numeric(

                    df[
                        column
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


        first_window = int(

            per_seed[
                "window_start"
            ].min()

        )


        last_window = int(

            per_seed[
                "window_start"
            ].max()

        )


        complete_index = np.arange(

            first_window,

            last_window
            +
            WINDOW_STEPS,

            WINDOW_STEPS,

        )


        per_seed = (

            per_seed

            .set_index(
                "window_start"
            )

            .reindex(
                complete_index
            )

        )


        per_seed.index.name = (
            "window_start"
        )


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


        per_seed = (
            per_seed
            .reset_index()
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


        per_seed[
            "seed"
        ] = seed


        per_seed[
            "method"
        ] = method_key


        frames.append(
            per_seed
        )


        audit_records.append({

            "method":
                method_key,

            "label":
                METHODS[
                    method_key
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
                str(
                    column
                ),

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
        observed_seeds

    )


    if missing_seeds:

        raise RuntimeError(

            f"{method_key}: "
            f"seeds ausentes para "
            f"{metric_name}: "
            f"{missing_seeds}"

        )


    print(

        f"[OK] "
        f"{METHODS[method_key]['label']}: "
        f"{len(observed_seeds)} seeds "
        f"for {metric_name}"

    )


    return (

        pd.concat(

            frames,

            ignore_index=True,

        ),

        pd.DataFrame(
            audit_records
        ),

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
# CUMULATIVE JAIN — ONE SEED
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


    grouped = {

        int(window):
            group

        for window, group
        in work.groupby(
            "window_start"
        )

    }


    provider_counts = {}

    rows = []


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
# LOAD FAIRNESS
# =============================================================================

def load_fairness(
    method_key,
):

    files = get_files(

        method_key,

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

                f"{method_key}: "
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
            "method"
        ] = method_key


        frames.append(
            per_seed
        )


        audit_records.append({

            "method":
                method_key,

            "label":
                METHODS[
                    method_key
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
                float(

                    per_seed[
                        "metric_mean"
                    ]

                    .dropna()

                    .iloc[-1]

                ),

            "final_unique_providers":
                int(

                    per_seed[
                        "cumulative_unique_providers"
                    ]

                    .iloc[-1]

                ),

        })


    missing_seeds = sorted(

        EXPECTED_SEEDS
        -
        observed_seeds

    )


    if missing_seeds:

        raise RuntimeError(

            f"{method_key}: "
            f"seeds ausentes para fairness: "
            f"{missing_seeds}"

        )


    print(

        f"[OK] "
        f"{METHODS[method_key]['label']}: "
        f"{len(observed_seeds)} fairness seeds"

    )


    return (

        pd.concat(

            frames,

            ignore_index=True,

        ),

        pd.DataFrame(
            audit_records
        ),

    )


# =============================================================================
# SUMMARIZE TEMPORAL DATA
# =============================================================================

def summarize_temporal(
    per_seed,
    operational_context,
    cumulative=False,
):

    summary = (

        per_seed

        .groupby(

            "simulation_step",

            as_index=False,

        )

        .agg(

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


    if cumulative:

        summary[
            "passes_observation_support"
        ] = True


    else:

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

                "Nenhuma janela com "
                "cobertura suficiente."

            )


        typical_observations = float(

            reference[
                "total_observations"
            ]

            .median()

        )


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
        "inside_common_operational_interval"
    ] = (

        (
            summary[
                "simulation_step"
            ]

            >=

            operational_context[
                "common_start"
            ]

        )

        &

        (
            summary[
                "simulation_step"
            ]

            <

            operational_context[
                "common_end"
            ]

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


    return summary


# =============================================================================
# PREPARE METRIC
# =============================================================================

def prepare_standard_metric(
    aliases,
    metric_name,
    operational_context,
):

    summaries = {}

    per_seed_frames = []

    audit_frames = []


    for method_key in METHOD_ORDER:

        (
            per_seed,
            audit,

        ) = load_temporal_metric(

            method_key=
                method_key,

            aliases=
                aliases,

            metric_name=
                metric_name,

        )


        summary = summarize_temporal(

            per_seed=
                per_seed,

            operational_context=
                operational_context,

            cumulative=False,

        )


        summaries[
            method_key
        ] = summary


        per_seed[
            "method"
        ] = method_key


        audit[
            "method"
        ] = method_key


        per_seed_frames.append(
            per_seed
        )


        audit_frames.append(
            audit
        )


    return {

        "summaries":
            summaries,

        "per_seed":

            pd.concat(

                per_seed_frames,

                ignore_index=True,

            ),

        "audit":

            pd.concat(

                audit_frames,

                ignore_index=True,

            ),

    }


# =============================================================================
# PREPARE FAIRNESS
# =============================================================================

def prepare_fairness(
    operational_context,
):

    summaries = {}

    per_seed_frames = []

    audit_frames = []


    for method_key in METHOD_ORDER:

        (
            per_seed,
            audit,

        ) = load_fairness(
            method_key
        )


        summary = summarize_temporal(

            per_seed=
                per_seed,

            operational_context=
                operational_context,

            cumulative=True,

        )


        summaries[
            method_key
        ] = summary


        per_seed_frames.append(
            per_seed
        )


        audit_frames.append(
            audit
        )


    return {

        "summaries":
            summaries,

        "per_seed":

            pd.concat(

                per_seed_frames,

                ignore_index=True,

            ),

        "audit":

            pd.concat(

                audit_frames,

                ignore_index=True,

            ),

    }


# =============================================================================
# DRAW COMPARISON
# =============================================================================

def draw_comparison(
    data,
    operational_context,
    ylabel,
    filename,
    show_ci=False,
    y_min=None,
    y_max=None,
    legend_location="best",
):

    fig, ax = plt.subplots(

        figsize=(
            10.2,
            5.8,
        )

    )


    all_values = []


    for method_key in METHOD_ORDER:

        summary = data[
            "summaries"
        ][
            method_key
        ]


        config = METHODS[
            method_key
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


        valid_values = plot_y[

            np.isfinite(
                plot_y
            )

        ]


        if len(valid_values) > 0:

            all_values.extend(

                valid_values.tolist()

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

            valid = (

                summary[
                    "used_in_plot"
                ]

            )


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


    ax.set_xlim(

        operational_context[
            "common_start"
        ]

        -
        WINDOW_STEPS
        *
        0.20,

        operational_context[
            "common_end"
        ]

        -
        WINDOW_STEPS
        *
        0.30,

    )


    if (

        y_min is not None

        and

        y_max is not None

    ):

        ax.set_ylim(

            y_min,

            y_max,

        )


    elif all_values:

        minimum = float(
            np.min(
                all_values
            )
        )


        maximum = float(
            np.max(
                all_values
            )
        )


        span = (

            maximum
            -
            minimum

        )


        padding = max(

            span
            *
            0.10,

            1.0,

        )


        ax.set_ylim(

            minimum
            -
            padding,

            maximum
            +
            padding,

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


    save_figure(

        fig,

        filename,

    )


    plt.close(
        fig
    )


# =============================================================================
# SAVE DATA
# =============================================================================

def save_metric_data(
    data,
    prefix,
):

    data[
        "per_seed"
    ].to_csv(

        OUTPUT_DIR
        /
        f"{prefix}_per_seed.csv",

        index=False,

    )


    data[
        "audit"
    ].to_csv(

        OUTPUT_DIR
        /
        f"{prefix}_audit.csv",

        index=False,

    )


    summary_frames = []


    for method_key in METHOD_ORDER:

        summary = (

            data[
                "summaries"
            ][
                method_key
            ]

            .copy()

        )


        summary[
            "method"
        ] = method_key


        summary[
            "label"
        ] = METHODS[
            method_key
        ][
            "label"
        ]


        summary_frames.append(
            summary
        )


    all_summary = pd.concat(

        summary_frames,

        ignore_index=True,

    )


    all_summary.to_csv(

        OUTPUT_DIR
        /
        f"{prefix}_all_windows.csv",

        index=False,

    )


    all_summary[

        all_summary[
            "used_in_plot"
        ]

    ].to_csv(

        OUTPUT_DIR
        /
        f"{prefix}_plot_data.csv",

        index=False,

    )


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def print_diagnostics(
    title,
    data,
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


    rows = []


    for method_key in METHOD_ORDER:

        summary = data[
            "summaries"
        ][
            method_key
        ]


        valid = summary[
            "used_in_plot"
        ]


        rows.append({

            "method":
                METHODS[
                    method_key
                ][
                    "label"
                ],

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

            "mean":
                float(

                    summary.loc[
                        valid,
                        "mean_smooth",
                    ].mean()

                ),

            "minimum":
                float(

                    summary.loc[
                        valid,
                        "mean_smooth",
                    ].min()

                ),

            "maximum":
                float(

                    summary.loc[
                        valid,
                        "mean_smooth",
                    ].max()

                ),

        })


    table = pd.DataFrame(
        rows
    )


    print(

        table

        .round(4)

        .to_string(
            index=False
        )

    )


# =============================================================================
# FIGURE 1 — REPUTATION
# =============================================================================

def plot_reputation(
    operational_context,
):

    data = prepare_standard_metric(

        aliases=
            REPUTATION_ALIASES,

        metric_name=
            "reputation",

        operational_context=
            operational_context,

    )


    draw_comparison(

        data=
            data,

        operational_context=
            operational_context,

        ylabel=
            "Average Normalized Reputation",

        filename=
            "reputation_shadai_vs_revs",

        show_ci=
            SHOW_REPUTATION_CI,

        y_min=
            REPUTATION_Y_MIN,

        y_max=
            REPUTATION_Y_MAX,

        legend_location=
            "lower right",

    )


    save_metric_data(

        data,

        "reputation_shadai_vs_revs",

    )


    print_diagnostics(

        "NORMALIZED REPUTATION — SHADAI VS REVS-BASED",

        data,

    )


# =============================================================================
# FIGURE 2 — AVAILABLE ENERGY AT SELECTION
# =============================================================================

def plot_energy(
    operational_context,
):

    data = prepare_standard_metric(

        aliases=
            ENERGY_ALIASES,

        metric_name=
            "energy",

        operational_context=
            operational_context,

    )


    draw_comparison(

        data=
            data,

        operational_context=
            operational_context,

        ylabel=
            "Available Energy at Selection",

        filename=
            "energy_shadai_vs_revs",

        show_ci=
            SHOW_ENERGY_CI,

        y_min=
            None,

        y_max=
            None,

        legend_location=
            "upper right",

    )


    save_metric_data(

        data,

        "energy_shadai_vs_revs",

    )


    print_diagnostics(

        "AVAILABLE ENERGY AT SELECTION — SHADAI VS REVS-BASED",

        data,

    )


# =============================================================================
# FIGURE 3 — CUMULATIVE JAIN FAIRNESS
# =============================================================================

def plot_fairness(
    operational_context,
):

    data = prepare_fairness(
        operational_context
    )


    draw_comparison(

        data=
            data,

        operational_context=
            operational_context,

        ylabel=
            "Cumulative Jain Fairness Index",

        filename=
            "fairness_shadai_vs_revs",

        show_ci=
            SHOW_FAIRNESS_CI,

        y_min=
            FAIRNESS_Y_MIN,

        y_max=
            FAIRNESS_Y_MAX,

        legend_location=
            "lower right",

    )


    save_metric_data(

        data,

        "fairness_shadai_vs_revs",

    )


    print_diagnostics(

        "CUMULATIVE JAIN FAIRNESS — SHADAI VS REVS-BASED",

        data,

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
        "SHADAI VS REVS-BASED PROVIDER-SELECTION BENCHMARK"
    )


    print(
        "#" * 100
    )


    print(
        "\nDirectly comparable temporal metrics:"
    )


    print(
        "  1. Normalized Reputation"
    )


    print(
        "  2. Available Energy at Selection"
    )


    print(
        "  3. Cumulative Jain Fairness Index"
    )


    print(
        "\nExcluded from direct comparison:"
    )


    print(
        "  - raw method score"
    )


    print(
        "  - distance"
    )


    # =========================================================================
    # OPERATIONAL CONTEXT
    # =========================================================================

    operational_context = (

        prepare_operational_context()

    )


    # =========================================================================
    # FIGURE 1 — REPUTATION
    # =========================================================================

    plot_reputation(

        operational_context

    )


    # =========================================================================
    # FIGURE 2 — ENERGY
    # =========================================================================

    plot_energy(

        operational_context

    )


    # =========================================================================
    # FIGURE 3 — FAIRNESS
    # =========================================================================

    plot_fairness(

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
        "ALL SHADAI VS REVS FIGURES GENERATED"
    )


    print(
        "#" * 100
    )


    print(
        "\nFigures:"
    )


    figures = [

        "reputation_shadai_vs_revs.png",

        "energy_shadai_vs_revs.png",

        "fairness_shadai_vs_revs.png",

    ]


    for figure in figures:

        print(

            f"  "
            f"{OUTPUT_DIR / figure}"

        )


if __name__ == "__main__":

    main()