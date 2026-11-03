# -*- coding: utf-8 -*-

"""
graficos_estrategias.py

Comparação interna entre estratégias/ablation do SHADAI.

FIGURAS
===============================================================================

1. Average Normalized Reputation vs. Simulation Step
   - SHADAI
   - Greedy
   - No Reputation

2. Average Score vs. Simulation Step
   - SHADAI
   - Greedy
   - No Reputation

3. Average Utility vs. Simulation Step
   - SHADAI
   - Greedy
   - No Reputation


IMPORTANTE
===============================================================================

REVS-based não é incluído neste bloco.

Ele será tratado separadamente como benchmark externo SHADAI vs. REVS-based.

Isso evita misturar:

- comparação interna / ablation;
- benchmark externo;
- scores com definições diferentes.


METODOLOGIA
===============================================================================

- cenário fixo: 400 Providers;
- 30 seeds;
- rng_scheme = split_v2;
- attack_type = none;
- malicious_ratio = 0;

- janelas temporais de 50 simulation steps;

- suavização:
      centered moving average
      3 janelas x 50 steps = 150 steps;

- suavização feita separadamente dentro de cada seed;

- somente depois ocorre agregação entre as seeds;

- warm-up e drain-out determinados por active_vehicles;

- os três gráficos usam o mesmo intervalo operacional comum;

- IC95% calculado, mas não exibido por padrão.

Nenhum CSV, analyzer ou resultado de simulação é alterado.
"""

from pathlib import Path
import glob
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURAÇÃO GERAL
# =============================================================================

OUTPUT_DIR = Path(
    "figures_estrategias"
)

EXPECTED_SEEDS = set(
    range(1, 31)
)

EXPECTED_RNG = (
    "split_v2"
)


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


# =============================================================================
# EIXOS Y DEFINITIVOS
# =============================================================================

# Reputation
REPUTATION_Y_MIN = 0.73
REPUTATION_Y_MAX = 0.95


# Score
SCORE_Y_MIN = 0.84
SCORE_Y_MAX = 0.93


# Utility
UTILITY_Y_MIN = 0.125
UTILITY_Y_MAX = 0.182


# =============================================================================
# ESTRATÉGIAS INTERNAS
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
            "black",

        "linestyle":
            "-",

        "linewidth":
            2.4,

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
            "#243B5A",

        "linestyle":
            ":",

        "linewidth":
            2.2,

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
            "#2F5D50",

        "linestyle":
            "-.",

        "linewidth":
            2.0,

    },

}


INTERNAL_STRATEGIES = [

    "SHADAI",

    "GREEDY",

    "NOREP",

]


# =============================================================================
# ALIASES DAS MÉTRICAS
# =============================================================================

METRIC_ALIASES = {

    "reputation": [

        "reputacao_norm",

        "reputation_norm",

        "normalized_reputation",

        "reputation_normalized",

    ],


    "score": [

        "score",

    ],


    "utility": [

        "utilidade",

        "utility",

        "utility_norm",

        "normalized_utility",

        "utilidade_norm",

    ],

}


# =============================================================================
# ESTILO GLOBAL
# =============================================================================

plt.rcParams.update({

    "font.family":
        "DejaVu Sans",

    "font.size":
        12,

    "axes.labelsize":
        15,

    "xtick.labelsize":
        13,

    "ytick.labelsize":
        13,

    "legend.fontsize":
        11.5,

    "pdf.fonttype":
        42,

    "ps.fonttype":
        42,

})


# =============================================================================
# SALVAR FIGURA
# =============================================================================

def save_figure(
    fig,
    filename,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    fig.savefig(

        OUTPUT_DIR
        /
        f"{filename}.png",

        dpi=400,

        bbox_inches="tight",

    )


    fig.savefig(

        OUTPUT_DIR
        /
        f"{filename}.pdf",

        bbox_inches="tight",

    )


    fig.savefig(

        OUTPUT_DIR
        /
        f"{filename}.eps",

        format="eps",

        bbox_inches="tight",

    )


# =============================================================================
# T CRÍTICO PARA IC95%
# =============================================================================

def t_critical_95(
    n,
):

    if n <= 1:

        return np.nan


    table = {

        23:
            2.068658,

        24:
            2.063899,

        25:
            2.059539,

        26:
            2.055529,

        27:
            2.051831,

        28:
            2.048407,

        29:
            2.045230,

    }


    df = int(
        n - 1
    )


    if df in table:

        return table[
            df
        ]


    if df > 29:

        return 1.96


    return 2.069


# =============================================================================
# LOCALIZA COLUNA DA MÉTRICA
# =============================================================================

def find_metric_column(
    df,
    metric_name,
):

    aliases = (

        METRIC_ALIASES[
            metric_name
        ]

    )


    lookup = {

        str(column)
        .strip()
        .lower():
            column

        for column
        in df.columns

    }


    for alias in aliases:

        key = (

            str(alias)
            .strip()
            .lower()

        )


        if key in lookup:

            return lookup[
                key
            ]


    raise RuntimeError(

        f"Métrica '{metric_name}' "
        "não encontrada.\n"

        f"Aliases procurados: "
        f"{aliases}\n"

        f"Colunas disponíveis: "
        f"{list(df.columns)}"

    )


# =============================================================================
# LOCALIZA ARQUIVOS
# =============================================================================

def get_files(
    strategy_key,
    kind,
):

    config = (

        STRATEGIES[
            strategy_key
        ]

    )


    if kind == "metric":

        pattern = (

            config[
                "metric_pattern"
            ]

        )


    elif kind == "request":

        pattern = (

            config[
                "request_pattern"
            ]

        )


    else:

        raise ValueError(

            f"kind inválido: "
            f"{kind}"

        )


    files = [

        Path(
            item
        )

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
            f"esperados 30 arquivos "
            f"'{kind}', encontrados "
            f"{len(files)}.\n"

            f"Pattern: "
            f"{pattern}"

        )


    return files


# =============================================================================
# VALIDA METADADOS
# =============================================================================

def validate_metadata(
    df,
    path,
):

    # =========================================================================
    # RNG
    # =========================================================================

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

            observed = (

                values.iloc[0]

            )


            if observed != EXPECTED_RNG:

                raise RuntimeError(

                    f"{path.name}: "
                    f"rng_scheme={observed}, "
                    f"esperado={EXPECTED_RNG}"

                )


    # =========================================================================
    # ATTACK
    # =========================================================================

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

            observed = (

                values.iloc[0]

            )


            allowed = {

                "none",

                "no_attack",

                "disabled",

                "off",

            }


            if observed not in allowed:

                raise RuntimeError(

                    f"{path.name}: "
                    f"attack_type="
                    f"{observed}"

                )


    # =========================================================================
    # MALICIOUS RATIO
    # =========================================================================

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


            if abs(
                observed
            ) > 1e-12:

                raise RuntimeError(

                    f"{path.name}: "
                    f"malicious_ratio="
                    f"{observed}"

                )


# =============================================================================
# CARREGA REQUESTS
# =============================================================================

def load_activity(
    strategy_key,
):

    files = (

        get_files(

            strategy_key,

            "request",

        )

    )


    frames = []

    observed_seeds = set()


    for path in files:

        df = pd.read_csv(
            path
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


        validate_metadata(
            df,
            path,
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
                f"seed duplicada "
                f"{seed}"

            )


        observed_seeds.add(
            seed
        )


        work = df[

            [
                "step",

                "active_vehicles",

            ]

        ].copy()


        work[
            "step"
        ] = pd.to_numeric(

            work[
                "step"
            ],

            errors="coerce",

        )


        work[
            "active_vehicles"
        ] = pd.to_numeric(

            work[
                "active_vehicles"
            ],

            errors="coerce",

        )


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
# RESUMO DA ATIVIDADE
# =============================================================================

def build_activity_summary(
    activity,
):

    work = (

        activity
        .copy()

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


    # =========================================================================
    # DENTRO DE CADA SEED
    # =========================================================================

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


    # =========================================================================
    # ENTRE SEEDS
    # =========================================================================

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
# DETECÇÃO DO INTERVALO OPERACIONAL
# =============================================================================

def detect_operational_interval(
    strategy_key,
):

    activity = (

        load_activity(
            strategy_key
        )

    )


    summary = (

        build_activity_summary(
            activity
        )

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


    # =========================================================================
    # REFERÊNCIA
    # =========================================================================

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


    needed = (

        ACTIVE_CONSECUTIVE_WINDOWS

    )


    # =========================================================================
    # INÍCIO
    # =========================================================================

    operational_start = None


    for index in range(

        0,

        peak_position
        +
        1,

    ):

        block = values[

            index
            :
            index
            +
            needed

        ]


        if (

            len(
                block
            )
            ==
            needed

            and

            np.all(

                block
                >=
                threshold

            )

        ):

            operational_start = float(

                steps[
                    index
                ]

            )

            break


    # =========================================================================
    # FIM
    # =========================================================================

    operational_end = None


    for index in range(

        peak_position
        +
        1,

        len(work)
        -
        needed
        +
        1,

    ):

        block = values[

            index
            :
            index
            +
            needed

        ]


        if np.all(

            block
            <
            threshold

        ):

            operational_end = float(

                steps[
                    index
                ]

            )

            break


    if operational_start is None:

        raise RuntimeError(

            f"{strategy_key}: "
            "início operacional "
            "não detectado."

        )


    if operational_end is None:

        raise RuntimeError(

            f"{strategy_key}: "
            "fim operacional "
            "não detectado."

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
# INTERVALO OPERACIONAL COMUM
# =============================================================================

def prepare_operational_context():

    records = []


    for strategy_key in INTERNAL_STRATEGIES:

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
        ]

        .max()

    )


    common_end = float(

        table[
            "operational_end"
        ]

        .min()

    )


    if (
        common_start
        >=
        common_end
    ):

        raise RuntimeError(

            "Não existe intervalo operacional "
            "comum entre as estratégias."

        )


    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True,

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
        "=" * 92
    )


    print(
        "COMMON OPERATIONAL INTERVAL — INTERNAL STRATEGIES"
    )


    print(
        "=" * 92
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

        .round(
            2
        )

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
# CARREGA UMA MÉTRICA
# =============================================================================

def load_metric(
    strategy_key,
    metric_name,
):

    files = (

        get_files(

            strategy_key,

            "metric",

        )

    )


    datasets = {}

    audit_records = []


    for path in files:

        df = pd.read_csv(
            path
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


        validate_metadata(
            df,
            path,
        )


        metric_column = (

            find_metric_column(

                df,

                metric_name,

            )

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
                f"seed duplicada "
                f"{seed}"

            )


        work = pd.DataFrame(

            {

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
                            metric_column
                        ],

                        errors="coerce",

                    ),

            }

        )


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


        # =========================================================================
        # VALIDAÇÃO EXTRA — REPUTATION NORMALIZADA
        # =========================================================================

        if metric_name == "reputation":

            if (

                work[
                    "metric"
                ].min()

                <
                -0.05

                or

                work[
                    "metric"
                ].max()

                >
                1.05

            ):

                raise RuntimeError(

                    f"{path.name}: "
                    "Reputation não parece "
                    "estar normalizada em [0,1]."

                )


        datasets[
            seed
        ] = work


        audit_records.append(

            {

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
                    str(
                        metric_column
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

            }

        )


    observed_seeds = set(
        datasets
    )


    missing_seeds = sorted(

        EXPECTED_SEEDS
        -
        observed_seeds

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
# SUAVIZAÇÃO TEMPORAL POR SEED
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


    if per_seed.empty:

        return per_seed


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


    work = (

        work

        .reset_index()

    )


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
# RESUMO DA MÉTRICA
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


    # =========================================================================
    # DENTRO DE CADA SEED
    # =========================================================================

    for seed in sorted(
        datasets
    ):

        work = (

            datasets[
                seed
            ]

            .copy()

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


        per_seed = (

            smooth_seed_series(
                per_seed
            )

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


    # =========================================================================
    # ENTRE SEEDS
    # =========================================================================

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

            median_observations_per_seed=(

                "n_observations",

                "median",

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


    # =========================================================================
    # COBERTURA
    # =========================================================================

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


    # =========================================================================
    # SUPORTE AMOSTRAL
    # =========================================================================

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


    # =========================================================================
    # IC95%
    # =========================================================================

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
# PREPARA COMPARAÇÃO
# =============================================================================

def prepare_metric_comparison(
    metric_name,
    operational_context,
):

    summaries = {}

    per_seed_frames = []

    audit_frames = []

    diagnostic_records = []


    common_start = (

        operational_context[
            "common_start"
        ]

    )


    common_end = (

        operational_context[
            "common_end"
        ]

    )


    for strategy_key in INTERNAL_STRATEGIES:

        result = (

            build_metric_summary(

                strategy_key,

                metric_name,

            )

        )


        summary = (

            result[
                "summary"
            ]

            .copy()

        )


        # =========================================================================
        # INTERVALO OPERACIONAL COMUM
        # =========================================================================

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


        valid = (

            summary[
                "used_in_plot"
            ]

        )


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


        diagnostic_records.append(

            {

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

                "typical_observations":
                    result[
                        "typical_observations"
                    ],

                "minimum_observations":
                    result[
                        "minimum_observations"
                    ],

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

            }

        )


    all_summary = pd.concat(

        [

            summaries[
                key
            ]

            for key
            in INTERNAL_STRATEGIES

        ],

        ignore_index=True,

    )


    all_per_seed = pd.concat(

        per_seed_frames,

        ignore_index=True,

    )


    all_audit = pd.concat(

        audit_frames,

        ignore_index=True,

    )


    diagnostics = pd.DataFrame(
        diagnostic_records
    )


    return {

        "summaries":
            summaries,

        "all_summary":
            all_summary,

        "all_per_seed":
            all_per_seed,

        "audit":
            all_audit,

        "diagnostics":
            diagnostics,

        "common_start":
            common_start,

        "common_end":
            common_end,

    }


# =============================================================================
# DESENHO GENÉRICO
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


    for strategy_key in INTERNAL_STRATEGIES:

        summary = (

            data[
                "summaries"
            ][
                strategy_key
            ]

        )


        config = (

            STRATEGIES[
                strategy_key
            ]

        )


        plot_y = np.where(

            summary[
                "used_in_plot"
            ],

            summary[
                "mean_smooth"
            ],

            np.nan,

        )


        # =========================================================================
        # CURVA
        # =========================================================================

        ax.plot(

            summary[
                "simulation_step"
            ],

            plot_y,

            label=(
                config[
                    "label"
                ]
            ),

            color=(
                config[
                    "color"
                ]
            ),

            linestyle=(
                config[
                    "linestyle"
                ]
            ),

            linewidth=(
                config[
                    "linewidth"
                ]
            ),

            zorder=3,

        )


        # =========================================================================
        # IC95%
        # =========================================================================

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

                color=(
                    config[
                        "color"
                    ]
                ),

                alpha=0.10,

                linewidth=0,

                zorder=1,

            )


    # =========================================================================
    # EIXOS
    # =========================================================================

    ax.set_xlabel(

        "Simulation Step",

        fontsize=15,

    )


    ax.set_ylabel(

        ylabel,

        fontsize=15,

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


    # =========================================================================
    # GRID
    # =========================================================================

    ax.grid(

        True,

        linestyle="--",

        linewidth=0.75,

        alpha=0.28,

    )


    ax.set_axisbelow(
        True
    )


    # =========================================================================
    # LEGENDA
    # =========================================================================

    ax.legend(

        loc=legend_location,

        frameon=True,

        framealpha=0.95,

        edgecolor="#D0D0D0",

        fontsize=11.5,

    )


    # =========================================================================
    # BORDAS
    # =========================================================================

    for spine in ax.spines.values():

        spine.set_color(
            "#505050"
        )

        spine.set_linewidth(
            0.8
        )


    ax.tick_params(

        axis="both",

        labelsize=13,

    )


    fig.tight_layout()


    return fig


# =============================================================================
# SALVAR TABELAS
# =============================================================================

def save_metric_tables(
    data,
    prefix,
):

    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True,

    )


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
# PRINT DIAGNÓSTICOS
# =============================================================================

def print_diagnostics(
    title,
    data,
    y_min,
    y_max,
):

    print(
        "\n"
        +
        "=" * 92
    )


    print(
        title
    )


    print(
        "=" * 92
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

        .round(
            4
        )

        .to_string(
            index=False
        )

    )


    print(
        "\nSource-column audit:"
    )


    source_audit = (

        data[
            "audit"
        ][

            [
                "strategy",

                "label",

                "metric",

                "source_column",

            ]

        ]

        .drop_duplicates()

    )


    print(

        source_audit

        .to_string(
            index=False
        )

    )


# =============================================================================
# GRÁFICO 1 — REPUTATION
# =============================================================================

def plot_reputation_comparison(
    operational_context,
):

    data = (

        prepare_metric_comparison(

            metric_name=
                "reputation",

            operational_context=
                operational_context,

        )

    )


    fig = (

        draw_metric_comparison(

            data=
                data,

            ylabel=
                "Average Normalized Reputation",

            y_min=
                REPUTATION_Y_MIN,

            y_max=
                REPUTATION_Y_MAX,

            show_ci=
                SHOW_REPUTATION_CI,

            legend_location=
                "upper right",

        )

    )


    save_figure(

        fig,

        "reputation_comparison_strategies",

    )


    plt.close(
        fig
    )


    save_metric_tables(

        data,

        "reputation",

    )


    print_diagnostics(

        "REPUTATION COMPARISON BETWEEN INTERNAL STRATEGIES",

        data,

        REPUTATION_Y_MIN,

        REPUTATION_Y_MAX,

    )


    print(

        "\n[DONE] "
        "Reputation comparison generated."

    )


# =============================================================================
# GRÁFICO 2 — SCORE
# =============================================================================

def plot_score_comparison(
    operational_context,
):

    data = (

        prepare_metric_comparison(

            metric_name=
                "score",

            operational_context=
                operational_context,

        )

    )


    fig = (

        draw_metric_comparison(

            data=
                data,

            ylabel=
                "Average Score",

            y_min=
                SCORE_Y_MIN,

            y_max=
                SCORE_Y_MAX,

            show_ci=
                SHOW_SCORE_CI,

            legend_location=
                "upper right",

        )

    )


    save_figure(

        fig,

        "score_comparison_strategies",

    )


    plt.close(
        fig
    )


    save_metric_tables(

        data,

        "score",

    )


    print_diagnostics(

        "SCORE COMPARISON BETWEEN INTERNAL STRATEGIES",

        data,

        SCORE_Y_MIN,

        SCORE_Y_MAX,

    )


    print(

        "\n[DONE] "
        "Score comparison generated."

    )


# =============================================================================
# GRÁFICO 3 — UTILITY
# =============================================================================

def plot_utility_comparison(
    operational_context,
):

    data = (

        prepare_metric_comparison(

            metric_name=
                "utility",

            operational_context=
                operational_context,

        )

    )


    fig = (

        draw_metric_comparison(

            data=
                data,

            ylabel=
                "Average Utility",

            y_min=
                UTILITY_Y_MIN,

            y_max=
                UTILITY_Y_MAX,

            show_ci=
                SHOW_UTILITY_CI,

            # ================================================================
            # ÚNICA FIGURA COM LEGENDA EMBAIXO À DIREITA
            # ================================================================

            legend_location=
                "lower right",

        )

    )


    save_figure(

        fig,

        "utility_comparison_strategies",

    )


    plt.close(
        fig
    )


    save_metric_tables(

        data,

        "utility",

    )


    print_diagnostics(

        "UTILITY COMPARISON BETWEEN INTERNAL STRATEGIES",

        data,

        UTILITY_Y_MIN,

        UTILITY_Y_MAX,

    )


    print(

        "\n[DONE] "
        "Utility comparison generated."

    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "\n"
        +
        "#" * 92
    )


    print(
        "SHADAI - INTERNAL STRATEGY COMPARISON"
    )


    print(
        "#" * 92
    )


    # =========================================================================
    # UM ÚNICO INTERVALO OPERACIONAL PARA AS TRÊS FIGURAS
    # =========================================================================

    operational_context = (

        prepare_operational_context()

    )


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
    # FINAL
    # =========================================================================

    print(
        "\n"
        +
        "#" * 92
    )


    print(
        "ALL INTERNAL STRATEGY-COMPARISON FIGURES GENERATED"
    )


    print(
        "#" * 92
    )


    print(
        "\nFigures:"
    )


    print(

        f"  "
        f"{OUTPUT_DIR / 'reputation_comparison_strategies.png'}"

    )


    print(

        f"  "
        f"{OUTPUT_DIR / 'score_comparison_strategies.png'}"

    )


    print(

        f"  "
        f"{OUTPUT_DIR / 'utility_comparison_strategies.png'}"

    )


if __name__ == "__main__":

    main()