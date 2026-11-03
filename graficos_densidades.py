# -*- coding: utf-8 -*-

"""
graficos_densidades.py

Comparação temporal entre diferentes cenários de Providers do SHADAI.

GRÁFICOS
===============================================================================

1. Average Reputation vs. Simulation Step
   - 100 Providers
   - 400 Providers
   - 800 Providers

2. Average Score vs. Simulation Step
   - 100 Providers
   - 400 Providers
   - 800 Providers

3. Average Utility vs. Simulation Step
   - 100 Providers
   - 400 Providers
   - 800 Providers


METODOLOGIA
===============================================================================

- 30 seeds independentes por cenário;
- rng_scheme = split_v2;
- pricing_mode = dynamic;
- attack_type = none;
- malicious_ratio = 0;

- janelas básicas de 50 simulation steps;

- suavização temporal:
      média móvel centrada de 3 janelas
      3 x 50 = 150 simulation steps;

- a suavização ocorre separadamente dentro de cada seed;

- somente depois as 30 seeds são agregadas;

- warm-up e drain-out são determinados por active_vehicles;

- cada cenário possui seu próprio intervalo operacional;

- os gráficos usam a interseção dos intervalos operacionais
  de 100 / 400 / 800 Providers.

IMPORTANTE
===============================================================================

Os valores 100 / 400 / 800 correspondem aos cenários
experimentais definidos por max_vehicles.

Na apresentação gráfica usamos a nomenclatura:

    100 Providers
    400 Providers
    800 Providers

Este script NÃO altera:
- simulações;
- analyzers;
- CSVs originais;
- RNG;
- lógica do SHADAI.
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

DATA_DIR = Path(
    "scalability_final"
)

OUTPUT_DIR = Path(
    "figures_densidades"
)


SCENARIOS = [
    100,
    400,
    800,
]


EXPECTED_SEEDS = set(
    range(1, 31)
)


EXPECTED_RNG = (
    "split_v2"
)

EXPECTED_PRICING_MODE = (
    "dynamic"
)

EXPECTED_ATTACK_TYPE = (
    "none"
)


# =============================================================================
# JANELAS TEMPORAIS
# =============================================================================

WINDOW_STEPS = 50


# =============================================================================
# SUAVIZAÇÃO TEMPORAL
# =============================================================================

# 3 janelas x 50 steps = 150 simulation steps.

SMOOTHING_WINDOWS = 3


# =============================================================================
# COBERTURA
# =============================================================================

MIN_SEED_COVERAGE = (
    0.80
)


MIN_OBSERVATION_SUPPORT_RATIO = (
    0.50
)


# =============================================================================
# INTERVALO OPERACIONAL
# =============================================================================

ACTIVE_MIN_SEED_COVERAGE = (
    0.80
)


ACTIVE_REFERENCE_QUANTILE = (
    0.75
)


ACTIVE_OPERATIONAL_FRACTION = (
    0.50
)


ACTIVE_CONSECUTIVE_WINDOWS = (
    3
)


# =============================================================================
# IC95%
# =============================================================================

# Mantemos as figuras principais sem IC
# para evitar sobreposição visual.

SHOW_REPUTATION_CI = False

SHOW_SCORE_CI = False

SHOW_UTILITY_CI = False


# =============================================================================
# EIXOS Y
# =============================================================================

# -------------------------------------------------------------------------
# Reputation
# -------------------------------------------------------------------------

REPUTATION_Y_MIN = (
    8.70
)

REPUTATION_Y_MAX = (
    9.20
)


# -------------------------------------------------------------------------
# Score
# -------------------------------------------------------------------------

SCORE_Y_MIN = (
    0.84
)

SCORE_Y_MAX = (
    0.90
)


# -------------------------------------------------------------------------
# Utility
# -------------------------------------------------------------------------

# A utility final é normalizada.
#
# Na primeira renderização deixamos automático.
# Depois de observar a figura, definimos uma escala fixa.

UTILITY_Y_MIN = 0.10

UTILITY_Y_MAX = 0.19


# =============================================================================
# ESTILO DOS CENÁRIOS
# =============================================================================

SCENARIO_STYLE = {

    100: {

        "label":
            "100 Providers",

        "color":
            "#B23A1B",

        "linestyle":
            "--",

        "linewidth":
            2.0,

    },


    400: {

        "label":
            "400 Providers",

        "color":
            "black",

        "linestyle":
            "-",

        "linewidth":
            2.4,

    },


    800: {

        "label":
            "800 Providers",

        "color":
            "#243B5A",

        "linestyle":
            ":",

        "linewidth":
            2.2,

    },

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


    # -------------------------------------------------------------------------
    # PNG
    # -------------------------------------------------------------------------

    fig.savefig(

        OUTPUT_DIR
        /
        f"{filename}.png",

        dpi=400,

        bbox_inches="tight",

    )


    # -------------------------------------------------------------------------
    # PDF
    # -------------------------------------------------------------------------

    fig.savefig(

        OUTPUT_DIR
        /
        f"{filename}.pdf",

        bbox_inches="tight",

    )


    # -------------------------------------------------------------------------
    # EPS
    # -------------------------------------------------------------------------

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
# PADRÕES DOS ARQUIVOS
# =============================================================================

def metric_pattern(
    scenario,
):

    return str(

        DATA_DIR

        /

        (
            f"metricas_shadai_{scenario}_"
            f"scalability_v{scenario}_"
            f"seed_*.csv"
        )

    )


def request_pattern(
    scenario,
):

    return str(

        DATA_DIR

        /

        (
            f"requests_shadai_{scenario}_"
            f"scalability_v{scenario}_"
            f"seed_*.csv"
        )

    )


# =============================================================================
# LOCALIZA MÉTRICAS
# =============================================================================

def find_metric_files(
    scenario,
):

    files = sorted(

        glob.glob(

            metric_pattern(
                scenario
            )

        )

    )


    if not files:

        raise FileNotFoundError(

            "\nNenhum arquivo de métricas encontrado.\n\n"

            f"Scenario: {scenario}\n"

            f"Pattern:\n"
            f"  {metric_pattern(scenario)}\n"

        )


    return [

        Path(
            file
        )

        for file
        in files

    ]


# =============================================================================
# LOCALIZA REQUESTS
# =============================================================================

def find_request_files(
    scenario,
):

    files = sorted(

        glob.glob(

            request_pattern(
                scenario
            )

        )

    )


    if not files:

        raise FileNotFoundError(

            "\nNenhum arquivo de requests encontrado.\n\n"

            f"Scenario: {scenario}\n"

            f"Pattern:\n"
            f"  {request_pattern(scenario)}\n"

        )


    return [

        Path(
            file
        )

        for file
        in files

    ]


# =============================================================================
# CARREGAMENTO DAS MÉTRICAS
# =============================================================================

def load_metric_scenario(
    scenario,
):
    """
    Carrega Reputation, Score e Utility
    das 30 seeds de um cenário.
    """

    required_columns = {

        "seed",

        "max_vehicles",

        "rng_scheme",

        "pricing_mode",

        "attack_type",

        "malicious_ratio",

        "step",

        "reputacao",

        "score",

        "utilidade",

    }


    datasets = {}


    for path in find_metric_files(
        scenario
    ):

        df = pd.read_csv(
            path
        )


        # =========================================================================
        # COLUNAS
        # =========================================================================

        missing = (

            required_columns
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


        if df.empty:

            raise RuntimeError(

                f"{path.name}: "
                "arquivo vazio."

            )


        # =========================================================================
        # METADADOS
        # =========================================================================

        seed = int(

            float(
                df[
                    "seed"
                ].iloc[0]
            )

        )


        max_vehicles = int(

            float(
                df[
                    "max_vehicles"
                ].iloc[0]
            )

        )


        rng = str(

            df[
                "rng_scheme"
            ].iloc[0]

        )


        pricing = str(

            df[
                "pricing_mode"
            ].iloc[0]

        ).lower()


        attack = str(

            df[
                "attack_type"
            ].iloc[0]

        ).lower()


        malicious_ratio = float(

            df[
                "malicious_ratio"
            ].iloc[0]

        )


        # =========================================================================
        # VALIDAÇÕES
        # =========================================================================

        if (
            max_vehicles
            !=
            scenario
        ):

            raise RuntimeError(

                f"{path.name}: "
                f"max_vehicles={max_vehicles}, "
                f"esperado={scenario}"

            )


        if (
            rng
            !=
            EXPECTED_RNG
        ):

            raise RuntimeError(

                f"{path.name}: "
                f"rng_scheme={rng}, "
                f"esperado={EXPECTED_RNG}"

            )


        if (
            pricing
            !=
            EXPECTED_PRICING_MODE
        ):

            raise RuntimeError(

                f"{path.name}: "
                f"pricing_mode={pricing}, "
                f"esperado={EXPECTED_PRICING_MODE}"

            )


        if (
            attack
            !=
            EXPECTED_ATTACK_TYPE
        ):

            raise RuntimeError(

                f"{path.name}: "
                f"attack_type={attack}, "
                f"esperado={EXPECTED_ATTACK_TYPE}"

            )


        if (

            abs(
                malicious_ratio
            )

            >
            1e-12

        ):

            raise RuntimeError(

                f"{path.name}: "
                f"malicious_ratio="
                f"{malicious_ratio}"

            )


        if seed in datasets:

            raise RuntimeError(

                f"Cenário {scenario}: "
                f"seed duplicada "
                f"{seed}"

            )


        # =========================================================================
        # DADOS
        # =========================================================================

        work = df[

            [
                "step",
                "reputacao",
                "score",
                "utilidade",
            ]

        ].copy()


        for column in [

            "step",

            "reputacao",

            "score",

            "utilidade",

        ]:

            work[
                column
            ] = pd.to_numeric(

                work[
                    column
                ],

                errors="coerce",

            )


        work = work.replace(

            [
                np.inf,
                -np.inf,
            ],

            np.nan,

        )


        # Cada métrica será tratada separadamente.
        # Exigimos aqui somente step válido.

        work = (

            work

            .dropna(
                subset=[
                    "step",
                ]
            )

        )


        if work.empty:

            raise RuntimeError(

                f"{path.name}: "
                "nenhum registro válido."

            )


        datasets[
            seed
        ] = work


    # =========================================================================
    # CONFERE AS SEEDS
    # =========================================================================

    observed = set(
        datasets
    )


    missing_seeds = sorted(

        EXPECTED_SEEDS
        -
        observed

    )


    extra_seeds = sorted(

        observed
        -
        EXPECTED_SEEDS

    )


    if missing_seeds:

        raise RuntimeError(

            f"Cenário {scenario}: "
            f"seeds ausentes: "
            f"{missing_seeds}"

        )


    if extra_seeds:

        raise RuntimeError(

            f"Cenário {scenario}: "
            f"seeds inesperadas: "
            f"{extra_seeds}"

        )


    print(

        f"[OK] Scenario {scenario}: "
        f"{len(datasets)} metric seeds"

    )


    return datasets


# =============================================================================
# CARREGAMENTO DA ATIVIDADE
# =============================================================================

def load_activity_scenario(
    scenario,
):

    frames = []


    for path in find_request_files(
        scenario
    ):

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


        if df.empty:

            continue


        seed = int(

            float(
                df[
                    "seed"
                ].iloc[0]
            )

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


        if work.empty:

            continue


        work[
            "seed"
        ] = seed


        frames.append(
            work
        )


    if not frames:

        raise RuntimeError(

            f"Cenário {scenario}: "
            "nenhum dado de atividade válido."

        )


    return pd.concat(

        frames,

        ignore_index=True,

    )


# =============================================================================
# RESUMO DA ATIVIDADE
# =============================================================================

def build_activity_summary(
    scenario,
):

    requests = (

        load_activity_scenario(
            scenario
        )

    )


    requests[
        "window_start"
    ] = (

        np.floor(

            requests[
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

        requests

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
    scenario,
):
    """
    Detecta warm-up e drain-out usando active_vehicles.

    As métricas Reputation, Score e Utility
    NÃO participam da definição do intervalo.
    """

    activity = (

        build_activity_summary(
            scenario
        )

    )


    work = (

        activity[

            activity[
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

            f"Scenario {scenario}: "
            "sem atividade suficiente."

        )


    # =========================================================================
    # REFERÊNCIA OPERACIONAL
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


    required_windows = (

        ACTIVE_CONSECUTIVE_WINDOWS

    )


    # =========================================================================
    # WARM-UP
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
            required_windows

        ]


        if (

            len(
                block
            )
            ==
            required_windows

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
    # DRAIN-OUT
    # =========================================================================

    operational_end = None


    for index in range(

        peak_position
        +
        1,

        len(work)
        -
        required_windows
        +
        1,

    ):

        block = values[

            index
            :
            index
            +
            required_windows

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

            f"Scenario {scenario}: "
            "início operacional não detectado."

        )


    if operational_end is None:

        raise RuntimeError(

            f"Scenario {scenario}: "
            "fim operacional não detectado."

        )


    return {

        "scenario":
            scenario,

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
# CONTEXTO OPERACIONAL COMUM
# =============================================================================

def prepare_operational_context():
    """
    Calcula uma única vez os intervalos operacionais
    dos três cenários.
    """

    records = []


    for scenario in SCENARIOS:

        records.append(

            detect_operational_interval(
                scenario
            )

        )


    operational_df = pd.DataFrame(
        records
    )


    common_start = float(

        operational_df[
            "operational_start"
        ]

        .max()

    )


    common_end = float(

        operational_df[
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

            "Não existe intervalo operacional comum "
            "entre 100 / 400 / 800 Providers."

        )


    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True,

    )


    operational_df.to_csv(

        OUTPUT_DIR
        /
        "provider_operational_intervals.csv",

        index=False,

    )


    return {

        "table":
            operational_df,

        "common_start":
            common_start,

        "common_end":
            common_end,

    }


# =============================================================================
# SUAVIZAÇÃO POR SEED
# =============================================================================

def smooth_seed_series(
    per_seed,
    mean_column,
):
    """
    Média móvel centrada de 3 janelas.

    3 x 50 = 150 simulation steps.

    O reindex impede que a média móvel atravesse
    artificialmente gaps temporais.
    """

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
            mean_column
        ]

        .rolling(

            window=(
                SMOOTHING_WINDOWS
            ),

            center=True,

            min_periods=(
                SMOOTHING_WINDOWS
            ),

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
# RESUMO GENÉRICO DE MÉTRICA
# =============================================================================

def build_metric_summary(
    scenario,
    metric_column,
):
    """
    Processamento utilizado por:

        reputacao
        score
        utilidade

    1. média por seed em janelas de 50 steps;
    2. suavização de 150 steps por seed;
    3. agregação entre as 30 seeds;
    4. cálculo de IC95%.
    """

    datasets = (

        load_metric_scenario(
            scenario
        )

    )


    frames = []


    # =========================================================================
    # CADA SEED
    # =========================================================================

    for seed in sorted(
        datasets
    ):

        work = (

            datasets[
                seed
            ][
                [
                    "step",
                    metric_column,
                ]
            ]

            .copy()

            .dropna(

                subset=[

                    "step",

                    metric_column,

                ]

            )

        )


        if work.empty:

            continue


        # =========================================================================
        # JANELAS DE 50 STEPS
        # =========================================================================

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

                    metric_column,

                    "mean",

                ),

                n_observations=(

                    metric_column,

                    "size",

                ),

            )

        )


        # =========================================================================
        # SUAVIZAÇÃO DE 150 STEPS
        # =========================================================================

        per_seed = (

            smooth_seed_series(

                per_seed,

                "metric_mean",

            )

        )


        per_seed[
            "seed"
        ] = seed


        per_seed[
            "scenario"
        ] = scenario


        frames.append(
            per_seed
        )


    if not frames:

        raise RuntimeError(

            f"Scenario {scenario}: "
            f"nenhum dado válido para "
            f"{metric_column}."

        )


    # =========================================================================
    # TODAS AS SEEDS
    # =========================================================================

    per_seed_df = pd.concat(

        frames,

        ignore_index=True,

    )


    # =========================================================================
    # AGREGAÇÃO ENTRE SEEDS
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
        "scenario"
    ] = scenario


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


    # =========================================================================
    # SUPORTE AMOSTRAL
    # =========================================================================

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

            f"Scenario {scenario}: "
            f"{metric_column} sem janelas "
            "com cobertura suficiente."

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

        "typical_observations":
            typical_observations,

        "minimum_observations":
            minimum_observations,

        "minimum_seed_count":
            minimum_seed_count,

    }


# =============================================================================
# PREPARA COMPARAÇÃO DE UMA MÉTRICA
# =============================================================================

def prepare_metric_comparison(
    metric_column,
    operational_context,
):

    summaries = {}

    per_seed_frames = []

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


    # =========================================================================
    # CENÁRIOS
    # =========================================================================

    for scenario in SCENARIOS:

        result = (

            build_metric_summary(

                scenario,

                metric_column,

            )

        )


        per_seed_df = (

            result[
                "per_seed"
            ]

        )


        summary = (

            result[
                "summary"
            ]

            .copy()

        )


        per_seed_frames.append(
            per_seed_df
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


        # =========================================================================
        # USADO NA FIGURA
        # =========================================================================

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


        summaries[
            scenario
        ] = summary


        valid = (

            summary[
                "used_in_plot"
            ]

        )


        if not valid.any():

            raise RuntimeError(

                f"Scenario {scenario}: "
                f"nenhum ponto válido para "
                f"{metric_column}."

            )


        # =========================================================================
        # DIAGNÓSTICOS
        # =========================================================================

        diagnostic_records.append(

            {

                "scenario":
                    scenario,

                "metric":
                    metric_column,

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

                "mean_raw":
                    float(

                        summary.loc[
                            valid,
                            "mean_raw",
                        ].mean()

                    ),

                "mean_smoothed":
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


    # =========================================================================
    # CONCATENA
    # =========================================================================

    all_summary = pd.concat(

        [

            summaries[
                scenario
            ]

            for scenario
            in SCENARIOS

        ],

        ignore_index=True,

    )


    all_per_seed = pd.concat(

        per_seed_frames,

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
):

    summaries = (

        data[
            "summaries"
        ]

    )


    common_start = (

        data[
            "common_start"
        ]

    )


    common_end = (

        data[
            "common_end"
        ]

    )


    # =========================================================================
    # FIGURA
    # =========================================================================

    fig, ax = plt.subplots(

        figsize=(
            10.2,
            5.8,
        )

    )


    # =========================================================================
    # CENÁRIOS
    # =========================================================================

    for scenario in SCENARIOS:

        summary = (

            summaries[
                scenario
            ]

            .copy()

        )


        style = (

            SCENARIO_STYLE[
                scenario
            ]

        )


        # =========================================================================
        # CURVA
        # =========================================================================

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

            label=(
                style[
                    "label"
                ]
            ),

            color=(
                style[
                    "color"
                ]
            ),

            linestyle=(
                style[
                    "linestyle"
                ]
            ),

            linewidth=(
                style[
                    "linewidth"
                ]
            ),

            zorder=3,

        )


        # =========================================================================
        # IC95% OPCIONAL
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
                    style[
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


    # -------------------------------------------------------------------------
    # Eixo Y
    #
    # Reputation e Score:
    #     fixos.
    #
    # Utility:
    #     automático na primeira renderização.
    # -------------------------------------------------------------------------

    if (

        y_min is not None

        and

        y_max is not None

    ):

        ax.set_ylim(

            y_min,

            y_max,

        )


    # =========================================================================
    # EIXO X
    # =========================================================================

    ax.set_xlim(

        common_start
        -
        WINDOW_STEPS
        *
        0.20,

        common_end
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
    #
    # Dentro do gráfico, canto superior direito.
    # =========================================================================

    ax.legend(

        loc="upper right",

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


    # =========================================================================
    # TICKS
    # =========================================================================

    ax.tick_params(

        axis="both",

        labelsize=13,

    )


    fig.tight_layout()


    return fig


# =============================================================================
# SALVA CSVs DE UMA MÉTRICA
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
        f"{prefix}_comparison_per_seed.csv",

        index=False,

    )


    data[
        "all_summary"
    ].to_csv(

        OUTPUT_DIR
        /
        f"{prefix}_comparison_all_windows.csv",

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
        f"{prefix}_comparison_plot_data.csv",

        index=False,

    )


    data[
        "diagnostics"
    ].to_csv(

        OUTPUT_DIR
        /
        f"{prefix}_comparison_diagnostics.csv",

        index=False,

    )


# =============================================================================
# DIAGNÓSTICOS
# =============================================================================

def print_metric_diagnostics(
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


    if (

        y_min is None

        or

        y_max is None

    ):

        print(

            "Y-axis range              : "
            "AUTO"

        )


    else:

        print(

            f"Y-axis range              : "
            f"{y_min} - {y_max}"

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


# =============================================================================
# GRÁFICO 1 — REPUTATION
# =============================================================================

def plot_reputation_comparison(
    operational_context,
):

    data = (

        prepare_metric_comparison(

            metric_column="reputacao",

            operational_context=
                operational_context,

        )

    )


    fig = (

        draw_metric_comparison(

            data=data,

            ylabel="Average Reputation",

            y_min=REPUTATION_Y_MIN,

            y_max=REPUTATION_Y_MAX,

            show_ci=SHOW_REPUTATION_CI,

        )

    )


    save_figure(

        fig,

        "reputation_comparison_providers",

    )


    plt.close(
        fig
    )


    save_metric_tables(

        data,

        "reputation",

    )


    print_metric_diagnostics(

        "REPUTATION COMPARISON",

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

            metric_column="score",

            operational_context=
                operational_context,

        )

    )


    fig = (

        draw_metric_comparison(

            data=data,

            ylabel="Average Score",

            y_min=SCORE_Y_MIN,

            y_max=SCORE_Y_MAX,

            show_ci=SHOW_SCORE_CI,

        )

    )


    save_figure(

        fig,

        "score_comparison_providers",

    )


    plt.close(
        fig
    )


    save_metric_tables(

        data,

        "score",

    )


    print_metric_diagnostics(

        "SCORE COMPARISON",

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

            metric_column="utilidade",

            operational_context=
                operational_context,

        )

    )


    fig = (

        draw_metric_comparison(

            data=data,

            ylabel="Average Utility",

            y_min=UTILITY_Y_MIN,

            y_max=UTILITY_Y_MAX,

            show_ci=SHOW_UTILITY_CI,

        )

    )


    save_figure(

        fig,

        "utility_comparison_providers",

    )


    plt.close(
        fig
    )


    save_metric_tables(

        data,

        "utility",

    )


    print_metric_diagnostics(

        "UTILITY COMPARISON",

        data,

        UTILITY_Y_MIN,

        UTILITY_Y_MAX,

    )


    print(

        "\n[DONE] "
        "Utility comparison generated."

    )


# =============================================================================
# RELATÓRIO DO INTERVALO OPERACIONAL
# =============================================================================

def print_operational_context(
    operational_context,
):

    print(
        "\n"
        +
        "=" * 92
    )


    print(
        "COMMON OPERATIONAL INTERVAL"
    )


    print(
        "=" * 92
    )


    print(

        operational_context[
            "table"
        ][

            [
                "scenario",
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
        "\nCommon interval:"
    )


    print(

        f"  Start : "
        f"{operational_context['common_start']:.0f}"

    )


    print(

        f"  End   : "
        f"{operational_context['common_end']:.0f}"

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
        "SHADAI - PROVIDER COMPARISON"
    )


    print(
        "#" * 92
    )


    # =========================================================================
    # INTERVALO OPERACIONAL
    #
    # Calculado uma única vez e reutilizado
    # por todos os gráficos.
    # =========================================================================

    operational_context = (

        prepare_operational_context()

    )


    print_operational_context(
        operational_context
    )


    # =========================================================================
    # GRÁFICO 1
    # REPUTATION
    # =========================================================================

    plot_reputation_comparison(
        operational_context
    )


    # =========================================================================
    # GRÁFICO 2
    # SCORE
    # =========================================================================

    plot_score_comparison(
        operational_context
    )


    # =========================================================================
    # GRÁFICO 3
    # UTILITY
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
        "ALL PROVIDER-COMPARISON FIGURES GENERATED"
    )


    print(
        "#" * 92
    )


    print(
        "\nFigures:"
    )


    print(

        f"  "
        f"{OUTPUT_DIR / 'reputation_comparison_providers.png'}"

    )


    print(

        f"  "
        f"{OUTPUT_DIR / 'score_comparison_providers.png'}"

    )


    print(

        f"  "
        f"{OUTPUT_DIR / 'utility_comparison_providers.png'}"

    )


if __name__ == "__main__":

    main()