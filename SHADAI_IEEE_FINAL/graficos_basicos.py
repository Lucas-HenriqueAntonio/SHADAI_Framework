# -*- coding: utf-8 -*-

"""
graficos_basicos.py

Gráficos básicos do SHADAI:
1) Correlation Analysis
2) Score Evolution

O gráfico temporal usa apenas o intervalo operacional sustentado:
- warm-up cutoff: primeira sequência de janelas com atividade >= limiar;
- drain-out cutoff: primeira sequência, após o pico, com atividade < limiar.

O Score NÃO participa da definição desses cortes.
Nenhuma simulação, analyzer ou CSV original é alterado.
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

DATA_DIR = Path("scalability_final")
OUTPUT_DIR = Path("figures_basicos")

EXPECTED_SEEDS = set(range(1, 31))

EXPECTED_RNG = "split_v2"
EXPECTED_MAX_VEHICLES = 400
EXPECTED_PRICING_MODE = "dynamic"
EXPECTED_ATTACK_TYPE = "none"


# =============================================================================
# CORRELAÇÃO
# =============================================================================

CORRELATION_METHOD = "fisher"

SHOW_CORRELATION_TITLE = True

VARIABLES = [

    (
        "score",
        "Score",
    ),

    (
        "utilidade",
        "Utility",
    ),

    (
        "reputacao",
        "Reputation",
    ),

    (
        "energia",
        "Energy",
    ),

]


# =============================================================================
# SCORE EVOLUTION
# =============================================================================

SCORE_WINDOW_STEPS = 50


# Exige pelo menos 80% das 30 seeds.
#
# 30 × 0.80 = 24 seeds.

MIN_SCORE_SEED_COVERAGE = 0.80


# Suporte mínimo relativo ao suporte típico.

MIN_SCORE_OBS_SUPPORT_RATIO = 0.50


# Mantemos False para a figura principal.
#
# O IC95% continua sendo calculado e salvo.

SHOW_SCORE_CI = False


# =============================================================================
# INTERVALO OPERACIONAL
# =============================================================================

# Cobertura mínima das seeds para considerar
# uma janela válida para análise de atividade.

ACTIVE_MIN_SEED_COVERAGE = 0.80


# Referência operacional:
#
# percentil 75 da mediana de active_vehicles.

ACTIVE_REFERENCE_QUANTILE = 0.75


# Limiar operacional:
#
# 50% da atividade de referência.
#
# O MESMO limiar é usado para:
#
# - fim do warm-up;
# - início do drain-out.

ACTIVE_OPERATIONAL_FRACTION = 0.50


# Persistência:
#
# 3 janelas consecutivas.
#
# 3 × 50 = 150 simulation steps.

ACTIVE_CONSECUTIVE_WINDOWS = 3


# Salva gráfico completo para auditoria.

SAVE_FULL_SCORE_DIAGNOSTIC = True


# =============================================================================
# ESTILO GLOBAL
# =============================================================================

plt.rcParams.update({

    "font.family":
        "DejaVu Sans",

    "font.size":
        12,

    "axes.titlesize":
        15,

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
# AUXILIARES
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
# T CRÍTICO
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
# ARQUIVOS
# =============================================================================

def find_baseline_metric_files():

    pattern = str(

        DATA_DIR
        /
        "metricas_shadai_400_"
        "scalability_v400_seed_*.csv"

    )


    files = sorted(

        glob.glob(
            pattern
        )

    )


    if not files:

        raise FileNotFoundError(

            "\nNenhum arquivo de métricas "
            "do baseline final foi encontrado.\n\n"

            f"Padrão procurado:\n"
            f"  {pattern}\n\n"

            "Execute este script a partir da raiz:\n"
            "  SHADAI_IEEE_FINAL\n"

        )


    return [

        Path(
            file
        )

        for file
        in files

    ]


def request_path_for_seed(
    seed,
):

    return (

        DATA_DIR

        /

        (
            f"requests_shadai_400_"
            f"scalability_v400_seed_{seed:02d}.csv"
        )

    )


# =============================================================================
# CARREGAMENTO DAS MÉTRICAS
# =============================================================================

def load_and_validate_seed_files():

    required = {

        "seed",

        "max_vehicles",

        "rng_scheme",

        "pricing_mode",

        "attack_type",

        "malicious_ratio",

        "step",

        "score",

        "utilidade",

        "reputacao",

        "energia",

    }


    datasets = {}

    rejected = []


    for path in find_baseline_metric_files():

        df = pd.read_csv(
            path
        )


        missing = (

            required
            -
            set(
                df.columns
            )

        )


        if missing:

            rejected.append(

                (
                    path.name,

                    (
                        "colunas ausentes: "
                        f"{sorted(missing)}"
                    ),

                )

            )

            continue


        if df.empty:

            rejected.append(

                (
                    path.name,
                    "arquivo vazio",
                )

            )

            continue


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


        valid = (

            max_vehicles
            ==
            EXPECTED_MAX_VEHICLES

            and

            rng
            ==
            EXPECTED_RNG

            and

            pricing
            ==
            EXPECTED_PRICING_MODE

            and

            attack
            ==
            EXPECTED_ATTACK_TYPE

            and

            abs(
                malicious_ratio
            )
            <
            1e-12

        )


        if not valid:

            rejected.append(

                (
                    path.name,

                    (
                        "metadados incompatíveis: "
                        f"seed={seed}, "
                        f"max={max_vehicles}, "
                        f"rng={rng}, "
                        f"pricing={pricing}, "
                        f"attack={attack}, "
                        f"malicious_ratio={malicious_ratio}"
                    ),

                )

            )

            continue


        work = df[

            [
                "step",
                "score",
                "utilidade",
                "reputacao",
                "energia",
            ]

        ].copy()


        for column in work.columns:

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


        if len(
            work
        ) < 3:

            rejected.append(

                (
                    path.name,
                    "menos de 3 observações válidas",
                )

            )

            continue


        datasets[
            seed
        ] = work


    # =========================================================================
    # RELATÓRIO
    # =========================================================================

    if rejected:

        print(
            "\nArquivos rejeitados:"
        )

        for (
            name,
            reason,
        ) in rejected:

            print(

                f"  [REJECT] "
                f"{name}: "
                f"{reason}"

            )


    observed = set(
        datasets
    )


    missing = sorted(

        EXPECTED_SEEDS
        -
        observed

    )


    extra = sorted(

        observed
        -
        EXPECTED_SEEDS

    )


    print(
        "\n"
        +
        "=" * 92
    )


    print(
        "SHADAI - BASIC FIGURES"
    )


    print(
        "=" * 92
    )


    print(

        f"Baseline                : "
        f"{EXPECTED_MAX_VEHICLES} "
        f"active-provider cap"

    )


    print(

        f"RNG scheme              : "
        f"{EXPECTED_RNG}"

    )


    print(

        f"Pricing mode            : "
        f"{EXPECTED_PRICING_MODE}"

    )


    print(

        f"Attack type             : "
        f"{EXPECTED_ATTACK_TYPE}"

    )


    print(

        f"Valid seeds             : "
        f"{len(observed)}/30"

    )


    print(

        f"Total metric records    : "
        f"{sum(len(x) for x in datasets.values())}"

    )


    if missing:

        print(

            f"[WARNING] "
            f"Missing seeds: "
            f"{missing}"

        )


    if extra:

        print(

            f"[WARNING] "
            f"Extra seeds: "
            f"{extra}"

        )


    if not datasets:

        raise RuntimeError(

            "Nenhuma seed válida encontrada."

        )


    return datasets


# =============================================================================
# CORRELAÇÃO — FISHER
# =============================================================================

def fisher_aggregate_correlation(
    datasets,
):

    cols = [

        item[0]

        for item
        in VARIABLES

    ]


    labels = [

        item[1]

        for item
        in VARIABLES

    ]


    pairs = {

        (
            i,
            j,
        ): []

        for i
        in range(
            len(cols)
        )

        for j
        in range(
            i + 1,
            len(cols)
        )

    }


    per_seed_records = []


    for seed in sorted(
        datasets
    ):

        df = (

            datasets[
                seed
            ][cols]

            .copy()

            .dropna()

        )


        if len(
            df
        ) < 3:

            continue


        corr = df.corr(
            method="pearson"
        )


        record = {

            "seed":
                seed,

            "n_records":
                len(df),

        }


        for i in range(
            len(cols)
        ):

            for j in range(

                i + 1,
                len(cols),

            ):

                r = float(

                    corr.iloc[
                        i,
                        j
                    ]

                )


                if np.isfinite(
                    r
                ):

                    pairs[

                        (
                            i,
                            j,
                        )

                    ].append(

                        float(

                            np.clip(

                                r,

                                -0.999999,

                                0.999999,

                            )

                        )

                    )


                record[

                    f"{labels[i]}"
                    f"__"
                    f"{labels[j]}"

                ] = r


        per_seed_records.append(
            record
        )


    result = np.eye(

        len(cols),

        dtype=float,

    )


    for i in range(
        len(cols)
    ):

        for j in range(

            i + 1,
            len(cols),

        ):

            values = np.asarray(

                pairs[
                    (
                        i,
                        j,
                    )
                ],

                dtype=float,

            )


            if len(
                values
            ) == 0:

                aggregate = (
                    np.nan
                )

            else:

                aggregate = float(

                    np.tanh(

                        np.mean(

                            np.arctanh(
                                values
                            )

                        )

                    )

                )


            result[
                i,
                j
            ] = aggregate


            result[
                j,
                i
            ] = aggregate


    matrix = pd.DataFrame(

        result,

        index=labels,

        columns=labels,

    )


    per_seed_df = pd.DataFrame(
        per_seed_records
    )


    return (

        matrix,

        per_seed_df,

    )


# =============================================================================
# CORRELAÇÃO — POOLED
# =============================================================================

def pooled_correlation(
    datasets,
):

    cols = [

        item[0]

        for item
        in VARIABLES

    ]


    labels = [

        item[1]

        for item
        in VARIABLES

    ]


    pooled = pd.concat(

        [

            datasets[
                seed
            ][cols]

            for seed
            in sorted(
                datasets
            )

        ],

        ignore_index=True,

    ).dropna()


    matrix = pooled.corr(
        method="pearson"
    )


    matrix.index = (
        labels
    )


    matrix.columns = (
        labels
    )


    return matrix


# =============================================================================
# FIGURA — CORRELAÇÃO
# =============================================================================

def draw_correlation_matrix(
    matrix,
):

    labels = list(
        matrix.columns
    )


    fig, ax = plt.subplots(

        figsize=(
            8.0,
            6.2,
        )

    )


    image = ax.imshow(

        matrix.to_numpy(
            dtype=float
        ),

        interpolation="nearest",

        cmap="RdBu_r",

        vmin=-1.0,

        vmax=1.0,

        aspect="equal",

    )


    cbar = fig.colorbar(

        image,

        ax=ax,

        fraction=0.046,

        pad=0.055,

    )


    cbar.set_ticks(

        np.linspace(

            -1.0,

            1.0,

            9,

        )

    )


    cbar.ax.tick_params(
        labelsize=10
    )


    ax.set_xticks(
        range(
            len(labels)
        )
    )


    ax.set_yticks(
        range(
            len(labels)
        )
    )


    ax.set_xticklabels(
        labels
    )


    ax.set_yticklabels(
        labels
    )


    ax.set_xticks(

        np.arange(

            -0.5,

            len(labels),

            1,

        ),

        minor=True,

    )


    ax.set_yticks(

        np.arange(

            -0.5,

            len(labels),

            1,

        ),

        minor=True,

    )


    ax.grid(

        which="minor",

        color="#505050",

        linestyle="-",

        linewidth=0.65,

        alpha=0.65,

    )


    ax.tick_params(

        which="minor",

        bottom=False,

        left=False,

    )


    for i in range(
        len(labels)
    ):

        for j in range(
            len(labels)
        ):

            value = matrix.iloc[
                i,
                j
            ]


            ax.text(

                j,

                i,

                (
                    f"{value:.2f}"
                    if np.isfinite(value)
                    else "NaN"
                ),

                ha="center",

                va="center",

                color="black",

                fontsize=11.5,

                fontweight="bold",

            )


    if SHOW_CORRELATION_TITLE:

        ax.set_title(

            "Correlation Analysis",

            pad=14,

            fontweight="normal",

        )


    ax.set_xlim(

        -0.5,

        len(labels)
        -
        0.5,

    )


    ax.set_ylim(

        len(labels)
        -
        0.5,

        -0.5,

    )


    fig.tight_layout()


    return fig


# =============================================================================
# EXECUÇÃO DA CORRELAÇÃO
# =============================================================================

def plot_correlation_analysis(
    datasets,
):

    fisher_matrix, per_seed = (

        fisher_aggregate_correlation(
            datasets
        )

    )


    pooled_matrix = (

        pooled_correlation(
            datasets
        )

    )


    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True,

    )


    fisher_matrix.to_csv(

        OUTPUT_DIR
        /
        "correlation_matrix_fisher.csv",

        index=True,

    )


    pooled_matrix.to_csv(

        OUTPUT_DIR
        /
        "correlation_matrix_pooled_reference.csv",

        index=True,

    )


    per_seed.to_csv(

        OUTPUT_DIR
        /
        "correlation_per_seed.csv",

        index=False,

    )


    if (
        CORRELATION_METHOD.lower()
        ==
        "fisher"
    ):

        matrix = (
            fisher_matrix
        )

        method = (

            "Per-seed Pearson "
            "+ Fisher-z aggregation"

        )


    elif (
        CORRELATION_METHOD.lower()
        ==
        "pooled"
    ):

        matrix = (
            pooled_matrix
        )

        method = (
            "Pooled Pearson correlation"
        )


    else:

        raise ValueError(

            "CORRELATION_METHOD deve ser "
            "'fisher' ou 'pooled'."

        )


    print(
        "\n"
        +
        "=" * 92
    )


    print(
        "CORRELATION ANALYSIS"
    )


    print(
        "=" * 92
    )


    print(

        f"Correlation method      : "
        f"{method}"

    )


    print(
        "\nMatrix used in figure:"
    )


    print(

        matrix

        .round(
            4
        )

        .to_string()

    )


    fig = draw_correlation_matrix(
        matrix
    )


    save_figure(

        fig,

        "correlation_analysis",

    )


    plt.close(
        fig
    )


    print(

        "\n[DONE] "
        "Correlation Analysis generated."

    )


# =============================================================================
# ATIVIDADE OPERACIONAL
# =============================================================================

def load_request_activity():

    frames = []


    for seed in sorted(
        EXPECTED_SEEDS
    ):

        path = (
            request_path_for_seed(
                seed
            )
        )


        if not path.exists():

            raise FileNotFoundError(

                f"Arquivo de requests "
                f"não encontrado: "
                f"{path}"

            )


        df = pd.read_csv(
            path
        )


        required = {

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


    if not frames:

        raise RuntimeError(

            "Nenhum arquivo de requests "
            "válido encontrado."

        )


    return pd.concat(

        frames,

        ignore_index=True,

    )


# =============================================================================
# RESUMO DA ATIVIDADE
# =============================================================================

def build_activity_summary():

    requests = (
        load_request_activity()
    )


    requests[
        "window_start"
    ] = (

        np.floor(

            requests[
                "step"
            ]

            /
            SCORE_WINDOW_STEPS

        )

        *
        SCORE_WINDOW_STEPS

    )


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

            median_active_vehicles=(

                "active_vehicles",

                "median",

            ),

            request_count=(

                "active_vehicles",

                "size",

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
        SCORE_WINDOW_STEPS
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

            active_q25=(

                "mean_active_vehicles",

                lambda x:
                    x.quantile(
                        0.25
                    ),

            ),

            active_q75=(

                "mean_active_vehicles",

                lambda x:
                    x.quantile(
                        0.75
                    ),

            ),

            n_seeds=(

                "seed",

                "nunique",

            ),

            total_requests=(

                "request_count",

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


    return (

        per_seed,

        summary,

    )


# =============================================================================
# DETECÇÃO DO INTERVALO OPERACIONAL
# =============================================================================

def detect_operational_interval(
    activity_summary,
):
    """
    Detecta:

    - fim do warm-up / início operacional;
    - início do drain-out / fim operacional.

    O Score NÃO participa da decisão.
    """

    work = (

        activity_summary[

            activity_summary[
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

            "Nenhuma janela de atividade "
            "atingiu a cobertura mínima."

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
    # WARM-UP CUTOFF
    # =========================================================================

    operational_start = None


    for index in range(

        0,

        peak_position
        -
        needed
        +
        2,

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
    # DRAIN-OUT CUTOFF
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

            "Não foi possível detectar "
            "um início operacional sustentado."

        )


    if operational_end is None:

        print(

            "[WARNING] "
            "Drain-out sustentado não detectado. "
            "O final disponível será mantido."

        )


    if (

        operational_end
        is not None

        and

        operational_start
        >=
        operational_end

    ):

        raise RuntimeError(

            "Intervalo operacional inválido: "
            "início >= fim."

        )


    return {

        "reference_active":
            reference_active,

        "operational_threshold":
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

        "operational_start_step":
            operational_start,

        "operational_end_step":
            operational_end,

    }


# =============================================================================
# RESUMO DO SCORE
# =============================================================================

def build_score_summary(
    datasets,
):

    frames = []


    for seed in sorted(
        datasets
    ):

        work = (

            datasets[
                seed
            ][
                [
                    "step",
                    "score",
                ]
            ]

            .copy()

            .dropna()

        )


        if work.empty:

            continue


        work[
            "window_start"
        ] = (

            np.floor(

                work[
                    "step"
                ]

                /
                SCORE_WINDOW_STEPS

            )

            *
            SCORE_WINDOW_STEPS

        )


        per_seed = (

            work

            .groupby(

                "window_start",

                as_index=False,

            )

            .agg(

                score_mean=(

                    "score",

                    "mean",

                ),

                n_observations=(

                    "score",

                    "size",

                ),

            )

        )


        per_seed[
            "seed"
        ] = seed


        per_seed[
            "simulation_step"
        ] = (

            per_seed[
                "window_start"
            ]

            +
            SCORE_WINDOW_STEPS
            /
            2

        )


        frames.append(
            per_seed
        )


    if not frames:

        raise RuntimeError(

            "Nenhum dado de Score encontrado."

        )


    per_seed_df = pd.concat(

        frames,

        ignore_index=True,

    )


    summary = (

        per_seed_df

        .groupby(

            "simulation_step",

            as_index=False,

        )

        .agg(

            mean_score=(

                "score_mean",

                "mean",

            ),

            std_score=(

                "score_mean",

                "std",

            ),

            n_seeds=(

                "seed",

                "nunique",

            ),

            total_observations=(

                "n_observations",

                "sum",

            ),

            mean_observations_per_seed=(

                "n_observations",

                "mean",

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


    # =========================================================================
    # COBERTURA
    # =========================================================================

    minimum_seed_count = int(

        math.ceil(

            len(
                EXPECTED_SEEDS
            )

            *
            MIN_SCORE_SEED_COVERAGE

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
    # SUPORTE
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

            "Nenhuma janela de Score atingiu "
            "a cobertura mínima."

        )


    reference_total_observations = float(

        reference[
            "total_observations"
        ]

        .median()

    )


    reference_median_obs_per_seed = float(

        reference[
            "median_observations_per_seed"
        ]

        .median()

    )


    minimum_total_observations = (

        reference_total_observations

        *
        MIN_SCORE_OBS_SUPPORT_RATIO

    )


    minimum_median_obs_per_seed = (

        reference_median_obs_per_seed

        *
        MIN_SCORE_OBS_SUPPORT_RATIO

    )


    summary[
        "passes_total_observation_support"
    ] = (

        summary[
            "total_observations"
        ]

        >=

        minimum_total_observations

    )


    summary[
        "passes_per_seed_observation_support"
    ] = (

        summary[
            "median_observations_per_seed"
        ]

        >=

        minimum_median_obs_per_seed

    )


    # =========================================================================
    # IC95%
    # =========================================================================

    summary[
        "sem"
    ] = (

        summary[
            "std_score"
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
            "mean_score"
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
            "mean_score"
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

        "minimum_seed_count":
            minimum_seed_count,

        "reference_total_observations":
            reference_total_observations,

        "minimum_total_observations":
            minimum_total_observations,

        "reference_median_obs_per_seed":
            reference_median_obs_per_seed,

        "minimum_median_obs_per_seed":
            minimum_median_obs_per_seed,

    }


# =============================================================================
# BLOCO CONTÍNUO
# =============================================================================

def keep_initial_contiguous_block(
    summary,
):

    valid = (

        summary[

            summary[
                "passes_all_filters"
            ]

        ]

        .copy()

        .sort_values(
            "simulation_step"
        )

        .reset_index(
            drop=True
        )

    )


    if (

        valid.empty

        or

        len(
            valid
        )
        ==
        1

    ):

        return valid


    differences = np.diff(

        valid[
            "simulation_step"
        ]

        .to_numpy(
            dtype=float
        )

    )


    breaks = np.where(

        differences

        >

        SCORE_WINDOW_STEPS
        *
        1.5

    )[0]


    if len(
        breaks
    ) == 0:

        return valid


    return (

        valid

        .iloc[

            :
            int(
                breaks[
                    0
                ]
                +
                1
            )

        ]

        .copy()

    )


# =============================================================================
# FIGURA PRINCIPAL DO SCORE
# =============================================================================

def draw_score_main(
    plot_summary,
):

    fig, ax = plt.subplots(

        figsize=(
            10.2,
            5.6,
        )

    )


    if SHOW_SCORE_CI:

        ax.fill_between(

            plot_summary[
                "simulation_step"
            ],

            plot_summary[
                "ci95_low"
            ],

            plot_summary[
                "ci95_high"
            ],

            color="#BEBEBE",

            alpha=0.25,

            linewidth=0,

            label="95% CI",

            zorder=1,

        )


    ax.plot(

        plot_summary[
            "simulation_step"
        ],

        plot_summary[
            "mean_score"
        ],

        color="black",

        linewidth=2.0,

        label=(

            f"Mean Score "
            f"({SCORE_WINDOW_STEPS}-Step Windows)"

        ),

        zorder=3,

    )


    ax.set_xlabel(

        "Simulation Step",

        fontsize=15,

    )


    ax.set_ylabel(

        "Average Score",

        fontsize=15,

    )

    ax.set_ylim(
    0.82,
    0.92,
    )


    ax.grid(

        True,

        linestyle="--",

        linewidth=0.75,

        alpha=0.28,

    )


    ax.set_axisbelow(
        True
    )


    ax.tick_params(

        axis="both",

        labelsize=13,

    )


    for spine in ax.spines.values():

        spine.set_color(
            "#505050"
        )

        spine.set_linewidth(
            0.8
        )


    ax.legend(

        loc="upper right",

        fontsize=11.5,

        frameon=True,

        framealpha=0.95,

        edgecolor="#D0D0D0",

    )


    fig.tight_layout()


    return fig


# =============================================================================
# FIGURA DIAGNÓSTICA
# =============================================================================

def draw_score_diagnostic(
    summary,
    activity_summary,
    operational,
):

    fig, ax1 = plt.subplots(

        figsize=(
            10.2,
            5.6,
        )

    )


    # =========================================================================
    # SCORE
    # =========================================================================

    ax1.plot(

        summary[
            "simulation_step"
        ],

        summary[
            "mean_score"
        ],

        color="black",

        linewidth=1.8,

        label="Mean Score",

        zorder=3,

    )


    ax1.set_xlabel(
        "Simulation Step"
    )


    ax1.set_ylabel(
        "Average Score"
    )


    ax1.grid(

        True,

        linestyle="--",

        linewidth=0.75,

        alpha=0.25,

    )


    # =========================================================================
    # ATIVIDADE
    # =========================================================================

    ax2 = ax1.twinx()


    ax2.plot(

        activity_summary[
            "simulation_step"
        ],

        activity_summary[
            "median_active_vehicles"
        ],

        color="#4C78A8",

        linewidth=1.8,

        alpha=0.85,

        label=(
            "Median Active Vehicles"
        ),

    )


    ax2.axhline(

        operational[
            "operational_threshold"
        ],

        color="#4C78A8",

        linestyle=":",

        linewidth=1.4,

        label=(
            "Operational Threshold"
        ),

    )


    ax2.set_ylabel(
        "Active Vehicles"
    )


    # =========================================================================
    # INÍCIO OPERACIONAL
    # =========================================================================

    ax1.axvline(

        operational[
            "operational_start_step"
        ],

        color="#2CA02C",

        linestyle="--",

        linewidth=1.6,

        label=(
            "Operational Start"
        ),

    )


    # =========================================================================
    # FIM OPERACIONAL
    # =========================================================================

    if (

        operational[
            "operational_end_step"
        ]

        is not None

    ):

        ax1.axvline(

            operational[
                "operational_end_step"
            ],

            color="#D62728",

            linestyle="--",

            linewidth=1.6,

            label=(
                "Operational End"
            ),

        )


    handles1, labels1 = (
        ax1.get_legend_handles_labels()
    )


    handles2, labels2 = (
        ax2.get_legend_handles_labels()
    )


    ax1.legend(

        handles1
        +
        handles2,

        labels1
        +
        labels2,

        loc="best",

        fontsize=9.5,

        frameon=True,

        framealpha=0.95,

    )


    fig.tight_layout()


    return fig


# =============================================================================
# SCORE EVOLUTION
# =============================================================================

def plot_score_evolution(
    datasets,
):

    score_data = (

        build_score_summary(
            datasets
        )

    )


    per_seed_df = (

        score_data[
            "per_seed"
        ]

    )


    summary = (

        score_data[
            "summary"
        ]

        .copy()

    )


    # =========================================================================
    # ATIVIDADE
    # =========================================================================

    (
        activity_per_seed,
        activity_summary,

    ) = build_activity_summary()


    operational = (

        detect_operational_interval(
            activity_summary
        )

    )


    operational_start = (

        operational[
            "operational_start_step"
        ]

    )


    operational_end = (

        operational[
            "operational_end_step"
        ]

    )


    # =========================================================================
    # FILTRO DO WARM-UP
    # =========================================================================

    summary[
        "passes_operational_start"
    ] = (

        summary[
            "simulation_step"
        ]

        >=

        operational_start

    )


    # =========================================================================
    # FILTRO DO DRAIN-OUT
    # =========================================================================

    if operational_end is None:

        summary[
            "passes_operational_end"
        ] = True


    else:

        summary[
            "passes_operational_end"
        ] = (

            summary[
                "simulation_step"
            ]

            <

            operational_end

        )


    summary[
        "passes_operational_interval"
    ] = (

        summary[
            "passes_operational_start"
        ]

        &

        summary[
            "passes_operational_end"
        ]

    )


    # =========================================================================
    # FILTRO COMPLETO
    # =========================================================================

    summary[
        "passes_all_filters"
    ] = (

        summary[
            "passes_seed_coverage"
        ]

        &

        summary[
            "passes_total_observation_support"
        ]

        &

        summary[
            "passes_per_seed_observation_support"
        ]

        &

        summary[
            "passes_operational_interval"
        ]

    )


    plot_summary = (

        keep_initial_contiguous_block(
            summary
        )

    )


    if plot_summary.empty:

        raise RuntimeError(

            "Nenhum bloco temporal confiável "
            "restou após os filtros."

        )


    summary[
        "used_in_main_plot"
    ] = (

        summary[
            "simulation_step"
        ]

        .isin(

            plot_summary[
                "simulation_step"
            ]

        )

    )


    # =========================================================================
    # FIGURA PRINCIPAL
    # =========================================================================

    fig = draw_score_main(
        plot_summary
    )


    save_figure(

        fig,

        "score_evolution",

    )


    plt.close(
        fig
    )


    # =========================================================================
    # FIGURA DIAGNÓSTICA
    # =========================================================================

    if SAVE_FULL_SCORE_DIAGNOSTIC:

        diagnostic = (

            draw_score_diagnostic(

                summary,

                activity_summary,

                operational,

            )

        )


        save_figure(

            diagnostic,

            "score_evolution_full_diagnostic",

        )


        plt.close(
            diagnostic
        )


    # =========================================================================
    # CSVs
    # =========================================================================

    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True,

    )


    per_seed_df.to_csv(

        OUTPUT_DIR
        /
        "score_evolution_per_seed.csv",

        index=False,

    )


    summary.to_csv(

        OUTPUT_DIR
        /
        "score_evolution_all_windows.csv",

        index=False,

    )


    plot_summary.to_csv(

        OUTPUT_DIR
        /
        "score_evolution_plot_data.csv",

        index=False,

    )


    activity_per_seed.to_csv(

        OUTPUT_DIR
        /
        "operational_activity_per_seed.csv",

        index=False,

    )


    activity_summary.to_csv(

        OUTPUT_DIR
        /
        "operational_activity_summary.csv",

        index=False,

    )


    interval_df = pd.DataFrame(

        [

            {

                "reference_quantile":
                    ACTIVE_REFERENCE_QUANTILE,

                "reference_active_vehicles":
                    operational[
                        "reference_active"
                    ],

                "operational_fraction":
                    ACTIVE_OPERATIONAL_FRACTION,

                "operational_threshold":
                    operational[
                        "operational_threshold"
                    ],

                "consecutive_windows":
                    ACTIVE_CONSECUTIVE_WINDOWS,

                "peak_step":
                    operational[
                        "peak_step"
                    ],

                "peak_active_vehicles":
                    operational[
                        "peak_active"
                    ],

                "operational_start_step":
                    operational_start,

                "operational_end_step":
                    operational_end,

            }

        ]

    )


    interval_df.to_csv(

        OUTPUT_DIR
        /
        "operational_interval.csv",

        index=False,

    )


    # =========================================================================
    # TERMINAL
    # =========================================================================

    print(
        "\n"
        +
        "=" * 92
    )


    print(
        "SCORE EVOLUTION"
    )


    print(
        "=" * 92
    )


    print(

        f"Seeds available                   : "
        f"{per_seed_df['seed'].nunique()}"

    )


    print(

        f"Window size                       : "
        f"{SCORE_WINDOW_STEPS} "
        f"simulation steps"

    )


    print(

        f"Minimum Score seed coverage       : "
        f"{MIN_SCORE_SEED_COVERAGE:.0%}"

    )


    print(

        f"Minimum Score seeds/window        : "
        f"{score_data['minimum_seed_count']}/"
        f"{len(EXPECTED_SEEDS)}"

    )


    print(

        f"Reference Score observations      : "
        f"{score_data['reference_total_observations']:.2f}"

    )


    print(

        f"Minimum Score observations        : "
        f"{score_data['minimum_total_observations']:.2f}"

    )


    # =========================================================================
    # INTERVALO OPERACIONAL
    # =========================================================================

    print(
        "\nOperational interval criterion"
    )


    print(
        "-" * 92
    )


    print(

        f"Activity reference quantile       : "
        f"{ACTIVE_REFERENCE_QUANTILE:.0%}"

    )


    print(

        f"Reference active vehicles         : "
        f"{operational['reference_active']:.2f}"

    )


    print(

        f"Operational fraction              : "
        f"{ACTIVE_OPERATIONAL_FRACTION:.0%}"

    )


    print(

        f"Operational threshold             : "
        f"{operational['operational_threshold']:.2f}"

    )


    print(

        f"Consecutive windows               : "
        f"{ACTIVE_CONSECUTIVE_WINDOWS}"

    )


    print(

        f"Activity peak step                : "
        f"{operational['peak_step']:.0f}"

    )


    print(

        f"Activity peak                     : "
        f"{operational['peak_active']:.2f}"

    )


    print(

        f"Operational start                 : "
        f"{operational_start:.0f}"

    )


    if operational_end is None:

        print(

            "Operational end                   : "
            "NOT DETECTED"

        )


    else:

        print(

            f"Operational end                   : "
            f"{operational_end:.0f}"

        )


    # =========================================================================
    # FIGURA PRINCIPAL
    # =========================================================================

    print(
        "\nMain figure"
    )


    print(
        "-" * 92
    )


    print(

        f"Windows used                      : "
        f"{len(plot_summary)}"

    )


    print(

        f"First plotted step                : "
        f"{plot_summary['simulation_step'].min():.0f}"

    )


    print(

        f"Last plotted step                 : "
        f"{plot_summary['simulation_step'].max():.0f}"

    )


    print(

        f"Min seeds in plotted data         : "
        f"{plot_summary['n_seeds'].min()}"

    )


    print(

        f"Mean Score plotted                : "
        f"{plot_summary['mean_score'].mean():.4f}"

    )


    print(

        f"95% CI displayed                  : "
        f"{SHOW_SCORE_CI}"

    )


    print(
        "\nOutputs:"
    )


    print(

        f"  "
        f"{OUTPUT_DIR / 'score_evolution.png'}"

    )


    print(

        f"  "
        f"{OUTPUT_DIR / 'score_evolution.pdf'}"

    )


    print(

        f"  "
        f"{OUTPUT_DIR / 'score_evolution.eps'}"

    )


    print(

        f"  "
        f"{OUTPUT_DIR / 'score_evolution_full_diagnostic.png'}"

    )


    print(

        f"  "
        f"{OUTPUT_DIR / 'operational_interval.csv'}"

    )


    print(

        "\n[DONE] "
        "Score Evolution generated."

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
        "SHADAI - BASIC FIGURES"
    )


    print(
        "#" * 92
    )


    datasets = (

        load_and_validate_seed_files()

    )


    # =========================================================================
    # GRÁFICO 1
    # =========================================================================

    plot_correlation_analysis(
        datasets
    )


    # =========================================================================
    # GRÁFICO 2
    # =========================================================================

    plot_score_evolution(
        datasets
    )


    print(
        "\n"
        +
        "#" * 92
    )


    print(
        "ALL BASIC FIGURES GENERATED"
    )


    print(
        "#" * 92
    )


if __name__ == "__main__":

    main()