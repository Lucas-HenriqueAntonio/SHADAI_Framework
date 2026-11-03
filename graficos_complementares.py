# -*- coding: utf-8 -*-

"""
graficos_complementares.py

Figuras complementares da campanha final SHADAI.

FIGURAS
===============================================================================

1. Static vs. Dynamic Pricing
   C25

   - Average Offered Price
   - Eligibility Rate
   - Conditional Allocation Rate
   - Average Utility


2. Pricing Sensitivity
   C25b

   Painel esquerdo:
   - beta sensitivity
   - Request Interval = 20

   Painel direito:
   - alpha sensitivity
   - Request Interval = 10

   Curvas:
   - Eligibility Rate
   - Conditional Allocation Rate


3. Runtime Scalability
   C15

   Painel esquerdo:
   - Average Decision Runtime
   - Selection Runtime P95

   Painel direito:
   - Total Algorithm Runtime


4. Dynamic-Pricing Stability
   C26

   Painel esquerdo:
   - Average Price Multiplier

   Painel direito:
   - Coefficient of Variation
   - Mean |Delta|
   - P95 |Delta|


PALETA OFICIAL DO ARTIGO
===============================================================================

BLACK       #000000
TERRACOTTA  #B23A1B
DARK_BLUE   #243B5A
DARK_GREEN  #2F5D50
GRAY        #6B6B6B


METODOLOGIA
===============================================================================

- Resultados da campanha final congelada;
- 30 seeds por configuração;
- IC95% utilizado quando disponível;
- C25/C25b/C26 usam outputs finais dos analyzers;
- C15 é reconstruído diretamente da telemetria runtime multi-seed;
- nenhum CSV ou resultado de simulação é modificado.
"""

from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# OUTPUT
# =============================================================================

OUTPUT_DIR = Path(
    "figures_complementares"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# INPUTS
# =============================================================================

C25_CANDIDATES = [

    Path(
        "pricing_ablation/analysis/"
        "c25_pricing_multiseed.csv"
    ),

    Path(
        "c25_pricing_multiseed.csv"
    ),

]


C25B_CANDIDATES = [

    Path(
        "pricing_sensitivity/analysis/"
        "c25b_multiseed.csv"
    ),

    Path(
        "c25b_sensitivity/analysis/"
        "c25b_multiseed.csv"
    ),

    Path(
        "c25b_multiseed.csv"
    ),

]


C26_CANDIDATES = [

    Path(
        "pricing_stability/analysis/"
        "c26_stability_multiseed.csv"
    ),

    Path(
        "c26_stability/analysis/"
        "c26_stability_multiseed.csv"
    ),

    Path(
        "c26_stability_multiseed.csv"
    ),

]


# =============================================================================
# C15 — RUNTIME
# =============================================================================

RUNTIME_DIRECTORIES = [

    Path(
        "runtime_ablation"
    ),

    Path(
        "runtime_scalability"
    ),

    Path(
        "runtime_final"
    ),

]


RUNTIME_INTERVALS = [

    60,
    40,
    20,
    10,

]


RUNTIME_LOAD_LABELS = {

    60:
        "Low",

    40:
        "Moderate",

    20:
        "High",

    10:
        "Very High",

}


EXPECTED_SEEDS = set(
    range(1, 31)
)


EXPECTED_RNG = (
    "split_v2"
)


# =============================================================================
# PALETA OFICIAL
# =============================================================================

BLACK = (
    "#000000"
)

TERRACOTTA = (
    "#B23A1B"
)

DARK_BLUE = (
    "#243B5A"
)

DARK_GREEN = (
    "#2F5D50"
)

GRAY = (
    "#6B6B6B"
)

LIGHT_GRID = (
    "#DEDEDE"
)


# =============================================================================
# ESPESSURAS
# =============================================================================

LW_MAIN = 2.4

LW_SECONDARY = 2.0

LW_DOTTED = 2.2


# =============================================================================
# ESTILO GLOBAL
# =============================================================================

plt.rcParams.update({

    "font.family":
        "DejaVu Sans",

    "font.size":
        11.5,

    "axes.labelsize":
        14,

    "xtick.labelsize":
        11.5,

    "ytick.labelsize":
        11.5,

    "legend.fontsize":
        10.5,

    "pdf.fonttype":
        42,

    "ps.fonttype":
        42,

})


# =============================================================================
# HELPERS
# =============================================================================

def find_first_existing(
    candidates,
):

    for path in candidates:

        if path.exists():

            return path


    return None


def load_required(
    candidates,
    label,
):

    path = (

        find_first_existing(
            candidates
        )

    )


    if path is None:

        raise FileNotFoundError(

            f"{label}: arquivo não encontrado.\n\n"

            "Candidatos pesquisados:\n"

            +

            "\n".join(

                f"  {item}"

                for item
                in candidates

            )

        )


    print(

        f"[OK] "
        f"{label:<30}: "
        f"{path}"

    )


    return (

        pd.read_csv(
            path
        ),

        path,

    )


# =============================================================================
# SAVE
# =============================================================================

def save_figure(
    fig,
    filename,
):

    paths = {

        "png":
            OUTPUT_DIR
            /
            f"{filename}.png",

        "pdf":
            OUTPUT_DIR
            /
            f"{filename}.pdf",

        "eps":
            OUTPUT_DIR
            /
            f"{filename}.eps",

    }


    fig.savefig(

        paths[
            "png"
        ],

        dpi=400,

        bbox_inches="tight",

    )


    fig.savefig(

        paths[
            "pdf"
        ],

        bbox_inches="tight",

    )


    fig.savefig(

        paths[
            "eps"
        ],

        format="eps",

        bbox_inches="tight",

    )


    print(

        f"[SAVE] "
        f"{paths['png']}"

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

        linewidth=0.7,

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
# ERROR BAR
# =============================================================================

def asymmetric_yerr(
    mean,
    low,
    high,
    scale=1.0,
):

    mean = np.asarray(
        mean,
        dtype=float,
    ) * scale


    low = np.asarray(
        low,
        dtype=float,
    ) * scale


    high = np.asarray(
        high,
        dtype=float,
    ) * scale


    lower = np.maximum(

        mean
        -
        low,

        0.0,

    )


    upper = np.maximum(

        high
        -
        mean,

        0.0,

    )


    return np.vstack(

        [
            lower,
            upper,
        ]

    )


# =============================================================================
# BAR LABELS
# =============================================================================

def label_bars(
    ax,
    bars,
    fmt,
    offset_fraction=0.015,
):

    ymin, ymax = (

        ax.get_ylim()

    )


    span = (

        ymax
        -
        ymin

    )


    for bar in bars:

        value = (

            bar.get_height()

        )


        if value >= 0:

            y = (

                value
                +
                span
                *
                offset_fraction

            )

            va = (
                "bottom"
            )


        else:

            y = (

                value
                -
                span
                *
                offset_fraction

            )

            va = (
                "top"
            )


        ax.text(

            bar.get_x()
            +
            bar.get_width()
            /
            2,

            y,

            fmt.format(
                value
            ),

            ha="center",

            va=va,

            fontsize=9.5,

            color=BLACK,

        )


# =============================================================================
# =============================================================================
# FIGURE 1 — STATIC VS DYNAMIC PRICING
# =============================================================================
# =============================================================================

def plot_pricing_static_vs_dynamic():

    df, path = (

        load_required(

            C25_CANDIDATES,

            "C25 pricing",

        )

    )


    required = [

        "pricing_mode",

        "avg_offered_price_mean",
        "avg_offered_price_ci95_low",
        "avg_offered_price_ci95_high",

        "eligibility_rate_mean",
        "eligibility_rate_ci95_low",
        "eligibility_rate_ci95_high",

        "conditional_allocation_rate_mean",
        "conditional_allocation_rate_ci95_low",
        "conditional_allocation_rate_ci95_high",

        "avg_utility_mean",
        "avg_utility_ci95_low",
        "avg_utility_ci95_high",

    ]


    missing = [

        column

        for column
        in required

        if column
        not in df.columns

    ]


    if missing:

        raise RuntimeError(

            "C25: colunas ausentes:\n"
            f"{missing}"

        )


    modes = [

        "static",
        "dynamic",

    ]


    labels = [

        "Static",
        "Dynamic",

    ]


    work = (

        df

        .set_index(
            "pricing_mode"
        )

        .loc[
            modes
        ]

        .reset_index()

    )


    # =========================================================================
    # PALETA
    #
    # Static  -> Gray
    # Dynamic -> Dark Blue
    # =========================================================================

    colors = [

        GRAY,
        DARK_BLUE,

    ]


    fig, axes = plt.subplots(

        2,
        2,

        figsize=(
            10.4,
            7.0,
        )

    )


    configurations = [

        (
            axes[0, 0],

            "avg_offered_price",

            "Average Offered Price",

            1.0,

            "{:.2f}",

        ),

        (
            axes[0, 1],

            "eligibility_rate",

            "Eligibility Rate (%)",

            100.0,

            "{:.1f}",

        ),

        (
            axes[1, 0],

            "conditional_allocation_rate",

            "Conditional Allocation Rate (%)",

            100.0,

            "{:.1f}",

        ),

        (
            axes[1, 1],

            "avg_utility",

            "Average Utility",

            1.0,

            "{:.3f}",

        ),

    ]


    for (

        ax,
        metric,
        ylabel,
        scale,
        fmt,

    ) in configurations:

        mean = (

            work[
                f"{metric}_mean"
            ]

            .to_numpy(
                dtype=float
            )

            *
            scale

        )


        yerr = (

            asymmetric_yerr(

                work[
                    f"{metric}_mean"
                ],

                work[
                    f"{metric}_ci95_low"
                ],

                work[
                    f"{metric}_ci95_high"
                ],

                scale=scale,

            )

        )


        x = np.arange(
            len(
                modes
            )
        )


        bars = ax.bar(

            x,

            mean,

            width=0.58,

            color=colors,

            edgecolor=BLACK,

            linewidth=0.8,

            yerr=yerr,

            capsize=4,

            error_kw={

                "ecolor":
                    BLACK,

                "elinewidth":
                    1.2,

                "capthick":
                    1.2,

            },

        )


        ax.set_xticks(
            x
        )


        ax.set_xticklabels(
            labels
        )


        ax.set_ylabel(
            ylabel
        )


        # =====================================================================
        # UTILITY ZERO REFERENCE
        # =====================================================================

        if metric == "avg_utility":

            ax.axhline(

                0.0,

                color=GRAY,

                linewidth=1.1,

                linestyle="--",

                zorder=1,

            )


        style_axis(
            ax
        )


        ax.relim()

        ax.autoscale_view()


        label_bars(

            ax,

            bars,

            fmt,

        )


    fig.tight_layout()


    save_figure(

        fig,

        "pricing_static_vs_dynamic",

    )


    plt.close(
        fig
    )


    print(
        "\n"
        +
        "=" * 92
    )


    print(
        "C25 - STATIC VS DYNAMIC PRICING"
    )


    print(
        "=" * 92
    )


    print(

        work[

            [
                "pricing_mode",

                "avg_offered_price_mean",

                "eligibility_rate_mean",

                "conditional_allocation_rate_mean",

                "avg_utility_mean",

            ]

        ]

        .round(
            5
        )

        .to_string(
            index=False
        )

    )


# =============================================================================
# =============================================================================
# FIGURE 2 — PRICING SENSITIVITY
# =============================================================================
# =============================================================================

def plot_pricing_sensitivity():

    df, path = (

        load_required(

            C25B_CANDIDATES,

            "C25b sensitivity",

        )

    )


    required = [

        "block",

        "alpha",

        "beta",

        "request_interval",

        "eligibility_rate_mean",

        "eligibility_rate_ci95_low",

        "eligibility_rate_ci95_high",

        "conditional_allocation_rate_mean",

        "conditional_allocation_rate_ci95_low",

        "conditional_allocation_rate_ci95_high",

    ]


    missing = [

        column

        for column
        in required

        if column
        not in df.columns

    ]


    if missing:

        raise RuntimeError(

            "C25b: colunas ausentes:\n"
            f"{missing}"

        )


    fig, axes = plt.subplots(

        1,
        2,

        figsize=(
            10.6,
            4.9,
        ),

        sharey=True,

    )


    # =========================================================================
    # BETA
    # =========================================================================

    beta = (

        df[

            df[
                "block"
            ]

            .astype(str)

            .str.lower()

            ==
            "beta"

        ]

        .copy()

        .sort_values(
            "beta"
        )

    )


    plot_sensitivity_panel(

        ax=
            axes[0],

        df=
            beta,

        parameter=
            "beta",

        xlabel=(

            r"$\beta$"

            "\n"

            "(Request Interval = 20)"

        ),

    )


    # =========================================================================
    # ALPHA
    # =========================================================================

    alpha = (

        df[

            df[
                "block"
            ]

            .astype(str)

            .str.lower()

            ==
            "alpha"

        ]

        .copy()

        .sort_values(
            "alpha"
        )

    )


    plot_sensitivity_panel(

        ax=
            axes[1],

        df=
            alpha,

        parameter=
            "alpha",

        xlabel=(

            r"$\alpha$"

            "\n"

            "(Request Interval = 10)"

        ),

    )


    axes[0].set_ylabel(

        "Rate (%)"

    )


    axes[1].legend(

        loc="lower right",

        frameon=True,

        framealpha=0.95,

        edgecolor="#D0D0D0",

    )


    fig.tight_layout()


    save_figure(

        fig,

        "pricing_sensitivity",

    )


    plt.close(
        fig
    )


    print(
        "\n"
        +
        "=" * 92
    )


    print(
        "C25b - PRICING SENSITIVITY"
    )


    print(
        "=" * 92
    )


    print(

        df[

            [
                "block",

                "alpha",

                "beta",

                "request_interval",

                "eligibility_rate_mean",

                "conditional_allocation_rate_mean",

            ]

        ]

        .round(
            5
        )

        .to_string(
            index=False
        )

    )


def plot_sensitivity_panel(
    ax,
    df,
    parameter,
    xlabel,
):

    x = (

        pd.to_numeric(

            df[
                parameter
            ],

            errors="coerce",

        )

        .to_numpy(
            dtype=float
        )

    )


    # =========================================================================
    # ELIGIBILITY
    #
    # Official dark green
    # =========================================================================

    eligibility = (

        df[
            "eligibility_rate_mean"
        ]

        .to_numpy(
            dtype=float
        )

        *
        100.0

    )


    eligibility_err = (

        asymmetric_yerr(

            df[
                "eligibility_rate_mean"
            ],

            df[
                "eligibility_rate_ci95_low"
            ],

            df[
                "eligibility_rate_ci95_high"
            ],

            scale=100.0,

        )

    )


    ax.errorbar(

        x,

        eligibility,

        yerr=
            eligibility_err,

        color=
            DARK_GREEN,

        linestyle=
            "-.",

        linewidth=
            LW_SECONDARY,

        marker=
            "s",

        markersize=
            6,

        capsize=
            4,

        label=
            "Eligibility Rate",

    )


    # =========================================================================
    # CONDITIONAL ALLOCATION
    #
    # Official dark blue
    # =========================================================================

    allocation = (

        df[
            "conditional_allocation_rate_mean"
        ]

        .to_numpy(
            dtype=float
        )

        *
        100.0

    )


    allocation_err = (

        asymmetric_yerr(

            df[
                "conditional_allocation_rate_mean"
            ],

            df[
                "conditional_allocation_rate_ci95_low"
            ],

            df[
                "conditional_allocation_rate_ci95_high"
            ],

            scale=100.0,

        )

    )


    ax.errorbar(

        x,

        allocation,

        yerr=
            allocation_err,

        color=
            DARK_BLUE,

        linestyle=
            ":",

        linewidth=
            LW_DOTTED,

        marker=
            "o",

        markersize=
            6,

        capsize=
            4,

        label=
            "Conditional Allocation",

    )


    ax.set_xlabel(
        xlabel
    )


    ax.set_xticks(
        x
    )


    ax.set_xticklabels(

        [

            f"{value:.1f}"

            for value
            in x

        ]

    )


    ax.set_ylim(
        40,
        102,
    )


    style_axis(
        ax
    )


# =============================================================================
# =============================================================================
# C15 — RUNTIME
# =============================================================================
# =============================================================================

def to_bool(
    series,
):

    return (

        series

        .astype(str)

        .str.strip()

        .str.lower()

        .isin(

            [
                "true",
                "1",
                "yes",
            ]

        )

    )


def extract_seed(
    df,
    path,
):

    if "seed" in df.columns:

        values = (

            pd.to_numeric(

                df[
                    "seed"
                ],

                errors="coerce",

            )

            .dropna()

        )


        if not values.empty:

            return int(

                values.iloc[0]

            )


    match = re.search(

        r"seed[_-]?0*(\d+)",

        path.stem,

        flags=re.IGNORECASE,

    )


    if match:

        return int(

            match.group(1)

        )


    return None


def find_runtime_files():

    candidates = []


    # =========================================================================
    # PRIMARY SEARCH
    # =========================================================================

    for directory in RUNTIME_DIRECTORIES:

        if not directory.exists():

            continue


        for path in directory.rglob(
            "*.csv"
        ):

            name = (

                path.name
                .lower()

            )


            if not name.startswith(
                "runtime"
            ):

                continue


            candidates.append(
                path
            )


    # =========================================================================
    # FALLBACK SEARCH
    # =========================================================================

    if not candidates:

        for path in Path(".").rglob(
            "runtime*.csv"
        ):

            lower_path = (

                str(path)
                .lower()

            )


            if (

                "runtime_ablation"
                in lower_path

                or

                "c15"
                in lower_path

            ):

                candidates.append(
                    path
                )


    return sorted(

        set(
            candidates
        )

    )


def build_runtime_per_seed():

    files = (

        find_runtime_files()

    )


    if not files:

        raise FileNotFoundError(

            "Nenhum CSV runtime C15 foi encontrado."

        )


    required = {

        "request_interval",

        "selection_runtime_ms",

        "decision_runtime_ms",

    }


    rows = []


    for path in files:

        try:

            df = pd.read_csv(
                path
            )

        except Exception:

            continue


        if df.empty:

            continue


        if not required.issubset(
            df.columns
        ):

            continue


        # =========================================================================
        # RNG
        # =========================================================================

        if "rng_scheme" in df.columns:

            rng = (

                df[
                    "rng_scheme"
                ]

                .dropna()

                .astype(str)

                .str.strip()

            )


            if (

                not rng.empty

                and

                rng.iloc[0]
                !=
                EXPECTED_RNG

            ):

                continue


        # =========================================================================
        # ATTACK
        # =========================================================================

        if "attack_type" in df.columns:

            attack = (

                df[
                    "attack_type"
                ]

                .dropna()

                .astype(str)

                .str.strip()

                .str.lower()

            )


            if (

                not attack.empty

                and

                attack.iloc[0]
                not in {

                    "none",

                    "off",

                    "disabled",

                    "no_attack",

                }

            ):

                continue


        # =========================================================================
        # INTERVAL
        # =========================================================================

        interval_values = (

            pd.to_numeric(

                df[
                    "request_interval"
                ],

                errors="coerce",

            )

            .dropna()

        )


        if interval_values.empty:

            continue


        interval = int(

            interval_values.iloc[0]

        )


        if interval not in RUNTIME_INTERVALS:

            continue


        seed = (

            extract_seed(
                df,
                path,
            )

        )


        if (

            seed is None

            or

            seed not in EXPECTED_SEEDS

        ):

            continue


        # =========================================================================
        # SELECTION MASK
        # =========================================================================

        if "selection_executed" in df.columns:

            selection_mask = (

                to_bool(

                    df[
                        "selection_executed"
                    ]

                )

            )


        else:

            selection_mask = (

                pd.to_numeric(

                    df[
                        "selection_runtime_ms"
                    ],

                    errors="coerce",

                )

                .notna()

            )


        selection_runtime = (

            pd.to_numeric(

                df.loc[

                    selection_mask,

                    "selection_runtime_ms",

                ],

                errors="coerce",

            )

            .dropna()

        )


        decision_runtime = (

            pd.to_numeric(

                df[
                    "decision_runtime_ms"
                ],

                errors="coerce",

            )

            .dropna()

        )


        if selection_runtime.empty:

            continue


        if decision_runtime.empty:

            continue


        # =========================================================================
        # OFFER
        # =========================================================================

        if "offer_runtime_ms" in df.columns:

            offer_runtime = (

                pd.to_numeric(

                    df[
                        "offer_runtime_ms"
                    ],

                    errors="coerce",

                )

                .dropna()

            )


            avg_offer = (

                float(
                    offer_runtime.mean()
                )

                if not offer_runtime.empty

                else np.nan

            )


        else:

            avg_offer = np.nan


        # =========================================================================
        # CANDIDATES
        # =========================================================================

        if "candidate_count" in df.columns:

            candidate_series = (

                pd.to_numeric(

                    df[
                        "candidate_count"
                    ],

                    errors="coerce",

                )

                .dropna()

            )


            avg_candidates = (

                float(
                    candidate_series.mean()
                )

                if not candidate_series.empty

                else np.nan

            )


        else:

            avg_candidates = np.nan


        rows.append(

            {

                "seed":
                    seed,

                "request_interval":
                    interval,

                "selections":
                    int(
                        len(
                            selection_runtime
                        )
                    ),

                "avg_candidates":
                    avg_candidates,

                "avg_offer_runtime_ms":
                    avg_offer,

                "avg_selection_runtime_ms":
                    float(

                        selection_runtime.mean()

                    ),

                "selection_p95_ms":
                    float(

                        selection_runtime
                        .quantile(
                            0.95
                        )

                    ),

                "selection_p99_ms":
                    float(

                        selection_runtime
                        .quantile(
                            0.99
                        )

                    ),

                "avg_decision_runtime_ms":
                    float(

                        decision_runtime.mean()

                    ),

                "total_algorithm_runtime_ms":
                    float(

                        decision_runtime.sum()

                    ),

                "source_file":
                    str(
                        path
                    ),

            }

        )


    result = pd.DataFrame(
        rows
    )


    if result.empty:

        raise RuntimeError(

            "Nenhum runtime C15 final "
            "foi identificado."

        )


    # =========================================================================
    # DUPLICATES
    # =========================================================================

    duplicate = (

        result

        .duplicated(

            subset=[

                "request_interval",

                "seed",

            ],

            keep=False,

        )

    )


    if duplicate.any():

        problem = (

            result.loc[

                duplicate,

                [

                    "request_interval",

                    "seed",

                    "source_file",

                ],

            ]

        )


        raise RuntimeError(

            "C15: arquivos duplicados:\n\n"

            +

            problem.to_string(
                index=False
            )

        )


    # =========================================================================
    # COMPLETENESS
    # =========================================================================

    for interval in RUNTIME_INTERVALS:

        subset = (

            result[

                result[
                    "request_interval"
                ]

                ==
                interval

            ]

        )


        observed = set(

            subset[
                "seed"
            ]

            .astype(int)

        )


        missing = sorted(

            EXPECTED_SEEDS
            -
            observed

        )


        print(

            f"[C15] "
            f"interval={interval:<2}: "
            f"{len(observed)}/30 seeds"

        )


        if missing:

            raise RuntimeError(

                f"C15 interval={interval}: "
                f"seeds ausentes: "
                f"{missing}"

            )


    result.to_csv(

        OUTPUT_DIR
        /
        "runtime_c15_per_seed.csv",

        index=False,

    )


    return result


# =============================================================================
# CI95 RUNTIME
# =============================================================================

def ci95_from_series(
    values,
):

    values = (

        pd.Series(
            values,
            dtype=float,
        )

        .dropna()

    )


    n = len(
        values
    )


    if n == 0:

        return (

            np.nan,
            np.nan,
            np.nan,

        )


    mean = float(

        values.mean()

    )


    if n == 1:

        return (

            mean,
            np.nan,
            np.nan,

        )


    sd = float(

        values.std(
            ddof=1
        )

    )


    critical = (

        2.04523

        if n <= 30

        else 1.96

    )


    margin = (

        critical

        *
        sd

        /
        math.sqrt(
            n
        )

    )


    return (

        mean,

        mean
        -
        margin,

        mean
        +
        margin,

    )


def aggregate_runtime(
    per_seed,
):

    metrics = [

        "selections",

        "avg_candidates",

        "avg_offer_runtime_ms",

        "avg_selection_runtime_ms",

        "selection_p95_ms",

        "selection_p99_ms",

        "avg_decision_runtime_ms",

        "total_algorithm_runtime_ms",

    ]


    rows = []


    for interval in RUNTIME_INTERVALS:

        subset = (

            per_seed[

                per_seed[
                    "request_interval"
                ]

                ==
                interval

            ]

        )


        row = {

            "request_interval":
                interval,

        }


        for metric in metrics:

            mean, low, high = (

                ci95_from_series(

                    subset[
                        metric
                    ]

                )

            )


            row[
                f"{metric}_mean"
            ] = mean


            row[
                f"{metric}_ci95_low"
            ] = low


            row[
                f"{metric}_ci95_high"
            ] = high


        rows.append(
            row
        )


    result = pd.DataFrame(
        rows
    )


    result.to_csv(

        OUTPUT_DIR
        /
        "runtime_c15_multiseed.csv",

        index=False,

    )


    return result


# =============================================================================
# RUNTIME LINE
# =============================================================================

def plot_runtime_line(
    ax,
    x,
    df,
    metric,
    label,
    color,
    linestyle,
    marker,
    linewidth,
):

    mean = (

        df[
            f"{metric}_mean"
        ]

        .to_numpy(
            dtype=float
        )

    )


    yerr = (

        asymmetric_yerr(

            df[
                f"{metric}_mean"
            ],

            df[
                f"{metric}_ci95_low"
            ],

            df[
                f"{metric}_ci95_high"
            ],

        )

    )


    ax.errorbar(

        x,

        mean,

        yerr=
            yerr,

        color=
            color,

        linestyle=
            linestyle,

        linewidth=
            linewidth,

        marker=
            marker,

        markersize=
            6,

        capsize=
            4,

        label=
            label,

    )


# =============================================================================
# FIGURE 3 — RUNTIME SCALABILITY
# =============================================================================

def plot_runtime_scalability():

    per_seed = (

        build_runtime_per_seed()

    )


    df = (

        aggregate_runtime(
            per_seed
        )

    )


    order = (

        pd.DataFrame(

            {

                "request_interval":
                    RUNTIME_INTERVALS

            }

        )

        .merge(

            df,

            on=
                "request_interval",

            how=
                "left",

        )

    )


    x = np.arange(

        len(
            RUNTIME_INTERVALS
        )

    )


    labels = [

        (
            f"{interval}\n"
            f"{RUNTIME_LOAD_LABELS[interval]}"
        )

        for interval
        in RUNTIME_INTERVALS

    ]


    fig, axes = plt.subplots(

        1,
        2,

        figsize=(
            11.0,
            4.9,
        )

    )


    # =========================================================================
    # LEFT
    # =========================================================================

    ax = axes[0]


    # Average Decision = black main line

    plot_runtime_line(

        ax=
            ax,

        x=
            x,

        df=
            order,

        metric=
            "avg_decision_runtime_ms",

        label=
            "Average Decision Runtime",

        color=
            BLACK,

        linestyle=
            "-",

        marker=
            "o",

        linewidth=
            LW_MAIN,

    )


    # Selection P95 = dark blue dotted

    plot_runtime_line(

        ax=
            ax,

        x=
            x,

        df=
            order,

        metric=
            "selection_p95_ms",

        label=
            "Selection Runtime P95",

        color=
            DARK_BLUE,

        linestyle=
            ":",

        marker=
            "s",

        linewidth=
            LW_DOTTED,

    )


    ax.set_xticks(
        x
    )


    ax.set_xticklabels(
        labels
    )


    ax.set_xlabel(

        "Request Interval / Offered Load"

    )


    ax.set_ylabel(

        "Runtime per Decision (ms)"

    )


    ax.legend(

        loc="upper right",

        frameon=True,

        framealpha=0.95,

        edgecolor="#D0D0D0",

    )


    style_axis(
        ax
    )


    # =========================================================================
    # RIGHT — TOTAL RUNTIME
    #
    # Terracotta = accumulated workload effect
    # =========================================================================

    ax = axes[1]


    mean = (

        order[
            "total_algorithm_runtime_ms_mean"
        ]

        .to_numpy(
            dtype=float
        )

    )


    yerr = (

        asymmetric_yerr(

            order[
                "total_algorithm_runtime_ms_mean"
            ],

            order[
                "total_algorithm_runtime_ms_ci95_low"
            ],

            order[
                "total_algorithm_runtime_ms_ci95_high"
            ],

        )

    )


    ax.errorbar(

        x,

        mean,

        yerr=
            yerr,

        color=
            TERRACOTTA,

        linestyle=
            "--",

        linewidth=
            LW_SECONDARY,

        marker=
            "o",

        markersize=
            6,

        capsize=
            4,

    )


    ax.set_xticks(
        x
    )


    ax.set_xticklabels(
        labels
    )


    ax.set_xlabel(

        "Request Interval / Offered Load"

    )


    ax.set_ylabel(

        "Total Algorithm Runtime (ms)"

    )


    style_axis(
        ax
    )


    fig.tight_layout()


    save_figure(

        fig,

        "runtime_scalability",

    )


    plt.close(
        fig
    )


    print(
        "\n"
        +
        "=" * 92
    )


    print(
        "C15 - RUNTIME SCALABILITY"
    )


    print(
        "=" * 92
    )


    print(

        order[

            [
                "request_interval",

                "avg_decision_runtime_ms_mean",

                "selection_p95_ms_mean",

                "selection_p99_ms_mean",

                "total_algorithm_runtime_ms_mean",

            ]

        ]

        .round(
            4
        )

        .to_string(
            index=False
        )

    )


# =============================================================================
# =============================================================================
# FIGURE 4 — PRICING STABILITY
# =============================================================================
# =============================================================================

def plot_pricing_stability():

    df, path = (

        load_required(

            C26_CANDIDATES,

            "C26 stability",

        )

    )


    required = [

        "request_interval",

        "avg_multiplier_mean",

        "avg_multiplier_ci95_low",

        "avg_multiplier_ci95_high",

        "multiplier_cv_mean",

        "multiplier_cv_ci95_low",

        "multiplier_cv_ci95_high",

        "mean_abs_multiplier_delta_mean",

        "mean_abs_multiplier_delta_ci95_low",

        "mean_abs_multiplier_delta_ci95_high",

        "p95_abs_multiplier_delta_mean",

        "p95_abs_multiplier_delta_ci95_low",

        "p95_abs_multiplier_delta_ci95_high",

    ]


    missing = [

        column

        for column
        in required

        if column
        not in df.columns

    ]


    if missing:

        raise RuntimeError(

            "C26: colunas ausentes:\n"
            f"{missing}"

        )


    order = (

        pd.DataFrame(

            {

                "request_interval":
                    RUNTIME_INTERVALS

            }

        )

        .merge(

            df,

            on=
                "request_interval",

            how=
                "left",

        )

    )


    x = np.arange(

        len(
            RUNTIME_INTERVALS
        )

    )


    labels = [

        (
            f"{interval}\n"
            f"{RUNTIME_LOAD_LABELS[interval]}"
        )

        for interval
        in RUNTIME_INTERVALS

    ]


    fig, axes = plt.subplots(

        1,
        2,

        figsize=(
            11.0,
            4.9,
        )

    )


    # =========================================================================
    # LEFT — MULTIPLIER
    #
    # Main system response = black
    # =========================================================================

    ax = axes[0]


    multiplier_mean = (

        order[
            "avg_multiplier_mean"
        ]

        .to_numpy(
            dtype=float
        )

    )


    multiplier_err = (

        asymmetric_yerr(

            order[
                "avg_multiplier_mean"
            ],

            order[
                "avg_multiplier_ci95_low"
            ],

            order[
                "avg_multiplier_ci95_high"
            ],

        )

    )


    ax.errorbar(

        x,

        multiplier_mean,

        yerr=
            multiplier_err,

        color=
            BLACK,

        linestyle=
            "-",

        linewidth=
            LW_MAIN,

        marker=
            "o",

        markersize=
            6,

        capsize=
            4,

    )


    ax.set_xticks(
        x
    )


    ax.set_xticklabels(
        labels
    )


    ax.set_xlabel(

        "Request Interval / Offered Load"

    )


    ax.set_ylabel(

        "Average Price Multiplier"

    )


    ax.set_ylim(
        1.10,
        1.33,
    )


    style_axis(
        ax
    )


    # =========================================================================
    # RIGHT — TEMPORAL VARIATION
    #
    # CV         = dark green
    # Mean Delta = dark blue
    # P95 Delta  = terracotta
    # =========================================================================

    ax = axes[1]


    stability_series = [

        (
            "multiplier_cv",

            "Coefficient of Variation",

            DARK_GREEN,

            "-.",

            "s",

            LW_SECONDARY,

        ),

        (
            "mean_abs_multiplier_delta",

            r"Mean $|\Delta|$",

            DARK_BLUE,

            ":",

            "o",

            LW_DOTTED,

        ),

        (
            "p95_abs_multiplier_delta",

            r"P95 $|\Delta|$",

            TERRACOTTA,

            "--",

            "^",

            LW_SECONDARY,

        ),

    ]


    for (

        metric,
        label,
        color,
        linestyle,
        marker,
        linewidth,

    ) in stability_series:

        mean = (

            order[
                f"{metric}_mean"
            ]

        )


        err = (

            asymmetric_yerr(

                order[
                    f"{metric}_mean"
                ],

                order[
                    f"{metric}_ci95_low"
                ],

                order[
                    f"{metric}_ci95_high"
                ],

            )

        )


        ax.errorbar(

            x,

            mean,

            yerr=
                err,

            color=
                color,

            linestyle=
                linestyle,

            linewidth=
                linewidth,

            marker=
                marker,

            markersize=
                6,

            capsize=
                4,

            label=
                label,

        )


    ax.set_xticks(
        x
    )


    ax.set_xticklabels(
        labels
    )


    ax.set_xlabel(

        "Request Interval / Offered Load"

    )


    ax.set_ylabel(

        "Temporal Variation"

    )


    ax.set_ylim(
        0.0,
        0.45,
    )


    ax.legend(

        loc="upper left",

        frameon=True,

        framealpha=0.95,

        edgecolor="#D0D0D0",

    )


    style_axis(
        ax
    )


    fig.tight_layout()


    save_figure(

        fig,

        "pricing_stability",

    )


    plt.close(
        fig
    )


    print(
        "\n"
        +
        "=" * 92
    )


    print(
        "C26 - PRICING STABILITY"
    )


    print(
        "=" * 92
    )


    print(

        order[

            [
                "request_interval",

                "avg_multiplier_mean",

                "multiplier_cv_mean",

                "mean_abs_multiplier_delta_mean",

                "p95_abs_multiplier_delta_mean",

            ]

        ]

        .round(
            4
        )

        .to_string(
            index=False
        )

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
        "SHADAI - COMPLEMENTARY FIGURES"
    )


    print(
        "#" * 100
    )


    # =========================================================================
    # 1 — C25
    # =========================================================================

    plot_pricing_static_vs_dynamic()


    # =========================================================================
    # 2 — C25b
    # =========================================================================

    plot_pricing_sensitivity()


    # =========================================================================
    # 3 — C15
    # =========================================================================

    plot_runtime_scalability()


    # =========================================================================
    # 4 — C26
    # =========================================================================

    plot_pricing_stability()


    # =========================================================================
    # FINAL
    # =========================================================================

    print(
        "\n"
        +
        "#" * 100
    )


    print(
        "ALL COMPLEMENTARY FIGURES GENERATED"
    )


    print(
        "#" * 100
    )


    print(
        "\nFigures:"
    )


    filenames = [

        "pricing_static_vs_dynamic.png",

        "pricing_sensitivity.png",

        "runtime_scalability.png",

        "pricing_stability.png",

    ]


    for filename in filenames:

        print(

            f"  "
            f"{OUTPUT_DIR / filename}"

        )


if __name__ == "__main__":

    main()