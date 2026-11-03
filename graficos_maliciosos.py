# -*- coding: utf-8 -*-

"""
graficos_maliciosos.py

Figuras dos cenários adversariais da campanha final SHADAI.

FIGURAS
===============================================================================

1. Average Normalized Reputation vs. Simulation Step

   - Baseline
   - Malicious Attack (40%)
   - Flood Attack (intensity 5)


2. Average Utility vs. Simulation Step

   - Baseline
   - Malicious Attack (40%)
   - Flood Attack (intensity 5)


3. Malicious Selection Rate vs. Simulation Step

   - Malicious Attack (20%)
   - Flood Attack (intensity 5)


OBSERVAÇÃO IMPORTANTE SOBRE O GRÁFICO 3
===============================================================================

A comparação da taxa de seleção utiliza 20% de agentes maliciosos
nos dois cenários:

C17:
    malicious_ratio = 0.20

C18:
    malicious_ratio = 0.20
    flood_intensity = 5

Assim, diferenças na taxa de seleção não são consequência de uma
diferença artificial na proporção física de veículos maliciosos.

A antiga linha horizontal "Expected Malicious Ratio (20%)" foi
deliberadamente removida da figura final.


METODOLOGIA
===============================================================================

- cenário de 400 Providers;
- 30 seeds;
- rng_scheme = split_v2;
- janelas temporais básicas de 50 simulation steps;
- centered moving average de 3 janelas = 150 steps;
- suavização realizada separadamente por seed;
- agregação entre seeds somente após a suavização;
- warm-up e drain-out determinados por active_vehicles;
- comparação limitada ao intervalo operacional comum;
- IC95% calculado, mas não mostrado por padrão.

Nenhum CSV, analyzer ou resultado de simulação é modificado.
"""

from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURAÇÃO GERAL
# =============================================================================

OUTPUT_DIR = Path(
    "figures_maliciosos"
)

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

SHOW_UTILITY_CI = False

SHOW_SELECTION_CI = False


# =============================================================================
# EIXOS Y FINAIS
# =============================================================================

# Reputation
REPUTATION_Y_MIN = 0.855

REPUTATION_Y_MAX = 0.920


# Utility
UTILITY_Y_MIN = 0.105

UTILITY_Y_MAX = 0.185


# Malicious Selection Rate
SELECTION_Y_MIN = 0.120

SELECTION_Y_MAX = 0.305


# =============================================================================
# CENÁRIOS
# =============================================================================

SCENARIOS = {

    # =========================================================================
    # BASELINE
    # =========================================================================

    "BASELINE": {

        "label":
            "Baseline",

        "directory":
            Path("scalability_final"),

        "color":
            "black",

        "linestyle":
            "-",

        "linewidth":
            2.4,

    },


    # =========================================================================
    # C17 — MALICIOUS 40%
    #
    # Stress máximo usado nos gráficos de Reputation e Utility.
    # =========================================================================

    "MALICIOUS40": {

        "label":
            "Malicious Attack (40%)",

        "directory":
            Path("malicious_ablation"),

        "color":
            "#B23A1B",

        "linestyle":
            "--",

        "linewidth":
            2.0,

    },


    # =========================================================================
    # C17 — MALICIOUS 20%
    #
    # Usado na comparação da taxa de seleção maliciosa.
    # =========================================================================

    "MALICIOUS20": {

        "label":
            "Malicious Attack (20%)",

        "directory":
            Path("malicious_ablation"),

        "color":
            "#B23A1B",

        "linestyle":
            "--",

        "linewidth":
            2.0,

    },


    # =========================================================================
    # C18 — FLOOD 5
    # =========================================================================

    "FLOOD5": {

        "label":
            "Flood Attack (intensity 5)",

        "directory":
            Path("flood_ablation"),

        "color":
            "#243B5A",

        "linestyle":
            ":",

        "linewidth":
            2.2,

    },

}


# =============================================================================
# CENÁRIOS UTILIZADOS EM CADA GRÁFICO
# =============================================================================

ATTACK_SCENARIOS = [

    "BASELINE",

    "MALICIOUS40",

    "FLOOD5",

]


SELECTION_SCENARIOS = [

    "MALICIOUS20",

    "FLOOD5",

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


UTILITY_ALIASES = [

    "utilidade",

    "utility",

    "utility_norm",

    "normalized_utility",

    "utilidade_norm",

]


ACTIVE_ALIASES = [

    "active_vehicles",

    "active_vehicle_count",

    "n_active_vehicles",

    "num_active_vehicles",

]


MALICIOUS_FLAG_ALIASES = [

    "malicioso",

    "malicious",

    "is_malicious",

    "selected_malicious",

    "selected_is_malicious",

    "provider_malicious",

    "provider_is_malicious",

    "vehicle_malicious",

]


MALICIOUS_RATE_ALIASES = [

    "malicious_selection_rate",

    "malicious_selected_rate",

    "malicious_selection_share",

    "selected_malicious_rate",

]


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
# CACHE DE ARQUIVOS
# =============================================================================

FILE_CACHE = {}


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
# T CRÍTICO 95%
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
# LOCALIZA COLUNA
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

            return lookup[
                key
            ]


    if required:

        raise RuntimeError(

            "Nenhuma coluna encontrada entre:\n"
            f"  {candidates}\n\n"

            "Colunas disponíveis:\n"
            f"  {list(df.columns)}"

        )


    return None


# =============================================================================
# SEED
# =============================================================================

def extract_seed_from_filename(
    path,
):

    match = re.search(

        r"seed[_-]?0*(\d+)",

        path.stem,

        flags=re.IGNORECASE,

    )


    if match is None:

        return None


    return int(
        match.group(1)
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

                float(
                    values.iloc[0]
                )

            )


    return (

        extract_seed_from_filename(
            path
        )

    )


# =============================================================================
# METADADOS
# =============================================================================

def first_numeric(
    df,
    column,
):

    if column not in df.columns:

        return None


    values = (

        pd.to_numeric(

            df[
                column
            ],

            errors="coerce",

        )

        .dropna()

    )


    if values.empty:

        return None


    return float(
        values.iloc[0]
    )


def first_text(
    df,
    column,
):

    if column not in df.columns:

        return None


    values = (

        df[
            column
        ]

        .dropna()

        .astype(str)

        .str.strip()

    )


    if values.empty:

        return None


    return str(
        values.iloc[0]
    )


# =============================================================================
# RATIO
# =============================================================================

def ratio_matches(
    observed,
    target,
):

    if observed is None:

        return False


    return (

        np.isclose(

            observed,

            target,

            atol=1e-9,

        )

        or

        np.isclose(

            observed,

            target * 100.0,

            atol=1e-9,

        )

    )


# =============================================================================
# RNG
# =============================================================================

def validate_rng(
    df,
    path,
):

    rng = (

        first_text(

            df,

            "rng_scheme",

        )

    )


    if rng is None:

        return


    if rng != EXPECTED_RNG:

        raise RuntimeError(

            f"{path.name}: "
            f"rng_scheme={rng}, "
            f"esperado={EXPECTED_RNG}"

        )


# =============================================================================
# MATCH DE CENÁRIO
# =============================================================================

def matches_scenario(
    df,
    path,
    scenario_key,
):

    name = (

        path.name
        .lower()

    )


    # =========================================================================
    # RNG
    # =========================================================================

    rng = (

        first_text(

            df,

            "rng_scheme",

        )

    )


    if (

        rng is not None

        and

        rng != EXPECTED_RNG

    ):

        return False


    # =========================================================================
    # PROVIDER CAP
    # =========================================================================

    max_vehicles = (

        first_numeric(

            df,

            "max_vehicles",

        )

    )


    if (

        max_vehicles is not None

        and

        int(
            max_vehicles
        )
        !=
        PROVIDER_SCENARIO

    ):

        return False


    # =========================================================================
    # METADADOS DOS ATAQUES
    # =========================================================================

    attack_type = (

        first_text(

            df,

            "attack_type",

        )

    )


    if attack_type is not None:

        attack_type = (

            attack_type
            .strip()
            .lower()

        )


    malicious_ratio = (

        first_numeric(

            df,

            "malicious_ratio",

        )

    )


    flood_intensity = (

        first_numeric(

            df,

            "flood_intensity",

        )

    )


    # =========================================================================
    # BASELINE
    # =========================================================================

    if scenario_key == "BASELINE":

        if (
            "scalability_v400"
            not in name
        ):

            return False


        if attack_type is not None:

            if attack_type not in {

                "none",

                "off",

                "disabled",

                "no_attack",

            }:

                return False


        if malicious_ratio is not None:

            if not np.isclose(

                malicious_ratio,

                0.0,

                atol=1e-12,

            ):

                return False


        return True


    # =========================================================================
    # MALICIOUS 40%
    # =========================================================================

    if scenario_key == "MALICIOUS40":

        if malicious_ratio is not None:

            if not ratio_matches(

                malicious_ratio,

                0.40,

            ):

                return False


        else:

            tokens = [

                "malicious_40",

                "malicious40",

                "ratio_40",

                "ratio40",

                "_40_",

            ]


            if not any(

                token in name

                for token
                in tokens

            ):

                return False


        if (

            attack_type is not None

            and

            attack_type
            in {

                "flood",

                "candidate_flood",

                "candidacy_flood",

            }

        ):

            return False


        return True


    # =========================================================================
    # MALICIOUS 20%
    # =========================================================================

    if scenario_key == "MALICIOUS20":

        if malicious_ratio is not None:

            if not ratio_matches(

                malicious_ratio,

                0.20,

            ):

                return False


        else:

            tokens = [

                "malicious_20",

                "malicious20",

                "ratio_20",

                "ratio20",

                "_20_",

            ]


            if not any(

                token in name

                for token
                in tokens

            ):

                return False


        if (

            attack_type is not None

            and

            attack_type
            in {

                "flood",

                "candidate_flood",

                "candidacy_flood",

            }

        ):

            return False


        return True


    # =========================================================================
    # FLOOD 5
    # =========================================================================

    if scenario_key == "FLOOD5":

        if flood_intensity is not None:

            if not np.isclose(

                flood_intensity,

                5.0,

                atol=1e-9,

            ):

                return False


        else:

            tokens = [

                "flood_05",

                "flood05",

                "flood_5",

                "flood5",

                "intensity_05",

                "intensity_5",

            ]


            if not any(

                token in name

                for token
                in tokens

            ):

                return False


        # C18 usa 20% de base maliciosa.
        if malicious_ratio is not None:

            if not ratio_matches(

                malicious_ratio,

                0.20,

            ):

                return False


        return True


    raise ValueError(

        f"Cenário inválido: "
        f"{scenario_key}"

    )


# =============================================================================
# LOCALIZA CSVs CANDIDATOS
# =============================================================================

def find_candidate_csvs(
    scenario_key,
    kind,
):

    directory = (

        SCENARIOS[
            scenario_key
        ][
            "directory"
        ]

    )


    if not directory.exists():

        raise FileNotFoundError(

            f"Diretório não encontrado:\n"
            f"  {directory}"

        )


    if kind == "metric":

        prefixes = (

            "metricas",

            "metrics",

        )


    elif kind == "request":

        prefixes = (

            "requests",

            "request",

        )


    else:

        raise ValueError(

            f"kind inválido: "
            f"{kind}"

        )


    candidates = []


    for path in directory.rglob(
        "*.csv"
    ):

        lower_parts = {

            part.lower()

            for part
            in path.parts

        }


        # Ignora outputs de analyzers.
        if "analysis" in lower_parts:

            continue


        name = (

            path.name
            .lower()

        )


        if not name.startswith(
            prefixes
        ):

            continue


        candidates.append(
            path
        )


    return sorted(
        candidates
    )


# =============================================================================
# DESCOBRE ARQUIVOS CORRETOS
# =============================================================================

def discover_scenario_files(
    scenario_key,
    kind,
):

    cache_key = (

        scenario_key,

        kind,

    )


    if cache_key in FILE_CACHE:

        return FILE_CACHE[
            cache_key
        ]


    candidates = (

        find_candidate_csvs(

            scenario_key,

            kind,

        )

    )


    selected = {}


    for path in candidates:

        try:

            preview = pd.read_csv(

                path,

                nrows=5,

            )

        except Exception:

            continue


        if preview.empty:

            continue


        if not matches_scenario(

            preview,

            path,

            scenario_key,

        ):

            continue


        seed = (

            extract_seed(

                preview,

                path,

            )

        )


        if seed is None:

            continue


        if seed not in EXPECTED_SEEDS:

            continue


        if seed in selected:

            raise RuntimeError(

                f"{SCENARIOS[scenario_key]['label']}: "
                f"mais de um arquivo válido "
                f"para seed {seed}.\n\n"

                f"Arquivo 1:\n"
                f"  {selected[seed]}\n\n"

                f"Arquivo 2:\n"
                f"  {path}"

            )


        selected[
            seed
        ] = path


    missing = sorted(

        EXPECTED_SEEDS
        -
        set(
            selected
        )

    )


    if missing:

        raise RuntimeError(

            f"{SCENARIOS[scenario_key]['label']}: "
            f"seeds ausentes para {kind}:\n"
            f"  {missing}\n\n"

            f"Diretório:\n"
            f"  {SCENARIOS[scenario_key]['directory']}\n\n"

            f"CSV candidatos encontrados: "
            f"{len(candidates)}"

        )


    ordered = [

        selected[
            seed
        ]

        for seed
        in sorted(
            selected
        )

    ]


    print(

        f"[FILES] "
        f"{SCENARIOS[scenario_key]['label']:<28} "
        f"{kind:<8}: "
        f"{len(ordered)} selected"

    )


    print(

        f"        first: "
        f"{ordered[0].name}"

    )


    print(

        f"        last : "
        f"{ordered[-1].name}"

    )


    FILE_CACHE[
        cache_key
    ] = ordered


    return ordered


# =============================================================================
# CONVERTE FLAG MALICIOSA
# =============================================================================

def coerce_binary_series(
    series,
):

    if pd.api.types.is_bool_dtype(
        series
    ):

        return (

            series
            .astype(float)

        )


    numeric = pd.to_numeric(

        series,

        errors="coerce",

    )


    text = (

        series

        .astype(str)

        .str.strip()

        .str.lower()

    )


    mapping = {

        "true":
            1.0,

        "false":
            0.0,

        "yes":
            1.0,

        "no":
            0.0,

        "malicious":
            1.0,

        "honest":
            0.0,

        "malicioso":
            1.0,

        "honesto":
            0.0,

    }


    mapped = (

        text
        .map(
            mapping
        )

    )


    result = (

        numeric
        .where(

            numeric.notna(),

            mapped,

        )

    )


    return result


# =============================================================================
# REPUTATION
# =============================================================================

def extract_reputation(
    df,
    path,
):

    normalized_col = (

        find_column(

            df,

            REPUTATION_NORMALIZED_ALIASES,

            required=False,

        )

    )


    if normalized_col is not None:

        values = pd.to_numeric(

            df[
                normalized_col
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


        if not finite.empty:

            if (

                finite.min()
                <
                -0.05

                or

                finite.max()
                >
                1.05

            ):

                raise RuntimeError(

                    f"{path.name}: "
                    f"{normalized_col} não parece "
                    "normalizada em [0,1]."

                )


        return (

            values,

            normalized_col,

            "already_normalized",

            1.0,

        )


    raw_col = (

        find_column(

            df,

            REPUTATION_RAW_ALIASES,

            required=True,

        )

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
            "nenhuma Reputation válida."

        )


    q99 = float(

        finite.quantile(
            0.99
        )

    )


    if q99 <= 1.05:

        factor = 1.0

        mode = (
            "raw_already_0_1"
        )


    elif q99 <= 10.5:

        factor = 10.0

        mode = (
            "raw_0_10_div10"
        )


    else:

        raise RuntimeError(

            f"{path.name}: "
            "escala de Reputation não reconhecida."

        )


    return (

        values
        /
        factor,

        raw_col,

        mode,

        factor,

    )


# =============================================================================
# UTILITY
# =============================================================================

def extract_utility(
    df,
    path,
):

    utility_col = (

        find_column(

            df,

            UTILITY_ALIASES,

            required=True,

        )

    )


    values = pd.to_numeric(

        df[
            utility_col
        ],

        errors="coerce",

    )


    return (

        values,

        utility_col,

        "direct",

        1.0,

    )


# =============================================================================
# MALICIOUS SELECTION RATE
# =============================================================================

def extract_malicious_selection(
    df,
    path,
):

    # =========================================================================
    # FLAG BINÁRIA DO PROVEDOR SELECIONADO
    # =========================================================================

    flag_col = (

        find_column(

            df,

            MALICIOUS_FLAG_ALIASES,

            required=False,

        )

    )


    if flag_col is not None:

        values = (

            coerce_binary_series(

                df[
                    flag_col
                ]

            )

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
                f"{flag_col} sem valores válidos."

            )


        if (

            finite.min()
            <
            -1e-9

            or

            finite.max()
            >
            1.0
            +
            1e-9

        ):

            raise RuntimeError(

                f"{path.name}: "
                f"{flag_col} deveria estar "
                "em escala binária [0,1]."

            )


        return (

            values,

            flag_col,

            "binary_selected_provider_flag",

            1.0,

        )


    # =========================================================================
    # FALLBACK: TAXA JÁ REGISTRADA
    # =========================================================================

    rate_col = (

        find_column(

            df,

            MALICIOUS_RATE_ALIASES,

            required=False,

        )

    )


    if rate_col is not None:

        values = pd.to_numeric(

            df[
                rate_col
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


        if not finite.empty:

            if finite.max() > 1.05:

                if finite.max() <= 100.5:

                    values = (

                        values
                        /
                        100.0

                    )


                    return (

                        values,

                        rate_col,

                        "direct_rate_percent_div100",

                        100.0,

                    )


                raise RuntimeError(

                    f"{path.name}: "
                    f"escala de {rate_col} "
                    "não reconhecida."

                )


        return (

            values,

            rate_col,

            "direct_rate_0_1",

            1.0,

        )


    raise RuntimeError(

        f"{path.name}: "
        "não foi encontrada coluna para "
        "identificar seleção maliciosa.\n\n"

        f"Flags procuradas:\n"
        f"  {MALICIOUS_FLAG_ALIASES}\n\n"

        f"Rates procuradas:\n"
        f"  {MALICIOUS_RATE_ALIASES}\n\n"

        f"Colunas disponíveis:\n"
        f"  {list(df.columns)}"

    )


# =============================================================================
# EXTRAÇÃO GENÉRICA
# =============================================================================

def extract_metric(
    df,
    path,
    metric_name,
):

    if metric_name == "reputation":

        return (

            extract_reputation(

                df,

                path,

            )

        )


    if metric_name == "utility":

        return (

            extract_utility(

                df,

                path,

            )

        )


    if metric_name == "malicious_selection":

        return (

            extract_malicious_selection(

                df,

                path,

            )

        )


    raise ValueError(

        f"Métrica inválida: "
        f"{metric_name}"

    )


# =============================================================================
# CARREGA MÉTRICA
# =============================================================================

def load_metric(
    scenario_key,
    metric_name,
):

    files = (

        discover_scenario_files(

            scenario_key,

            "metric",

        )

    )


    datasets = {}

    audit_records = []


    for path in files:

        df = pd.read_csv(
            path
        )


        validate_rng(
            df,
            path,
        )


        if "step" not in df.columns:

            raise RuntimeError(

                f"{path.name}: "
                "coluna 'step' ausente."

            )


        seed = (

            extract_seed(

                df,

                path,

            )

        )


        if seed in datasets:

            raise RuntimeError(

                f"{scenario_key}: "
                f"seed duplicada "
                f"{seed}"

            )


        (
            metric_values,

            source_column,

            extraction_mode,

            scale_factor,

        ) = extract_metric(

            df,

            path,

            metric_name,

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

                    metric_values,

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


        datasets[
            seed
        ] = work


        audit_records.append(

            {

                "scenario":
                    scenario_key,

                "label":
                    SCENARIOS[
                        scenario_key
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

                "scale_factor":
                    scale_factor,

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


    missing = sorted(

        EXPECTED_SEEDS
        -
        set(
            datasets
        )

    )


    if missing:

        raise RuntimeError(

            f"{SCENARIOS[scenario_key]['label']}: "
            f"seeds ausentes para "
            f"{metric_name}: "
            f"{missing}"

        )


    print(

        f"[OK] "
        f"{SCENARIOS[scenario_key]['label']}: "
        f"{len(datasets)} seeds "
        f"for {metric_name}"

    )


    return (

        datasets,

        pd.DataFrame(
            audit_records
        ),

    )


# =============================================================================
# CARREGA ATIVIDADE
# =============================================================================

def load_activity(
    scenario_key,
):

    files = (

        discover_scenario_files(

            scenario_key,

            "request",

        )

    )


    frames = []


    for path in files:

        df = pd.read_csv(
            path
        )


        validate_rng(
            df,
            path,
        )


        if "step" not in df.columns:

            raise RuntimeError(

                f"{path.name}: "
                "coluna 'step' ausente."

            )


        active_col = (

            find_column(

                df,

                ACTIVE_ALIASES,

                required=True,

            )

        )


        seed = (

            extract_seed(

                df,

                path,

            )

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


                "active_vehicles":

                    pd.to_numeric(

                        df[
                            active_col
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


    return pd.concat(

        frames,

        ignore_index=True,

    )


# =============================================================================
# RESUMO DA ATIVIDADE
# =============================================================================

def build_activity_summary(
    scenario_key,
):

    activity = (

        load_activity(
            scenario_key
        )

    )


    activity[
        "window_start"
    ] = (

        np.floor(

            activity[
                "step"
            ]

            /
            WINDOW_STEPS

        )

        *
        WINDOW_STEPS

    )


    per_seed = (

        activity

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
# INTERVALO OPERACIONAL
# =============================================================================

def detect_operational_interval(
    scenario_key,
):

    summary = (

        build_activity_summary(
            scenario_key
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

            f"{scenario_key}: "
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


    needed = (

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
    # DRAIN-OUT
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

            f"{scenario_key}: "
            "início operacional não detectado."

        )


    if operational_end is None:

        raise RuntimeError(

            f"{scenario_key}: "
            "fim operacional não detectado."

        )


    return {

        "scenario":
            scenario_key,

        "label":
            SCENARIOS[
                scenario_key
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
# CONTEXTO OPERACIONAL
# =============================================================================

def prepare_operational_context(
    scenario_order,
    context_name,
):

    records = []


    for scenario_key in scenario_order:

        records.append(

            detect_operational_interval(
                scenario_key
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


    if common_start >= common_end:

        raise RuntimeError(

            f"{context_name}: "
            "não existe intervalo operacional comum."

        )


    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True,

    )


    table.to_csv(

        OUTPUT_DIR
        /
        f"operational_intervals_{context_name}.csv",

        index=False,

    )


    print(
        "\n"
        +
        "=" * 100
    )


    print(

        f"COMMON OPERATIONAL INTERVAL — "
        f"{context_name.upper()}"

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
# SUAVIZAÇÃO POR SEED
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
    scenario_key,
    metric_name,
):

    (
        datasets,

        audit,

    ) = load_metric(

        scenario_key,

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
            "scenario"
        ] = scenario_key


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
        "scenario"
    ] = scenario_key


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

            f"{scenario_key}: "
            f"sem cobertura suficiente "
            f"para {metric_name}."

        )


    # =========================================================================
    # SUPORTE DE OBSERVAÇÕES
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
    scenario_order,
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


    for scenario_key in scenario_order:

        result = (

            build_metric_summary(

                scenario_key,

                metric_name,

            )

        )


        summary = (

            result[
                "summary"
            ]

            .copy()

        )


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

                f"{scenario_key}: "
                f"nenhum ponto válido "
                f"para {metric_name}."

            )


        summaries[
            scenario_key
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

                "scenario":
                    scenario_key,

                "label":
                    SCENARIOS[
                        scenario_key
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


    return {

        "summaries":
            summaries,

        "all_summary":

            pd.concat(

                [

                    summaries[
                        key
                    ]

                    for key
                    in scenario_order

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
# DESENHO GENÉRICO
# =============================================================================

def draw_metric_comparison(
    data,
    scenario_order,
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


    for scenario_key in scenario_order:

        summary = (

            data[
                "summaries"
            ][
                scenario_key
            ]

        )


        config = (

            SCENARIOS[
                scenario_key
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
# SALVA TABELAS
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
        f"{prefix}_adversarial_per_seed.csv",

        index=False,

    )


    data[
        "all_summary"
    ].to_csv(

        OUTPUT_DIR
        /
        f"{prefix}_adversarial_all_windows.csv",

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
        f"{prefix}_adversarial_plot_data.csv",

        index=False,

    )


    data[
        "audit"
    ].to_csv(

        OUTPUT_DIR
        /
        f"{prefix}_adversarial_audit.csv",

        index=False,

    )


    data[
        "diagnostics"
    ].to_csv(

        OUTPUT_DIR
        /
        f"{prefix}_adversarial_diagnostics.csv",

        index=False,

    )


# =============================================================================
# DIAGNÓSTICOS
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


    print(

        data[
            "audit"
        ][

            [
                "scenario",

                "label",

                "metric",

                "source_column",

                "extraction_mode",

                "scale_factor",

            ]

        ]

        .drop_duplicates()

        .to_string(
            index=False
        )

    )


# =============================================================================
# GRÁFICO 1 — REPUTATION
# =============================================================================

def plot_reputation_under_attacks(
    operational_context,
):

    data = (

        prepare_metric_comparison(

            metric_name=
                "reputation",

            scenario_order=
                ATTACK_SCENARIOS,

            operational_context=
                operational_context,

        )

    )


    fig = (

        draw_metric_comparison(

            data=
                data,

            scenario_order=
                ATTACK_SCENARIOS,

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

        "reputation_under_adversarial_scenarios",

    )


    plt.close(
        fig
    )


    save_metric_tables(

        data,

        "reputation",

    )


    print_diagnostics(

        "REPUTATION UNDER ADVERSARIAL SCENARIOS",

        data,

        REPUTATION_Y_MIN,

        REPUTATION_Y_MAX,

    )


# =============================================================================
# GRÁFICO 2 — UTILITY
# =============================================================================

def plot_utility_under_attacks(
    operational_context,
):

    data = (

        prepare_metric_comparison(

            metric_name=
                "utility",

            scenario_order=
                ATTACK_SCENARIOS,

            operational_context=
                operational_context,

        )

    )


    fig = (

        draw_metric_comparison(

            data=
                data,

            scenario_order=
                ATTACK_SCENARIOS,

            ylabel=
                "Average Utility",

            y_min=
                UTILITY_Y_MIN,

            y_max=
                UTILITY_Y_MAX,

            show_ci=
                SHOW_UTILITY_CI,

            legend_location=
                "upper left",

        )

    )


    save_figure(

        fig,

        "utility_under_adversarial_scenarios",

    )


    plt.close(
        fig
    )


    save_metric_tables(

        data,

        "utility",

    )


    print_diagnostics(

        "UTILITY UNDER ADVERSARIAL SCENARIOS",

        data,

        UTILITY_Y_MIN,

        UTILITY_Y_MAX,

    )


# =============================================================================
# GRÁFICO 3 — MALICIOUS SELECTION RATE
# =============================================================================

def plot_malicious_selection_rate():

    # =========================================================================
    # C17 20% vs. C18 flood-5
    # Ambos possuem 20% de base maliciosa.
    # =========================================================================

    operational_context = (

        prepare_operational_context(

            SELECTION_SCENARIOS,

            "malicious_selection_rate",

        )

    )


    data = (

        prepare_metric_comparison(

            metric_name=
                "malicious_selection",

            scenario_order=
                SELECTION_SCENARIOS,

            operational_context=
                operational_context,

        )

    )


    fig = (

        draw_metric_comparison(

            data=
                data,

            scenario_order=
                SELECTION_SCENARIOS,

            ylabel=
                "Malicious Selection Rate",

            y_min=
                SELECTION_Y_MIN,

            y_max=
                SELECTION_Y_MAX,

            show_ci=
                SHOW_SELECTION_CI,

            legend_location=
                "upper right",

        )

    )


    save_figure(

        fig,

        "malicious_selection_rate",

    )


    plt.close(
        fig
    )


    save_metric_tables(

        data,

        "malicious_selection_rate",

    )


    print_diagnostics(

        "MALICIOUS SELECTION RATE",

        data,

        SELECTION_Y_MIN,

        SELECTION_Y_MAX,

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
        "SHADAI - ADVERSARIAL SCENARIOS"
    )


    print(
        "#" * 100
    )


    # =========================================================================
    # CONTEXTO COMUM PARA REPUTATION + UTILITY
    # =========================================================================

    attack_operational_context = (

        prepare_operational_context(

            ATTACK_SCENARIOS,

            "adversarial_stress",

        )

    )


    # =========================================================================
    # 1 — REPUTATION
    # =========================================================================

    plot_reputation_under_attacks(

        attack_operational_context

    )


    # =========================================================================
    # 2 — UTILITY
    # =========================================================================

    plot_utility_under_attacks(

        attack_operational_context

    )


    # =========================================================================
    # 3 — MALICIOUS SELECTION RATE
    # =========================================================================

    plot_malicious_selection_rate()


    # =========================================================================
    # FINAL
    # =========================================================================

    print(
        "\n"
        +
        "#" * 100
    )


    print(
        "ADVERSARIAL FIGURES GENERATED"
    )


    print(
        "#" * 100
    )


    print(
        "\nFigures:"
    )


    print(

        f"  "
        f"{OUTPUT_DIR / 'reputation_under_adversarial_scenarios.png'}"

    )


    print(

        f"  "
        f"{OUTPUT_DIR / 'utility_under_adversarial_scenarios.png'}"

    )


    print(

        f"  "
        f"{OUTPUT_DIR / 'malicious_selection_rate.png'}"

    )


if __name__ == "__main__":

    main()