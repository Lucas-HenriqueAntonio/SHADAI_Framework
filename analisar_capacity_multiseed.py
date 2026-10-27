# analisar_capacity_multiseed.py

import os
import math
import xml.etree.ElementTree as ET

import pandas as pd


try:
    from scipy.stats import t
except ImportError:
    t = None


# ============================================================
# CONFIGURAÇÃO
# ============================================================

CAPACITY_DIR = "capacity_ablation"

SUMOCFG = "cidade.sumocfg"


CONFIGURACOES = {

    "capacity_low": {
        "label": "Low",
        "interval": 60
    },

    "capacity_moderate": {
        "label": "Moderate",
        "interval": 40
    },

    "capacity_high": {
        "label": "High",
        "interval": 20
    },

    "capacity_very_high": {
        "label": "Very High",
        "interval": 10
    }
}


ORDEM = [

    "capacity_low",

    "capacity_moderate",

    "capacity_high",

    "capacity_very_high"
]


BASELINE = "capacity_high"


SEEDS = list(
    range(
        1,
        31
    )
)


RNG_SCHEME_REQUIRED = "split_v1"

TELEMETRY_SCHEME_REQUIRED = "c14_capacity_v1"


# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_PER_SEED = os.path.join(

    CAPACITY_DIR,

    "capacity_por_seed.csv"
)


OUTPUT_AGG = os.path.join(

    CAPACITY_DIR,

    "capacity_multiseed.csv"
)


OUTPUT_PAIRED = os.path.join(

    CAPACITY_DIR,

    "capacity_pareado_vs_baseline.csv"
)


OUTPUT_LATEX = os.path.join(

    CAPACITY_DIR,

    "tabela_capacity_tnsm.tex"
)


# ============================================================
# SIMULATION TIME
# ============================================================

def ler_tempo_final(
    caminho
):

    tree = ET.parse(
        caminho
    )

    root = tree.getroot()

    end = root.find(
        "./time/end"
    )

    if end is None:
        return 3600

    return int(
        float(
            end.attrib[
                "value"
            ]
        )
    )


SIMULATION_STEPS = ler_tempo_final(
    SUMOCFG
)


# ============================================================
# IC95
# ============================================================

def calcular_ic95(
    valores
):

    serie = pd.Series(
        valores
    ).dropna()

    n = len(
        serie
    )

    if n == 0:

        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan")
        )

    media = serie.mean()

    if n == 1:

        return (
            media,
            float("nan"),
            float("nan"),
            float("nan")
        )

    sd = serie.std(
        ddof=1
    )

    se = (
        sd
        /
        math.sqrt(
            n
        )
    )

    if t is not None:

        crit = t.ppf(
            0.975,
            n - 1
        )

    else:

        crit = 1.96

    margem = (
        crit
        *
        se
    )

    return (
        media,
        sd,
        media - margem,
        media + margem
    )


# ============================================================
# BOOL
# ============================================================

def converter_bool(
    serie
):

    return (

        serie
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1"
            ]
        )
    )


# ============================================================
# VALIDA REQUEST LOG
# ============================================================

def validar_requests(
    df,
    intervalo
):

    obrigatorias = [

        "rng_scheme",

        "telemetry_scheme",

        "request_interval",

        "step",

        "active_vehicles",

        "eligible_candidates",

        "allocated",

        "failure_reason",

        "completion_step"
    ]

    for coluna in obrigatorias:

        if coluna not in df.columns:
            return False

    if not (
        df[
            "rng_scheme"
        ]
        .astype(str)
        .eq(
            RNG_SCHEME_REQUIRED
        )
        .all()
    ):
        return False

    if not (
        df[
            "telemetry_scheme"
        ]
        .astype(str)
        .eq(
            TELEMETRY_SCHEME_REQUIRED
        )
        .all()
    ):
        return False

    intervalos = (

        pd.to_numeric(
            df[
                "request_interval"
            ],
            errors="coerce"
        )

        .dropna()

        .unique()
    )

    if len(
        intervalos
    ) != 1:
        return False

    return (
        int(
            intervalos[0]
        )
        ==
        intervalo
    )


# ============================================================
# CABEÇALHO
# ============================================================

print("\n")

print("=" * 175)

print(
    "SHADAI - MULTI-SEED CAPACITY / THROUGHPUT ANALYSIS"
)

print("=" * 175)

print(
    f"Directory        : {CAPACITY_DIR}"
)

print(
    f"Simulation steps : {SIMULATION_STEPS}"
)

print(
    f"Seeds            : {len(SEEDS)}"
)

print(
    f"Baseline         : High (interval=20)"
)

print("=" * 175)


# ============================================================
# LEITURA
# ============================================================

resultados = []

faltantes = []

invalidos = []


for configuracao in ORDEM:

    label = CONFIGURACOES[
        configuracao
    ][
        "label"
    ]

    interval = CONFIGURACOES[
        configuracao
    ][
        "interval"
    ]

    load_factor = (
        CONFIGURACOES[
            BASELINE
        ][
            "interval"
        ]
        /
        interval
    )

    for seed in SEEDS:

        experiment_id = (

            f"{configuracao}_"
            f"interval_"
            f"{interval:03d}_"
            f"seed_"
            f"{seed:02d}"
        )

        requests_file = os.path.join(

            CAPACITY_DIR,

            (
                f"requests_shadai_400_"
                f"{experiment_id}.csv"
            )
        )

        if not os.path.exists(
            requests_file
        ):

            faltantes.append(
                (
                    configuracao,
                    seed
                )
            )

            continue

        df = pd.read_csv(
            requests_file
        )

        if (
            df.empty
            or
            not validar_requests(
                df,
                interval
            )
        ):

            invalidos.append(
                (
                    configuracao,
                    seed
                )
            )

            continue

        allocated = converter_bool(
            df[
                "allocated"
            ]
        )

        active = (

            pd.to_numeric(
                df[
                    "active_vehicles"
                ],
                errors="coerce"
            )
            >
            0
        )

        total_requests = len(
            df
        )

        allocated_count = int(
            allocated.sum()
        )

        unserved_count = (
            total_requests
            -
            allocated_count
        )

        no_active_count = (

            df[
                "failure_reason"
            ]
            .fillna("")
            .eq(
                "no_active_vehicles"
            )
            .sum()
        )

        no_eligible_count = (

            df[
                "failure_reason"
            ]
            .fillna("")
            .eq(
                "no_eligible_candidates"
            )
            .sum()
        )

        selection_failed_count = (

            df[
                "failure_reason"
            ]
            .fillna("")
            .eq(
                "selection_failed"
            )
            .sum()
        )

        active_request_count = int(
            active.sum()
        )

        # ====================================================
        # END-TO-END SERVICE RATIO
        # ====================================================

        service_ratio = (

            allocated_count
            /
            total_requests

            if total_requests > 0

            else float("nan")
        )

        unserved_ratio = (

            unserved_count
            /
            total_requests

            if total_requests > 0

            else float("nan")
        )

        # ====================================================
        # CONDITIONAL SERVICE RATIO
        # ====================================================

        conditional_service_ratio = (

            allocated_count
            /
            active_request_count

            if active_request_count > 0

            else float("nan")
        )

        conditional_unserved_ratio = (

            (
                active_request_count
                -
                allocated_count
            )
            /
            active_request_count

            if active_request_count > 0

            else float("nan")
        )

        # ====================================================
        # AVAILABILITY
        # ====================================================

        vehicle_availability_ratio = (

            active_request_count
            /
            total_requests

            if total_requests > 0

            else float("nan")
        )

        no_active_ratio = (

            no_active_count
            /
            total_requests

            if total_requests > 0

            else float("nan")
        )

        no_eligible_ratio = (

            no_eligible_count
            /
            total_requests

            if total_requests > 0

            else float("nan")
        )

        no_eligible_given_active = (

            no_eligible_count
            /
            active_request_count

            if active_request_count > 0

            else float("nan")
        )

        # ====================================================
        # COMPLETIONS
        # ====================================================

        completion_numeric = pd.to_numeric(

            df[
                "completion_step"
            ],

            errors="coerce"
        )

        completed = (

            allocated

            &

            completion_numeric.notna()

            &

            (
                completion_numeric
                <=
                SIMULATION_STEPS
            )
        )

        completed_count = int(
            completed.sum()
        )

        pending_at_end = (

            allocated_count
            -
            completed_count
        )

        completion_ratio_of_allocated = (

            completed_count
            /
            allocated_count

            if allocated_count > 0

            else float("nan")
        )

        # ====================================================
        # THROUGHPUT
        # ====================================================

        offered_throughput = (

            total_requests
            /
            SIMULATION_STEPS
            *
            1000.0
        )

        allocation_throughput = (

            allocated_count
            /
            SIMULATION_STEPS
            *
            1000.0
        )

        completion_throughput = (

            completed_count
            /
            SIMULATION_STEPS
            *
            1000.0
        )

        # ====================================================
        # ACTIVE / ELIGIBLE COUNTS
        # ====================================================

        avg_active_vehicles = (

            pd.to_numeric(
                df[
                    "active_vehicles"
                ],
                errors="coerce"
            )
            .mean()
        )

        avg_eligible_candidates = (

            pd.to_numeric(
                df[
                    "eligible_candidates"
                ],
                errors="coerce"
            )
            .mean()
        )

        avg_eligible_when_active = (

            pd.to_numeric(
                df.loc[
                    active,
                    "eligible_candidates"
                ],
                errors="coerce"
            )
            .mean()
        )

        # ====================================================
        # ACCOUNTING SANITY
        # ====================================================

        accounting_ok = (

            allocated_count
            +
            no_active_count
            +
            no_eligible_count
            +
            selection_failed_count
            ==
            total_requests
        )

        resultados.append({

            "configuration":
                configuracao,

            "workload":
                label,

            "request_interval":
                interval,

            "load_factor_vs_baseline":
                load_factor,

            "seed":
                seed,

            "total_requests":
                total_requests,

            "active_request_count":
                active_request_count,

            "allocated_count":
                allocated_count,

            "completed_count":
                completed_count,

            "unserved_count":
                unserved_count,

            "no_active_count":
                no_active_count,

            "no_eligible_count":
                no_eligible_count,

            "selection_failed_count":
                selection_failed_count,

            "service_ratio":
                service_ratio,

            "conditional_service_ratio":
                conditional_service_ratio,

            "unserved_ratio":
                unserved_ratio,

            "conditional_unserved_ratio":
                conditional_unserved_ratio,

            "vehicle_availability_ratio":
                vehicle_availability_ratio,

            "no_active_ratio":
                no_active_ratio,

            "no_eligible_ratio":
                no_eligible_ratio,

            "no_eligible_given_active":
                no_eligible_given_active,

            "completion_ratio_of_allocated":
                completion_ratio_of_allocated,

            "pending_at_end":
                pending_at_end,

            "offered_throughput_per_1000":
                offered_throughput,

            "allocation_throughput_per_1000":
                allocation_throughput,

            "completion_throughput_per_1000":
                completion_throughput,

            "avg_active_vehicles":
                avg_active_vehicles,

            "avg_eligible_candidates":
                avg_eligible_candidates,

            "avg_eligible_when_active":
                avg_eligible_when_active,

            "accounting_ok":
                accounting_ok
        })


# ============================================================
# DATAFRAME
# ============================================================

df_seed = pd.DataFrame(
    resultados
)


if df_seed.empty:

    raise RuntimeError(
        "Nenhum experimento C14 válido encontrado."
    )


df_seed.to_csv(
    OUTPUT_PER_SEED,
    index=False
)


# ============================================================
# COMPLETUDE
# ============================================================

print("\n")

print(
    "DATASET COMPLETENESS"
)

print("-" * 175)


for configuracao in ORDEM:

    n = len(

        df_seed[
            df_seed[
                "configuration"
            ]
            ==
            configuracao
        ]
    )

    print(

        f"{configuracao:<20}: "
        f"{n:>2}/30 runs"
    )


if faltantes:

    print(

        f"\n[WARNING] Missing: "
        f"{len(faltantes)}"
    )


if invalidos:

    print(

        f"\n[WARNING] Invalid: "
        f"{len(invalidos)}"
    )


# ============================================================
# SANITY
# ============================================================

if (
    df_seed[
        "accounting_ok"
    ].all()
):

    print(
        "\n[PASS] Request accounting valid in all runs."
    )

else:

    print(
        "\n[FAIL] Request accounting mismatch detected."
    )


# ============================================================
# AGREGAÇÃO
# ============================================================

METRICAS = [

    "total_requests",

    "active_request_count",

    "allocated_count",

    "completed_count",

    "unserved_count",

    "service_ratio",

    "conditional_service_ratio",

    "unserved_ratio",

    "vehicle_availability_ratio",

    "no_active_ratio",

    "no_eligible_ratio",

    "no_eligible_given_active",

    "completion_ratio_of_allocated",

    "pending_at_end",

    "offered_throughput_per_1000",

    "allocation_throughput_per_1000",

    "completion_throughput_per_1000",

    "avg_active_vehicles",

    "avg_eligible_when_active"
]


resumo = []


for configuracao in ORDEM:

    dados = df_seed[
        df_seed[
            "configuration"
        ]
        ==
        configuracao
    ]

    config = CONFIGURACOES[
        configuracao
    ]

    linha = {

        "configuration":
            configuracao,

        "workload":
            config[
                "label"
            ],

        "request_interval":
            config[
                "interval"
            ],

        "load_factor_vs_baseline":

            CONFIGURACOES[
                BASELINE
            ][
                "interval"
            ]
            /
            config[
                "interval"
            ],

        "n_runs":
            len(
                dados
            )
    }

    for metrica in METRICAS:

        (
            media,
            sd,
            low,
            high
        ) = calcular_ic95(

            dados[
                metrica
            ]
        )

        linha[
            f"{metrica}_mean"
        ] = media

        linha[
            f"{metrica}_sd"
        ] = sd

        linha[
            f"{metrica}_ci95_low"
        ] = low

        linha[
            f"{metrica}_ci95_high"
        ] = high

    resumo.append(
        linha
    )


df_agg = pd.DataFrame(
    resumo
)


df_agg.to_csv(
    OUTPUT_AGG,
    index=False
)


# ============================================================
# MAIN RESULTS
# ============================================================

print("\n")

print(
    "MAIN CAPACITY RESULTS"
)

print("=" * 175)


header = (

    f"{'LOAD':<12}"

    f"{'INT':>6}"

    f"{'REQ':>8}"

    f"{'ACTIVE':>9}"

    f"{'ALLOC':>8}"

    f"{'E2E%':>9}"

    f"{'COND%':>9}"

    f"{'NOACT%':>9}"

    f"{'NOELIG%':>10}"

    f"{'ALLOC TP':>11}"

    f"{'COMP TP':>11}"

    f"{'PENDING':>10}"
)


print(
    header
)

print("-" * 175)


for _, row in df_agg.iterrows():

    print(

        f"{row['workload']:<12}"

        f"{int(row['request_interval']):>6}"

        f"{row['total_requests_mean']:>8.2f}"

        f"{row['active_request_count_mean']:>9.2f}"

        f"{row['allocated_count_mean']:>8.2f}"

        f"{row['service_ratio_mean'] * 100:>8.2f}%"

        f"{row['conditional_service_ratio_mean'] * 100:>8.2f}%"

        f"{row['no_active_ratio_mean'] * 100:>8.2f}%"

        f"{row['no_eligible_ratio_mean'] * 100:>9.2f}%"

        f"{row['allocation_throughput_per_1000_mean']:>11.2f}"

        f"{row['completion_throughput_per_1000_mean']:>11.2f}"

        f"{row['pending_at_end_mean']:>10.2f}"
    )


# ============================================================
# PAIRED VS BASELINE
# ============================================================

baseline = (

    df_seed[
        df_seed[
            "configuration"
        ]
        ==
        BASELINE
    ]

    .copy()
)


metricas_pareadas = [

    "service_ratio",

    "conditional_service_ratio",

    "vehicle_availability_ratio",

    "no_active_ratio",

    "no_eligible_ratio",

    "allocation_throughput_per_1000",

    "completion_throughput_per_1000"
]


comparacoes = []


for configuracao in ORDEM:

    if configuracao == BASELINE:
        continue

    alternativa = (

        df_seed[
            df_seed[
                "configuration"
            ]
            ==
            configuracao
        ]

        .copy()
    )

    pareado = pd.merge(

        baseline,

        alternativa,

        on="seed",

        suffixes=(
            "_baseline",
            "_alternative"
        )
    )

    for metrica in metricas_pareadas:

        diferencas = (

            pareado[
                f"{metrica}_alternative"
            ]

            -

            pareado[
                f"{metrica}_baseline"
            ]
        )

        (
            media,
            sd,
            low,
            high
        ) = calcular_ic95(
            diferencas
        )

        comparacoes.append({

            "configuration":
                configuracao,

            "workload":
                CONFIGURACOES[
                    configuracao
                ][
                    "label"
                ],

            "metric":
                metrica,

            "n_pairs":
                len(
                    pareado
                ),

            "mean_difference":
                media,

            "difference_sd":
                sd,

            "ci95_low":
                low,

            "ci95_high":
                high,

            "ci95_excludes_zero":

                (
                    low > 0
                    or
                    high < 0
                )

                if (
                    not pd.isna(
                        low
                    )
                    and
                    not pd.isna(
                        high
                    )
                )

                else False
        })


df_paired = pd.DataFrame(
    comparacoes
)


df_paired.to_csv(
    OUTPUT_PAIRED,
    index=False
)


# ============================================================
# BASELINE DETAIL
# ============================================================

print("\n")

print(
    "BASELINE CAPACITY DETAIL (INTERVAL=20)"
)

print("=" * 175)


row = df_agg[
    df_agg[
        "configuration"
    ]
    ==
    BASELINE
].iloc[0]


print(

    f"End-to-end service ratio       : "
    f"{row['service_ratio_mean'] * 100:.2f}%"
)

print(

    f"Conditional service ratio      : "
    f"{row['conditional_service_ratio_mean'] * 100:.2f}%"
)

print(

    f"Vehicle availability ratio     : "
    f"{row['vehicle_availability_ratio_mean'] * 100:.2f}%"
)

print(

    f"No-active-vehicle ratio        : "
    f"{row['no_active_ratio_mean'] * 100:.2f}%"
)

print(

    f"No-eligible-candidate ratio    : "
    f"{row['no_eligible_ratio_mean'] * 100:.2f}%"
)

print(

    f"No eligible given availability : "
    f"{row['no_eligible_given_active_mean'] * 100:.2f}%"
)

print(

    f"Allocation throughput / 1000   : "
    f"{row['allocation_throughput_per_1000_mean']:.2f}"
)

print(

    f"Completion throughput / 1000   : "
    f"{row['completion_throughput_per_1000_mean']:.2f}"
)

print(

    f"Completion ratio of allocated  : "
    f"{row['completion_ratio_of_allocated_mean'] * 100:.2f}%"
)

print(

    f"Pending services at end        : "
    f"{row['pending_at_end_mean']:.2f}"
)


# ============================================================
# LATEX
# ============================================================

latex = []


latex.append(
    r"\begin{table*}[t]"
)

latex.append(
    r"\centering"
)

latex.append(
    r"\caption{SHADAI capacity under increasing offered workload. End-to-end service ratio includes requests generated when no vehicle is active, whereas conditional service ratio considers only requests generated when at least one vehicle is available. Values are means over 30 simulation seeds.}"
)

latex.append(
    r"\label{tab:capacity_load}"
)

latex.append(
    r"\resizebox{\textwidth}{!}{%"
)

latex.append(
    r"\begin{tabular}{l|cccccccc}"
)

latex.append(
    r"\hline"
)

latex.append(
    r"Workload & Interval & Offered Req. & E2E Service & Conditional Service & No Active Vehicle & No Eligible Candidate & Alloc. Throughput & Completion Throughput \\"
)

latex.append(
    r"\hline"
)


for _, row in df_agg.iterrows():

    nome = row[
        "workload"
    ]

    if row[
        "configuration"
    ] == BASELINE:

        nome = (
            r"\textbf{High (baseline)}"
        )

    latex.append(

        f"{nome} & "

        f"{int(row['request_interval'])} & "

        f"{row['total_requests_mean']:.1f} & "

        f"{row['service_ratio_mean'] * 100:.2f}\\% & "

        f"{row['conditional_service_ratio_mean'] * 100:.2f}\\% & "

        f"{row['no_active_ratio_mean'] * 100:.2f}\\% & "

        f"{row['no_eligible_ratio_mean'] * 100:.2f}\\% & "

        f"{row['allocation_throughput_per_1000_mean']:.2f} & "

        f"{row['completion_throughput_per_1000_mean']:.2f} "

        r"\\"
    )


latex.append(
    r"\hline"
)

latex.append(
    r"\end{tabular}%"
)

latex.append(
    r"}"
)

latex.append(
    r"\vspace{1mm}"
)

latex.append(
    r"\parbox{0.98\textwidth}{\footnotesize Throughput is reported per 1000 simulation steps. Requests without active vehicles are classified separately from requests for which vehicles are present but no candidate satisfies the SHADAI eligibility conditions. The current implementation is a loss system: unserved requests are not queued for later processing.}"
)

latex.append(
    r"\end{table*}"
)


with open(
    OUTPUT_LATEX,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(
            latex
        )
    )


# ============================================================
# OUTPUT
# ============================================================

print("\n")

print(
    "OUTPUT FILES"
)

print("=" * 175)

print(
    f"Per seed   : {OUTPUT_PER_SEED}"
)

print(
    f"Aggregated : {OUTPUT_AGG}"
)

print(
    f"Paired     : {OUTPUT_PAIRED}"
)

print(
    f"LaTeX      : {OUTPUT_LATEX}"
)

print("\n")

print("=" * 175)

print(
    "C14 CAPACITY ANALYSIS COMPLETED"
)

print("=" * 175)

print()