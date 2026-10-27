# shadai_sumo.py

import argparse
import csv
import os
import random
import time
import xml.etree.ElementTree as ET

import traci

from lider import Plataforma
from veiculo import Veiculo


# ============================================================
# ARGUMENTOS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "SHADAI SUMO simulation with normalized utility, split RNG "
        "streams, malicious/flood experiments, task reliability "
        "telemetry, and optional dynamic pricing."
    )
)

# Utility lambdas
parser.add_argument("--lambda1", type=float, default=0.5)
parser.add_argument("--lambda2", type=float, default=0.3)
parser.add_argument("--lambda3", type=float, default=0.2)

# Score weights
parser.add_argument("--w1", type=float, default=0.4)
parser.add_argument("--w2", type=float, default=0.3)
parser.add_argument("--w3", type=float, default=0.3)

# epsilon-greedy
parser.add_argument("--epsilon", type=float, default=0.2)

# cooldown and workload
parser.add_argument("--cooldown-steps", type=int, default=40)
parser.add_argument("--request-interval", type=int, default=20)

# C25 - pricing
parser.add_argument(
    "--pricing-mode",
    type=str,
    choices=["static", "dynamic"],
    default="static",
    help=(
        "static preserves the original service-specific prices; "
        "dynamic applies supply/demand dependent pricing."
    ),
)
parser.add_argument(
    "--pricing-alpha",
    type=float,
    default=0.20,
    help="Sensitivity to exogenous demand pressure.",
)
parser.add_argument(
    "--pricing-beta",
    type=float,
    default=0.20,
    help="Sensitivity to active-vehicle supply scarcity.",
)
parser.add_argument(
    "--pricing-reference-interval",
    type=int,
    default=20,
    help="Reference request interval used by the dynamic-pricing rule.",
)

# Adversarial configuration
parser.add_argument("--malicious-ratio", type=float, default=0.20)
parser.add_argument(
    "--attack-type",
    type=str,
    choices=["none", "failure", "flood"],
    default="none",
)
parser.add_argument(
    "--attack-enabled",
    action="store_true",
    help=(
        "Legacy compatibility: if supplied while attack-type is none, "
        "the attack becomes deliberate failure."
    ),
)
parser.add_argument(
    "--flood-intensity",
    type=int,
    default=3,
    help=(
        "Number of additional artificial candidacies per eligible "
        "malicious vehicle for the same request."
    ),
)

# Experiment
parser.add_argument("--experiment", type=str, default="normalized_original")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--output-dir", type=str, default=".")

args = parser.parse_args()


# ============================================================
# PARÂMETROS
# ============================================================

LAMBDA1 = args.lambda1
LAMBDA2 = args.lambda2
LAMBDA3 = args.lambda3

W1 = args.w1
W2 = args.w2
W3 = args.w3

EPSILON = args.epsilon

COOLDOWN_STEPS = args.cooldown_steps
REQUEST_INTERVAL = args.request_interval

PRICING_MODE = args.pricing_mode
PRICING_ALPHA = args.pricing_alpha
PRICING_BETA = args.pricing_beta
PRICING_REFERENCE_INTERVAL = args.pricing_reference_interval

MALICIOUS_RATIO = args.malicious_ratio
FLOOD_INTENSITY = args.flood_intensity

EXPERIMENTO = args.experiment
SEED = args.seed
OUTPUT_DIR = args.output_dir

ATTACK_TYPE = args.attack_type
if args.attack_enabled and ATTACK_TYPE == "none":
    ATTACK_TYPE = "failure"

ATTACK_ENABLED = ATTACK_TYPE != "none"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# VERSIONAMENTO
# ============================================================

RNG_SCHEME = "split_v1"

# Kept for compatibility with C18-generated telemetry.
TELEMETRY_SCHEME = "c18_flood_intensity_v1"

SUMMARY_SCHEME = "c19_task_reliability_v1"

PRICING_SCHEME = "c25_supply_demand_dynamic_v1"

ATTACK_SCHEME = {
    "none": "none_v1",
    "failure": "deliberate_service_failure_v1",
    "flood": "flood_candidate_duplication_v1",
}[ATTACK_TYPE]

FLOOD_SCHEME = "c18_flood_param_v1"


# ============================================================
# RNG STREAMS
# ============================================================

SEED_REQUEST = SEED + 100_000
SEED_VEHICLE = SEED + 200_000
SEED_RELIABILITY = SEED + 300_000
SEED_POLICY_GATE = SEED + 400_000
SEED_POLICY_CHOICE = SEED + 500_000
SEED_ACTIVE_SAMPLE = SEED + 600_000
SEED_MALICIOUS = SEED + 700_000
SEED_FLOOD = SEED + 800_000  # reserved for future stochastic flood variants

RNG_REQUEST = random.Random(SEED_REQUEST)
RNG_VEHICLE = random.Random(SEED_VEHICLE)
RNG_RELIABILITY = random.Random(SEED_RELIABILITY)
RNG_POLICY_GATE = random.Random(SEED_POLICY_GATE)
RNG_POLICY_CHOICE = random.Random(SEED_POLICY_CHOICE)
RNG_ACTIVE_SAMPLE = random.Random(SEED_ACTIVE_SAMPLE)
RNG_MALICIOUS = random.Random(SEED_MALICIOUS)
RNG_FLOOD = random.Random(SEED_FLOOD)


# ============================================================
# VALIDAÇÕES
# ============================================================

if abs(LAMBDA1 + LAMBDA2 + LAMBDA3 - 1.0) > 1e-6:
    raise ValueError("lambda1 + lambda2 + lambda3 must sum to 1.0.")

if any(x < 0 for x in [LAMBDA1, LAMBDA2, LAMBDA3]):
    raise ValueError("Lambda weights cannot be negative.")

if abs(W1 + W2 + W3 - 1.0) > 1e-6:
    raise ValueError("w1 + w2 + w3 must sum to 1.0.")

if any(x < 0 for x in [W1, W2, W3]):
    raise ValueError("Score weights cannot be negative.")

if not 0.0 <= EPSILON <= 1.0:
    raise ValueError("epsilon must be between 0.0 and 1.0.")

if COOLDOWN_STEPS < 0:
    raise ValueError("cooldown_steps must be >= 0.")

if REQUEST_INTERVAL <= 0:
    raise ValueError("request_interval must be > 0.")

if PRICING_ALPHA < 0:
    raise ValueError("pricing_alpha must be >= 0.")

if PRICING_BETA < 0:
    raise ValueError("pricing_beta must be >= 0.")

if PRICING_REFERENCE_INTERVAL <= 0:
    raise ValueError("pricing_reference_interval must be > 0.")

if not 0.0 <= MALICIOUS_RATIO <= 1.0:
    raise ValueError("malicious_ratio must be between 0.0 and 1.0.")

if FLOOD_INTENSITY < 0:
    raise ValueError("flood_intensity must be >= 0.")


# ============================================================
# CENÁRIO
# ============================================================

MODO = "shadai"
MAX_VEICULOS = 400

CUSTO_BASE_MIN = 3.0
CUSTO_BASE_MAX = 8.0


# ============================================================
# SERVIÇOS
# ============================================================

SERVICOS = [
    {
        "nome": "Processamento de Imagem Urbana",
        "energia": 18,
        "duracao": 60,
        "preco": 45.0,
    },
    {
        "nome": "Agregação de Dados de Sensores",
        "energia": 8,
        "duracao": 30,
        "preco": 22.0,
    },
    {
        "nome": "Reencaminhamento de Dados (Edge Relay)",
        "energia": 5,
        "duracao": 25,
        "preco": 15.0,
    },
    {
        "nome": "Atualização de Modelos Locais (ML)",
        "energia": 22,
        "duracao": 80,
        "preco": 60.0,
    },
    {
        "nome": "Diagnóstico Veicular",
        "energia": 10,
        "duracao": 40,
        "preco": 28.0,
    },
]


# ============================================================
# NORMALIZAÇÃO
# ============================================================

PRECO_MAX = max(s["preco"] for s in SERVICOS)
ENERGIA_SERVICO_MAX = 25.0
DISTANCIA_MAX = 1000.0
TEMPO_MAX = 100.0


# ============================================================
# ESTADO
# ============================================================

veiculos = {}
servicos_ativos = []

plataforma = Plataforma(
    epsilon=EPSILON,
    w1=W1,
    w2=W2,
    w3=W3,
    cooldown_steps=COOLDOWN_STEPS,
    rng_policy_gate=RNG_POLICY_GATE,
    rng_policy_choice=RNG_POLICY_CHOICE,
)


# ============================================================
# ARQUIVOS
# ============================================================

METRICS_FILE = os.path.join(
    OUTPUT_DIR, f"metricas_{MODO}_{MAX_VEICULOS}_{EXPERIMENTO}.csv"
)

ELIGIBILITY_FILE = os.path.join(
    OUTPUT_DIR, f"eligibilidade_{MODO}_{MAX_VEICULOS}_{EXPERIMENTO}.csv"
)

REQUESTS_FILE = os.path.join(
    OUTPUT_DIR, f"requests_{MODO}_{MAX_VEICULOS}_{EXPERIMENTO}.csv"
)

RUNTIME_FILE = os.path.join(
    OUTPUT_DIR, f"runtime_{MODO}_{MAX_VEICULOS}_{EXPERIMENTO}.csv"
)

VEHICLES_FILE = os.path.join(
    OUTPUT_DIR, f"vehicles_{MODO}_{MAX_VEICULOS}_{EXPERIMENTO}.csv"
)

OUTCOMES_FILE = os.path.join(
    OUTPUT_DIR, f"outcomes_{MODO}_{MAX_VEICULOS}_{EXPERIMENTO}.csv"
)

FLOOD_FILE = os.path.join(
    OUTPUT_DIR, f"flood_{MODO}_{MAX_VEICULOS}_{EXPERIMENTO}.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR, f"summary_{MODO}_{MAX_VEICULOS}_{EXPERIMENTO}.csv"
)


# ============================================================
# METADADOS COMUNS
# ============================================================

COMMON_COLUMNS = [
    "experimento",
    "seed",
    "rng_scheme",
    "telemetry_scheme",
    "attack_scheme",
    "flood_scheme",
    "attack_type",
    "malicious_ratio",
    "flood_intensity",
    "pricing_scheme",
    "pricing_mode",
    "pricing_alpha",
    "pricing_beta",
    "pricing_reference_interval",
]


def common_values():
    return [
        EXPERIMENTO,
        SEED,
        RNG_SCHEME,
        TELEMETRY_SCHEME,
        ATTACK_SCHEME,
        FLOOD_SCHEME,
        ATTACK_TYPE,
        MALICIOUS_RATIO,
        FLOOD_INTENSITY,
        PRICING_SCHEME,
        PRICING_MODE,
        PRICING_ALPHA,
        PRICING_BETA,
        PRICING_REFERENCE_INTERVAL,
    ]


# ============================================================
# CSV HEADERS
# ============================================================

with open(VEHICLES_FILE, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(
        COMMON_COLUMNS
        + [
            "vehicle",
            "malicious_draw",
            "malicious",
            "initial_reputation",
            "initial_energy",
        ]
    )


with open(METRICS_FILE, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(
        COMMON_COLUMNS
        + [
            "lambda1",
            "lambda2",
            "lambda3",
            "w1",
            "w2",
            "w3",
            "epsilon",
            "cooldown_steps",
            "request_interval",
            "modo",
            "step",
            "veiculo",
            "selected_malicious",
            "servico",
            "base_price",
            "offered_price",
            "demand_pressure",
            "supply_scarcity",
            "price_multiplier",
            "price_premium_rate",
            "score_base",
            "score",
            "reputacao_norm",
            "energia_norm",
            "proximidade_norm",
            "distancia",
            "cooldown_penalty",
            "last_selection_step",
            "steps_since_last_selection",
            "selection_mode",
            "candidate_count",
            "unique_candidate_count",
            "artificial_candidate_count",
            "malicious_candidate_entries",
            "unique_malicious_candidates",
            "chosen_rank",
            "greedy_best_vehicle",
            "greedy_best_score",
            "greedy_best_reputation_norm",
            "greedy_best_energy_norm",
            "greedy_best_proximity_norm",
            "greedy_best_distance",
            "exploration_regret",
            "exploration_regret_relative",
            "selected_greedy_best",
            "offer_runtime_ms",
            "selection_runtime_ms",
            "decision_runtime_ms",
            "utilidade",
            "reputacao",
            "energia",
        ]
    )


with open(ELIGIBILITY_FILE, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(
        COMMON_COLUMNS
        + [
            "lambda1",
            "lambda2",
            "lambda3",
            "w1",
            "w2",
            "w3",
            "epsilon",
            "cooldown_steps",
            "request_interval",
            "step",
            "servico",
            "base_price",
            "offered_price",
            "demand_pressure",
            "supply_scarcity",
            "price_multiplier",
            "active_vehicles",
            "positive_utility",
            "sufficient_energy",
            "eligible",
            "rejected_utility",
            "rejected_energy",
            "rejected_utility_only",
            "rejected_energy_only",
            "rejected_both",
            "eligibility_rate",
            "avg_utility",
            "min_utility",
            "max_utility",
        ]
    )


with open(REQUESTS_FILE, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(
        COMMON_COLUMNS
        + [
            "request_interval",
            "cooldown_steps",
            "step",
            "servico",
            "active_vehicles",
            "unique_eligible_candidates",
            "candidate_entries_after_flood",
            "artificial_candidate_count",
            "base_price",
            "offered_price",
            "demand_pressure",
            "supply_scarcity",
            "price_multiplier",
            "price_premium",
            "price_premium_rate",
            "allocated",
            "failure_reason",
            "selected_vehicle",
            "selected_malicious",
            "completion_step",
        ]
    )


with open(RUNTIME_FILE, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(
        COMMON_COLUMNS
        + [
            "request_interval",
            "cooldown_steps",
            "step",
            "active_vehicles",
            "unique_candidate_count",
            "candidate_count",
            "artificial_candidate_count",
            "allocated",
            "offer_executed",
            "selection_executed",
            "offer_runtime_ms",
            "selection_runtime_ms",
            "decision_runtime_ms",
        ]
    )


with open(OUTCOMES_FILE, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(
        COMMON_COLUMNS
        + [
            "selection_step",
            "completion_step",
            "vehicle",
            "malicious",
            "service",
            "reliability_draw",
            "success",
            "reputation_before",
            "reputation_after",
            "reputation_delta",
        ]
    )


with open(FLOOD_FILE, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(
        COMMON_COLUMNS
        + [
            "step",
            "unique_candidate_count",
            "unique_malicious_candidates",
            "honest_candidate_count",
            "artificial_candidate_count",
            "candidate_entries_after_flood",
            "malicious_candidate_entries",
            "malicious_entry_share",
            "selected_vehicle",
            "selected_malicious",
            "selection_mode",
        ]
    )


# ============================================================
# CONTADORES
# ============================================================

total_requests = 0
successful_allocations = 0
unserved_requests = 0

completed_services = 0
successful_services = 0
failed_services = 0

malicious_selections = 0
honest_selections = 0

total_artificial_candidates = 0

# C25
total_base_price = 0.0
total_offered_price = 0.0
total_price_multiplier = 0.0
total_demand_pressure = 0.0
total_supply_scarcity = 0.0
dynamic_price_increase_count = 0


# ============================================================
# HELPERS
# ============================================================

def ler_tempo_final_sumocfg(sumocfg_path):
    tree = ET.parse(sumocfg_path)
    root = tree.getroot()
    end = root.find("./time/end")

    if end is not None:
        return int(float(end.attrib["value"]))

    return 3600


def is_malicious(vehicle):
    return bool(
        getattr(
            vehicle,
            "malicioso",
            False,
        )
    )


def clip(value, minimum, maximum):
    return max(
        minimum,
        min(value, maximum),
    )


# ============================================================
# C25 - DYNAMIC PRICING
# ============================================================

def calcular_preco_dinamico(
    servico,
    active_vehicle_count,
):
    """
    Static:
        p_t = p_base

    Dynamic:
        p_t = clip(
            p_base * [1 + alpha*D_t + beta*S_t],
            p_base,
            PRECO_MAX
        )

    Demand:
        D_t = clip(I_ref / I_t - 1, 0, 1)

    Scarcity:
        S_t = 1 - clip(N_active / N_max, 0, 1)

    Eligibility is NOT used here, avoiding
    p -> utility -> eligibility -> p circularity.
    """

    base_price = float(
        servico["preco"]
    )

    if PRICING_MODE == "static":
        return {
            "base_price": base_price,
            "offered_price": base_price,
            "demand_pressure": 0.0,
            "supply_scarcity": 0.0,
            "price_multiplier": 1.0,
            "price_premium": 0.0,
            "price_premium_rate": 0.0,
        }

    relative_request_rate = (
        PRICING_REFERENCE_INTERVAL
        /
        float(REQUEST_INTERVAL)
    )

    demand_pressure = clip(
        relative_request_rate - 1.0,
        0.0,
        1.0,
    )

    active_ratio = clip(
        active_vehicle_count
        /
        float(MAX_VEICULOS),
        0.0,
        1.0,
    )

    supply_scarcity = (
        1.0 - active_ratio
    )

    raw_multiplier = (
        1.0
        + PRICING_ALPHA * demand_pressure
        + PRICING_BETA * supply_scarcity
    )

    raw_price = (
        base_price
        *
        raw_multiplier
    )

    offered_price = clip(
        raw_price,
        base_price,
        PRECO_MAX,
    )

    effective_multiplier = (
        offered_price / base_price
        if base_price > 0
        else 1.0
    )

    price_premium = (
        offered_price
        -
        base_price
    )

    price_premium_rate = (
        price_premium / base_price
        if base_price > 0
        else 0.0
    )

    return {
        "base_price":
            base_price,

        "offered_price":
            offered_price,

        "demand_pressure":
            demand_pressure,

        "supply_scarcity":
            supply_scarcity,

        "price_multiplier":
            effective_multiplier,

        "price_premium":
            price_premium,

        "price_premium_rate":
            price_premium_rate,
    }


# ============================================================
# VEHICLE LOG
# ============================================================

def registrar_veiculo(vehicle):
    with open(
        VEHICLES_FILE,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        csv.writer(f).writerow(
            common_values()
            + [
                vehicle.id,
                round(
                    getattr(
                        vehicle,
                        "malicious_draw",
                        0.0,
                    ),
                    9,
                ),
                is_malicious(vehicle),
                round(vehicle.reputacao, 6),
                round(vehicle.energia, 6),
            ]
        )


# ============================================================
# FLOOD
# ============================================================

def aplicar_flood(candidatos):
    if (
        ATTACK_TYPE != "flood"
        or
        FLOOD_INTENSITY == 0
    ):
        return (
            list(candidatos),
            0,
        )

    resultado = list(candidatos)
    artificial_count = 0

    for candidato in candidatos:
        v = candidato["veiculo"]

        if not is_malicious(v):
            continue

        for _ in range(
            FLOOD_INTENSITY
        ):
            duplicate = (
                candidato.copy()
            )

            duplicate[
                "flood_duplicate"
            ] = True

            resultado.append(
                duplicate
            )

            artificial_count += 1

    return (
        resultado,
        artificial_count,
    )


# ============================================================
# LOGGING
# ============================================================

def registrar_runtime(
    step,
    active_vehicles,
    unique_candidate_count,
    candidate_count,
    artificial_candidate_count,
    allocated,
    offer_executed,
    selection_executed,
    offer_runtime_ms,
    selection_runtime_ms,
):
    decision_runtime_ms = (
        offer_runtime_ms
        +
        selection_runtime_ms
    )

    with open(
        RUNTIME_FILE,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        csv.writer(f).writerow(
            common_values()
            + [
                REQUEST_INTERVAL,
                COOLDOWN_STEPS,
                step,
                active_vehicles,
                unique_candidate_count,
                candidate_count,
                artificial_candidate_count,
                allocated,
                offer_executed,
                selection_executed,
                round(
                    offer_runtime_ms,
                    9,
                ),
                round(
                    selection_runtime_ms,
                    9,
                ),
                round(
                    decision_runtime_ms,
                    9,
                ),
            ]
        )


def registrar_request(
    step,
    servico,
    active_vehicles,
    unique_eligible_candidates,
    candidate_entries_after_flood,
    artificial_candidate_count,
    pricing_info,
    allocated,
    failure_reason,
    selected_vehicle="",
    selected_malicious="",
    completion_step="",
):
    with open(
        REQUESTS_FILE,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        csv.writer(f).writerow(
            common_values()
            + [
                REQUEST_INTERVAL,
                COOLDOWN_STEPS,
                step,
                servico["nome"],
                active_vehicles,
                unique_eligible_candidates,
                candidate_entries_after_flood,
                artificial_candidate_count,
                round(
                    pricing_info["base_price"],
                    6,
                ),
                round(
                    pricing_info["offered_price"],
                    6,
                ),
                round(
                    pricing_info["demand_pressure"],
                    9,
                ),
                round(
                    pricing_info["supply_scarcity"],
                    9,
                ),
                round(
                    pricing_info["price_multiplier"],
                    9,
                ),
                round(
                    pricing_info["price_premium"],
                    6,
                ),
                round(
                    pricing_info["price_premium_rate"],
                    9,
                ),
                allocated,
                failure_reason,
                selected_vehicle,
                selected_malicious,
                completion_step,
            ]
        )


def registrar_flood(
    step,
    candidatos_unicos,
    candidatos_flood,
    artificial_count,
    escolhido=None,
):
    unique_malicious = sum(
        1
        for c in candidatos_unicos
        if is_malicious(
            c["veiculo"]
        )
    )

    honest_unique = (
        len(candidatos_unicos)
        -
        unique_malicious
    )

    malicious_entries = sum(
        1
        for c in candidatos_flood
        if is_malicious(
            c["veiculo"]
        )
    )

    total_entries = len(
        candidatos_flood
    )

    malicious_entry_share = (
        malicious_entries
        /
        total_entries
        if total_entries > 0
        else 0.0
    )

    selected_vehicle = ""
    selected_malicious = ""
    selection_mode = ""

    if escolhido is not None:
        selected_vehicle = (
            escolhido[
                "veiculo"
            ].id
        )

        selected_malicious = (
            is_malicious(
                escolhido[
                    "veiculo"
                ]
            )
        )

        selection_mode = (
            escolhido[
                "selection_mode"
            ]
        )

    with open(
        FLOOD_FILE,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        csv.writer(f).writerow(
            common_values()
            + [
                step,
                len(
                    candidatos_unicos
                ),
                unique_malicious,
                honest_unique,
                artificial_count,
                total_entries,
                malicious_entries,
                round(
                    malicious_entry_share,
                    9,
                ),
                selected_vehicle,
                selected_malicious,
                selection_mode,
            ]
        )


def registrar_metricas(
    step,
    escolhido,
    servico,
    pricing_info,
    offer_runtime_ms,
    selection_runtime_ms,
    candidatos_unicos,
    candidatos_flood,
    artificial_count,
):
    v = escolhido["veiculo"]

    decision_runtime_ms = (
        offer_runtime_ms
        +
        selection_runtime_ms
    )

    unique_malicious = sum(
        1
        for c in candidatos_unicos
        if is_malicious(
            c["veiculo"]
        )
    )

    malicious_entries = sum(
        1
        for c in candidatos_flood
        if is_malicious(
            c["veiculo"]
        )
    )

    with open(
        METRICS_FILE,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        csv.writer(f).writerow(
            common_values()
            + [
                LAMBDA1,
                LAMBDA2,
                LAMBDA3,
                W1,
                W2,
                W3,
                EPSILON,
                COOLDOWN_STEPS,
                REQUEST_INTERVAL,
                MODO,
                step,
                v.id,
                is_malicious(v),
                servico["nome"],
                round(
                    pricing_info["base_price"],
                    6,
                ),
                round(
                    pricing_info["offered_price"],
                    6,
                ),
                round(
                    pricing_info["demand_pressure"],
                    9,
                ),
                round(
                    pricing_info["supply_scarcity"],
                    9,
                ),
                round(
                    pricing_info["price_multiplier"],
                    9,
                ),
                round(
                    pricing_info["price_premium_rate"],
                    9,
                ),
                round(
                    escolhido["score_base"],
                    6,
                ),
                round(
                    escolhido["score"],
                    6,
                ),
                round(
                    escolhido["reputacao_norm"],
                    6,
                ),
                round(
                    escolhido["energia_norm"],
                    6,
                ),
                round(
                    escolhido["proximidade_norm"],
                    6,
                ),
                round(
                    escolhido["distancia"],
                    6,
                ),
                escolhido[
                    "cooldown_penalty"
                ],
                escolhido[
                    "last_selection_step"
                ],
                escolhido[
                    "steps_since_last_selection"
                ],
                escolhido[
                    "selection_mode"
                ],
                len(
                    candidatos_flood
                ),
                len(
                    candidatos_unicos
                ),
                artificial_count,
                malicious_entries,
                unique_malicious,
                escolhido[
                    "chosen_rank"
                ],
                escolhido[
                    "greedy_best_vehicle"
                ],
                round(
                    escolhido["greedy_best_score"],
                    6,
                ),
                round(
                    escolhido["greedy_best_reputation_norm"],
                    6,
                ),
                round(
                    escolhido["greedy_best_energy_norm"],
                    6,
                ),
                round(
                    escolhido["greedy_best_proximity_norm"],
                    6,
                ),
                round(
                    escolhido["greedy_best_distance"],
                    6,
                ),
                round(
                    escolhido["exploration_regret"],
                    6,
                ),
                round(
                    escolhido["exploration_regret_relative"],
                    6,
                ),
                escolhido[
                    "selected_greedy_best"
                ],
                round(
                    offer_runtime_ms,
                    9,
                ),
                round(
                    selection_runtime_ms,
                    9,
                ),
                round(
                    decision_runtime_ms,
                    9,
                ),
                round(
                    escolhido["utilidade"],
                    6,
                ),
                round(
                    v.reputacao,
                    6,
                ),
                # Available energy at selection time,
                # before service execution.
                round(
                    v.energia,
                    6,
                ),
            ]
        )


# ============================================================
# ELIGIBILITY
# ============================================================

def analisar_elegibilidade(
    step,
    servico,
    pos_tarefa,
    veiculos_ativos,
    pricing_info,
):
    utilities = []

    positive_utility = 0
    sufficient_energy = 0
    eligible = 0

    rejected_utility = 0
    rejected_energy = 0

    rejected_utility_only = 0
    rejected_energy_only = 0
    rejected_both = 0

    for v in veiculos_ativos:
        (
            utilidade,
            _,
            _,
            _,
        ) = v.calcular_utilidade(
            plataforma.preco,
            servico,
            pos_tarefa,
        )

        utilities.append(
            utilidade
        )

        utility_ok = (
            utilidade > 0
        )

        energy_ok = (
            v.energia
            >=
            servico["energia"]
        )

        if utility_ok:
            positive_utility += 1
        else:
            rejected_utility += 1

        if energy_ok:
            sufficient_energy += 1
        else:
            rejected_energy += 1

        if utility_ok and energy_ok:
            eligible += 1

        elif (
            not utility_ok
            and energy_ok
        ):
            rejected_utility_only += 1

        elif (
            utility_ok
            and not energy_ok
        ):
            rejected_energy_only += 1

        else:
            rejected_both += 1

    total = len(
        veiculos_ativos
    )

    eligibility_rate = (
        eligible / total
        if total > 0
        else 0.0
    )

    avg_utility = (
        sum(utilities)
        /
        len(utilities)
        if utilities
        else 0.0
    )

    min_utility = (
        min(utilities)
        if utilities
        else 0.0
    )

    max_utility = (
        max(utilities)
        if utilities
        else 0.0
    )

    with open(
        ELIGIBILITY_FILE,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        csv.writer(f).writerow(
            common_values()
            + [
                LAMBDA1,
                LAMBDA2,
                LAMBDA3,
                W1,
                W2,
                W3,
                EPSILON,
                COOLDOWN_STEPS,
                REQUEST_INTERVAL,
                step,
                servico["nome"],
                round(
                    pricing_info["base_price"],
                    6,
                ),
                round(
                    pricing_info["offered_price"],
                    6,
                ),
                round(
                    pricing_info["demand_pressure"],
                    9,
                ),
                round(
                    pricing_info["supply_scarcity"],
                    9,
                ),
                round(
                    pricing_info["price_multiplier"],
                    9,
                ),
                total,
                positive_utility,
                sufficient_energy,
                eligible,
                rejected_utility,
                rejected_energy,
                rejected_utility_only,
                rejected_energy_only,
                rejected_both,
                round(
                    eligibility_rate,
                    6,
                ),
                round(
                    avg_utility,
                    6,
                ),
                round(
                    min_utility,
                    6,
                ),
                round(
                    max_utility,
                    6,
                ),
            ]
        )


# ============================================================
# VEÍCULOS
# ============================================================

def atualizar_veiculos():
    ids_ativos = list(
        traci.vehicle.getIDList()
    )

    if (
        len(ids_ativos)
        >
        MAX_VEICULOS
    ):
        ids_ativos = (
            RNG_ACTIVE_SAMPLE.sample(
                ids_ativos,
                MAX_VEICULOS,
            )
        )

    ativos = []

    for vid in ids_ativos:
        pos = (
            traci.vehicle
            .getPosition(
                vid
            )[0]
        )

        if vid not in veiculos:
            v = Veiculo(
                vid,

                custo_base=
                    RNG_VEHICLE.uniform(
                        CUSTO_BASE_MIN,
                        CUSTO_BASE_MAX,
                    ),

                capacidade=
                    RNG_VEHICLE.uniform(
                        1,
                        10,
                    ),

                pos=pos,

                energia=
                    RNG_VEHICLE.uniform(
                        60,
                        120,
                    ),

                lambda1=LAMBDA1,
                lambda2=LAMBDA2,
                lambda3=LAMBDA3,

                custo_max=
                    CUSTO_BASE_MAX,

                preco_max=
                    PRECO_MAX,

                energia_servico_max=
                    ENERGIA_SERVICO_MAX,

                distancia_max=
                    DISTANCIA_MAX,

                tempo_max=
                    TEMPO_MAX,
            )

            malicious_draw = (
                RNG_MALICIOUS.random()
            )

            malicious = (
                ATTACK_ENABLED
                and
                malicious_draw
                <
                MALICIOUS_RATIO
            )

            setattr(
                v,
                "malicious_draw",
                malicious_draw,
            )

            setattr(
                v,
                "malicioso",
                malicious,
            )

            veiculos[vid] = v

            registrar_veiculo(v)

        else:
            veiculos[
                vid
            ].posicao = pos

        ativos.append(
            veiculos[vid]
        )

    return ativos


# ============================================================
# FINALIZA SERVIÇOS
# ============================================================

def finalizar_servicos(step):
    global completed_services
    global successful_services
    global failed_services

    finalizados = [
        s
        for s in servicos_ativos
        if s["fim"] <= step
    ]

    for s in finalizados:
        v = s["veiculo"]

        # Always consume a reliability draw to preserve stream alignment.
        reliability_draw = (
            RNG_RELIABILITY.random()
        )

        if (
            ATTACK_TYPE == "failure"
            and
            is_malicious(v)
        ):
            sucesso = False

        else:
            sucesso = (
                reliability_draw
                >
                0.1
            )

        reputacao_before = (
            v.reputacao
        )

        v.atualizar_reputacao(
            sucesso
        )

        reputacao_after = (
            v.reputacao
        )

        completed_services += 1

        if sucesso:
            successful_services += 1
        else:
            failed_services += 1

        with open(
            OUTCOMES_FILE,
            "a",
            newline="",
            encoding="utf-8",
        ) as f:
            csv.writer(f).writerow(
                common_values()
                + [
                    s["inicio"],
                    step,
                    v.id,
                    is_malicious(v),
                    s["nome"],
                    round(
                        reliability_draw,
                        9,
                    ),
                    sucesso,
                    round(
                        reputacao_before,
                        6,
                    ),
                    round(
                        reputacao_after,
                        6,
                    ),
                    round(
                        reputacao_after
                        -
                        reputacao_before,
                        6,
                    ),
                ]
            )

        servicos_ativos.remove(
            s
        )


# ============================================================
# NOVA REQUISIÇÃO
# ============================================================

def nova_requisicao(
    step,
    veiculos_ativos,
):
    global total_requests
    global successful_allocations
    global unserved_requests
    global malicious_selections
    global honest_selections
    global total_artificial_candidates

    global total_base_price
    global total_offered_price
    global total_price_multiplier
    global total_demand_pressure
    global total_supply_scarcity
    global dynamic_price_increase_count

    servico = (
        RNG_REQUEST.choice(
            SERVICOS
        )
    )

    # C25: define price before utility / eligibility.
    pricing_info = (
        calcular_preco_dinamico(
            servico,
            len(
                veiculos_ativos
            ),
        )
    )

    plataforma.preco = (
        pricing_info[
            "offered_price"
        ]
    )

    # Preserve request-stream consumption between pricing modes.
    pos_tarefa = (
        RNG_REQUEST.uniform(
            0,
            DISTANCIA_MAX,
        )
    )

    total_requests += 1

    total_base_price += (
        pricing_info[
            "base_price"
        ]
    )

    total_offered_price += (
        pricing_info[
            "offered_price"
        ]
    )

    total_price_multiplier += (
        pricing_info[
            "price_multiplier"
        ]
    )

    total_demand_pressure += (
        pricing_info[
            "demand_pressure"
        ]
    )

    total_supply_scarcity += (
        pricing_info[
            "supply_scarcity"
        ]
    )

    if (
        pricing_info[
            "offered_price"
        ]
        >
        pricing_info[
            "base_price"
        ]
        +
        1e-12
    ):
        dynamic_price_increase_count += 1

    # ========================================================
    # SEM VEÍCULOS ATIVOS
    # ========================================================

    if len(
        veiculos_ativos
    ) == 0:
        unserved_requests += 1

        registrar_request(
            step,
            servico,
            0,
            0,
            0,
            0,
            pricing_info,
            False,
            "no_active_vehicles",
        )

        registrar_runtime(
            step,
            0,
            0,
            0,
            0,
            False,
            False,
            False,
            0.0,
            0.0,
        )

        return

    # ========================================================
    # ELIGIBILITY
    # ========================================================

    analisar_elegibilidade(
        step,
        servico,
        pos_tarefa,
        veiculos_ativos,
        pricing_info,
    )

    # ========================================================
    # OFFER
    # ========================================================

    offer_start = (
        time.perf_counter_ns()
    )

    candidatos_unicos = (
        plataforma.ofertar(
            veiculos_ativos,
            servico,
            pos_tarefa,
        )
    )

    offer_end = (
        time.perf_counter_ns()
    )

    offer_runtime_ms = (
        offer_end
        -
        offer_start
    ) / 1_000_000.0

    # ========================================================
    # SEM CANDIDATOS ELEGÍVEIS
    # ========================================================

    if not candidatos_unicos:
        unserved_requests += 1

        registrar_request(
            step,
            servico,
            len(
                veiculos_ativos
            ),
            0,
            0,
            0,
            pricing_info,
            False,
            "no_eligible_candidates",
        )

        registrar_runtime(
            step,
            len(
                veiculos_ativos
            ),
            0,
            0,
            0,
            False,
            True,
            False,
            offer_runtime_ms,
            0.0,
        )

        registrar_flood(
            step,
            [],
            [],
            0,
            None,
        )

        return

    # ========================================================
    # C18 FLOOD
    # ========================================================

    (
        candidatos_para_selecao,
        artificial_count,
    ) = aplicar_flood(
        candidatos_unicos
    )

    total_artificial_candidates += (
        artificial_count
    )

    # ========================================================
    # SELECTION
    # ========================================================

    selection_start = (
        time.perf_counter_ns()
    )

    escolhidos = (
        plataforma.selecionar(
            candidatos_para_selecao,
            pos_tarefa,
            step,
        )
    )

    selection_end = (
        time.perf_counter_ns()
    )

    selection_runtime_ms = (
        selection_end
        -
        selection_start
    ) / 1_000_000.0

    if not escolhidos:
        unserved_requests += 1

        registrar_request(
            step,
            servico,
            len(
                veiculos_ativos
            ),
            len(
                candidatos_unicos
            ),
            len(
                candidatos_para_selecao
            ),
            artificial_count,
            pricing_info,
            False,
            "selection_failed",
        )

        registrar_runtime(
            step,
            len(
                veiculos_ativos
            ),
            len(
                candidatos_unicos
            ),
            len(
                candidatos_para_selecao
            ),
            artificial_count,
            False,
            True,
            True,
            offer_runtime_ms,
            selection_runtime_ms,
        )

        registrar_flood(
            step,
            candidatos_unicos,
            candidatos_para_selecao,
            artificial_count,
            None,
        )

        return

    escolhido = (
        escolhidos[0]
    )

    v = escolhido[
        "veiculo"
    ]

    selected_malicious = (
        is_malicious(v)
    )

    if selected_malicious:
        malicious_selections += 1
    else:
        honest_selections += 1

    registrar_metricas(
        step,
        escolhido,
        servico,
        pricing_info,
        offer_runtime_ms,
        selection_runtime_ms,
        candidatos_unicos,
        candidatos_para_selecao,
        artificial_count,
    )

    registrar_runtime(
        step,
        len(
            veiculos_ativos
        ),
        len(
            candidatos_unicos
        ),
        len(
            candidatos_para_selecao
        ),
        artificial_count,
        True,
        True,
        True,
        offer_runtime_ms,
        selection_runtime_ms,
    )

    registrar_flood(
        step,
        candidatos_unicos,
        candidatos_para_selecao,
        artificial_count,
        escolhido,
    )

    successful_allocations += 1

    fim = (
        step
        +
        servico[
            "duracao"
        ]
    )

    registrar_request(
        step,
        servico,
        len(
            veiculos_ativos
        ),
        len(
            candidatos_unicos
        ),
        len(
            candidatos_para_selecao
        ),
        artificial_count,
        pricing_info,
        True,
        "",
        v.id,
        selected_malicious,
        fim,
    )

    servicos_ativos.append(
        {
            "veiculo":
                v,

            "inicio":
                step,

            "fim":
                fim,

            "nome":
                servico[
                    "nome"
                ],
        }
    )

    # One physical execution regardless of duplicate flood entries.
    v.executar_servico(
        servico
    )


# ============================================================
# SUMMARY C19 + C25
# ============================================================

def registrar_summary():
    encountered = len(
        veiculos
    )

    malicious_encountered = sum(
        1
        for v in veiculos.values()
        if is_malicious(v)
    )

    realized_malicious_ratio = (
        malicious_encountered
        /
        encountered
        if encountered > 0
        else 0.0
    )

    task_success_rate = (
        successful_services
        /
        completed_services
        if completed_services > 0
        else 0.0
    )

    task_failure_rate = (
        failed_services
        /
        completed_services
        if completed_services > 0
        else 0.0
    )

    allocation_rate = (
        successful_allocations
        /
        total_requests
        if total_requests > 0
        else 0.0
    )

    malicious_selection_share = (
        malicious_selections
        /
        successful_allocations
        if successful_allocations > 0
        else 0.0
    )

    avg_base_price = (
        total_base_price
        /
        total_requests
        if total_requests > 0
        else 0.0
    )

    avg_offered_price = (
        total_offered_price
        /
        total_requests
        if total_requests > 0
        else 0.0
    )

    avg_price_multiplier = (
        total_price_multiplier
        /
        total_requests
        if total_requests > 0
        else 0.0
    )

    avg_demand_pressure = (
        total_demand_pressure
        /
        total_requests
        if total_requests > 0
        else 0.0
    )

    avg_supply_scarcity = (
        total_supply_scarcity
        /
        total_requests
        if total_requests > 0
        else 0.0
    )

    dynamic_price_increase_rate = (
        dynamic_price_increase_count
        /
        total_requests
        if total_requests > 0
        else 0.0
    )

    with open(
        SUMMARY_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "experimento",
                "seed",
                "rng_scheme",
                "telemetry_scheme",
                "summary_scheme",
                "attack_scheme",
                "flood_scheme",
                "attack_type",
                "malicious_ratio",
                "realized_malicious_ratio",
                "flood_intensity",
                "request_interval",
                "cooldown_steps",
                "epsilon",
                "pricing_scheme",
                "pricing_mode",
                "pricing_alpha",
                "pricing_beta",
                "pricing_reference_interval",
                "avg_base_price",
                "avg_offered_price",
                "avg_price_multiplier",
                "avg_demand_pressure",
                "avg_supply_scarcity",
                "dynamic_price_increase_rate",
                "total_requests",
                "successful_allocations",
                "unserved_requests",
                "allocation_rate",
                "completed_tasks",
                "successful_tasks",
                "failed_tasks",
                "task_success_rate",
                "task_failure_rate",
                "pending_tasks",
                "malicious_selections",
                "honest_selections",
                "malicious_selection_share",
                "total_artificial_candidates",
            ]
        )

        writer.writerow(
            [
                EXPERIMENTO,
                SEED,
                RNG_SCHEME,
                TELEMETRY_SCHEME,
                SUMMARY_SCHEME,
                ATTACK_SCHEME,
                FLOOD_SCHEME,
                ATTACK_TYPE,
                MALICIOUS_RATIO,
                realized_malicious_ratio,
                FLOOD_INTENSITY,
                REQUEST_INTERVAL,
                COOLDOWN_STEPS,
                EPSILON,
                PRICING_SCHEME,
                PRICING_MODE,
                PRICING_ALPHA,
                PRICING_BETA,
                PRICING_REFERENCE_INTERVAL,
                avg_base_price,
                avg_offered_price,
                avg_price_multiplier,
                avg_demand_pressure,
                avg_supply_scarcity,
                dynamic_price_increase_rate,
                total_requests,
                successful_allocations,
                unserved_requests,
                allocation_rate,
                completed_services,
                successful_services,
                failed_services,
                task_success_rate,
                task_failure_rate,
                len(
                    servicos_ativos
                ),
                malicious_selections,
                honest_selections,
                malicious_selection_share,
                total_artificial_candidates,
            ]
        )

    return {
        "encountered":
            encountered,

        "malicious_encountered":
            malicious_encountered,

        "realized_ratio":
            realized_malicious_ratio,

        "task_success_rate":
            task_success_rate,

        "task_failure_rate":
            task_failure_rate,

        "allocation_rate":
            allocation_rate,

        "malicious_selection_share":
            malicious_selection_share,

        "avg_base_price":
            avg_base_price,

        "avg_offered_price":
            avg_offered_price,

        "avg_price_multiplier":
            avg_price_multiplier,

        "avg_demand_pressure":
            avg_demand_pressure,

        "avg_supply_scarcity":
            avg_supply_scarcity,

        "dynamic_price_increase_rate":
            dynamic_price_increase_rate,
    }


# ============================================================
# EXECUÇÃO
# ============================================================

def executar_shadai(sumocfg):
    tempo_final = (
        ler_tempo_final_sumocfg(
            sumocfg
        )
    )

    sumo_binary = (
        "sumo"
        if args.headless
        else "sumo-gui"
    )

    print()
    print("=" * 125)
    print("SHADAI - C25 DYNAMIC PRICING READY")
    print("=" * 125)
    print(f"Experiment                 : {EXPERIMENTO}")
    print(f"Seed                       : {SEED}")
    print(f"Attack type                : {ATTACK_TYPE}")
    print(f"Malicious ratio            : {MALICIOUS_RATIO * 100:.1f}%")
    print(f"Flood intensity            : {FLOOD_INTENSITY}")
    print(f"Pricing mode               : {PRICING_MODE}")
    print(f"Pricing alpha              : {PRICING_ALPHA}")
    print(f"Pricing beta               : {PRICING_BETA}")
    print(f"Pricing reference interval : {PRICING_REFERENCE_INTERVAL}")
    print(f"Request interval           : {REQUEST_INTERVAL}")
    print(f"Output directory           : {OUTPUT_DIR}")
    print("=" * 125)

    traci.start(
        [
            sumo_binary,
            "-c",
            sumocfg,
            "--seed",
            str(SEED),
        ]
    )

    for step in range(
        tempo_final
    ):
        traci.simulationStep()

        ativos = (
            atualizar_veiculos()
        )

        finalizar_servicos(
            step
        )

        if (
            step > 0
            and
            step
            %
            REQUEST_INTERVAL
            ==
            0
        ):
            nova_requisicao(
                step,
                ativos,
            )

    traci.close()

    summary = (
        registrar_summary()
    )

    print()
    print("=" * 125)
    print("FINAL SUMMARY")
    print("=" * 125)

    print(f"Total requests               : {total_requests}")
    print(f"Successful allocations       : {successful_allocations}")
    print(f"Unserved requests            : {unserved_requests}")
    print(
        f"Allocation rate              : "
        f"{summary['allocation_rate'] * 100:.2f}%"
    )

    print("-" * 125)

    print(f"Completed tasks              : {completed_services}")
    print(f"Successful tasks             : {successful_services}")
    print(f"Failed tasks                 : {failed_services}")
    print(
        f"Task success rate            : "
        f"{summary['task_success_rate'] * 100:.2f}%"
    )
    print(
        f"Task failure rate            : "
        f"{summary['task_failure_rate'] * 100:.2f}%"
    )
    print(f"Pending tasks                : {len(servicos_ativos)}")

    print("-" * 125)

    print(f"Pricing mode                 : {PRICING_MODE}")
    print(
        f"Average base price           : "
        f"{summary['avg_base_price']:.2f}"
    )
    print(
        f"Average offered price        : "
        f"{summary['avg_offered_price']:.2f}"
    )
    print(
        f"Average price multiplier     : "
        f"{summary['avg_price_multiplier']:.4f}"
    )
    print(
        f"Average demand pressure      : "
        f"{summary['avg_demand_pressure']:.4f}"
    )
    print(
        f"Average supply scarcity      : "
        f"{summary['avg_supply_scarcity']:.4f}"
    )
    print(
        f"Requests with price increase : "
        f"{summary['dynamic_price_increase_rate'] * 100:.2f}%"
    )

    print("-" * 125)

    print(
        f"Realized malicious ratio     : "
        f"{summary['realized_ratio'] * 100:.2f}%"
    )
    print(f"Malicious selections         : {malicious_selections}")
    print(
        f"Malicious selection share    : "
        f"{summary['malicious_selection_share'] * 100:.2f}%"
    )
    print(
        f"Artificial candidacies total : "
        f"{total_artificial_candidates}"
    )

    print("-" * 125)

    print(f"Summary CSV                  : {SUMMARY_FILE}")
    print("=" * 125)
    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    executar_shadai(
        "cidade.sumocfg"
    )
