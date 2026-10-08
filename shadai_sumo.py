# shadai_sumo.py

import traci
import xml.etree.ElementTree as ET
import random
import csv
import argparse
import os

from veiculo import Veiculo
from lider import Plataforma


# ============================================================
# ARGUMENTOS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "SHADAI SUMO simulation with normalized utility, "
        "parameterized score weights, epsilon-greedy, "
        "and separated random-number streams."
    )
)


# ============================================================
# LAMBDA
# ============================================================

parser.add_argument(
    "--lambda1",
    type=float,
    default=0.5
)

parser.add_argument(
    "--lambda2",
    type=float,
    default=0.3
)

parser.add_argument(
    "--lambda3",
    type=float,
    default=0.2
)


# ============================================================
# SCORE WEIGHTS
# ============================================================

parser.add_argument(
    "--w1",
    type=float,
    default=0.4
)

parser.add_argument(
    "--w2",
    type=float,
    default=0.3
)

parser.add_argument(
    "--w3",
    type=float,
    default=0.3
)


# ============================================================
# EPSILON
# ============================================================

parser.add_argument(
    "--epsilon",
    type=float,
    default=0.2
)


# ============================================================
# EXPERIMENTO
# ============================================================

parser.add_argument(
    "--experiment",
    type=str,
    default="normalized_original"
)

parser.add_argument(
    "--seed",
    type=int,
    default=42
)

parser.add_argument(
    "--headless",
    action="store_true"
)

parser.add_argument(
    "--output-dir",
    type=str,
    default="."
)


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

EXPERIMENTO = args.experiment
SEED = args.seed

OUTPUT_DIR = args.output_dir


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# ESQUEMA RNG
# ============================================================
#
# Não utilizamos mais random.seed(SEED) nem o gerador global.
#
# Cada fonte de aleatoriedade recebe um stream independente.
#
# ============================================================

RNG_SCHEME = "split_v1"


SEED_REQUEST = (
    SEED + 100_000
)

SEED_VEHICLE = (
    SEED + 200_000
)

SEED_RELIABILITY = (
    SEED + 300_000
)

SEED_POLICY_GATE = (
    SEED + 400_000
)

SEED_POLICY_CHOICE = (
    SEED + 500_000
)

SEED_ACTIVE_SAMPLE = (
    SEED + 600_000
)


RNG_REQUEST = random.Random(
    SEED_REQUEST
)

RNG_VEHICLE = random.Random(
    SEED_VEHICLE
)

RNG_RELIABILITY = random.Random(
    SEED_RELIABILITY
)

RNG_POLICY_GATE = random.Random(
    SEED_POLICY_GATE
)

RNG_POLICY_CHOICE = random.Random(
    SEED_POLICY_CHOICE
)

RNG_ACTIVE_SAMPLE = random.Random(
    SEED_ACTIVE_SAMPLE
)


# ============================================================
# VALIDAÇÕES
# ============================================================

SOMA_LAMBDA = (
    LAMBDA1
    + LAMBDA2
    + LAMBDA3
)

if abs(
    SOMA_LAMBDA - 1.0
) > 1e-6:

    raise ValueError(
        "lambda1 + lambda2 + lambda3 must sum to 1.0."
    )


if any(
    valor < 0
    for valor in [
        LAMBDA1,
        LAMBDA2,
        LAMBDA3
    ]
):

    raise ValueError(
        "Lambda weights cannot be negative."
    )


SOMA_W = (
    W1
    + W2
    + W3
)

if abs(
    SOMA_W - 1.0
) > 1e-6:

    raise ValueError(
        "w1 + w2 + w3 must sum to 1.0."
    )


if any(
    valor < 0
    for valor in [
        W1,
        W2,
        W3
    ]
):

    raise ValueError(
        "Score weights cannot be negative."
    )


if not (
    0.0 <= EPSILON <= 1.0
):

    raise ValueError(
        "epsilon must be between 0.0 and 1.0."
    )


# ============================================================
# CENÁRIO
# ============================================================

MODO = "shadai"

MAX_VEICULOS = 400

INTERVALO_REQUISICAO = 20

COOLDOWN_STEPS = 40


# ============================================================
# ATAQUES
# ============================================================

ATAQUE = False

PERCENTUAL_MALICIOSOS = 0.0


# ============================================================
# CUSTOS
# ============================================================

CUSTO_BASE_MIN = 3.0

CUSTO_BASE_MAX = 8.0


# ============================================================
# SERVIÇOS
# ============================================================

SERVICOS = [

    {
        "nome":
            "Processamento de Imagem Urbana",

        "energia":
            18,

        "duracao":
            60,

        "preco":
            45.0
    },

    {
        "nome":
            "Agregação de Dados de Sensores",

        "energia":
            8,

        "duracao":
            30,

        "preco":
            22.0
    },

    {
        "nome":
            "Reencaminhamento de Dados (Edge Relay)",

        "energia":
            5,

        "duracao":
            25,

        "preco":
            15.0
    },

    {
        "nome":
            "Atualização de Modelos Locais (ML)",

        "energia":
            22,

        "duracao":
            80,

        "preco":
            60.0
    },

    {
        "nome":
            "Diagnóstico Veicular",

        "energia":
            10,

        "duracao":
            40,

        "preco":
            28.0
    }
]


# ============================================================
# NORMALIZAÇÃO
# ============================================================

PRECO_MAX = max(
    servico["preco"]
    for servico in SERVICOS
)

ENERGIA_SERVICO_MAX = 25.0

DISTANCIA_MAX = 1000.0

TEMPO_MAX = 100.0


# ============================================================
# ESTRUTURAS
# ============================================================

veiculos = {}

servicos_ativos = []


plataforma = Plataforma(

    epsilon=EPSILON,

    w1=W1,

    w2=W2,

    w3=W3,

    cooldown_steps=COOLDOWN_STEPS,

    rng_policy_gate=
        RNG_POLICY_GATE,

    rng_policy_choice=
        RNG_POLICY_CHOICE
)


# ============================================================
# ARQUIVOS
# ============================================================

METRICS_FILE = os.path.join(

    OUTPUT_DIR,

    (
        f"metricas_"
        f"{MODO}_"
        f"{MAX_VEICULOS}_"
        f"{EXPERIMENTO}.csv"
    )
)


ELIGIBILITY_FILE = os.path.join(

    OUTPUT_DIR,

    (
        f"eligibilidade_"
        f"{MODO}_"
        f"{MAX_VEICULOS}_"
        f"{EXPERIMENTO}.csv"
    )
)


# ============================================================
# CSV MÉTRICAS
# ============================================================

with open(
    METRICS_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow([

        "experimento",

        "seed",

        "rng_scheme",

        "lambda1",
        "lambda2",
        "lambda3",

        "w1",
        "w2",
        "w3",

        "epsilon",

        "modo",

        "step",

        "veiculo",

        "servico",

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

        "utilidade",

        "reputacao",

        "energia"
    ])


# ============================================================
# CSV ELEGIBILIDADE
# ============================================================

with open(
    ELIGIBILITY_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow([

        "experimento",

        "seed",

        "rng_scheme",

        "lambda1",
        "lambda2",
        "lambda3",

        "w1",
        "w2",
        "w3",

        "epsilon",

        "step",

        "servico",

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

        "max_utility"
    ])


# ============================================================
# CONTADORES
# ============================================================

total_requests = 0

soma_taxa_elegibilidade = 0.0

total_avaliacoes = 0

total_rejeitados_utility = 0

total_rejeitados_energy = 0

total_elegiveis = 0

successful_allocations = 0


# ============================================================
# TEMPO FINAL
# ============================================================

def ler_tempo_final_sumocfg(
    sumocfg_path
):

    tree = ET.parse(
        sumocfg_path
    )

    root = tree.getroot()

    end = root.find(
        "./time/end"
    )

    if end is not None:

        return int(
            end.attrib["value"]
        )

    return 3600


# ============================================================
# REGISTRA MÉTRICAS
# ============================================================

def registrar_metricas(
    step,
    escolhido,
    servico
):

    veiculo = escolhido[
        "veiculo"
    ]

    with open(
        METRICS_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow([

            EXPERIMENTO,

            SEED,

            RNG_SCHEME,

            LAMBDA1,
            LAMBDA2,
            LAMBDA3,

            W1,
            W2,
            W3,

            EPSILON,

            MODO,

            step,

            veiculo.id,

            servico["nome"],

            round(
                escolhido[
                    "score_base"
                ],
                6
            ),

            round(
                escolhido[
                    "score"
                ],
                6
            ),

            round(
                escolhido[
                    "reputacao_norm"
                ],
                6
            ),

            round(
                escolhido[
                    "energia_norm"
                ],
                6
            ),

            round(
                escolhido[
                    "proximidade_norm"
                ],
                6
            ),

            round(
                escolhido[
                    "distancia"
                ],
                6
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

            round(
                escolhido[
                    "utilidade"
                ],
                6
            ),

            round(
                veiculo.reputacao,
                6
            ),

            round(
                veiculo.energia,
                6
            )
        ])


# ============================================================
# VEÍCULOS ATIVOS
# ============================================================

def atualizar_veiculos():

    ids_ativos = list(
        traci.vehicle.getIDList()
    )


    if len(
        ids_ativos
    ) > MAX_VEICULOS:

        ids_ativos = (
            RNG_ACTIVE_SAMPLE.sample(
                ids_ativos,
                MAX_VEICULOS
            )
        )


    veiculos_ativos = []


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
                        CUSTO_BASE_MAX
                    ),

                capacidade=
                    RNG_VEHICLE.uniform(
                        1,
                        10
                    ),

                pos=pos,

                energia=
                    RNG_VEHICLE.uniform(
                        60,
                        120
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
                    TEMPO_MAX
            )


            veiculos[
                vid
            ] = v

        else:

            veiculos[
                vid
            ].posicao = pos


        veiculos_ativos.append(
            veiculos[
                vid
            ]
        )


    return veiculos_ativos


# ============================================================
# FINALIZA SERVIÇOS
# ============================================================

def finalizar_servicos(
    step
):

    finalizados = [

        servico

        for servico in servicos_ativos

        if servico[
            "fim"
        ] <= step
    ]


    for servico in finalizados:

        v = servico[
            "veiculo"
        ]


        sucesso = (
            RNG_RELIABILITY.random()
            > 0.1
        )


        v.atualizar_reputacao(
            sucesso
        )


        servicos_ativos.remove(
            servico
        )


# ============================================================
# ELEGIBILIDADE
# ============================================================

def analisar_elegibilidade(
    step,
    servico,
    pos_tarefa,
    veiculos_ativos
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
            _
        ) = v.calcular_utilidade(

            plataforma.preco,
            servico,
            pos_tarefa
        )


        utilities.append(
            utilidade
        )


        utility_ok = (
            utilidade > 0
        )


        energy_ok = (
            v.energia
            >= servico[
                "energia"
            ]
        )


        if utility_ok:

            positive_utility += 1

        else:

            rejected_utility += 1


        if energy_ok:

            sufficient_energy += 1

        else:

            rejected_energy += 1


        if (
            utility_ok
            and
            energy_ok
        ):

            eligible += 1

        elif (
            not utility_ok
            and
            energy_ok
        ):

            rejected_utility_only += 1

        elif (
            utility_ok
            and
            not energy_ok
        ):

            rejected_energy_only += 1

        else:

            rejected_both += 1


    total = len(
        veiculos_ativos
    )


    eligibility_rate = (

        eligible
        / total

        if total > 0

        else 0.0
    )


    avg_utility = (

        sum(
            utilities
        )
        / len(
            utilities
        )

        if utilities

        else 0.0
    )


    min_utility = (

        min(
            utilities
        )

        if utilities

        else 0.0
    )


    max_utility = (

        max(
            utilities
        )

        if utilities

        else 0.0
    )


    with open(
        ELIGIBILITY_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow([

            EXPERIMENTO,

            SEED,

            RNG_SCHEME,

            LAMBDA1,
            LAMBDA2,
            LAMBDA3,

            W1,
            W2,
            W3,

            EPSILON,

            step,

            servico["nome"],

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
                6
            ),

            round(
                avg_utility,
                6
            ),

            round(
                min_utility,
                6
            ),

            round(
                max_utility,
                6
            )
        ])


    return {

        "total":
            total,

        "eligible":
            eligible,

        "rejected_utility":
            rejected_utility,

        "rejected_energy":
            rejected_energy,

        "eligibility_rate":
            eligibility_rate
    }


# ============================================================
# NOVA REQUISIÇÃO
# ============================================================

def nova_requisicao(
    step,
    veiculos_ativos
):

    global total_requests
    global soma_taxa_elegibilidade
    global total_avaliacoes
    global total_rejeitados_utility
    global total_rejeitados_energy
    global total_elegiveis
    global successful_allocations


    # ========================================================
    # SOMENTE O RNG DO AMBIENTE/REQUEST É USADO AQUI
    # ========================================================

    servico = (
        RNG_REQUEST.choice(
            SERVICOS
        )
    )


    plataforma.preco = (
        servico[
            "preco"
        ]
    )


    pos_tarefa = (
        RNG_REQUEST.uniform(
            0,
            DISTANCIA_MAX
        )
    )


    diagnostico = (
        analisar_elegibilidade(

            step,
            servico,
            pos_tarefa,
            veiculos_ativos
        )
    )


    total_requests += 1


    total_avaliacoes += (
        diagnostico[
            "total"
        ]
    )


    soma_taxa_elegibilidade += (
        diagnostico[
            "eligibility_rate"
        ]
    )


    total_rejeitados_utility += (
        diagnostico[
            "rejected_utility"
        ]
    )


    total_rejeitados_energy += (
        diagnostico[
            "rejected_energy"
        ]
    )


    total_elegiveis += (
        diagnostico[
            "eligible"
        ]
    )


    candidatos = (
        plataforma.ofertar(

            veiculos_ativos,
            servico,
            pos_tarefa
        )
    )


    if not candidatos:

        return


    escolhidos = (
        plataforma.selecionar(

            candidatos,
            pos_tarefa,
            step
        )
    )


    if not escolhidos:

        return


    escolhido = (
        escolhidos[
            0
        ]
    )


    v = escolhido[
        "veiculo"
    ]


    registrar_metricas(

        step,
        escolhido,
        servico
    )


    successful_allocations += 1


    fim = (
        step
        + servico[
            "duracao"
        ]
    )


    servicos_ativos.append({

        "veiculo":
            v,

        "fim":
            fim,

        "nome":
            servico[
                "nome"
            ]
    })


    v.executar_servico(
        servico
    )


# ============================================================
# EXECUÇÃO
# ============================================================

def executar_shadai(
    sumocfg
):

    tempo_final = (
        ler_tempo_final_sumocfg(
            sumocfg
        )
    )


    if args.headless:

        sumo_binary = "sumo"

        execution_mode = "HEADLESS"

    else:

        sumo_binary = "sumo-gui"

        execution_mode = "GUI"


    print("\n")
    print("=" * 86)

    print(
        "SHADAI - SEPARATED RANDOM STREAMS"
    )

    print("=" * 86)


    print(
        f"Experiment       : {EXPERIMENTO}"
    )

    print(
        f"Seed             : {SEED}"
    )

    print(
        f"RNG scheme       : {RNG_SCHEME}"
    )

    print(
        f"Execution mode   : {execution_mode}"
    )

    print(
        f"Lambda           : "
        f"({LAMBDA1}, {LAMBDA2}, {LAMBDA3})"
    )

    print(
        f"Score weights    : "
        f"({W1}, {W2}, {W3})"
    )

    print(
        f"Epsilon          : {EPSILON}"
    )

    print(
        f"Cooldown steps   : {COOLDOWN_STEPS}"
    )

    print(
        f"Output directory : {OUTPUT_DIR}"
    )


    print("-" * 86)

    print(
        f"Request RNG      : {SEED_REQUEST}"
    )

    print(
        f"Vehicle RNG      : {SEED_VEHICLE}"
    )

    print(
        f"Reliability RNG  : {SEED_RELIABILITY}"
    )

    print(
        f"Policy gate RNG  : {SEED_POLICY_GATE}"
    )

    print(
        f"Policy choice RNG: {SEED_POLICY_CHOICE}"
    )

    print(
        f"Sampling RNG     : {SEED_ACTIVE_SAMPLE}"
    )

    print("=" * 86)


    traci.start([

        sumo_binary,

        "-c",
        sumocfg,

        "--seed",
        str(
            SEED
        )
    ])


    for step in range(
        tempo_final
    ):

        traci.simulationStep()


        veiculos_ativos = (
            atualizar_veiculos()
        )


        finalizar_servicos(
            step
        )


        if (
            step > 0
            and
            step
            % INTERVALO_REQUISICAO
            == 0
            and
            len(
                veiculos_ativos
            ) > 0
        ):

            nova_requisicao(

                step,
                veiculos_ativos
            )


    traci.close()


    # ========================================================
    # SUMÁRIO
    # ========================================================

    print("\n")
    print("=" * 86)

    print(
        "SIMULATION SUMMARY"
    )

    print("=" * 86)


    print(
        f"Total requests: "
        f"{total_requests}"
    )


    print(
        f"Total vehicle evaluations: "
        f"{total_avaliacoes}"
    )


    if total_requests > 0:

        mean_eligibility = (
            soma_taxa_elegibilidade
            / total_requests
        )


        allocation_rate = (
            successful_allocations
            / total_requests
        )


        print(
            f"Mean eligibility rate: "
            f"{mean_eligibility:.4f} "
            f"({mean_eligibility * 100:.2f}%)"
        )


        print(
            f"Request allocation rate: "
            f"{allocation_rate:.4f} "
            f"({allocation_rate * 100:.2f}%)"
        )


    if total_avaliacoes > 0:

        utility_rejection_rate = (
            total_rejeitados_utility
            / total_avaliacoes
        )


        energy_rejection_rate = (
            total_rejeitados_energy
            / total_avaliacoes
        )


        overall_eligibility_rate = (
            total_elegiveis
            / total_avaliacoes
        )


        print(
            f"Overall utility rejection rate: "
            f"{utility_rejection_rate:.4f} "
            f"({utility_rejection_rate * 100:.2f}%)"
        )


        print(
            f"Overall energy rejection rate: "
            f"{energy_rejection_rate:.4f} "
            f"({energy_rejection_rate * 100:.2f}%)"
        )


        print(
            f"Overall eligibility rate: "
            f"{overall_eligibility_rate:.4f} "
            f"({overall_eligibility_rate * 100:.2f}%)"
        )


    print(
        f"Successful allocations: "
        f"{successful_allocations}"
    )


    print("=" * 86)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    executar_shadai(
        "cidade.sumocfg"
    )