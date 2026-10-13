# ablation_load.py

import subprocess
import sys
import os
import time
import csv


# ============================================================
# DIRETÓRIO
# ============================================================

OUTPUT_DIR = "load_ablation"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# C13 - NÍVEIS DE CARGA
#
# Menor intervalo = maior taxa de requisições.
#
# baseline original = 20 steps
# ============================================================

CONFIGURACOES = {

    "load_low": {
        "label": "Low",
        "request_interval": 60
    },

    "load_moderate": {
        "label": "Moderate",
        "request_interval": 40
    },

    "load_high": {
        "label": "High",
        "request_interval": 20
    },

    "load_very_high": {
        "label": "Very High",
        "request_interval": 10
    }
}


BASELINE_CONFIG = "load_high"

BASELINE_INTERVAL = 20


# ============================================================
# SEEDS
# ============================================================

SEEDS = list(
    range(
        1,
        31
    )
)


# ============================================================
# PARÂMETROS FIXOS
# ============================================================

LAMBDA1 = 0.5
LAMBDA2 = 0.3
LAMBDA3 = 0.2

W1 = 0.4
W2 = 0.3
W3 = 0.3

EPSILON = 0.2

COOLDOWN_STEPS = 40


# ============================================================
# INFRAESTRUTURA EXPERIMENTAL
# ============================================================

RNG_SCHEME_REQUIRED = "split_v1"

TELEMETRY_SCHEME_REQUIRED = "c9_regret_v1"

SHADAI_SCRIPT = "shadai_sumo.py"


# ============================================================
# CONTROLE
# ============================================================

FORCAR_REEXECUCAO = False


# ============================================================
# VALIDA OUTPUT
# ============================================================

def arquivo_compativel(
    caminho,
    request_interval_esperado
):

    if not os.path.exists(
        caminho
    ):

        return False


    try:

        with open(
            caminho,
            "r",
            newline="",
            encoding="utf-8"
        ) as f:

            reader = csv.DictReader(
                f
            )


            if reader.fieldnames is None:

                return False


            obrigatorias = [

                "rng_scheme",

                "telemetry_scheme",

                "cooldown_steps",

                "request_interval"
            ]


            for coluna in obrigatorias:

                if coluna not in reader.fieldnames:

                    return False


            primeira = next(
                reader,
                None
            )


            if primeira is None:

                return False


            if (
                primeira[
                    "rng_scheme"
                ]
                !=
                RNG_SCHEME_REQUIRED
            ):

                return False


            if (
                primeira[
                    "telemetry_scheme"
                ]
                !=
                TELEMETRY_SCHEME_REQUIRED
            ):

                return False


            intervalo_csv = int(
                float(
                    primeira[
                        "request_interval"
                    ]
                )
            )


            if (
                intervalo_csv
                !=
                request_interval_esperado
            ):

                return False


            cooldown_csv = int(
                float(
                    primeira[
                        "cooldown_steps"
                    ]
                )
            )


            if (
                cooldown_csv
                !=
                COOLDOWN_STEPS
            ):

                return False


            return True


    except Exception:

        return False


# ============================================================
# VERIFICA SCRIPT PRINCIPAL
# ============================================================

if not os.path.exists(
    SHADAI_SCRIPT
):

    raise FileNotFoundError(
        f"Arquivo não encontrado: "
        f"{SHADAI_SCRIPT}"
    )


# ============================================================
# TOTAL
# ============================================================

TOTAL = (
    len(CONFIGURACOES)
    *
    len(SEEDS)
)


# ============================================================
# CABEÇALHO
# ============================================================

print("\n")

print("=" * 125)

print(
    "SHADAI - MULTI-SEED WORKLOAD / REQUEST-LOAD ANALYSIS"
)

print("=" * 125)


print(
    f"Output directory  : {OUTPUT_DIR}"
)

print(
    f"RNG scheme        : {RNG_SCHEME_REQUIRED}"
)

print(
    f"Telemetry         : {TELEMETRY_SCHEME_REQUIRED}"
)

print(
    f"Configurations    : {len(CONFIGURACOES)}"
)

print(
    f"Seeds             : {len(SEEDS)}"
)

print(
    f"Total runs        : {TOTAL}"
)

print(
    f"Baseline interval : {BASELINE_INTERVAL}"
)

print(
    f"Cooldown fixed    : {COOLDOWN_STEPS}"
)

print(
    f"Epsilon fixed     : {EPSILON}"
)

print("=" * 125)


# ============================================================
# CONTADORES
# ============================================================

executadas = 0

puladas = 0

falhas = []

contador = 0

inicio_geral = time.time()


# ============================================================
# EXECUÇÕES
# ============================================================

for nome, config in CONFIGURACOES.items():

    label = config[
        "label"
    ]

    request_interval = config[
        "request_interval"
    ]


    for seed in SEEDS:

        contador += 1


        experiment_id = (

            f"{nome}_"
            f"interval_"
            f"{request_interval:03d}_"
            f"seed_"
            f"{seed:02d}"
        )


        metrics_file = os.path.join(

            OUTPUT_DIR,

            (
                f"metricas_shadai_400_"
                f"{experiment_id}.csv"
            )
        )


        eligibility_file = os.path.join(

            OUTPUT_DIR,

            (
                f"eligibilidade_shadai_400_"
                f"{experiment_id}.csv"
            )
        )


        metrics_validas = (
            arquivo_compativel(

                metrics_file,

                request_interval
            )
        )


        eligibility_valida = (
            arquivo_compativel(

                eligibility_file,

                request_interval
            )
        )


        # ====================================================
        # SKIP
        # ====================================================

        if (
            not FORCAR_REEXECUCAO
            and
            metrics_validas
            and
            eligibility_valida
        ):

            print(

                f"[{contador:03d}/{TOTAL:03d}] "

                f"SKIP | "

                f"{label:<10} | "

                f"interval="
                f"{request_interval:>3} | "

                f"seed="
                f"{seed:02d}"
            )


            puladas += 1

            continue


        # ====================================================
        # AVISO DE OUTPUT ANTIGO
        # ====================================================

        if (
            os.path.exists(
                metrics_file
            )

            or

            os.path.exists(
                eligibility_file
            )
        ):

            print(

                f"[INFO] Existing output is stale "
                f"or incompatible: "

                f"{nome}, "

                f"seed={seed:02d}. "

                f"Re-running."
            )


        # ====================================================
        # COMANDO
        # ====================================================

        comando = [

            sys.executable,

            SHADAI_SCRIPT,

            "--lambda1",
            str(
                LAMBDA1
            ),

            "--lambda2",
            str(
                LAMBDA2
            ),

            "--lambda3",
            str(
                LAMBDA3
            ),

            "--w1",
            str(
                W1
            ),

            "--w2",
            str(
                W2
            ),

            "--w3",
            str(
                W3
            ),

            "--epsilon",
            str(
                EPSILON
            ),

            "--cooldown-steps",
            str(
                COOLDOWN_STEPS
            ),

            "--request-interval",
            str(
                request_interval
            ),

            "--experiment",
            experiment_id,

            "--seed",
            str(
                seed
            ),

            "--headless",

            "--output-dir",
            OUTPUT_DIR
        ]


        print(

            f"[{contador:03d}/{TOTAL:03d}] "

            f"RUN  | "

            f"{label:<10} | "

            f"interval="
            f"{request_interval:>3} | "

            f"seed="
            f"{seed:02d}"
        )


        inicio = time.time()


        resultado = subprocess.run(
            comando
        )


        duracao = (
            time.time()
            - inicio
        )


        # ====================================================
        # PROCESS FAILURE
        # ====================================================

        if resultado.returncode != 0:

            falhas.append({

                "configuration":
                    nome,

                "label":
                    label,

                "request_interval":
                    request_interval,

                "seed":
                    seed,

                "reason":
                    resultado.returncode
            })


            print(

                f"[FAIL] "

                f"{label} | "

                f"interval="
                f"{request_interval} | "

                f"seed="
                f"{seed:02d}"
            )


            continue


        # ====================================================
        # OUTPUT VALIDATION
        # ====================================================

        if (
            not arquivo_compativel(

                metrics_file,

                request_interval
            )

            or

            not arquivo_compativel(

                eligibility_file,

                request_interval
            )
        ):

            falhas.append({

                "configuration":
                    nome,

                "label":
                    label,

                "request_interval":
                    request_interval,

                "seed":
                    seed,

                "reason":
                    "invalid_output"
            })


            print(

                f"[FAIL] Invalid output | "

                f"{label} | "

                f"interval="
                f"{request_interval} | "

                f"seed="
                f"{seed:02d}"
            )


            continue


        executadas += 1


        print(

            f"[OK]   "

            f"{label:<10} | "

            f"interval="
            f"{request_interval:>3} | "

            f"seed="
            f"{seed:02d} | "

            f"{duracao:.2f}s"
        )


# ============================================================
# CHECAGEM FINAL
# ============================================================

completos = 0


for nome, config in CONFIGURACOES.items():

    request_interval = config[
        "request_interval"
    ]


    for seed in SEEDS:

        experiment_id = (

            f"{nome}_"
            f"interval_"
            f"{request_interval:03d}_"
            f"seed_"
            f"{seed:02d}"
        )


        metrics_file = os.path.join(

            OUTPUT_DIR,

            (
                f"metricas_shadai_400_"
                f"{experiment_id}.csv"
            )
        )


        eligibility_file = os.path.join(

            OUTPUT_DIR,

            (
                f"eligibilidade_shadai_400_"
                f"{experiment_id}.csv"
            )
        )


        if (
            arquivo_compativel(

                metrics_file,

                request_interval
            )

            and

            arquivo_compativel(

                eligibility_file,

                request_interval
            )
        ):

            completos += 1


# ============================================================
# SUMÁRIO
# ============================================================

tempo_total = (
    time.time()
    - inicio_geral
)


print("\n")

print("=" * 125)

print(
    "WORKLOAD ABLATION SUMMARY"
)

print("=" * 125)


print(
    f"Expected            : {TOTAL}"
)

print(
    f"Executed            : {executadas}"
)

print(
    f"Skipped             : {puladas}"
)

print(
    f"Failures            : {len(falhas)}"
)

print(
    f"Complete compatible : "
    f"{completos}/{TOTAL}"
)

print(
    f"RNG scheme          : "
    f"{RNG_SCHEME_REQUIRED}"
)

print(
    f"Telemetry scheme    : "
    f"{TELEMETRY_SCHEME_REQUIRED}"
)

print(
    f"Total time          : "
    f"{tempo_total / 60:.2f} min"
)

print(
    f"Output directory    : "
    f"{OUTPUT_DIR}"
)


if falhas:

    print("\n")

    print(
        "FAILED RUNS"
    )

    print("-" * 125)


    for falha in falhas:

        print(

            f"{falha['configuration']} | "

            f"interval="
            f"{falha['request_interval']} | "

            f"seed="
            f"{falha['seed']:02d} | "

            f"reason="
            f"{falha['reason']}"
        )


if completos == TOTAL:

    print(
        "\nALL WORKLOAD EXPERIMENTS COMPLETE"
    )

else:

    print(
        "\nWORKLOAD EXPERIMENT SET INCOMPLETE"
    )


print("=" * 125)

print()