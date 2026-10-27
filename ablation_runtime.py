# ablation_runtime.py

import subprocess
import sys
import os
import time
import csv


# ============================================================
# DIRETÓRIO
# ============================================================

OUTPUT_DIR = "runtime_ablation"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# WORKLOADS
# ============================================================

CONFIGURACOES = {

    "runtime_low": {
        "label": "Low",
        "request_interval": 60
    },

    "runtime_moderate": {
        "label": "Moderate",
        "request_interval": 40
    },

    "runtime_high": {
        "label": "High",
        "request_interval": 20
    },

    "runtime_very_high": {
        "label": "Very High",
        "request_interval": 10
    }
}


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
# VERSÕES
# ============================================================

RNG_SCHEME_REQUIRED = "split_v1"

TELEMETRY_SCHEME_REQUIRED = "c15_runtime_v1"

SHADAI_SCRIPT = "shadai_sumo.py"

FORCAR_REEXECUCAO = False


# ============================================================
# VALIDA RUNTIME CSV
# ============================================================

def runtime_compativel(
    caminho,
    intervalo_esperado
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

                "request_interval",

                "cooldown_steps",

                "active_vehicles",

                "candidate_count",

                "offer_executed",

                "selection_executed",

                "offer_runtime_ms",

                "selection_runtime_ms",

                "decision_runtime_ms"
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
                intervalo_esperado
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
# SCRIPT CHECK
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
# HEADER
# ============================================================

print("\n")

print("=" * 130)

print(
    "SHADAI - MULTI-SEED COMPUTATIONAL RUNTIME EXPERIMENT"
)

print("=" * 130)


print(
    f"Output directory : {OUTPUT_DIR}"
)

print(
    f"RNG scheme       : {RNG_SCHEME_REQUIRED}"
)

print(
    f"Telemetry        : {TELEMETRY_SCHEME_REQUIRED}"
)

print(
    f"Workloads        : {len(CONFIGURACOES)}"
)

print(
    f"Seeds            : {len(SEEDS)}"
)

print(
    f"Total runs       : {TOTAL}"
)

print(
    f"Cooldown         : {COOLDOWN_STEPS}"
)

print(
    f"Epsilon          : {EPSILON}"
)

print("=" * 130)


# ============================================================
# EXECUÇÕES
# ============================================================

executadas = 0

puladas = 0

falhas = []

contador = 0

inicio_geral = time.time()


for nome, config in CONFIGURACOES.items():

    label = config[
        "label"
    ]

    interval = config[
        "request_interval"
    ]


    for seed in SEEDS:

        contador += 1


        experiment_id = (

            f"{nome}_"
            f"interval_"
            f"{interval:03d}_"
            f"seed_"
            f"{seed:02d}"
        )


        runtime_file = os.path.join(

            OUTPUT_DIR,

            (
                f"runtime_shadai_400_"
                f"{experiment_id}.csv"
            )
        )


        if (
            not FORCAR_REEXECUCAO

            and

            runtime_compativel(
                runtime_file,
                interval
            )
        ):

            print(

                f"[{contador:03d}/{TOTAL:03d}] "

                f"SKIP | "

                f"{label:<10} | "

                f"interval="
                f"{interval:>3} | "

                f"seed="
                f"{seed:02d}"
            )


            puladas += 1

            continue


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
                interval
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
            f"{interval:>3} | "

            f"seed="
            f"{seed:02d}"
        )


        inicio = time.time()


        resultado = subprocess.run(
            comando
        )


        duracao = (
            time.time()
            -
            inicio
        )


        if resultado.returncode != 0:

            falhas.append({

                "configuration":
                    nome,

                "interval":
                    interval,

                "seed":
                    seed,

                "reason":
                    resultado.returncode
            })


            print(

                f"[FAIL] "

                f"{nome} | "

                f"seed="
                f"{seed:02d}"
            )


            continue


        if not runtime_compativel(
            runtime_file,
            interval
        ):

            falhas.append({

                "configuration":
                    nome,

                "interval":
                    interval,

                "seed":
                    seed,

                "reason":
                    "invalid_runtime_output"
            })


            print(

                f"[FAIL] Invalid runtime output | "

                f"{nome} | "

                f"seed="
                f"{seed:02d}"
            )


            continue


        executadas += 1


        print(

            f"[OK]   "

            f"{label:<10} | "

            f"interval="
            f"{interval:>3} | "

            f"seed="
            f"{seed:02d} | "

            f"{duracao:.2f}s"
        )


# ============================================================
# FINAL COMPLETENESS
# ============================================================

completos = 0


for nome, config in CONFIGURACOES.items():

    interval = config[
        "request_interval"
    ]


    for seed in SEEDS:

        experiment_id = (

            f"{nome}_"
            f"interval_"
            f"{interval:03d}_"
            f"seed_"
            f"{seed:02d}"
        )


        arquivo = os.path.join(

            OUTPUT_DIR,

            (
                f"runtime_shadai_400_"
                f"{experiment_id}.csv"
            )
        )


        if runtime_compativel(
            arquivo,
            interval
        ):

            completos += 1


# ============================================================
# SUMMARY
# ============================================================

tempo_total = (
    time.time()
    -
    inicio_geral
)


print("\n")

print("=" * 130)

print(
    "C15 RUNTIME EXPERIMENT SUMMARY"
)

print("=" * 130)


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
    f"Total experiment time: "
    f"{tempo_total / 60:.2f} min"
)

print(
    f"Output directory      : "
    f"{OUTPUT_DIR}"
)


if falhas:

    print("\nFAILED RUNS")

    print("-" * 130)


    for falha in falhas:

        print(

            f"{falha['configuration']} | "

            f"interval="
            f"{falha['interval']} | "

            f"seed="
            f"{falha['seed']:02d} | "

            f"reason="
            f"{falha['reason']}"
        )


if completos == TOTAL:

    print(
        "\nALL C15 RUNTIME EXPERIMENTS COMPLETE"
    )

else:

    print(
        "\nC15 RUNTIME EXPERIMENT SET INCOMPLETE"
    )


print("=" * 130)

print()