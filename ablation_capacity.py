# ablation_capacity.py

import subprocess
import sys
import os
import time
import csv


# ============================================================
# DIRETÓRIO
# ============================================================

OUTPUT_DIR = "capacity_ablation"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# C14 - WORKLOAD LEVELS
# ============================================================

CONFIGURACOES = {

    "capacity_low": {
        "label": "Low",
        "request_interval": 60
    },

    "capacity_moderate": {
        "label": "Moderate",
        "request_interval": 40
    },

    "capacity_high": {
        "label": "High",
        "request_interval": 20
    },

    "capacity_very_high": {
        "label": "Very High",
        "request_interval": 10
    }
}


BASELINE = "capacity_high"


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
# INFRAESTRUTURA
# ============================================================

RNG_SCHEME_REQUIRED = "split_v1"

TELEMETRY_SCHEME_REQUIRED = "c14_capacity_v1"

SHADAI_SCRIPT = "shadai_sumo.py"

FORCAR_REEXECUCAO = False


# ============================================================
# VALIDA CSV
# ============================================================

def arquivo_compativel(
    caminho,
    request_interval
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

                "cooldown_steps"
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
                request_interval
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
# VALIDA REQUEST CSV
# ============================================================

def requests_compativel(
    caminho,
    request_interval
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

                "allocated",

                "failure_reason",

                "active_vehicles",

                "eligible_candidates",

                "completion_step"
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
                request_interval
            ):
                return False

            return True

    except Exception:

        return False


# ============================================================
# CHECK
# ============================================================

if not os.path.exists(
    SHADAI_SCRIPT
):

    raise FileNotFoundError(
        f"Arquivo não encontrado: {SHADAI_SCRIPT}"
    )


TOTAL = (
    len(CONFIGURACOES)
    *
    len(SEEDS)
)


# ============================================================
# CABEÇALHO
# ============================================================

print("\n")

print("=" * 130)

print(
    "SHADAI - MULTI-SEED CAPACITY / THROUGHPUT EXPERIMENT"
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

        requests_file = os.path.join(

            OUTPUT_DIR,

            (
                f"requests_shadai_400_"
                f"{experiment_id}.csv"
            )
        )

        valido = (

            arquivo_compativel(
                metrics_file,
                interval
            )

            and

            arquivo_compativel(
                eligibility_file,
                interval
            )

            and

            requests_compativel(
                requests_file,
                interval
            )
        )

        if (
            valido
            and
            not FORCAR_REEXECUCAO
        ):

            print(

                f"[{contador:03d}/{TOTAL:03d}] "

                f"SKIP | "

                f"{label:<10} | "

                f"interval={interval:>3} | "

                f"seed={seed:02d}"
            )

            puladas += 1

            continue

        if (
            os.path.exists(
                metrics_file
            )
            or
            os.path.exists(
                requests_file
            )
        ):

            print(

                f"[INFO] Stale/incompatible output: "
                f"{nome}, seed={seed:02d}. "
                f"Re-running."
            )

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

            f"interval={interval:>3} | "

            f"seed={seed:02d}"
        )

        inicio = time.time()

        resultado = subprocess.run(
            comando
        )

        duracao = (
            time.time()
            - inicio
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
                f"[FAIL] {nome} seed={seed:02d}"
            )

            continue

        valido = (

            arquivo_compativel(
                metrics_file,
                interval
            )

            and

            arquivo_compativel(
                eligibility_file,
                interval
            )

            and

            requests_compativel(
                requests_file,
                interval
            )
        )

        if not valido:

            falhas.append({

                "configuration":
                    nome,

                "interval":
                    interval,

                "seed":
                    seed,

                "reason":
                    "invalid_output"
            })

            print(

                f"[FAIL] Invalid C14 output | "
                f"{nome} | seed={seed:02d}"
            )

            continue

        executadas += 1

        print(

            f"[OK]   "

            f"{label:<10} | "

            f"interval={interval:>3} | "

            f"seed={seed:02d} | "

            f"{duracao:.2f}s"
        )


# ============================================================
# CHECAGEM FINAL
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

        requests_file = os.path.join(

            OUTPUT_DIR,

            (
                f"requests_shadai_400_"
                f"{experiment_id}.csv"
            )
        )

        if (

            arquivo_compativel(
                metrics_file,
                interval
            )

            and

            arquivo_compativel(
                eligibility_file,
                interval
            )

            and

            requests_compativel(
                requests_file,
                interval
            )
        ):

            completos += 1


tempo_total = (
    time.time()
    - inicio_geral
)


print("\n")

print("=" * 130)

print(
    "C14 CAPACITY EXPERIMENT SUMMARY"
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
    f"Total time          : "
    f"{tempo_total / 60:.2f} min"
)

print(
    f"Output directory    : {OUTPUT_DIR}"
)


if falhas:

    print("\nFAILED RUNS")

    print("-" * 130)

    for falha in falhas:

        print(

            f"{falha['configuration']} | "

            f"interval={falha['interval']} | "

            f"seed={falha['seed']:02d} | "

            f"reason={falha['reason']}"
        )


if completos == TOTAL:

    print(
        "\nALL C14 CAPACITY EXPERIMENTS COMPLETE"
    )

else:

    print(
        "\nC14 CAPACITY EXPERIMENT SET INCOMPLETE"
    )


print("=" * 130)
print()