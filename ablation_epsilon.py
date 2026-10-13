# ablation_epsilon.py

import subprocess
import sys
import os
import time
import csv


# ============================================================
# DIRETÓRIO
# ============================================================

EPSILON_DIR = "epsilon_ablation"

os.makedirs(
    EPSILON_DIR,
    exist_ok=True
)


# ============================================================
# VERSÕES EXIGIDAS
# ============================================================

RNG_SCHEME_REQUIRED = "split_v1"

TELEMETRY_SCHEME_REQUIRED = "c9_regret_v1"


# ============================================================
# EPSILON
# ============================================================

CONFIGURACOES = {

    "epsilon_000": 0.0,

    "epsilon_010": 0.1,

    "epsilon_020": 0.2,

    "epsilon_030": 0.3,

    "epsilon_050": 0.5,

    "epsilon_100": 1.0
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


SHADAI_SCRIPT = "shadai_sumo.py"


FORCAR_REEXECUCAO = False


# ============================================================
# VALIDA CSV
# ============================================================

def arquivo_compativel(
    caminho
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

                "telemetry_scheme"
            ]

            for coluna in obrigatorias:

                if coluna not in reader.fieldnames:
                    return False

            primeira_linha = next(
                reader,
                None
            )

            if primeira_linha is None:
                return False

            return (

                primeira_linha[
                    "rng_scheme"
                ]
                ==
                RNG_SCHEME_REQUIRED

                and

                primeira_linha[
                    "telemetry_scheme"
                ]
                ==
                TELEMETRY_SCHEME_REQUIRED
            )

    except Exception:

        return False


# ============================================================
# VERIFICA SCRIPT
# ============================================================

if not os.path.exists(
    SHADAI_SCRIPT
):

    raise FileNotFoundError(
        f"Arquivo não encontrado: "
        f"{SHADAI_SCRIPT}"
    )


TOTAL_EXECUCOES = (
    len(CONFIGURACOES)
    *
    len(SEEDS)
)


# ============================================================
# CABEÇALHO
# ============================================================

print("\n")
print("=" * 115)

print(
    "SHADAI - MULTI-SEED EPSILON EXPERIMENT WITH C9 REGRET TELEMETRY"
)

print("=" * 115)

print(
    f"Output directory : {EPSILON_DIR}"
)

print(
    f"RNG scheme       : {RNG_SCHEME_REQUIRED}"
)

print(
    f"Telemetry        : {TELEMETRY_SCHEME_REQUIRED}"
)

print(
    f"Configurations   : {len(CONFIGURACOES)}"
)

print(
    f"Seeds            : {len(SEEDS)}"
)

print(
    f"Total runs       : {TOTAL_EXECUCOES}"
)

print("=" * 115)


executadas = 0

puladas = 0

falhas = []

contador = 0

inicio_geral = time.time()


# ============================================================
# EXECUÇÕES
# ============================================================

for nome, epsilon in CONFIGURACOES.items():

    for seed in SEEDS:

        contador += 1

        experiment_id = (

            f"{nome}_"
            f"seed_"
            f"{seed:02d}"
        )

        metrics_file = os.path.join(

            EPSILON_DIR,

            (
                f"metricas_shadai_400_"
                f"{experiment_id}.csv"
            )
        )

        eligibility_file = os.path.join(

            EPSILON_DIR,

            (
                f"eligibilidade_shadai_400_"
                f"{experiment_id}.csv"
            )
        )

        if (
            not FORCAR_REEXECUCAO
            and
            arquivo_compativel(
                metrics_file
            )
            and
            arquivo_compativel(
                eligibility_file
            )
        ):

            print(

                f"[{contador:03d}/"
                f"{TOTAL_EXECUCOES:03d}] "

                f"SKIP | "

                f"{nome:<12} | "

                f"epsilon="
                f"{epsilon:.2f} | "

                f"seed="
                f"{seed:02d}"
            )

            puladas += 1

            continue

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

                f"[INFO] Stale telemetry: "
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
                epsilon
            ),

            "--experiment",
            experiment_id,

            "--seed",
            str(
                seed
            ),

            "--headless",

            "--output-dir",
            EPSILON_DIR
        ]

        print(

            f"[{contador:03d}/"
            f"{TOTAL_EXECUCOES:03d}] "

            f"RUN  | "

            f"{nome:<12} | "

            f"epsilon="
            f"{epsilon:.2f} | "

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

        if resultado.returncode != 0:

            falhas.append({

                "configuration":
                    nome,

                "seed":
                    seed,

                "reason":
                    resultado.returncode
            })

            print(
                f"[FAIL] "
                f"{nome} "
                f"seed={seed:02d}"
            )

            continue

        if (
            not arquivo_compativel(
                metrics_file
            )
            or
            not arquivo_compativel(
                eligibility_file
            )
        ):

            falhas.append({

                "configuration":
                    nome,

                "seed":
                    seed,

                "reason":
                    "invalid_telemetry_output"
            })

            print(

                f"[FAIL] Invalid telemetry | "
                f"{nome} "
                f"seed={seed:02d}"
            )

            continue

        executadas += 1

        print(

            f"[OK]   "

            f"{nome:<12} | "

            f"seed="
            f"{seed:02d} | "

            f"{duracao:.2f}s"
        )


# ============================================================
# CHECAGEM FINAL
# ============================================================

completos = 0


for nome in CONFIGURACOES:

    for seed in SEEDS:

        experiment_id = (

            f"{nome}_"
            f"seed_"
            f"{seed:02d}"
        )

        metrics_file = os.path.join(

            EPSILON_DIR,

            (
                f"metricas_shadai_400_"
                f"{experiment_id}.csv"
            )
        )

        eligibility_file = os.path.join(

            EPSILON_DIR,

            (
                f"eligibilidade_shadai_400_"
                f"{experiment_id}.csv"
            )
        )

        if (
            arquivo_compativel(
                metrics_file
            )
            and
            arquivo_compativel(
                eligibility_file
            )
        ):

            completos += 1


tempo_total = (
    time.time()
    - inicio_geral
)


print("\n")
print("=" * 115)

print(
    "C9 EPSILON EXPERIMENT SUMMARY"
)

print("=" * 115)

print(
    f"Expected            : {TOTAL_EXECUCOES}"
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
    f"{completos}/{TOTAL_EXECUCOES}"
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


if completos == TOTAL_EXECUCOES:

    print(
        "\nALL C9 EPSILON EXPERIMENTS COMPLETE"
    )

else:

    print(
        "\nC9 EPSILON EXPERIMENT SET INCOMPLETE"
    )


print("=" * 115)
print()