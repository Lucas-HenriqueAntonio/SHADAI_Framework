# ablation_cooldown.py

import subprocess
import sys
import os
import time
import csv


# ============================================================
# DIRETÓRIO
# ============================================================

OUTPUT_DIR = "cooldown_ablation"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# COOLDOWN CONFIGURATIONS
# ============================================================

CONFIGURACOES = {

    "cooldown_000": 0,

    "cooldown_020": 20,

    "cooldown_040": 40,

    "cooldown_060": 60,

    "cooldown_080": 80
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


# ============================================================
# INFRAESTRUTURA EXPERIMENTAL
# ============================================================

RNG_SCHEME_REQUIRED = "split_v1"

TELEMETRY_SCHEME_REQUIRED = "c9_regret_v1"


SHADAI_SCRIPT = "shadai_sumo.py"


# Se True, executa tudo novamente mesmo com CSV válido.
FORCAR_REEXECUCAO = False


# ============================================================
# VALIDA OUTPUT
# ============================================================

def arquivo_compativel(
    caminho,
    cooldown_esperado
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


            valor_cooldown = int(
                float(
                    primeira[
                        "cooldown_steps"
                    ]
                )
            )


            if (
                valor_cooldown
                !=
                cooldown_esperado
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

    len(
        CONFIGURACOES
    )

    *

    len(
        SEEDS
    )
)


# ============================================================
# CABEÇALHO
# ============================================================

print("\n")

print("=" * 120)

print(
    "SHADAI - MULTI-SEED COOLDOWN SENSITIVITY ANALYSIS"
)

print("=" * 120)


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
    f"Configurations   : {len(CONFIGURACOES)}"
)

print(
    f"Seeds            : {len(SEEDS)}"
)

print(
    f"Total runs       : {TOTAL}"
)

print(
    f"Lambda fixed     : "
    f"({LAMBDA1}, {LAMBDA2}, {LAMBDA3})"
)

print(
    f"Weights fixed    : "
    f"({W1}, {W2}, {W3})"
)

print(
    f"Epsilon fixed    : {EPSILON}"
)

print("=" * 120)


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

for nome, cooldown in CONFIGURACOES.items():

    for seed in SEEDS:

        contador += 1


        experiment_id = (

            f"{nome}_"
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

                cooldown
            )
        )


        eligibility_valida = (
            arquivo_compativel(

                eligibility_file,

                cooldown
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

                f"{nome:<13} | "

                f"cooldown="
                f"{cooldown:>3} | "

                f"seed="
                f"{seed:02d}"
            )


            puladas += 1

            continue


        # ====================================================
        # INFORMA OUTPUT ANTIGO
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
                cooldown
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

            f"{nome:<13} | "

            f"cooldown="
            f"{cooldown:>3} | "

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

                "cooldown":
                    cooldown,

                "seed":
                    seed,

                "reason":
                    resultado.returncode
            })


            print(

                f"[FAIL] "

                f"{nome} | "

                f"cooldown="
                f"{cooldown} | "

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

                cooldown
            )

            or

            not arquivo_compativel(

                eligibility_file,

                cooldown
            )
        ):

            falhas.append({

                "configuration":
                    nome,

                "cooldown":
                    cooldown,

                "seed":
                    seed,

                "reason":
                    "invalid_output"
            })


            print(

                f"[FAIL] Invalid output | "

                f"{nome} | "

                f"cooldown="
                f"{cooldown} | "

                f"seed="
                f"{seed:02d}"
            )


            continue


        executadas += 1


        print(

            f"[OK]   "

            f"{nome:<13} | "

            f"cooldown="
            f"{cooldown:>3} | "

            f"seed="
            f"{seed:02d} | "

            f"{duracao:.2f}s"
        )


# ============================================================
# VALIDAÇÃO FINAL
# ============================================================

completos = 0


for nome, cooldown in CONFIGURACOES.items():

    for seed in SEEDS:

        experiment_id = (

            f"{nome}_"
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

                cooldown
            )

            and

            arquivo_compativel(

                eligibility_file,

                cooldown
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

print("=" * 120)

print(
    "COOLDOWN ABLATION SUMMARY"
)

print("=" * 120)


print(
    f"Expected            : "
    f"{TOTAL}"
)

print(
    f"Executed            : "
    f"{executadas}"
)

print(
    f"Skipped             : "
    f"{puladas}"
)

print(
    f"Failures            : "
    f"{len(falhas)}"
)

print(
    f"Complete compatible : "
    f"{completos}/"
    f"{TOTAL}"
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

    print("-" * 120)


    for falha in falhas:

        print(

            f"{falha['configuration']} | "

            f"cooldown="
            f"{falha['cooldown']} | "

            f"seed="
            f"{falha['seed']:02d} | "

            f"reason="
            f"{falha['reason']}"
        )


if completos == TOTAL:

    print(
        "\nALL COOLDOWN EXPERIMENTS COMPLETE"
    )

else:

    print(
        "\nCOOLDOWN EXPERIMENT SET INCOMPLETE"
    )


print("=" * 120)

print()