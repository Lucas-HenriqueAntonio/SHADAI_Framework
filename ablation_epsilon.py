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
# ESQUEMA RNG EXIGIDO
# ============================================================

RNG_SCHEME_REQUIRED = "split_v1"


# ============================================================
# CONFIGURAÇÕES DE EPSILON
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


# ============================================================
# SCRIPT PRINCIPAL
# ============================================================

SHADAI_SCRIPT = "shadai_sumo.py"


# ============================================================
# REEXECUÇÃO
# ============================================================

FORCAR_REEXECUCAO = False


# ============================================================
# FUNÇÃO PARA VALIDAR CSV
# ============================================================

def arquivo_rng_compativel(
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

            if (
                reader.fieldnames is None
                or
                "rng_scheme"
                not in reader.fieldnames
            ):

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


# ============================================================
# TOTAL
# ============================================================

TOTAL_EXECUCOES = (
    len(CONFIGURACOES)
    *
    len(SEEDS)
)


# ============================================================
# CABEÇALHO
# ============================================================

print("\n")
print("=" * 110)

print(
    "SHADAI - MULTI-SEED EPSILON SENSITIVITY ANALYSIS"
)

print("=" * 110)

print(
    f"Output directory : {EPSILON_DIR}"
)

print(
    f"RNG scheme       : {RNG_SCHEME_REQUIRED}"
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

print(
    f"Lambda fixed     : "
    f"({LAMBDA1}, {LAMBDA2}, {LAMBDA3})"
)

print(
    f"Score weights    : "
    f"({W1}, {W2}, {W3})"
)

print("=" * 110)


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


        metricas_validas = (
            arquivo_rng_compativel(
                metrics_file
            )
        )


        eligibility_valida = (
            arquivo_rng_compativel(
                eligibility_file
            )
        )


        # ====================================================
        # SKIP
        # ====================================================

        if (
            not FORCAR_REEXECUCAO
            and
            metricas_validas
            and
            eligibility_valida
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


        # ====================================================
        # AVISA SOBRE OUTPUT ANTIGO
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


        # ====================================================
        # ERRO DO PROCESSO
        # ====================================================

        if resultado.returncode != 0:

            falhas.append({

                "configuration":
                    nome,

                "epsilon":
                    epsilon,

                "seed":
                    seed,

                "reason":
                    resultado.returncode
            })


            print(

                f"[FAIL] "
                f"{nome} | "
                f"seed={seed:02d}"
            )

            continue


        # ====================================================
        # VERIFICA OUTPUT
        # ====================================================

        if (
            not arquivo_rng_compativel(
                metrics_file
            )
            or
            not arquivo_rng_compativel(
                eligibility_file
            )
        ):

            falhas.append({

                "configuration":
                    nome,

                "epsilon":
                    epsilon,

                "seed":
                    seed,

                "reason":
                    "invalid_rng_output"
            })


            print(

                f"[FAIL] "
                f"Invalid RNG output | "
                f"{nome} | "
                f"seed={seed:02d}"
            )

            continue


        executadas += 1


        print(

            f"[OK]   "

            f"{nome:<12} | "

            f"epsilon="
            f"{epsilon:.2f} | "

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
            arquivo_rng_compativel(
                metrics_file
            )
            and
            arquivo_rng_compativel(
                eligibility_file
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
print("=" * 110)

print(
    "MULTI-SEED EPSILON ABLATION SUMMARY"
)

print("=" * 110)


print(
    f"Expected runs       : "
    f"{TOTAL_EXECUCOES}"
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
    f"{TOTAL_EXECUCOES}"
)

print(
    f"RNG scheme          : "
    f"{RNG_SCHEME_REQUIRED}"
)

print(
    f"Total time          : "
    f"{tempo_total / 60:.2f} min"
)

print(
    f"Output directory    : "
    f"{EPSILON_DIR}"
)


if falhas:

    print(
        "\nFAILED RUNS"
    )

    print("-" * 110)

    for falha in falhas:

        print(

            f"{falha['configuration']} | "

            f"epsilon="
            f"{falha['epsilon']:.2f} | "

            f"seed="
            f"{falha['seed']:02d} | "

            f"reason="
            f"{falha['reason']}"
        )


if completos == TOTAL_EXECUCOES:

    print(
        "\nALL EPSILON EXPERIMENTS COMPLETE"
    )

else:

    print(
        "\nEPSILON EXPERIMENT SET INCOMPLETE"
    )


print("=" * 110)
print()