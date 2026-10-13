# ablation_weights.py

import subprocess
import sys
import os
import time
import csv


# ============================================================
# DIRETÓRIO
# ============================================================

WEIGHT_DIR = "weight_ablation"

os.makedirs(
    WEIGHT_DIR,
    exist_ok=True
)


# ============================================================
# RNG OBRIGATÓRIO
# ============================================================

RNG_SCHEME_REQUIRED = "split_v1"


# ============================================================
# CONFIGURAÇÕES DOS PESOS
# ============================================================
#
# w1 = reputação
# w2 = energia residual
# w3 = proximidade
#
# ============================================================

CONFIGURACOES = {

    "original": (
        0.4,
        0.3,
        0.3
    ),

    "equal": (
        0.3333,
        0.3333,
        0.3334
    ),

    "reputation_high": (
        0.7,
        0.15,
        0.15
    ),

    "energy_high": (
        0.15,
        0.7,
        0.15
    ),

    "proximity_high": (
        0.15,
        0.15,
        0.7
    ),

    "no_reputation": (
        0.0,
        0.5,
        0.5
    ),

    "no_energy": (
        0.5714,
        0.0,
        0.4286
    ),

    "no_proximity": (
        0.5714,
        0.4286,
        0.0
    )
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
# LAMBDA FIXO
# ============================================================

LAMBDA1 = 0.5
LAMBDA2 = 0.3
LAMBDA3 = 0.2


# ============================================================
# EPSILON FIXO
# ============================================================

EPSILON = 0.2


# ============================================================
# SCRIPT PRINCIPAL
# ============================================================

SHADAI_SCRIPT = "shadai_sumo.py"


# ============================================================
# REEXECUÇÃO
# ============================================================

FORCAR_REEXECUCAO = False


# ============================================================
# FUNÇÃO DE VALIDAÇÃO
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
    "SHADAI - MULTI-SEED WEIGHT SENSITIVITY AND ABLATION"
)

print("=" * 110)

print(
    f"Output directory : {WEIGHT_DIR}"
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
    f"Epsilon fixed    : {EPSILON}"
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

for nome, pesos in CONFIGURACOES.items():

    w1, w2, w3 = pesos


    for seed in SEEDS:

        contador += 1


        experiment_id = (

            f"weights_"
            f"{nome}_"
            f"seed_"
            f"{seed:02d}"
        )


        metrics_file = os.path.join(

            WEIGHT_DIR,

            (
                f"metricas_shadai_400_"
                f"{experiment_id}.csv"
            )
        )


        eligibility_file = os.path.join(

            WEIGHT_DIR,

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

                f"{nome:<17} | "

                f"seed="
                f"{seed:02d}"
            )

            puladas += 1

            continue


        # ====================================================
        # OUTPUT ANTIGO / INCOMPATÍVEL
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
                w1
            ),

            "--w2",
            str(
                w2
            ),

            "--w3",
            str(
                w3
            ),

            "--epsilon",
            str(
                EPSILON
            ),

            "--experiment",
            experiment_id,

            "--seed",
            str(
                seed
            ),

            "--headless",

            "--output-dir",
            WEIGHT_DIR
        ]


        print(

            f"[{contador:03d}/"
            f"{TOTAL_EXECUCOES:03d}] "

            f"RUN  | "

            f"{nome:<17} | "

            f"w=("
            f"{w1:.4f}, "
            f"{w2:.4f}, "
            f"{w3:.4f}) | "

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
        # ERRO
        # ====================================================

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
                f"{nome} | "
                f"seed={seed:02d}"
            )

            continue


        # ====================================================
        # VALIDA NOVOS CSVs
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

                "seed":
                    seed,

                "reason":
                    "invalid_rng_output"
            })


            print(

                f"[FAIL] Invalid RNG output | "
                f"{nome} | "
                f"seed={seed:02d}"
            )

            continue


        executadas += 1


        print(

            f"[OK]   "

            f"{nome:<17} | "

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

            f"weights_"
            f"{nome}_"
            f"seed_"
            f"{seed:02d}"
        )


        metrics_file = os.path.join(

            WEIGHT_DIR,

            (
                f"metricas_shadai_400_"
                f"{experiment_id}.csv"
            )
        )


        eligibility_file = os.path.join(

            WEIGHT_DIR,

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
    "MULTI-SEED WEIGHT ABLATION SUMMARY"
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
    f"{WEIGHT_DIR}"
)


if falhas:

    print(
        "\nFAILED RUNS"
    )

    print("-" * 110)


    for falha in falhas:

        print(

            f"{falha['configuration']} | "

            f"seed="
            f"{falha['seed']:02d} | "

            f"reason="
            f"{falha['reason']}"
        )


if completos == TOTAL_EXECUCOES:

    print(
        "\nALL WEIGHT EXPERIMENTS COMPLETE"
    )

else:

    print(
        "\nWEIGHT EXPERIMENT SET INCOMPLETE"
    )


print("=" * 110)
print()