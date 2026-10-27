# ablation_malicious_ratio.py

import subprocess
import sys
import os
import csv
import time


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = "malicious_ablation"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

CONFIGURACOES = {

    "malicious_00": {
        "label": "0%",
        "ratio": 0.00
    },

    "malicious_10": {
        "label": "10%",
        "ratio": 0.10
    },

    "malicious_20": {
        "label": "20%",
        "ratio": 0.20
    },

    "malicious_30": {
        "label": "30%",
        "ratio": 0.30
    },

    "malicious_40": {
        "label": "40%",
        "ratio": 0.40
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
# PARÂMETROS BASELINE
# ============================================================

LAMBDA1 = 0.5
LAMBDA2 = 0.3
LAMBDA3 = 0.2

W1 = 0.4
W2 = 0.3
W3 = 0.3

EPSILON = 0.2

COOLDOWN_STEPS = 40

REQUEST_INTERVAL = 20


# ============================================================
# VERSIONAMENTO
# ============================================================

RNG_REQUIRED = "split_v1"

TELEMETRY_REQUIRED = (
    "c17_malicious_ratio_v1"
)

ATTACK_SCHEME_REQUIRED = (
    "deliberate_service_failure_v1"
)


SHADAI_SCRIPT = "shadai_sumo.py"

FORCAR_REEXECUCAO = False


# ============================================================
# VALIDAÇÃO CSV
# ============================================================

def csv_compativel(
    path,
    expected_ratio
):

    if not os.path.exists(
        path
    ):

        return False


    try:

        with open(
            path,
            "r",
            newline="",
            encoding="utf-8"
        ) as f:

            reader = csv.DictReader(
                f
            )


            if reader.fieldnames is None:

                return False


            required = [

                "rng_scheme",

                "telemetry_scheme",

                "attack_scheme",

                "malicious_ratio"
            ]


            for col in required:

                if col not in reader.fieldnames:

                    return False


            row = next(
                reader,
                None
            )


            if row is None:

                return False


            if (
                row[
                    "rng_scheme"
                ]
                !=
                RNG_REQUIRED
            ):

                return False


            if (
                row[
                    "telemetry_scheme"
                ]
                !=
                TELEMETRY_REQUIRED
            ):

                return False


            if (
                row[
                    "attack_scheme"
                ]
                !=
                ATTACK_SCHEME_REQUIRED
            ):

                return False


            ratio = float(
                row[
                    "malicious_ratio"
                ]
            )


            if abs(
                ratio
                -
                expected_ratio
            ) > 1e-9:

                return False


            return True


    except Exception:

        return False


# ============================================================
# CHECK SCRIPT
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

print("=" * 135)

print(
    "SHADAI - C17 MULTI-SEED MALICIOUS RATIO ABLATION"
)

print("=" * 135)


print(
    f"Output directory : {OUTPUT_DIR}"
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
    f"RNG scheme       : {RNG_REQUIRED}"
)

print(
    f"Telemetry        : {TELEMETRY_REQUIRED}"
)

print(
    f"Attack scheme    : {ATTACK_SCHEME_REQUIRED}"
)

print(
    f"Request interval : {REQUEST_INTERVAL}"
)

print(
    f"Cooldown         : {COOLDOWN_STEPS}"
)

print(
    f"Epsilon          : {EPSILON}"
)

print("=" * 135)


# ============================================================
# EXECUÇÕES
# ============================================================

contador = 0

executadas = 0

puladas = 0

falhas = []

inicio_geral = time.time()


for config_name, cfg in CONFIGURACOES.items():

    ratio = cfg[
        "ratio"
    ]

    label = cfg[
        "label"
    ]


    for seed in SEEDS:

        contador += 1


        experiment_id = (

            f"{config_name}_"
            f"ratio_{int(ratio * 100):02d}_"
            f"seed_{seed:02d}"
        )


        vehicles_file = os.path.join(

            OUTPUT_DIR,

            (
                f"vehicles_shadai_400_"
                f"{experiment_id}.csv"
            )
        )


        outcomes_file = os.path.join(

            OUTPUT_DIR,

            (
                f"outcomes_shadai_400_"
                f"{experiment_id}.csv"
            )
        )


        metrics_file = os.path.join(

            OUTPUT_DIR,

            (
                f"metricas_shadai_400_"
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


        eligibility_file = os.path.join(

            OUTPUT_DIR,

            (
                f"eligibilidade_shadai_400_"
                f"{experiment_id}.csv"
            )
        )


        runtime_file = os.path.join(

            OUTPUT_DIR,

            (
                f"runtime_shadai_400_"
                f"{experiment_id}.csv"
            )
        )


        valido = (

            csv_compativel(
                vehicles_file,
                ratio
            )

            and

            csv_compativel(
                metrics_file,
                ratio
            )

            and

            csv_compativel(
                outcomes_file,
                ratio
            )

            and

            csv_compativel(
                requests_file,
                ratio
            )

            and

            csv_compativel(
                eligibility_file,
                ratio
            )

            and

            csv_compativel(
                runtime_file,
                ratio
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

                f"ratio={label:<4} | "

                f"seed={seed:02d}"
            )


            puladas += 1

            continue


        cmd = [

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
                REQUEST_INTERVAL
            ),

            "--attack-enabled",

            "--malicious-ratio",
            str(
                ratio
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

            f"ratio={label:<4} | "

            f"seed={seed:02d}"
        )


        start = time.time()


        result = subprocess.run(
            cmd
        )


        duration = (
            time.time()
            -
            start
        )


        if result.returncode != 0:

            falhas.append({

                "configuration":
                    config_name,

                "ratio":
                    ratio,

                "seed":
                    seed,

                "reason":
                    result.returncode
            })


            print(
                "[FAIL] Simulation error."
            )

            continue


        valido = (

            csv_compativel(
                vehicles_file,
                ratio
            )

            and

            csv_compativel(
                metrics_file,
                ratio
            )

            and

            csv_compativel(
                outcomes_file,
                ratio
            )

            and

            csv_compativel(
                requests_file,
                ratio
            )

            and

            csv_compativel(
                eligibility_file,
                ratio
            )

            and

            csv_compativel(
                runtime_file,
                ratio
            )
        )


        if not valido:

            falhas.append({

                "configuration":
                    config_name,

                "ratio":
                    ratio,

                "seed":
                    seed,

                "reason":
                    "invalid_output"
            })


            print(
                "[FAIL] Invalid C17 output."
            )

            continue


        executadas += 1


        print(

            f"[OK]   "

            f"ratio={label:<4} | "

            f"seed={seed:02d} | "

            f"{duration:.2f}s"
        )


# ============================================================
# CHECAGEM FINAL
# ============================================================

completos = 0


for config_name, cfg in CONFIGURACOES.items():

    ratio = cfg[
        "ratio"
    ]


    for seed in SEEDS:

        experiment_id = (

            f"{config_name}_"
            f"ratio_{int(ratio * 100):02d}_"
            f"seed_{seed:02d}"
        )


        files = {

            "vehicles":
                os.path.join(
                    OUTPUT_DIR,
                    (
                        f"vehicles_shadai_400_"
                        f"{experiment_id}.csv"
                    )
                ),

            "metrics":
                os.path.join(
                    OUTPUT_DIR,
                    (
                        f"metricas_shadai_400_"
                        f"{experiment_id}.csv"
                    )
                ),

            "outcomes":
                os.path.join(
                    OUTPUT_DIR,
                    (
                        f"outcomes_shadai_400_"
                        f"{experiment_id}.csv"
                    )
                ),

            "requests":
                os.path.join(
                    OUTPUT_DIR,
                    (
                        f"requests_shadai_400_"
                        f"{experiment_id}.csv"
                    )
                ),

            "eligibility":
                os.path.join(
                    OUTPUT_DIR,
                    (
                        f"eligibilidade_shadai_400_"
                        f"{experiment_id}.csv"
                    )
                ),

            "runtime":
                os.path.join(
                    OUTPUT_DIR,
                    (
                        f"runtime_shadai_400_"
                        f"{experiment_id}.csv"
                    )
                )
        }


        if all(

            csv_compativel(
                path,
                ratio
            )

            for path in files.values()
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

print("=" * 135)

print(
    "C17 EXPERIMENT SUMMARY"
)

print("=" * 135)


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
    f"Output directory    : "
    f"{OUTPUT_DIR}"
)


if completos == TOTAL:

    print(
        "\nALL C17 MALICIOUS-RATIO RUNS COMPLETE"
    )

else:

    print(
        "\nC17 EXPERIMENT SET INCOMPLETE"
    )


if falhas:

    print("\nFAILED RUNS")

    print("-" * 135)


    for failure in falhas:

        print(

            f"{failure['configuration']} | "

            f"ratio={failure['ratio']:.2f} | "

            f"seed={failure['seed']:02d} | "

            f"reason={failure['reason']}"
        )


print("=" * 135)

print()