# ablation_flood_intensity.py

import os
import csv
import sys
import time
import subprocess


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = "flood_ablation"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# INTENSIDADES
# ============================================================

CONFIGS = {

    "flood_00": {
        "label": "0",
        "intensity": 0
    },

    "flood_01": {
        "label": "1",
        "intensity": 1
    },

    "flood_02": {
        "label": "2",
        "intensity": 2
    },

    "flood_03": {
        "label": "3",
        "intensity": 3
    },

    "flood_05": {
        "label": "5",
        "intensity": 5
    }
}


SEEDS = list(
    range(
        1,
        31
    )
)


# ============================================================
# BASELINE
# ============================================================

MALICIOUS_RATIO = 0.20

LAMBDA1 = 0.5
LAMBDA2 = 0.3
LAMBDA3 = 0.2

W1 = 0.4
W2 = 0.3
W3 = 0.3

EPSILON = 0.2

COOLDOWN = 40

REQUEST_INTERVAL = 20


# ============================================================
# VERSIONAMENTO
# ============================================================

EXPECTED_RNG = "split_v1"

EXPECTED_TELEMETRY = (
    "c18_flood_intensity_v1"
)

EXPECTED_ATTACK = (
    "flood_candidate_duplication_v1"
)

EXPECTED_FLOOD_SCHEME = (
    "c18_flood_param_v1"
)


SHADAI_SCRIPT = "shadai_sumo.py"

FORCAR_REEXECUCAO = False


# ============================================================
# VALIDATOR
# ============================================================

def csv_compativel(
    path,
    intensity
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


            row = next(
                reader,
                None
            )


            if row is None:

                return False


            checks = [

                (
                    row.get(
                        "rng_scheme"
                    )
                    ==
                    EXPECTED_RNG
                ),

                (
                    row.get(
                        "telemetry_scheme"
                    )
                    ==
                    EXPECTED_TELEMETRY
                ),

                (
                    row.get(
                        "attack_scheme"
                    )
                    ==
                    EXPECTED_ATTACK
                ),

                (
                    row.get(
                        "flood_scheme"
                    )
                    ==
                    EXPECTED_FLOOD_SCHEME
                ),

                (
                    row.get(
                        "attack_type"
                    )
                    ==
                    "flood"
                ),

                (
                    abs(
                        float(
                            row.get(
                                "malicious_ratio"
                            )
                        )
                        -
                        MALICIOUS_RATIO
                    )
                    <
                    1e-9
                ),

                (
                    int(
                        float(
                            row.get(
                                "flood_intensity"
                            )
                        )
                    )
                    ==
                    intensity
                )
            ]


            return all(
                checks
            )


    except Exception:

        return False


# ============================================================
# RUNS
# ============================================================

TOTAL = (
    len(CONFIGS)
    *
    len(SEEDS)
)


contador = 0
executadas = 0
puladas = 0
falhas = []

start_all = time.time()


print("\n")

print("=" * 135)

print(
    "SHADAI - C18 FLOOD INTENSITY ABLATION"
)

print("=" * 135)


print(
    f"Output directory : {OUTPUT_DIR}"
)

print(
    f"Malicious ratio  : "
    f"{MALICIOUS_RATIO * 100:.0f}%"
)

print(
    f"Intensities      : "
    f"{[x['intensity'] for x in CONFIGS.values()]}"
)

print(
    f"Seeds            : {len(SEEDS)}"
)

print(
    f"Total runs       : {TOTAL}"
)

print("=" * 135)


for config_name, cfg in CONFIGS.items():

    intensity = cfg[
        "intensity"
    ]


    for seed in SEEDS:

        contador += 1


        experiment = (

            f"{config_name}_"
            f"intensity_{intensity:02d}_"
            f"seed_{seed:02d}"
        )


        flood_file = os.path.join(

            OUTPUT_DIR,

            (
                f"flood_shadai_400_"
                f"{experiment}.csv"
            )
        )


        metrics_file = os.path.join(

            OUTPUT_DIR,

            (
                f"metricas_shadai_400_"
                f"{experiment}.csv"
            )
        )


        if (
            not FORCAR_REEXECUCAO

            and

            csv_compativel(
                flood_file,
                intensity
            )

            and

            csv_compativel(
                metrics_file,
                intensity
            )
        ):

            print(

                f"[{contador:03d}/{TOTAL:03d}] "

                f"SKIP | "

                f"intensity="
                f"{intensity} | "

                f"seed="
                f"{seed:02d}"
            )


            puladas += 1

            continue


        cmd = [

            sys.executable,

            SHADAI_SCRIPT,

            "--attack-type",
            "flood",

            "--malicious-ratio",
            str(
                MALICIOUS_RATIO
            ),

            "--flood-intensity",
            str(
                intensity
            ),

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
                COOLDOWN
            ),

            "--request-interval",
            str(
                REQUEST_INTERVAL
            ),

            "--seed",
            str(
                seed
            ),

            "--experiment",
            experiment,

            "--output-dir",
            OUTPUT_DIR,

            "--headless"
        ]


        print(

            f"[{contador:03d}/{TOTAL:03d}] "

            f"RUN  | "

            f"intensity="
            f"{intensity} | "

            f"seed="
            f"{seed:02d}"
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

            falhas.append(
                (
                    config_name,
                    seed,
                    "simulation_error"
                )
            )

            print(
                "[FAIL]"
            )

            continue


        if not (
            csv_compativel(
                flood_file,
                intensity
            )
            and
            csv_compativel(
                metrics_file,
                intensity
            )
        ):

            falhas.append(
                (
                    config_name,
                    seed,
                    "invalid_output"
                )
            )

            print(
                "[FAIL] Invalid output."
            )

            continue


        executadas += 1


        print(

            f"[OK]   "
            f"{duration:.2f}s"
        )


# ============================================================
# COMPLETENESS
# ============================================================

complete = 0


for config_name, cfg in CONFIGS.items():

    intensity = cfg[
        "intensity"
    ]


    for seed in SEEDS:

        experiment = (

            f"{config_name}_"
            f"intensity_{intensity:02d}_"
            f"seed_{seed:02d}"
        )


        flood_file = os.path.join(

            OUTPUT_DIR,

            (
                f"flood_shadai_400_"
                f"{experiment}.csv"
            )
        )


        metrics_file = os.path.join(

            OUTPUT_DIR,

            (
                f"metricas_shadai_400_"
                f"{experiment}.csv"
            )
        )


        if (
            csv_compativel(
                flood_file,
                intensity
            )
            and
            csv_compativel(
                metrics_file,
                intensity
            )
        ):

            complete += 1


print("\n")

print("=" * 135)

print(
    "C18 FLOOD ABLATION SUMMARY"
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
    f"{complete}/{TOTAL}"
)

print(
    f"Total time          : "
    f"{(time.time() - start_all) / 60:.2f} min"
)


if complete == TOTAL:

    print(
        "\nALL C18 FLOOD-INTENSITY RUNS COMPLETE"
    )

else:

    print(
        "\nC18 FLOOD EXPERIMENT SET INCOMPLETE"
    )


if falhas:

    print("\nFailures:")

    for failure in falhas:

        print(
            failure
        )


print("=" * 135)

print()