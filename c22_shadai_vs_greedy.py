# c22_shadai_vs_greedy.py

import csv
import os
import subprocess
import sys


# ============================================================
# CONFIGURATION
# ============================================================

SHADAI_SCRIPT = "shadai_sumo.py"

OUTPUT_DIR = "greedy_alignment"

SEEDS = list(
    range(1, 31)
)


# ============================================================
# SHARED PARAMETERS
# ============================================================

LAMBDA1 = 0.5
LAMBDA2 = 0.3
LAMBDA3 = 0.2

W1 = 0.4
W2 = 0.3
W3 = 0.3

EPSILON_SHADAI = 0.2
EPSILON_GREEDY = 0.0

COOLDOWN_STEPS = 40

REQUEST_INTERVAL = 20


# ============================================================
# EXPECTED SCHEMES
# ============================================================

RNG_SCHEME = "split_v1"


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HELPERS
# ============================================================

def experiment_name(
    policy,
    seed
):

    return (
        f"c22_{policy}_"
        f"seed_{seed:02d}"
    )


def metrics_path(
    experiment
):

    return os.path.join(

        OUTPUT_DIR,

        (
            f"metricas_shadai_400_"
            f"{experiment}.csv"
        )
    )


def run_is_compatible(
    path,
    seed,
    epsilon
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


            first = next(
                reader,
                None
            )


            if first is None:

                return False


            file_seed = int(
                float(
                    first.get(
                        "seed",
                        -1
                    )
                )
            )


            file_epsilon = float(
                first.get(
                    "epsilon",
                    -1
                )
            )


            file_rng = first.get(
                "rng_scheme",
                ""
            )


            attack_type = first.get(
                "attack_type",
                "none"
            )


            malicious_ratio = float(
                first.get(
                    "malicious_ratio",
                    0.0
                )
            )


            return (

                file_seed
                ==
                seed

                and

                abs(
                    file_epsilon
                    -
                    epsilon
                )
                <
                1e-12

                and

                file_rng
                ==
                RNG_SCHEME

                and

                attack_type
                ==
                "none"

                and

                abs(
                    malicious_ratio
                )
                <
                1e-12
            )


    except Exception:

        return False


def run_policy(
    policy,
    seed,
    epsilon
):

    experiment = (
        experiment_name(
            policy,
            seed
        )
    )


    path = (
        metrics_path(
            experiment
        )
    )


    if run_is_compatible(
        path,
        seed,
        epsilon
    ):

        print(
            f"[SKIP] "
            f"{policy:<7} "
            f"seed={seed:02d} "
            f"already exists and is compatible."
        )

        return


    print(
        f"[RUN ] "
        f"{policy:<7} "
        f"seed={seed:02d} "
        f"epsilon={epsilon:.2f}"
    )


    cmd = [

        sys.executable,

        SHADAI_SCRIPT,

        "--lambda1",
        str(LAMBDA1),

        "--lambda2",
        str(LAMBDA2),

        "--lambda3",
        str(LAMBDA3),

        "--w1",
        str(W1),

        "--w2",
        str(W2),

        "--w3",
        str(W3),

        "--epsilon",
        str(epsilon),

        "--cooldown-steps",
        str(COOLDOWN_STEPS),

        "--request-interval",
        str(REQUEST_INTERVAL),

        "--malicious-ratio",
        "0.0",

        "--attack-type",
        "none",

        "--flood-intensity",
        "0",

        "--seed",
        str(seed),

        "--experiment",
        experiment,

        "--output-dir",
        OUTPUT_DIR,

        "--headless"
    ]


    result = subprocess.run(
        cmd,
        check=False
    )


    if result.returncode != 0:

        raise RuntimeError(
            f"{policy} failed for seed {seed}. "
            f"Return code = {result.returncode}"
        )


# ============================================================
# HEADER
# ============================================================

print("\n")

print("=" * 130)

print(
    "C22 - SHADAI VS CORRECTED GREEDY BASELINE"
)

print("=" * 130)


print(
    f"Seeds               : "
    f"{len(SEEDS)}"
)

print(
    f"SHADAI epsilon      : "
    f"{EPSILON_SHADAI}"
)

print(
    f"Greedy epsilon      : "
    f"{EPSILON_GREEDY}"
)

print(
    "Greedy definition   : "
    "argmax final Score_i"
)

print(
    "Attack              : disabled"
)

print(
    f"Cooldown            : "
    f"{COOLDOWN_STEPS}"
)

print(
    f"Request interval    : "
    f"{REQUEST_INTERVAL}"
)

print(
    f"Output directory    : "
    f"{OUTPUT_DIR}"
)

print("=" * 130)

print()


# ============================================================
# RUN PAIRED SEEDS
# ============================================================

for seed in SEEDS:

    print(
        "\n"
        +
        "-" * 130
    )

    print(
        f"PAIRED SEED {seed:02d}"
    )

    print(
        "-" * 130
    )


    run_policy(
        "shadai",
        seed,
        EPSILON_SHADAI
    )


    run_policy(
        "greedy",
        seed,
        EPSILON_GREEDY
    )


# ============================================================
# DONE
# ============================================================

print("\n")

print("=" * 130)

print(
    "C22 SHADAI VS GREEDY RUNS COMPLETED"
)

print("=" * 130)

print()