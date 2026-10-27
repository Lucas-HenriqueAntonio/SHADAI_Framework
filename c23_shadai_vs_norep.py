# c23_shadai_vs_norep.py

import csv
import os
import subprocess
import sys


# ============================================================
# CONFIGURATION
# ============================================================

SHADAI_SCRIPT = "shadai_sumo.py"

OUTPUT_DIR = "norep_alignment"

SEEDS = list(
    range(1, 31)
)


# ============================================================
# COMMON PARAMETERS
# ============================================================

LAMBDA1 = 0.5
LAMBDA2 = 0.3
LAMBDA3 = 0.2

EPSILON = 0.2

COOLDOWN_STEPS = 40

REQUEST_INTERVAL = 20


# ============================================================
# POLICY WEIGHTS
# ============================================================

POLICIES = {

    "shadai": {

        "w1": 0.4,

        "w2": 0.3,

        "w3": 0.3
    },

    "norep": {

        "w1": 0.0,

        "w2": 0.5,

        "w3": 0.5
    }
}


# ============================================================
# EXPECTED SCHEME
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
        f"c23_{policy}_"
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


def compatible_float(
    value,
    target
):

    try:

        return (
            abs(
                float(value)
                -
                target
            )
            <
            1e-12
        )

    except Exception:

        return False


def run_is_compatible(
    path,
    seed,
    weights
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


            return (

                file_seed
                ==
                seed

                and

                first.get(
                    "rng_scheme",
                    ""
                )
                ==
                RNG_SCHEME

                and

                compatible_float(
                    first.get(
                        "w1",
                        -1
                    ),
                    weights[
                        "w1"
                    ]
                )

                and

                compatible_float(
                    first.get(
                        "w2",
                        -1
                    ),
                    weights[
                        "w2"
                    ]
                )

                and

                compatible_float(
                    first.get(
                        "w3",
                        -1
                    ),
                    weights[
                        "w3"
                    ]
                )

                and

                compatible_float(
                    first.get(
                        "epsilon",
                        -1
                    ),
                    EPSILON
                )

                and

                first.get(
                    "attack_type",
                    "none"
                )
                ==
                "none"

                and

                compatible_float(
                    first.get(
                        "malicious_ratio",
                        0
                    ),
                    0.0
                )
            )


    except Exception:

        return False


def run_policy(
    policy,
    seed
):

    weights = POLICIES[
        policy
    ]


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

        weights
    ):

        print(
            f"[SKIP] "
            f"{policy:<7} "
            f"seed={seed:02d} "
            f"weights=("
            f"{weights['w1']}, "
            f"{weights['w2']}, "
            f"{weights['w3']})"
        )

        return


    print(
        f"[RUN ] "
        f"{policy:<7} "
        f"seed={seed:02d} "
        f"weights=("
        f"{weights['w1']}, "
        f"{weights['w2']}, "
        f"{weights['w3']})"
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
        str(
            weights[
                "w1"
            ]
        ),

        "--w2",
        str(
            weights[
                "w2"
            ]
        ),

        "--w3",
        str(
            weights[
                "w3"
            ]
        ),

        "--epsilon",
        str(EPSILON),

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
    "C23 - SHADAI VS RENORMALIZED NO-REPUTATION"
)

print("=" * 130)


print(
    "SHADAI weights        : "
    "(0.4, 0.3, 0.3)"
)

print(
    "No-Reputation weights : "
    "(0.0, 0.5, 0.5)"
)

print(
    f"Epsilon               : "
    f"{EPSILON}"
)

print(
    f"Cooldown              : "
    f"{COOLDOWN_STEPS}"
)

print(
    f"Request interval      : "
    f"{REQUEST_INTERVAL}"
)

print(
    "Attack                : none"
)

print(
    f"Paired seeds          : "
    f"{len(SEEDS)}"
)

print(
    f"Output directory      : "
    f"{OUTPUT_DIR}"
)

print("=" * 130)

print()


# ============================================================
# RUN
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
        seed
    )


    run_policy(
        "norep",
        seed
    )


# ============================================================
# DONE
# ============================================================

print("\n")

print("=" * 130)

print(
    "C23 SHADAI VS NO-REPUTATION RUNS COMPLETED"
)

print("=" * 130)

print()