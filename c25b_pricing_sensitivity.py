# c25b_pricing_sensitivity.py

import csv
import os
import subprocess
import sys


# ============================================================
# CONFIGURATION
# ============================================================

SHADAI_SCRIPT = "shadai_sumo.py"

OUTPUT_DIR = "pricing_sensitivity"

SEEDS = list(range(1, 31))

LAMBDA1 = 0.5
LAMBDA2 = 0.3
LAMBDA3 = 0.2

W1 = 0.4
W2 = 0.3
W3 = 0.3

EPSILON = 0.2
COOLDOWN_STEPS = 40

PRICING_REFERENCE_INTERVAL = 20

RNG_SCHEME = "split_v1"
PRICING_SCHEME = "c25_supply_demand_dynamic_v1"


# ============================================================
# EXPERIMENTAL DESIGN
# ============================================================
#
# Block 1: beta sensitivity (supply scarcity)
#   request_interval = 20
#   D_t = 0
#   alpha fixed at 0.20
#   beta = 0.10, 0.20, 0.30
#
# Block 2: alpha sensitivity (demand pressure)
#   request_interval = 10
#   D_t = 1
#   beta fixed at 0.20
#   alpha = 0.10, 0.20, 0.30
#
# The design isolates the effect of each coefficient instead of
# varying alpha and beta jointly.
# ============================================================

CONFIGS = [
    {
        "block": "beta",
        "label": "beta_010",
        "alpha": 0.20,
        "beta": 0.10,
        "request_interval": 20,
    },
    {
        "block": "beta",
        "label": "beta_020",
        "alpha": 0.20,
        "beta": 0.20,
        "request_interval": 20,
    },
    {
        "block": "beta",
        "label": "beta_030",
        "alpha": 0.20,
        "beta": 0.30,
        "request_interval": 20,
    },

    {
        "block": "alpha",
        "label": "alpha_010",
        "alpha": 0.10,
        "beta": 0.20,
        "request_interval": 10,
    },
    {
        "block": "alpha",
        "label": "alpha_020",
        "alpha": 0.20,
        "beta": 0.20,
        "request_interval": 10,
    },
    {
        "block": "alpha",
        "label": "alpha_030",
        "alpha": 0.30,
        "beta": 0.20,
        "request_interval": 10,
    },
]


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# HELPERS
# ============================================================

def experiment_name(config, seed):
    return (
        f"c25b_{config['label']}_"
        f"seed_{seed:02d}"
    )


def metrics_path(experiment):
    return os.path.join(
        OUTPUT_DIR,
        f"metricas_shadai_400_{experiment}.csv",
    )


def compatible_float(value, target):
    try:
        return abs(float(value) - target) < 1e-12
    except Exception:
        return False


def run_is_compatible(
    path,
    config,
    seed,
):
    if not os.path.exists(path):
        return False

    try:
        with open(
            path,
            "r",
            newline="",
            encoding="utf-8",
        ) as f:
            reader = csv.DictReader(f)
            first = next(reader, None)

            if first is None:
                return False

            return (
                int(float(first.get("seed", -1)))
                == seed

                and first.get("rng_scheme", "")
                == RNG_SCHEME

                and first.get("pricing_scheme", "")
                == PRICING_SCHEME

                and first.get("pricing_mode", "")
                == "dynamic"

                and compatible_float(
                    first.get("pricing_alpha", -1),
                    config["alpha"],
                )

                and compatible_float(
                    first.get("pricing_beta", -1),
                    config["beta"],
                )

                and int(
                    float(
                        first.get(
                            "pricing_reference_interval",
                            -1,
                        )
                    )
                )
                == PRICING_REFERENCE_INTERVAL

                and int(
                    float(
                        first.get(
                            "request_interval",
                            -1,
                        )
                    )
                )
                == config["request_interval"]

                and compatible_float(
                    first.get("epsilon", -1),
                    EPSILON,
                )

                and first.get("attack_type", "none")
                == "none"

                and compatible_float(
                    first.get("malicious_ratio", 0),
                    0.0,
                )

                and int(
                    float(
                        first.get("flood_intensity", 0)
                    )
                )
                == 0
            )

    except Exception:
        return False


def run_config(config, seed):
    experiment = experiment_name(
        config,
        seed,
    )

    path = metrics_path(
        experiment
    )

    if run_is_compatible(
        path,
        config,
        seed,
    ):
        print(
            f"[SKIP] "
            f"{config['label']:<10} "
            f"seed={seed:02d}"
        )
        return

    print(
        f"[RUN ] "
        f"{config['label']:<10} "
        f"seed={seed:02d} "
        f"alpha={config['alpha']:.2f} "
        f"beta={config['beta']:.2f} "
        f"interval={config['request_interval']}"
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
        str(EPSILON),

        "--cooldown-steps",
        str(COOLDOWN_STEPS),

        "--request-interval",
        str(config["request_interval"]),

        "--pricing-mode",
        "dynamic",

        "--pricing-alpha",
        str(config["alpha"]),

        "--pricing-beta",
        str(config["beta"]),

        "--pricing-reference-interval",
        str(PRICING_REFERENCE_INTERVAL),

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

        "--headless",
    ]

    result = subprocess.run(
        cmd,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Run failed: "
            f"{config['label']} "
            f"seed={seed}. "
            f"Return code={result.returncode}"
        )


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 140)
print("C25b - DYNAMIC PRICING PARAMETER SENSITIVITY")
print("=" * 140)

print(
    "Block beta : request interval=20, "
    "alpha=0.20, beta={0.10,0.20,0.30}"
)

print(
    "Block alpha: request interval=10, "
    "beta=0.20, alpha={0.10,0.20,0.30}"
)

print(
    f"Paired seeds per configuration : "
    f"{len(SEEDS)}"
)

print(
    f"Total runs                    : "
    f"{len(CONFIGS) * len(SEEDS)}"
)

print(
    "Attacks                       : disabled"
)

print(
    f"Output directory              : "
    f"{OUTPUT_DIR}"
)

print("=" * 140)


for config in CONFIGS:
    print()
    print("=" * 140)

    print(
        f"BLOCK={config['block']} | "
        f"{config['label']} | "
        f"alpha={config['alpha']:.2f} | "
        f"beta={config['beta']:.2f} | "
        f"interval={config['request_interval']}"
    )

    print("=" * 140)

    for seed in SEEDS:
        run_config(
            config,
            seed,
        )


print()
print("=" * 140)
print("C25b RUNS COMPLETED")
print("=" * 140)
print()
