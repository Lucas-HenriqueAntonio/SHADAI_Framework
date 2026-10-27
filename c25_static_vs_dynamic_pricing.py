# c25_static_vs_dynamic_pricing.py

import csv
import os
import subprocess
import sys

SHADAI_SCRIPT = "shadai_sumo.py"
OUTPUT_DIR = "pricing_ablation"
SEEDS = list(range(1, 31))

LAMBDA1 = 0.5
LAMBDA2 = 0.3
LAMBDA3 = 0.2

W1 = 0.4
W2 = 0.3
W3 = 0.3

EPSILON = 0.2
COOLDOWN_STEPS = 40
REQUEST_INTERVAL = 20

PRICING_ALPHA = 0.20
PRICING_BETA = 0.20
PRICING_REFERENCE_INTERVAL = 20

RNG_SCHEME = "split_v1"
PRICING_SCHEME = "c25_supply_demand_dynamic_v1"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def experiment_name(mode, seed):
    return f"c25_{mode}_seed_{seed:02d}"


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


def run_is_compatible(path, seed, pricing_mode):
    if not os.path.exists(path):
        return False

    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            first = next(reader, None)

            if first is None:
                return False

            return (
                int(float(first.get("seed", -1))) == seed
                and first.get("rng_scheme", "") == RNG_SCHEME
                and first.get("pricing_scheme", "") == PRICING_SCHEME
                and first.get("pricing_mode", "") == pricing_mode
                and compatible_float(first.get("pricing_alpha", -1), PRICING_ALPHA)
                and compatible_float(first.get("pricing_beta", -1), PRICING_BETA)
                and int(float(first.get("pricing_reference_interval", -1)))
                == PRICING_REFERENCE_INTERVAL
                and compatible_float(first.get("epsilon", -1), EPSILON)
                and first.get("attack_type", "none") == "none"
                and compatible_float(first.get("malicious_ratio", 0), 0.0)
                and int(float(first.get("flood_intensity", 0))) == 0
            )
    except Exception:
        return False


def run_config(pricing_mode, seed):
    experiment = experiment_name(pricing_mode, seed)
    path = metrics_path(experiment)

    if run_is_compatible(path, seed, pricing_mode):
        print(
            f"[SKIP] {pricing_mode:<8} seed={seed:02d} "
            f"already exists and is compatible."
        )
        return

    print(f"[RUN ] {pricing_mode:<8} seed={seed:02d}")

    cmd = [
        sys.executable,
        SHADAI_SCRIPT,
        "--lambda1", str(LAMBDA1),
        "--lambda2", str(LAMBDA2),
        "--lambda3", str(LAMBDA3),
        "--w1", str(W1),
        "--w2", str(W2),
        "--w3", str(W3),
        "--epsilon", str(EPSILON),
        "--cooldown-steps", str(COOLDOWN_STEPS),
        "--request-interval", str(REQUEST_INTERVAL),
        "--pricing-mode", pricing_mode,
        "--pricing-alpha", str(PRICING_ALPHA),
        "--pricing-beta", str(PRICING_BETA),
        "--pricing-reference-interval", str(PRICING_REFERENCE_INTERVAL),
        "--malicious-ratio", "0.0",
        "--attack-type", "none",
        "--flood-intensity", "0",
        "--seed", str(seed),
        "--experiment", experiment,
        "--output-dir", OUTPUT_DIR,
        "--headless",
    ]

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        raise RuntimeError(
            f"{pricing_mode} failed for seed {seed}. "
            f"Return code = {result.returncode}"
        )


print()
print("=" * 130)
print("C25 - STATIC VS DYNAMIC PRICING")
print("=" * 130)
print(f"Seeds                       : {len(SEEDS)}")
print("Static mode                 : original service prices")
print(f"Dynamic alpha               : {PRICING_ALPHA}")
print(f"Dynamic beta                : {PRICING_BETA}")
print(f"Reference request interval  : {PRICING_REFERENCE_INTERVAL}")
print(f"Experiment request interval : {REQUEST_INTERVAL}")
print("Attack                      : disabled")
print(f"Output directory            : {OUTPUT_DIR}")
print("=" * 130)

for seed in SEEDS:
    print()
    print("-" * 130)
    print(f"PAIRED SEED {seed:02d}")
    print("-" * 130)

    run_config("static", seed)
    run_config("dynamic", seed)

print()
print("=" * 130)
print("C25 RUNS COMPLETED")
print("=" * 130)
print()
