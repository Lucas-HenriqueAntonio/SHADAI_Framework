# c26_dynamic_pricing_stability.py

import csv
import os
import subprocess
import sys

SHADAI_SCRIPT = "shadai_sumo.py"
OUTPUT_DIR = "pricing_stability"
SEEDS = list(range(1, 31))
REQUEST_INTERVALS = [60, 40, 20, 10]

LAMBDA1, LAMBDA2, LAMBDA3 = 0.5, 0.3, 0.2
W1, W2, W3 = 0.4, 0.3, 0.3
EPSILON = 0.2
COOLDOWN_STEPS = 40

PRICING_ALPHA = 0.20
PRICING_BETA = 0.20
PRICING_REFERENCE_INTERVAL = 20

RNG_SCHEME = "split_v1"
PRICING_SCHEME = "c25_supply_demand_dynamic_v1"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def label_for_interval(interval):
    return f"ri_{interval:02d}"


def experiment_name(interval, seed):
    return f"c26_{label_for_interval(interval)}_seed_{seed:02d}"


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


def run_is_compatible(path, interval, seed):
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
                and first.get("pricing_mode", "") == "dynamic"
                and compatible_float(first.get("pricing_alpha", -1), PRICING_ALPHA)
                and compatible_float(first.get("pricing_beta", -1), PRICING_BETA)
                and int(float(first.get("pricing_reference_interval", -1)))
                    == PRICING_REFERENCE_INTERVAL
                and int(float(first.get("request_interval", -1))) == interval
                and compatible_float(first.get("epsilon", -1), EPSILON)
                and first.get("attack_type", "none") == "none"
                and compatible_float(first.get("malicious_ratio", 0), 0.0)
                and int(float(first.get("flood_intensity", 0))) == 0
            )
    except Exception:
        return False


def run_config(interval, seed):
    experiment = experiment_name(interval, seed)
    path = metrics_path(experiment)

    if run_is_compatible(path, interval, seed):
        print(f"[SKIP] interval={interval:<2} seed={seed:02d}")
        return

    print(f"[RUN ] interval={interval:<2} seed={seed:02d}")

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
        "--request-interval", str(interval),
        "--pricing-mode", "dynamic",
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
            f"C26 failed for interval={interval}, seed={seed}. "
            f"Return code={result.returncode}"
        )


print()
print("=" * 140)
print("C26 - DYNAMIC PRICING STABILITY")
print("=" * 140)
print(f"Request intervals        : {REQUEST_INTERVALS}")
print(f"Seeds per interval       : {len(SEEDS)}")
print(f"Total runs               : {len(REQUEST_INTERVALS) * len(SEEDS)}")
print(f"Pricing alpha            : {PRICING_ALPHA}")
print(f"Pricing beta             : {PRICING_BETA}")
print(f"Reference interval       : {PRICING_REFERENCE_INTERVAL}")
print("Attacks                  : disabled")
print(f"Output directory         : {OUTPUT_DIR}")
print("=" * 140)

for interval in REQUEST_INTERVALS:
    print()
    print("=" * 140)
    print(f"REQUEST INTERVAL = {interval}")
    print("=" * 140)

    for seed in SEEDS:
        run_config(interval, seed)

print()
print("=" * 140)
print("C26 RUNS COMPLETED")
print("=" * 140)
print()
