# run_scalability_final.py
# Final 30-seed scalability campaign for SHADAI.
# Explicitly isolates the 100/400/800 active-vehicle caps.

import csv
import os
import subprocess
import sys

SHADAI_SCRIPT = "shadai_sumo.py"
OUTPUT_DIR = "scalability_final"
SEEDS = list(range(1, 31))
VEHICLE_CAPS = [100, 400, 800]

LAMBDA = (0.5, 0.3, 0.2)
WEIGHTS = (0.4, 0.3, 0.3)
EPSILON = 0.2
COOLDOWN = 40
REQUEST_INTERVAL = 20

# The final integrated SHADAI baseline uses dynamic pricing.
PRICING_MODE = "dynamic"
PRICING_ALPHA = 0.20
PRICING_BETA = 0.20
PRICING_REFERENCE_INTERVAL = 20

EXPECTED_RNG = "split_v2"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def experiment_name(max_vehicles, seed):
    return f"scalability_v{max_vehicles}_seed_{seed:02d}"


def metrics_path(max_vehicles, seed):
    exp = experiment_name(max_vehicles, seed)
    return os.path.join(
        OUTPUT_DIR,
        f"metricas_shadai_{max_vehicles}_{exp}.csv",
    )


def compatible(path, max_vehicles, seed):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False

    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            row = next(csv.DictReader(f), None)

        if row is None:
            return False

        checks = [
            int(float(row.get("seed", -1))) == seed,
            int(float(row.get("max_vehicles", -1))) == max_vehicles,
            row.get("rng_scheme", "") == EXPECTED_RNG,
            row.get("pricing_mode", "") == PRICING_MODE,
            row.get("attack_type", "none") == "none",
            abs(float(row.get("malicious_ratio", 0.0))) < 1e-12,
        ]

        return all(checks)

    except Exception:
        return False


def run_one(max_vehicles, seed):
    path = metrics_path(max_vehicles, seed)

    if compatible(path, max_vehicles, seed):
        print(f"[SKIP] vehicles={max_vehicles} seed={seed:02d}")
        return

    exp = experiment_name(max_vehicles, seed)

    cmd = [
        sys.executable,
        SHADAI_SCRIPT,
        "--max-vehicles", str(max_vehicles),
        "--lambda1", str(LAMBDA[0]),
        "--lambda2", str(LAMBDA[1]),
        "--lambda3", str(LAMBDA[2]),
        "--w1", str(WEIGHTS[0]),
        "--w2", str(WEIGHTS[1]),
        "--w3", str(WEIGHTS[2]),
        "--epsilon", str(EPSILON),
        "--cooldown-steps", str(COOLDOWN),
        "--request-interval", str(REQUEST_INTERVAL),
        "--pricing-mode", PRICING_MODE,
        "--pricing-alpha", str(PRICING_ALPHA),
        "--pricing-beta", str(PRICING_BETA),
        "--pricing-reference-interval", str(PRICING_REFERENCE_INTERVAL),
        "--attack-type", "none",
        "--malicious-ratio", "0.0",
        "--flood-intensity", "0",
        "--seed", str(seed),
        "--experiment", exp,
        "--output-dir", OUTPUT_DIR,
        "--headless",
    ]

    print(f"[RUN ] vehicles={max_vehicles} seed={seed:02d}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    print("=" * 90)
    print("FINAL SCALABILITY CAMPAIGN")
    print("Caps: 100 / 400 / 800 | 30 seeds | dynamic pricing | split_v2")
    print("=" * 90)

    for max_vehicles in VEHICLE_CAPS:
        for seed in SEEDS:
            run_one(max_vehicles, seed)

    print("\n[DONE] scalability_final")
