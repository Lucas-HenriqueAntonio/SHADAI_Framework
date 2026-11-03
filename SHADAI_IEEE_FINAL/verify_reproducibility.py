# verify_reproducibility.py
# Runs the same final SHADAI configuration twice with the same seed and
# checks stochastic/scientific outputs after excluding wall-clock timings.

from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd

BASE = Path("reproducibility_check")
RUN_A = BASE / "run_a"
RUN_B = BASE / "run_b"

SEED = 7
EXP_A = "repro_a"
EXP_B = "repro_b"

IGNORE_COLUMNS = {
    "experimento",
    "experiment",
    "offer_runtime_ms",
    "selection_runtime_ms",
    "decision_runtime_ms",
}

for p in [RUN_A, RUN_B]:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def run(out_dir, experiment):
    cmd = [
        sys.executable,
        "shadai_sumo.py",
        "--max-vehicles", "400",
        "--lambda1", "0.5",
        "--lambda2", "0.3",
        "--lambda3", "0.2",
        "--w1", "0.4",
        "--w2", "0.3",
        "--w3", "0.3",
        "--epsilon", "0.2",
        "--cooldown-steps", "40",
        "--request-interval", "20",
        "--pricing-mode", "dynamic",
        "--pricing-alpha", "0.2",
        "--pricing-beta", "0.2",
        "--pricing-reference-interval", "20",
        "--attack-type", "none",
        "--malicious-ratio", "0.0",
        "--flood-intensity", "0",
        "--seed", str(SEED),
        "--experiment", experiment,
        "--output-dir", str(out_dir),
        "--headless",
    ]
    subprocess.run(cmd, check=True)


def canonical(path):
    df = pd.read_csv(path)

    drop = [
        c for c in df.columns
        if c in IGNORE_COLUMNS or "runtime" in c.lower()
    ]

    if drop:
        df = df.drop(columns=drop)

    return df.reset_index(drop=True)


run(RUN_A, EXP_A)
run(RUN_B, EXP_B)

prefixes = [
    "metricas",
    "eligibilidade",
    "requests",
    "vehicles",
    "outcomes",
    "flood",
    "summary",
]

failures = []

for prefix in prefixes:
    a_files = list(RUN_A.glob(f"{prefix}_*.csv"))
    b_files = list(RUN_B.glob(f"{prefix}_*.csv"))

    if len(a_files) != 1 or len(b_files) != 1:
        failures.append(f"{prefix}: missing/ambiguous output")
        continue

    a = canonical(a_files[0])
    b = canonical(b_files[0])

    try:
        pd.testing.assert_frame_equal(
            a,
            b,
            check_dtype=False,
            check_exact=True,
        )
        print(f"[PASS] {prefix}")
    except AssertionError as exc:
        failures.append(f"{prefix}: {exc}")
        print(f"[FAIL] {prefix}")

if failures:
    print("\nREPRODUCIBILITY CHECK FAILED")
    for item in failures:
        print(" -", item)
    raise SystemExit(1)

print("\nREPRODUCIBILITY CHECK PASSED")
print("Same seed produced identical scientific/stochastic outputs.")
print("Wall-clock runtime fields were intentionally excluded.")
