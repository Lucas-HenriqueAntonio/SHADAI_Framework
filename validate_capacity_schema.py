# validate_capacity_schema.py
from pathlib import Path
import pandas as pd

ROOT = Path("capacity_ablation")
CONFIGS = [
    ("capacity_low", 60),
    ("capacity_moderate", 40),
    ("capacity_high", 20),
    ("capacity_very_high", 10),
]

required = {
    "rng_scheme",
    "telemetry_scheme",
    "request_interval",
    "step",
    "active_vehicles",
    "unique_eligible_candidates",
    "allocated",
    "failure_reason",
    "completion_step",
}

ok_all = True

for config, interval in CONFIGS:
    path = ROOT / (
        f"requests_shadai_400_{config}_"
        f"interval_{interval:03d}_seed_01.csv"
    )

    if not path.exists():
        print(f"[FAIL] missing: {path}")
        ok_all = False
        continue

    df = pd.read_csv(path)
    missing = sorted(required - set(df.columns))

    if missing:
        print(f"[FAIL] {config}: missing columns {missing}")
        ok_all = False
        continue

    print(
        f"[PASS] {config}: "
        f"rows={len(df)}, "
        f"rng={df['rng_scheme'].iloc[0]}, "
        f"telemetry={df['telemetry_scheme'].iloc[0]}, "
        f"eligible_col=unique_eligible_candidates"
    )

if ok_all:
    print("\nCAPACITY SCHEMA VALIDATION PASSED")
else:
    raise SystemExit("\nCAPACITY SCHEMA VALIDATION FAILED")
