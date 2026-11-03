# validate_final_outputs.py
# Diagnostic validator for the definitive SHADAI campaign.
# It does NOT modify files and does NOT rerun simulations.

from pathlib import Path
import csv

EXPECTED_TELEMETRY = "c18_flood_intensity_v1"
EXPECTED_RNG = "split_v2"

CHECKS = [
    ("scalability_final", 90, ("metricas_",)),
    ("lambda_ablation", 240, ("metricas_shadai_",)),
    ("weight_ablation", 240, ("metricas_shadai_",)),
    ("epsilon_ablation", 180, ("metricas_shadai_",)),
    ("cooldown_ablation", 150, ("metricas_shadai_",)),
    ("load_ablation", 120, ("metricas_shadai_",)),
    ("capacity_ablation", 120, ("metricas_shadai_",)),
    ("runtime_ablation", 120, ("runtime_", "metricas_shadai_")),
    ("malicious_ablation", 150, ("metricas_shadai_",)),
    ("flood_ablation", 150, ("metricas_shadai_",)),
    ("greedy_alignment", 60, ("metricas_",)),
    ("norep_alignment", 60, ("metricas_",)),
    ("c24_revs_benchmark", 60, ("metricas_",)),
    ("pricing_ablation", 60, ("metricas_shadai_",)),
    ("pricing_sensitivity", 180, ("metricas_shadai_",)),
    ("pricing_stability", 120, ("metricas_shadai_",)),
]

def first_row(path):
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader, None)
            return reader.fieldnames or [], row
    except Exception:
        return [], None

def candidate_primary_files(folder, prefixes):
    files = []
    for p in folder.glob("*.csv"):
        if any(p.name.startswith(prefix) for prefix in prefixes):
            # Exclude aggregate/analysis CSVs from primary run counting.
            lowered = p.name.lower()
            if any(token in lowered for token in [
                "multiseed", "por_seed", "pareado", "comparison",
                "comparacao", "resumo_ablation", "analysis", "source_used"
            ]):
                continue
            files.append(p)
    return sorted(files)

print("=" * 118)
print("FINAL CAMPAIGN OUTPUT VALIDATION")
print("=" * 118)
print(f"Expected RNG       : {EXPECTED_RNG}")
print(f"Expected telemetry : {EXPECTED_TELEMETRY}")
print()

overall_ok = True

for dirname, expected, prefixes in CHECKS:
    folder = Path(dirname)

    if not folder.exists():
        print(f"[MISSING DIR] {dirname}")
        overall_ok = False
        continue

    files = candidate_primary_files(folder, prefixes)

    # For runtime_ablation, prefer runtime_ files if they exist to avoid double-count.
    if dirname == "runtime_ablation":
        runtime_files = [p for p in files if p.name.startswith("runtime_")]
        if runtime_files:
            files = runtime_files

    rng_values = set()
    telemetry_values = set()
    empty = 0
    unreadable = 0

    for path in files:
        fields, row = first_row(path)
        if row is None:
            empty += 1
            continue

        if "rng_scheme" in fields:
            rng_values.add(str(row.get("rng_scheme", "")))

        if "telemetry_scheme" in fields:
            telemetry_values.add(str(row.get("telemetry_scheme", "")))

    count_ok = len(files) >= expected

    if rng_values and rng_values != {EXPECTED_RNG}:
        overall_ok = False

    # Some benchmark-specific files may not expose telemetry_scheme.
    if telemetry_values and telemetry_values != {EXPECTED_TELEMETRY}:
        overall_ok = False

    if not count_ok:
        overall_ok = False

    status = "PASS" if count_ok else "INCOMPLETE"

    print(
        f"[{status:<10}] {dirname:<24} "
        f"primary={len(files):>4}/{expected:<4} "
        f"empty={empty:<3} "
        f"rng={sorted(rng_values) if rng_values else '-'} "
        f"telemetry={sorted(telemetry_values) if telemetry_values else '-'}"
    )

print()
print("=" * 118)

if overall_ok:
    print("VALIDATION PASSED: expected final-campaign outputs are present/compatible.")
else:
    print("VALIDATION FOUND GAPS.")
    print("Do not rerun everything automatically.")
    print("Use the reported directory/count to rerun only the affected experiment runner.")

print("=" * 118)
