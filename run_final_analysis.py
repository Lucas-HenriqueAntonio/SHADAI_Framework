# run_final_analysis.py
# Final analysis orchestration.
# Continues through all analyzers so one failure does not hide later issues.

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "analisar_scalability_final.py",
    "analisar_ablation_lambda_multiseed.py",
    "analisar_ablation_weights_multiseed.py",
    "analisar_ablation_epsilon_multiseed.py",
    "analisar_ablation_load_multiseed.py",
    "analisar_capacity_multiseed.py",
    "analisar_runtime_multiseed.py",
    "analisar_malicious_ratio_multiseed.py",
    "analisar_flood_multiseed.py",
    "analisar_c22_shadai_vs_greedy.py",
    "analisar_c23_shadai_vs_norep.py",
    "analisar_c24_shadai_vs_revs.py",
    "analisar_c25_pricing.py",
    "analisar_c25b_pricing_sensitivity.py",
    "analisar_c26_pricing_stability.py",
]

failures = []

for script in SCRIPTS:
    print("\n" + "=" * 100)
    print(f"[ANALYSIS] {script}")
    print("=" * 100)

    if not Path(script).exists():
        failures.append((script, "missing_script"))
        print(f"[FAIL] Script not found: {script}")
        continue

    result = subprocess.run([sys.executable, script], check=False)

    if result.returncode != 0:
        failures.append((script, f"exit_code={result.returncode}"))
        print(f"[FAIL] {script} -> exit code {result.returncode}")
    else:
        print(f"[PASS] {script}")

print("\n" + "=" * 100)
print("FINAL ANALYSIS SUMMARY")
print("=" * 100)
print(f"Analyzers expected : {len(SCRIPTS)}")
print(f"Analyzers passed   : {len(SCRIPTS) - len(failures)}")
print(f"Analyzers failed   : {len(failures)}")

if failures:
    print("\nFailures:")
    for script, reason in failures:
        print(f" - {script}: {reason}")
    print("\nAnalysis phase completed with failures.")
    sys.exit(1)

print("\nAll final analyzers completed successfully.")
