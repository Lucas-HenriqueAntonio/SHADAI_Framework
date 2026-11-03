# run_final_plots.py
# Generates all final figure families after analysis.

import subprocess
import sys

SCRIPTS = [
    "graficos.py",
    "gera_graficos.py",
    "graficos_maliciosos.py",
    "graficos_pricing.py",
    "graficos_sensibilidade.py",
    "graficos_ablations.py",
]

for script in SCRIPTS:
    print("\n" + "=" * 100)
    print(f"[PLOTS] {script}")
    print("=" * 100)
    subprocess.run([sys.executable, script], check=True)

print("\n[DONE] Final figures are under figures_final/.")
