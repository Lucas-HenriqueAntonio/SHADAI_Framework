# run_final_plots_v2.py
# Does not modify simulations or analyses. It only regenerates figures.

import subprocess
import sys

SCRIPTS = [
    "graficos_principais_tnsm_v2.py",
    "graficos_suplementares_tnsm_v2.py",
]

for script in SCRIPTS:
    print("\n" + "="*92)
    print("[PLOTS V2]", script)
    print("="*92)
    subprocess.run([sys.executable, script], check=True)

print("\n[DONE] figures_final_v2 generated successfully.")
