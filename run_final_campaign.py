# run_final_campaign.py
# Master orchestration for the definitive SHADAI campaign.
#
# Usage:
#   python run_final_campaign.py --phase verify
#   python run_final_campaign.py --phase experiments
#   python run_final_campaign.py --phase analysis
#   python run_final_campaign.py --phase plots
#   python run_final_campaign.py --phase all
#
# The "experiments" phase is intentionally long.

import argparse
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument(
    "--phase",
    choices=["verify", "experiments", "analysis", "plots", "all"],
    default="verify",
)
args = parser.parse_args()

EXPERIMENTS = [
    # New isolated 100/400/800 campaign.
    "run_scalability_final.py",

    # Utility/score/policy sensitivities.
    "Ablation_Lambda.py",
    "ablation_weights.py",
    "ablation_epsilon.py",
    "ablation_cooldown.py",
    "ablation_load.py",
    "ablation_capacity.py",
    "ablation_runtime.py",

    # Security.
    "ablation_malicious_ratio.py",
    "ablation_flood_intensity.py",

    # Strategy / literature baselines.
    "c22_shadai_vs_greedy.py",
    "c23_shadai_vs_norep.py",
    "c24_run_shadai_vs_revs.py",

    # Pricing.
    "c25_static_vs_dynamic_pricing.py",
    "c25b_pricing_sensitivity.py",
    "c26_dynamic_pricing_stability.py",
]


def call(script):
    print("\n" + "=" * 110)
    print(f"[RUN] {script}")
    print("=" * 110)
    subprocess.run([sys.executable, script], check=True)


if args.phase in ["verify", "all"]:
    call("verify_reproducibility.py")

if args.phase in ["experiments", "all"]:
    for script in EXPERIMENTS:
        call(script)

if args.phase in ["analysis", "all"]:
    call("run_final_analysis.py")

if args.phase in ["plots", "all"]:
    call("run_final_plots.py")

print(f"\n[DONE] phase={args.phase}")
