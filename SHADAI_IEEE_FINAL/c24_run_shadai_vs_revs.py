# c24_run_shadai_vs_revs.py
import csv
import os
import subprocess
import sys

SHADAI_SCRIPT="shadai_sumo.py"
REVS_SCRIPT="revs_sumo.py"
OUTPUT_DIR="c24_revs_benchmark"
SEEDS=list(range(1,31))

# Final SHADAI configuration
L1,L2,L3=.5,.3,.2
W1,W2,W3=.4,.3,.3
EPSILON=.2
COOLDOWN=40
REQUEST_INTERVAL=20

# Final dynamic pricing configuration
PRICING_MODE="dynamic"
PRICING_ALPHA=.20
PRICING_BETA=.20
PRICING_REFERENCE_INTERVAL=20

# REVS paper parameters
REVS_W1=.7
REVS_W2=.3
REPUTATION_THRESHOLD=.5
COMMUNICATION_RADIUS=250.
DIRECTION_TOLERANCE_DEG=90.
STAY_TIME_MODE="raw"

os.makedirs(OUTPUT_DIR,exist_ok=True)

def run(cmd,label):
    print(f"[RUN ] {label}")
    r=subprocess.run(cmd,check=False)
    if r.returncode!=0:
        raise RuntimeError(f"{label} failed with return code {r.returncode}")

def shadai_exp(seed): return f"c24_shadai_seed_{seed:02d}"
def revs_exp(seed): return f"c24_revs_seed_{seed:02d}"

def existing_nonempty(path):
    return os.path.exists(path) and os.path.getsize(path)>0

def run_shadai(seed):
    exp=shadai_exp(seed)
    path=os.path.join(OUTPUT_DIR,f"metricas_shadai_400_{exp}.csv")
    if existing_nonempty(path):
        print(f"[SKIP] SHADAI seed={seed:02d}")
        return
    cmd=[
        sys.executable,SHADAI_SCRIPT,
        "--lambda1",str(L1),"--lambda2",str(L2),"--lambda3",str(L3),
        "--w1",str(W1),"--w2",str(W2),"--w3",str(W3),
        "--epsilon",str(EPSILON),
        "--cooldown-steps",str(COOLDOWN),
        "--request-interval",str(REQUEST_INTERVAL),
        "--pricing-mode",PRICING_MODE,
        "--pricing-alpha",str(PRICING_ALPHA),
        "--pricing-beta",str(PRICING_BETA),
        "--pricing-reference-interval",str(PRICING_REFERENCE_INTERVAL),
        "--malicious-ratio","0.0",
        "--attack-type","none",
        "--flood-intensity","0",
        "--seed",str(seed),
        "--experiment",exp,
        "--output-dir",OUTPUT_DIR,
        "--headless",
    ]
    run(cmd,f"SHADAI seed={seed:02d}")

def run_revs(seed):
    exp=revs_exp(seed)
    path=os.path.join(OUTPUT_DIR,f"metricas_revs_400_{exp}.csv")
    if existing_nonempty(path):
        print(f"[SKIP] REVS seed={seed:02d}")
        return
    cmd=[
        sys.executable,REVS_SCRIPT,
        "--lambda1",str(L1),"--lambda2",str(L2),"--lambda3",str(L3),
        "--request-interval",str(REQUEST_INTERVAL),
        "--pricing-mode",PRICING_MODE,
        "--pricing-alpha",str(PRICING_ALPHA),
        "--pricing-beta",str(PRICING_BETA),
        "--pricing-reference-interval",str(PRICING_REFERENCE_INTERVAL),
        "--revs-w1",str(REVS_W1),
        "--revs-w2",str(REVS_W2),
        "--reputation-threshold",str(REPUTATION_THRESHOLD),
        "--communication-radius",str(COMMUNICATION_RADIUS),
        "--direction-tolerance-deg",str(DIRECTION_TOLERANCE_DEG),
        "--stay-time-mode",STAY_TIME_MODE,
        "--seed",str(seed),
        "--experiment",exp,
        "--output-dir",OUTPUT_DIR,
        "--headless",
    ]
    run(cmd,f"REVS seed={seed:02d}")

print("="*120)
print("C24 - SHADAI vs REVS-BASED PROVIDER SELECTION")
print("="*120)
print(f"30 paired seeds | pricing={PRICING_MODE} | alpha={PRICING_ALPHA} | beta={PRICING_BETA}")
print("SHADAI source code is not modified by this runner.")
print("REVS parameters: w=(0.7,0.3), Rth=0.5, radius=250 m, raw stay time.")
print("="*120)

for seed in SEEDS:
    print(f"\n--- paired seed {seed:02d} ---")
    run_shadai(seed)
    run_revs(seed)

print("\nC24 runs completed.")
