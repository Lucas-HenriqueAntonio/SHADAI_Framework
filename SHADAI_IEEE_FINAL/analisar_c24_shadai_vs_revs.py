# analisar_c24_shadai_vs_revs.py
import math
import os
import pandas as pd

try:
    from scipy.stats import t
except ImportError:
    t=None

INPUT_DIR="c24_revs_benchmark"
OUTPUT_DIR=os.path.join(INPUT_DIR,"analysis")
os.makedirs(OUTPUT_DIR,exist_ok=True)

SEEDS=list(range(1,31))

PER_SEED=os.path.join(OUTPUT_DIR,"c24_shadai_vs_revs_por_seed.csv")
AGG=os.path.join(OUTPUT_DIR,"c24_shadai_vs_revs_multiseed.csv")
PAIRED=os.path.join(OUTPUT_DIR,"c24_shadai_minus_revs_paired.csv")
PLOT_READY=os.path.join(OUTPUT_DIR,"c24_plot_ready.csv")
LATEX=os.path.join(OUTPUT_DIR,"tabela_c24_shadai_vs_revs.tex")

def to_bool(s):
    return s.astype(str).str.strip().str.lower().isin(["true","1"])

def ci95(values):
    s=pd.Series(values,dtype="float64").dropna()
    n=len(s)
    if n==0: return (float("nan"),)*4
    m=s.mean()
    if n==1: return m,float("nan"),float("nan"),float("nan")
    sd=s.std(ddof=1); se=sd/math.sqrt(n)
    crit=t.ppf(.975,n-1) if t is not None else 1.96
    return m,sd,m-crit*se,m+crit*se

def jain(counts):
    s=pd.Series(counts,dtype="float64")
    if len(s)==0: return float("nan")
    den=len(s)*(s*s).sum()
    return (s.sum()**2)/den if den else float("nan")

def shadai_paths(seed):
    e=f"c24_shadai_seed_{seed:02d}"
    return (
        os.path.join(INPUT_DIR,f"requests_shadai_400_{e}.csv"),
        os.path.join(INPUT_DIR,f"metricas_shadai_400_{e}.csv"),
        os.path.join(INPUT_DIR,f"outcomes_shadai_400_{e}.csv"),
    )

def revs_paths(seed):
    e=f"c24_revs_seed_{seed:02d}"
    return (
        os.path.join(INPUT_DIR,f"requests_revs_400_{e}.csv"),
        os.path.join(INPUT_DIR,f"metricas_revs_400_{e}.csv"),
        os.path.join(INPUT_DIR,f"outcomes_revs_400_{e}.csv"),
    )

def require(paths):
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

def process_shadai(seed):
    rp,mp,op=shadai_paths(seed); require((rp,mp,op))
    req=pd.read_csv(rp); met=pd.read_csv(mp); out=pd.read_csv(op)

    allocated=to_bool(req["allocated"])
    active=pd.to_numeric(req["active_vehicles"],errors="coerce").fillna(0)
    n=len(req); alloc=int(allocated.sum())
    native_active=int((active>0).sum())
    ge2=int((active>=2).sum())

    counts=met["veiculo"].astype(str).value_counts() if len(met) else pd.Series(dtype=float)
    success=to_bool(out["success"]).mean() if len(out) else float("nan")

    return {
        "method":"SHADAI",
        "seed":seed,
        "requests":n,
        "allocations":alloc,
        "allocation_rate":alloc/n if n else float("nan"),
        "conditional_allocation_native":alloc/native_active if native_active else float("nan"),
        "conditional_allocation_ge2":alloc/ge2 if ge2 else float("nan"),
        "avg_selected_reputation_norm":pd.to_numeric(met["reputacao_norm"],errors="coerce").mean() if len(met) else float("nan"),
        "avg_available_energy_at_selection":pd.to_numeric(met["energia"],errors="coerce").mean() if len(met) else float("nan"),
        "unique_selected_vehicles":len(counts),
        "tasks_per_selected_vehicle":counts.mean() if len(counts) else float("nan"),
        "jain_selected_vehicles":jain(counts.values),
        "task_success_rate":success,
        # Method-specific, retained for diagnostics only:
        "avg_method_distance":pd.to_numeric(met["distancia"],errors="coerce").mean() if len(met) else float("nan"),
        "avg_revs_stay_time":float("nan"),
        "avg_revs_score":float("nan"),
        "avg_revs_candidates":float("nan"),
        "no_candidate_rate":((req["failure_reason"].astype(str)=="no_eligible_candidates").mean() if n else float("nan")),
    }

def process_revs(seed):
    rp,mp,op=revs_paths(seed); require((rp,mp,op))
    req=pd.read_csv(rp); met=pd.read_csv(mp); out=pd.read_csv(op)

    allocated=to_bool(req["allocated"])
    active=pd.to_numeric(req["active_vehicles"],errors="coerce").fillna(0)
    n=len(req); alloc=int(allocated.sum())
    native_active=int((active>=2).sum())
    ge2=native_active

    counts=met["veiculo"].astype(str).value_counts() if len(met) else pd.Series(dtype=float)
    success=to_bool(out["success"]).mean() if len(out) else float("nan")

    return {
        "method":"REVS-based",
        "seed":seed,
        "requests":n,
        "allocations":alloc,
        "allocation_rate":alloc/n if n else float("nan"),
        "conditional_allocation_native":alloc/native_active if native_active else float("nan"),
        "conditional_allocation_ge2":alloc/ge2 if ge2 else float("nan"),
        "avg_selected_reputation_norm":pd.to_numeric(met["reputacao_norm"],errors="coerce").mean() if len(met) else float("nan"),
        "avg_available_energy_at_selection":pd.to_numeric(met["energia"],errors="coerce").mean() if len(met) else float("nan"),
        "unique_selected_vehicles":len(counts),
        "tasks_per_selected_vehicle":counts.mean() if len(counts) else float("nan"),
        "jain_selected_vehicles":jain(counts.values),
        "task_success_rate":success,
        # REVS-specific geometry:
        "avg_method_distance":pd.to_numeric(met["distancia"],errors="coerce").mean() if len(met) else float("nan"),
        "avg_revs_stay_time":pd.to_numeric(met["stay_time"],errors="coerce").mean() if len(met) else float("nan"),
        "avg_revs_score":pd.to_numeric(met["revs_score"],errors="coerce").mean() if len(met) else float("nan"),
        "avg_revs_candidates":pd.to_numeric(req["eligible_revs"],errors="coerce").mean() if n else float("nan"),
        "no_candidate_rate":((req["failure_reason"].astype(str)=="no_revs_candidate").mean() if n else float("nan")),
    }

rows=[]; missing=[]
for seed in SEEDS:
    try: rows.append(process_shadai(seed))
    except FileNotFoundError as e: missing.append(("SHADAI",seed,str(e)))
    try: rows.append(process_revs(seed))
    except FileNotFoundError as e: missing.append(("REVS-based",seed,str(e)))

df=pd.DataFrame(rows)
if df.empty: raise RuntimeError("No C24 runs found.")
df.to_csv(PER_SEED,index=False)

print("="*170)
print("C24 - SHADAI VS REVS-BASED PROVIDER SELECTION")
print("="*170)
for method in ["SHADAI","REVS-based"]:
    print(f"{method:<12}: {len(df[df.method==method])}/30")
if missing: print(f"[WARNING] missing runs: {len(missing)}")

METRICS=[
    "allocation_rate",
    "conditional_allocation_native",
    "conditional_allocation_ge2",
    "avg_selected_reputation_norm",
    "avg_available_energy_at_selection",
    "unique_selected_vehicles",
    "tasks_per_selected_vehicle",
    "jain_selected_vehicles",
    "task_success_rate",
    "avg_revs_stay_time",
    "avg_revs_candidates",
    "no_candidate_rate",
]

agg=[]
for method in ["SHADAI","REVS-based"]:
    sub=df[df.method==method]
    row={"method":method,"n_runs":len(sub)}
    for m in METRICS:
        mean,sd,lo,hi=ci95(sub[m])
        row[f"{m}_mean"]=mean; row[f"{m}_sd"]=sd; row[f"{m}_ci95_low"]=lo; row[f"{m}_ci95_high"]=hi
    agg.append(row)

agg=pd.DataFrame(agg)
agg.to_csv(AGG,index=False)

print("\nMULTI-SEED RESULTS")
print("-"*170)
print(f"{'METHOD':<12}{'ALLOC%':>10}{'COND-N%':>10}{'COND-2%':>10}{'REP':>10}{'ENERGY':>10}{'UNIQUE':>10}{'JAIN':>10}{'SUCCESS%':>11}{'NO-CAND%':>11}")
for _,r in agg.iterrows():
    print(
        f"{r['method']:<12}"
        f"{r['allocation_rate_mean']*100:>10.2f}"
        f"{r['conditional_allocation_native_mean']*100:>10.2f}"
        f"{r['conditional_allocation_ge2_mean']*100:>10.2f}"
        f"{r['avg_selected_reputation_norm_mean']:>10.4f}"
        f"{r['avg_available_energy_at_selection_mean']:>10.2f}"
        f"{r['unique_selected_vehicles_mean']:>10.2f}"
        f"{r['jain_selected_vehicles_mean']:>10.4f}"
        f"{r['task_success_rate_mean']*100:>11.2f}"
        f"{r['no_candidate_rate_mean']*100:>11.2f}"
    )

# Only directly comparable metrics enter paired inferential comparison.
COMPARABLE=[
    "allocation_rate",
    "conditional_allocation_ge2",
    "avg_selected_reputation_norm",
    "avg_available_energy_at_selection",
    "unique_selected_vehicles",
    "tasks_per_selected_vehicle",
    "jain_selected_vehicles",
    "task_success_rate",
]

s=df[df.method=="SHADAI"]
r=df[df.method=="REVS-based"]
paired=pd.merge(s,r,on="seed",suffixes=("_shadai","_revs"))

prows=[]
print("\nPAIRED DIFFERENCES: SHADAI - REVS-BASED")
print("-"*170)
for m in COMPARABLE:
    d=paired[f"{m}_shadai"]-paired[f"{m}_revs"]
    mean,sd,lo,hi=ci95(d)
    ex=(lo>0 or hi<0) if pd.notna(lo) and pd.notna(hi) else False
    prows.append({"metric":m,"n_pairs":len(d.dropna()),"mean_difference_shadai_minus_revs":mean,"sd_difference":sd,"ci95_low":lo,"ci95_high":hi,"ci95_excludes_zero":ex})
    print(f"{m:<40} diff={mean:+.5f} | CI95=[{lo:+.5f}, {hi:+.5f}] | excludes zero={ex}")

pd.DataFrame(prows).to_csv(PAIRED,index=False)

# Long-form file for later figures. Only comparable metrics are included.
plot_rows=[]
for _,row in df.iterrows():
    for m in COMPARABLE:
        plot_rows.append({"method":row["method"],"seed":row["seed"],"metric":m,"value":row[m]})
pd.DataFrame(plot_rows).to_csv(PLOT_READY,index=False)

# Compact LaTeX table.
A=agg.set_index("method")
latex=[
r"\begin{table}[t]",
r"\centering",
r"\caption{Comparison between SHADAI and the REVS-based provider-selection benchmark over 30 paired seeds.}",
r"\label{tab:c24_revs}",
r"\begin{tabular}{lccccc}",
r"\hline",
r"Method & Allocation & Cond. Alloc. & Reputation & Energy & Jain \\",
r"\hline",
f"SHADAI & {A.loc['SHADAI','allocation_rate_mean']*100:.2f}\\% & {A.loc['SHADAI','conditional_allocation_ge2_mean']*100:.2f}\\% & {A.loc['SHADAI','avg_selected_reputation_norm_mean']:.4f} & {A.loc['SHADAI','avg_available_energy_at_selection_mean']:.2f} & {A.loc['SHADAI','jain_selected_vehicles_mean']:.4f} \\\\",
f"REVS-based & {A.loc['REVS-based','allocation_rate_mean']*100:.2f}\\% & {A.loc['REVS-based','conditional_allocation_ge2_mean']*100:.2f}\\% & {A.loc['REVS-based','avg_selected_reputation_norm_mean']:.4f} & {A.loc['REVS-based','avg_available_energy_at_selection_mean']:.2f} & {A.loc['REVS-based','jain_selected_vehicles_mean']:.4f} \\\\",
r"\hline",
r"\end{tabular}",
r"\end{table}",
]
with open(LATEX,"w",encoding="utf-8") as f: f.write("\n".join(latex))

print("\nIMPORTANT INTERPRETATION NOTE")
print("-"*170)
print("Do NOT compare raw method scores: SHADAI score and REVS trust score have different definitions.")
print("Do NOT directly compare 'distance': SHADAI distance is vehicle-to-task; REVS distance is requester-to-provider.")
print("Use c24_plot_ready.csv for later common-metric figures.")

print("\nOUTPUTS")
for p in [PER_SEED,AGG,PAIRED,PLOT_READY,LATEX]: print(p)
