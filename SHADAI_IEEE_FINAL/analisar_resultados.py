import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("metricas_shadai_400.csv")

print("\nDataset loaded successfully.")
print(df.head())

# ============================================================
# GLOBAL VISUAL SETTINGS
# ============================================================

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

# IEEE-compatible font embedding
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# ============================================================
# SAVE FIGURE FUNCTION
# ============================================================

def save_figure(filename):

    plt.savefig(
        f"{filename}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.savefig(
        f"{filename}.pdf",
        bbox_inches="tight"
    )

    plt.savefig(
        f"{filename}.eps",
        format="eps",
        bbox_inches="tight"
    )

# ============================================================
# 1. CORRELATION MATRIX
# ============================================================

columns = [
    "score",
    "utilidade",
    "reputacao",
    "energia"
]

corr = df[columns].corr()

plt.figure(figsize=(8, 6))

image = plt.imshow(
    corr,
    interpolation="nearest",
    cmap="RdBu_r",
    vmin=-1,
    vmax=1
)

plt.colorbar(image)

plt.xticks(
    range(len(columns)),
    ["Score", "Utility", "Reputation", "Energy"]
)

plt.yticks(
    range(len(columns)),
    ["Score", "Utility", "Reputation", "Energy"]
)

for i in range(len(columns)):
    for j in range(len(columns)):
        plt.text(
            j,
            i,
            f"{corr.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            color="black",
            fontsize=11,
            fontweight="bold"
        )

plt.title("Correlation Analysis")

plt.tight_layout()

save_figure(
    "correlation_analysis"
)

plt.show()

# ============================================================
# 2. UTILITY VS RESIDUAL ENERGY
# ============================================================

energy = df["energia"]
utility = df["utilidade"]

plt.figure(figsize=(10, 6))

plt.scatter(
    energy,
    utility,
    alpha=0.25,
    s=18,
    color="#F4A261"
)

# ============================================================
# AVERAGE UTILITY BY ENERGY BIN
# ============================================================

bins = np.linspace(
    energy.min(),
    energy.max(),
    15
)

df["energy_bin"] = pd.cut(
    energy,
    bins
)

average_by_bin = (
    df.groupby("energy_bin")["utilidade"]
    .mean()
)

centers = [
    interval.mid
    for interval in average_by_bin.index
]

plt.plot(
    centers,
    average_by_bin.values,
    linewidth=3,
    marker="D",
    markersize=6,
    color="#D62828",
    label="Average Utility per Energy Bin"
)

# ============================================================
# LINEAR TREND
# ============================================================

z = np.polyfit(
    energy,
    utility,
    1
)

p = np.poly1d(z)

x_values = np.linspace(
    energy.min(),
    energy.max(),
    200
)

plt.plot(
    x_values,
    p(x_values),
    linestyle=":",
    linewidth=2.5,
    color="black",
    label="Linear Trend"
)

plt.xlabel("Residual Energy")
plt.ylabel("Utility")

plt.title("Utility versus Residual Energy")

plt.grid(
    True,
    linestyle="--",
    alpha=0.5
)

plt.legend()

plt.tight_layout()

save_figure(
    "utility_vs_energy"
)

plt.show()

# ============================================================
# 3. SCORE EVOLUTION
# ============================================================

df = df.sort_values("step")

window = 50

df["score_smooth"] = (
    df["score"]
    .rolling(
        window=window,
        min_periods=1
    )
    .mean()
)

plt.figure(figsize=(12, 6))

plt.plot(
    df["step"],
    df["score_smooth"],
    color="black",
    linewidth=1.5,
    label=f"Moving Average ({window} Steps)"
)

plt.xlabel("Simulation Step")
plt.ylabel("Average Score")

plt.title("Score Evolution During Simulation")

plt.grid(
    True,
    linestyle="--",
    alpha=0.5
)

plt.legend()

plt.tight_layout()

save_figure(
    "score_evolution"
)

plt.show()

# ============================================================
# GENERAL STATISTICS
# ============================================================

print("\n========== GENERAL METRICS ==========")

print(f"Total Records: {len(df)}")

print(f"Average Score: {df['score'].mean():.4f}")

print(f"Average Utility: {df['utilidade'].mean():.4f}")

print(f"Average Residual Energy: {df['energia'].mean():.4f}")

print(f"Average Reputation: {df['reputacao'].mean():.4f}")

print(f"Minimum Score: {df['score'].min():.4f}")

print(f"Maximum Score: {df['score'].max():.4f}")

print(f"Executed Services: {len(df)}")

print("\nGenerated files:")

print(" - correlation_analysis.(png/pdf/eps)")
print(" - utility_vs_energy.(png/pdf/eps)")
print(" - score_evolution.(png/pdf/eps)")