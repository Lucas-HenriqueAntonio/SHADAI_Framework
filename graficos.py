import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

files = [
    "metricas_shadai_100.csv",
    "metricas_shadai_400.csv",
    "metricas_shadai_800.csv"
]

labels = [
    "100 Vehicles",
    "400 Vehicles",
    "800 Vehicles"
]

# ============================================================
# IEEE FIGURE SETTINGS
# ============================================================

plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

# Embed fonts correctly in PDF/EPS
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# ============================================================
# COLORS AND LINE STYLES
# ============================================================

colors = {
    "100 Vehicles": "#1B4332",  # Dark Green
    "400 Vehicles": "#B02E0C",  # Dark Red
    "800 Vehicles": "#1D3557"   # Dark Blue
}

styles = {
    "100 Vehicles": "-",
    "400 Vehicles": "--",
    "800 Vehicles": ":"
}

# ============================================================
# LOAD DATASETS
# ============================================================

dfs = []

for file, label in zip(files, labels):

    df = pd.read_csv(file)

    df["scenario"] = label

    dfs.append(df)

data = pd.concat(dfs)

print("\nDatasets loaded successfully.")

# ============================================================
# AVERAGE PER STEP
# ============================================================

average = (
    data
    .groupby(["step", "scenario"])
    .mean(numeric_only=True)
    .reset_index()
)

# ============================================================
# SMOOTHING
# ============================================================

WINDOW = 20

metrics = [
    (
        "score",
        "Average Score",
        "Score Evolution Across Traffic Densities",
        "score_comparison"
    ),
    (
        "utilidade",
        "Average Utility",
        "Utility Evolution Across Traffic Densities",
        "utility_comparison"
    ),
    (
        "reputacao",
        "Average Reputation",
        "Reputation Evolution Across Traffic Densities",
        "reputation_comparison"
    )
]

# ============================================================
# PLOTTING
# ============================================================

for column, ylabel, title, filename in metrics:

    plt.figure(figsize=(12, 6))

    for label in labels:

        subset = average[
            average["scenario"] == label
        ].copy()

        subset = subset.sort_values("step")

        subset["smooth"] = (
            subset[column]
            .rolling(
                window=WINDOW,
                min_periods=1
            )
            .mean()
        )

        plt.plot(
            subset["step"],
            subset["smooth"],
            color=colors[label],
            linestyle=styles[label],
            linewidth=2.2,
            label=label
        )

    plt.xlabel("Simulation Step")
    plt.ylabel(ylabel)

    plt.title(
        title,
        fontweight="bold"
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.5
    )

    plt.legend(
        frameon=True
    )

    plt.tight_layout()

    # ========================================================
    # SAVE PNG
    # ========================================================

    plt.savefig(
        f"{filename}.png",
        dpi=300,
        bbox_inches="tight"
    )

    # ========================================================
    # SAVE PDF (VECTOR)
    # ========================================================

    plt.savefig(
        f"{filename}.pdf",
        bbox_inches="tight"
    )

    # ========================================================
    # SAVE EPS (VECTOR)
    # ========================================================

    plt.savefig(
        f"{filename}.eps",
        format="eps",
        bbox_inches="tight"
    )

    plt.show()

print("\nComparative plots generated successfully!")
print("Files saved in PNG, PDF, and EPS formats.")