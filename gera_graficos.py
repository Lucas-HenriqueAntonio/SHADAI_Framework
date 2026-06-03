import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

MODES = [
    "shadai",
    "random",
    "greedy",
    "no_reputation"
]

VEHICLES = 400

WINDOW = 20

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
# LABELS
# ============================================================

MODE_LABELS = {
    "shadai": "SHADAI",
    "random": "Random",
    "greedy": "Greedy",
    "no_reputation": "No Reputation"
}

# ============================================================
# COLORS
# ============================================================

COLORS = {
    "shadai": "#000000",        # Black
    "random": "#B02E0C",        # Dark Red
    "greedy": "#1D3557",        # Dark Blue
    "no_reputation": "#1B4332"  # Dark Green
}

# ============================================================
# LINE STYLES
# ============================================================

LINESTYLES = {
    "shadai": "-",
    "random": "--",
    "greedy": ":",
    "no_reputation": "-."
}

# ============================================================
# LINE WIDTHS
# ============================================================

LINEWIDTHS = {
    "shadai": 3.0,
    "random": 2.0,
    "greedy": 2.0,
    "no_reputation": 2.0
}

# ============================================================
# LOAD DATASETS
# ============================================================

dfs = []

for mode in MODES:

    filename = f"metricas_{mode}_{VEHICLES}.csv"

    try:

        df = pd.read_csv(filename)

        df["mode"] = mode

        dfs.append(df)

        print(f"Loaded: {filename}")

    except Exception:

        print(f"File not found: {filename}")

if not dfs:
    raise Exception("No CSV files were found.")

data = pd.concat(dfs)

# ============================================================
# GROUP BY STEP
# ============================================================

average = (
    data
    .groupby(["step", "mode"])
    .mean(numeric_only=True)
    .reset_index()
)

# ============================================================
# SMOOTHING FUNCTION
# ============================================================

def smooth(df, column):

    return (
        df[column]
        .rolling(
            window=WINDOW,
            min_periods=1
        )
        .mean()
    )

# ============================================================
# PLOTTING FUNCTION
# ============================================================

def generate_plot(
    column,
    title,
    ylabel,
    filename
):

    plt.figure(figsize=(12, 6))

    for mode in MODES:

        subset = average[
            average["mode"] == mode
        ].copy()

        subset = subset.sort_values("step")

        subset["smooth"] = smooth(
            subset,
            column
        )

        plt.plot(
            subset["step"],
            subset["smooth"],
            color=COLORS[mode],
            linestyle=LINESTYLES[mode],
            linewidth=LINEWIDTHS[mode],
            label=MODE_LABELS[mode]
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
    # PNG
    # ========================================================

    plt.savefig(
        f"{filename}.png",
        dpi=300,
        bbox_inches="tight"
    )

    # ========================================================
    # PDF (VECTOR)
    # ========================================================

    plt.savefig(
        f"{filename}.pdf",
        bbox_inches="tight"
    )

    # ========================================================
    # EPS (VECTOR)
    # ========================================================

    plt.savefig(
        f"{filename}.eps",
        format="eps",
        bbox_inches="tight"
    )

    plt.show()

# ============================================================
# GENERATE FIGURES
# ============================================================

generate_plot(
    "score",
    f"Average Score Comparison Across Selection Strategies",
    "Average Score",
    f"score_comparison_{VEHICLES}"
)

generate_plot(
    "utilidade",
    f"Average Utility Comparison Across Selection Strategies",
    "Average Utility",
    f"utility_comparison_{VEHICLES}"
)

generate_plot(
    "reputacao",
    f"Average Reputation Comparison Across Selection Strategies",
    "Average Reputation",
    f"reputation_comparison_{VEHICLES}"
)

print("\nComparative figures generated successfully!")
print("Files saved in PNG, PDF, and EPS formats.")