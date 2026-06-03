import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# FILES
# ============================================================

FILE_NORMAL = "metricas_shadai_400.csv"
FILE_GREEDY = "metricas_shadai_400_ganancia.csv"
FILE_FLOOD = "metricas_shadai_400_flood.csv"

# ============================================================
# IEEE FIGURE SETTINGS
# ============================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

WINDOW = 30

# ============================================================
# COLORS
# ============================================================

COLORS = {
    "Normal": "#000000",          # Black
    "Greedy Attack": "#B02E0C",   # Dark Red
    "Flood Attack": "#1D3557"     # Dark Blue
}

# ============================================================
# LOAD DATA
# ============================================================

df_normal = pd.read_csv(FILE_NORMAL)
df_greedy = pd.read_csv(FILE_GREEDY)
df_flood = pd.read_csv(FILE_FLOOD)

# ============================================================
# SMOOTHING
# ============================================================

def smooth(series):

    return series.rolling(
        WINDOW,
        min_periods=1
    ).mean()

# ============================================================
# AVERAGE BY STEP
# ============================================================

def average_per_step(df, column):

    data = (
        df
        .groupby("step")[column]
        .mean()
        .reset_index()
    )

    data[column] = smooth(
        data[column]
    )

    return data

# ============================================================
# SAVE FIGURE FUNCTION
# ============================================================

def save_figure(filename):

    plt.tight_layout()

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

    plt.close()

# ============================================================
# REPUTATION EVOLUTION
# ============================================================

normal_rep = average_per_step(
    df_normal,
    "reputacao"
)

greedy_rep = average_per_step(
    df_greedy,
    "reputacao"
)

flood_rep = average_per_step(
    df_flood,
    "reputacao"
)

plt.figure(figsize=(11,5))

plt.plot(
    normal_rep["step"],
    normal_rep["reputacao"],
    color=COLORS["Normal"],
    linewidth=3,
    label="Normal"
)

plt.plot(
    greedy_rep["step"],
    greedy_rep["reputacao"],
    color=COLORS["Greedy Attack"],
    linewidth=2.2,
    linestyle="--",
    label="Greedy Attack"
)

plt.plot(
    flood_rep["step"],
    flood_rep["reputacao"],
    color=COLORS["Flood Attack"],
    linewidth=2.2,
    linestyle=":",
    label="Flood Attack"
)

plt.xlabel("Simulation Step")
plt.ylabel("Average Reputation")

plt.title(
    "Average Reputation Under Adversarial Conditions",
    fontweight="bold"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

save_figure(
    "reputation_under_attacks"
)

# ============================================================
# UTILITY EVOLUTION
# ============================================================

normal_util = average_per_step(
    df_normal,
    "utilidade"
)

greedy_util = average_per_step(
    df_greedy,
    "utilidade"
)

flood_util = average_per_step(
    df_flood,
    "utilidade"
)

plt.figure(figsize=(11,5))

plt.plot(
    normal_util["step"],
    normal_util["utilidade"],
    color=COLORS["Normal"],
    linewidth=3,
    label="Normal"
)

plt.plot(
    greedy_util["step"],
    greedy_util["utilidade"],
    color=COLORS["Greedy Attack"],
    linewidth=2.2,
    linestyle="--",
    label="Greedy Attack"
)

plt.plot(
    flood_util["step"],
    flood_util["utilidade"],
    color=COLORS["Flood Attack"],
    linewidth=2.2,
    linestyle=":",
    label="Flood Attack"
)

plt.xlabel("Simulation Step")
plt.ylabel("Average Utility")

plt.title(
    "Average Utility Under Adversarial Conditions",
    fontweight="bold"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

save_figure(
    "utility_under_attacks"
)

# ============================================================
# MALICIOUS VEHICLE SELECTION RATE
# ============================================================

def malicious_rate(df):

    data = (
        df.groupby("step")["malicioso"]
        .mean()
        .reset_index()
    )

    data["malicioso"] = smooth(
        data["malicioso"]
    )

    return data

greedy_rate = malicious_rate(
    df_greedy
)

flood_rate = malicious_rate(
    df_flood
)

plt.figure(figsize=(11,5))

plt.plot(
    greedy_rate["step"],
    greedy_rate["malicioso"],
    color=COLORS["Greedy Attack"],
    linewidth=2.5,
    label="Greedy Attack"
)

plt.plot(
    flood_rate["step"],
    flood_rate["malicioso"],
    color=COLORS["Flood Attack"],
    linewidth=2.5,
    linestyle="--",
    label="Flood Attack"
)

# Expected malicious ratio (20%)

plt.axhline(
    y=0.20,
    color="gray",
    linestyle=":",
    linewidth=1.5,
    label="Expected Malicious Ratio (20%)"
)

plt.xlabel("Simulation Step")
plt.ylabel("Malicious Selection Rate")

plt.title(
    "Malicious Vehicle Selection Rate",
    fontweight="bold"
)

plt.ylim(0, 1)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

save_figure(
    "malicious_selection_rate"
)

# ============================================================
# SUMMARY STATISTICS
# ============================================================

print("\n===================================")
print("SCENARIO SUMMARY")
print("===================================")

for name, df in [
    ("Normal", df_normal),
    ("Greedy Attack", df_greedy),
    ("Flood Attack", df_flood)
]:

    print(f"\n{name}")

    print(
        f"Average Score: "
        f"{df['score'].mean():.4f}"
    )

    print(
        f"Average Utility: "
        f"{df['utilidade'].mean():.4f}"
    )

    print(
        f"Average Reputation: "
        f"{df['reputacao'].mean():.4f}"
    )

# ============================================================
# FINISH
# ============================================================

print("\nFigures generated successfully!")

print("\nGenerated files:")
print(" - reputation_under_attacks.(png/pdf/eps)")
print(" - utility_under_attacks.(png/pdf/eps)")
print(" - malicious_selection_rate.(png/pdf/eps)")