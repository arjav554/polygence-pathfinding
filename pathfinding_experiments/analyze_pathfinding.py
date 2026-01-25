import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
FILE_NAME = "pathfinding_research_results.csv" 

# --- 1. LOAD AND CLEAN DATA ---
print("Loading data...")
try:
    df = pd.read_csv(FILE_NAME)
except FileNotFoundError:
    print(f"Error: Could not find file '{FILE_NAME}'. Make sure it matches exactly.")
    exit()

# Filter: Only keep successful runs. Analyzing failed runs skews the average.
# We explicitly copy() to avoid SettingWithCopy warnings.
df_clean = df[df['Success'] == True].copy()

# Conversion: Nanoseconds to Milliseconds (ns is too hard to read on charts)
df_clean['Time_ms'] = df_clean['Execution Time (ns)'] / 1_000_000

print(f"Data Loaded. {len(df)} total rows. {len(df_clean)} successful rows used for analysis.")

# Set the visual style for academic charts
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# --- 2. PLOT 1: SCALABILITY LINE PLOT (Grid Size vs Time) ---
# We use 'relplot' to create side-by-side charts (facets) for each Environment Type.
# If we mixed Mazes and Empty Maps on one chart, the data would look noisy.
print("Generating Scalability Plot...")

grid_plot = sns.relplot(
    data=df_clean,
    x="Grid Size",
    y="Time_ms",
    hue="Algorithm",       # Different color per algorithm
    style="Algorithm",     # Different line style per algorithm
    col="Environment Type",# Separate chart for each map type
    kind="line",
    markers=True,
    dashes=False,
    height=5,
    aspect=1,
    linewidth=2.5
)

# Fix Labels
grid_plot.set_titles("{col_name}")  # Titles for each subplot
grid_plot.set_axis_labels("Grid Dimension (N x N)", "Execution Time (ms)")
grid_plot.figure.suptitle("Algorithm Scalability by Environment", y=1.05, fontsize=16, weight='bold')

# Save
plt.savefig("1_Scalability_Analysis.png", dpi=300, bbox_inches='tight')
print("Saved: 1_Scalability_Analysis.png")

# --- 3. PLOT 2: EFFICIENCY BOX PLOT (Nodes Expanded) ---
# A Box Plot is statistically better than a Bar Chart because it shows
# variance (stability). It proves if an algorithm is "lucky" or consistently good.
print("Generating Efficiency Plot...")
plt.figure(figsize=(12, 6))

efficiency_plot = sns.boxplot(
    data=df_clean,
    x="Algorithm",
    y="Nodes Expanded",
    hue="Environment Type",
    palette="vlag"
)

plt.yscale('log') # IMPORTANT: Use Log Scale because BFS expands 100x more nodes than A*
plt.title("Search Efficiency Distribution (Log Scale)", fontsize=16, weight='bold')
plt.ylabel("Nodes Expanded (Log Scale)")
plt.xlabel("Algorithm")
plt.legend(title="Map Type", bbox_to_anchor=(1.05, 1), loc='upper left')

# Save
plt.savefig("2_Efficiency_Analysis.png", dpi=300, bbox_inches='tight')
print("Saved: 2_Efficiency_Analysis.png")

# --- 4. STATISTICAL SUMMARY TABLE ---
# This aggregates the raw numbers for you to quote in your paper.
print("\n--- STATISTICAL SUMMARY (Copy to Paper) ---")
summary = df_clean.groupby(['Environment Type', 'Algorithm']).agg({
    'Time_ms': ['mean', 'std'],            # Mean time + consistency (std dev)
    'Nodes Expanded': ['mean'],            # Work done
    'Peak Memory (Nodes in Open)': ['max'] # Worst-case memory usage
}).round(2)

print(summary)

# Optional: Save summary to CSV
summary.to_csv("3_Statistical_Summary.csv")
print("\nSaved: 3_Statistical_Summary.csv")
