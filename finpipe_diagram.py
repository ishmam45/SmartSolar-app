# -*- coding: utf-8 -*-
"""
Board-presentation pipeline diagram for solar_pipeline.py
Output: outputs/finpipe.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

os.makedirs("outputs", exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────
BG       = "#FFFFFF"
C1       = "#1D4ED8"   # step number circles  (rich blue)
BOX_COLS = [
    "#DBEAFE",  # 1 Data Generation  – light blue
    "#EDE9FE",  # 2 EDA              – light purple
    "#FEF3C7",  # 3 Preprocessing    – light amber
    "#FEE2E2",  # 4 Model Training   – light red/pink
    "#DCFCE7",  # 5 Evaluation       – light green
    "#F0FDF4",  # 6 Save & Deploy    – pale green
]
BDR_COLS = [
    "#3B82F6",
    "#7C3AED",
    "#D97706",
    "#DC2626",
    "#16A34A",
    "#15803D",
]
TXT_DARK = "#1E293B"
TXT_DIM  = "#64748B"

# ── Layout constants ───────────────────────────────────────────────────
FIG_W, FIG_H = 11, 15        # inches
CW,    CH    = 10,  14       # coordinate space
BW           = 7.8           # box width
BH           = 1.55          # box height
GAP          = 0.44          # gap between boxes
LEFT         = (CW - BW) / 2 # box left edge (centred)
CX           = CW / 2        # centre x

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, CW)
ax.set_ylim(0, CH)
ax.axis("off")

# ── Title ──────────────────────────────────────────────────────────────
ax.text(CX, CH - 0.30,
        "SolarCast  —  ML Training Pipeline",
        ha="center", va="top",
        fontsize=17, fontweight="bold", color=TXT_DARK)
ax.text(CX, CH - 0.82,
        "solar_pipeline.py",
        ha="center", va="top",
        fontsize=10, color=TXT_DIM, style="italic")

# ── Step definitions ───────────────────────────────────────────────────
steps = [
    {
        "num"  : "1",
        "title": "Data Generation",
        "body" : [
            "Simulate 50,000 realistic solar panel readings",
            "Physics formula gives the ideal output (P = h * A * G * [1 - b(T-25)])",
            "Four real-world loss factors applied: soiling, aging, shading, wiring",
            "Result: a labelled dataset the ML models can learn from",
        ],
        "note" : "50,000 rows   12 input features   saved to data/solar_dataset.csv",
    },
    {
        "num"  : "2",
        "title": "Exploratory Data Analysis  (EDA)",
        "body" : [
            "Visualise how irradiance, temperature and losses affect output",
            "Confirm that ML is needed  (formula alone cannot account for losses)",
            "Check feature correlations and data distribution",
        ],
        "note" : "Saved charts: outputs/eda_A_distributions.png   eda_B_correlations.png",
    },
    {
        "num"  : "3",
        "title": "Preprocessing",
        "body" : [
            "Remove night rows where irradiance = 0  (trivial to predict, not useful)",
            "Normalise all 12 features to 0-1 range  (MinMaxScaler)",
            "Split into Training set (80%) and Test set (20%)",
        ],
        "note" : "Scaler saved to models/scaler.pkl  |  No data leakage (scaler fit on train only)",
    },
    {
        "num"  : "4",
        "title": "Model Training  (5 Algorithms)",
        "body" : [
            "Random Forest         |   Gradient Boosting",
            "Neural Network (MLP)  |   XGBoost  (or SVR)",
            "LightGBM  (or Extra Trees)",
            "Every model learns to predict real-world DC power output in Watts",
        ],
        "note" : "All 5 models trained on the same scaled training set",
    },
    {
        "num"  : "5",
        "title": "Evaluation",
        "body" : [
            "Test each model on unseen data (the 20% test set)",
            "Metrics: R-squared   RMSE   MAE   MAPE",
            "Best model selected automatically by highest R-squared score",
            "Residual analysis and feature importance charts generated",
        ],
        "note" : "MAPE calculated only on outputs > 50W  (excludes meaningless dawn/dusk values)",
    },
    {
        "num"  : "6",
        "title": "Save  &  Hand Off to Dashboard",
        "body" : [
            "All 5 trained models saved as  models/*.pkl",
            "Best model also saved as  models/best_model.pkl",
            "Dashboard (app.py) loads these files to run live predictions",
        ],
        "note" : "Output charts: model_comparison.png   results_table.png   formula_vs_ml.png",
    },
]

# ── Draw boxes top-to-bottom ───────────────────────────────────────────
n = len(steps)
# total height used by boxes and gaps
total = n * BH + (n - 1) * GAP
# starting y (top of first box)
y_top = CH - 1.15

def draw_step(ax, step_dict, i, y):
    fc  = BOX_COLS[i]
    bc  = BDR_COLS[i]
    cy  = y - BH / 2     # centre y of box

    # ── Main rounded box ──────────────────────────────────────────────
    patch = mpatches.FancyBboxPatch(
        (LEFT, y - BH), BW, BH,
        boxstyle="round,pad=0.12",
        facecolor=fc, edgecolor=bc, linewidth=1.6, zorder=3)
    ax.add_patch(patch)

    # ── Step number circle (left, vertically centred) ─────────────────
    circle = plt.Circle((LEFT + 0.42, cy), 0.28, color=bc, zorder=5)
    ax.add_patch(circle)
    ax.text(LEFT + 0.42, cy,
            step_dict["num"],
            ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", zorder=6)

    # ── Title (sits at top of box) ────────────────────────────────────
    title_y = y - 0.26
    ax.text(LEFT + 0.88, title_y,
            step_dict["title"],
            ha="left", va="center",
            fontsize=11.5, fontweight="bold", color=TXT_DARK, zorder=4)

    # ── Body bullet points ────────────────────────────────────────────
    body_lines = step_dict["body"]
    n_lines    = len(body_lines)
    # bullets start just below title, end just above note
    usable     = BH - 0.55     # space for bullets
    y_step     = usable / (n_lines + 0.5)
    y_start    = y - 0.52

    for j, line in enumerate(body_lines):
        ax.text(LEFT + 0.92, y_start - j * y_step,
                u"\u2022  " + line,
                ha="left", va="center",
                fontsize=9.0, color=TXT_DARK, zorder=4)

    # ── Note strip (pinned to bottom of box) ──────────────────────────
    note_y = y - BH + 0.21
    ax.text(LEFT + 0.92, note_y,
            step_dict["note"],
            ha="left", va="center",
            fontsize=7.5, color=bc,
            style="italic", zorder=4)


# ── Draw all steps and arrows ──────────────────────────────────────────
y_cursor = y_top
for i, step in enumerate(steps):
    draw_step(ax, step, i, y_cursor)

    # Arrow to next box
    if i < len(steps) - 1:
        ax.annotate(
            "", xy=(CX, y_cursor - BH - GAP + 0.06),
            xytext=(CX, y_cursor - BH),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#94A3B8", lw=1.8,
                mutation_scale=14),
            zorder=2)

    y_cursor -= BH + GAP

# ── Footer ─────────────────────────────────────────────────────────────
ax.text(CX, 0.12,
        "After training completes  ->  launch dashboard:   streamlit run app.py",
        ha="center", va="bottom",
        fontsize=8, color=TXT_DIM, style="italic")

plt.tight_layout(rect=[0, 0.01, 1, 1])
fig.savefig("outputs/finpipe.png", dpi=180,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("[OK] outputs/finpipe.png")
