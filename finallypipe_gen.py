# -*- coding: utf-8 -*-
"""
Generates outputs/finallypipe.png
Same visual style as the original pipeline_diagram.png:
  - Light grey page background
  - Dark wide row boxes for major stages
  - Coloured model boxes
  - Clean arrows, no FIX annotations
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

os.makedirs("outputs", exist_ok=True)

# ── Canvas ─────────────────────────────────────────────────────────────
FW, FH = 22, 12          # figure size in inches
fig, ax = plt.subplots(figsize=(FW, FH))
fig.patch.set_facecolor("#F1F5F9")
ax.set_facecolor("#F1F5F9")
ax.set_xlim(0, 22)
ax.set_ylim(0, 12)
ax.axis("off")

# ── Colours ────────────────────────────────────────────────────────────
DARK1 = "#0F4C81"   # input boxes
DARK2 = "#164E63"   # physics engine box
DARK3 = "#1E293B"   # dataset row
DARK4 = "#1E3A5F"   # evaluation row
DARK5 = "#374151"   # scaled sets row
PURPLE= "#7C3AED"   # EDA
TEAL  = "#0891B2"   # preprocessing
GREEN = "#166534"   # saved models
VIOL  = "#6D28D9"   # dashboard

MODEL_COLORS = {
    "RF" : "#F59E0B",
    "GB" : "#22C55E",
    "ANN": "#7F77DD",
    "XGB": "#EF4444",
    "LGB": "#0891B2",
}


# ── Helpers ────────────────────────────────────────────────────────────
def box(ax, cx, cy, w, h, title, sub="",
        fc="#1E3A5F", tc="white", fs=10, sub_fs=8):
    patch = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.12",
        facecolor=fc, edgecolor="#475569",
        linewidth=0.9, zorder=3)
    ax.add_patch(patch)
    dy = 0.15 if sub else 0
    ax.text(cx, cy + dy, title,
            ha="center", va="center",
            fontsize=fs, fontweight="bold", color=tc, zorder=4)
    if sub:
        ax.text(cx, cy - 0.22, sub,
                ha="center", va="center",
                fontsize=sub_fs, color=tc, alpha=0.82, zorder=4)


def arr(ax, x1, y1, x2, y2, color="#64748B", lw=1.6):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color, lw=lw,
                    mutation_scale=12), zorder=2)


# ══════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════
ax.text(11, 11.60,
        "SolarCast  \u2014  ML Pipeline Architecture",
        ha="center", va="center",
        fontsize=16, fontweight="bold", color="#1E293B")
ax.text(11, 11.25,
        "How solar panel output is predicted using machine learning",
        ha="center", va="center",
        fontsize=9.5, color="#64748B")

# ══════════════════════════════════════════════════════════════════════
# ROW 1  \u2014  INPUT BOXES
# ══════════════════════════════════════════════════════════════════════
r1 = 10.55
box(ax,  3.0, r1, 3.8, 0.90, "Weather Inputs",
    "Irradiance \u00b7 Temp \u00b7 Humidity \u00b7 Wind", DARK1)
box(ax,  7.5, r1, 3.8, 0.90, "Panel Datasheet",
    "Efficiency \u00b7 Area \u00b7 Temp-coeff \u00b7 NOCT", DARK1)
box(ax, 12.0, r1, 3.8, 0.90, "Time Features",
    "Hour \u00b7 Month \u00b7 Day of Year", DARK1)
box(ax, 17.0, r1, 4.4, 0.90, "Physics Engine",
    "T_mod = T_amb + (NOCT\u221220)/800 \u00d7 G", DARK2)

for cx in [3.0, 7.5, 12.0, 17.0]:
    arr(ax, cx, r1 - 0.45, cx, r1 - 0.98)

# ══════════════════════════════════════════════════════════════════════
# ROW 2  \u2014  DATASET (wide dark row)
# ══════════════════════════════════════════════════════════════════════
r2 = 9.20
box(ax, 11.0, r2, 21.0, 0.88,
    "DATASET  \u2014  50,000 rows \u00d7 12 features",
    "Labels = Ideal formula \u00d7 soiling \u00d7 aging \u00d7 shading \u00d7 wiring + noise  "
    "(real-world DC power output in Watts)",
    DARK3, fs=11)
arr(ax, 11.0, r2 - 0.44, 11.0, r2 - 1.00)

# ══════════════════════════════════════════════════════════════════════
# ROW 3  \u2014  EDA  +  PREPROCESSING  (side by side)
# ══════════════════════════════════════════════════════════════════════
r3 = 7.88
box(ax,  4.8, r3, 7.5, 0.88,
    "Exploratory Data Analysis",
    "Visualise distributions \u00b7 Correlations \u00b7 Confirm ML is needed",
    PURPLE, fs=10.5)
box(ax, 17.8, r3, 7.5, 0.88,
    "Preprocessing Pipeline",
    "Remove night rows \u00b7 MinMaxScaler \u00b7 80% train / 20% test split",
    TEAL, fs=10.5)

arr(ax,  7.5, r2 - 0.44,  4.8, r3 + 0.44)   # dataset -> EDA
arr(ax, 14.5, r2 - 0.44, 17.8, r3 + 0.44)   # dataset -> preprocessing

for cx in [4.8, 17.8]:
    arr(ax, cx, r3 - 0.44, 11.0, r3 - 1.00)

# ══════════════════════════════════════════════════════════════════════
# ROW 4  \u2014  SCALED FEATURE SETS
# ══════════════════════════════════════════════════════════════════════
r4 = 6.55
box(ax, 11.0, r4, 21.0, 0.88,
    "Scaled Feature Sets   \u2014   X_train  +  y_train      |      X_test  +  y_test",
    "All 12 features normalised 0\u20131 (MinMaxScaler)  \u00b7  Scaler saved to models/scaler.pkl",
    DARK5, fs=10.5)
arr(ax, 11.0, r4 - 0.44, 11.0, r4 - 1.03)

# ══════════════════════════════════════════════════════════════════════
# ROW 5  \u2014  5 MODEL BOXES
# ══════════════════════════════════════════════════════════════════════
r5 = 5.12
xm = [2.8, 6.1, 9.5, 13.0, 16.5]
labels = [
    ("RF",  "Random\nForest"),
    ("GB",  "Gradient\nBoosting"),
    ("ANN", "MLP /\nANN"),
    ("XGB", "XGBoost\nor SVR"),
    ("LGB", "LightGBM\nor ET"),
]
for (key, full), mx in zip(labels, xm):
    ax.annotate("", xy=(mx, r5 + 0.52), xytext=(11.0, r4 - 0.44),
                arrowprops=dict(arrowstyle="-|>",
                                color="#94A3B8", lw=1.2,
                                mutation_scale=10), zorder=2)
    box(ax, mx, r5, 2.9, 1.04, key, full, MODEL_COLORS[key], fs=11, sub_fs=9)

for mx in xm:
    arr(ax, mx, r5 - 0.52, mx, r5 - 1.04)

# ══════════════════════════════════════════════════════════════════════
# ROW 6  \u2014  EVALUATION
# ══════════════════════════════════════════════════════════════════════
r6 = 3.68
box(ax, 11.0, r6, 21.0, 0.80,
    "EVALUATION   \u2192   R\u00b2  \u00b7  RMSE  \u00b7  MAE  \u00b7  MAPE  \u00b7  Residual Analysis  \u00b7  Feature Importance",
    "", DARK4, fs=11)
arr(ax, 11.0, r6 - 0.40, 11.0, r6 - 0.94)

# ══════════════════════════════════════════════════════════════════════
# ROW 7  \u2014  OUTPUTS  (two boxes side by side)
# ══════════════════════════════════════════════════════════════════════
r7 = 2.44
box(ax,  6.2, r7, 9.0, 0.76,
    "Saved Models   (models/*.pkl)",
    "Best model + all 5 models + scaler",
    GREEN, fs=10)
box(ax, 16.2, r7, 9.0, 0.76,
    "Streamlit Dashboard   (app.py)",
    "Live prediction \u00b7 Sensitivity analysis \u00b7 Model performance",
    VIOL, fs=10)

ax.annotate("", xy=(16.2 - 4.5, r7), xytext=(6.2 + 4.5, r7),
            arrowprops=dict(arrowstyle="-|>",
                            color="#94A3B8", lw=1.4,
                            mutation_scale=11), zorder=2)

plt.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig("outputs/finallypipe.png", dpi=180,
            bbox_inches="tight", facecolor="#F1F5F9")
plt.close()
print("[OK] outputs/finallypipe.png")
