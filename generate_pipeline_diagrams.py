# -*- coding: utf-8 -*-
"""
Pipeline Diagram Generator  -  compact, light-background, report/slide ready
Produces:
  outputs/solarpipe.png      - solar_pipeline.py flow
  outputs/apppipe.png        - app.py flow
  outputs/Finalpipeline.png  - combined end-to-end
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

os.makedirs("outputs", exist_ok=True)

# ── Light-theme colour palette ─────────────────────────────────────────
BG        = "#FFFFFF"
GRID_LINE = "#E2E8F0"

# Box fills  (soft, readable on white)
C_NAVY    = "#1E3A5F"   # dark header text boxes
C_BLUE    = "#DBEAFE"   # input boxes (light blue)
C_AMBER   = "#FEF3C7"   # model boxes
C_GREEN   = "#DCFCE7"   # output / save boxes
C_PURPLE  = "#EDE9FE"   # EDA / app tab boxes
C_TEAL    = "#CCFBF1"   # preprocessing boxes
C_SLATE   = "#F1F5F9"   # neutral section boxes
C_CORAL   = "#FEE2E2"   # evaluation boxes
C_INDIGO  = "#E0E7FF"   # formula / dataset boxes

# Border / text colours
BD_BLUE   = "#3B82F6"
BD_AMBER  = "#D97706"
BD_GREEN  = "#16A34A"
BD_PURPLE = "#7C3AED"
BD_TEAL   = "#0D9488"
BD_SLATE  = "#64748B"
BD_CORAL  = "#DC2626"
BD_INDIGO = "#4338CA"
BD_NAVY   = "#1E3A5F"


def pbox(ax, cx, cy, w, h, title, sub="",
         fc=C_SLATE, bc=BD_SLATE, tc="#1E293B",
         fs=7.5, sub_fs=6.5, pad=0.08, lw=1.1):
    """Compact rounded box."""
    patch = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad={pad}",
        facecolor=fc, edgecolor=bc,
        linewidth=lw, zorder=3)
    ax.add_patch(patch)
    dy = 0.14 if sub else 0
    ax.text(cx, cy + dy, title,
            ha="center", va="center",
            fontsize=fs, fontweight="bold", color=tc, zorder=4)
    if sub:
        ax.text(cx, cy - 0.17, sub,
                ha="center", va="center",
                fontsize=sub_fs, color="#475569", zorder=4)


def arr(ax, x1, y1, x2, y2, color=BD_SLATE, lw=1.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color, lw=lw,
                    mutation_scale=8), zorder=2)


def step_tag(ax, x, y, text, color=BD_SLATE):
    ax.text(x, y, text,
            ha="left", va="center",
            fontsize=5.5, color=color,
            fontweight="bold", style="italic")


# ══════════════════════════════════════════════════════════════════════
# DIAGRAM 1  -  solar_pipeline.py
# ══════════════════════════════════════════════════════════════════════
def draw_solar():
    FW, FH = 13, 10          # figure inches  (compact for report)
    CW, CH = 13, 10          # axis coordinate space
    fig, ax = plt.subplots(figsize=(FW, FH))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, CW); ax.set_ylim(0, CH); ax.axis("off")

    # thin horizontal grid lines as guides
    fig.text(0.5, 0.98,
             "SolarCast — Training Pipeline  (solar_pipeline.py)",
             ha="center", va="top",
             fontsize=10, fontweight="bold", color=C_NAVY)
    fig.text(0.5, 0.955,
             "Data generation  →  EDA  →  Preprocessing  →  Model training  →  Evaluation  →  Saved artifacts",
             ha="center", va="top", fontsize=6.5, color="#64748B")

    top = CH - 0.95
    BH  = 0.60     # standard box height
    GAP = 0.42     # vertical gap between rows

    # ── Row 0: Inputs ─────────────────────────────────────────────────
    r0 = top
    step_tag(ax, 0.05, r0, "INPUTS")
    xs = [1.85, 4.15, 6.45, 8.75, 11.15]
    labels = [
        ("Weather", "G  T_amb  humidity\nwind_speed"),
        ("Panel Specs", "eta  area  beta  NOCT\n(random per row)"),
        ("Time", "hour  month\nday_of_year"),
        ("NOCT Engine", "T_mod = T_amb +\n(NOCT-20)/800 * G"),
        ("Physics\nFormula", "P_ideal = eta*A*G*\n[1-beta*(T_mod-25)]"),
    ]
    fills  = [C_BLUE,  C_BLUE,  C_BLUE,  C_INDIGO, C_INDIGO]
    bds    = [BD_BLUE, BD_BLUE, BD_BLUE, BD_INDIGO, BD_INDIGO]
    for x, (title, sub), fc, bc in zip(xs, labels, fills, bds):
        pbox(ax, x, r0, 2.1, BH, title, sub, fc, bc, fs=7)
        arr(ax, x, r0 - BH/2, x, r0 - BH/2 - GAP + 0.05)

    # ── Row 1: Loss factors ────────────────────────────────────────────
    r1 = r0 - BH - GAP
    step_tag(ax, 0.05, r1, "REAL-WORLD\nLOSSES")
    loss_xs = [2.1, 4.4, 6.7, 9.0, 11.1]
    loss_lbl = [
        ("Soiling", "0.80-1.00"),
        ("Aging", "0.92-1.00"),
        ("Shading", "0.50-0.85\n(12% prob.)"),
        ("Wiring", "0.95-0.99"),
        ("Noise", "Gaussian\n+/-5 W"),
    ]
    for x, (t, s) in zip(loss_xs, loss_lbl):
        pbox(ax, x, r1, 1.95, BH + 0.08, t, s, C_CORAL, BD_CORAL, fs=7)
        arr(ax, x, r1 - (BH+0.08)/2, 6.5, r1 - (BH+0.08)/2 - GAP + 0.05,
            color=BD_CORAL, lw=0.8)

    # ── Row 2: Dataset ────────────────────────────────────────────────
    r2 = r1 - BH - GAP - 0.08
    step_tag(ax, 0.05, r2, "DATASET")
    pbox(ax, 6.5, r2, 12.6, BH,
         "Generated Dataset  |  50,000 rows x 12 features + target (dc_power)",
         "P_dc = P_ideal x soiling x aging x shading x wiring + noise",
         C_INDIGO, BD_INDIGO, fs=7.5)
    arr(ax, 6.5, r2 - BH/2, 3.3, r2 - BH/2 - GAP + 0.05)
    arr(ax, 6.5, r2 - BH/2, 9.7, r2 - BH/2 - GAP + 0.05)

    # ── Row 3: EDA + Preprocessing ────────────────────────────────────
    r3 = r2 - BH - GAP
    step_tag(ax, 0.05, r3, "EDA &\nPREPROC")
    pbox(ax, 3.3, r3, 5.8, BH + 0.1,
         "Exploratory Data Analysis",
         "Distributions  Correlations  Stats\nou.outputs/eda_A/B/C.png",
         C_PURPLE, BD_PURPLE, fs=7)
    pbox(ax, 9.7, r3, 5.8, BH + 0.1,
         "Preprocessing",
         "Drop night rows  MinMaxScaler (train only)\n80/20 split  MAPE threshold = 50W",
         C_TEAL, BD_TEAL, fs=7)
    for x in [3.3, 9.7]:
        arr(ax, x, r3 - (BH+0.1)/2, 6.5, r3 - (BH+0.1)/2 - GAP + 0.05)

    # ── Row 4: Scaled sets ────────────────────────────────────────────
    r4 = r3 - BH - GAP - 0.1
    step_tag(ax, 0.05, r4, "SCALED\nSETS")
    pbox(ax, 6.5, r4, 12.6, BH,
         "X_train_sc  y_train    |    X_test_sc  y_test",
         "12 features normalised 0-1  |  scaler saved: models/scaler.pkl",
         C_SLATE, BD_SLATE, fs=7.5)
    arr(ax, 6.5, r4 - BH/2, 6.5, r4 - BH/2 - GAP + 0.05)

    # ── Row 5: 5 Models ──────────────────────────────────────────────
    r5 = r4 - BH - GAP
    step_tag(ax, 0.05, r5, "5 ML\nMODELS")
    models = [
        (2.1,  "Random\nForest",   C_AMBER,  BD_AMBER),
        (4.4,  "Gradient\nBoost",  C_AMBER,  BD_AMBER),
        (6.7,  "MLP / ANN",        C_AMBER,  BD_AMBER),
        (9.0,  "XGBoost\nor SVR",  C_AMBER,  BD_AMBER),
        (11.3, "LightGBM\nor ET",  C_AMBER,  BD_AMBER),
    ]
    for mx, label, fc, bc in models:
        arr(ax, mx, r4 - BH/2, mx, r5 + (BH+0.1)/2, color=BD_AMBER, lw=0.8)
        pbox(ax, mx, r5, 1.95, BH + 0.1, label, "", fc, bc, fs=7)
        arr(ax, mx, r5 - (BH+0.1)/2, 6.5, r5 - (BH+0.1)/2 - GAP + 0.05,
            color=BD_AMBER, lw=0.8)

    # ── Row 6: Evaluation ─────────────────────────────────────────────
    r6 = r5 - BH - GAP - 0.1
    step_tag(ax, 0.05, r6, "EVAL")
    pbox(ax, 6.5, r6, 12.6, BH,
         "Evaluation  |  R2   RMSE   MAE   MAPE (>50W)   Residuals   Feature Importance",
         "compute_metrics()  |  Best model selected by R2",
         C_CORAL, BD_CORAL, fs=7.5)
    arr(ax, 6.5, r6 - BH/2, 3.8, r6 - BH/2 - GAP + 0.05)
    arr(ax, 6.5, r6 - BH/2, 9.2, r6 - BH/2 - GAP + 0.05)

    # ── Row 7: Outputs ────────────────────────────────────────────────
    r7 = r6 - BH - GAP
    step_tag(ax, 0.05, r7, "ARTIFACTS")
    pbox(ax, 3.8, r7, 5.8, BH,
         "Saved Models",
         "models/  rf  gb  mlp  xgb  lgb\nsvr  et  best_model  scaler  .pkl",
         C_GREEN, BD_GREEN, fs=7)
    pbox(ax, 9.2, r7, 5.8, BH,
         "Output Charts",
         "data/solar_dataset.csv\nou.outputs/  model_comparison  results_table\nresidual_analysis  formula_vs_ml  .png",
         C_GREEN, BD_GREEN, fs=7)

    plt.tight_layout(rect=[0.06, 0, 1, 0.95])
    fig.savefig("outputs/solarpipe.png", dpi=200,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  [OK] outputs/solarpipe.png")


# ══════════════════════════════════════════════════════════════════════
# DIAGRAM 2  -  app.py
# ══════════════════════════════════════════════════════════════════════
def draw_app():
    FW, FH = 13, 10
    CW, CH = 13, 10
    fig, ax = plt.subplots(figsize=(FW, FH))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, CW); ax.set_ylim(0, CH); ax.axis("off")

    fig.text(0.5, 0.98,
             "SolarCast — Dashboard Flow  (app.py)",
             ha="center", va="top",
             fontsize=10, fontweight="bold", color=C_NAVY)
    fig.text(0.5, 0.955,
             "User inputs  ->  ML inference  ->  Physics baseline  ->  Real-world adjustment  ->  Visualisations",
             ha="center", va="top", fontsize=6.5, color="#64748B")

    top = CH - 0.95
    BH  = 0.60
    GAP = 0.40

    # ── Row 0: Sidebar inputs ─────────────────────────────────────────
    r0 = top
    step_tag(ax, 0.05, r0, "SIDEBAR\nINPUTS")
    inp = [
        (2.0,  "Panel Preset\n& Specs",       "Rated Wp  eta  area\nbeta  NOCT  #panels",  C_BLUE,   BD_BLUE),
        (4.6,  "Real-World\nConditions",       "Soiling  Aging\nWiring  Shading %",         C_CORAL,  BD_CORAL),
        (7.2,  "System Config",                "Inverter  Battery\nLoad W  Sun hrs",         C_TEAL,   BD_TEAL),
        (9.8,  "Weather",                      "G  T_amb  humidity\nwind  hour  month  DoY", C_BLUE,   BD_BLUE),
        (12.1, "Loaded Assets",                "scaler.pkl +\n7 model .pkl files",           C_GREEN,  BD_GREEN),
    ]
    for x, t, s, fc, bc in inp:
        pbox(ax, x, r0, 2.3, BH + 0.1, t, s, fc, bc, fs=6.8)
        arr(ax, x, r0 - (BH+0.1)/2, x, r0 - (BH+0.1)/2 - GAP + 0.05)

    # ── Row 1: Input dicts ────────────────────────────────────────────
    r1 = r0 - BH - GAP - 0.1
    step_tag(ax, 0.05, r1, "INPUT\nDICTS")
    pbox(ax, 6.5, r1, 12.6, BH,
         "Build  panel_spec  +  weather_now",
         "panel_spec: eff * num_panels  area  beta  noct  inv_eff  battery_wh  batt_dod  load_w  sun_hours\n"
         "weather_now: irradiance  ambient_temp  humidity  wind_speed  hour  month  day_of_year",
         C_SLATE, BD_SLATE, fs=7)
    arr(ax, 4.5, r1 - BH/2, 3.8, r1 - BH/2 - GAP + 0.05)
    arr(ax, 8.5, r1 - BH/2, 9.2, r1 - BH/2 - GAP + 0.05)

    # ── Row 2: NOCT + Physics formula ─────────────────────────────────
    r2 = r1 - BH - GAP
    step_tag(ax, 0.05, r2, "PHYSICS\nHELPERS")
    pbox(ax, 3.8, r2, 5.8, BH,
         "NOCT Module Temp",
         "T_mod = T_amb + (NOCT-20)/800 * G\n(IEC 61215 standard)",
         C_INDIGO, BD_INDIGO, fs=7)
    pbox(ax, 9.2, r2, 5.8, BH,
         "Physics Baseline  P_ideal",
         "eta * A * G * [1 - beta*(T_mod-25)]\ntheoretical max (no losses)",
         C_INDIGO, BD_INDIGO, fs=7)
    for x in [3.8, 9.2]:
        arr(ax, x, r2 - BH/2, 6.5, r2 - BH/2 - GAP + 0.05)

    # ── Row 3: predict_all_models ─────────────────────────────────────
    r3 = r2 - BH - GAP
    step_tag(ax, 0.05, r3, "ML\nINFERENCE")
    pbox(ax, 6.5, r3, 12.6, BH,
         "predict_all_models(panel_spec, weather_now)",
         "12-feature row  ->  scaler.transform([row])  ->  model.predict(x_sc)  for each loaded model",
         C_AMBER, BD_AMBER, fs=7.5)
    arr(ax, 4.0, r3 - BH/2, 4.0, r3 - BH/2 - GAP + 0.05)
    arr(ax, 9.0, r3 - BH/2, 9.0, r3 - BH/2 - GAP + 0.05)

    # ── Row 4: Ensemble + Adjustment ──────────────────────────────────
    r4 = r3 - BH - GAP
    step_tag(ax, 0.05, r4, "ENSEMBLE\n& ADJUST")
    pbox(ax, 4.0, r4, 5.8, BH,
         "Ensemble Average  ens_pdc",
         "mean of all model predictions",
         C_AMBER, BD_AMBER, fs=7)
    pbox(ax, 9.0, r4, 5.8, BH,
         "Real-World Adjusted  adjusted_pdc",
         "ens_pdc * (100 - soiling - aging - wiring - shading) / 100",
         C_CORAL, BD_CORAL, fs=7)
    for x in [4.0, 9.0]:
        arr(ax, x, r4 - BH/2, 6.5, r4 - BH/2 - GAP + 0.05)

    # ── Row 5: Backup calc ────────────────────────────────────────────
    r5 = r4 - BH - GAP
    step_tag(ax, 0.05, r5, "ENERGY &\nBACKUP")
    pbox(ax, 6.5, r5, 12.6, BH,
         "compute_backup(adjusted_pdc, panel_spec)",
         "P_ac = P_dc * inv_eff   |   E_panel = P_dc * sun_hrs   |   E_batt = Wh * DoD * inv_eff   |   backup_hrs = E_total / load_W",
         C_TEAL, BD_TEAL, fs=7)
    arr(ax, 6.5, r5 - BH/2, 6.5, r5 - BH/2 - GAP + 0.05)

    # ── Row 6: Dashboard ─────────────────────────────────────────────
    r6 = r5 - BH - GAP
    step_tag(ax, 0.05, r6, "DASHBOARD\nOUTPUTS")
    dash = [
        (1.8,  "Metrics Row",    "Per-model W\nEnsemble  Adjusted  Backup",    C_AMBER, BD_AMBER),
        (4.4,  "Loss Breakdown", "Physics vs adjusted\nSoiling Aging Wiring Shade", C_CORAL, BD_CORAL),
        (7.0,  "Output Gauge",   "Adjusted vs rated Wp\ndelta = formula-adjusted", C_INDIGO, BD_INDIGO),
        (9.6,  "Temp & Status",  "T_mod  thermal loss %\nsite loss banner",         C_TEAL, BD_TEAL),
        (12.1, "Backup Chart",   "E_panel  E_batt\nE_total Wh bar chart",           C_GREEN, BD_GREEN),
    ]
    for x, t, s, fc, bc in dash:
        pbox(ax, x, r6, 2.3, BH + 0.1, t, s, fc, bc, fs=6.8)

    # ── Row 7: Analysis tabs ──────────────────────────────────────────
    r7 = r6 - BH - GAP - 0.1
    step_tag(ax, 0.05, r7, "ANALYSIS\nTABS")
    tabs = [
        (1.8,  "Hourly Profile",       "ML  ideal  adjusted  AC\nkWh/day estimate",        C_PURPLE, BD_PURPLE),
        (4.4,  "Irradiance\nSensitivity", "G 0->1100 W/m2 sweep\nML vs formula vs adjusted", C_PURPLE, BD_PURPLE),
        (7.0,  "Temp\nSensitivity",    "T 0->50 deg C sweep\nML vs formula",                C_PURPLE, BD_PURPLE),
        (9.6,  "Formula\nvs ML",       "Test scatter  R2\nformula overestimates",            C_PURPLE, BD_PURPLE),
        (12.1, "Model\nPerformance",   "MAPE>50W  R2  RMSE\nFeature importance RF",          C_PURPLE, BD_PURPLE),
    ]
    for x, t, s, fc, bc in tabs:
        pbox(ax, x, r7, 2.3, BH + 0.1, t, s, fc, bc, fs=6.8)

    plt.tight_layout(rect=[0.06, 0, 1, 0.95])
    fig.savefig("outputs/apppipe.png", dpi=200,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  [OK] outputs/apppipe.png")


# ══════════════════════════════════════════════════════════════════════
# DIAGRAM 3  -  Combined end-to-end
# ══════════════════════════════════════════════════════════════════════
def draw_final():
    FW, FH = 13, 16
    CW, CH = 13, 16
    fig, ax = plt.subplots(figsize=(FW, FH))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, CW); ax.set_ylim(0, CH); ax.axis("off")

    fig.text(0.5, 0.99,
             "SolarCast — End-to-End ML System Pipeline",
             ha="center", va="top",
             fontsize=11, fontweight="bold", color=C_NAVY)
    fig.text(0.5, 0.975,
             "solar_pipeline.py  (offline training)                                   app.py  (live dashboard)",
             ha="center", va="top", fontsize=6.5, color="#64748B")

    DIVIDER_Y = CH / 2 + 0.3      # horizontal divider between training & dashboard halves
    BH   = 0.52     # box height
    GAP  = 0.36     # row vertical gap
    MID  = CW / 2   # 6.5

    # ── DIVIDER LINE ──────────────────────────────────────────────────
    ax.plot([0.45, CW - 0.1], [DIVIDER_Y, DIVIDER_Y],
            color="#CBD5E1", lw=1.2, ls="--", zorder=1)
    ax.text(0.5, DIVIDER_Y + 0.08,
            "solar_pipeline.py  (training & artifact generation)",
            fontsize=5.5, color=BD_SLATE, va="bottom", fontweight="bold")
    ax.text(0.5, DIVIDER_Y - 0.08,
            "app.py  (Streamlit dashboard  -  inference & visualisation)",
            fontsize=5.5, color=BD_AMBER, va="top", fontweight="bold")

    # ═══════════════════════
    # TOP  -  solar_pipeline.py
    # ═══════════════════════
    top = CH - 0.70

    # Row A: 4 input sources
    rA = top
    step_tag(ax, 0.05, rA, "INPUTS")
    for x, t, s in [
        (2.0,  "Weather",        "G  T_amb  humidity  wind"),
        (4.5,  "Panel Specs",    "eta  area  beta  NOCT"),
        (7.0,  "Time Features",  "hour  month  day_of_year"),
        (9.5,  "NOCT + Formula", "T_mod  ->  P_ideal"),
    ]:
        pbox(ax, x, rA, 2.25, BH, t, s, C_BLUE, BD_BLUE, fs=6.8)
        arr(ax, x, rA - BH/2, x, rA - BH/2 - GAP + 0.04)

    # Row B: loss factors
    rB = rA - BH - GAP
    step_tag(ax, 0.05, rB, "LOSS\nFACTORS")
    for x, t, s in [
        (1.7,  "Soiling",  "0.80-1.00"),
        (3.9,  "Aging",    "0.92-1.00"),
        (6.1,  "Shading",  "0.50-0.85\n12% prob"),
        (8.3,  "Wiring",   "0.95-0.99"),
        (10.5, "Noise",    "+/-5 W"),
    ]:
        pbox(ax, x, rB, 2.0, BH + 0.04, t, s, C_CORAL, BD_CORAL, fs=6.5)
        arr(ax, x, rB - (BH+0.04)/2, MID, rB - (BH+0.04)/2 - GAP + 0.04,
            color=BD_CORAL, lw=0.7)

    # Row C: dataset box
    rC = rB - BH - GAP - 0.04
    step_tag(ax, 0.05, rC, "DATASET")
    pbox(ax, MID, rC, 12.4, BH,
         "50,000 rows  x  12 features  +  dc_power target",
         "P_dc = P_ideal x soiling x aging x shading x wiring + noise   ->   data/solar_dataset.csv",
         C_INDIGO, BD_INDIGO, fs=7.2)
    arr(ax, 3.5, rC - BH/2, 3.0, rC - BH/2 - GAP + 0.04)
    arr(ax, 9.5, rC - BH/2, 10.0, rC - BH/2 - GAP + 0.04)

    # Row D: EDA + Preprocessing
    rD = rC - BH - GAP
    step_tag(ax, 0.05, rD, "EDA &\nPREPROC")
    pbox(ax, 3.0, rD, 5.4, BH,
         "EDA  (Figures A, B, C)",
         "Distributions  Correlations  Stats",
         C_PURPLE, BD_PURPLE, fs=6.8)
    pbox(ax, 10.0, rD, 5.4, BH,
         "Preprocessing",
         "MinMaxScaler  80/20 split  MAPE=50W",
         C_TEAL, BD_TEAL, fs=6.8)
    for x in [3.0, 10.0]:
        arr(ax, x, rD - BH/2, MID, rD - BH/2 - GAP + 0.04)

    # Row E: scaled sets
    rE = rD - BH - GAP
    step_tag(ax, 0.05, rE, "SCALED\nSETS")
    pbox(ax, MID, rE, 12.4, BH,
         "X_train_sc   y_train    |    X_test_sc   y_test",
         "12 features 0-1  |  models/scaler.pkl saved",
         C_SLATE, BD_SLATE, fs=7.2)
    arr(ax, MID, rE - BH/2, MID, rE - BH/2 - GAP + 0.04)

    # Row F: 5 models
    rF = rE - BH - GAP
    step_tag(ax, 0.05, rF, "5 MODELS")
    for x, t in [(1.8, "Random\nForest"), (3.9, "Gradient\nBoost"),
                 (6.1, "MLP\n/ ANN"),  (8.3, "XGBoost\nor SVR"),
                 (10.5,"LightGBM\nor ET")]:
        arr(ax, x, rE - BH/2, x, rF + (BH+0.06)/2, color=BD_AMBER, lw=0.7)
        pbox(ax, x, rF, 1.9, BH + 0.06, t, "", C_AMBER, BD_AMBER, fs=6.8)
        arr(ax, x, rF - (BH+0.06)/2, MID, rF - (BH+0.06)/2 - GAP + 0.04,
            color=BD_AMBER, lw=0.7)

    # Row G: Evaluation + save
    rG = rF - BH - GAP - 0.06
    step_tag(ax, 0.05, rG, "EVAL &\nSAVE")
    pbox(ax, MID, rG, 12.4, BH,
         "Evaluation:  R2   RMSE   MAE   MAPE(>50W)   Residuals   Feature Importance",
         "Best model by R2  ->  models/best_model.pkl   |   All models  ->  models/*.pkl   |   outputs/model_comparison.png",
         C_GREEN, BD_GREEN, fs=7)

    # ── artifact handoff arrows (cross divider) ───────────────────────
    arr(ax, 3.0, rG - BH/2, 3.0, DIVIDER_Y - 0.02,
        color=BD_GREEN, lw=1.4)
    arr(ax, 10.0, rG - BH/2, 10.0, DIVIDER_Y - 0.02,
        color=BD_TEAL, lw=1.4)
    ax.text(3.2, (rG - BH/2 + DIVIDER_Y) / 2,
            "models/*.pkl\nscaler.pkl", fontsize=5.2,
            color=BD_GREEN, va="center", style="italic")
    ax.text(10.2, (rG - BH/2 + DIVIDER_Y) / 2,
            "solar_dataset.csv", fontsize=5.2,
            color=BD_TEAL, va="center", style="italic")

    # ═══════════════════════
    # BOTTOM  -  app.py
    # ═══════════════════════
    base = DIVIDER_Y - 0.45

    # Row H: sidebar inputs
    rH = base
    step_tag(ax, 0.05, rH, "SIDEBAR\nINPUTS")
    for x, t, s in [
        (1.7,  "Panel\nPreset & Specs",    "Rated Wp  eta  area\nbeta  NOCT  #panels"),
        (4.25, "Real-World\nConditions",   "Soiling  Aging\nWiring  Shading %"),
        (6.8,  "System\nConfig",           "Inverter  Battery\nLoad W  Sun hrs"),
        (9.35, "Weather",                  "G  T_amb  humidity\nwind  hour  month  DoY"),
        (11.9, "Models\n(cached)",         "scaler.pkl\n7 model .pkl files"),
    ]:
        pbox(ax, x, rH, 2.2, BH + 0.08, t, s, C_BLUE, BD_BLUE, fs=6.2)
        arr(ax, x, rH - (BH+0.08)/2, x, rH - (BH+0.08)/2 - GAP + 0.04)

    # Row I: predict_all_models
    rI = rH - BH - GAP - 0.08
    step_tag(ax, 0.05, rI, "ML\nINFERENCE")
    pbox(ax, MID, rI, 12.4, BH,
         "predict_all_models(panel_spec, weather_now)",
         "12-feature row  ->  scaler.transform  ->  model.predict  for all loaded models  +  T_mod via NOCT",
         C_AMBER, BD_AMBER, fs=7.2)
    arr(ax, 4.0, rI - BH/2, 4.0, rI - BH/2 - GAP + 0.04)
    arr(ax, 9.0, rI - BH/2, 9.0, rI - BH/2 - GAP + 0.04)

    # Row J: ensemble + adjustment
    rJ = rI - BH - GAP
    step_tag(ax, 0.05, rJ, "ENSEMBLE\n& ADJUST")
    pbox(ax, 4.0, rJ, 5.4, BH,
         "Ensemble Average  ens_pdc",
         "mean of all model predictions",
         C_AMBER, BD_AMBER, fs=6.8)
    pbox(ax, 9.0, rJ, 5.4, BH,
         "Real-World Adjustment  adjusted_pdc",
         "ens_pdc * (100 - site_losses%) / 100",
         C_CORAL, BD_CORAL, fs=6.8)
    for x in [4.0, 9.0]:
        arr(ax, x, rJ - BH/2, MID, rJ - BH/2 - GAP + 0.04)

    # Row K: backup
    rK = rJ - BH - GAP
    step_tag(ax, 0.05, rK, "ENERGY\n& BACKUP")
    pbox(ax, MID, rK, 12.4, BH,
         "compute_backup()  ->  P_ac   E_panel   E_batt   E_total   backup_hrs",
         "P_ac = P_dc * inv_eff  |  E_panel = P_dc * sun_hrs  |  backup = E_total / load_W",
         C_TEAL, BD_TEAL, fs=7.2)
    arr(ax, MID, rK - BH/2, MID, rK - BH/2 - GAP + 0.04)

    # Row L: dashboard widgets
    rL = rK - BH - GAP
    step_tag(ax, 0.05, rL, "DASHBOARD")
    for x, t, s in [
        (1.7,  "Metrics Row",     "W per model\nEnsemble  Adjusted  Backup"),
        (4.25, "Loss Expander",   "Physics vs adjusted\nSoiling Aging Wiring Shade"),
        (6.8,  "Output Gauge",    "Adjusted vs rated Wp"),
        (9.35, "Temp & Status",   "T_mod  thermal loss\nsite loss banner"),
        (11.9, "Backup Chart",    "E_panel  E_batt\nE_total Wh bars"),
    ]:
        pbox(ax, x, rL, 2.2, BH + 0.08, t, s, C_GREEN, BD_GREEN, fs=6.2)

    # Row M: analysis tabs
    rM = rL - BH - GAP - 0.08
    step_tag(ax, 0.05, rM, "ANALYSIS\nTABS")
    for x, t, s in [
        (1.7,  "Hourly\nProfile",     "ML  ideal  adjusted\nkWh/day"),
        (4.25, "Irradiance\nSensitivity", "G sweep 0-1100 W/m2\nML vs formula"),
        (6.8,  "Temp\nSensitivity",   "T sweep 0-50 C\nML vs formula"),
        (9.35, "Formula\nvs ML",      "Test scatter + R2\nformula overestimates"),
        (11.9, "Model\nPerformance",  "MAPE>50W  R2  RMSE\nFeature importance RF"),
    ]:
        pbox(ax, x, rM, 2.2, BH + 0.08, t, s, C_PURPLE, BD_PURPLE, fs=6.2)

    plt.tight_layout(rect=[0.06, 0, 1, 0.97])
    fig.savefig("outputs/Finalpipeline.png", dpi=200,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  [OK] outputs/Finalpipeline.png")


# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating compact light-theme pipeline diagrams ...")
    draw_solar()
    draw_app()
    draw_final()
    print("Done!  All 3 saved to outputs/")
