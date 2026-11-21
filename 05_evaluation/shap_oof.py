#!/usr/bin/env python3
"""
TreeSHAP analysis with feature-family grouping
==============================================

Produces the mean |SHAP| bar chart, one bar per feature column under its own
name (rac_metal__D_mc-chi-2-all, topo_td10, LCD, ...).

Out-of-fold SHAP
----------------
For each fold the model is fitted on the training folds and SHAP values are
computed on the held-out fold, then pooled. Explaining training-set rows
would describe what the model memorised rather than how it generalises.

Output
------
    shap/<level>/<stimulus>_shap_importance.csv mean |SHAP| per column
    shap/<level>/<stimulus>_shap_values.npz     full matrix, for re-plotting
    shap/<level>/<stimulus>_shap_bar.png        the figure
    shap/<level>/<stimulus>_beeswarm.png        top ungrouped columns

Usage
-----
    python shap_oof.py --levels 5_comb
    python shap_oof.py --levels 5_comb --params tuning/5_comb/best_params.json
    python shap_oof.py --top 15
"""

import argparse
import json
import os
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from common.data import read_labels, read_features, read_mech, read_master  # released-data loaders


TRAIN_ROOT = "../results/folds"          # frozen folds from model/make_folds.py
LABEL_FILE = "../CR_Main_LLM_Extraction_overall.xlsx"
FEAT_FILE = "../features_master.csv"
MECH_FILE = "CR_Main_Mechanical_pp_final.csv"

GEOM = ["LCD", "PLD", "density", "VF", "ASA", "VSA", "PV", "LFPD", "GSA", "GPV"]
MECH = ["K_vrh", "G_vrh", "anisotropy", "lambda_min", "born_stable"]

STIMULI = {
    "solvent": "Overall_Flexibility_solvent",
    "temperature": "Overall_Flexibility_temperature",
    "gas": "Overall_Flexibility_gas adsorption/desorption",
}
LEVELS = ["1_random", "2_mofkey", "3_chemistry", "4_doi", "5_comb"]

N_FOLDS = 5
SEED = 20260724
BASELINE = dict(n_estimators=500, min_samples_leaf=2, max_features="sqrt",
                bootstrap=False, class_weight="balanced")

BAR_COLOR = "#E8384F"


# ---------------------------------------------------------------- data
def load_features():
    lab = read_labels()
    feat = read_features()
    rac = [c for c in feat.columns if c.startswith("rac_")]
    table = lab[["cif_name"] + GEOM].merge(feat[["cif_name"] + rac],
                                           on="cif_name", how="inner")
    cols = GEOM + rac
    if True:
        mech = read_mech()
        table = table.merge(mech[["cif_name"] + MECH], on="cif_name", how="inner")
        cols = cols + MECH
        print(f"[load] geometric {len(GEOM)} + rac {len(rac)} "
              f"+ mechanical {len(MECH)} = {len(cols)}")
    else:
        print(f"[load] geometric {len(GEOM)} + rac {len(rac)} = {len(cols)} "
              f"({MECH_FILE} not found)")
    table = table.dropna(subset=cols).reset_index(drop=True)
    print(f"[load] {len(table)} CIFs with complete values")
    return lab, table, cols


def assemble(lab, table, cols, level, stim):
    fp = os.path.join(TRAIN_ROOT, level, "folds", f"{stim}_folds.csv")
    if not os.path.exists(fp):
        return None
    col = STIMULI[stim]
    s = lab[["cif_name", col]].copy()
    s[col] = s[col].astype(str).str.strip()
    s = s[s[col].isin(["Yes", "No"])]
    s["y"] = (s[col] == "Yes").astype(int)
    folds = pd.read_csv(fp)[["cif_name", "fold"]]
    return (s[["cif_name", "y"]]
            .merge(folds, on="cif_name", how="inner")
            .merge(table, on="cif_name", how="inner")
            .dropna(subset=cols).reset_index(drop=True))


# ---------------------------------------------------------------- shap
def oof_shap(X, y, f, cols, params):
    """Fit per fold, explain the held-out rows, return the pooled matrix."""
    import shap
    from sklearn.ensemble import RandomForestClassifier

    out = np.zeros((len(X), len(cols)), dtype=float)
    filled = np.zeros(len(X), dtype=bool)

    for k in range(N_FOLDS):
        te = f == k
        if te.sum() == 0 or len(np.unique(y[~te])) < 2:
            continue
        model = RandomForestClassifier(random_state=SEED, n_jobs=-1, **params)
        model.fit(X[~te], y[~te])

        vals = shap.TreeExplainer(model).shap_values(X[te])
        # binary RF returns either a list of two arrays or a 3-D array;
        # take the positive class either way
        if isinstance(vals, list):
            vals = vals[1]
        elif getattr(vals, "ndim", 2) == 3:
            vals = vals[:, :, 1]
        out[te] = vals
        filled[te] = True
        print(f"    fold {k}: explained {te.sum()} rows", flush=True)

    return out[filled], filled


def importance(shap_matrix, cols):
    """Mean |SHAP| per column, largest first."""
    return pd.Series(np.abs(shap_matrix).mean(axis=0),
                     index=cols).sort_values(ascending=False)


def bar_plot(importance, title, path, top=10):
    imp = importance.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 0.34 * len(imp) + 0.9))
    ax.barh(range(len(imp)), imp.values, color=BAR_COLOR,
            edgecolor="white", height=0.72)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp.index, fontsize=8.5)
    ax.set_xlabel("Mean (|SHAP value|)", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def beeswarm_plot(shap_matrix, X, cols, path, top=10):
    import shap
    fig = plt.figure()
    shap.summary_plot(shap_matrix, X, feature_names=cols,
                      max_display=top, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="*", default=["5_comb"])
    ap.add_argument("--stimuli", nargs="*", default=list(STIMULI))
    ap.add_argument("--params", default=None,
                    help="rf_best_params.csv from tune_rf.py; the most "
                         "frequent setting per stimulus is used. "
                         "Omit to use the fixed baseline settings.")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--no-beeswarm", action="store_true")
    ap.add_argument("--out", default="shap")
    args = ap.parse_args()

    try:
        import shap  # noqa: F401
    except ImportError:
        raise SystemExit("[error] pip install shap")

    tuned = {}
    if args.params:
        bp = pd.read_csv(args.params)
        keys = ["n_estimators", "max_depth", "min_samples_leaf", "max_features"]
        for stim, g in bp.groupby("stimulus"):
            # the setting chosen most often across outer folds
            mode = g[keys].apply(lambda r: "|".join(str(v) for v in r), axis=1).mode()[0].split("|")
            p = dict(zip(keys, mode))
            p["n_estimators"] = int(float(p["n_estimators"]))
            p["min_samples_leaf"] = int(float(p["min_samples_leaf"]))
            p["max_depth"] = (None if p["max_depth"] in ("nan", "None")
                              else int(float(p["max_depth"])))
            try:
                p["max_features"] = float(p["max_features"])
            except ValueError:
                pass
            p["bootstrap"] = False
            p["class_weight"] = "balanced"
            tuned[stim] = p
        print(f"[params] loaded from {args.params}: {tuned}")

    lab, table, cols = load_features()

    for level in args.levels:
        if level not in LEVELS:
            print(f"[skip] unknown level {level}")
            continue
        outdir = os.path.join(args.out, level)
        os.makedirs(outdir, exist_ok=True)

        for stim in args.stimuli:
            data = assemble(lab, table, cols, level, stim)
            if data is None:
                print(f"[skip] {level}/{stim}: fold file not found")
                continue
            params = tuned.get(stim, BASELINE)
            print(f"[shap] {level}/{stim}  n={len(data)}  params={params}", flush=True)

            X, y = data[cols].values, data["y"].values
            f = data["fold"].values
            sv, filled = oof_shap(X, y, f, cols, params)
            Xf = X[filled]

            imp = importance(sv, cols)
            imp.rename("mean_abs_shap").to_csv(
                os.path.join(outdir, f"{stim}_shap_importance.csv"))
            np.savez_compressed(os.path.join(outdir, f"{stim}_shap_values.npz"),
                                shap=sv, X=Xf, cols=np.array(cols),
                                cif_name=data.cif_name.values[filled],
                                y=y[filled], fold=f[filled])

            bar_plot(imp, f"{stim}", os.path.join(outdir, f"{stim}_shap_bar.png"),
                     top=args.top)
            if not args.no_beeswarm:
                beeswarm_plot(sv, Xf, cols,
                              os.path.join(outdir, f"{stim}_beeswarm.png"))

            print(f"  top {min(args.top, len(imp))} features:")
            for name, v in imp.head(args.top).items():
                print(f"    {name:26s} {v:.4f}")
            print()

    print(f"[done] figures and tables under {args.out}/")


if __name__ == "__main__":
    main()
