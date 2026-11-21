#!/usr/bin/env python3
"""
Mechanical-descriptor ablation on frozen folds — class-balanced version
=======================================================================

Same design as run_mechanical_ablation.py, with two changes so results line
up with tune_rf_nested_cv.py:

    * every model is fitted with balanced class weights
        rf   class_weight="balanced"
        gbr  sample_weight = compute_sample_weight("balanced")
        xgb  sample_weight = compute_sample_weight("balanced")
    * metrics are reported at threshold 0.5:
        macro_f1, f1_pos, accuracy, balanced_accuracy, roc_auc, pr_auc, brier

Outputs go to <feature_set>_balanced/<level>/ so the original AUC-only run
is left untouched.

Question
--------
Do MLIP-derived elastic descriptors add anything on top of the geometric +
RAC baseline?

Design
------
Two feature sets are trained on EXACTLY the same rows and the same folds:

    rac             geometric(10) + RAC(176)                = 186
    rac_mechanical  geometric(10) + RAC(176) + mechanical(5) = 191

Rows are restricted to CIFs that have complete values in BOTH sets, so any
difference in AUC comes from the extra descriptors and not from a change in
sample. Comparing the 186-feature model on 2,533 rows against the
191-feature model on 2,307 rows would confound the two.

Folds are read from the existing ladder directories and are NOT regenerated.
A fold is a property of the CIF's group, not of feature availability, so
dropping rows leaves every group intact within its fold; this is asserted at
runtime.

Mechanical features
-------------------
    K_vrh        isotropic bulk modulus
    G_vrh        shear modulus
    anisotropy   directional stiffness contrast
    lambda_min   lowest Kelvin eigenvalue
    born_stable  soft-mode criticality flag

The raw stiffness components C11..C66 are not used; the five scalars above
already summarise the tensor.

relax_fmax, dV_relax and collapse are deliberately excluded: they define the
convergence filter, so using them as predictors would feed the selection
criterion back into the model.

Caveat
------
Filter membership is not independent of the label
(solvent p=0.03, temperature p=0.004, gas p=0.03; gas retains proportionally
fewer flexible MOFs). Report this alongside any gain.

Layout
------
    rf_with_mechanical_sorted/
      CR_Main_Mechanical_pp_final.csv
      run_mechanical_ablation_balanced.py
      rac_balanced/<level>/{folds,train_test,results}/
      rac_mechanical_balanced/<level>/{folds,train_test,results}/
      summary_all_levels_balanced.csv

Usage
-----
    python run_mechanical_ablation_balanced.py
    python run_mechanical_ablation_balanced.py --levels 5_comb --stimuli gas
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from common.data import read_labels, read_features, read_mech, read_master  # released-data loaders


warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- paths
TRAIN_ROOT = "../results/folds"          # frozen folds from model/make_folds.py
LABEL_FILE = "../CR_Main_LLM_Extraction_overall.xlsx"
FEAT_FILE = "../features_master.csv"
MASTER_FILE = "../master_groups.csv"
MECH_FILE = "CR_Main_Mechanical_pp_final.csv"

# ---------------------------------------------------------------- features
GEOM = ["LCD", "PLD", "density", "VF", "ASA", "VSA", "PV", "LFPD", "GSA", "GPV"]
MECH = ["K_vrh", "G_vrh", "anisotropy", "lambda_min", "born_stable"]

STIMULI = {
    "solvent": "Overall_Flexibility_solvent",
    "temperature": "Overall_Flexibility_temperature",
    "gas": "Overall_Flexibility_gas adsorption/desorption",
}
LEVELS = {
    "1_random": None,
    "2_mofkey": "material_id",
    "3_chemistry": "chemistry_id",
    "4_doi": "doi",
    "5_comb": "combined_id",
}

N_FOLDS = 5
SEED = 20260724
THR = 0.5
RF_KW = dict(n_estimators=500, min_samples_leaf=2, max_features="sqrt",
             bootstrap=False, class_weight="balanced", n_jobs=-1,
             random_state=SEED)
GBR_KW = dict(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=SEED)
XGB_KW = dict(n_estimators=400, max_depth=4, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
              random_state=SEED, n_jobs=-1)


# ---------------------------------------------------------------- loading
def load_inputs():
    lab = read_labels()
    feat = read_features()
    mech = read_mech()

    for name, cols, df in [("geometric", GEOM, lab), ("mechanical", MECH, mech)]:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise SystemExit(f"[error] {name} columns missing: {missing}")

    rac = [c for c in feat.columns if c.startswith("rac_")]
    if not rac:
        raise SystemExit(f"[error] no rac_* columns in {FEAT_FILE}")

    table = (lab[["cif_name"] + GEOM]
             .merge(feat[["cif_name"] + rac], on="cif_name", how="inner")
             .merge(mech[["cif_name"] + MECH], on="cif_name", how="inner"))

    sets = {"rac": GEOM + rac, "rac_mechanical": GEOM + rac + MECH}
    print(f"[load] geometric {len(GEOM)} | rac {len(rac)} | mechanical {len(MECH)}")
    for k, v in sets.items():
        print(f"       {k:16s} {len(v)} features")

    # the common subset: complete in the largest feature set
    full = sets["rac_mechanical"]
    before = len(table)
    table = table.dropna(subset=full).reset_index(drop=True)
    print(f"[load] CIFs with complete geom+rac+mech: {len(table)} "
          f"(dropped {before - len(table)} incomplete)")
    return lab, table, sets


def label_table(lab, stim):
    col = STIMULI[stim]
    s = lab[["cif_name", col]].copy()
    s[col] = s[col].astype(str).str.strip()
    s = s[s[col].isin(["Yes", "No"])]
    s["y"] = (s[col] == "Yes").astype(int)
    return s[["cif_name", "y"]]


def load_folds(level, stim):
    """Read the frozen fold assignment produced by the ladder run."""
    path = os.path.join(TRAIN_ROOT, level, "folds", f"{stim}_folds.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)[["cif_name", "fold"]]


def check_fold_integrity(data, master, group_col, tag):
    """No group may straddle two folds after rows are dropped."""
    if group_col is None:
        return
    j = data.merge(master[["cif_name", group_col]], on="cif_name", how="left")
    j["g"] = j[group_col].where(j[group_col].notna(), "NA::" + j["cif_name"])
    crossed = (j.groupby("g")["fold"].nunique() > 1).sum()
    if crossed:
        raise SystemExit(f"[error] {tag}: {crossed} group(s) span multiple folds")


# ---------------------------------------------------------------- models
def get_models():
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    models = {"rf": RandomForestClassifier(**RF_KW),
              "gbr": GradientBoostingClassifier(**GBR_KW)}
    try:
        from xgboost import XGBClassifier
        models["xgb"] = XGBClassifier(**XGB_KW)
    except ImportError:
        print("[warn] xgboost not installed -> skipping xgb")
    return models


def metrics(y, p, thr=THR):
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 brier_score_loss, f1_score, accuracy_score,
                                 balanced_accuracy_score)
    keys = ["macro_f1", "f1_pos", "accuracy", "balanced_accuracy",
            "roc_auc", "pr_auc", "brier"]
    if len(np.unique(y)) < 2:
        return dict(n=len(y), pos=int(y.sum()), **{k: np.nan for k in keys})
    yhat = (p >= thr).astype(int)
    return dict(n=len(y), pos=int(y.sum()),
                macro_f1=f1_score(y, yhat, average="macro"),
                f1_pos=f1_score(y, yhat),
                accuracy=accuracy_score(y, yhat),
                balanced_accuracy=balanced_accuracy_score(y, yhat),
                roc_auc=roc_auc_score(y, p),
                pr_auc=average_precision_score(y, p),
                brier=brier_score_loss(y, p))


# ---------------------------------------------------------------- driver
def run(fs_name, features, level, group_col, lab, table, master, stimuli):
    out = os.path.join(fs_name + "_balanced", level)
    for sub in ("folds", "train_test", "results"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    models = get_models()
    oof_rows, summary_rows = [], {m: [] for m in models}
    summaries = []

    for stim in stimuli:
        folds = load_folds(level, stim)
        if folds is None:
            print(f"[skip] {level}/{stim}: fold file not found")
            continue

        data = (label_table(lab, stim)
                .merge(folds, on="cif_name", how="inner")
                .merge(table, on="cif_name", how="inner")
                .dropna(subset=features)
                .reset_index(drop=True))
        check_fold_integrity(data, master, group_col, f"{fs_name}/{level}/{stim}")
        data[["cif_name", "y", "fold"]].to_csv(
            os.path.join(out, "folds", f"{stim}_folds.csv"), index=False)

        X, y = data[features].values, data["y"].values
        f, names = data["fold"].values, data["cif_name"].values
        oof = {m: np.full(len(data), np.nan) for m in models}

        for k in range(N_FOLDS):
            tr, te = f != k, f == k
            if te.sum() == 0:
                continue
            pd.Series(names[tr], name="cif_name").to_csv(
                os.path.join(out, "train_test", f"{stim}_fold{k}_train.csv"), index=False)
            pd.Series(names[te], name="cif_name").to_csv(
                os.path.join(out, "train_test", f"{stim}_fold{k}_test.csv"), index=False)
            if len(np.unique(y[tr])) < 2:
                continue
            from sklearn.base import clone
            from sklearn.utils.class_weight import compute_sample_weight
            sw = compute_sample_weight("balanced", y[tr])
            for mname, model in models.items():
                if mname == "rf":
                    m = clone(model).fit(X[tr], y[tr])       # class_weight inside
                else:
                    m = clone(model).fit(X[tr], y[tr], sample_weight=sw)
                p = m.predict_proba(X[te])[:, 1]
                oof[mname][te] = p
                summary_rows[mname].append(dict(stimulus=stim, fold=k, **metrics(y[te], p)))

        for mname in models:
            ok = ~np.isnan(oof[mname])
            pooled = metrics(y[ok], oof[mname][ok])
            fa = [r["roc_auc"] for r in summary_rows[mname]
                  if r["stimulus"] == stim and not np.isnan(r["roc_auc"])]
            ff = [r["macro_f1"] for r in summary_rows[mname]
                  if r["stimulus"] == stim and not np.isnan(r["macro_f1"])]
            summaries.append(dict(
                feature_set=fs_name, level=level, stimulus=stim, model=mname,
                thr=THR,
                pooled_macro_f1=pooled["macro_f1"], pooled_f1_pos=pooled["f1_pos"],
                pooled_accuracy=pooled["accuracy"],
                pooled_balanced_accuracy=pooled["balanced_accuracy"],
                pooled_roc_auc=pooled["roc_auc"], pooled_pr_auc=pooled["pr_auc"],
                pooled_brier=pooled["brier"],
                fold_f1_std=float(np.std(ff)) if ff else np.nan,
                fold_auc_std=float(np.std(fa)) if fa else np.nan,
                n=pooled["n"], pos=pooled["pos"]))
            for i in range(len(data)):
                if ok[i]:
                    oof_rows.append(dict(cif_name=names[i], stimulus=stim,
                                         model=mname, fold=int(f[i]),
                                         y=int(y[i]), score=float(oof[mname][i])))

    for mname, rows in summary_rows.items():
        if rows:
            pd.DataFrame(rows).to_csv(
                os.path.join(out, "results", f"{mname}_metrics_by_fold.csv"), index=False)
    if oof_rows:
        pd.DataFrame(oof_rows).to_csv(
            os.path.join(out, "results", "oof_predictions.csv"), index=False)
    s = pd.DataFrame(summaries)
    if not s.empty:
        s.to_csv(os.path.join(out, "results", "summary.csv"), index=False)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="*", default=list(LEVELS))
    ap.add_argument("--stimuli", nargs="*", default=list(STIMULI))
    ap.add_argument("--feature-sets", nargs="*",
                    default=["rac", "rac_mechanical"])
    args = ap.parse_args()

    lab, table, sets = load_inputs()
    master = read_master()

    all_s = []
    for fs_name in args.feature_sets:
        if fs_name not in sets:
            print(f"[skip] unknown feature set {fs_name}")
            continue
        print(f"\n########## {fs_name} ({len(sets[fs_name])} features) ##########")
        for level in args.levels:
            if level not in LEVELS:
                print(f"[skip] unknown level {level}")
                continue
            s = run(fs_name, sets[fs_name], level, LEVELS[level],
                    lab, table, master, args.stimuli)
            if s.empty:
                continue
            all_s.append(s)
            print(f"\n=== {fs_name} / {level} ===")
            print(s[["stimulus", "model", "pooled_macro_f1", "pooled_accuracy",
                     "pooled_roc_auc", "n"]].round(3).to_string(index=False))

    if not all_s:
        return
    comb = pd.concat(all_s, ignore_index=True)
    comb.to_csv("summary_all_levels_balanced.csv", index=False)
    print("\n[written] summary_all_levels_balanced.csv")

    rf = comb[comb.model == "rf"]
    for metric in ("pooled_roc_auc", "pooled_macro_f1"):
        piv = rf.pivot_table(index=["level", "stimulus"], columns="feature_set",
                             values=metric)
        if {"rac", "rac_mechanical"} <= set(piv.columns):
            piv["delta"] = piv["rac_mechanical"] - piv["rac"]
            print(f"\n=== RF: mechanical gain, {metric} (same rows, same folds) ===")
            print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
