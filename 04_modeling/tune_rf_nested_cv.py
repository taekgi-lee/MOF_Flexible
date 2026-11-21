#!/usr/bin/env python3
"""
Nested-CV hyper-parameter tuning for the Random Forest — macro-F1 objective
=========================================================================

Same protocol as tune_rf.py (AUC version); two changes:

    RandomForestClassifier(class_weight="balanced")   fixed
    RandomizedSearchCV(scoring="f1_macro")            objective

class_weight="balanced" re-weights the minority class during fitting so that
the 0.5 decision threshold is meaningful; macro-F1 is then evaluated at 0.5.
This mirrors the earlier common_rf / rfecv_tune pipeline.

    RandomizedSearchCV, n_iter=40, scoring=f1_macro
    inner CV     StratifiedGroupKFold(4) when the level defines groups,
                 StratifiedKFold(4) for the random level
    tuned        n_estimators      200..1000
                 max_depth         None, 8, 12, 16, 24, 32
                 min_samples_leaf  1..8
                 max_features      sqrt, log2, 0.3, 0.5, 0.7
    fixed        bootstrap=False, class_weight="balanced", random_state=SEED

Reported (pooled OOF and per fold, all at threshold 0.5)
    macro_f1, f1_pos, accuracy, balanced_accuracy, roc_auc, pr_auc, brier

Why nested
----------
The outer fold is held out and predicted once; it never influences which
parameters are chosen. The inner split is stratified AND grouped, so a
material cannot sit in both the inner-train and inner-validation portion.

Output
------
    rac_mechanical_tune_f1/<level>/results/rf_best_params.csv
    rac_mechanical_tune_f1/<level>/results/rf_tuned_metrics_by_fold.csv
    rac_mechanical_tune_f1/<level>/results/oof_predictions.csv
    rac_mechanical_tune_f1/<level>/results/summary.csv
    rac_mechanical_tune_f1/best_params_all_levels.csv
    rac_mechanical_tune_f1/tuning_summary.csv

Mechanical descriptors are included automatically when CR_Main_Mechanical_pp_final.csv
is present in the working directory (same as tune_rf.py).

Usage
-----
    python tune_rf_nested_cv.py --levels 1_random 2_mofkey 3_chemistry 4_doi 5_comb
    python tune_rf_nested_cv.py --levels 2_mofkey --stimuli gas
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import randint
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from common.data import read_labels, read_features, read_mech, read_master  # released-data loaders


warnings.filterwarnings("ignore")

TRAIN_ROOT = "../results/folds"          # frozen folds from model/make_folds.py
LABEL_FILE = "../CR_Main_LLM_Extraction_overall.xlsx"
FEAT_FILE = "../features_master.csv"
MASTER_FILE = "../master_groups.csv"
MECH_FILE = "CR_Main_Mechanical_pp_final.csv"

GEOM = ["LCD", "PLD", "density", "VF", "ASA", "VSA", "PV", "LFPD", "GSA", "GPV"]
MECH = ["K_vrh", "G_vrh", "anisotropy", "lambda_min", "born_stable"]

STIMULI = {
    "solvent": "Overall_Flexibility_solvent",
    "temperature": "Overall_Flexibility_temperature",
    "gas": "Overall_Flexibility_gas adsorption/desorption",
}
LEVELS = {"1_random": None, "2_mofkey": "material_id",
          "3_chemistry": "chemistry_id", "4_doi": "doi",
          "5_comb": "combined_id"}

N_FOLDS = 5
SEED = 20260724
THR = 0.5

# ---- tuning settings ----
N_ITER = 40
INNER_FOLDS = 4
SCORING = "f1_macro"
SEARCH_JOBS = -1
RF_JOBS = 1           # single-threaded forest; the search parallelises instead

PARAM_DIST = {
    "n_estimators": randint(200, 1001),
    "max_depth": [None, 8, 12, 16, 24, 32],
    "min_samples_leaf": randint(1, 9),
    "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
}
RF_FIXED = dict(bootstrap=False, class_weight="balanced",
                random_state=SEED, n_jobs=RF_JOBS)


# ---------------------------------------------------------------- data
def load_inputs():
    lab = read_labels()
    feat = read_features()
    master = read_master()

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
    return lab, table, master, cols


def assemble(lab, table, cols, master, level, stim):
    fp = os.path.join(TRAIN_ROOT, level, "folds", f"{stim}_folds.csv")
    if not os.path.exists(fp):
        return None
    col = STIMULI[stim]
    s = lab[["cif_name", col]].copy()
    s[col] = s[col].astype(str).str.strip()
    s = s[s[col].isin(["Yes", "No"])]
    s["y"] = (s[col] == "Yes").astype(int)
    folds = pd.read_csv(fp)[["cif_name", "fold"]]
    gcol = LEVELS[level]
    keep = ["cif_name"] + ([gcol] if gcol else [])
    return (s[["cif_name", "y"]]
            .merge(folds, on="cif_name", how="inner")
            .merge(table, on="cif_name", how="inner")
            .merge(master[keep], on="cif_name", how="left")
            .dropna(subset=cols).reset_index(drop=True))


def make_search(group_col, n_iter, inner_folds):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                         StratifiedGroupKFold)
    base = RandomForestClassifier(**RF_FIXED)
    if group_col is None:
        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True,
                                random_state=SEED)
    else:
        inner = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True,
                                     random_state=SEED)
    return RandomizedSearchCV(base, PARAM_DIST, n_iter=n_iter, scoring=SCORING,
                              cv=inner, random_state=SEED, n_jobs=SEARCH_JOBS,
                              refit=True, error_score=np.nan)


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
def run_level(level, lab, table, master, cols, stimuli, outdir, n_iter, inner_folds):
    group_col = LEVELS[level]
    rdir = os.path.join(outdir, level, "results")
    os.makedirs(rdir, exist_ok=True)

    fold_rows, summary_rows, best_rows, oof_rows = [], [], [], []

    for stim in stimuli:
        data = assemble(lab, table, cols, master, level, stim)
        if data is None:
            print(f"[skip] {level}/{stim}: fold file not found")
            continue

        X, y = data[cols].values, data["y"].values
        f, names = data["fold"].values, data["cif_name"].values
        groups = (data[group_col].fillna("NA::" + data["cif_name"]).values
                  if group_col else None)
        oof = np.full(len(data), np.nan)

        for k in range(N_FOLDS):
            te = f == k
            tr = ~te
            if te.sum() == 0 or len(np.unique(y[tr])) < 2:
                continue
            search = make_search(group_col, n_iter, inner_folds)
            if group_col:
                search.fit(X[tr], y[tr], groups=groups[tr])
            else:
                search.fit(X[tr], y[tr])

            p = search.best_estimator_.predict_proba(X[te])[:, 1]
            oof[te] = p
            mt = metrics(y[te], p)
            fold_rows.append(dict(stimulus=stim, fold=k, **mt))

            bp = search.best_params_
            best_rows.append(dict(
                level=level, stimulus=stim, fold=k,
                n_estimators=bp.get("n_estimators"), max_depth=bp.get("max_depth"),
                min_samples_leaf=bp.get("min_samples_leaf"),
                max_features=bp.get("max_features"),
                inner_best_macro_f1=float(search.best_score_),
                test_macro_f1=mt["macro_f1"], test_roc_auc=mt["roc_auc"]))
            print(f"  {level}/{stim} fold {k}: inner_f1={search.best_score_:.4f} "
                  f"test_f1={mt['macro_f1']:.4f} test_auc={mt['roc_auc']:.4f} {bp}",
                  flush=True)

        ok = ~np.isnan(oof)
        pooled = metrics(y[ok], oof[ok])
        ff = [r["macro_f1"] for r in fold_rows
              if r["stimulus"] == stim and not np.isnan(r["macro_f1"])]
        fa = [r["roc_auc"] for r in fold_rows
              if r["stimulus"] == stim and not np.isnan(r["roc_auc"])]
        summary_rows.append(dict(
            level=level, stimulus=stim, model="rf_tuned_f1", thr=THR,
            pooled_macro_f1=pooled["macro_f1"], pooled_f1_pos=pooled["f1_pos"],
            pooled_accuracy=pooled["accuracy"],
            pooled_balanced_accuracy=pooled["balanced_accuracy"],
            pooled_roc_auc=pooled["roc_auc"], pooled_pr_auc=pooled["pr_auc"],
            pooled_brier=pooled["brier"],
            fold_f1_std=float(np.std(ff)) if ff else np.nan,
            fold_auc_std=float(np.std(fa)) if fa else np.nan,
            n=pooled["n"], pos=pooled["pos"]))
        for i in np.where(ok)[0]:
            oof_rows.append(dict(cif_name=names[i], stimulus=stim,
                                 model="rf_tuned_f1", fold=int(f[i]),
                                 y=int(y[i]), score=float(oof[i])))

    if fold_rows:
        pd.DataFrame(fold_rows).to_csv(
            os.path.join(rdir, "rf_tuned_metrics_by_fold.csv"), index=False)
    if best_rows:
        pd.DataFrame(best_rows).to_csv(
            os.path.join(rdir, "rf_best_params.csv"), index=False)
    if oof_rows:
        pd.DataFrame(oof_rows).to_csv(
            os.path.join(rdir, "oof_predictions.csv"), index=False)
    summ = pd.DataFrame(summary_rows)
    if not summ.empty:
        summ.to_csv(os.path.join(rdir, "summary.csv"), index=False)
    return summ, pd.DataFrame(best_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="*", default=["5_comb"])
    ap.add_argument("--stimuli", nargs="*", default=list(STIMULI))
    ap.add_argument("--n_iter", type=int, default=N_ITER)
    ap.add_argument("--inner_folds", type=int, default=INNER_FOLDS)
    ap.add_argument("--out", default="rac_mechanical_tune_f1")
    args = ap.parse_args()

    print(f"[tune] RandomizedSearchCV n_iter={args.n_iter} "
          f"inner={args.inner_folds} scoring={SCORING} class_weight=balanced thr={THR}")
    lab, table, master, cols = load_inputs()
    os.makedirs(args.out, exist_ok=True)

    all_s, all_bp = [], []
    for level in args.levels:
        if level not in LEVELS:
            print(f"[skip] unknown level {level}")
            continue
        s, bp = run_level(level, lab, table, master, cols, args.stimuli,
                          args.out, args.n_iter, args.inner_folds)
        if not s.empty:
            all_s.append(s)
        if not bp.empty:
            all_bp.append(bp)

    if all_s:
        comb = pd.concat(all_s, ignore_index=True)
        comb.to_csv(os.path.join(args.out, "tuning_summary.csv"), index=False)
        print("\n=== tuned RF (f1_macro, balanced), pooled OOF @ 0.5 ===")
        print(comb[["level", "stimulus", "pooled_macro_f1", "pooled_accuracy",
                    "pooled_balanced_accuracy", "pooled_roc_auc",
                    "fold_f1_std", "n"]].round(3).to_string(index=False))
    if all_bp:
        pd.concat(all_bp, ignore_index=True).to_csv(
            os.path.join(args.out, "best_params_all_levels.csv"), index=False)
        print(f"\n[written] {args.out}/tuning_summary.csv "
              f"+ best_params_all_levels.csv")


if __name__ == "__main__":
    main()
