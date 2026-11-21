"""
interval_precision.py - Table 3 with the FINAL hyperparameters held fixed.

Trains the final RF configuration (majority-vote params) on 4 of the 5 frozen
material-split folds, predicts the 5th, pools OOF over the 1,838 gas seeds,
and reports the interval-wise observed yes fraction. The deployed model is not
refit here; only the calibration table is recomputed.

Run from prediction/label_spreading/  (same data files as label_spreading.py):
    python interval_precision.py --folds ../../train/rf_with_mechanical_sorted/rac_mechanical_balanced/2_mofkey/results/oof_predictions.csv

If the OOF file has no fold column, folds are rebuilt with
StratifiedGroupKFold(material_id, shuffle, seed) - set --fold_seed to the value
used in make_folds (default 20260724).

Outputs -> results_calib/
    oof_fixed_params.csv        cif_name, y, fold, P_yes  (seeds)
    table3_fixed_params.csv     interval table
"""
import argparse, os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from common.data import read_labels, read_features, read_mech, read_master  # released-data loaders


LABEL_FILE = "CR_Main_LLM_Extraction_overall.xlsx"
FEAT_FILE = "features_master.csv"
MECH_FILE = "CR_Main_Mechanical_pp_final.csv"
MASTER_FILE = "master_groups.csv"
GEOM = ["LCD", "PLD", "density", "VF", "ASA", "VSA", "PV", "LFPD", "GSA", "GPV"]
MECH = ["K_vrh", "G_vrh", "anisotropy", "lambda_min", "born_stable"]
LABEL_COL = "Overall_Flexibility_gas adsorption/desorption"

# majority-vote hyperparameters (Table S6)
PARAMS = dict(n_estimators=941, max_depth=12, min_samples_leaf=8, max_features="sqrt",
              class_weight="balanced", n_jobs=-1, random_state=0)
BINS = [-1e-9, 0.2, 0.4, 0.6, 0.8, 1.0]
LABELS = ["<=0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", ">0.8"]


def load():
    lab = read_labels()
    feat = read_features()
    mech = read_mech()
    rac = [c for c in feat.columns if c.startswith("rac_")]
    cols = GEOM + rac + MECH
    t = (lab[["cif_name", LABEL_COL] + GEOM]
         .merge(feat[["cif_name"] + rac], on="cif_name")
         .merge(mech[["cif_name"] + MECH], on="cif_name")
         .dropna(subset=cols).reset_index(drop=True))
    t["y"] = t[LABEL_COL].map({"Yes": 1, "No": 0})
    m = read_master()[["cif_name", "material_id"]]
    t = t.merge(m, on="cif_name", how="left")
    t["material_id"] = t.material_id.fillna(t.cif_name)
    print(f"[load] {len(t)} rows, seeds {t.y.notna().sum()}")
    return t, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default=None, help="oof_predictions.csv with cif_name + fold")
    ap.add_argument("--fold_seed", type=int, default=20260724)
    ap.add_argument("--out", default="results_calib")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    t, cols = load()
    seed = t[t.y.notna()].copy()
    X, y, g = seed[cols].to_numpy(float), seed.y.astype(int).to_numpy(), seed.material_id.to_numpy()

    # ---- frozen folds
    fold = None
    if a.folds and os.path.exists(a.folds):
        f = pd.read_csv(a.folds)
        fcol = next((c for c in f.columns if c.lower() in ("fold", "outer_fold", "test_fold")), None)
        if fcol is not None:
            f = f[f.model.str.startswith("rf")] if "model" in f.columns else f
            f = f[f.get("stimulus", "gas") == "gas"] if "stimulus" in f.columns else f
            fold = seed[["cif_name"]].merge(f[["cif_name", fcol]].drop_duplicates(), on="cif_name", how="left")[fcol].to_numpy()
            print(f"[folds] loaded from {a.folds} ({fcol}); missing:", int(pd.isna(fold).sum()))
            if pd.isna(fold).any():
                fold = None
    if fold is None:
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=a.fold_seed)
        fold = np.zeros(len(y), int)
        for k, (_, te) in enumerate(skf.split(X, y, groups=g)):
            fold[te] = k
        print(f"[folds] rebuilt with StratifiedGroupKFold seed={a.fold_seed}")
    fold = fold.astype(int)

    # ---- fixed-param CV
    oof = np.full(len(y), np.nan)
    for k in range(5):
        tr, te = fold != k, fold == k
        assert not set(g[tr]) & set(g[te]), f"material crosses fold {k}"
        rf = RandomForestClassifier(**PARAMS).fit(X[tr], y[tr])
        oof[te] = rf.predict_proba(X[te])[:, 1]
        print(f"  fold {k}: n_test={te.sum()} AUC={roc_auc_score(y[te], oof[te]):.3f}")
    print(f"[pooled] AUC={roc_auc_score(y, oof):.3f}  macroF1@0.5={f1_score(y, oof > .5, average='macro'):.3f}")
    pd.DataFrame(dict(cif_name=seed.cif_name, y=y, fold=fold, P_yes=oof)).to_csv(f"{a.out}/oof_fixed_params.csv", index=False)

    # ---- Table 3
    b = pd.cut(oof, BINS, labels=LABELS)
    tab = pd.DataFrame({"interval": LABELS})
    tab["n"] = [int((b == L).sum()) for L in LABELS]
    tab["yes_fraction"] = [round(float(y[b == L].mean()), 3) if (b == L).sum() else np.nan for L in LABELS]
    tab["precision_for_call"] = [round(1 - r, 3) if L == "<=0.2" else (round(r, 3) if L == ">0.8" else np.nan)
                                 for L, r in zip(LABELS, tab.yes_fraction)]
    print("\n== Table 3 (fixed params) ==\n", tab.to_string(index=False))
    tab.to_csv(f"{a.out}/table3_fixed_params.csv", index=False)

    print(f"saved to {a.out}/")

if __name__ == "__main__":
    main()
