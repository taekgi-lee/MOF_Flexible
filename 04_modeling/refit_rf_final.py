"""
refit_rf_final.py — refit the final RF on ALL labeled CIFs with the tuned parameters
and score the unlabeled CIFs.

Run from:  train/rf_with_mechanical_sorted/rac_mechanical_tune_refit/
Inputs are read from the parent folders; nothing needs to be copied here.

    ../../CR_Main_LLM_Extraction_overall.xlsx   labels + geom 10
    ../../features_master.csv                   RAC 176
    ../CR_Main_Mechanical_pp_final.csv          mechanical 5
    ../../master_groups.csv                     material_id (for --weight-by-material)

Row set: CIFs with complete geom + RAC + mechanical (7,188) — identical to
label_spreading.py. Labeled (Yes/No) rows are the training set, the rest
(Unknown/none) are scored.

Outputs (this folder)
    rf_<stimulus>.joblib               fitted model
    rf_<stimulus>_meta.json            params, feature list, counts
    rf_<stimulus>_train_rows.csv       cif_name, y used for fitting
    rf_<stimulus>_predictions.csv      ALL 7,188 rows: label_in, is_train, P_yes

Usage
-----
    python refit_rf_final.py                          # gas, F1-tuned params (941/12/8/sqrt, balanced)
    python refit_rf_final.py --stimuli gas solvent temperature
    python refit_rf_final.py --weight-by-material     # sample_weight = 1/n_ms
    python refit_rf_final.py --params '{"n_estimators": 500, "min_samples_leaf": 2, "max_features": "sqrt"}'
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from common.data import read_labels, read_features, read_mech, read_master  # released-data loaders


LABEL_FILE = "../../CR_Main_LLM_Extraction_overall.xlsx"
FEAT_FILE = "../../features_master.csv"
MECH_FILE = "../CR_Main_Mechanical_pp_final.csv"
MASTER_FILE = "../../master_groups.csv"

GEOM = ["LCD", "PLD", "density", "VF", "ASA", "VSA", "PV", "LFPD", "GSA", "GPV"]
MECH = ["K_vrh", "G_vrh", "anisotropy", "lambda_min", "born_stable"]

STIMULI = {
    "solvent": "Overall_Flexibility_solvent",
    "temperature": "Overall_Flexibility_temperature",
    "gas": "Overall_Flexibility_gas adsorption/desorption",
}

SEED = 20260724

# plurality over 2_mofkey outer folds
# (tune_rf_nested_cv.py: scoring=f1_macro, class_weight=balanced; rf_best_params.csv)
TUNED = {
    "gas": dict(n_estimators=941, max_depth=12, min_samples_leaf=8,
                max_features="sqrt", bootstrap=False, class_weight="balanced"),
    "solvent": dict(n_estimators=200, max_depth=16, min_samples_leaf=5,
                    max_features=0.5, bootstrap=False, class_weight="balanced"),
    "temperature": dict(n_estimators=297, max_depth=8, min_samples_leaf=4,
                        max_features=0.5, bootstrap=False, class_weight="balanced"),
}
BASELINE = dict(n_estimators=500, min_samples_leaf=2,
                max_features="sqrt", bootstrap=False, class_weight="balanced")


# ---------------------------------------------------------------- data
def load_inputs():
    lab = read_labels()
    feat = read_features()
    mech = read_mech()

    rac = [c for c in feat.columns if c.startswith("rac_")]
    cols = GEOM + rac + MECH
    label_cols = list(STIMULI.values())

    table = (lab[["cif_name"] + label_cols + GEOM]
             .merge(feat[["cif_name"] + rac], on="cif_name", how="inner")
             .merge(mech[["cif_name"] + MECH], on="cif_name", how="inner"))
    n_before = len(table)
    table = table.dropna(subset=cols).reset_index(drop=True)
    print(f"[load] geom {len(GEOM)} + rac {len(rac)} + mech {len(MECH)} = {len(cols)}")
    print(f"[load] {len(table)} CIFs with complete values (dropped {n_before - len(table)})")
    return table, cols


def material_weights(cif_names):
    """w = 1 / n_ms : total mass 1 per (material, stimulus) pair."""
    master = read_master()[["cif_name", "material_id"]]
    m = pd.DataFrame({"cif_name": cif_names}).merge(master, on="cif_name", how="left")
    if m["material_id"].isna().any():
        print(f"[warn] {int(m['material_id'].isna().sum())} rows without material_id -> own group")
        m["material_id"] = m["material_id"].fillna(m["cif_name"])
    n_ms = m.groupby("material_id")["cif_name"].transform("count")
    return (1.0 / n_ms).to_numpy()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", nargs="+", default=["gas"], choices=list(STIMULI))
    ap.add_argument("--params", default=None,
                    help="JSON string/file with RF kwargs; default = TUNED[stimulus] "
                         "if defined, else baseline")
    ap.add_argument("--weight-by-material", action="store_true")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    table, cols = load_inputs()
    X_all = table[cols].to_numpy(dtype=float)

    for stim in args.stimuli:
        col = STIMULI[stim]
        y_all = table[col].map({"Yes": 1, "No": 0})
        is_train = y_all.notna().to_numpy()
        y = y_all[is_train].astype(int).to_numpy()
        X = X_all[is_train]

        if args.params is None:
            params = dict(TUNED.get(stim, BASELINE))
            src = "tuned" if stim in TUNED else "baseline"
        elif os.path.exists(args.params):
            with open(args.params) as f:
                params = json.load(f)
            src = args.params
        else:
            params = json.loads(args.params)
            src = "cli"
        params.setdefault("bootstrap", False)
        params.setdefault("class_weight", "balanced")

        print(f"\n[{stim}] train {len(y)}  Yes {int(y.sum())} / No {int((y == 0).sum())}"
              f"  |  predict {int((~is_train).sum())}")
        print(f"[{stim}] params ({src}): {params}")

        sw = None
        if args.weight_by_material:
            sw = material_weights(table.loc[is_train, "cif_name"].to_numpy())
            print(f"[{stim}] weights: min {sw.min():.4f} max {sw.max():.4f} sum {sw.sum():.1f}")

        rf = RandomForestClassifier(**params, random_state=SEED, n_jobs=args.n_jobs)
        rf.fit(X, y, sample_weight=sw)
        pos = list(rf.classes_).index(1)
        p_yes = rf.predict_proba(X_all)[:, pos]

        tag = f"rf_{stim}"
        joblib.dump(rf, os.path.join(args.out_dir, f"{tag}.joblib"))

        pd.DataFrame({"cif_name": table.loc[is_train, "cif_name"].to_numpy(), "y": y}) \
          .to_csv(os.path.join(args.out_dir, f"{tag}_train_rows.csv"), index=False)

        pred = pd.DataFrame({
            "cif_name": table["cif_name"],
            "label_in": table[col],
            "is_train": is_train,
            "P_yes": p_yes,
        })
        pred.to_csv(os.path.join(args.out_dir, f"{tag}_predictions.csv"), index=False)

        tgt = pred[~pred["is_train"]]
        meta = {
            "stimulus": stim, "label_column": col,
            "feature_columns": cols, "n_features": len(cols),
            "params": {k: (v if isinstance(v, (int, float, str, bool)) or v is None
                           else str(v)) for k, v in params.items()},
            "params_source": src, "seed": SEED,
            "weight_by_material": args.weight_by_material,
            "n_train": int(len(y)), "n_yes": int(y.sum()), "n_no": int((y == 0).sum()),
            "n_predict": int(len(tgt)),
            "predict_P_yes_gt_0.5": int((tgt["P_yes"] > 0.5).sum()),
        }
        with open(os.path.join(args.out_dir, f"{tag}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[{stim}] predict: P_yes > 0.5 -> {meta['predict_P_yes_gt_0.5']} / {len(tgt)}")
        print(f"[{stim}] saved {tag}.joblib, {tag}_predictions.csv")


if __name__ == "__main__":
    main()