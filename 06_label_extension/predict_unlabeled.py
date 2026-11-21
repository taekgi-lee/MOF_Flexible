"""
predict_unlabeled.py — score the unlabeled CIFs with the refit RF.

Run from:  prediction/rf/

Inputs
    model : ../../train/rf_with_mechanical_sorted/rac_mechanical_tune_refit/
            rf_gas.joblib, rf_gas_meta.json, rf_gas_train_rows.csv
    data  : ../label_spreading/  (same three files label spreading used)

Row set is rebuilt exactly as in label_spreading.py / refit_rf_final.py
(complete geom + RAC + mechanical, 7,188) and checked against the model's
training rows so the target set is identical to label spreading's.

Outputs (results/)
    rf_gas_predictions.csv   all 7,188 rows: label_in, is_train, P_yes,
                             Flexibility (Yes/No at 0.5),
                             Flexibility_conf (Yes if P>0.8, No if P<=0.2, else Unsure)
    rf_gas_summary.json

Usage
-----
    python predict_unlabeled.py
    python predict_unlabeled.py --conf_lo 0.2 --conf_hi 0.8
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from common.data import read_labels, read_features, read_mech, read_master  # released-data loaders


MODEL_DIR = "../results/rf_final"
DATA_DIR = "../data"

LABEL_FILE = os.path.join(DATA_DIR, "CR_Main_LLM_Extraction_overall.xlsx")
FEAT_FILE = os.path.join(DATA_DIR, "features_master.csv")
MECH_FILE = os.path.join(DATA_DIR, "CR_Main_Mechanical_pp_final.csv")

GEOM = ["LCD", "PLD", "density", "VF", "ASA", "VSA", "PV", "LFPD", "GSA", "GPV"]
MECH = ["K_vrh", "G_vrh", "anisotropy", "lambda_min", "born_stable"]
LABEL_COL = "Overall_Flexibility_gas adsorption/desorption"


def load_table(cols):
    lab = read_labels()
    feat = read_features()
    mech = read_mech()
    rac = [c for c in feat.columns if c.startswith("rac_")]
    table = (lab[["cif_name", LABEL_COL] + GEOM]
             .merge(feat[["cif_name"] + rac], on="cif_name", how="inner")
             .merge(mech[["cif_name"] + MECH], on="cif_name", how="inner"))
    n_before = len(table)
    table = table.dropna(subset=cols).reset_index(drop=True)
    print(f"[load] {len(table)} CIFs with complete values (dropped {n_before - len(table)})")
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimulus", default="gas")
    ap.add_argument("--thr", type=float, default=0.5,
                    help="P_yes threshold for the Flexibility column (Yes/No)")
    ap.add_argument("--conf_lo", type=float, default=0.2,
                    help="P_yes <= conf_lo -> Flexibility_conf = No   (OOF prec 0.92)")
    ap.add_argument("--conf_hi", type=float, default=0.8,
                    help="P_yes >  conf_hi -> Flexibility_conf = Yes  (OOF prec 0.97)")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tag = f"rf_{args.stimulus}"
    rf = joblib.load(os.path.join(MODEL_DIR, f"{tag}.joblib"))
    with open(os.path.join(MODEL_DIR, f"{tag}_meta.json")) as f:
        meta = json.load(f)
    cols = meta["feature_columns"]
    train_rows = pd.read_csv(os.path.join(MODEL_DIR, f"{tag}_train_rows.csv"))
    print(f"[model] {tag}: {meta['n_features']} features, params {meta['params']}")

    table = load_table(cols)

    # consistency check: model's training rows must be exactly the Yes/No rows here
    y_here = table[LABEL_COL].map({"Yes": 1, "No": 0})
    is_train = y_here.notna().to_numpy()
    here = set(table.loc[is_train, "cif_name"])
    there = set(train_rows["cif_name"])
    if here != there:
        raise SystemExit(f"[error] training rows differ: here {len(here)} / model {len(there)} "
                         f"/ overlap {len(here & there)}")
    print(f"[check] training rows match model: {len(here)}")

    pos = list(rf.classes_).index(1)
    p_yes = rf.predict_proba(table[cols].to_numpy(dtype=float))[:, pos]

    flex = np.where(p_yes > args.thr, "Yes", "No")
    conf = np.where(p_yes > args.conf_hi, "Yes",
                    np.where(p_yes <= args.conf_lo, "No", "Unsure"))
    out = pd.DataFrame({
        "cif_name": table["cif_name"],
        "label_in": table[LABEL_COL],
        "is_train": is_train,
        "P_yes": p_yes,
        "Flexibility": flex,
        "Flexibility_conf": conf,
    })
    out.to_csv(os.path.join(args.out_dir, f"{tag}_predictions.csv"), index=False)

    tgt = out[~out["is_train"]]
    summary = {
        "stimulus": args.stimulus, "thr": args.thr,
        "conf_lo": args.conf_lo, "conf_hi": args.conf_hi,
        "n_rows": int(len(out)), "n_train": int(is_train.sum()),
        "n_target": int(len(tgt)),
        "target_Flexibility": tgt["Flexibility"].value_counts().to_dict(),
        "target_Flexibility_conf": tgt["Flexibility_conf"].value_counts().to_dict(),
        "target_P_yes_quantiles": {q: float(tgt["P_yes"].quantile(q))
                                   for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "model_dir": MODEL_DIR,
    }
    with open(os.path.join(args.out_dir, f"{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[predict] targets {len(tgt)}: Flexibility (thr {args.thr}) -> "
          f"{summary['target_Flexibility']}")
    print(f"[predict] Flexibility_conf (No<= {args.conf_lo} / Yes> {args.conf_hi}) -> "
          f"{summary['target_Flexibility_conf']}")
    print(f"[predict] target P_yes quantiles: "
          + "  ".join(f"q{int(q*100)}={v:.3f}" for q, v in summary["target_P_yes_quantiles"].items()))
    print(f"[predict] saved {args.out_dir}/{tag}_predictions.csv")


if __name__ == "__main__":
    main()
