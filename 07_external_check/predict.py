"""
predict.py — assemble geom 10 + RAC 176 + mechanical 5 and run the refit
gas RF on NU-2002.

Run from:  NU_2002/

Inputs
    ../features/geom/nu2002_geom.csv
    ../features/RACs/nu2002_racs.csv
    ../features/mechanical_pp/results_orb_nu2002.csv
    model: ../../train/rf_with_mechanical_sorted/rac_mechanical_tune_refit/rf_gas.joblib
           (override with --model)

The feature vector is aligned to model.feature_names_in_. RAC name mismatches
are resolved by suffix matching (text after '__'); anything unresolved is
reported and aborts — no silent zero-filling.
"""

import argparse
import glob
import os

import joblib
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="../../train/rf_with_mechanical_sorted/rac_mechanical_tune_refit/rf_gas.joblib")
ap.add_argument("--geom", default="../features/geom/nu2002_geom.csv")
ap.add_argument("--racs", default="../features/RACs/nu2002_racs.csv")
ap.add_argument("--mech", default="../features/mechanical_pp/results_orb_nu2002.csv")
a = ap.parse_args()

model = joblib.load(a.model)
fm = pd.read_csv("../data/descriptors.csv", nrows=1)
GEOM = ["LCD","PLD","density","VF","ASA","VSA","PV","LFPD","GSA","GPV"]
MECH = ["K_vrh","G_vrh","anisotropy","lambda_min","born_stable"]
rac_cols = [c for c in fm.columns if c.startswith("rac")]
feats = GEOM + rac_cols + MECH
assert len(feats) == model.n_features_in_, (len(feats), model.n_features_in_)
print(f"[model] n_features={model.n_features_in_}  assembled={len(feats)}")

geom = pd.read_csv(a.geom).iloc[0].to_dict()
racs = pd.read_csv(a.racs).iloc[0].to_dict()
mech_df = pd.read_csv(a.mech)
mech_df = mech_df[mech_df.status == "ok"]
if not len(mech_df):
    raise SystemExit("mechanical csv has no status=ok row")
mech = mech_df.iloc[0].to_dict()
print(f"[mech qc] relax_fmax={mech['relax_fmax']:.5f}  dV={mech['dV_relax']:+.3f}  "
      f"qc_ok={mech['qc_ok']}  lambda_min={mech['lambda_min']:.3f}  born_stable={mech['born_stable']}")

pool = {}
pool.update({k: v for k, v in geom.items() if k != "cif_name"})
pool.update({k: v for k, v in racs.items() if k != "cif_name"})
for k in ["K_vrh", "G_vrh", "anisotropy", "lambda_min", "born_stable"]:
    pool[k] = float(mech[k]) if k != "born_stable" else bool(mech[k])

# suffix index for RAC fuzzy matching
suffix = {}
for k in pool:
    if "__" in str(k):
        suffix.setdefault(str(k).split("__", 1)[1], []).append(k)

row, missing, fuzzy = {}, [], []
for f in feats:
    if f in pool:
        row[f] = pool[f]
    elif "__" in f and f.split("__", 1)[1] in suffix and len(suffix[f.split("__", 1)[1]]) == 1:
        src = suffix[f.split("__", 1)[1]][0]
        row[f] = pool[src]
        fuzzy.append((f, src))
    else:
        missing.append(f)

if fuzzy:
    print(f"[map] {len(fuzzy)} RAC names matched by suffix, e.g. {fuzzy[:3]}")
if missing:
    print(f"[FAIL] {len(missing)} model features not found. First 20:")
    for m in missing[:20]:
        print("   ", m)
    have = [k for k in pool if str(k).startswith("rac")][:10]
    print("available rac keys look like:", have)
    raise SystemExit(1)

X = pd.DataFrame([row])[feats]
p = float(model.predict_proba(X)[0, 1])
flex = "Yes" if p > 0.5 else "No"
conf = "Yes" if p > 0.8 else ("No" if p <= 0.2 else "Unsure")
print("\n================ NU-2002 (gas axis) ================")
print(f"P(yes)            : {p:.4f}")
print(f"Flexibility (0.5) : {flex}")
print(f"Flexibility_conf  : {conf}   (No<=0.2 / Yes>0.8, OOF prec 0.92/0.97)")
print("====================================================")
