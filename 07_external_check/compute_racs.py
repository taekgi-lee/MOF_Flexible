"""
compute_racs.py — RAC 176 features for NU-2002.

Run from:  this directory
Input   :  ../NU2002_348K_BCP_ordered_H.cif  (or --cif)

Output: nu2002_racs.csv (columns rac_*), nu2002_racs_raw.json
Check the printed 'first keys' against features_master rac_* column names.
"""
import argparse
import json
import pandas as pd
from CoREMOF.calculation import mof_features

ap = argparse.ArgumentParser()
ap.add_argument("--cif", default="../geom/NU2002_348K_BCP_ordered_H.cif")
ap.add_argument("--name", default="NU2002_348K_ordered")
a = ap.parse_args()

r = mof_features.RACs(a.cif)
json.dump(r, open("nu2002_racs_raw.json", "w"), indent=1, default=float)

flat = {}
if all(isinstance(v, dict) for v in r.values()):
    prefix = {"Metal": "rac_metal__", "Linker": "rac_linker__",
              "Function-group": "rac_func__", "Functional-group": "rac_func__",
              "Func": "rac_func__"}
    for grp, d in r.items():
        p = prefix.get(grp, f"rac_{grp.lower()}__")
        for k, v in d.items():
            flat[p + str(k)] = float(v)
else:
    for k, v in r.items():
        flat["rac_" + str(k)] = float(v)

pd.DataFrame([{"cif_name": a.name, **flat}]).to_csv("nu2002_racs.csv", index=False)
print(f"[RACs] {len(flat)} descriptors -> nu2002_racs.csv")
print("first keys:", list(flat)[:6])
