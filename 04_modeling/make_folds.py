"""
make_folds.py - frozen 5-fold assignments for the split ladder (run before the model scripts).

For each split level and stimulus, rows sharing a group id stay in one fold and the
fold sizes / yes counts are balanced greedily (largest groups first). Deterministic
(seed 20260724); reproduces the folds used in the paper.

Writes  <out>/<level>/folds/<stimulus>_folds.csv   (cif_name, y, fold)
Usage   python make_folds.py --out .          # run from model/
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.data import read_labels, read_master

STIMULI = {"solvent": "Overall_Flexibility_solvent", "temperature": "Overall_Flexibility_temperature",
           "gas": "Overall_Flexibility_gas adsorption/desorption"}
LEVELS = {"1_random": None, "2_mofkey": "material_id", "3_chemistry": "chemistry_id",
          "4_doi": "doi", "5_comb": "combined_id"}
N_FOLDS, SEED = 5, 20260724


def make_folds(df, group_col, master, k=N_FOLDS, seed=SEED):
    d = df.merge(master[["cif_name"] + ([group_col] if group_col else [])], on="cif_name", how="left")
    if group_col is None:
        d["grp"] = d["cif_name"]
    else:
        d["grp"] = d[group_col].where(d[group_col].notna(), "NA::" + d["cif_name"])
    agg = d.groupby("grp")["y"].agg(n="size", pos="sum").reset_index()
    agg = agg.sample(frac=1, random_state=seed).sort_values("n", ascending=False)
    load = np.zeros((k, 2)); fold_of = {}
    for _, r in agg.iterrows():
        v = np.array([r["n"], r["pos"]], float)
        b = int(np.argmin(((load + v) ** 2).sum(axis=1)))
        fold_of[r["grp"]] = b; load[b] += v
    d["fold"] = d["grp"].map(fold_of)
    return d[["cif_name", "y", "fold"]]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="../results/folds"); a = ap.parse_args()
    lab, master = read_labels(), read_master()
    for level, gcol in LEVELS.items():
        os.makedirs(os.path.join(a.out, level, "folds"), exist_ok=True)
        for stim, col in STIMULI.items():
            s = lab[["cif_name", col]].copy(); s[col] = s[col].astype(str).str.strip()
            s = s[s[col].isin(["Yes", "No"])]; s["y"] = (s[col] == "Yes").astype(int)
            f = make_folds(s[["cif_name", "y"]], gcol, master)
            f.to_csv(os.path.join(a.out, level, "folds", f"{stim}_folds.csv"), index=False)
            print(f"{level:12s} {stim:12s} n={len(f)} folds={f.fold.value_counts().sort_index().tolist()}")


if __name__ == "__main__":
    main()
