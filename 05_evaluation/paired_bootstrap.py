"""
paired_bootstrap.py - SI Table S1: paired bootstrap model comparisons.

For each stimulus x split level, compares RF vs GBR, RF vs XGB, GBR vs XGB
on the same pooled OOF predictions (rac_mechanical_balanced baselines),
resampling material groups with replacement (2,000 draws).
Reports dAUC and d(macro-F1) with 95% CI; a CI excluding 0 = significant.

Run from train/rf_with_mechanical_sorted/:
    python paired_bootstrap.py
    python paired_bootstrap.py --dir rac_balanced --out si_S1_rac.csv

Output: si_S1_model_comparison.csv  (+ markdown print for the SI document)
"""

import argparse
import os
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

LEVELS = ["1_random", "2_mofkey", "3_chemistry", "4_doi", "5_comb"]
NICE = {"1_random": "Random", "2_mofkey": "Material", "3_chemistry": "Chemistry",
        "4_doi": "DOI", "5_comb": "Combined"}
MODELS = ["rf", "gbr", "xgb"]
N_BOOT = 2000
SEED = 20260901


def pooled_paired(df, m1, m2, idx_by_group, groups, rng):
    da, df1 = [], []
    for _ in range(N_BOOT):
        g = rng.choice(groups, len(groups))
        ii = np.concatenate([idx_by_group[x] for x in g])
        s = df.iloc[ii]
        if s.y.nunique() < 2:
            continue
        da.append(roc_auc_score(s.y, s[f"score_{m1}"]) -
                  roc_auc_score(s.y, s[f"score_{m2}"]))
        df1.append(f1_score(s.y, s[f"score_{m1}"] > .5, average="macro") -
                   f1_score(s.y, s[f"score_{m2}"] > .5, average="macro"))
    return np.array(da), np.array(df1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="../04_modeling/rac_mechanical_balanced")
    ap.add_argument("--groups", default="../data/master_groups.csv")
    ap.add_argument("--out", default="si_S1_model_comparison.csv")
    args = ap.parse_args()

    mg = pd.read_csv(args.groups)[["cif_name", "material_id"]]
    rng = np.random.default_rng(SEED)
    rows = []

    for lv in LEVELS:
        p = f"{args.dir}/{lv}/results/oof_predictions.csv"
        if not os.path.exists(p):
            print(f"[skip] {p} not found"); continue
        oof = pd.read_csv(p)
        for stim in ["gas", "solvent", "temperature"]:
            d = oof[oof.stimulus == stim]
            wide = None
            for m in MODELS:
                s = (d[d.model == m][["cif_name", "y", "score"]]
                     .rename(columns={"score": f"score_{m}"}))
                wide = s if wide is None else wide.merge(
                    s[["cif_name", f"score_{m}"]], on="cif_name")
            wide = wide.merge(mg, on="cif_name", how="left")
            wide["material_id"] = wide.material_id.fillna(wide.cif_name)
            wide = wide.reset_index(drop=True)
            idx = wide.groupby("material_id").indices
            groups = np.array(list(idx))

            for m1, m2 in itertools.combinations(MODELS, 2):
                da, df1 = pooled_paired(wide, m1, m2, idx, groups, rng)
                rows.append(dict(
                    stimulus=stim, split=NICE[lv], pair=f"{m1} - {m2}",
                    n=len(wide),
                    dAUC=np.mean(da), dAUC_lo=np.percentile(da, 2.5),
                    dAUC_hi=np.percentile(da, 97.5),
                    dF1=np.mean(df1), dF1_lo=np.percentile(df1, 2.5),
                    dF1_hi=np.percentile(df1, 97.5),
                ))
                r = rows[-1]
                sig = "*" if r["dAUC_lo"] * r["dAUC_hi"] > 0 else " "
                print(f"[{stim:11s} {NICE[lv]:9s}] {r['pair']}: "
                      f"dAUC {r['dAUC']:+.3f} [{r['dAUC_lo']:+.3f},{r['dAUC_hi']:+.3f}]{sig}")

    out = pd.DataFrame(rows).round(4)
    out.to_csv(args.out, index=False)
    ns = int((out.dAUC_lo * out.dAUC_hi > 0).sum())
    print(f"\nsaved {args.out}  ({len(out)} comparisons, {ns} with CI excluding 0)")


if __name__ == "__main__":
    main()
