"""
ls_gate_validation.py - assemble SI S7 (label-spreading gate validation).

Does two things:
  (a) S7a: excerpt table (support=3) from the two existing validation grids
      ls_gas_validation.csv        (no-threshold  P > 0.995)
      ls_gas_validation_0_885.csv  (yes-threshold P > 0.885)
  (b) S7b: rerun the material-masked validation with a second fold seed,
      restricted to the two selected gates (k=50 a=0.1, k=75 a=0.2),
      support 1-5, both thresholds.

Run from the folder that contains the validation csvs and the LS input
(features/labels used by label_spreading.py):
    python ls_gate_validation.py                       # uses defaults below
Outputs -> results_si/
    S7a_gate_grid_support3.csv / .md
    S7b_seed_reproduction.csv / .md
"""

import os
import numpy as np
import pandas as pd
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from common.data import read_labels, read_features, read_mech, read_master  # released-data loaders


# ---------------------------------------------------------------- config
VAL_995 = "../results/ls_gas_validation_thr0995.csv"
VAL_885 = "../results/ls_gas_validation_thr0885.csv"
OUT = "../results/si_S13"

# for (b): mirrors label_spreading.py load_inputs() exactly
LABEL_FILE = "CR_Main_LLM_Extraction_overall.xlsx"
FEAT_FILE = "features_master.csv"
MECH_FILE = "CR_Main_Mechanical_pp_final.csv"
MASTER_FILE = "master_groups.csv"
LABEL_COL = "Overall_Flexibility_gas adsorption/desorption"
GEOM = ["LCD", "PLD", "density", "VF", "ASA", "VSA", "PV", "LFPD", "GSA", "GPV"]
MECH = ["K_vrh", "G_vrh", "anisotropy", "lambda_min", "born_stable"]
SEEDS = [20260724, 20260725]
GATES = [dict(k=50, alpha=0.1, thr=0.995, cls="no", name="No gate"),
         dict(k=75, alpha=0.2, thr=0.885, cls="yes", name="Yes gate")]
SUPPORTS = [1, 2, 3, 4, 5]
N_FOLDS = 5

os.makedirs(OUT, exist_ok=True)


def part_a():
    a995 = pd.read_csv(VAL_995)
    a885 = pd.read_csv(VAL_885)
    s995 = a995[a995.min_support == 3][["k", "alpha", "n_conf_no", "acc_conf_no"]]
    s885 = a885[a885.min_support == 3][["k", "alpha", "n_conf_yes", "acc_conf_yes"]]
    m = s995.merge(s885, on=["k", "alpha"], how="outer").sort_values(["k", "alpha"])
    m.columns = ["k", "alpha", "n_no (P>0.995)", "acc_no", "n_yes (P>0.885)", "acc_yes"]
    m.to_csv(f"{OUT}/S7a_gate_grid_support3.csv", index=False)
    md = m.round(3).to_markdown(index=False)
    open(f"{OUT}/S7a_gate_grid_support3.md", "w").write(md)
    print("== S7a (support = 3) ==")
    print(md)


def load_inputs():
    from sklearn.preprocessing import StandardScaler
    lab = read_labels()
    feat = read_features()
    mech = read_mech()
    rac = [c for c in feat.columns if c.startswith("rac_")]
    cols = GEOM + rac + MECH
    table = (lab[["cif_name", LABEL_COL] + GEOM]
             .merge(feat[["cif_name"] + rac], on="cif_name", how="inner")
             .merge(mech[["cif_name"] + MECH], on="cif_name", how="inner"))
    table = table.dropna(subset=cols).reset_index(drop=True)
    y = table[LABEL_COL].map({"Yes": 1, "No": 0}).fillna(-1).astype(int).to_numpy()
    X = StandardScaler().fit_transform(table[cols].to_numpy(dtype=float))
    print(f"[load] {len(table)} rows, seeds yes {(y==1).sum()} / no {(y==0).sum()}")
    return table, X, y


def part_b():
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.semi_supervised import LabelSpreading
    from sklearn.neighbors import NearestNeighbors

    table, X, y = load_inputs()
    master = read_master()[["cif_name", "material_id"]]
    g = table[["cif_name"]].merge(master, on="cif_name", how="left")["material_id"]
    g = g.fillna(table["cif_name"]).to_numpy()
    mats = g
    lab_idx = np.where(y != -1)[0]
    rows = []

    for seed in SEEDS:
        skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        folds = list(skf.split(lab_idx, y[lab_idx], groups=g[lab_idx]))
        for gate in GATES:
            k, a, thr, cls = gate["k"], gate["alpha"], gate["thr"], gate["cls"]
            nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
            _, nbidx = nn.kneighbors(X)
            pass_n = {s_: 0 for s_ in SUPPORTS}
            pass_c = {s_: 0 for s_ in SUPPORTS}
            for tr, te in folds:
                hide = lab_idx[te]
                y_masked = y.copy()
                y_masked[hide] = -1
                ls = LabelSpreading(kernel="knn", n_neighbors=k, alpha=a,
                                    max_iter=1000, tol=1e-3, n_jobs=-1).fit(X, y_masked)
                P = ls.label_distributions_
                order = {c: i for i, c in enumerate(ls.classes_)}
                p_no, p_yes = P[:, order[0]], P[:, order[1]]
                p = p_yes if cls == "yes" else p_no
                true_val = 1 if cls == "yes" else 0
                visible_seed = (y_masked != -1)
                for i in hide:
                    nb = nbidx[i]
                    nb = nb[nb != i][:k]
                    nb = nb[visible_seed[nb]]
                    same = {mats[j] for j in nb if y_masked[j] == true_val}
                    ns = len(same)
                    if p[i] > thr:
                        for s_ in SUPPORTS:
                            if ns >= s_:
                                pass_n[s_] += 1
                                if y[i] == true_val:
                                    pass_c[s_] += 1
            for s_ in SUPPORTS:
                n_, c_ = pass_n[s_], pass_c[s_]
                rows.append(dict(seed=seed, gate=gate["name"], k=k, alpha=a,
                                 thr=thr, cls=cls, support=s_, n_pass=n_,
                                 acc=(c_ / n_ if n_ else np.nan)))
                print(f"seed={seed} {gate['name']} sup>={s_}: n={n_} acc={(c_/n_ if n_ else float('nan')):.3f}",
                      flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/S7b_seed_reproduction.csv", index=False)
    sel = out[out.support == 3]
    md = sel.pivot_table(index=["gate", "k", "alpha", "thr"], columns="seed",
                         values=["n_pass", "acc"]).round(3).to_markdown()
    open(f"{OUT}/S7b_seed_reproduction.md", "w").write(md)
    print("== S7b (support >= 3) ==")
    print(md)


if __name__ == "__main__":
    part_a()
    try:
        part_b()
    except FileNotFoundError as e:
        print(f"\n[part b skipped] {e}")
