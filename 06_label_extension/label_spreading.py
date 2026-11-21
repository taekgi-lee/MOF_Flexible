"""
label_spreading.py — semi-supervised label spreading for the gas axis,
with a seed-support gate.

Seeds   : CIFs with Overall_Flexibility_gas == Yes / No
Targets : Unknown / none
Rows    : only CIFs with complete geom 10 + RAC 176 + mechanical 5 (7,188)

sklearn convention: unlabeled MUST be -1, so encoding is
    Yes -> 1, No -> 0, Unknown/none -> -1

Confidence gate (two conditions, both required)
    P_yes > thr                       (purity of label mass reaching the node)
    n_yes_seed_mats >= min_support    (number of DISTINCT seed materials with
                                       label Yes among the k nearest neighbours)
    symmetric for No.
Purity alone cannot tell "3 flexible materials nearby" from "one Yes CIF in an
otherwise unlabeled cluster"; both give P_yes = 1. The support count does.

Outputs (default out_dir = results/)
    ls_gas_predictions.csv    every row: seed label, P_no, P_yes, spread label,
                              neighbour seed counts, conf flags (raw and gated)
    ls_gas_seed_flips.csv     seeds whose spread label != input label
    ls_gas_summary.json
    ls_gas_validation.csv     (--validate only) grouped-masked metrics per (k, alpha, support)

Usage
-----
    python label_spreading.py --k 50 --alpha 0.1                  # min_support 3
    python label_spreading.py --k 50 --alpha 0.1 --min_support 1  # = old behaviour
    python label_spreading.py --validate --k_grid 30,50 --alpha_grid 0.1
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
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

SEED = 20260724
N_FOLDS = 5


# ---------------------------------------------------------------- data
def load_inputs():
    lab = read_labels()
    feat = read_features()
    mech = read_mech()
    master = read_master()[["cif_name", "material_id"]]

    rac = [c for c in feat.columns if c.startswith("rac_")]
    cols = GEOM + rac + MECH

    table = (lab[["cif_name", LABEL_COL] + GEOM]
             .merge(feat[["cif_name"] + rac], on="cif_name", how="inner")
             .merge(mech[["cif_name"] + MECH], on="cif_name", how="inner"))
    n_before = len(table)
    table = table.dropna(subset=cols).reset_index(drop=True)
    table = table.merge(master, on="cif_name", how="left")
    n_nomat = int(table["material_id"].isna().sum())
    if n_nomat:
        print(f"[load] {n_nomat} rows without material_id -> own material")
        table["material_id"] = table["material_id"].fillna(table["cif_name"])

    print(f"[load] geom {len(GEOM)} + rac {len(rac)} + mech {len(MECH)} = {len(cols)}")
    print(f"[load] {len(table)} CIFs with complete values (dropped {n_before - len(table)})")

    y = table[LABEL_COL].map({"Yes": 1, "No": 0}).fillna(-1).astype(int).to_numpy()
    vc = table[LABEL_COL].value_counts()
    print(f"[load] seeds  Yes {int((y == 1).sum())} / No {int((y == 0).sum())}")
    print(f"[load] target Unknown {int(vc.get('Unknown', 0))} / none {int(vc.get('none', 0))}"
          f"  -> {int((y == -1).sum())} rows to spread")

    X = StandardScaler().fit_transform(table[cols].to_numpy(dtype=float))
    return table, X, y, cols


def make_ls(k, alpha, n_jobs):
    return LabelSpreading(kernel="knn", n_neighbors=k, alpha=alpha,
                          max_iter=1000, tol=1e-3, n_jobs=n_jobs)


def spread(X, y, k, alpha, n_jobs):
    ls = make_ls(k, alpha, n_jobs).fit(X, y)
    P = ls.label_distributions_
    order = {c: i for i, c in enumerate(ls.classes_)}
    return P[:, order[0]], P[:, order[1]], ls.transduction_


def neighbour_seed_support(X, y, mats, k, n_jobs):
    """For every row: number of Yes / No seed CIFs and distinct seed materials
    among its k nearest neighbours (self excluded)."""
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=n_jobs).fit(X)
    _, idx = nn.kneighbors(X)
    idx = idx[:, 1:]                                  # drop self
    ny = np.zeros(len(X), int); nn_ = np.zeros(len(X), int)
    my = np.zeros(len(X), int); mn = np.zeros(len(X), int)
    for i, nb in enumerate(idx):
        yy = y[nb]
        ny[i] = int((yy == 1).sum()); nn_[i] = int((yy == 0).sum())
        my[i] = len(set(mats[nb[yy == 1]])); mn[i] = len(set(mats[nb[yy == 0]]))
    return ny, nn_, my, mn


# ---------------------------------------------------------------- validate
def validate(table, X, y, args):
    g = table["material_id"].to_numpy()
    mats = g
    lab_idx = np.where(y != -1)[0]
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(lab_idx, y[lab_idx], groups=g[lab_idx]))
    for f, (tr, te) in enumerate(folds):
        assert not set(g[lab_idx[tr]]) & set(g[lab_idx[te]]), f"group crossed fold {f}"

    ks = [int(v) for v in args.k_grid.split(",")]
    alphas = [float(v) for v in args.alpha_grid.split(",")]
    supports = [int(v) for v in args.support_grid.split(",")]
    rows = []
    for k, a in itertools.product(ks, alphas):
        oof = np.full(len(y), np.nan)
        sup_y = np.zeros(len(y), int); sup_n = np.zeros(len(y), int)
        for tr, te in folds:
            y_m = y.copy(); y_m[lab_idx[te]] = -1
            p_no, p_yes, _ = spread(X, y_m, k, a, args.n_jobs)
            p_yes = np.where(np.isclose(p_no + p_yes, 0.0), 0.5, p_yes)
            oof[lab_idx[te]] = p_yes[lab_idx[te]]
            _, _, my, mn = neighbour_seed_support(X, y_m, mats, k, args.n_jobs)
            sup_y[lab_idx[te]] = my[lab_idx[te]]; sup_n[lab_idx[te]] = mn[lab_idx[te]]
        yt, p = y[lab_idx], oof[lab_idx]
        auc = roc_auc_score(yt, p)
        for s in supports:
            cy = (p > args.thr) & (sup_y[lab_idx] >= s)
            cn = (p < 1 - args.thr) & (sup_n[lab_idx] >= s)
            conf = cy | cn
            acc = float(((p[conf] > 0.5).astype(int) == yt[conf]).mean()) if conf.sum() else np.nan
            acc_y = float((yt[cy] == 1).mean()) if cy.sum() else np.nan
            acc_n = float((yt[cn] == 0).mean()) if cn.sum() else np.nan
            rows.append(dict(k=k, alpha=a, min_support=s, auc_pooled_oof=auc,
                             frac_confident=float(conf.mean()), n_confident=int(conf.sum()),
                             acc_confident=acc, n_conf_yes=int(cy.sum()), acc_conf_yes=acc_y,
                             n_conf_no=int(cn.sum()), acc_conf_no=acc_n))
            print(f"[validate] k={k:3d} alpha={a:.2f} support>={s}  AUC {auc:.3f}  "
                  f"conf {conf.mean():.3f} (n={conf.sum()})  acc {acc:.3f}  "
                  f"| Yes n={cy.sum()} acc {acc_y:.3f} | No n={cn.sum()} acc {acc_n:.3f}")
    out = pd.DataFrame(rows)
    path = os.path.join(args.out_dir, "ls_gas_validation.csv")
    out.to_csv(path, index=False)
    print(f"[validate] saved {path}")
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--thr", type=float, default=0.995)
    ap.add_argument("--min_support", type=int, default=3,
                    help="min distinct seed materials of the same label among k neighbours")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--k_grid", default="30,50")
    ap.add_argument("--alpha_grid", default="0.1")
    ap.add_argument("--support_grid", default="1,2,3,5")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    table, X, y, cols = load_inputs()

    if args.validate:
        validate(table, X, y, args)
        return

    p_no, p_yes, y_spread = spread(X, y, args.k, args.alpha, args.n_jobs)
    is_seed = y != -1
    unreachable = np.isclose(p_no + p_yes, 0.0)
    y_spread = np.where(unreachable, -1, y_spread)

    mats = table["material_id"].to_numpy()
    ny, nn_, my, mn = neighbour_seed_support(X, y, mats, args.k, args.n_jobs)

    raw_yes = (~is_seed) & (p_yes > args.thr)
    raw_no = (~is_seed) & (p_no > args.thr)
    conf_yes = raw_yes & (my >= args.min_support)
    conf_no = raw_no & (mn >= args.min_support)

    out = pd.DataFrame({
        "cif_name": table["cif_name"], "material_id": mats,
        "label_in": table[LABEL_COL], "y_seed": y,
        "P_no": p_no, "P_yes": p_yes, "y_spread": y_spread,
        "is_seed": is_seed, "seed_flipped": is_seed & (y_spread != y),
        "unreachable": unreachable,
        "n_yes_seeds": ny, "n_no_seeds": nn_,
        "n_yes_seed_mats": my, "n_no_seed_mats": mn,
        "conf_yes_raw": raw_yes, "conf_no_raw": raw_no,
        "conf_yes": conf_yes, "conf_no": conf_no,
    })
    out.to_csv(os.path.join(args.out_dir, "ls_gas_predictions.csv"), index=False)
    out[out["seed_flipped"]].to_csv(os.path.join(args.out_dir, "ls_gas_seed_flips.csv"), index=False)

    tgt = out[~out["is_seed"]]
    summary = {
        "k": args.k, "alpha": args.alpha, "thr": args.thr, "min_support": args.min_support,
        "n_rows": int(len(out)), "n_features": len(cols),
        "n_seed_yes": int((y == 1).sum()), "n_seed_no": int((y == 0).sum()),
        "n_target": int(len(tgt)), "target_unreachable": int(tgt["unreachable"].sum()),
        "raw_conf_yes": int(tgt["conf_yes_raw"].sum()), "raw_conf_no": int(tgt["conf_no_raw"].sum()),
        "conf_yes": int(tgt["conf_yes"].sum()), "conf_no": int(tgt["conf_no"].sum()),
        "undecided": int(len(tgt) - tgt["conf_yes"].sum() - tgt["conf_no"].sum()),
        "conf_yes_materials": int(tgt.loc[tgt["conf_yes"], "material_id"].nunique()),
        "conf_no_materials": int(tgt.loc[tgt["conf_no"], "material_id"].nunique()),
        "seed_flipped": int(out["seed_flipped"].sum()),
    }
    with open(os.path.join(args.out_dir, "ls_gas_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[spread] k={args.k} alpha={args.alpha} thr={args.thr} min_support={args.min_support}")
    print(f"[spread] purity only : conf Yes {summary['raw_conf_yes']} / conf No {summary['raw_conf_no']}")
    print(f"[spread] + support   : conf Yes {summary['conf_yes']} ({summary['conf_yes_materials']} materials)"
          f" / conf No {summary['conf_no']} ({summary['conf_no_materials']} materials)"
          f" / undecided {summary['undecided']}")
    print(f"[spread] unreachable {summary['target_unreachable']}, seeds flipped {summary['seed_flipped']}")
    print(f"[spread] saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
