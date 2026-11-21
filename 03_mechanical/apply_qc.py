"""
apply_qc.py - merge MLIP elastic-tensor shard results and apply acceptance criteria.

Step 1  merge: fill rows of the master mechanical table from per-shard result files
        (results_orb_shardNN.csv written by compute_elastic_tensors.py) -> *_merged.csv (8,802 ok of 8,804)
Step 2  QC:    keep rows with round(relax_fmax, 3) <= 0.009  (i.e. below 0.01 eV/A after rounding)
               and |dV_relax| <= 0.15                          -> *_final.csv (7,828)

Usage
  python apply_qc.py --master data/CR_Main_Mechanical_pp.csv --shards "results_orb/**/results_orb_shard*.csv"
  python apply_qc.py --merged data/CR_Main_Mechanical_pp_merged.csv      # QC only
"""
import argparse, glob
import pandas as pd

MECH = ["K_vrh", "G_vrh", "anisotropy", "lambda_min", "born_stable"]


def merge(master_path, shard_glob):
    g = pd.read_csv(master_path)
    shards = pd.concat([pd.read_csv(f) for f in glob.glob(shard_glob, recursive=True)], ignore_index=True)
    shards = shards.drop_duplicates("cif_name", keep="last").set_index("cif_name")
    fill = g.status.isna() | (g.status != "ok")
    cols = [c for c in shards.columns if c in g.columns and c != "cif_name"]
    for c in cols:
        g.loc[fill, c] = g.loc[fill, "cif_name"].map(shards[c])
    print("merged:", g.status.value_counts(dropna=False).to_dict())
    return g


def qc(g):
    fmax_r = g["relax_fmax"].round(3)
    keep = (fmax_r <= 0.009) & (g["dV_relax"].abs() <= 0.15)
    final = g[keep].copy()
    ok = g["relax_fmax"].notna()
    print("kept:", len(final), "| excluded:", len(g) - len(final),
          "| fmax only:", int((~(fmax_r <= 0.009) & (g.dV_relax.abs() <= 0.15) & ok).sum()),
          "| dV only:", int(((fmax_r <= 0.009) & ~(g.dV_relax.abs() <= 0.15)).sum()),
          "| both:", int((~(fmax_r <= 0.009) & ~(g.dV_relax.abs() <= 0.15) & ok).sum()),
          "| fail:", int((~ok).sum()))
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master"); ap.add_argument("--shards")
    ap.add_argument("--merged", default="data/CR_Main_Mechanical_pp_merged.csv")
    ap.add_argument("--out", default="data/CR_Main_Mechanical_pp_final.csv")
    a = ap.parse_args()
    if a.master and a.shards:
        g = merge(a.master, a.shards); g.to_csv(a.merged, index=False)
    else:
        g = pd.read_csv(a.merged)
    qc(g).to_csv(a.out, index=False); print("saved", a.out)


if __name__ == "__main__":
    main()
