"""
make_material_groups.py - build master_groups.csv (grouping columns for the split ladder).

Inputs
  --grouping  topo_grouping output workbook (sheet 'topo_grouping_cif'), produced by
              topo_grouping_en.py (MOFkey identity + CrystalNets fingerprint fallback for
              non-keyable CIFs) from cr_main_mofkey.xlsx and the CrystalNets descriptor file
              (topo_n_vertices, topo_n_edges, topo_td10).
  --features  features_master.csv (only the cif_name, doi columns are used)

Output columns: cif_name, doi, material_id, chemistry_id, combined_id, material_source
  material_id  {metal}||{linker keys}||{topology or fingerprint}   -> Material split
  chemistry_id {metal}||{linker keys}                              -> Chemistry split
  doi          source publication                                   -> DOI stress test
  combined_id  connected component over material_id, chemistry_id, doi -> Combined split
  material_source  mofkey | absorbed | topo_fp
"""
import argparse
import numpy as np
import pandas as pd


class UF:
    def __init__(s): s.p = {}
    def f(s, x):
        s.p.setdefault(x, x)
        while s.p[x] != x:
            s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def u(s, a, b):
        ra, rb = s.f(a), s.f(b)
        if ra != rb: s.p[ra] = rb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grouping", default="data/topo_grouping.xlsx")
    ap.add_argument("--features", default="data/features_master.csv")
    ap.add_argument("--out", default="data/master_groups.csv")
    a = ap.parse_args()

    g = pd.read_excel(a.grouping, sheet_name="topo_grouping_cif")
    feat = pd.read_csv(a.features, usecols=["cif_name", "doi"])
    m = g[["cif_name", "composition", "material_id", "material_source", "topology_status"]].merge(
        feat, on="cif_name", how="left")
    m["chemistry_id"] = m["composition"].where(m["composition"].notna(), m["material_id"])
    m["doi_norm"] = m["doi"].fillna("NA::" + m["cif_name"].astype(str))

    uf = UF()
    for i in m.index: uf.f(("r", i))
    for col in ["material_id", "chemistry_id", "doi_norm"]:
        for v, idx in m.groupby(col).groups.items():
            if pd.isna(v): continue
            base = ("r", idx[0])
            for j in idx[1:]: uf.u(base, ("r", j))
    roots, comb = {}, []
    for i in m.index:
        if pd.isna(m.loc[i, "material_id"]):
            comb.append(np.nan); continue
        r = uf.f(("r", i))
        if r not in roots: roots[r] = "PC::%06d" % (len(roots) + 1)
        comb.append(roots[r])
    m["combined_id"] = comb

    master = m[["cif_name", "doi_norm", "material_id", "chemistry_id", "combined_id", "material_source"]].copy()
    master.columns = ["cif_name", "doi", "material_id", "chemistry_id", "combined_id", "material_source"]
    master.to_csv(a.out, index=False)
    print("rows:", len(master))
    for c in ["material_id", "chemistry_id", "doi", "combined_id"]:
        print(f"  {c:13s} unique {master[c].nunique():5d} | missing {master[c].isna().sum()}")
    print(master.material_source.value_counts().to_dict())


if __name__ == "__main__":
    main()
