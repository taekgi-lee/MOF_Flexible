"""
Recompute the topology of every CIF in the CR_data set (~16k) with CrystalNets,
and derive continuous topological descriptors directly from each MOF's
periodic graph (PeriodicGraph).

Because the descriptors are computed the same way for both recognized nets and
UNKNOWN genomes, they can be used directly as features for XGBoost + SHAP
instead of a high-cardinality one-hot encoding of net names.

Run from the folder that contains juliapkg.json (e.g. ~/crystalnet_work):
  $ cd ~/crystalnet_work
  $ python compute_topology_descriptors.py

Columns added to the CSV:
  topology_CrystalNet  : topology label (same string format as the previous run)
  topo_dim             : net dimensionality (1/2/3; 0 if identification failed)
  topo_catenation      : number of interpenetrated substructures
  topo_is_known        : 1 = named net in RCSR/etc., 0 = UNKNOWN genome
  topo_mismatch        : 1 = SingleNodes result differs from AllNodes
  topo_n_vertices      : vertices per unit cell (AllNodes clustering)
  topo_n_edges         : edges per unit cell
  topo_mean_degree     : mean coordination number
  topo_min_degree / topo_max_degree
  topo_td10            : per-vertex mean of the cumulative 10-term coordination
                         sequence (NaN if it cannot be computed)
"""

import os
import math
import pandas as pd

# ───────────────────────── Configuration ─────────────────────────
CSV_IN     = "../data/LLM_Extraction.xlsx"   # structure list
CIF_DIR    = "cifs"   # CoRE MOF 2024 CIFs (download separately)
CIF_EXT    = ".cif"
CSV_OUT    = "../data/topology.csv"
CLUSTERING = "AllNodes"   # clustering used for the descriptors ("AllNodes" recommended / "SingleNodes")
# ──────────────────────────────────────────────────────────────────

import juliacall
jl = juliacall.newmodule("CrystalNetsModule")
jl.seval("using CrystalNets")
jl.CrystalNets.toggle_warning(False)
jl.CrystalNets.toggle_export(False)

# Reach the modules already loaded by CrystalNets through their fully qualified
# paths (so juliapkg.json does not need to be modified).
jl.seval(r'''
const PGmod = CrystalNets.PeriodicGraphs
const GRmod = CrystalNets.PeriodicGraphs.Graphs

# genome (TopologicalGenome) -> (dim, nv, ne, meandeg, mindeg, maxdeg, td10)
function cn_features(genome)
    g = PGmod.PeriodicGraph(genome)
    n = GRmod.nv(g)
    if n == 0
        return (0, 0, 0, NaN, 0, 0, NaN)   # identification failed (PeriodicGraph{0})
    end
    degs = Int[GRmod.degree(g, v) for v in 1:n]
    md = sum(degs) / n
    td = NaN
    try
        s = 0.0
        for v in 1:n
            s += sum(PGmod.coordination_sequence(g, v, 10))
        end
        td = s / n
    catch
        td = NaN
    end
    return (ndims(g), n, GRmod.ne(g), md, minimum(degs), maximum(degs), td)
end
''')

CL = getattr(jl.Clustering, CLUSTERING)
CL_OTHER = jl.Clustering.SingleNodes if CLUSTERING == "AllNodes" else jl.Clustering.AllNodes


def is_known_label(s):
    # Recognized net names are short tokens without spaces (pcu, tfz-d, sqc11259, ...).
    # UNKNOWN genomes are integer encodings (contain spaces) or start with 'UNKNOWN'.
    return (" " not in s) and (not s.startswith("UNKNOWN"))


def label_and_descriptors(cif_path):
    options = jl.CrystalNets.Options(structure=jl.StructureType.MOF)
    result = jl.determine_topology(cif_path, options)
    subs = list(result)                 # interpenetrated substructures
    catenation = len(subs)

    # Label (same format as the previous run)
    parts = []
    for x in subs:
        sn = str(x[jl.Clustering.SingleNodes])
        an = str(x[jl.Clustering.AllNodes])
        parts.append(an if sn == an else f"single={sn}|all={an}")
    label = " + ".join(parts) if parts else "EMPTY"

    # Descriptors are taken from the representative (first) substructure,
    # using the selected clustering genome.
    rep = subs[0]
    g_main  = rep[CL]
    g_other = rep[CL_OTHER]
    mismatch = 1 if str(g_main) != str(g_other) else 0
    known    = 1 if is_known_label(str(g_main)) else 0

    feats = list(jl.cn_features(g_main))   # (dim, nv, ne, meandeg, mindeg, maxdeg, td10)
    dim, nv_, ne_, md, mn, mx, td = feats

    return {
        "topology_CrystalNet": label,
        "topo_dim":         int(dim),
        "topo_catenation":  int(catenation),
        "topo_is_known":    int(known),
        "topo_mismatch":    int(mismatch),
        "topo_n_vertices":  int(nv_),
        "topo_n_edges":     int(ne_),
        "topo_mean_degree": float(md),
        "topo_min_degree":  int(mn),
        "topo_max_degree":  int(mx),
        "topo_td10":        float(td),
    }


NAN_ROW = {
    "topology_CrystalNet": None, "topo_dim": math.nan, "topo_catenation": math.nan,
    "topo_is_known": math.nan, "topo_mismatch": math.nan, "topo_n_vertices": math.nan,
    "topo_n_edges": math.nan, "topo_mean_degree": math.nan, "topo_min_degree": math.nan,
    "topo_max_degree": math.nan, "topo_td10": math.nan,
}


def probe():
    """Self-check at startup using pcu. Expected: dim 3, nv 1, mean_degree 6, td10 1560."""
    g = jl.parse(jl.TopologicalGenome, "pcu")
    f = list(jl.cn_features(g))
    print(f"[probe] pcu -> dim={int(f[0])}, nv={int(f[1])}, ne={int(f[2])}, "
          f"mean_deg={float(f[3])}, td10={float(f[6])}  (expected: dim3, nv1, mean_deg6, td10~1560)")
    if not (int(f[0]) == 3 and abs(float(f[3]) - 6) < 1e-6):
        raise RuntimeError("probe failed: cannot access the graph functions correctly. "
                           "Check td10 / degree computation.")
    if math.isnan(float(f[6])):
        print("[probe][warning] td10 is disabled (coordination_sequence signature mismatch). "
              "The td10 column will be filled with NaN; all other descriptors are still computed.")


def main():
    probe()
    df = pd.read_csv(CSV_IN)
    names = df["cif_name"].astype(str).tolist()
    total = len(names)
    print(f"Processing {total} structures (clustering={CLUSTERING})")

    rows = []
    for i, name in enumerate(names, 1):
        cif_path = os.path.join(CIF_DIR, name + CIF_EXT)
        if not os.path.exists(cif_path):
            row = dict(NAN_ROW); row["topology_CrystalNet"] = "FILE_NOT_FOUND"
        else:
            try:
                row = label_and_descriptors(cif_path)
            except Exception as e:
                row = dict(NAN_ROW)
                row["topology_CrystalNet"] = "ERROR: " + str(e).replace("\n", " ")[:200]
        rows.append(row)
        if i % 50 == 0 or i == total:
            print(f"  {i}/{total} (last: {name} -> {str(row['topology_CrystalNet'])[:35]})")

    feat_df = pd.DataFrame(rows, index=df.index)
    for col in feat_df.columns:        # overwrite if a column with the same name exists
        df[col] = feat_df[col]
    df.to_csv(CSV_OUT, index=False)
    print(f"Saved -> {CSV_OUT}")

    lab = df["topology_CrystalNet"].astype(str)
    print(f"success {(~lab.str.startswith(('ERROR','FILE_NOT_FOUND'))).sum()} | "
          f"error {lab.str.startswith('ERROR').sum()} | "
          f"file_not_found {(lab == 'FILE_NOT_FOUND').sum()} | "
          f"unknown_net {int((df['topo_is_known'] == 0).sum())} | "
          f"td10_missing {int(df['topo_td10'].isna().sum())}")


if __name__ == "__main__":
    main()
