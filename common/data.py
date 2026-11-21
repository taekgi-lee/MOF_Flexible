"""
common/data.py - single entry point for the released data files.

The analysis scripts were written against three input tables
(label workbook with geometric columns, RAC feature table, QC-passed
mechanical table). The released data merges these into
data/descriptors.csv; the helpers below rebuild the three views so the
scripts run unchanged on the released files.

    data/LLM_Extraction.xlsx   labels, evidence, DOI (8,804 rows)
    data/descriptors.csv       geometric 10 + RAC 176 + mechanical + QC flags (8,804 rows)
    data/master_groups.csv     material / chemistry / doi / combined group ids
"""
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
LABELS_XLSX = os.path.join(DATA, "LLM_Extraction.xlsx")
DESCRIPTORS = os.path.join(DATA, "descriptors.csv")
MASTER = os.path.join(DATA, "master_groups.csv")

GEOM = ["LCD", "PLD", "density", "VF", "ASA", "VSA", "PV", "LFPD", "GSA", "GPV"]
MECH = ["K_vrh", "G_vrh", "anisotropy", "lambda_min", "born_stable"]

_desc = None
def _d():
    global _desc
    if _desc is None:
        _desc = pd.read_csv(DESCRIPTORS)
    return _desc

def read_labels():
    """Label workbook + the 10 geometric descriptors (as in the original label file)."""
    lab = pd.read_excel(LABELS_XLSX)
    return lab.merge(_d()[["cif_name"] + GEOM], on="cif_name", how="left")

def read_features():
    """RAC feature table (cif_name, doi, rac_*)."""
    d = _d()
    rac = [c for c in d.columns if c.startswith("rac_")]
    return d[["cif_name", "DOI"] + rac].rename(columns={"DOI": "doi"})

def read_mech():
    """QC-passed mechanical table (7,828 rows)."""
    d = _d()
    d = d[d.mech_qc_pass.fillna(False).astype(bool)]
    cols = ["cif_name"] + MECH + [c for c in ["C11","C22","C33","C44","C55","C66","dV_relax","relax_fmax","relax_converged","n_atoms"] if c in d.columns]
    return d[cols].reset_index(drop=True)

def read_master():
    return pd.read_csv(MASTER)
