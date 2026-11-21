"""
compute_geometry.py — geometric 10 features for NU-2002.

Run from:  this directory
Input   :  ../NU2002_348K_BCP_ordered_H.cif  (or --cif)

Zeo++ settings calibrated against CoRE MOF 2025 (HADLOQ01 check):
    PLD/LCD/LFPD : network -ha -res
    PV/GPV/VF    : probe 1.86 A
    ASA/VSA/GSA  : probe 1.655 A, 50k samples
Units: ASA A^2, VSA m^2/cm^3, GSA m^2/g, PV A^3, GPV cm^3/g.

Output: nu2002_geom.csv
"""
import argparse
import pandas as pd
from pymatgen.core import Structure
from CoREMOF.calculation import Zeopp

GEOM = ["LCD", "PLD", "density", "VF", "ASA", "VSA", "PV", "LFPD", "GSA", "GPV"]

ap = argparse.ArgumentParser()
ap.add_argument("--cif", default="NU2002_348K_BCP_ordered_H.cif")
ap.add_argument("--name", default="NU2002_348K_ordered")
a = ap.parse_args()

pd_ = Zeopp.PoreDiameter(a.cif)
sa = Zeopp.SurfaceArea(a.cif, chan_radius=1.655, probe_radius=1.655, num_samples=50000)
pv = Zeopp.PoreVolume(a.cif)
dens = Structure.from_file(a.cif).density

row = {"cif_name": a.name,
       "LCD": pd_["LCD"], "PLD": pd_["PLD"], "LFPD": pd_["LFPD"],
       "density": float(dens), "VF": pv["VF"],
       "ASA": sa["ASA"][0], "VSA": sa["ASA"][1], "GSA": sa["ASA"][2],
       "PV": pv["PV"][0], "GPV": pv["PV"][1]}
g = pd.DataFrame([row])[["cif_name"] + GEOM]
g.to_csv("nu2002_geom.csv", index=False)
print(g.T.to_string())
