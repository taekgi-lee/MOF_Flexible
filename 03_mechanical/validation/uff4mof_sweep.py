"""
uff4mof_sweep.py -- G1 reference validation with UFF4MOF classical force field

Follows the exact structure and output format of fmax_sweep.py so results merge
directly with the 7net/orb tables.  Only the calculator block differs:

    POSCAR -> CIF -> lammps-interface (-ff UFF4MOF) -> in./data. files
           -> ASE LAMMPSlib calculator
           -> matcalc RelaxCalc + ElasticityCalc   (identical protocol)

Requirements (any env with ase + matcalc + pymatgen):
    pip install lammps-interface
    conda install -c conda-forge lammps      # provides the python 'lammps' module

Key differences from the MLIP runs -- read before interpreting results:
  * NO D3 correction.  UFF4MOF carries its own Lennard-Jones dispersion;
    adding torch-dftd would double-count vdW.
  * CPU only.  Classical FF -- runs in minutes, no GPU needed.
  * Literature expectation (Wieser et al., npj Comput. Mater. 2024):
    UFF4MOF gets K_vrh roughly right through error cancellation, but
    individual C_ij can be off by up to 5x.  This run quantifies that on
    our exact protocol for the 3-way table DFT / UFF4MOF / MLIP.

Usage:
    python uff4mof_sweep.py
"""

import os
import re
import glob
import subprocess
import warnings
import numpy as np
import pandas as pd
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.elasticity import ElasticTensor
import matcalc as mtc
from ase.calculators.lammpslib import LAMMPSlib
from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor

MODEL = "uff4mof"
EV_A3_TO_GPA = 160.21766208

# -- protocol: identical to the MLIP fmax sweep at its final point ----------
FMAX_LIST = [0.001]
MAX_STEPS = 1000
STRAINS = (-0.01, -0.005, 0.005, 0.01)

CUTOFF = 12.5          # A -- standard UFF4MOF cutoff in the screening literature

REF_DIR = "reference_structures"   # DFT-optimized POSCAR/OUTCAR from Wieser & Zojer (2024); obtain from the original SI
WORK = "uff4mof_work"

structures = [
    ("UiO-66",     f"{REF_DIR}/UiO-66_111_900_POSCAR",     f"{REF_DIR}/UiO-66_111_900_OUTCAR"),
    ("MOF-5",      f"{REF_DIR}/MOF-5_111_800_POSCAR",      f"{REF_DIR}/MOF-5_111_800_OUTCAR"),
    ("MIL-53(np)", f"{REF_DIR}/MIL-53(np)_224_900_POSCAR", f"{REF_DIR}/MIL-53(np)_224_900_OUTCAR"),
    ("MIL-53(lp)", f"{REF_DIR}/MIL-53(lp)_121_900_POSCAR", f"{REF_DIR}/MIL-53(lp)_121_900_OUTCAR"),
]


def read_outcar_elastic(p):
    """OUTCAR TOTAL ELASTIC MODULI (kBar) -> 6x6 GPa, pymatgen Voigt order."""
    lines = open(p).readlines()
    start = None
    for i, ln in enumerate(lines):
        if "TOTAL ELASTIC MODULI" in ln:
            start = i
    if start is None:
        for i, ln in enumerate(lines):
            if "SYMMETRIZED ELASTIC MODULI" in ln:
                start = i
    if start is None:
        raise ValueError(f"elastic block not found: {p}")
    rows = []
    for ln in lines[start + 3: start + 9]:
        nums = re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", ln)
        rows.append([float(x) for x in nums][-6:])
    C_vasp = np.array(rows) / 10.0
    perm = [0, 1, 2, 4, 5, 3]
    return C_vasp[np.ix_(perm, perm)]


def voigt_to_kelvin(C):
    C = 0.5 * (np.asarray(C, float) + np.asarray(C, float).T)
    K = C.copy()
    r2 = 2 ** 0.5
    K[0:3, 3:6] *= r2
    K[3:6, 0:3] *= r2
    K[3:6, 3:6] *= 2.0
    return K


def lam_min(C):
    return float(np.linalg.eigvalsh(voigt_to_kelvin(C)).min())


# ---------------------------------------------------------------------------
# UFF4MOF calculator construction (the only block that differs from MLIP runs)
# ---------------------------------------------------------------------------
def build_uff4mof_calc(sid, structure):
    """structure -> CIF -> lammps-interface -> (LAMMPSlib calc, matcalc-ready structure).

    Returns BOTH the calculator and the structure re-read from the LAMMPS data
    file.  matcalc must operate on the data-file structure, not the original
    POSCAR one: LAMMPSlib scatters positions by atom index, so the ASE atom
    ordering has to match the data file exactly.
    """
    wdir = os.path.join(WORK, sid.replace("(", "_").replace(")", ""))
    os.makedirs(wdir, exist_ok=True)

    cif_path = os.path.join(wdir, "struct.cif")
    structure.to(filename=cif_path)

    cmd = ["lammps-interface", "-ff", "UFF4MOF",
           "--cutoff", "12.5", "struct.cif"]
    r = subprocess.run(cmd, cwd=wdir, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"lammps-interface failed: {r.stderr[-400:]}")

    in_files = glob.glob(os.path.join(wdir, "in.*"))
    data_files = glob.glob(os.path.join(wdir, "data.*"))
    if not in_files or not data_files:
        raise RuntimeError("lammps-interface produced no in./data. files")
    in_file, data_file = in_files[0], data_files[0]

    style_cmds, post_cmds = [], []
    seen_read = False
    skip_prefix = ("min_style", "minimize", "fix", "unfix", "run", "dump",
                   "thermo", "timestep", "velocity", "reset_timestep",
                   "compute", "variable", "print", "log", "restart",
                   "write_restart", "write_data", "units", "atom_style",
                   "boundary", "box")
    for ln in open(in_file):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        low = ln.lower()
        if low.startswith("read_data"):
            seen_read = True
            continue
        if any(low.startswith(k) for k in skip_prefix):
            continue
        (post_cmds if seen_read else style_cmds).append(ln)

    # read_data must live in the HEADER: LAMMPSlib issues change_box (to sync
    # the ASE cell) before lmpcmds run, so the box has to exist by then.
    # Styles must precede read_data because the data file contains coeffs.
    header = (["units real", "atom_style full", "boundary p p p"]
              + style_cmds
              + [f"read_data {os.path.abspath(data_file)}"])

    calc = LAMMPSlib(
        lmpcmds=post_cmds,
        lammps_header=header,
        create_atoms=False,
        create_box=False,
        boundary=False,
        keep_alive=True,
        log_file=os.path.join(wdir, "lammps.log"),
    )

    # Re-read the data file so the structure handed to matcalc has EXACTLY the
    # atom ordering LAMMPS sees (element identities recovered from masses).
    atoms = ase_read(data_file, format="lammps-data", units="real",
                     sort_by_id=True)
    s_lmp = AseAtomsAdaptor.get_structure(atoms)
    return calc, s_lmp


# ---------------------------------------------------------------------------
# Main -- same loop/output structure as fmax_sweep.py
# ---------------------------------------------------------------------------
os.makedirs(WORK, exist_ok=True)
print(f"### model   = {MODEL} via lammps-interface (cutoff {CUTOFF} A, no D3)")
print(f"### matcalc = {getattr(mtc, '__version__', '?')}")
print(f"### strain  = +-1.0% fixed (validated optimum; same max strain as ref DFT)")
print(f"### fmax    = {FMAX_LIST}")
print()

rows = []

for sid, poscar, outcar in structures:
    print("=" * 66)
    print(f"[{sid}]")

    C_dft = read_outcar_elastic(outcar)
    lam_dft = lam_min(C_dft)
    et_dft = ElasticTensor.from_voigt(C_dft)
    K_dft, G_dft = float(et_dft.k_vrh), float(et_dft.g_vrh)
    A_dft = float(et_dft.universal_anisotropy)
    print(f"  DFT: lam_min={lam_dft:.3f}  K={K_dft:.2f}  G={G_dft:.2f}  A_U={A_dft:.2f}")

    s0 = Poscar.from_file(poscar).structure

    print(f"\n  {'fmax':>7} {'dV%':>7} {'lam_min':>9} {'ratio':>7} {'K_vrh':>8} "
          f"{'G_vrh':>8} {'A_U':>8}")

    for fm in FMAX_LIST:
        try:
            calc, s_lmp = build_uff4mof_calc(sid, s0)

            relaxer = mtc.RelaxCalc(calc, relax_cell=True, fmax=fm,
                                    max_steps=MAX_STEPS)
            s1 = relaxer.calc(s_lmp)["final_structure"]
            dV = (s1.volume - s0.volume) / s0.volume

            elast = mtc.ElasticityCalc(
                calc,
                relax_structure=False,
                relax_deformed_structures=True,
                fmax=fm,
                norm_strains=STRAINS,
                shear_strains=STRAINS,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = elast.calc(s1)

            C = getattr(res["elastic_tensor"], "voigt", res["elastic_tensor"])
            C = np.asarray(C, float) * EV_A3_TO_GPA
            C = 0.5 * (C + C.T)
            et = ElasticTensor.from_voigt(C)
            lam = lam_min(C)

            rows.append(dict(
                model=MODEL, id=sid, fmax=fm, dV=dV,
                lambda_min=lam, lambda_min_DFT=lam_dft,
                lambda_min_ratio=lam / lam_dft,
                K_vrh=float(et.k_vrh), K_vrh_DFT=K_dft,
                G_vrh=float(et.g_vrh), G_vrh_DFT=G_dft,
                aniso=float(et.universal_anisotropy), aniso_DFT=A_dft,
                tensor=str(np.round(C, 3).tolist()),
            ))
            print(f"  {fm:7.3f} {dV*100:+7.2f} {lam:9.3f} {lam/lam_dft:7.2f} "
                  f"{float(et.k_vrh):8.2f} {float(et.g_vrh):8.2f} "
                  f"{float(et.universal_anisotropy):8.2f}")

            # save ELATE-ready tensor
            with open(f"tensor_{MODEL}_{sid}.txt", "w") as fh:
                for r_ in C:
                    fh.write("  ".join(f"{v:9.3f}" for v in r_) + "\n")

        except Exception as e:
            print(f"  {fm:7.3f}  FAILED: {repr(e)[:70]}")
            rows.append(dict(model=MODEL, id=sid, fmax=fm, err=repr(e)[:200]))
    print()

df = pd.DataFrame(rows)
out = f"fmax_sweep_{MODEL}.csv"
df.to_csv(out, index=False)

print("=" * 66)
print("### lambda_min ratio -- UFF4MOF vs DFT")
print("=" * 66)
ok = df[df.get("lambda_min_ratio").notna()] if "lambda_min_ratio" in df else pd.DataFrame()
if not ok.empty:
    print(ok.pivot(index="id", columns="fmax", values="lambda_min_ratio")
            .round(2).to_string())
    if len(ok) == 4:
        try:
            from scipy.stats import spearmanr
            for k, dc, mc in [("K", "K_vrh_DFT", "K_vrh"),
                              ("G", "G_vrh_DFT", "G_vrh"),
                              ("lam", "lambda_min_DFT", "lambda_min"),
                              ("A_U", "aniso_DFT", "aniso")]:
                rho = spearmanr(ok[dc], ok[mc]).correlation
                lr = np.abs(np.log((ok[mc] / ok[dc]).astype(float)
                                   .clip(lower=1e-9))).mean()
                print(f"  {k:>4}: Spearman rho = {rho:5.2f}   "
                      f"|log ratio| = {lr:.3f}")
        except ImportError:
            pass

print(f"\nsaved: {out}  +  tensor_{MODEL}_*.txt")
