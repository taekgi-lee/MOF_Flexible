"""
compute_elastic_tensors.py -- CoRE MOF elastic tensor screening with ORB v3 + DFT-D3

Shard 25 of 25.  Run from inside this folder:

    conda activate mlip_orb
    python compute_elastic_tensors.py                  # full shard
    python compute_elastic_tensors.py --limit 10       # pilot: first 10 structures only
    python compute_elastic_tensors.py --fresh          # discard previous output, start over

Everything is hardcoded for this shard -- no arguments needed for normal runs.


PROTOCOL
--------
Model      : orb-v3-conservative-inf-omat (float32-highest; ORB has no float64)
Dispersion : torch-dftd D3(BJ), xc=pbe, cutoff=40 Bohr
             Matches the reference DFT, which used VASP IVDW=12 (DFT-D3 BJ)
             with S6=1.0000, S8=0.7875, A1=0.4289, A2=4.4407.
Strain     : +-1.0% (4-point fit).  Validated by strain sweep -- smaller windows
             degrade even the rigid control (UiO-66 G_vrh +71% at +-0.1%) because
             the relaxation residual, not the strain size, sets the noise floor.
             The reference DFT used VASP IBRION=6 / POTIM=0.01 / NFREE=2, i.e. the
             same maximum strain via central differences.
fmax       : 0.001 eV/A.  Validated by fmax sweep:
               MIL-53(np) lambda_min ratio : 6.50 (0.01) -> 2.58 (0.003) -> 1.58 (0.001)
               UiO-66     lambda_min ratio : 1.11 -> 1.01 -> 0.97  (crosses 1.0)
             Loose fmax under-relaxes the deformed cells and biases the tensor
             toward clamped-ion values.  The reference OUTCAR reports TOTAL ELASTIC
             MODULI, which *includes* the ionic relaxation contribution, so the
             deformed cells must be relaxed to match.

Reference validation (orb @ fmax=0.001, 4 structures with published DFT):
    K_vrh   |log ratio| mean = 0.023      (+-4%)
    G_vrh   |log ratio| mean = 0.030      (+-5%)
    lambda_min                = 0.308     (Spearman rho = 1.000 vs DFT)
    anisotropy                = 0.257     (up to 2x on the softest structure)


QC POLICY
---------
* collapse: dV_relax < -15% means the framework fell in during relaxation, so the
  tensor computed on top of it is meaningless.
* relax convergence: a failed relaxation does NOT raise -- matcalc logs
  "Maximum steps reached" and returns the non-equilibrium structure anyway, which
  would silently produce a garbage tensor under status='ok'.  matcalc reports this
  through `logging`, not `warnings`, so a log handler is installed; in addition the
  actual max|F| is measured directly after relaxation.
* born_stable is NOT a QC filter.  lambda_min <= 0 indicates proximity to a
  soft-mode instability, i.e. a *flexible candidate*.  Filtering on it would
  correlate the missingness with the label.  Recorded as signal only.
* R^2 is not recorded.  The reported "mean R^2" averages over stress components
  that are zero by symmetry, so it sits at 0.33-0.55 even when the tensor is exact
  (verified on UiO-66) and does not respond to the strain window.  It is an
  artifact, not a reliability metric.
* Failures are recorded, never dropped.  If failures concentrate in particular
  chemistries that is itself a bias worth auditing.
"""

import os
import time
import logging
import argparse
import warnings

import numpy as np
import pandas as pd
import torch

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.elasticity import ElasticTensor
import matcalc as mtc

from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from ase.calculators.mixing import SumCalculator
from ase.units import Bohr

# ---------------------------------------------------------------------------
# Hardcoded for this shard
# ---------------------------------------------------------------------------
SHARD = 0                       # NU-2002 one-off
MANIFEST = None                 # manifest not used: all *.cif in CIF_DIR
OUT_CSV = "results_orb_nu2002.csv"
CIF_DIR = "."                   # cifs sit next to this script

# ---------------------------------------------------------------------------
# Model / protocol
# ---------------------------------------------------------------------------
MODEL_NAME = "orb-v3-conservative-inf-omat"
PRECISION = "float32-highest"      # ORB does not support float64
DEVICE = "cuda"

FMAX = 0.001                       # eV/A -- see docstring
MAX_STEPS = 10000                   # default 500 can be short at this fmax
FMAX_TOL = 2.0                     # converged if measured max|F| <= FMAX * FMAX_TOL

NORM_STRAINS = (-0.01, -0.005, 0.005, 0.01)
SHEAR_STRAINS = (-0.01, -0.005, 0.005, 0.01)

COLLAPSE_DV = -0.15                # dV_relax below this => framework collapse

EV_A3_TO_GPA = 160.21766208

COLS = [
    "cif_name", "shard", "order_in_shard", "status", "error", "n_atoms",
    "K_vrh", "G_vrh", "anisotropy", "lambda_min", "born_stable",
    "C11", "C22", "C33", "C44", "C55", "C66",
    "dV_relax", "collapse", "relax_fmax", "relax_converged", "relax_maxsteps_hit",
    "fit_resid", "qc_ok", "sec", "tensor",
]

parser = argparse.ArgumentParser(description="ORB elastic + relaxed structure, NU-2002 one-off")
parser.add_argument("--limit", type=int, default=None,
                    help="pilot mode: process only the first N structures")
parser.add_argument("--fresh", action="store_true",
                    help="delete existing output and start from scratch")
parser.add_argument("--max-atoms", type=int, default=None,
                    help="skip structures above this atom count (recorded as skip_large)")
args = parser.parse_args()

if args.fresh and os.path.exists(OUT_CSV):
    os.remove(OUT_CSV)
    print(f"[fresh] removed {OUT_CSV}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def voigt_to_kelvin(c):
    """Convert a Voigt 6x6 stiffness matrix to the Kelvin (Mandel) representation.

    Voigt notation carries conventional factors of 2 on the shear components, so
    the raw 6x6 matrix is not an isometric representation and its eigenvalues are
    NOT physical eigenstiffnesses.  The Kelvin form rescales the blocks so that
    eigenvalues are meaningful.  lambda_min must be taken from this form.
    """
    c = 0.5 * (np.asarray(c, float) + np.asarray(c, float).T)
    k = c.copy()
    r2 = 2 ** 0.5
    k[0:3, 3:6] *= r2
    k[3:6, 0:3] *= r2
    k[3:6, 3:6] *= 2.0
    return k


class MaxStepsCatcher(logging.Handler):
    """matcalc reports relaxation failure through `logging`, not `warnings`.

    A relaxation that hits max_steps does not raise; it returns a non-equilibrium
    structure. Without catching this, such rows would land in the CSV as
    status='ok' with a meaningless tensor.
    """

    def __init__(self):
        super().__init__()
        self.hit = False

    def emit(self, record):
        if "Maximum steps reached" in record.getMessage():
            self.hit = True

    def reset(self):
        self.hit = False


_catcher = MaxStepsCatcher()
logging.getLogger("matcalc").addHandler(_catcher)
logging.getLogger("matcalc").setLevel(logging.WARNING)


def measure_fmax(struct):
    """Measure the actual max|F| of a relaxed structure.

    One extra force evaluation, negligible next to the relaxation itself. This is
    the definitive convergence check -- it does not depend on parsing log text.
    """
    atoms = AseAtomsAdaptor.get_atoms(struct)
    atoms.calc = calc
    return float(np.abs(atoms.get_forces()).max())


def empty_row(cif_name, status, error="", n_atoms=np.nan, order=np.nan, sec=np.nan):
    row = {c: np.nan for c in COLS}
    row.update(cif_name=cif_name, shard=SHARD, order_in_shard=order,
               status=status, error=error, n_atoms=n_atoms, sec=sec)
    return row


def append_row(row):
    """Append one row immediately, so a crash never loses completed work."""
    frame = pd.DataFrame([row], columns=COLS)
    frame.to_csv(OUT_CSV, mode="a", header=not os.path.exists(OUT_CSV), index=False)


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available -- this must run on a GPU node.")

print("=" * 70)
print(f" shard      : {SHARD}   ({MANIFEST} -> {OUT_CSV})")
print(f" model      : {MODEL_NAME} ({PRECISION}) + torch-dftd D3(bj/pbe/40 Bohr)")
print(f" matcalc    : {getattr(mtc, '__version__', '?')}")
print(f" torch      : {torch.__version__}   arch={torch.cuda.get_arch_list()}")
print(f" gpu        : {torch.cuda.get_device_name(0)}")
print(f" protocol   : fmax={FMAX}  max_steps={MAX_STEPS}")
print(f"              norm_strains={NORM_STRAINS}")
print(f"              shear_strains={SHEAR_STRAINS}")
print("=" * 70)

orbff = pretrained.orb_v3_conservative_inf_omat(device=DEVICE, precision=PRECISION)
base = ORBCalculator(orbff, device=DEVICE)
d3 = TorchDFTD3Calculator(damping="bj", xc="pbe", cutoff=40 * Bohr,
                          device=DEVICE, dtype=torch.float64)
calc = SumCalculator([base, d3])

relaxer = mtc.RelaxCalc(calc, relax_cell=True, fmax=FMAX, max_steps=MAX_STEPS)
elast = mtc.ElasticityCalc(
    calc,
    relax_structure=False,             # relaxed above, kept separate to record dV
    relax_deformed_structures=True,    # essential for flexible frameworks
    fmax=FMAX,
    norm_strains=NORM_STRAINS,
    shear_strains=SHEAR_STRAINS,
)


def process_one(cif_name, n_atoms, order):
    t0 = time.time()
    cif_path = os.path.join(CIF_DIR, f"{cif_name}.cif")
    if not os.path.exists(cif_path):
        return empty_row(cif_name, "fail", "file_not_found", n_atoms, order, 0.0)

    try:
        s0 = Structure.from_file(cif_path)

        # -- relaxation (cell + atoms) --
        _catcher.reset()
        s1 = relaxer.calc(s0)["final_structure"]
        s1.to(filename=os.path.join(CIF_DIR, f"{cif_name}_relaxed.cif"))
        maxsteps_hit = _catcher.hit
        d_vol = (s1.volume - s0.volume) / s0.volume
        collapse = bool(d_vol < COLLAPSE_DV)

        # -- convergence check: no exception is raised on failure, so measure it --
        relax_fmax = measure_fmax(s1)
        converged = bool(relax_fmax <= FMAX * FMAX_TOL)

        # -- elastic tensor --
        # Suppress matcalc's R^2 warning: it fires on every structure because of
        # symmetry-zero stress components (verified artifact), and would otherwise
        # flood the log.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = elast.calc(s1)

        c = getattr(res["elastic_tensor"], "voigt", res["elastic_tensor"])
        c = np.asarray(c, float) * EV_A3_TO_GPA
        c = 0.5 * (c + c.T)

        eig = np.linalg.eigvalsh(voigt_to_kelvin(c))   # Kelvin form, see helper
        lambda_min = float(eig.min())
        et = ElasticTensor.from_voigt(c)

        try:
            resid = float(res.get("residuals_sum", np.nan))
        except (TypeError, ValueError):
            resid = np.nan

        return {
            "cif_name": cif_name, "shard": SHARD, "order_in_shard": order,
            "status": "ok", "error": "", "n_atoms": len(s0),
            "K_vrh": float(et.k_vrh), "G_vrh": float(et.g_vrh),
            "anisotropy": float(et.universal_anisotropy),
            "lambda_min": lambda_min,
            "born_stable": bool(lambda_min > 0),   # signal only, not a QC filter
            "C11": float(c[0, 0]), "C22": float(c[1, 1]), "C33": float(c[2, 2]),
            "C44": float(c[3, 3]), "C55": float(c[4, 4]), "C66": float(c[5, 5]),
            "dV_relax": float(d_vol), "collapse": collapse,
            "relax_fmax": relax_fmax,
            "relax_converged": converged,
            "relax_maxsteps_hit": bool(maxsteps_hit),
            "fit_resid": resid,
            "qc_ok": bool((not collapse) and converged),
            "sec": time.time() - t0,
            "tensor": str(np.round(c, 4).tolist()),
        }

    except Exception as exc:
        return empty_row(cif_name, "fail", repr(exc)[:300], n_atoms, order,
                         time.time() - t0)
    finally:
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
import glob as _glob
from pymatgen.core import Structure as _S
_cifs = sorted(_glob.glob(os.path.join(CIF_DIR, "*.cif")))
manifest = pd.DataFrame({
    "cif_name": [os.path.splitext(os.path.basename(f))[0] for f in _cifs],
    "n_atoms": [len(_S.from_file(f)) for f in _cifs],
})
manifest["order_in_shard"] = range(len(manifest))
print(f"[local] {len(manifest)} cif(s) found in {os.path.abspath(CIF_DIR)}")
if args.limit:
    manifest = manifest.head(args.limit)

# Resume from this shard's own output. Re-submitting after a wall-time kill picks
# up where it left off; use --fresh to force a clean run.
done = set()
if os.path.exists(OUT_CSV):
    try:
        done = set(pd.read_csv(OUT_CSV)["cif_name"].astype(str))
        print(f"[resume] {len(done)} structures already done -- skipping")
    except Exception as exc:
        print(f"[resume] could not read {OUT_CSV} ({exc!r}); appending anyway")

todo = manifest[~manifest.cif_name.astype(str).isin(done)]
total = len(todo)
print(f"to process: {total}   (manifest {len(manifest)}, done {len(done)})")
if total:
    print(f"n_atoms range: {int(todo.n_atoms.min())} - {int(todo.n_atoms.max())}\n")

n_ok = n_fail = n_skip = 0
t_start = time.time()

for i, r in enumerate(todo.itertuples(), 1):
    n_atoms = float(r.n_atoms)

    if args.max_atoms and n_atoms > args.max_atoms:
        append_row(empty_row(r.cif_name, "skip_large",
                             f"n_atoms={int(n_atoms)}>{args.max_atoms}",
                             n_atoms, r.order_in_shard, 0.0))
        n_skip += 1
        print(f"[{i}/{total}] {r.cif_name}: SKIP (n_atoms={int(n_atoms)})", flush=True)
        continue

    row = process_one(r.cif_name, n_atoms, r.order_in_shard)
    append_row(row)

    if row["status"] == "ok":
        n_ok += 1
        flags = ""
        if row["collapse"]:
            flags += " [COLLAPSE]"
        if not row["relax_converged"]:
            flags += " [NOT-CONVERGED]"
        print(f"[{i}/{total}] {r.cif_name} (n={int(n_atoms)}): OK  "
              f"K={row['K_vrh']:.1f} G={row['G_vrh']:.1f} "
              f"lam={row['lambda_min']:.2f} A={row['anisotropy']:.1f} "
              f"dV={row['dV_relax'] * 100:+.1f}% F={row['relax_fmax']:.5f} "
              f"{row['sec']:.0f}s{flags}", flush=True)
    else:
        n_fail += 1
        print(f"[{i}/{total}] {r.cif_name} (n={int(n_atoms)}): "
              f"{row['status'].upper()}  {row['error'][:80]}", flush=True)

    if i % 20 == 0:
        elapsed = time.time() - t_start
        rate = elapsed / i
        print(f"  --- {i}/{total} | elapsed {elapsed / 3600:.1f} h | "
              f"{rate:.0f} s/structure | eta {(total - i) * rate / 3600:.1f} h ---",
              flush=True)

print(f"\ndone: OK {n_ok}, FAIL {n_fail}, SKIP {n_skip}  "
      f"in {(time.time() - t_start) / 3600:.1f} h  ->  {OUT_CSV}")
