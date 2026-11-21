"""
Strain-window sweep — 실행계획 §4.4 / §8 P1

목적: λ_min 과대(최대 9.4배)가
  (A) 모델이 soft mode 를 못 배움          -> window 줄여도 그대로
  (B) ±1% 가 MLIP PES 의 조화영역 밖       -> window 줄이면 ratio -> 1 수렴
중 어느 쪽인지 판정.

설계:
  - relax 는 window 당 반복하지 않고 '한 번만' 수행한 뒤 모든 window 가
    같은 평형 구조를 공유. -> window 효과만 분리됨 (통제).
    (RelaxCalc(fmax=0.01) 로 이완 후 ElasticityCalc(relax_structure=False))
  - 참조 DFT: VASP IBRION=6, POTIM=0.01, NFREE=2 -> ±1% 중심차분 2점.
    우리 기본은 ±1% 4점 회귀. 최대 strain 은 같지만 방식이 다름.
  - rigid 대조군(UiO-66) 포함: window 를 줄여도 rigid 는 안 변해야 정상.
    rigid 까지 흔들리면 수치 잡음 문제(변형이 너무 작음).

사용법:
  conda activate mlip_7net && python strain_sweep.py 7net
  conda activate mlip_orb  && python strain_sweep.py orb
"""

import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.elasticity import ElasticTensor
import matcalc as mtc

from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from ase.calculators.mixing import SumCalculator
from ase.units import Bohr
import torch

MODEL = sys.argv[1] if len(sys.argv) > 1 else "7net"
DEVICE = "cuda"
EV_A3_TO_GPA = 160.21766208
FMAX = 0.01

REF_DIR = "reference_structures"   # DFT-optimized POSCAR/OUTCAR from Wieser & Zojer (2024); obtain from the original SI

# rigid 대조군 1 + flexible 2
structures = [
    ("UiO-66",     f"{REF_DIR}/UiO-66_111_900_POSCAR",     f"{REF_DIR}/UiO-66_111_900_OUTCAR"),
    ("MIL-53(np)", f"{REF_DIR}/MIL-53(np)_224_900_POSCAR", f"{REF_DIR}/MIL-53(np)_224_900_OUTCAR"),
    ("MIL-53(lp)", f"{REF_DIR}/MIL-53(lp)_121_900_POSCAR", f"{REF_DIR}/MIL-53(lp)_121_900_OUTCAR"),
]

# (라벨, norm_strains, shear_strains) — 크→소
WINDOWS = [
    ("±1.0%",  (-0.01,  -0.005,   0.005,   0.01 )),
    ("±0.5%",  (-0.005, -0.0025,  0.0025,  0.005)),
    ("±0.2%",  (-0.002, -0.001,   0.001,   0.002)),
    ("±0.1%",  (-0.001, -0.0005,  0.0005,  0.001)),
]


def read_outcar_elastic(p):
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
        raise ValueError(f"elastic 블록을 못 찾음: {p}")
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


def extract_r2(res, wlist):
    for k in ["r2_score", "r2_scores", "fit_r2", "r2", "mean_r2", "r2_mean"]:
        if k in res:
            try:
                return float(np.asarray(res[k], dtype=float).mean())
            except (TypeError, ValueError):
                pass
    for w in wlist:
        m = re.search(r"R²\s*=\s*([0-9.eE+-]+)", str(w.message))
        if m:
            return float(m.group(1))
    return np.nan


# ── calculator ───────────────────────────────────────────
if MODEL == "7net":
    from sevenn.calculator import SevenNetCalculator
    base = SevenNetCalculator(model="7net-mf-ompa", modal="mpa", device=DEVICE)
elif MODEL == "orb":
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator
    base = ORBCalculator(
        pretrained.orb_v3_conservative_inf_omat(device=DEVICE, precision="float32-highest"),
        device=DEVICE,
    )
else:
    raise SystemExit(f"모르는 모델: {MODEL}  (7net | orb)")

d3 = TorchDFTD3Calculator(damping="bj", xc="pbe", cutoff=40 * Bohr,
                          device=DEVICE, dtype=torch.float64)
calc = SumCalculator([base, d3])

relaxer = mtc.RelaxCalc(calc, relax_cell=True, fmax=FMAX)

print(f"### model   = {MODEL} + torch-dftd D3(bj/pbe/40Bohr)")
print(f"### matcalc = {getattr(mtc, '__version__', '?')}")
print(f"### 설계    = relax 1회 후 모든 window 가 동일 평형구조 공유 (fmax={FMAX})")
print(f"### 참조DFT = VASP IBRION=6 POTIM=0.01 NFREE=2 -> ±1% 중심차분 2점\n")

rows = []

for sid, poscar, outcar in structures:
    print("=" * 62)
    print(f"[{sid}]")

    C_dft = read_outcar_elastic(outcar)
    lam_dft = lam_min(C_dft)
    et_dft = ElasticTensor.from_voigt(C_dft)
    K_dft, G_dft = float(et_dft.k_vrh), float(et_dft.g_vrh)
    A_dft = float(et_dft.universal_anisotropy)
    print(f"  DFT: λ_min={lam_dft:.3f}  K={K_dft:.2f}  G={G_dft:.2f}  A_U={A_dft:.2f}")

    # ── relax 1회 (모든 window 공유) ──
    s0 = Poscar.from_file(poscar).structure
    try:
        s1 = relaxer.calc(s0)["final_structure"]
        dV = (s1.volume - s0.volume) / s0.volume
        print(f"  relax: dV={dV*100:+.2f}%  (vol {s0.volume:.1f} -> {s1.volume:.1f})")
    except Exception as e:
        print(f"  relax 실패: {e!r}")
        continue

    print(f"\n  {'window':>8} {'λ_min':>9} {'ratio':>7} {'K_vrh':>8} {'G_vrh':>8} "
          f"{'A_U':>8} {'R²':>8}")

    for wlabel, strains in WINDOWS:
        try:
            elast = mtc.ElasticityCalc(
                calc,
                relax_structure=False,          # ★ 이미 이완됨
                relax_deformed_structures=True,
                fmax=FMAX,
                norm_strains=strains,
                shear_strains=strains,
            )
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter("always")
                res = elast.calc(s1)

            C = getattr(res["elastic_tensor"], "voigt", res["elastic_tensor"])
            C = np.asarray(C, float) * EV_A3_TO_GPA
            C = 0.5 * (C + C.T)
            et = ElasticTensor.from_voigt(C)

            lam = lam_min(C)
            r2 = extract_r2(res, wl)
            rec = dict(
                model=MODEL, id=sid, window=wlabel,
                max_strain=max(abs(x) for x in strains),
                lambda_min=lam, lambda_min_DFT=lam_dft, lambda_min_ratio=lam / lam_dft,
                K_vrh=float(et.k_vrh), K_vrh_DFT=K_dft,
                G_vrh=float(et.g_vrh), G_vrh_DFT=G_dft,
                aniso=float(et.universal_anisotropy), aniso_DFT=A_dft,
                r2=r2,
                tensor=str(np.round(C, 3).tolist()),
            )
            rows.append(rec)
            print(f"  {wlabel:>8} {lam:9.3f} {lam/lam_dft:7.2f} {rec['K_vrh']:8.2f} "
                  f"{rec['G_vrh']:8.2f} {rec['aniso']:8.2f} {r2:8.4f}")
        except Exception as e:
            print(f"  {wlabel:>8}  실패: {repr(e)[:70]}")
            rows.append(dict(model=MODEL, id=sid, window=wlabel, err=repr(e)[:200]))
    print()

df = pd.DataFrame(rows)
out = f"strain_sweep_{MODEL}.csv"
df.to_csv(out, index=False)

# ── 판정 표 ──────────────────────────────────────────────
print("\n" + "=" * 62)
print("### λ_min ratio vs strain window — A/B 판정")
print("=" * 62)
ok = df[df.get("lambda_min_ratio").notna()] if "lambda_min_ratio" in df else pd.DataFrame()
if not ok.empty:
    piv = ok.pivot(index="id", columns="window", values="lambda_min_ratio")
    cols = [w for w, _ in WINDOWS if w in piv.columns]
    print(piv[cols].round(2).to_string())
    print("\n  window 줄일수록 ratio -> 1 수렴  => (B) 조화영역 이탈. window 고쳐서 전부 재계산")
    print("  window 줄여도 ratio 그대로       => (A) 모델이 soft mode 를 못 배움. G1 실패 확정")
    print("  rigid(UiO-66)까지 흔들림         => 수치 잡음. 그 window 는 너무 작음")

    print("\n### R² vs window")
    print(ok.pivot(index="id", columns="window", values="r2")[cols].round(3).to_string())

print(f"\ncsv 저장: {out}")
