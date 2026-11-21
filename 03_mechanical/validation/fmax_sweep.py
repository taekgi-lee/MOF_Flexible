"""
fmax sweep — strain_sweep.py 후속

배경:
  strain sweep 결과, window 를 줄일수록 rigid 대조군(UiO-66)까지 단조로 붕괴
  (±0.1% 에서 G_vrh +71% 과대). 즉 잡음 바닥을 정하는 건 strain 크기가 아니라
  '변형셀 이완 잔차(fmax)' -> strain 을 줄이면 응력 신호만 줄고 잡음은 그대로.
  => ±1.0% 가 최적 작동점. (B) 조화영역 이탈 가설은 기각.

이 스크립트가 답하는 것:
  그렇다면 ±1% 결과 자체는 잡음이 아닌가?
  strain 을 ±1% 로 고정하고 fmax 만 조여서 (0.01 -> 0.003 -> 0.001)
  λ_min ratio 가 움직이는지 본다.

    ratio 그대로   => 잡음 아님. (A) 모델이 soft mode 를 못 배움. G1 실패 확정.
    ratio 하락     => fmax 가 지배. 조인 fmax 로 전부 재계산 필요.

설계:
  - relax(사전이완) 는 fmax 조건마다 다시 함. 이완 수렴도 자체가 변수라서.
  - 대조군 UiO-66(rigid) 포함: fmax 조여도 rigid 는 안 변해야 정상.
  - MIL-53(np) 가 주 타겟 (±1% 에서 ratio 9.40 으로 최악).

사용법:
  conda activate mlip_7net && python fmax_sweep.py 7net
"""

import re
import sys
import time
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

# ★ strain 은 고정 (sweep 에서 최적으로 판정됨 + 참조 DFT 와 동일한 최대 strain)
STRAINS = (-0.01, -0.005, 0.005, 0.01)

# ★ 이번에 흔드는 변수
FMAX_LIST = [0.001]

REF_DIR = "reference_structures"   # DFT-optimized POSCAR/OUTCAR from Wieser & Zojer (2024); obtain from the original SI

structures = [
    ("UiO-66",     f"{REF_DIR}/UiO-66_111_900_POSCAR",     f"{REF_DIR}/UiO-66_111_900_OUTCAR"),
    ("MOF-5",      f"{REF_DIR}/MOF-5_111_800_POSCAR",      f"{REF_DIR}/MOF-5_111_800_OUTCAR"),
    ("MIL-53(np)", f"{REF_DIR}/MIL-53(np)_224_900_POSCAR", f"{REF_DIR}/MIL-53(np)_224_900_OUTCAR"),
    ("MIL-53(lp)", f"{REF_DIR}/MIL-53(lp)_121_900_POSCAR", f"{REF_DIR}/MIL-53(lp)_121_900_OUTCAR"),
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
    raise SystemExit(f"모르는 모델: {MODEL}")

d3 = TorchDFTD3Calculator(damping="bj", xc="pbe", cutoff=40 * Bohr,
                          device=DEVICE, dtype=torch.float64)
calc = SumCalculator([base, d3])

print(f"### model   = {MODEL} + torch-dftd D3(bj/pbe/40Bohr)")
print(f"### matcalc = {getattr(mtc, '__version__', '?')}")
print(f"### strain  = ±1.0% 고정 (sweep 최적점, 참조 DFT 와 동일 최대 strain)")
print(f"### 변수    = fmax {FMAX_LIST}")
print(f"### 질문    = ±1% 결과가 이완 잔차(잡음)에 지배되는가?\n")

rows = []

for sid, poscar, outcar in structures:
    print("=" * 66)
    print(f"[{sid}]")

    C_dft = read_outcar_elastic(outcar)
    lam_dft = lam_min(C_dft)
    et_dft = ElasticTensor.from_voigt(C_dft)
    K_dft, G_dft = float(et_dft.k_vrh), float(et_dft.g_vrh)
    A_dft = float(et_dft.universal_anisotropy)
    print(f"  DFT: λ_min={lam_dft:.3f}  K={K_dft:.2f}  G={G_dft:.2f}  A_U={A_dft:.2f}")

    s0 = Poscar.from_file(poscar).structure

    print(f"\n  {'fmax':>7} {'dV%':>7} {'λ_min':>9} {'ratio':>7} {'K_vrh':>8} "
          f"{'G_vrh':>8} {'A_U':>8} {'R²':>7} {'sec':>6}")

    for fm in FMAX_LIST:
        t0 = time.time()
        try:
            # ── 이완도 fmax 조건에 맞춰 다시 (수렴도 자체가 변수) ──
            relaxer = mtc.RelaxCalc(calc, relax_cell=True, fmax=fm)
            s1 = relaxer.calc(s0)["final_structure"]
            dV = (s1.volume - s0.volume) / s0.volume

            elast = mtc.ElasticityCalc(
                calc,
                relax_structure=False,          # 위에서 이미 이완
                relax_deformed_structures=True,
                fmax=fm,                        # ★ 변형셀 이완도 같은 fmax
                norm_strains=STRAINS,
                shear_strains=STRAINS,
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
            dt = time.time() - t0

            rows.append(dict(
                model=MODEL, id=sid, fmax=fm, dV=dV,
                lambda_min=lam, lambda_min_DFT=lam_dft, lambda_min_ratio=lam / lam_dft,
                K_vrh=float(et.k_vrh), K_vrh_DFT=K_dft,
                G_vrh=float(et.g_vrh), G_vrh_DFT=G_dft,
                aniso=float(et.universal_anisotropy), aniso_DFT=A_dft,
                r2=r2, sec=dt,
                tensor=str(np.round(C, 3).tolist()),
            ))
            print(f"  {fm:7.3f} {dV*100:+7.2f} {lam:9.3f} {lam/lam_dft:7.2f} "
                  f"{float(et.k_vrh):8.2f} {float(et.g_vrh):8.2f} "
                  f"{float(et.universal_anisotropy):8.2f} {r2:7.3f} {dt:6.0f}")
        except Exception as e:
            print(f"  {fm:7.3f}  실패: {repr(e)[:64]}")
            rows.append(dict(model=MODEL, id=sid, fmax=fm, err=repr(e)[:200]))
    print()

df = pd.DataFrame(rows)
out = f"fmax_sweep_{MODEL}.csv"
df.to_csv(out, index=False)

print("=" * 66)
print("### λ_min ratio vs fmax — 최종 판정")
print("=" * 66)
if "lambda_min_ratio" in df:
    ok = df[df.lambda_min_ratio.notna()]
    if not ok.empty:
        print(ok.pivot(index="id", columns="fmax", values="lambda_min_ratio").round(2).to_string())
        print("\n  ratio 그대로 => 잡음 아님. (A) 모델이 soft mode 를 못 배움. G1 실패 확정.")
        print("  ratio 하락   => fmax 가 지배. 조인 fmax 로 전부 재계산.")
        print("  rigid 도 변함 => 이완 수렴 자체가 불안정.")
        print("\n### G_vrh (DFT 대비)")
        print(ok.pivot(index="id", columns="fmax", values="G_vrh").round(2).to_string())

print(f"\ncsv 저장: {out}")
