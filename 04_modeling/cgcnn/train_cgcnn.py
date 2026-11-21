#!/usr/bin/env python3
"""
train_cgcnn.py — CGCNN with balanced loss / F1 epoch selection
====================================================================

Same driver as run_cgcnn_folds.py, but for the patched main_balanced.py
(from patch_main.py) and with its own output tree so the original
AUC run is not touched:

    results_f1/
      <level>/
        <stim>_fold<k>_test_results.csv   raw per-fold output (copied out of data/)
        oof_predictions.csv               cif_name, stimulus, model, fold, y, score
        summary.csv                       pooled metrics @ 0.5
      summary_all_levels.csv

main.py still writes test_results.csv into the fold's data root; this
runner copies it into results_f1/ immediately, so a later run of the
original runner does not read balanced results by mistake.

--skip-done looks for results_f1/<level>/<stim>_fold<k>_test_results.csv
(NOT the file inside data/), so resuming never picks up the old AUC run.

Usage
-----
    python train_cgcnn.py --cgcnn-main ~/cgcnn/main_balanced.py --data data --epochs 100
    python train_cgcnn.py --cgcnn-main ~/cgcnn/main_balanced.py --levels 2_mofkey --stimuli gas
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

STIMULI = ["solvent", "temperature", "gas"]
LEVELS = ["1_random", "2_mofkey", "3_chemistry", "4_doi", "5_comb"]
N_FOLDS = 5
THR = 0.5
MODEL = "cgcnn_f1"


def run_fold(cgcnn_main, root, epochs, extra, python_exe, gpu=None):
    sizes = json.load(open(os.path.join(root, "split_sizes.json")))
    root_abs = os.path.abspath(root)
    cmd = [
        python_exe, os.path.abspath(cgcnn_main), root_abs,
        "--task", "classification",
        "--epochs", str(epochs),
        "--train-size", str(sizes["train_size"]),
        "--val-size", str(sizes["val_size"]),
        "--test-size", str(sizes["test_size"]),
    ] + extra

    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print("  $", " ".join(cmd[1:]), flush=True)
    res = subprocess.run(cmd, cwd=root, env=env)
    if res.returncode != 0:
        print(f"  [fail] returncode {res.returncode}")
        return None
    out = os.path.join(root, "test_results.csv")
    if not os.path.exists(out):
        print(f"  [fail] {out} not produced")
        return None
    return out


def load_and_check(path, root):
    df = pd.read_csv(path, header=None, names=["cif_name", "y", "score"])
    expected = pd.read_csv(os.path.join(root, "test_ids.csv"))
    got, want = set(df.cif_name), set(expected.cif_name)
    if got != want:
        print(f"  [SPLIT MISMATCH] predicted {len(got)} ids, expected {len(want)}, "
              f"overlap {len(got & want)}")
        print("  The fold assignment was not honoured -- results are invalid.")
        return None
    return df


def pooled_metrics(oof, level):
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 f1_score, accuracy_score, balanced_accuracy_score)
    rows = []
    for stim, g in oof.groupby("stimulus"):
        if g.y.nunique() < 2:
            continue
        yhat = (g.score >= THR).astype(int)
        fold_f1 = [f1_score(h.y, (h.score >= THR).astype(int), average="macro")
                   for _, h in g.groupby("fold") if h.y.nunique() > 1]
        fold_auc = [roc_auc_score(h.y, h.score)
                    for _, h in g.groupby("fold") if h.y.nunique() > 1]
        rows.append(dict(
            level=level, stimulus=stim, model=MODEL, thr=THR,
            pooled_macro_f1=f1_score(g.y, yhat, average="macro"),
            pooled_f1_pos=f1_score(g.y, yhat),
            pooled_accuracy=accuracy_score(g.y, yhat),
            pooled_balanced_accuracy=balanced_accuracy_score(g.y, yhat),
            pooled_roc_auc=roc_auc_score(g.y, g.score),
            pooled_pr_auc=average_precision_score(g.y, g.score),
            fold_f1_std=float(np.std(fold_f1)) if fold_f1 else np.nan,
            fold_auc_std=float(np.std(fold_auc)) if fold_auc else np.nan,
            n=len(g), pos=int(g.y.sum()), n_folds=g.fold.nunique()))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cgcnn-main", required=True,
                    help="path to main_balanced.py (patched)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--levels", nargs="*", default=LEVELS)
    ap.add_argument("--stimuli", nargs="*", default=STIMULI)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--gpu", default="2",
                    help="CUDA_VISIBLE_DEVICES for the child; '' to inherit")
    ap.add_argument("--skip-done", action="store_true",
                    help="reuse results_f1/<level>/<stim>_fold<k>_test_results.csv if present")
    ap.add_argument("--out", default="results_f1")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="anything after this is passed straight to main.py")
    args = ap.parse_args()

    if "balanced" not in os.path.basename(args.cgcnn_main):
        print(f"[warn] --cgcnn-main is {args.cgcnn_main}; expected main_balanced.py")

    failures, all_summ = [], []
    for level in args.levels:
        ldir = os.path.join(args.out, level)
        os.makedirs(ldir, exist_ok=True)
        rows = []
        for stim in args.stimuli:
            for k in range(N_FOLDS):
                root = os.path.join(args.data, level, f"{stim}_fold{k}")
                if not os.path.isdir(root):
                    print(f"[skip] {root}")
                    continue
                saved = os.path.join(ldir, f"{stim}_fold{k}_test_results.csv")

                if args.skip_done and os.path.exists(saved):
                    print(f"[skip] {level} / {stim} / fold {k} (found in {args.out})", flush=True)
                else:
                    print(f"[run] {level} / {stim} / fold {k}", flush=True)
                    produced = run_fold(args.cgcnn_main, root, args.epochs,
                                        args.extra, args.python,
                                        args.gpu if args.gpu else None)
                    if produced is None:
                        failures.append(f"{level}/{stim}/fold{k}")
                        continue
                    shutil.copy(produced, saved)          # keep out of data/

                df = load_and_check(saved, root)
                if df is None:
                    failures.append(f"{level}/{stim}/fold{k}")
                    continue
                df["stimulus"], df["model"], df["fold"] = stim, MODEL, k
                rows.append(df[["cif_name", "stimulus", "model", "fold", "y", "score"]])

        if not rows:
            print(f"[warn] no results for {level}")
            continue

        oof = pd.concat(rows, ignore_index=True)
        oof.to_csv(os.path.join(ldir, "oof_predictions.csv"), index=False)
        summ = pooled_metrics(oof, level)
        summ.to_csv(os.path.join(ldir, "summary.csv"), index=False)
        all_summ.append(summ)

        print(f"\n=== {level} pooled OOF @ {THR} ===")
        print(summ[["stimulus", "pooled_macro_f1", "pooled_accuracy",
                    "pooled_balanced_accuracy", "pooled_roc_auc", "n", "n_folds"]]
              .round(3).to_string(index=False))
        for _, r in summ.iterrows():
            if r.n_folds < N_FOLDS:
                print(f"  [warn] {r.stimulus}: only {r.n_folds}/{N_FOLDS} folds "
                      f"-- pooled metric is NOT comparable")
        print(f"[written] {ldir}/oof_predictions.csv, summary.csv")

    if all_summ:
        pd.concat(all_summ, ignore_index=True).to_csv(
            os.path.join(args.out, "summary_all_levels.csv"), index=False)
        print(f"\n[written] {args.out}/summary_all_levels.csv")

    if failures:
        print("\n" + "=" * 60)
        print(f"[FAILED] {len(failures)} fold(s):")
        for f in failures:
            print("   ", f)
        print("Re-run with --skip-done after fixing the cause.")
        print("=" * 60)
    else:
        print("\n[ok] every requested fold completed")


if __name__ == "__main__":
    main()
