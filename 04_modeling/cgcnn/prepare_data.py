#!/usr/bin/env python3
"""
Prepare CGCNN input directories from the existing ladder fold lists.
====================================================================

For every (split level, stimulus, fold) this writes one CGCNN data root
containing id_prop.csv, atom_init.json and symlinked CIF files.

CGCNN's get_train_val_test_loader slices the dataset by position when
--train-size / --val-size / --test-size are supplied, so id_prop.csv is
written in the order [train..., val..., test...] and the exact sizes are
recorded in split_sizes.json for the runner to pass through.

Prerequisite
------------
cgcnn/data.py CIFData shuffles id_prop_data on load
(random.seed(123); random.shuffle(...)). That must be disabled, otherwise
the position-based slicing ignores this file order and every split silently
becomes random. run_cgcnn_folds.py verifies each fold and aborts on mismatch.

Fold roles
----------
    test  = fold k
    val   = fold (k+1) % 5      (a whole fold, so groups stay intact)
    train = the remaining three folds

RF / GBR / XGBoost trained on four folds because they need no validation
set; CGCNN uses three. Report this when comparing.

Usage
-----
    python prepare_data.py \
        --train-root .. \
        --cif-dir cifs \
        --atom-init atom_init.json \
        --levels 5_comb \
        --out data
"""

import argparse
import json
import os
import shutil
import warnings

import pandas as pd

STIMULI = ["solvent", "temperature", "gas"]
LEVELS = ["1_random", "2_mofkey", "3_chemistry", "4_doi", "5_comb"]
N_FOLDS = 5


def validate_cifs(names, cif_dir, cache_path):
    """Parse each CIF once with pymatgen; return the set that fails.

    CGCNN reads structures through pymatgen, so a CIF it cannot parse kills
    the whole fold. Results are cached because parsing is slow.
    """
    cache = {}
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path))

    todo = [n for n in names if n not in cache]
    if todo:
        from pymatgen.core import Structure
        print(f"[validate] parsing {len(todo)} CIFs "
              f"({len(cache)} cached) ...", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, n in enumerate(todo, 1):
                path = os.path.join(cif_dir, f"{n}.cif")
                if not os.path.exists(path):
                    cache[n] = "missing"
                else:
                    try:
                        Structure.from_file(path)
                        cache[n] = "ok"
                    except Exception as exc:
                        cache[n] = f"parse_error: {str(exc)[:120]}"
                if i % 500 == 0:
                    print(f"  {i}/{len(todo)}", flush=True)
        with open(cache_path, "w") as fh:
            json.dump(cache, fh, indent=1)

    bad = {n: cache[n] for n in names if cache.get(n, "ok") != "ok"}
    return bad


def build_one(folds, level, stim, k, cif_dir, atom_init, out_root, copy_cifs):
    test_fold = k
    val_fold = (k + 1) % N_FOLDS

    test = folds[folds.fold == test_fold]
    val = folds[folds.fold == val_fold]
    train = folds[~folds.fold.isin([test_fold, val_fold])]

    # id_prop.csv order IS the split: get_train_val_test_loader slices by
    # position. Requires the shuffle in cgcnn/data.py CIFData to be disabled.
    ordered = pd.concat([train, val, test], ignore_index=True)

    root = os.path.join(out_root, level, f"{stim}_fold{k}")
    os.makedirs(root, exist_ok=True)

    # id_prop.csv: no header, two columns (cif_id, target)
    ordered[["cif_name", "y"]].to_csv(
        os.path.join(root, "id_prop.csv"), index=False, header=False)

    shutil.copy(atom_init, os.path.join(root, "atom_init.json"))

    missing = []
    for name in ordered.cif_name:
        src = os.path.join(cif_dir, f"{name}.cif")
        dst = os.path.join(root, f"{name}.cif")
        if not os.path.exists(src):
            missing.append(name)
            continue
        if os.path.exists(dst):
            continue
        if copy_cifs:
            shutil.copy(src, dst)
        else:
            os.symlink(os.path.abspath(src), dst)

    sizes = dict(train_size=len(train), val_size=len(val), test_size=len(test),
                 test_fold=test_fold, val_fold=val_fold,
                 n_missing_cif=len(missing),
                 pos_rate_train=float(train.y.mean()),
                 pos_rate_test=float(test.y.mean()))
    with open(os.path.join(root, "split_sizes.json"), "w") as fh:
        json.dump(sizes, fh, indent=1)

    # expected test membership; the runner checks predictions against this
    test[["cif_name", "y"]].to_csv(os.path.join(root, "test_ids.csv"), index=False)

    return sizes, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", default="..",
                    help="folder holding 1_random/ .. 5_comb/")
    ap.add_argument("--cif-dir", required=True)
    ap.add_argument("--atom-init", default="atom_init.json",
                    help="atom_init.json from the cgcnn repository")
    ap.add_argument("--levels", nargs="*", default=LEVELS)
    ap.add_argument("--stimuli", nargs="*", default=STIMULI)
    ap.add_argument("--out", default="data")
    ap.add_argument("--cif-cache", default="cif_parse_cache.json",
                    help="where to cache CIF parse results")
    ap.add_argument("--no-validate", action="store_true",
                    help="skip the pymatgen parse check")
    ap.add_argument("--copy-cifs", action="store_true",
                    help="copy instead of symlink (use if symlinks are unsupported)")
    args = ap.parse_args()

    if not os.path.exists(args.atom_init):
        raise SystemExit(
            f"[error] {args.atom_init} not found. Copy it from the cgcnn "
            f"repository (cgcnn/data/sample-regression/atom_init.json).")

    # collect every CIF referenced by any requested fold list, validate once
    referenced = set()
    for level in args.levels:
        for stim in args.stimuli:
            fp = os.path.join(args.train_root, level, "folds", f"{stim}_folds.csv")
            if os.path.exists(fp):
                referenced |= set(pd.read_csv(fp).cif_name)

    bad = {}
    if not args.no_validate:
        bad = validate_cifs(sorted(referenced), args.cif_dir, args.cif_cache)
        if bad:
            print(f"\n[validate] {len(bad)} CIF(s) unusable, excluded:")
            for n, why in list(bad.items())[:10]:
                print(f"    {n}  ->  {why}")
            if len(bad) > 10:
                print(f"    ... and {len(bad) - 10} more")
            pd.DataFrame({"cif_name": list(bad), "reason": list(bad.values())}) \
              .to_csv("excluded_cifs.csv", index=False)
            print("[validate] full list -> excluded_cifs.csv\n")
        else:
            print("[validate] all CIFs parse cleanly\n")

    all_missing = set()
    for level in args.levels:
        for stim in args.stimuli:
            fp = os.path.join(args.train_root, level, "folds", f"{stim}_folds.csv")
            if not os.path.exists(fp):
                print(f"[skip] {fp} not found")
                continue
            folds = pd.read_csv(fp)
            if bad:
                before = len(folds)
                folds = folds[~folds.cif_name.isin(bad)].reset_index(drop=True)
                if len(folds) != before:
                    print(f"  [{level}/{stim}] dropped {before - len(folds)} "
                          f"unusable CIF(s)")
            for k in range(N_FOLDS):
                sizes, missing = build_one(folds, level, stim, k, args.cif_dir,
                                           args.atom_init, args.out, args.copy_cifs)
                all_missing |= set(missing)
                print(f"  {level}/{stim}_fold{k}: train {sizes['train_size']} "
                      f"val {sizes['val_size']} test {sizes['test_size']} "
                      f"| missing cif {sizes['n_missing_cif']}")

    if all_missing:
        print(f"\n[warn] {len(all_missing)} CIF files not found, e.g. "
              f"{sorted(all_missing)[:3]}")
    print(f"\n[done] data roots under {args.out}/")


if __name__ == "__main__":
    main()
