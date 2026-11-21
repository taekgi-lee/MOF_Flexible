#!/usr/bin/env python3
"""
patch_main.py — make main_balanced.py from the reference CGCNN main.py

Changes (classification task only)
    1. nn.NLLLoss(weight=w)   w = balanced class weights computed from the
                              TRAIN slice of id_prop.csv (first train_size rows;
                              valid because the CIFData shuffle is disabled)
    2. class_eval fscore      average='binary' -> average='macro'
    3. best-epoch selection   val AUC -> val macro F1

Everything else (architecture, optimiser, test_results.csv format) is untouched.

Usage
-----
    python patch_main.py ~/cgcnn/main.py
    # -> writes ~/cgcnn/main_balanced.py
"""

import sys
import os

src = sys.argv[1]
dst = os.path.join(os.path.dirname(os.path.abspath(src)), "main_balanced.py")
s = open(src).read()


def rep(old, new, what):
    global s
    if s.count(old) != 1:
        raise SystemExit(f"[patch] '{what}': expected exactly 1 match, found {s.count(old)}.\n"
                         f"        Your main.py differs from the reference; edit by hand.")
    s = s.replace(old, new)
    print(f"[patch] {what}: ok")


# 1. balanced NLLLoss ----------------------------------------------------
rep("""    if args.task == 'classification':
        criterion = nn.NLLLoss()
""",
"""    if args.task == 'classification':
        # balanced class weights from the train slice (positional split,
        # CIFData shuffle disabled -> first train_size rows are train)
        _y = np.array([int(float(row[1])) for row in
                       dataset.id_prop_data[:args.train_size]])
        _cnt = np.bincount(_y, minlength=2).astype(float)
        _w = torch.tensor(len(_y) / (2.0 * np.maximum(_cnt, 1)),
                          dtype=torch.float)
        if args.cuda:
            _w = _w.cuda()
        print(f"[balanced] train counts {_cnt.astype(int).tolist()} "
              f"-> class weights {_w.cpu().numpy().round(3).tolist()}")
        criterion = nn.NLLLoss(weight=_w)
""", "balanced NLLLoss")

# 2. macro F1 in class_eval ------------------------------------------------
rep("""            target_label, pred_label, average='binary')""",
    """            target_label, pred_label, average='macro')""",
    "class_eval macro F1")

# 3. select best epoch by val macro F1 ------------------------------------
# validate() returns auc_scores.avg for classification; return fscores.avg
rep("""        return auc_scores.avg""",
    """        return fscores.avg   # best-epoch selection by macro F1 (patched)""",
    "epoch selection by F1")

open(dst, "w").write(s)
print(f"[patch] written {dst}")
