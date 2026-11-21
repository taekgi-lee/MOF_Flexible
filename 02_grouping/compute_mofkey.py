#!/usr/bin/env python3
"""
Fill empty mofkey values in cr_main_mofkey using mofid (cif2mofid).

- Reads cr_main_mofkey (csv or the xlsx sheet 'cr_main_mofkey')
- Optionally validates that your mofid reproduces EXISTING mofkeys (version check)
- Runs cif2mofid on every row whose mofkey is empty, using ../cif/<cif_name>.cif
- Writes an incremental per-cif log (resumable) and a filled output file

Run from:  this directory
Env:       conda activate mofid
Usage:     python compute_mofkey.py
"""
import os, sys, csv, time, tempfile, shutil
from collections import Counter
import pandas as pd

# ----------------------------- CONFIG -----------------------------
INPUT_FILE     = "cr_main_mofkey.csv"   # or "cr_main_mofkey.xlsx" (needs openpyxl)
CIF_DIR        = "../cif"               # where the .cif files live
OUT_FILE       = "cr_main_mofkey_filled.csv"
LOG_FILE       = "mofkey_fill_log.csv"  # incremental log (enables resume)
VALIDATE_N     = 20                     # reproduce N existing mofkeys first; 0 to skip
PROGRESS_EVERY = 10
# ------------------------------------------------------------------

from mofid.run_mofid import cif2mofid


def read_input(path):
    if path.lower().endswith((".xlsx", ".xlsm")):
        return pd.read_excel(path, sheet_name="cr_main_mofkey", dtype=str).fillna("")
    return pd.read_csv(path, dtype=str).fillna("")


def run_one(cif_name):
    """Return (mofkey, topology, status)."""
    cif_path = os.path.join(CIF_DIR, cif_name + ".cif")
    if not os.path.isfile(cif_path):
        return "", "", "cif_missing"
    tmp = tempfile.mkdtemp(prefix="mofid_")
    try:
        try:
            r = cif2mofid(cif_path, output_path=tmp)   # normal signature
        except TypeError:
            r = cif2mofid(cif_path)                     # fallback if kwarg differs
        mk = (r.get("mofkey") or "").strip()
        topo = (r.get("topology") or "").strip()
        return mk, topo, ("success" if mk else "empty_result")
    except Exception as e:
        return "", "", "mofid_failed:%s" % type(e).__name__
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def validate(df, n):
    have = df[df["mofkey"].str.strip() != ""].head(n)
    if have.empty:
        print("[validate] no existing mofkeys to check\n"); return
    print("[validate] reproducing %d existing mofkeys ..." % len(have))
    match = checked = 0
    for _, row in have.iterrows():
        mk, _, st = run_one(row["cif_name"])
        if not st.startswith("success"):
            print("  %s -> %s" % (row["cif_name"], st)); continue
        checked += 1
        if mk == row["mofkey"].strip():
            match += 1
        else:
            print("  MISMATCH %s" % row["cif_name"])
            print("    old: %s" % row["mofkey"])
            print("    new: %s" % mk)
    if checked:
        pct = match / checked * 100
        print("[validate] match %d/%d = %.1f%%" % (match, checked, pct))
        if match < checked:
            print("  !! 기존 값과 불일치 -> mofid 버전 확인 필요 "
                  "(git checkout mofid1.0_archive 후 재빌드 검토)")
    print()


def main():
    if not os.path.isfile(INPUT_FILE):
        sys.exit("input not found: %s (cr_main_mofkey 파일을 이 폴더에 두세요)" % INPUT_FILE)
    df = read_input(INPUT_FILE)
    if "mofkey" not in df.columns or "cif_name" not in df.columns:
        sys.exit("input must have 'cif_name' and 'mofkey' columns")

    if VALIDATE_N > 0:
        validate(df, VALIDATE_N)

    empty_mask = df["mofkey"].str.strip() == ""
    todo = df[empty_mask]
    print("[fill] empty mofkey rows: %d" % len(todo))

    # resume: skip cif_names already in the log
    done = set()
    if os.path.isfile(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
        done = set(pd.read_csv(LOG_FILE, dtype=str)["cif_name"])
        print("[fill] resuming, %d already logged" % len(done))

    new_log = not (os.path.isfile(LOG_FILE) and os.path.getsize(LOG_FILE) > 0)
    logf = open(LOG_FILE, "a", newline="")
    w = csv.writer(logf)
    if new_log:
        w.writerow(["cif_name", "mofkey", "topology", "status"]); logf.flush()

    t0 = time.time()
    n = len(todo)
    for i, (_, row) in enumerate(todo.iterrows(), 1):
        cn = row["cif_name"]
        if cn in done:
            continue
        mk, topo, st = run_one(cn)
        w.writerow([cn, mk, topo, st]); logf.flush()
        if i % PROGRESS_EVERY == 0 or i == n:
            print("  %d/%d  (%.0fs)  last=%s [%s]" % (i, n, time.time() - t0, cn, st))
    logf.close()

    # merge log back into the dataframe
    log = pd.read_csv(LOG_FILE, dtype=str).fillna("").drop_duplicates("cif_name", keep="last")
    mk_map = dict(zip(log["cif_name"], log["mofkey"]))
    tp_map = dict(zip(log["cif_name"], log["topology"]))
    st_map = dict(zip(log["cif_name"], log["status"]))

    df["fill_status"] = ""
    for idx in df.index[empty_mask]:
        cn = df.at[idx, "cif_name"]
        df.at[idx, "fill_status"] = st_map.get(cn, "not_run")
        newmk = mk_map.get(cn, "")
        if newmk:
            df.at[idx, "mofkey"] = newmk
            if "topology_status" in df.columns:
                df.at[idx, "topology_status"] = tp_map.get(cn, "")
            if "mofkey_status" in df.columns:
                df.at[idx, "mofkey_status"] = "filled"

    df.to_csv(OUT_FILE, index=False)

    filled = int((df.loc[empty_mask, "mofkey"].str.strip() != "").sum())
    print("\n=== SUMMARY ===")
    print("empty at start :", int(empty_mask.sum()))
    print("now filled     :", filled)
    print("still empty    :", int(empty_mask.sum()) - filled)
    print("fill_status    :", dict(Counter(df.loc[empty_mask, "fill_status"])))
    print("per-cif log    :", LOG_FILE)
    print("output         :", OUT_FILE)


if __name__ == "__main__":
    main()
