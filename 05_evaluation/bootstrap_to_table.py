"""Convert si_S1_model_comparison.csv to wide Table S1 (rows=split, cols=stimulus x pair).
Run next to the csv:  python bootstrap_to_table.py
"""
import pandas as pd

d = pd.read_csv("si_S1_model_comparison.csv")
splits = ["Random", "Material", "Chemistry", "DOI", "Combined"]
stims = ["gas", "solvent", "temperature"]
pairs = ["rf - gbr", "rf - xgb", "gbr - xgb"]

def cell(r):
    s = f"{r.dAUC:+.3f} [{r.dAUC_lo:+.3f}, {r.dAUC_hi:+.3f}]"
    return f"**{s}**" if r.dAUC_lo * r.dAUC_hi > 0 else s

hdr = ["Split"] + [f"{st.capitalize()} {p.replace(' - ', '−').upper()}" for st in stims for p in pairs]
print("| " + " | ".join(hdr) + " |")
print("|" + "---|" * len(hdr))
for sp in splits:
    row = [sp]
    for st in stims:
        for p in pairs:
            r = d[(d.stimulus == st) & (d.split == sp) & (d.pair == p)].iloc[0]
            row.append(cell(r))
    print("| " + " | ".join(row) + " |")
