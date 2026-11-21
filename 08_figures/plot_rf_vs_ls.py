"""
plot_rf_vs_ls.py - Figure 5: RF P(yes) vs label-spreading P(yes) on the 5,350 targets.

Inputs (two files):
  --rf  rf_gas_predictions.csv   (final refit RF; columns cif_name, is_train, P_yes)
  --ls  ls_gas_prediction.csv    (label spreading; columns cif_name, is_seed, P_yes,
                                  Flexibility in {No, Yes, Unsure} = gated LS calls)
Usage:
    python plot_rf_vs_ls.py --rf results/rf_gas_predictions.csv --ls ../label_spreading/ls_gas_prediction.csv
    python plot_rf_vs_ls.py ... --print      # 3.3 in single-column, 8 pt
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt

RED, GREEN, GREY = "#C1554D", "#2A9D8F", "#D9D9D9"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf", default="../results/rf_final/rf_gas_predictions.csv")
    ap.add_argument("--ls", default="../results/ls_gas_prediction.csv")
    ap.add_argument("--out", default="fig5_rf_vs_ls.png")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()

    rf = pd.read_csv(a.rf); rf = rf[~rf.is_train.astype(bool)][["cif_name", "P_yes"]]
    ls = pd.read_csv(a.ls)[["cif_name", "P_yes", "Flexibility"]].rename(columns={"P_yes": "ls_P_yes"})
    d = rf.merge(ls, on="cif_name", how="inner")
    x, y = d.ls_P_yes.to_numpy(float), d.P_yes.to_numpy(float)
    rf_no, rf_yes = (y <= 0.2), (y > 0.8)
    ls_no, ls_yes = (d.Flexibility == "No").to_numpy(), (d.Flexibility == "Yes").to_numpy()
    neither = ~(rf_no | rf_yes | ls_no | ls_yes)

    fs, size, ms, dm, lw, lfs = (8, (3.3, 3.3), 5, 16, 0.6, 7) if a.print else (11, (6.0, 5.6), 14, 60, 1.2, 10)
    plt.rcParams.update({"font.size": fs})
    fig, ax = plt.subplots(figsize=size)
    ax.scatter(x[neither], y[neither], s=ms * .6, c=GREY, alpha=.5, lw=0, rasterized=True, label="neither confident")
    ax.scatter(x[rf_no], y[rf_no], s=ms, c=RED, lw=0, label=f"RF conf No ({rf_no.sum()})", zorder=3)
    ax.scatter(x[rf_yes], y[rf_yes], s=ms, c=GREEN, lw=0, label=f"RF conf Yes ({rf_yes.sum()})", zorder=3)
    ax.scatter(x[ls_no], y[ls_no], s=dm, marker="D", facecolors="none", edgecolors=RED, lw=lw,
               label=f"LS conf No ({ls_no.sum()})", zorder=4)
    ax.scatter(x[ls_yes], y[ls_yes], s=dm, marker="D", facecolors="none", edgecolors=GREEN, lw=lw,
               label=f"LS conf Yes ({ls_yes.sum()})", zorder=4)
    for h in (0.2, 0.8):
        ax.axhline(h, ls=":", lw=.8, c="k", zorder=1)
    ax.set_xlim(-.03, 1.03); ax.set_ylim(-.03, 1.03)
    ax.set_xlabel("Label spreading P(Yes)"); ax.set_ylabel("RF P(Yes)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=lfs, loc="upper left", frameon=False, handletextpad=.4, borderaxespad=.3)
    fig.tight_layout(); fig.savefig(a.out, dpi=600, bbox_inches="tight")
    print(f"saved {a.out} targets={len(d)} RF no/yes={rf_no.sum()}/{rf_yes.sum()} "
          f"LS no/yes={ls_no.sum()}/{ls_yes.sum()} conflicts={((rf_no & ls_yes) | (rf_yes & ls_no)).sum()}")


if __name__ == "__main__":
    main()
