from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")                      # write files, no window
import matplotlib.pyplot as plt
import pandas as pd

BLUE, ORANGE = "#2a78d6", "#eb6834"        # validated pair (CVD-safe)
INK, MUTED = "#2c2c2a", "#8a8a85"

ap = argparse.ArgumentParser()
ap.add_argument("table")                   # cartography_table.csv
args = ap.parse_args()

t = pd.read_csv(args.table)
t = t.sample(frac=1, random_state=0)   # shuffle so neither colour hides the other
cut33 = t["variability"].quantile(1 - 0.33)
cut10 = t["variability"].quantile(1 - 0.10)
crit2 = ((t["confidence"] > 0.9) & (t["variability"] < 0.05)).mean()
print(f"n={len(t)}  cut33={cut33:.4f}  cut10={cut10:.4f}  "
      f"share conf>0.9 & var<0.05 = {crit2:.3f}")

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(t["variability"], t["confidence"], s=4, alpha=0.35, linewidths=0,
           c=t["gold"].map({0: ORANGE, 1: BLUE}), rasterized=True)
for x, lab in ((cut33, "top 33%"), (cut10, "top 10%")):
    ax.axvline(x, color=MUTED, lw=1, ls="--")
    ax.text(x, 0.02, f" {lab}", color=INK, fontsize=8, rotation=90, va="bottom")

ax.set_xlabel("variability (std of P(gold) across 9 snapshots)", color=INK)
ax.set_ylabel("confidence (mean P(gold))", color=INK)
ax.set_xlim(left=0); ax.set_ylim(0, 1.02)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()

out = Path("report/figures"); out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "data_map.pdf"); fig.savefig(out / "data_map.png", dpi=200)
print(f"wrote {out}/data_map.pdf and .png")