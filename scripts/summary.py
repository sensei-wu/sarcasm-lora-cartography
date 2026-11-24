from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import pandas as pd

SKIP = ("_limit", "map_all_seed1_train_", "cartography_")
PHASE2 = re.compile(r"^(amb|lm|rand)(10|33)_seed(\d)$")

ap = argparse.ArgumentParser()
ap.add_argument("results_root")   # e.g. $BACKUP/results
args = ap.parse_args()

rows = []
for d in sorted(Path(args.results_root).iterdir()):
    if not d.is_dir() or any(s in d.name for s in SKIP):
        continue
    test = json.loads((d / "test_metrics.json").read_text())
    tw = json.loads((d / "isarcasm_test_metrics.json").read_text())
    m = PHASE2.match(d.name)
    rows.append({
        "run": d.name,
        "arm": m.group(1) if m else d.name,
        "size": int(m.group(2)) if m else None,
        "seed": int(m.group(3)) if m else None,
        "test_acc": test["accuracy"],
        "tw_P": tw["precision"], "tw_R": tw["recall"], "tw_F1": tw["f1"],
        "tw_F1_lo": tw["ci95"]["f1"][0], "tw_F1_hi": tw["ci95"]["f1"][1],
        "tw_yes": tw["yes_rate"],
    })

df = pd.DataFrame(rows).set_index("run")
df[["size", "seed"]] = df[["size", "seed"]].astype("Int64")
print(df.round(3).to_string())

p2 = df.dropna(subset=["size"])
agg = p2.groupby(["size", "arm"])[["test_acc", "tw_F1", "tw_yes"]].agg(["mean", "min", "max"])
print(agg.round(3).to_string())

out = Path("analysis"); out.mkdir(exist_ok=True)
df.round(4).to_csv(out / "summary_runs.csv")
agg.round(4).to_csv(out / "summary_arms.csv")
print("wrote analysis/summary_runs.csv, analysis/summary_arms.csv")