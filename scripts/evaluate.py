import argparse
import json
import shutil
from pathlib import Path

import yaml

from modeling import load_model
from scoring import load_rows, score_batch

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--limit", type=int, default=None)
args = ap.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

run_name = cfg["run_name"] + (f"_limit{args.limit}" if args.limit else "")

missing = [p for p in cfg["splits"].values() if not Path(p).exists()]
assert not missing, f"missing data files: {missing}"

out_dir = Path(cfg["results_root"]) / run_name
out_dir.mkdir(parents=True, exist_ok=False)
shutil.copy(args.config, out_dir / "config.yaml")

tok, model, ids = load_model(cfg["model_id"], cfg["adapter"])

for split, path in cfg["splits"].items():
    rows = load_rows(path)
    if args.limit:
        rows = rows[:args.limit]
    res = score_batch(tok, model, ids, rows, batch_size=cfg["batch_size"], split=split)

    with open(out_dir / f"{split}_preds.jsonl", "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")

    acc = sum(r["pred"] == r["gold"] for r in res) / len(res)
    mean_mass = sum(r["mass"] for r in res) / len(res)
    yes_rate = sum(r["pred"] for r in res) / len(res)
    print(f"{split}: n={len(res)} acc={acc:.3f} mean_mass={mean_mass:.3f} yes_rate={yes_rate:.3f}")
