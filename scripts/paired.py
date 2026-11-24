from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from cartography import load_checkpoint
from metrics import compute_metrics, confusion_counts


def f1(g, p) -> float:
    return compute_metrics(confusion_counts(pd.Series(g), pd.Series(p)))["f1"]


def load_pair(root: Path, a: str, b: str):
    """Tweet gold + both runs' preds, aligned by id."""
    A = load_checkpoint(root / a / "isarcasm_test_preds.jsonl")
    B = load_checkpoint(root / b / "isarcasm_test_preds.jsonl")
    df = A[["gold", "pred"]].join(B[["gold", "pred"]], suffixes=("_a", "_b"), how="inner")
    assert len(df) == len(A) == len(B), "id sets differ"
    assert (df["gold_a"] == df["gold_b"]).all(), "gold differs"
    return df["gold_a"].to_numpy(), df["pred_a"].to_numpy(), df["pred_b"].to_numpy()


def paired_diff(g, pa, pb, n_boot=1000, alpha=0.05, seed=0):
    """F1(a) - F1(b): observed, CI from resampling the same rows for both."""
    rng = np.random.default_rng(seed)
    n = len(g)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = f1(g[idx], pa[idx]) - f1(g[idx], pb[idx])
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return f1(g, pa) - f1(g, pb), float(lo), float(hi)


PAIRS = []
for size in (33, 10):
    for seed in (1, 2):
        for a, b in (("amb", "rand"), ("amb", "lm"), ("lm", "rand")):
            PAIRS.append((f"{a}{size}_seed{seed}", f"{b}{size}_seed{seed}"))
PAIRS += [("random10_seed1", "zeroshot"), ("rand33_seed1", "zeroshot"),
          ("rand33_seed2", "zeroshot"), ("map_all_seed1_epoch3", "zeroshot")]

ap = argparse.ArgumentParser()
ap.add_argument("results_root", type=Path)
args = ap.parse_args()

rows = []
for a, b in PAIRS:
    obs, lo, hi = paired_diff(*load_pair(args.results_root, a, b))
    rows.append({"a": a, "b": b, "diff": obs, "lo": lo, "hi": hi,
                 "sig": lo > 0 or hi < 0})
res = pd.DataFrame(rows)
print(res.round(3).to_string(index=False))
Path("analysis").mkdir(exist_ok=True)
res.round(4).to_csv("analysis/paired_f1.csv", index=False)