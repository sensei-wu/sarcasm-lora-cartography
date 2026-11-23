"""Classification metrics for one results dir.

Usage:
    python scripts/metrics.py <results_dir>

For every {split}_preds.jsonl in <results_dir>, writes {split}_metrics.json
next to it and prints one summary line. Positive class = sarcastic = 1 = "Yes".

Pipeline:
    find_pred_files -> load_preds -> confusion_counts -> compute_metrics -> write_metrics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import numpy as np

from cartography import load_checkpoint  # indexed by id, has `gold` and `pred`

POSITIVE = 1


# --- 1. Find and read ---------------------------------------------------------

def find_pred_files(results_dir: Path) -> dict[str, Path]:
    """Locate every {split}_preds.jsonl directly inside results_dir.

    Takes: results dir.
    Returns: {split: path}, e.g. {"val": .../val_preds.jsonl, "isarcasm_test": ...}.
             Split name is the filename minus the `_preds.jsonl` suffix.
             Errors if none are found.
    """
    files = {p.name.removesuffix("_preds.jsonl"): p for p in sorted(results_dir.glob("*_preds.jsonl"))}
    assert files, f"no *_preds.jsonl in {results_dir}"
    return files


def load_preds(path: Path) -> pd.DataFrame:
    """Read one preds file (reuses cartography.load_checkpoint).

    Takes: path to a {split}_preds.jsonl.
    Returns: DataFrame indexed by id with at least `gold` and `pred` (both 0/1).
             Asserts the labels are only {0, 1} and ids are unique.
    """
    df = load_checkpoint(path)
    assert df.index.is_unique, f"duplicate ids in {path}"
    assert set(df["gold"]) <= {0, 1} and set(df["pred"]) <= {0, 1}, f"labels not 0/1 in {path}"
    return df


# --- 2. Counts ----------------------------------------------------------------

def confusion_counts(gold: pd.Series, pred: pd.Series) -> dict[str, int]:
    """Count the four cells, sarcastic = 1 as positive.

    Takes: gold and pred label series (aligned).
    Returns: {"tp": int, "fp": int, "fn": int, "tn": int}; sums to len(gold).
    """
    return {
        "tp": int(((gold == POSITIVE) & (pred == POSITIVE)).sum()),
        "fp": int(((gold != POSITIVE) & (pred == POSITIVE)).sum()),
        "fn": int(((gold == POSITIVE) & (pred != POSITIVE)).sum()),
        "tn": int(((gold != POSITIVE) & (pred != POSITIVE)).sum()),
    }


# --- 3. Metrics from counts ---------------------------------------------------

def safe_div(num: float, den: float, zero_division: float = 0.0) -> float:
    """num / den, or `zero_division` when den == 0.

    Decision for precision when TP + FP == 0 (model never said Yes):
    return 0.0 (sklearn's default), so F1 is also 0.0 and the JSON never holds NaN.
    The counts in the output make that case recoverable (tp + fp == 0).
    """
    return num / den if den else zero_division


def compute_metrics(counts: dict[str, int]) -> dict[str, float | int]:
    """Derive accuracy, precision, recall, F1 from the four counts only.

    Takes: output of confusion_counts.
    Returns: {"n", "tp", "fp", "fn", "tn", "accuracy", "precision", "recall", "f1"}.
             Pure function of the counts (no dataframes), so the later bootstrap
             can call it on resampled counts.
    """
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    n = tp + fp + fn + tn
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "n": n, **counts,
        "accuracy": safe_div(tp + tn, n),
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "yes_rate": safe_div(tp + fp, n),
    }


# --- 4. Output ----------------------------------------------------------------

def write_metrics(metrics: dict, preds_path: Path) -> Path:
    """Write {split}_metrics.json next to the preds file.

    Takes: metrics dict and the preds file it came from.
    Returns: path written (preds_path with `_preds.jsonl` -> `_metrics.json`).
    """
    out = preds_path.with_name(preds_path.name.replace("_preds.jsonl", "_metrics.json"))
    out.write_text(json.dumps(metrics, indent=2) + "\n")
    return out


def format_line(split: str, metrics: dict) -> str:
    """One-line summary, e.g. 'val  n=1000  acc=0.812  P=0.790  R=0.840  F1=0.814'.

    Takes: split name and metrics dict.
    Returns: the string (printed by main).
    """
    m = metrics
    return (f"{split:14s} n={m['n']:5d} acc={m['accuracy']:.3f} P={m['precision']:.3f} "
            f"R={m['recall']:.3f} F1={m['f1']:.3f} yes_rate={m['yes_rate']:.3f}")


def evaluate_split(split: str, preds_path: Path) -> dict:
    """load_preds -> confusion_counts -> compute_metrics -> write_metrics -> print line.

    Takes: split name and preds path.
    Returns: the metrics dict.
    """
    df = load_preds(preds_path)
    metrics = compute_metrics(confusion_counts(df["gold"], df["pred"]))
    metrics["ci95"] = bootstrap_ci(df["gold"], df["pred"])
    write_metrics(metrics, preds_path)
    print(format_line(split, metrics))
    return metrics


# --- 5. Later: bootstrap CIs (separate function, not wired into main yet) ----

def bootstrap_ci(gold: pd.Series, pred: pd.Series, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CIs over examples.

    Takes: gold, pred, number of resamples, two-sided alpha, rng seed.
    Returns: {"accuracy": (lo, hi), "precision": ..., "recall": ..., "f1": ...}.
    TODO: not implemented in this pass.
    """
    rng = np.random.default_rng(seed)
    g, p = gold.to_numpy(), pred.to_numpy()
    n = len(g)
    values = {k: [] for k in ("accuracy", "precision", "recall", "f1")}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = compute_metrics(confusion_counts(pd.Series(g[idx]), pd.Series(p[idx])))
        for k in values:
            values[k].append(m[k])
    return {k: (float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2)))
            for k, v in values.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=Path)
    args = ap.parse_args()

    for split, path in find_pred_files(args.results_dir).items():
        evaluate_split(split, path)


if __name__ == "__main__":
    main()
