"""Dataset-cartography metrics from the 9 mapping-run checkpoints.

Pipeline:
    load_checkpoints -> build_table -> collapse_check -> select_ids -> write_id_files

Output table (one row per headline id):
    id | epoch1 | epoch2_q1 | epoch2_q2 | epoch2_q3 | epoch2 | epoch3_q1 | epoch3_q2 | epoch3_q3 | epoch3 | confidence | variability
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("/workspace/results")
RUN_PREFIX = "map_all_seed1_train_"

# Chronological order. Explicit list (not a glob) so the smoke-test dir
# `map_all_seed1_train_epoch1_limit20` is not picked up.
CHECKPOINTS = [
    "epoch1",
    "epoch2_q1", "epoch2_q2", "epoch2_q3", "epoch2",
    "epoch3_q1", "epoch3_q2", "epoch3_q3", "epoch3",
]


# --- 1. Read the 9 files into one structure keyed by id ----------------------

def load_checkpoint(path: Path) -> pd.DataFrame:
    """Read one train_preds.jsonl.

    Takes: path to a train_preds.jsonl.
    Returns: DataFrame indexed by id with columns [gold, p_gold, p_yes, pred, mass].
    """
    df = pd.read_json(path, lines=True)
    return df.set_index("id")


def load_checkpoints(results_dir: Path = RESULTS_DIR, checkpoints: list[str] = CHECKPOINTS) -> dict[str, pd.DataFrame]:
    """Load every checkpoint listed in `checkpoints`.

    Takes: results dir and checkpoint names.
    Returns: {checkpoint_name: DataFrame from load_checkpoint}. Should assert that
             every file exists and that all 9 share the identical id set and gold labels.
    """
    preds = {}
    for name in checkpoints:
        path = results_dir / f"{RUN_PREFIX}{name}" / "train_preds.jsonl"
        assert path.exists(), f"missing: {path}"
        preds[name] = load_checkpoint(path)

    first = preds[checkpoints[0]]
    for name, df in preds.items():
        assert df.index.is_unique, f"duplicate ids in {name}"
        assert set(df.index) == set(first.index), f"id set differs in {name}"
        assert (df["gold"].reindex(first.index) == first["gold"]).all(), f"gold differs in {name}"
    return preds


def build_table(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pivot per-checkpoint frames into the wide p_gold table.

    Takes: output of load_checkpoints.
    Returns: DataFrame indexed by id, one p_gold column per checkpoint (CHECKPOINTS order),
             plus a `gold` column. No confidence/variability yet.
    """
    table = pd.DataFrame({name: df["p_gold"] for name, df in preds.items()})
    table["gold"] = preds[CHECKPOINTS[0]]["gold"]
    return table


# --- 2. Mean and std per id ---------------------------------------------------

def add_confidence_variability(table: pd.DataFrame) -> pd.DataFrame:
    """Add `confidence` (mean p_gold) and `variability` (std p_gold) across checkpoints.

    Takes: wide table from build_table.
    Returns: copy of the table with two new columns. (Decide ddof: population ddof=0
             as in Swayamdipta et al. 2020, vs pandas default ddof=1.)
    """
    out = table.copy()
    p = out[CHECKPOINTS]
    out["confidence"] = p.mean(axis=1)
    out["variability"] = p.std(axis=1, ddof=0)
    return out


# --- 3. Collapse-check numbers ------------------------------------------------

def collapse_check(table: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> dict[str, float | dict]:
    """Sanity numbers showing whether training collapsed (e.g. predicting one class).

    Takes: table with confidence/variability, and the raw per-checkpoint frames.
    Returns: dict, tentatively:
        - frac_pred_yes per checkpoint        (collapse to a constant label -> ~0 or ~1)
        - accuracy per checkpoint
        - mean `mass` per checkpoint          (p_yes+p_no mass; low -> model leaving the yes/no format)
        - frac_low_var                        (share of ids with variability < eps)
        - mean variability, mean confidence
    """
    raise NotImplementedError


# --- 4. Rank headlines and write the three id files ---------------------------

def select_ids(table: pd.DataFrame, frac: float = 0.33) -> dict[str, list[str]]:
    """Rank ids into the three cartography regions.

    Takes: table with confidence/variability; `frac` = share of the dataset per region.
    Returns: {"easy": top-frac by confidence (low variability tiebreak),
              "ambiguous": top-frac by variability,
              "hard": bottom-frac by confidence}  -> lists of ids.
    """
    n = round(len(table) * frac)
    easy = table.sort_values(["confidence", "variability"], ascending=[False, True]).index[:n]
    ambiguous = table.sort_values("variability", ascending=False).index[:n]
    hard = table.sort_values("confidence", ascending=True).index[:n]
    return {"easy": list(easy), "ambiguous": list(ambiguous), "hard": list(hard)}


def label_matched_random(table: pd.DataFrame, ref_ids: list[str], seed: int) -> list[str]:
    """Random ids with exactly as many sarcastic / non-sarcastic rows as `ref_ids`."""
    n_pos = int(table.loc[ref_ids, "gold"].sum())
    n_neg = len(ref_ids) - n_pos
    pos = table[table["gold"] == 1].sample(n=n_pos, random_state=seed).index
    neg = table[table["gold"] == 0].sample(n=n_neg, random_state=seed).index
    return sorted(pos.tolist() + neg.tolist())


def write_id_files(ids: dict[str, list[str]], out_dir: Path) -> list[Path]:
    """Write one id per line to <out_dir>/{easy,ambiguous,hard}_ids.txt.

    Takes: output of select_ids and an output dir (created if missing).
    Returns: the three paths written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, id_list in ids.items():
        p = out_dir / f"{name}_ids.txt"
        p.write_text("\n".join(id_list) + "\n")
        paths.append(p)
    return paths


FRACS = (0.33, 0.10)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)

    table = add_confidence_variability(build_table(load_checkpoints(args.results_dir)))
    table.to_csv(args.out_dir / "cartography_table.csv")

    for frac in FRACS:
        ids = select_ids(table, frac)
        for seed in args.seeds:
            ids[f"label_matched_seed{seed}"] = label_matched_random(table, ids["ambiguous"], seed)
        write_id_files(ids, args.out_dir / f"frac{round(frac * 100)}")
        for name, id_list in ids.items():
            sub = table.loc[id_list]
            print(f"{frac:.2f} {name:20s} n={len(sub)} sarcastic={sub.gold.mean():.3f} "
                  f"conf={sub.confidence.mean():.4f} var_median={sub.variability.median():.4f}")


if __name__ == "__main__":
    main()
