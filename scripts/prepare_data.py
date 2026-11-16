"""Prepare the News Headlines sarcasm dataset and the iSarcasmEval Task A test set.

Outputs (data/processed/):
    train.jsonl, val.jsonl, test.jsonl   headline splits (80/10/10, stratified)
    isarcasm_test.jsonl                  out-of-domain test set
    stats.json                           counts, balances, baseline, file hashes

Each .jsonl row: {"id", "text", "label", "domain"}. article_link is never written,
because the domain fully determines the label.

Run from the repo root:  python scripts/prepare_data.py
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------- constants
SEED = 13
MAX_WORDS = 60

ROOT = Path(__file__).resolve().parent.parent
RAW_HEADLINES = ROOT / "data" / "raw" / "Sarcasm_Headlines_Dataset_v2.json"
RAW_ISARCASM = ROOT / "data" / "raw" / "task_A_En_test.csv"
OUT_DIR = ROOT / "data" / "processed"

ISARCASM_TEXT_COL = "text"
ISARCASM_LABEL_COL = "sarcastic"

EXPECTED_RAW = 28_619
EXPECTED_DEDUP = 28_503  # 28,619 - 116 extra copies (80 headlines that appear 2+ times)
EXPECTED_DUP_ROWS = 116
EXPECTED_DUP_UNIQUE = 80
EXPECTED_LENGTH_DROPS = 1

FIELDS = ["id", "text", "label", "domain"]


# ---------------------------------------------------------------- helpers
def domain_of(link: str) -> str:
    # Substring match on purpose: ~580 HuffPost links are malformed
    # ("huffingtonpost.comhttp:..."), so urlparse would mislabel them.
    if "theonion" in link:
        return "onion"
    if "huffingtonpost" in link:
        return "huffpost"
    return "other"


def normalise(text: str) -> str:
    """Key for duplicate detection only; never used as model input."""
    return " ".join(text.split())


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in df[FIELDS].itertuples(index=False):
            rec = {"id": row.id, "text": row.text, "label": int(row.label), "domain": row.domain}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_stats(df: pd.DataFrame) -> dict:
    n = len(df)
    n_sarc = int(df.label.sum())
    return {
        "size": n,
        "label_counts": {"0": n - n_sarc, "1": n_sarc},
        "sarcastic_proportion": round(n_sarc / n, 6),
        "domain_counts": {k: int(v) for k, v in sorted(df.domain.value_counts().items())},
    }


# ---------------------------------------------------------------- headlines
def prepare_headlines() -> tuple[dict[str, pd.DataFrame], dict]:
    raw = pd.read_json(RAW_HEADLINES, lines=True, dtype={"headline": str, "is_sarcastic": int})
    n_raw = len(raw)
    assert n_raw == EXPECTED_RAW, f"raw count {n_raw} != {EXPECTED_RAW}"

    df = pd.DataFrame({
        "text": raw.headline,
        "label": raw.is_sarcastic.astype(int),
        "domain": raw.article_link.map(domain_of),
        "raw_idx": range(n_raw),
    })
    df["norm"] = df.text.map(normalise)

    # 3. drop every copy of headlines whose copies disagree on the label
    n_labels = df.groupby("norm").label.transform("nunique")
    conflicting = df[n_labels > 1]
    n_conflict_rows = len(conflicting)
    n_conflict_unique = conflicting.norm.nunique()
    df = df[n_labels == 1]
    assert n_conflict_rows == 0, f"found {n_conflict_rows} conflicting-label rows"

    # 4. exact duplicates on normalised text, keep first (original file order)
    dup_mask = df.norm.duplicated(keep="first")
    n_dup_rows = int(dup_mask.sum())
    n_dup_unique = df[dup_mask].norm.nunique()
    df = df[~dup_mask]
    assert n_dup_rows == EXPECTED_DUP_ROWS, f"dup rows {n_dup_rows} != {EXPECTED_DUP_ROWS}"
    assert n_dup_unique == EXPECTED_DUP_UNIQUE, f"dup headlines {n_dup_unique} != {EXPECTED_DUP_UNIQUE}"
    assert len(df) == EXPECTED_DEDUP, f"dedup count {len(df)} != {EXPECTED_DEDUP}"

    # 5. length filter (rows have no id yet, so log their raw file line index)
    n_words = df.text.str.split().str.len()
    too_long = df[n_words > MAX_WORDS]
    for r, nw in zip(too_long.itertuples(), n_words[n_words > MAX_WORDS]):
        print(f"[length drop] raw_idx={r.raw_idx} words={nw} text={r.text[:80]!r}")
    n_len_drops = len(too_long)
    df = df[n_words <= MAX_WORDS].reset_index(drop=True)
    assert n_len_drops == EXPECTED_LENGTH_DROPS, f"length drops {n_len_drops} != {EXPECTED_LENGTH_DROPS}"

    # 6. stable ids in original file order, before any shuffling
    width = len(str(len(df) - 1))
    df["id"] = ["h" + str(i).zfill(width) for i in range(len(df))]

    # 7. stratified 80/10/10
    train, rest = train_test_split(df, test_size=0.2, stratify=df.label, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest.label, random_state=SEED)
    splits = {"train": train, "val": val, "test": test}

    assert sum(len(s) for s in splits.values()) == len(df)
    assert len(set(train.id) | set(val.id) | set(test.id)) == len(df), "id overlap across splits"

    stats = {
        "raw_count": n_raw,
        "conflicting_label_drops": {"rows": n_conflict_rows, "unique_headlines": n_conflict_unique},
        "duplicate_drops": {"rows": n_dup_rows, "unique_headlines": n_dup_unique},
        "length_drops": n_len_drops,
        "final_count": len(df),
        "splits": {name: split_stats(s) for name, s in splits.items()},
    }
    return splits, stats


# ---------------------------------------------------------------- iSarcasmEval
def prepare_isarcasm() -> tuple[pd.DataFrame, dict]:
    # keep_default_na=False so tweets like "NA" or "null" stay text, verbatim
    raw = pd.read_csv(RAW_ISARCASM, dtype=str, keep_default_na=False)
    print(f"iSarcasmEval columns: {raw.columns.tolist()}")
    missing = {ISARCASM_TEXT_COL, ISARCASM_LABEL_COL} - set(raw.columns)
    if missing:
        raise SystemExit(f"iSarcasmEval: expected columns {missing} not found; stopping.")

    labels = raw[ISARCASM_LABEL_COL].str.strip()
    bad = set(labels) - {"0", "1"}
    assert not bad, f"unexpected iSarcasmEval labels: {bad}"

    width = len(str(len(raw) - 1))
    df = pd.DataFrame({
        "id": ["t" + str(i).zfill(width) for i in range(len(raw))],
        "text": raw[ISARCASM_TEXT_COL],
        "label": labels.astype(int),
        "domain": "twitter",
    })

    p = df.label.mean()
    stats = {
        "size": len(df),
        "sarcastic_proportion": round(float(p), 6),
        "all_sarcastic_baseline_f1": round(float(2 * p / (1 + p)), 6),
    }
    return df, stats


# ---------------------------------------------------------------- main
def print_summary(stats: dict) -> None:
    h = stats["headlines"]
    print("\n================ prepare_data summary ================")
    print(f"seed={stats['seed']}  max_words={stats['max_words']}")
    print("\nHeadlines")
    print(f"  raw rows                 {h['raw_count']:>7,}")
    print(f"  conflicting-label drops  {h['conflicting_label_drops']['rows']:>7,}")
    print(f"  duplicate drops          {h['duplicate_drops']['rows']:>7,}  "
          f"({h['duplicate_drops']['unique_headlines']} unique headlines)")
    print(f"  length drops (>{stats['max_words']} words) {h['length_drops']:>4,}")
    print(f"  final rows               {h['final_count']:>7,}")
    print(f"\n  {'split':<6} {'size':>7} {'not sarc':>9} {'sarc':>7} {'% sarc':>7}  domains")
    for name, s in h["splits"].items():
        doms = ", ".join(f"{k}={v}" for k, v in s["domain_counts"].items())
        print(f"  {name:<6} {s['size']:>7,} {s['label_counts']['0']:>9,} {s['label_counts']['1']:>7,} "
              f"{100 * s['sarcastic_proportion']:>6.2f}%  {doms}")
    i = stats["isarcasm_test"]
    print("\niSarcasmEval Task A (En) test")
    print(f"  size                     {i['size']:>7,}")
    print(f"  sarcastic proportion p   {i['sarcastic_proportion']:>7.4f}")
    print(f"  all-sarcastic F1 2p/(1+p) {i['all_sarcastic_baseline_f1']:>6.4f}")
    print("\nSHA-256")
    for name, digest in stats["sha256"].items():
        print(f"  {name:<20} {digest}")
    print("======================================================")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    splits, head_stats = prepare_headlines()
    isarcasm, isa_stats = prepare_isarcasm()

    outputs = {f"{name}.jsonl": df for name, df in splits.items()}
    outputs["isarcasm_test.jsonl"] = isarcasm
    for fname, df in outputs.items():
        write_jsonl(df, OUT_DIR / fname)

    stats = {
        "seed": SEED,
        "max_words": MAX_WORDS,
        "headlines": head_stats,
        "isarcasm_test": isa_stats,
        "sha256": {fname: sha256(OUT_DIR / fname) for fname in outputs},
    }
    with open(OUT_DIR / "stats.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print_summary(stats)


if __name__ == "__main__":
    main()
