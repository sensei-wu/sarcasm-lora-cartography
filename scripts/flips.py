from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from cartography import load_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("base_dir")   # e.g. results/zeroshot
ap.add_argument("new_dir")    # e.g. results/random10_seed1
args = ap.parse_args()

base = load_checkpoint(Path(args.base_dir) / "isarcasm_test_preds.jsonl")
new = load_checkpoint(Path(args.new_dir) / "isarcasm_test_preds.jsonl")

assert set(base.index) == set(new.index), "different tweet ids"
df = base[["gold", "pred", "p_yes"]].join(new[["gold", "pred", "p_yes"]],
                                          suffixes=("_base", "_new"))
assert (df["gold_base"] == df["gold_new"]).all(), "gold labels differ"
df = df.drop(columns="gold_new").rename(columns={"gold_base": "gold"})

right_base = (df["pred_base"] == df["gold"]).map({True: "right", False: "wrong"})
right_new = (df["pred_new"] == df["gold"]).map({True: "right", False: "wrong"})
df["flip"] = right_base + "->" + right_new
df["d_p_yes"] = df["p_yes_new"] - df["p_yes_base"]

import re
texts = pd.read_json("data/processed/isarcasm_test.jsonl", lines=True).set_index("id")["text"]
assert set(texts.index) == set(df.index), "text ids don't match"
df["text"] = texts

EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
df["n_words"] = df["text"].str.split().str.len()
df["short"] = df["n_words"] <= df["n_words"].median()
df["hashtag"] = df["text"].str.contains("#")
df["mention"] = df["text"].str.contains("@")
df["url"] = df["text"].str.contains("http")
df["emoji"] = df["text"].str.contains(EMOJI)
df["lowercase"] = df["text"] == df["text"].str.lower()

print(pd.crosstab(df["flip"], df["gold"], margins=True))

FEATS = ["short", "emoji", "lowercase"]
# (move this line up to replace the old FEATS)

out = Path(args.new_dir).name + "_vs_" + Path(args.base_dir).name + "_flips.csv"
cols = ["gold", "flip", "p_yes_base", "p_yes_new", "d_p_yes", "n_words", "text"]
flips = df[df["flip"].isin(["right->wrong", "wrong->right"])]
flips.sort_values("d_p_yes")[cols].to_csv(out)
print(f"wrote {len(flips)} rows to {out}")
print(df.groupby(["gold", "flip"])[FEATS].mean().round(2))