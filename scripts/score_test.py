import time
import torch
from modeling import load_model
from scoring import score_batch, load_rows

rows = load_rows("/workspace/data/processed/test.jsonl")[:64]

tok, model, ids = load_model()
rows = load_rows("/workspace/data/processed/test.jsonl")[:64]

for _ in range(2):
    t = time.perf_counter()
    res = score_batch(tok, model, ids, rows, split="test")
    print(round(time.perf_counter() - t, 2), "s")
print(res[0])

from scoring import score_one

rows20 = rows[:20]
batched = score_batch(tok, model, ids, rows20, batch_size=20)
diffs = []
for (rid, text, gold), b in zip(rows20, batched):
    p_yes, _, _ = score_one(tok, model, ids, text)
    diffs.append(abs(p_yes - b["p_yes"]))
print("max", max(diffs), "mean", sum(diffs) / len(diffs))
print("same preds:", all((score_one(tok, model, ids, t)[0] > 0.5) == bool(b["pred"])
                         for (_, t, _), b in zip(rows20, batched)))

from prompts import build_messages, render

lens = [len(tok(render(tok, build_messages(t)), add_special_tokens=False).input_ids)
        for _, t, _ in rows20]
pads = [max(lens) - n for n in lens]
for p, d in sorted(zip(pads, diffs)):
    print(p, round(d, 4))