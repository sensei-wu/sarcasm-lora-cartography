"""LoRA trainer for the sarcasm classifier. One script covers every training box in the diagram:

    phase1_map   full train split, log P(gold) on every train example after each epoch
    gonogo       random 10% of train (sanity check that LoRA beats the baselines at all)
    phase2       data-map subset (easy / ambiguous / hard) + random control, 2-3 seeds

Usage:  python scripts/train.py configs/<run>.yaml [--limit N]

The trainer only trains and saves. Test-set metrics come from the shared evaluator
(scripts/evaluate.py) pointed at the saved adapter, so every run is scored identically.

Outputs under {results_root}/{run_name}/:
    config.yaml            copy of the config
    adapter/               final LoRA adapter (PeftModel.from_pretrained-able)
    train_log.jsonl        one row per epoch: train loss, val acc
    map_log.jsonl          phase1 only: {"epoch", "id", "gold", "p_gold"} per train example
    selected_ids.json      ids actually trained on (for audit)
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import torch
import yaml

from modeling import load_model
from prompts import build_messages, render
from scoring import load_rows, score_batch

# Config keys (see configs/train_gonogo.yaml):
#   run_name, model_id, results_root, batch_size
#   train: path to train.jsonl, val: path to val.jsonl
#   seed, epochs, lr, train_batch_size, grad_accum
#   lora: {r, alpha, dropout, target_modules}
#   subset: {kind: all|random|easy|ambiguous|hard, fraction, map_path, plus_random}
#   log_map: bool   (phase 1 only)


# ---------------------------------------------------------------- reproducibility
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- data selection
def select_subset(rows, subset_cfg, seed):
    """Return the rows to train on.

    kind=all        every row (phase 1)
    kind=random     `fraction` of rows, sampled with `seed` (go/no-go and phase 2 control)
    kind=easy|ambiguous|hard
                    ids from the data map at `map_path`, plus `plus_random` extra random
                    rows from outside that subset (diagram: "+ random")
    """
    kind = subset_cfg["kind"]
    if kind == "all":
        return rows
    if kind == "random":
        rng = random.Random(seed)
        k = round(len(rows) * subset_cfg["fraction"])
        return rng.sample(rows, k)
    # TODO: load map_path (output of the cartography analysis step), take the ids for
    # `kind`, then add `plus_random` rows drawn with Random(seed) from the remainder.
    raise NotImplementedError(f"subset kind {kind!r}")


# ---------------------------------------------------------------- model
def add_lora(model, lora_cfg):
    """Wrap the base model with a trainable LoRA adapter."""
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------- batching and loss
def make_batch(tok, ids, chunk):
    """Tokenise a list of (id, text, gold) rows exactly as scoring.score_batch does
    (left padding, so the label slot is always position -1) and return the gold
    label-token ids as targets."""
    prompts = [render(tok, build_messages(text)) for _, text, _ in chunk]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    # TODO: map gold (1/0) -> ids["Yes"] / ids["No"]
    targets = torch.tensor([ids["Yes"] if gold == 1 else ids["No"] for _, _, gold in chunk])
    return enc, targets


def label_loss(logits_last, targets, ids):
    """Cross-entropy at the label slot.

    TODO: decide between full-vocab CE and CE restricted to the [Yes, No] pair. The
    evaluator reads the 2-way softmax, so restricting matches it; full-vocab also
    trains the model to put mass on the label tokens (the `mass` metric).
    """
    return torch.nn.functional.cross_entropy(logits_last.float(), targets)


# ---------------------------------------------------------------- training loop
def train_epoch(tok, model, ids, rows, optimizer, cfg, rng):
    model.train()
    rows = rows[:]
    rng.shuffle(rows)
    bs, accum = cfg["train_batch_size"], cfg["grad_accum"]
    total, steps = 0.0, 0
    optimizer.zero_grad()
    for i, start in enumerate(range(0, len(rows), bs)):
        enc, targets = make_batch(tok, ids, rows[start:start + bs])
        enc, targets = enc.to(model.device), targets.to(model.device)
        logits = model(**enc).logits[:, -1]
        loss = label_loss(logits, targets, ids) / accum
        loss.backward()
        total += loss.item() * accum
        steps += 1
        if (i + 1) % accum == 0:
            optimizer.step()
            optimizer.zero_grad()
    # TODO: flush the last partial accumulation; add LR schedule / grad clipping if wanted
    return total / max(steps, 1)


@torch.no_grad()
def eval_split(tok, model, ids, rows, batch_size):
    """Quick val accuracy for the training log (the real evaluation is evaluate.py)."""
    model.eval()
    res = score_batch(tok, model, ids, rows, batch_size=batch_size)
    return sum(r["pred"] == r["gold"] for r in res) / len(res)


@torch.no_grad()
def log_map_epoch(tok, model, ids, rows, batch_size, epoch, f):
    """Phase 1: P(gold) for every train example, written as one jsonl row each."""
    model.eval()
    res = score_batch(tok, model, ids, rows, batch_size=batch_size)
    for r in res:
        f.write(json.dumps({"epoch": epoch, "id": r["id"], "gold": r["gold"],
                            "p_gold": r["p_gold"]}) + "\n")
    f.flush()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--limit", type=int, default=None, help="smoke test on the first N rows")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_name = cfg["run_name"] + (f"_limit{args.limit}" if args.limit else "")
    out_dir = Path(cfg["results_root"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy(args.config, out_dir / "config.yaml")

    set_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])

    train_all = load_rows(cfg["train"])
    val_rows = load_rows(cfg["val"])
    if args.limit:
        train_all, val_rows = train_all[:args.limit], val_rows[:args.limit]

    # Selection is applied to the full train split; the map log (phase 1) always uses
    # kind=all, so its ids cover the whole split.
    train_rows = select_subset(train_all, cfg["subset"], cfg["seed"])
    (out_dir / "selected_ids.json").write_text(json.dumps([r[0] for r in train_rows]))
    print(f"training on {len(train_rows)} / {len(train_all)} rows")

    tok, model, ids = load_model(cfg["model_id"])
    model = add_lora(model, cfg["lora"])
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg["lr"]
    )

    map_f = open(out_dir / "map_log.jsonl", "w") if cfg.get("log_map") else None
    with open(out_dir / "train_log.jsonl", "w") as log_f:
        for epoch in range(1, cfg["epochs"] + 1):
            train_loss = train_epoch(tok, model, ids, train_rows, optimizer, cfg, rng)
            val_acc = eval_split(tok, model, ids, val_rows, cfg["batch_size"])
            row = {"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc}
            log_f.write(json.dumps(row) + "\n")
            log_f.flush()
            print(row)
            if map_f:
                log_map_epoch(tok, model, ids, train_rows, cfg["batch_size"], epoch, map_f)
    if map_f:
        map_f.close()

    model.save_pretrained(out_dir / "adapter")
    print(f"saved adapter to {out_dir / 'adapter'}")
    print("evaluate with: set `adapter` in an evaluate config to this path "
          "and run scripts/evaluate.py")


if __name__ == "__main__":
    main()
