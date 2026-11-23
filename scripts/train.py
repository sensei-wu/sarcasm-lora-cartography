"""LoRA trainer for the sarcasm classifier. One script covers every training box in the diagram:

    phase1_map   full train split, adapter saved after each epoch (data map scored afterwards)
    random10     random 10% of train (sanity check that LoRA beats the baselines at all)
    phase2       data-map subset (easy / ambiguous / hard) + random control, 2-3 seeds

Usage:  python scripts/train.py configs/<run>.yaml [--limit N]

The trainer only trains and saves. Test-set metrics come from the shared evaluator
(scripts/evaluate.py) pointed at the saved adapter, so every run is scored identically.

Outputs under {adapters_root}/{run_name}/ (separate from evaluate.py's results_root, so the
two scripts never claim the same folder):
    config.yaml            copy of the config
    epoch{N}/              LoRA adapter after each epoch (PeftModel.from_pretrained-able)
    train_log.jsonl        one row per epoch: train loss, val acc
    selected_ids.json     ids actually trained on (for audit)
"""

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch
import yaml
from transformers import get_linear_schedule_with_warmup

from modeling import load_model
from prompts import build_messages, render
from scoring import load_rows, score_batch

# Config keys (see configs/train_random10.yaml):
#   run_name, model_id, adapters_root, batch_size
#   train: path to train.jsonl, val: path to val.jsonl
#   seed, epochs, lr, train_batch_size
#   lora: {r, alpha, dropout, target_modules}
#   subset: {kind: all|random|easy|ambiguous|hard, fraction, map_path}


# ---------------------------------------------------------------- reproducibility
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- data selection
def select_subset(rows, subset_cfg, seed):
    """Return the rows to train on.

    kind=all        every row (phase 1)
    kind=random     `fraction` of rows, sampled with `seed` (random10 and phase 2 control)
    kind=ids        exactly the ids listed in `ids_path` (one per line; phase 2 map subsets)
    """
    kind = subset_cfg["kind"]
    if kind == "all":
        return rows
    if kind == "random":
        rng = random.Random(seed)
        k = round(len(rows) * subset_cfg["fraction"])
        return rng.sample(rows, k)
    if kind == "ids":
        wanted = set(Path(subset_cfg["ids_path"]).read_text().split())
        chosen = [r for r in rows if r[0] in wanted]
        assert len(chosen) == len(wanted), f"{len(wanted) - len(chosen)} ids not found in train"
        return chosen
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
    targets = torch.tensor([ids["Yes"] if gold == 1 else ids["No"] for _, _, gold in chunk])
    return enc, targets


def label_loss(logits_last, targets):
    """Full-vocabulary cross-entropy at the label slot (decided 22 Nov); float32 for numerical stability."""
    return torch.nn.functional.cross_entropy(logits_last.float(), targets)


# ---------------------------------------------------------------- training loop
def train_epoch(tok, model, ids, rows, optimizer, scheduler, trainable, cfg, rng, epoch, out_dir):  # NEW: out_dir
    model.train()
    rows = rows[:]
    rng.shuffle(rows)
    bs = cfg["train_batch_size"]
    n_steps = math.ceil(len(rows) / bs)                                   # NEW
    k = cfg.get("saves_per_epoch", 1)                                     # NEW
    save_at = {round(i * n_steps / k): i for i in range(1, k)}            # NEW
    total, steps, t0 = 0.0, 0, time.time()
    for start in range(0, len(rows), bs):
        enc, targets = make_batch(tok, ids, rows[start:start + bs])
        enc, targets = enc.to(model.device), targets.to(model.device)
        logits = model(**enc).logits[:, -1]
        loss = label_loss(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item(); steps += 1
        if steps in save_at:                                              # NEW
            model.save_pretrained(out_dir / f"epoch{epoch}_q{save_at[steps]}")  # NEW
        if steps % 10 == 0:
            print(f"epoch {epoch} step {steps}/{n_steps} "
                  f"loss {loss.item():.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                  f"{(time.time()-t0)/steps:.2f}s/step", flush=True)
    return total / max(steps, 1)


@torch.no_grad()
def eval_split(tok, model, ids, rows, batch_size):
    """Quick val accuracy for the training log (the real evaluation is evaluate.py)."""
    model.eval()
    res = score_batch(tok, model, ids, rows, batch_size=batch_size)
    return sum(r["pred"] == r["gold"] for r in res) / len(res)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--limit", type=int, default=None, help="smoke test on the first N rows")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_name = cfg["run_name"] + (f"_limit{args.limit}" if args.limit else "")
    out_dir = Path(cfg["adapters_root"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy(args.config, out_dir / "config.yaml")

    set_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])

    train_all = load_rows(cfg["train"])
    val_rows = load_rows(cfg["val"])

    # Selection is applied to the full train split; the mapping run uses kind=all,
    # so its ids cover the whole split. --limit (smoke test) cuts the selected
    # list afterwards, so the subset draw is not shrunk by it.
    train_rows = select_subset(train_all, cfg["subset"], cfg["seed"])
    if args.limit:
        train_rows, val_rows = train_rows[:args.limit], val_rows[:args.limit]
    (out_dir / "selected_ids.json").write_text(json.dumps([r[0] for r in train_rows]))
    print(f"training on {len(train_rows)} / {len(train_all)} rows")

    tok, model, ids = load_model(cfg["model_id"])
    model = add_lora(model, cfg["lora"])
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"])
    steps_per_epoch = math.ceil(len(train_rows) / cfg["train_batch_size"])
    total_steps = steps_per_epoch * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=round(0.05 * total_steps), num_training_steps=total_steps)

    with open(out_dir / "train_log.jsonl", "w") as log_f:
        for epoch in range(1, cfg["epochs"] + 1):
            train_loss = train_epoch(tok, model, ids, train_rows, optimizer, scheduler,
                                     trainable, cfg, rng, epoch, out_dir)
            val_acc = eval_split(tok, model, ids, val_rows, cfg["batch_size"])
            row = {"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc}
            log_f.write(json.dumps(row) + "\n")
            log_f.flush()
            print(row)
            model.save_pretrained(out_dir / f"epoch{epoch}")
    print(f"adapters saved under {out_dir}")


if __name__ == "__main__":
    main()
