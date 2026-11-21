import torch
from prompts import build_messages, render

@torch.no_grad()
def score_one(tok, model, ids, text):
    prompt = render(tok, build_messages(text))
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    logits = model(**enc).logits[0, -1].float()
    pair = logits[[ids["Yes"], ids["No"]]]
    p_yes, p_no = torch.softmax(pair, dim=-1).tolist()
    full = torch.softmax(logits, dim=-1)
    mass = (full[ids["Yes"]] + full[ids["No"]]).item()
    return p_yes, p_no, mass

@torch.no_grad()
def score_batch(tok, model, ids, rows, batch_size=32, split=None):
    out = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        prompts = [render(tok, build_messages(text)) for _, text, _ in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        logits = model(**enc).logits[:, -1].float()
        pair = torch.softmax(logits[:, [ids["Yes"], ids["No"]]], dim=-1)
        full = torch.softmax(logits, dim=-1)
        mass = full[:, ids["Yes"]] + full[:, ids["No"]]
        for (rid, _, gold), (p_yes, p_no), m in zip(chunk, pair.tolist(), mass.tolist()):
            out.append({"split": split, "id": rid, "gold": gold,
                        "p_yes": p_yes, "p_no": p_no,
                        "p_gold": p_yes if gold == 1 else p_no,
                        "pred": int(p_yes > p_no), "mass": m})
    return out