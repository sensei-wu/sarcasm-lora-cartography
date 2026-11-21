from prompts import LABEL_WORDS

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-2B"


def label_token_ids(tok):
    ids = {}
    for word in LABEL_WORDS.values():
        pieces = tok.encode(word, add_special_tokens=False)
        assert len(pieces) == 1, f"{word!r} is {len(pieces)} tokens: {pieces}"
        ids[word] = pieces[0]
    return ids

def check_label_append(tok, prompt, ids):
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    for word, wid in ids.items():
        full = tok(prompt + word, add_special_tokens=False)["input_ids"]
        assert len(full) == len(p_ids) + 1, (word, len(p_ids), len(full))
        assert full[:-1] == p_ids, f"{word!r} merged with the end of the prompt"
        assert full[-1] == wid, (word, full[-1], wid)
    return len(p_ids)

MODEL_ID = "Qwen/Qwen3.5-2B"

def load_model(model_id=MODEL_ID, adapter_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    assert tok.pad_token is not None, "tokenizer has no pad token"

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device)
    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    ids = label_token_ids(tok)
    return tok, model, ids