from prompts import LABEL_WORDS

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