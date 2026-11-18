import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-2B"

# Goal 1 (done): load only
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
model.eval()

# Goal 2 (done): plain-string content, tail = '<|im_start|>assistant\n<think>\n\n</think>\n\n'
tok = AutoTokenizer.from_pretrained(MODEL_ID)
kw = dict(tokenize=False, add_generation_prompt=True)

SYS_TMPL = "You are a sarcasm classifier. Decide whether the text is sarcastic. Answer with exactly one word: {y} or {n}."
USER_TMPL = "Text: {text}\nIs this text sarcastic?"
EXAMPLES = [
    "new york introduces shoe-sharing program for city's pedestrians",
    "texas republicans urge trump-style immigration crackdown",
    "Really happy that the weather has stayed like this for the whole weekend,",
]
LABELS = ["yes", "Yes", "no", "No", " yes", " Yes", " no", " No"]

# Goals 3 + 4: top-5 next tokens and label probabilities
for y, n in [("Yes", "No"), ("yes", "no")]:
    SYS = SYS_TMPL.format(y=y, n=n)
    print(f"\n===== instruction says: {y} or {n} =====")
    for text in EXAMPLES:
        msgs = [{"role": "system", "content": SYS},
                {"role": "user",   "content": USER_TMPL.format(text=text)}]
        prompt = tok.apply_chat_template(msgs, enable_thinking=False, **kw)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            probs = model(**enc).logits[0, -1].float().softmax(-1)

        print("\n", text[:60])
        top = probs.topk(5)
        for p, i in zip(top.values.tolist(), top.indices.tolist()):
            print(f"  {i:>7} {tok.convert_ids_to_tokens(i)!r:12} {p:.4f}")
        for v in LABELS:
            ids = tok.encode(v, add_special_tokens=False)
            merged = len(tok(prompt + v)["input_ids"]) == len(enc["input_ids"][0]) + 1
            print(f"  {v!r:7} ids={ids} p={probs[ids[0]].item():.4f} single_after_prompt={merged}")