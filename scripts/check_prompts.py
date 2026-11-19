from transformers import AutoTokenizer
from prompts import build_messages, render   # adjust import path to your layout

TAIL = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")

zero = render(tok, build_messages("x"))
assert zero.endswith(TAIL), repr(zero[-60:])
print(repr(zero))

few = render(tok, build_messages("x", [("a", 1), ("b", 0)]))
assert few.endswith(TAIL), repr(few[-60:])
print(few)