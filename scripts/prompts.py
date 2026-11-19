"""Single source of truth for the prompt wording seen by the model."""

from transformers import PreTrainedTokenizerBase

SYSTEM_MSG = (
    "You are a sarcasm classifier. Decide whether the text is sarcastic. "
    "Answer with exactly one word: Yes or No."
)
USER_TMPL = "Text: {text}\nIs this text sarcastic?"
LABEL_WORDS = {1: "Yes", 0: "No"}


def build_messages(text: str, demos: list[tuple[str, int]] | None = None) -> list[dict[str, str]]:
    """Return chat messages: system, then (user, assistant) per demo, then the user turn for `text`."""
    messages = [{"role": "system", "content": SYSTEM_MSG}]
    for demo_text, demo_label in demos or []:
        messages.append({"role": "user", "content": USER_TMPL.format(text=demo_text)})
        messages.append({"role": "assistant", "content": LABEL_WORDS[demo_label]})
    messages.append({"role": "user", "content": USER_TMPL.format(text=text)})
    return messages


def render(tok: PreTrainedTokenizerBase, messages: list[dict[str, str]]) -> str:
    """Render messages to the exact prompt string, ending at the assistant generation slot."""
    ...
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )