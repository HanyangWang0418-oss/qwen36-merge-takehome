"""Render a chat example with the base model's own chat template and mark
assistant tokens for the loss mask.

The conversation is rendered once with the template; assistant spans are then
located structurally in the rendered text (ChatML: from the end of
"<|im_start|>assistant\\n" to the end of the following "<|im_end|>\\n") and mapped
onto tokens with the fast tokenizer's offset mapping. Working on text offsets
is exact even when a token straddles a boundary (e.g. "<think>\\n" merging with
the first word of the reasoning), which breaks naive token-prefix matching.

Qwen3.6's template strips <think>...</think> from every assistant turn except
the last unless preserve_thinking=True is passed; for multi-turn agent data we
pass it so the model is trained on its reasoning in every turn (this matches
the preserve_thinking=True serving setting used in evaluation)."""
from __future__ import annotations

ASSISTANT_HEADER = "<|im_start|>assistant\n"
TURN_END = "<|im_end|>\n"


def assistant_spans(text: str) -> list[tuple[int, int]]:
    spans, pos = [], 0
    while True:
        h = text.find(ASSISTANT_HEADER, pos)
        if h < 0:
            break
        s = h + len(ASSISTANT_HEADER)
        e = text.find(TURN_END, s)
        e = len(text) if e < 0 else e + len(TURN_END)
        spans.append((s, e))
        pos = e
    return spans


def render_example(tokenizer, messages: list[dict], assistant_only: bool = True,
                   template_kwargs: dict | None = None) -> tuple[list[int], list[float]]:
    """Returns (token_ids, weights). weights[i] == 1.0 where token i lies inside an assistant turn."""
    kw = {"preserve_thinking": True, **(template_kwargs or {})}
    full = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False, **kw)
    spans = assistant_spans(full)
    n_assistant = sum(1 for m in messages if m["role"] == "assistant")
    if len(spans) != n_assistant:
        raise ValueError(f"found {len(spans)} assistant spans for {n_assistant} assistant messages")
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    weights = [0.0] * len(ids)
    for ti, (cs, ce) in enumerate(enc["offset_mapping"]):
        if ce > cs and any(cs >= s and ce <= e for s, e in spans):
            weights[ti] = 1.0
    if not assistant_only:
        weights = [1.0] * len(ids)
    return ids, weights


if __name__ == "__main__":  # inspect what the template keeps across turns
    import json
    from transformers import AutoTokenizer

    rev = json.load(open("data/manifests/base_model.json"))["hf_revision"]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-35B-A3B", revision=rev)
    msgs = [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "<think>\nr1\n</think>\n\nA1"},
            {"role": "user", "content": "Q2"}, {"role": "assistant", "content": "<think>\nr2\n</think>\n\nA2"}]
    for kw in ({"preserve_thinking": False}, {"preserve_thinking": True}):
        print(kw, repr(tok.apply_chat_template(msgs, tokenize=False, **kw)))
        ids, w = render_example(tok, msgs, template_kwargs=kw)
        print("  n_tokens", len(ids), "loss tokens:", repr(tok.decode([i for i, wi in zip(ids, w) if wi])))
    # round-trip check: token ids from offsets-tokenisation equal the template's own tokenisation
    ids_text = tok(tok.apply_chat_template(msgs, tokenize=False, preserve_thinking=True), add_special_tokens=False)["input_ids"]
    ids_tmpl = tok.apply_chat_template(msgs, tokenize=True, preserve_thinking=True)
    ids_tmpl = ids_tmpl["input_ids"] if not isinstance(ids_tmpl, list) else ids_tmpl
    assert ids_text == ids_tmpl, "tokenisation mismatch"
    print("round-trip ok")
