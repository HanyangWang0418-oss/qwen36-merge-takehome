"""Render a chat example with the base model's own chat template and mark
assistant tokens for the loss mask. Thinking traces (if any) stay inside the
assistant turn as `<think>...</think>` so the model learns to produce them."""
from __future__ import annotations


def render_example(tokenizer, messages: list[dict], assistant_only: bool = True) -> tuple[list[int], list[float]]:
    """Returns (token_ids, weights). weights[i] == 1.0 where token i is part of
    an assistant turn (including the EOS/turn-end token), else 0.0.

    Implementation: render the prefix up to each assistant turn with
    add_generation_prompt=True, render the prefix including the turn, and mark
    the difference. This is template-agnostic and exact for Qwen templates.
    """
    tokens: list[int] = []
    weights: list[float] = []
    prev_len = 0
    for i, m in enumerate(messages):
        prefix = messages[: i + 1]
        if m["role"] == "assistant":
            gen = tokenizer.apply_chat_template(messages[:i], add_generation_prompt=True, tokenize=True)
            full = tokenizer.apply_chat_template(prefix, add_generation_prompt=False, tokenize=True)
            assert full[: len(gen)] == gen, "template prefix mismatch"
            tokens = full
            weights = weights[: len(gen)] + [0.0] * (len(gen) - len(weights)) + [1.0] * (len(full) - len(gen))
        else:
            full = tokenizer.apply_chat_template(prefix, add_generation_prompt=False, tokenize=True)
            tokens = full
            weights = weights + [0.0] * (len(full) - len(weights))
        prev_len = len(tokens)
    if not assistant_only:
        weights = [1.0] * len(tokens)
    assert len(tokens) == len(weights) == prev_len
    return tokens, weights
