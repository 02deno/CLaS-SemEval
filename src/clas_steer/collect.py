"""Mean residual-stream states on FLORES sentences.

The collected position is the last real token of each sentence: the hidden
state after the model has read the whole line. Layer indices are 0-based
decoder layers. Hugging Face stores the embedding output at
``hidden_states[0]``, so layer ℓ is ``hidden_states[ℓ + 1]``.
"""

from __future__ import annotations

import torch


def last_real_index(attention_mask: torch.Tensor) -> torch.Tensor:
    """Index of the last non-padding token in each row.

    Works for both left and right padding. ``attention_mask`` uses 1 for
    real tokens and 0 for padding.
    """
    flipped = attention_mask.flip(dims=[1]).to(dtype=torch.long)
    offset = flipped.argmax(dim=1)
    return attention_mask.size(1) - 1 - offset


def decoder_layers(model) -> torch.nn.ModuleList:
    """Return the decoder block list for Llama-style and GPT-2-style models."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError(
        "Cannot find decoder layers. Expected model.model.layers "
        "(Llama, Aya, Gemma) or model.transformer.h (GPT-2)."
    )


def _embed_device(model) -> torch.device:
    embeddings = model.get_input_embeddings()
    return embeddings.weight.device


@torch.inference_mode()
def collect_residual_means(
    model,
    tokenizer,
    texts: list[str],
    layers: list[int],
    batch_size: int = 8,
    max_length: int = 512,
) -> dict[int, torch.Tensor]:
    """Average the residual stream at the last real token.

    Returns CPU float tensors, one vector per requested layer. ``layers``
    are 0-based indices into the decoder stack.
    """
    if not texts:
        raise ValueError("collect_residual_means received no texts")
    if not layers:
        raise ValueError("collect_residual_means received no layers")

    n_layers = len(decoder_layers(model))
    unknown = [layer for layer in layers if layer < 0 or layer >= n_layers]
    if unknown:
        raise IndexError(
            f"layers {unknown} are outside 0..{n_layers - 1} ({n_layers} layers)"
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = _embed_device(model)
    sums = {layer: None for layer in layers}
    seen = 0
    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded, output_hidden_states=True, use_cache=False)
            hidden_states = output.hidden_states
            if hidden_states is None:
                raise RuntimeError(
                    "model did not return hidden states; "
                    "output_hidden_states=True was ignored"
                )
            index = last_real_index(encoded["attention_mask"])
            batch_index = torch.arange(index.size(0), device=index.device)
            for layer in layers:
                states = hidden_states[layer + 1]
                picked = states[batch_index, index].float().cpu()
                sums[layer] = (
                    picked.sum(dim=0)
                    if sums[layer] is None
                    else sums[layer] + picked.sum(dim=0)
                )
            seen += index.size(0)
    finally:
        tokenizer.padding_side = previous_side
        model.train(was_training)

    return {layer: sums[layer] / seen for layer in layers}
