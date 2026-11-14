"""Add a residual-stream vector while the frozen model is generating.

The official baseline uses vLLM, which does not expose a forward hook.
Steering therefore runs under Hugging Face ``generate``. The weights are
not updated. Each hooked decoder layer computes:

    h = h + alpha * v

``v`` is expected to already be a unit vector (see ``vectors.diffmean``).
The same shift is applied at every token position of the forward pass,
which is the residual intervention written in CLaS-Bench.
"""

from __future__ import annotations

import torch

from clas_steer.collect import decoder_layers


def add_to_hidden(hidden: torch.Tensor, vector: torch.Tensor, alpha: float) -> torch.Tensor:
    """Broadcast ``alpha * vector`` onto a ``[batch, time, hidden]`` state."""
    shift = vector.to(device=hidden.device, dtype=hidden.dtype)
    return hidden + alpha * shift


def add_to_output(output, vector: torch.Tensor, alpha: float):
    """Shift a decoder-layer return value, whether it is a tensor or a tuple."""
    if isinstance(output, tuple):
        if not output or not torch.is_tensor(output[0]):
            raise TypeError(
                "decoder layer returned a tuple whose first item is not a tensor"
            )
        shifted = add_to_hidden(output[0], vector, alpha)
        return (shifted,) + output[1:]
    if not torch.is_tensor(output):
        raise TypeError(
            f"decoder layer returned {type(output).__name__}, expected a tensor"
        )
    return add_to_hidden(output, vector, alpha)


class SteeringHook:
    """Context manager that registers, and then removes, residual-stream hooks."""

    def __init__(
        self,
        model,
        vectors: dict[int, torch.Tensor],
        alpha: float,
    ) -> None:
        if not vectors:
            raise ValueError("SteeringHook needs at least one layer vector")
        self.model = model
        self.vectors = {int(layer): vector for layer, vector in vectors.items()}
        self.alpha = float(alpha)
        self._handles: list = []

    def __enter__(self) -> "SteeringHook":
        layers = decoder_layers(self.model)
        n_layers = len(layers)
        try:
            for index, vector in self.vectors.items():
                if index < 0 or index >= n_layers:
                    raise IndexError(
                        f"layer {index} is outside 0..{n_layers - 1} ({n_layers} layers)"
                    )
                handle = layers[index].register_forward_hook(self._hook_for(vector))
                self._handles.append(handle)
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _hook_for(self, vector: torch.Tensor):
        alpha = self.alpha

        def hook(module, inputs, output):
            del module, inputs
            return add_to_output(output, vector, alpha)

        return hook
