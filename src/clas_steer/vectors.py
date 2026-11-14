"""DiffMean language vectors on residual-stream means.

CLaS-Bench defines the direction at layer ℓ as the difference of mean
residual states, then L2-normalizes it before scaling by alpha:

    v = mean(h | target) - mean(h | contrast)
    h <- h + alpha * v / ||v||

``diffmean`` returns the unit vector, so the hook adds ``alpha * v``.
"""

from __future__ import annotations

from pathlib import Path

import torch


def diffmean(
    target_mean: torch.Tensor,
    other_mean: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Unit vector pointing from ``other_mean`` toward ``target_mean``.

    A zero difference (identical means) returns a zero vector. Adding it
    changes nothing, which is preferable to dividing by zero.
    """
    delta = target_mean.detach().float().cpu() - other_mean.detach().float().cpu()
    norm = torch.linalg.vector_norm(delta)
    if float(norm) < eps:
        return torch.zeros_like(delta)
    return delta / norm


def average_means(means: list[torch.Tensor]) -> torch.Tensor:
    """Unweighted mean of residual means, one vector per language."""
    if not means:
        raise ValueError("average_means received no vectors")
    stacked = torch.stack([item.detach().float().cpu() for item in means], dim=0)
    return stacked.mean(dim=0)


def save_vectors(path: Path, vectors: dict[int, torch.Tensor], meta: dict) -> None:
    """Write ``{layer: tensor}`` plus a JSON-like metadata dict to a ``.pt`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "vectors": {
            int(layer): vector.detach().float().cpu()
            for layer, vector in vectors.items()
        },
        "meta": meta,
    }
    torch.save(payload, path)


def load_vectors(path: Path) -> tuple[dict[int, torch.Tensor], dict]:
    """Load a file written by :func:`save_vectors`."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    vectors = {int(layer): tensor for layer, tensor in payload["vectors"].items()}
    return vectors, dict(payload.get("meta") or {})
