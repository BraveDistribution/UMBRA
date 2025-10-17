"""Utility functions for metrics."""

from __future__ import annotations

__all__ = [
    "effective_rank",
]

from typing import Literal

import torch

def effective_rank(
    features: torch.Tensor,
    eps: float = 1e-6,
    method: Literal["auto", "svd", "cov"] = "auto",
    subsample: int | None = None # e.g., 8192
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Effective rank via singular values. Safe for large queues.

    Returns:
      eff_r (scalar tensor), entropy (scalar tensor)
    """
    if features.ndim != 2:
        raise ValueError(f"Expected (N, D) got {features.shape}")
    N, D = features.shape
    if N < 1:
        z = torch.zeros((), device=features.device)
        return z, z

    with torch.no_grad():
        X = features.detach().float()  # keep it fast; switch to .double() if you truly need it
        if subsample is not None and N > subsample:
            idx = torch.randperm(N, device=X.device)[:subsample]
            X = X.index_select(0, idx)
            N = X.size(0)

        # center
        X = X - X.mean(dim=0, keepdim=True)

        # choose method
        if method == "auto":
            method = "cov" if N > 4 * D else "svd"

        if method == "cov":
            # D x D; cheaper when N >> D
            cov = (X.T @ X) / max(N - 1, 1)
            evals = torch.linalg.eigvalsh(cov).clamp_min(0)
            svals = torch.sqrt(evals.clamp_min(0))  # singular values
        elif method == "svd":
            svals = torch.linalg.svdvals(X)
        else:
            raise ValueError("method must be 'auto', 'svd', or 'cov'")

        # handle near-zero spectrum
        total = svals.sum()
        if total <= eps:
            # treat as collapsed
            zero = torch.tensor(0.0, device=X.device)
            return zero, zero

        s = svals.clamp_min(eps)
        p = s / s.sum()
        H = -(p * (p + eps).log()).sum()
        er = torch.exp(H)

    return er, H