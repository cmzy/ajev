"""Training objectives. All take padded logits [B, K] with -inf in padded slots."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def smooth(target: torch.Tensor, option_mask: torch.Tensor, eps: float) -> torch.Tensor:
    """Label smoothing restricted to real options: (1 - eps) * t + eps / K."""
    k = option_mask.sum(-1, keepdim=True).clamp(min=1)
    return ((1 - eps) * target + eps / k) * option_mask


def soft_ce(logits: torch.Tensor, target: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against a soft target distribution (per decision)."""
    logp = F.log_softmax(logits, dim=-1).masked_fill(~option_mask, 0.0)
    return -(target * logp).sum(-1)


def rps(logits: torch.Tensor, target: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
    """Ranked probability score for ordinal (score) questions, normalised by K-1."""
    p = F.softmax(logits, dim=-1).masked_fill(~option_mask, 0.0)
    diff = (p.cumsum(-1) - target.cumsum(-1)) * option_mask
    k = option_mask.sum(-1).clamp(min=2)
    return (diff**2).sum(-1) / (k - 1)


def symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
    """Symmetric KL between two views' distributions, already aligned to the same option order."""
    la = F.log_softmax(logits_a, dim=-1).masked_fill(~option_mask, 0.0)
    lb = F.log_softmax(logits_b, dim=-1).masked_fill(~option_mask, 0.0)
    pa, pb = la.exp() * option_mask, lb.exp() * option_mask
    return 0.5 * ((pa * (la - lb)).sum(-1) + (pb * (lb - la)).sum(-1))


def unpermute(logits: torch.Tensor, perm: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
    """Map view logits back to canonical option order: out[:, perm[j]] = logits[:, j]."""
    out = torch.full_like(logits, float("-inf"))
    src = logits.masked_fill(~option_mask, float("-inf"))
    return out.scatter(1, perm, src)
