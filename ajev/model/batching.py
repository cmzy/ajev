"""Padding / collation of encoded decisions into model tensors."""

from __future__ import annotations

import torch

from ajev.model.encoding import Encoded


def collate(encs: list[Encoded], pad_id: int) -> dict[str, torch.Tensor]:
    b = len(encs)
    t = max(len(e.input_ids) for e in encs)
    k = max(len(e.marker_pos) for e in encs)
    input_ids = torch.full((b, t), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((b, t), dtype=torch.long)
    marker_pos = torch.zeros((b, k), dtype=torch.long)
    option_mask = torch.zeros((b, k), dtype=torch.bool)
    for i, e in enumerate(encs):
        input_ids[i, : len(e.input_ids)] = torch.tensor(e.input_ids)
        attention_mask[i, : len(e.input_ids)] = 1
        marker_pos[i, : len(e.marker_pos)] = torch.tensor(e.marker_pos)
        option_mask[i, : len(e.marker_pos)] = True
    return {"input_ids": input_ids, "attention_mask": attention_mask, "marker_pos": marker_pos,
            "option_mask": option_mask}


def pad_targets(targets: list[list[float]], k: int) -> torch.Tensor:
    out = torch.zeros((len(targets), k))
    for i, t in enumerate(targets):
        out[i, : len(t)] = torch.tensor(t)
    return out


def length_sorted_batches(lengths: list[int], batch_size: int) -> list[list[int]]:
    """Deterministic inference batching: similar lengths together to minimise padding."""
    order = sorted(range(len(lengths)), key=lengths.__getitem__)
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
