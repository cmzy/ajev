"""Encoder decision model: a bidirectional backbone plus an option-marker scoring head.

Every option gets a ``<mask>`` marker in the input; the head maps each marker's hidden
state (together with the sequence's ``<bos>`` summary) to one logit, and a softmax over
the options of a decision gives its distribution. Option counts vary per decision, so
logits are padded to the batch maximum and masked.
"""

from __future__ import annotations

import json
import os

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

HEAD_FILE = "decision_head.pt"
AJEV_CONFIG = "ajev_config.json"


class DecisionHead(nn.Module):
    def __init__(self, hidden: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, opt: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # opt: [B, K, H] marker states, ctx: [B, H] <bos> state
        ctx = ctx.unsqueeze(1).expand_as(opt)
        return self.net(torch.cat([opt, ctx, opt * ctx], dim=-1)).squeeze(-1)


class DecisionModel(nn.Module):
    def __init__(self, backbone: nn.Module, head_dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = DecisionHead(backbone.config.hidden_size, head_dropout)

    def forward(self, input_ids, attention_mask, marker_pos, option_mask) -> torch.Tensor:
        """Returns logits [B, K]; padded option slots are -inf."""
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        idx = marker_pos.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        opt = torch.gather(hidden, 1, idx)
        logits = self.head(opt, hidden[:, 0]).float()
        return logits.masked_fill(~option_mask, float("-inf"))

    # ---- persistence ------------------------------------------------------------
    def save(self, path: str, tokenizer=None, extra: dict | None = None) -> None:
        os.makedirs(path, exist_ok=True)
        self.backbone.save_pretrained(path)
        torch.save(self.head.state_dict(), os.path.join(path, HEAD_FILE))
        if tokenizer is not None:
            tokenizer.save_pretrained(path)
        with open(os.path.join(path, AJEV_CONFIG), "w") as f:
            json.dump(extra or {}, f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str, **backbone_kw) -> "DecisionModel":
        """Load a saved AJev checkpoint, or start fresh from a HF backbone id."""
        model = cls(AutoModel.from_pretrained(path, **backbone_kw))
        head = os.path.join(path, HEAD_FILE)
        if os.path.exists(head):
            model.head.load_state_dict(torch.load(head, map_location="cpu"))
        return model


def load_tokenizer(path: str):
    return AutoTokenizer.from_pretrained(path)


def load_ajev_config(path: str) -> dict:
    p = os.path.join(path, AJEV_CONFIG)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


