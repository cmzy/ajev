"""Inference wrapper implementing the ``Predictor`` protocol for a trained checkpoint."""

from __future__ import annotations

import torch

from ajev.model.batching import collate, length_sorted_batches
from ajev.model.encoder import DecisionModel, load_ajev_config, load_tokenizer
from ajev.model.encoding import DecisionEncoder
from ajev.schema import Decision


def autocast_dtype(device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


@torch.no_grad()
def predict_logits(model: DecisionModel, encoder: DecisionEncoder, decisions: list[Decision],
                   device: torch.device, batch_size: int = 32) -> list[list[float]]:
    """Raw (uncalibrated) logits per decision, in option order."""
    model.eval()
    encs = [encoder.encode(d) for d in decisions]
    out: list[list[float]] = [[] for _ in decisions]
    dtype = autocast_dtype(device)
    for idx in length_sorted_batches([len(e.input_ids) for e in encs], batch_size):
        batch = {k: v.to(device) for k, v in collate([encs[i] for i in idx], encoder.tok.pad_token_id).items()}
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            logits = model(**batch)
        for row, i in zip(logits.float().cpu(), idx):
            out[i] = row[: len(decisions[i].options)].tolist()
    return out


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    t = torch.tensor(logits) / temperature
    return torch.softmax(t, dim=-1).tolist()


class EncoderPredictor:
    def __init__(self, path: str, device: str | None = None, batch_size: int = 32,
                 temperatures: dict[str, float] | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        cfg = load_ajev_config(path)
        self.encoder = DecisionEncoder(load_tokenizer(path), **cfg.get("encoding", {}))
        self.model = DecisionModel.from_pretrained(path).to(self.device)
        self.batch_size = batch_size
        # Per-type temperatures fitted by ajev.calibrate; 1.0 when uncalibrated.
        self.temperatures = temperatures if temperatures is not None else cfg.get("temperatures", {})

    def predict_logits(self, decisions: list[Decision]) -> list[list[float]]:
        return predict_logits(self.model, self.encoder, decisions, self.device, self.batch_size)

    def predict(self, decisions: list[Decision]) -> list[list[float]]:
        logits = self.predict_logits(decisions)
        return [softmax(lg, self.temperatures.get(d.type, 1.0)) for d, lg in zip(decisions, logits)]
