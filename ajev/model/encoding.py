"""Turn a Decision into encoder input ids with one marker token per option.

Layout (one sequence per decision, all options scored in a single pass):

    <bos> {type}: {instructions} <eos> <mask> {name}: {desc} <mask> {name}: {desc} ... <eos> {state} <eos>

The hidden state at each ``<mask>`` marker is scored by the decision head. Options come
before the state so that truncation only ever cuts the state. Budgets (in tokens):
instructions <= ``max_instr``, options section <= ``max_options``, state gets the rest.
"""

from __future__ import annotations

from dataclasses import dataclass

from ajev.schema import Decision

TYPE_PREFIX = {"noul": "yes/no", "choice": "choice", "score": "score"}


@dataclass
class Encoded:
    input_ids: list[int]
    marker_pos: list[int]  # index of each option's marker, in option order
    truncated_state: bool


class DecisionEncoder:
    def __init__(self, tokenizer, max_len: int = 1024, max_instr: int = 128, max_options: int = 384,
                 min_desc: int = 8) -> None:
        self.tok = tokenizer
        self.max_len = max_len
        self.max_instr = max_instr
        self.max_options = max_options
        self.min_desc = min_desc
        self.bos = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
        self.eos = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
        self.marker = tokenizer.mask_token_id
        if None in (self.bos, self.eos, self.marker):
            raise ValueError("tokenizer needs cls/bos, sep/eos and mask tokens")

    def _ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"] if text else []

    def _options(self, d: Decision) -> list[list[int]]:
        names = [self._ids(o.name) for o in d.options]
        descs = [self._ids(f": {o.desc}") if o.desc else [] for o in d.options]
        names_only = sum(1 + len(n) for n in names)
        if names_only > self.max_options:
            raise ValueError(
                f"{d.id}: {len(d.options)} option names need {names_only} tokens > max_options={self.max_options}"
            )
        # Largest per-option description length that fits the budget (binary search).
        longest = max((len(x) for x in descs), default=0)
        lo, hi = 0, longest
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if names_only + sum(min(len(x), mid) for x in descs) <= self.max_options:
                lo = mid
            else:
                hi = mid - 1
        cap = lo if lo >= self.min_desc or lo == longest else 0
        return [[self.marker] + n + x[:cap] for n, x in zip(names, descs)]

    def encode(self, d: Decision) -> Encoded:
        head = [self.bos] + self._ids(f"{TYPE_PREFIX[d.type]}: ")[:8] + self._ids(d.instructions)[: self.max_instr]
        head.append(self.eos)
        ids = list(head)
        marker_pos = []
        for chunk in self._options(d):
            marker_pos.append(len(ids))
            ids += chunk
        ids.append(self.eos)
        room = self.max_len - len(ids) - 1
        state = self._ids(d.state)
        truncated = len(state) > room
        ids += state[: max(room, 0)] + [self.eos]
        return Encoded(ids, marker_pos, truncated)
