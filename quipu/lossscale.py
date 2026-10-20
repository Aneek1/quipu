"""Dynamic loss scale for fp16 training (train.precision "fp16").

fp16 underflows below ~6e-8, where many gradients live, so each micro-batch's loss is
multiplied by `scale` before backward and the gradients divided by it afterwards
(unscale_, after the all-reduce, before clipping). A step whose gradients are not
finite is an overflow: the trainer skips it and calls overflow(), which halves the
scale; after growth_interval clean steps in a row good_step() doubles it. The scale
never drops below `floor`; an overflow at the floor is a real non-finite step and
counts toward the trainer's poison streak (at_floor).

Not torch.amp.GradScaler: the trainer reduces gradients at one known point, has its
own non-finite skip, and steps two optimizers (Muon + AdamW).
"""
from __future__ import annotations

from typing import Any, Iterable

import torch

INIT_SCALE = 2.0 ** 16
GROWTH_INTERVAL = 2000
FLOOR = 1.0


class LossScaler:
    def __init__(self, enabled: bool, init_scale: float = INIT_SCALE,
                 growth_interval: int = GROWTH_INTERVAL, floor: float = FLOOR) -> None:
        self.enabled = enabled
        self.scale = float(init_scale) if enabled else 1.0
        self.growth_interval = growth_interval
        self.floor = floor
        self.clean_steps = 0

    @property
    def at_floor(self) -> bool:
        return self.scale <= self.floor

    @torch.no_grad()
    def unscale_(self, params: Iterable[torch.nn.Parameter]) -> None:
        if not self.enabled or self.scale == 1.0:
            return
        inv = 1.0 / self.scale
        for p in params:
            if p.grad is not None:
                p.grad.mul_(inv)

    def overflow(self) -> None:
        if self.enabled:
            self.scale = max(self.floor, self.scale / 2)
            self.clean_steps = 0

    def good_step(self) -> None:
        if not self.enabled:
            return
        self.clean_steps += 1
        if self.clean_steps >= self.growth_interval:
            self.scale *= 2
            self.clean_steps = 0

    def state_dict(self) -> dict[str, Any]:
        return {"scale": self.scale, "clean_steps": self.clean_steps}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self.enabled:
            self.scale = float(state["scale"])
            self.clean_steps = int(state["clean_steps"])
