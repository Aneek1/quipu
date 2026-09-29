"""The FP8 training option (spec section 12): train.precision = "bf16" | "fp8".

"fp8" swaps the attention projections (q, k, v, o) and the shared-expert linears
(gate, up, down) of a QuipuMoE for torchao's Float8Linear, tensorwise dynamic
scaling (torchao's "tensorwise" recipe): input and weight cast to e4m3, the output
gradient to e5m2, each with one amax-based scale per tensor per matmul, through
torch._scaled_mm. Everything else stays as under bf16 autocast:

- The swap keeps each nn.Parameter object and its name (Float8Linear is an
  nn.Linear subclass holding a plain fp32 `weight`), so master weights, optimizer
  state, Muon's Newton-Schulz and the optimizer grouping (quipu.optim.groups, which
  matches Attention.q/k/v by module) are unchanged, and the state_dict keys are
  exactly bf16's: a checkpoint loads into either precision, strictly.
- The router, embeddings / tied lm_head, norms, AttnRes and the routed ExpertBank
  are not touched. The experts stay bf16 because on sm_120 (RTX 5060 / 5090) the
  grouped FP8 GEMM (torch._scaled_grouped_mm) is not available (sm_90 / sm_100
  only) and per-expert torch._scaled_mm measured slower than the bf16 loop on the
  laptop (see results/fp8/laptop.md). Rowwise scaling is not supported on sm_120
  either, so tensorwise is the only recipe here.

Order in the trainer: build the model, move it to the device, apply_precision, then
build the optimizers and torch.compile (so the compile trial runs the FP8 model).
torchao is imported only when "fp8" is used.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from quipu.config import PRECISIONS
from quipu.model import Attention
from quipu.moe import MoELayer

FP8_RECIPE = "tensorwise"
# torch._scaled_mm needs both matrix dims of each operand divisible by 16.
FP8_ALIGN = 16
# FP8 tensor cores: Ada (sm_89), Hopper (sm_90), Blackwell (sm_100, sm_120).
MIN_CAPABILITY = (8, 9)


class Fp8Unsupported(ValueError):
    """precision "fp8" asked for where FP8 matmuls cannot run (no CUDA device, or
    one without FP8 tensor cores). A usage error: retrying cannot fix it."""


def fp8_targets(model: nn.Module) -> list[str]:
    """Qualified names of the nn.Linear modules the FP8 option converts: every
    Attention's q, k, v, o and every MoELayer's shared-expert gate, up, down."""
    names = {id(m): n for n, m in model.named_modules()}
    out: list[str] = []
    for module in model.modules():
        if isinstance(module, Attention):
            out += [names[id(lin)] for lin in (module.q, module.k, module.v, module.o)]
        elif isinstance(module, MoELayer):
            for ffn in module.shared:
                out += [names[id(lin)] for lin in (ffn.gate, ffn.up, ffn.down)]
    return out


def convert_to_fp8(model: nn.Module) -> list[str]:
    """Swap fp8_targets(model) for torchao Float8Linear in place. Returns the names
    converted. Refuses a target whose in/out width is not a multiple of FP8_ALIGN."""
    targets = set(fp8_targets(model))
    if not targets:
        raise ValueError("no FP8 targets in this model (no Attention or shared expert)")
    modules = dict(model.named_modules())
    for name in sorted(targets):
        lin = modules[name]
        if lin.in_features % FP8_ALIGN or lin.out_features % FP8_ALIGN:
            raise ValueError(f"{name} is {lin.in_features} -> {lin.out_features}; FP8 "
                             f"matmuls need both widths to be multiples of {FP8_ALIGN}")

    from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    converted: list[str] = []

    def keep(module: nn.Module, fqn: str) -> bool:
        if fqn in targets:
            converted.append(fqn)
            return True
        return False

    convert_to_float8_training(model, module_filter_fn=keep,
                               config=Float8LinearConfig.from_recipe_name(FP8_RECIPE))
    if set(converted) != targets:
        missing = sorted(targets - set(converted))
        raise RuntimeError(f"torchao did not convert {missing}")
    return converted


def fp8_unsupported(device: str | torch.device) -> str | None:
    """Why FP8 training cannot run on `device`, or None if it can."""
    dev = torch.device(device)
    if dev.type != "cuda":
        return f"FP8 training needs a CUDA device, got {str(device)!r}"
    if not torch.cuda.is_available():
        return "FP8 training needs CUDA, which is not available"
    cap = torch.cuda.get_device_capability(dev)
    if cap < MIN_CAPABILITY:
        return (f"FP8 training needs compute capability >= {MIN_CAPABILITY[0]}."
                f"{MIN_CAPABILITY[1]}, this GPU is {cap[0]}.{cap[1]}")
    return None


def apply_precision(model: nn.Module, precision: str, device: str | torch.device) -> nn.Module:
    """The model for train.precision: "bf16" returns it untouched, "fp8" converts
    it in place (convert_to_fp8) after checking the device. Returns the model."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {list(PRECISIONS)}, got {precision!r}")
    if precision == "bf16":
        return model
    reason = fp8_unsupported(device)
    if reason is not None:
        raise Fp8Unsupported(reason)
    convert_to_fp8(model)
    return model
