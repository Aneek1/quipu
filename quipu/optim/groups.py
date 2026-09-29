"""build_optimizers(model, cfg): which optimizer, and which settings, each parameter gets.

optimizer "adamw" is train.py's grouping as it stands (quipu-114m): one AdamW, two
groups,
    decay    every parameter with p.dim() >= 2, the tied embedding included
             (GPT-2/nanoGPT decay a tied embedding), weight_decay = cfg.weight_decay
    no_decay everything else (RMSNorm scales, AttnRes pseudo-queries), weight_decay 0
in model.parameters() order.

optimizer "muon" (spec section 6.1): Muon for the weight matrices of the hidden
layers, AdamW for the rest.
    Muon, per head     attention q / k / v (head_dim = the Attention's head_dim;
                       whole-matrix when cfg.muon_per_head is False)
    Muon, whole matrix attention o, dense / shared-expert gate, up, down
    Muon, per expert   ExpertBank gate / up / down ([n, in, out]; in_out=True)
    AdamW, decayed     embeddings (and the lm_head tied to them), router weights
    AdamW, no decay    every 1-D parameter and every name in
                       model.no_decay_param_names() (norms, AttnRes queries)
The embedding and the lm_head see one token/class per row; their gradients are
sparse and row-wise, not a matrix transform, and the fp32 router is a tiny 2-D
classifier whose scale matters to the Top-k decision. Neither suits Muon's
fixed-spectral-norm steps.

Muon groups decay with cfg.muon_weight_decay, not cfg.weight_decay (see the
TrainConfig comment: lr * wd is the per-step shrink, and muon_lr is ~33x lr).

Every group carries base_lr (cfg.lr for AdamW, cfg.muon_lr for Muon): the LR schedule
sets group["lr"] = group["base_lr"] * factor on every group of every optimizer.
Optimizer.load_state_dict brings back the saved base_lr and weight_decay, so on
resume call apply_config_lrs(optimizers, cfg) after loading to make the config win.
The Quantile Balancing bias is a buffer, so no optimizer ever sees it. A parameter
the rules do not place is an error, not a silent default.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from quipu.config import OPTIMIZERS, TrainConfig
from quipu.model import Attention
from quipu.moe import ExpertBank, Router
from quipu.optim.muon import Muon

MUON_NESTEROV = True   # not a config field; every Muon group uses Nesterov momentum


def _no_decay_ids(model: nn.Module) -> set[int]:
    """Every 1-D parameter, plus the model's own no-decay list when it has one."""
    ids = {id(p) for p in model.parameters() if p.dim() < 2}
    named = dict(model.named_parameters())
    names_fn = getattr(model, "no_decay_param_names", None)
    if names_fn is not None:
        for name in names_fn():
            if name not in named:
                raise ValueError(f"no_decay_param_names lists {name!r}, not a parameter")
            ids.add(id(named[name]))
    return ids


def _adamw(decay: list[nn.Parameter], no_decay: list[nn.Parameter],
           cfg: TrainConfig) -> torch.optim.AdamW | None:
    groups = [
        {"params": params, "weight_decay": wd, "base_lr": cfg.lr, "decay": is_decay}
        for params, wd, is_decay in ((decay, cfg.weight_decay, True), (no_decay, 0.0, False))
        if params
    ]
    if not groups:
        return None
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))


def build_optimizers(model: nn.Module, cfg: TrainConfig) -> list[torch.optim.Optimizer]:
    """The optimizers for model under cfg.optimizer (module docstring). Step and
    zero_grad every one; their param groups partition model.parameters()."""
    if cfg.optimizer not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {list(OPTIMIZERS)}, got {cfg.optimizer!r}")
    params = list(model.parameters())          # deduplicated: the tied weight once
    no_decay = _no_decay_ids(model)

    if cfg.optimizer == "adamw":
        opt = _adamw([p for p in params if id(p) not in no_decay],
                     [p for p in params if id(p) in no_decay], cfg)
        return [opt] if opt is not None else []

    per_head: dict[int, int] = {}              # id -> head_dim, attention q / k / v
    adam_matrix: set[int] = set()              # embeddings, tied lm_head, router
    expert: set[int] = set()
    for module in model.modules():
        if isinstance(module, Attention):
            for lin in (module.q, module.k, module.v):
                per_head[id(lin.weight)] = module.head_dim
        elif isinstance(module, (nn.Embedding, Router)):
            adam_matrix.add(id(module.weight))
        elif isinstance(module, ExpertBank):
            expert.update(id(p) for p in module.parameters(recurse=False))
    lm_head = getattr(model, "lm_head", None)
    if isinstance(lm_head, nn.Linear):
        adam_matrix.add(id(lm_head.weight))

    adam_decay, adam_no_decay = [], []
    muon_head: dict[int, list[nn.Parameter]] = {}
    muon_matrix, muon_expert = [], []
    names = {id(p): n for n, p in model.named_parameters()}
    for p in params:
        pid = id(p)
        if pid in no_decay:
            adam_no_decay.append(p)
        elif pid in adam_matrix:
            adam_decay.append(p)
        elif pid in expert and p.dim() == 3:
            muon_expert.append(p)
        elif pid in per_head and cfg.muon_per_head:
            muon_head.setdefault(per_head[pid], []).append(p)
        elif p.dim() == 2:
            muon_matrix.append(p)
        else:
            raise ValueError(f"no optimizer rule for parameter {names[pid]!r} "
                             f"(shape {tuple(p.shape)})")

    muon_common = {"lr": cfg.muon_lr, "base_lr": cfg.muon_lr, "momentum": cfg.muon_momentum,
                   "weight_decay": cfg.muon_weight_decay, "ns_steps": cfg.muon_ns_steps,
                   "nesterov": MUON_NESTEROV}
    muon_groups = [
        {"params": ps, "head_dim": hd, "in_out": False, **muon_common}
        for hd, ps in muon_head.items()
    ]
    if muon_matrix:
        muon_groups.append({"params": muon_matrix, "head_dim": None, "in_out": False,
                            **muon_common})
    if muon_expert:
        muon_groups.append({"params": muon_expert, "head_dim": None, "in_out": True,
                            **muon_common})

    opts: list[torch.optim.Optimizer] = []
    if muon_groups:
        opts.append(Muon(muon_groups))
    adam = _adamw(adam_decay, adam_no_decay, cfg)
    if adam is not None:
        opts.append(adam)
    return opts


def apply_config_lrs(optimizers: list[torch.optim.Optimizer], cfg: TrainConfig) -> None:
    """Re-set base_lr and weight_decay on every group from cfg, for use right after
    load_state_dict (which restores the checkpoint's values). Muon groups get
    cfg.muon_lr / cfg.muon_weight_decay, and momentum / ns_steps / nesterov as
    build_optimizers sets them; AdamW groups cfg.lr, and cfg.weight_decay for the
    decay group (tagged group["decay"] at build) while the no-decay group stays at 0.
    group["lr"] is left alone: the LR schedule derives it from base_lr before the
    next step."""
    for opt in optimizers:
        is_muon = isinstance(opt, Muon)
        for group in opt.param_groups:
            if is_muon:
                group["base_lr"] = cfg.muon_lr
                group["weight_decay"] = cfg.muon_weight_decay
                group["momentum"] = cfg.muon_momentum
                group["ns_steps"] = cfg.muon_ns_steps
                group["nesterov"] = MUON_NESTEROV
            else:
                if "decay" not in group:
                    raise ValueError("AdamW group has no 'decay' tag; was it built by "
                                     "build_optimizers?")
                group["base_lr"] = cfg.lr
                group["weight_decay"] = cfg.weight_decay if group["decay"] else 0.0
