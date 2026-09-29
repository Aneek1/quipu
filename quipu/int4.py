"""Weight-only int4 for the quipu-moe export (spec 12: int4 for inference only).

Why not torchao: its int4 weight-only path (Int4WeightOnlyConfig) raises "Requires mslk
>= 1.0.0" on this setup (torch 2.11 cu128, torchao 0.18, Windows), it quantizes only
nn.Linear modules (the routed experts, ~90% of the parameters, are 3-D ExpertBank
tensors), and a checkpoint of its tensor subclasses needs torchao (and its CUDA
kernels) to load. So this is the simple, portable scheme instead; any machine with
torch dequantizes it (hf/modeling_quipu_moe.py has the same dequantize_int4).

Scheme: group-wise asymmetric int4. The last dimension of a tensor is cut into
groups of `group_size` consecutive values (zero-padded to a whole group); each group
keeps a float16 scale and minimum, and each value the 4-bit code
round((w - min) / scale) in 0..15, two codes per byte (low nibble first). Dequantized:
code * scale + min, so every value is within scale / 2 of the original (plus float16
rounding of scale and min, which the codes are computed against).

Size: 4 bits + 2 x 16 bits per group of 128 = 4.25 bits per quantized value.

What is quantized (select_int4): every weight matrix of the attention (q, k, v, o),
the shared experts and the routed experts. Not quantized: the embedding / tied output
head (bf16), and the router, norms, balancing bias and AttnRes queries (fp32: tiny,
and routing decisions are sensitive to them).
"""
from __future__ import annotations

import torch

GROUP_SIZE = 128
FORMAT = "int4-groupwise-asym-v1"
SMALL = 1 << 20   # unquantized tensors below this many values stay fp32


def select_int4(name: str, tensor: torch.Tensor) -> bool:
    """Whether the export quantizes this state_dict entry."""
    if tensor.ndim < 2 or not tensor.is_floating_point():
        return False
    if name in ("embed.weight", "lm_head.weight") or ".router." in name:
        return False
    return (".attn." in name) or (".moe.shared." in name) or (".moe.experts." in name)


def quantize_int4(w: torch.Tensor, group_size: int = GROUP_SIZE
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(qweight uint8 [..., groups * group_size / 2], scales fp16 [..., groups],
    mins fp16 [..., groups]) for w (any float dtype, >= 1-D)."""
    if group_size < 2 or group_size % 2:
        raise ValueError(f"group_size must be even and >= 2, got {group_size}")
    w = w.detach().float()
    last = w.shape[-1]
    groups = -(-last // group_size)
    pad = groups * group_size - last
    if pad:
        # Pad with the row's last value, so padding never widens a group's range.
        w = torch.cat([w, w[..., -1:].expand(*w.shape[:-1], pad)], dim=-1)
    g = w.reshape(*w.shape[:-1], groups, group_size)
    mins = g.amin(-1).half()
    maxs = g.amax(-1)
    scales = ((maxs - mins.float()) / 15).clamp_min(1e-7).half()
    codes = torch.round((g - mins.float()[..., None]) / scales.float()[..., None]).clamp(0, 15)
    codes = codes.to(torch.uint8).reshape(*w.shape[:-1], groups, group_size // 2, 2)
    packed = codes[..., 0] | (codes[..., 1] << 4)
    return packed.reshape(*w.shape[:-1], groups * group_size // 2), scales, mins


def dequantize_int4(qweight: torch.Tensor, scales: torch.Tensor, mins: torch.Tensor,
                    shape: list[int] | torch.Size, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """The fp32 tensor of `shape` that quantize_int4 encoded."""
    shape = list(shape)
    lead, last = shape[:-1], shape[-1]
    groups = scales.shape[-1]
    q = qweight.reshape(*lead, groups, group_size // 2)
    codes = torch.stack([q & 0x0F, q >> 4], dim=-1).reshape(*lead, groups, group_size).float()
    w = codes * scales.float()[..., None] + mins.float()[..., None]
    return w.reshape(*lead, groups * group_size)[..., :last].contiguous()


def quantize_state(state: dict[str, torch.Tensor], group_size: int = GROUP_SIZE,
                   keep_dtype: torch.dtype = torch.bfloat16
                   ) -> tuple[dict[str, torch.Tensor], dict[str, dict]]:
    """(tensors to save, the "int4" metadata {name: {"shape", "group_size"}}). Selected
    tensors become name.qweight / name.scales / name.mins; the rest are cast to
    keep_dtype (floating) or kept."""
    out: dict[str, torch.Tensor] = {}
    meta: dict[str, dict] = {}
    for name, t in state.items():
        if select_int4(name, t):
            q, s, m = quantize_int4(t, group_size)
            out[name + ".qweight"], out[name + ".scales"], out[name + ".mins"] = q, s, m
            meta[name] = {"shape": list(t.shape), "group_size": group_size}
        elif t.is_floating_point():
            # Small tensors (router, norms, balancing bias, AttnRes queries) stay fp32:
            # they cost nothing and the routing decision is sensitive to them.
            dtype = torch.float32 if t.numel() < SMALL else keep_dtype
            out[name] = t.detach().to(dtype).contiguous()
        else:
            out[name] = t
    return out, meta


def dequantize_state(saved: dict[str, torch.Tensor], meta: dict[str, dict]) -> dict[str, torch.Tensor]:
    """The inverse of quantize_state, every tensor fp32."""
    saved = dict(saved)
    out = {}
    for name, q in meta.items():
        out[name] = dequantize_int4(saved.pop(name + ".qweight"), saved.pop(name + ".scales"),
                                    saved.pop(name + ".mins"), q["shape"], q["group_size"])
    out.update({k: v.float() if v.is_floating_point() else v for k, v in saved.items()})
    return out
