"""M7b: the FP8 training option (spec section 12).

CPU tests build everything with device="cpu" explicitly (CUDA_VISIBLE_DEVICES=""
does not hide the laptop GPU). CUDA tests are skipped without a CUDA device, or
when less than 3 GB of it is free (Ollama may be holding it).
"""
import copy
import dataclasses
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

import quipu.train as train_mod
from quipu.config import load_config
from quipu.data import write_shard
from quipu.fp8 import Fp8Unsupported, apply_precision, convert_to_fp8, fp8_targets
from quipu.model_moe import QuipuMoE
from quipu.optim import build_optimizers
from quipu.train import Trainer
from tests.test_train_moe import SMOKE, smoke

FP8_TYPES = ("Float8Linear",)


def smoke_model(**over) -> QuipuMoE:
    cfg = load_config(SMOKE, {"model": over} if over else None)
    torch.manual_seed(0)
    return QuipuMoE(cfg.model)


def expected_targets(model: QuipuMoE) -> set[str]:
    """Written out by hand: attention q/k/v/o and every shared-expert linear."""
    names = set()
    for l in range(model.cfg.n_layer):
        names |= {f"blocks.{l}.attn.{p}" for p in ("q", "k", "v", "o")}
        for s in range(model.cfg.shared_experts):
            names |= {f"blocks.{l}.moe.shared.{s}.{p}" for p in ("gate", "up", "down")}
    return names


# ---- config -----------------------------------------------------------------------

def test_precision_defaults_to_bf16(tmp_path):
    assert smoke(tmp_path).train.precision == "bf16"


def test_precision_fp8_is_accepted_for_moe(tmp_path):
    assert smoke(tmp_path, precision="fp8").train.precision == "fp8"


@pytest.mark.parametrize("bad", ["fp16", "FP8", "", 8, None, True])
def test_an_unknown_precision_is_refused(tmp_path, bad):
    with pytest.raises(ValueError, match="precision"):
        smoke(tmp_path, precision=bad)


def test_fp8_is_refused_for_the_dense_model(tmp_path):
    dense = {"kind": "dense", "n_experts": 0, "top_k": 0, "expert_hidden": 0,
             "shared_experts": 0, "shared_hidden": 0, "activation": "swiglu",
             "attnres_blocks": 0}
    load_config(SMOKE, {"model": dense})                       # bf16 dense is fine
    with pytest.raises(ValueError, match="precision"):
        load_config(SMOKE, {"model": dense, "train": {"precision": "fp8"}})


# ---- which layers -----------------------------------------------------------------

def test_the_targets_are_exactly_attention_and_shared_expert_linears():
    model = smoke_model()
    assert set(fp8_targets(model)) == expected_targets(model)
    assert len(fp8_targets(model)) == len(set(fp8_targets(model)))


def test_conversion_swaps_exactly_the_targets_and_nothing_else():
    model = smoke_model()
    before = {n: type(m) for n, m in model.named_modules()}
    converted = convert_to_fp8(model)
    assert set(converted) == expected_targets(model)
    for name, module in model.named_modules():
        if name in expected_targets(model):
            assert type(module).__name__ in FP8_TYPES, name
            assert isinstance(module, nn.Linear), name          # still a Linear
        else:
            assert type(module) is before[name], name
    # Router, embedding, tied head, norms, AttnRes and the expert bank untouched.
    assert type(model.lm_head) is nn.Linear
    assert model.lm_head.weight is model.embed.weight


def test_router_head_and_experts_are_never_targets():
    model = smoke_model()
    targets = set(fp8_targets(model))
    for name, _ in model.named_modules():
        if name.endswith(("router", "experts", "balancer", "lm_head", "embed", "norm",
                          "norm1", "norm2")) or name.startswith("attnres"):
            assert name not in targets, name


def test_bf16_leaves_the_model_untouched():
    model = smoke_model()
    before = {n: (type(m), m) for n, m in model.named_modules()}
    params = {n: p for n, p in model.named_parameters()}
    out = apply_precision(model, "bf16", "cpu")
    assert out is model
    assert {n: (type(m), m) for n, m in model.named_modules()} == before
    assert {n: p for n, p in model.named_parameters()} == params


def test_apply_precision_refuses_an_unknown_precision():
    with pytest.raises(ValueError, match="precision"):
        apply_precision(smoke_model(), "fp4", "cpu")


def test_fp8_needs_a_cuda_device():
    with pytest.raises(Fp8Unsupported, match="CUDA"):
        apply_precision(smoke_model(), "fp8", "cpu")


def test_a_target_whose_width_is_not_a_multiple_of_16_is_refused():
    # shared_hidden 120: the shared expert's gate/up out_features and down in_features.
    model = smoke_model(shared_hidden=120)
    with pytest.raises(ValueError, match="16"):
        convert_to_fp8(model)


# ---- checkpoints and optimizers ---------------------------------------------------

def test_state_dict_keys_shapes_and_values_are_those_of_bf16():
    plain = smoke_model()
    fp8 = copy.deepcopy(plain)
    convert_to_fp8(fp8)
    a, b = plain.state_dict(), fp8.state_dict()
    assert list(a) == list(b)
    for k in a:
        assert a[k].shape == b[k].shape and a[k].dtype == b[k].dtype, k
        assert torch.equal(a[k], b[k]), k


def test_checkpoints_load_both_ways_strictly():
    plain, other = smoke_model(), smoke_model()
    fp8 = copy.deepcopy(other)
    convert_to_fp8(fp8)
    fp8.load_state_dict(plain.state_dict(), strict=True)          # bf16 -> fp8
    fresh = smoke_model()
    fresh.load_state_dict(fp8.state_dict(), strict=True)          # fp8 -> bf16
    for (k, v), (_, w) in zip(plain.state_dict().items(), fresh.state_dict().items()):
        assert torch.equal(v, w), k


def test_parameters_stay_plain_and_are_the_same_objects():
    model = smoke_model()
    before = dict(model.named_parameters())
    convert_to_fp8(model)
    after = dict(model.named_parameters())
    assert list(before) == list(after)
    for n, p in after.items():
        assert p is before[n], n
        assert type(p) is nn.Parameter, n


def _group_map(model, cfg):
    """name -> (optimizer type, head_dim, in_out, weight_decay) for every parameter;
    fails if any parameter is in no group or in two."""
    names = {id(p): n for n, p in model.named_parameters()}
    out: dict[str, tuple] = {}
    for opt in build_optimizers(model, cfg):
        for g in opt.param_groups:
            for p in g["params"]:
                n = names[id(p)]
                assert n not in out, f"{n} is in two groups"
                out[n] = (type(opt).__name__, g.get("head_dim"), g.get("in_out"),
                          g["weight_decay"])
    assert set(out) == set(names.values()), "a parameter is in no group"
    return out


@pytest.mark.parametrize("optimizer", ["muon", "adamw"])
def test_optimizer_groups_cover_every_parameter_once_and_do_not_change(tmp_path, optimizer):
    cfg = smoke(tmp_path, optimizer=optimizer).train
    plain = smoke_model()
    fp8 = copy.deepcopy(plain)
    convert_to_fp8(fp8)
    assert _group_map(fp8, cfg) == _group_map(plain, cfg)
    if optimizer == "muon":
        groups = _group_map(fp8, cfg)
        assert groups["blocks.0.attn.q.weight"][:2] == ("Muon", fp8.cfg.head_dim)
        assert groups["blocks.0.moe.shared.0.up.weight"][:2] == ("Muon", None)


def test_the_trainer_refuses_fp8_on_cpu(tmp_path):
    cfg = smoke(tmp_path, precision="fp8")
    data = tmp_path / "data"
    write_shard(data / "shard_000.bin", np.zeros(20_000, dtype=np.uint16))
    with pytest.raises(Fp8Unsupported):
        Trainer(model_cfg=cfg.model, train_cfg=cfg.train, shard_dir=data, device="cpu",
                run_dir=tmp_path / "runs", run_id="t")


def test_fp8_on_cpu_is_a_usage_error_not_a_crash(tmp_path, monkeypatch):
    path = tmp_path / "fp8.toml"
    path.write_text(SMOKE.read_text(encoding="utf-8").replace(
        'optimizer     = "muon"', 'optimizer     = "muon"\nprecision     = "fp8"'),
        encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    code = train_mod.run_main(["--config", str(path), "--device", "cpu", "--run-id", "x"])
    assert code == train_mod.EXIT_USAGE


# ---- CUDA -------------------------------------------------------------------------

def _cuda_free_gb() -> float:
    if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
        return 0.0
    try:
        return torch.cuda.mem_get_info()[0] / 2**30
    except RuntimeError:
        return 0.0


cuda = pytest.mark.skipif(_cuda_free_gb() < 3.0,
                          reason="needs a CUDA device with >= 3 GB free")


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@cuda
def test_fp8_forward_is_close_to_bf16():
    cfg = dataclasses.replace(load_config(SMOKE).model, d_model=256, n_head=8, n_kv_head=4,
                              shared_hidden=512, expert_hidden=128)
    torch.manual_seed(0)
    ref = QuipuMoE(cfg).cuda()
    fp8 = copy.deepcopy(ref)
    convert_to_fp8(fp8)
    x = torch.randint(cfg.vocab_size, (4, cfg.context), device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        a, b = ref(x), fp8(x)
    # bf16 vs fp32 on this model is ~0.01 and FP8 vs bf16 ~0.06 (e4m3 keeps 3
    # mantissa bits); 0.1 catches a wrong scale or a transposed operand.
    err = _rel_err(b, a)
    print(f"\nFP8 vs bf16 logits relative error: {err:.4f}")
    assert math.isfinite(err) and err < 0.1


def _markov_data(path, tokens=400_000, vocab=512, seed=0):
    """A learnable stream: each token has 4 possible successors."""
    rng = np.random.RandomState(seed)
    succ = rng.randint(0, vocab, (vocab, 4))
    choice = rng.randint(0, 4, tokens)
    out = np.empty(tokens, dtype=np.uint16)
    t = 0
    for i in range(tokens):
        out[i] = t
        t = succ[t, choice[i]]
    write_shard(path / "shard_000.bin", out)
    return path


def _train(tmp_path, data, precision, seed, steps=100, run_id=None, **train):
    cfg = smoke(tmp_path / f"{precision}-{seed}", precision=precision, seed=seed,
                milestones=[], **train)
    tr = Trainer(model_cfg=cfg.model, train_cfg=cfg.train, shard_dir=data, device="cuda",
                 run_dir=tmp_path / "runs", run_id=run_id or f"{precision}-{seed}")
    losses = [tr.train_step() for _ in range(steps)]
    return tr, losses


@cuda
def test_100_smoke_steps_of_fp8_track_bf16_within_seed_noise(tmp_path):
    data = _markov_data(tmp_path / "data")
    _, bf16 = _train(tmp_path, data, "bf16", 1337)
    _, bf16_b = _train(tmp_path, data, "bf16", 1338)
    tr, fp8 = _train(tmp_path, data, "fp8", 1337)
    assert any(type(m).__name__ in FP8_TYPES for m in tr.model.modules())

    def tail(ls):                      # mean of the last 10 steps: less step noise
        return sum(ls[-10:]) / 10
    noise = abs(tail(bf16) - tail(bf16_b))
    print(f"\n100-step smoke loss (last-10 mean): bf16 {tail(bf16):.4f}, "
          f"bf16 seed+1 {tail(bf16_b):.4f}, fp8 {tail(fp8):.4f}; "
          f"final step bf16 {bf16[-1]:.4f}, fp8 {fp8[-1]:.4f}")
    assert all(math.isfinite(l) for l in fp8)
    assert fp8[-1] < fp8[0] - 1.0                          # it learned
    assert abs(tail(fp8) - tail(bf16)) <= max(3 * noise, 0.05)


@cuda
def test_the_compile_trial_covers_the_fp8_model(tmp_path, monkeypatch):
    # aot_eager runs dynamo and AOT autograd (what inductor sees) without Triton.
    monkeypatch.setattr(train_mod, "COMPILE_BACKEND", "aot_eager")
    data = _markov_data(tmp_path / "data", tokens=50_000)
    tr, losses = _train(tmp_path, data, "fp8", 1337, steps=3, compile=True)
    assert tr.compiled
    assert tr.forward_model is not tr.model
    assert any(type(m).__name__ in FP8_TYPES for m in tr.model.modules())
    assert all(math.isfinite(l) for l in losses)
