"""scripts/milestone_eval.py: CPU only, tiny model, fake milestones built exactly the
way quipu/train.py builds them (bf16 state_dict; the tied "embed.weight" and
"lm_head.weight" are both present as keys, sharing one on-disk storage).
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import quipu.train as train_mod
from quipu.config import Config, DataConfig, ModelConfig, TrainConfig
from quipu.data import write_shard
from quipu.model import Quipu
from quipu.train import LATEST

_SPEC = importlib.util.spec_from_file_location(
    "milestone_eval", Path(__file__).resolve().parent.parent / "scripts" / "milestone_eval.py"
)
milestone_eval = importlib.util.module_from_spec(_SPEC)
sys.modules["milestone_eval"] = milestone_eval
_SPEC.loader.exec_module(milestone_eval)


VOCAB = 50257  # real gpt2 vocab, so the fixed prompts encode/decode meaningfully


def tiny_model_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=VOCAB, d_model=16, n_layer=1, n_head=2, n_kv_head=1,
        ffn_hidden=32, context=16, rope_base=10000.0, norm_eps=1e-6,
    )


def tiny_data_cfg(shard_dir: Path) -> DataConfig:
    return DataConfig(
        dataset="x", subset="x", shard_dir=str(shard_dir), shard_tokens=1000, val_tokens=1000,
        code_dataset="x", code_share=0.1, code_languages=("Python",), code_licenses=("mit",),
        html_cap=0.1, code_val_tokens=1000, code_heldout_first_file=1, code_files_total=2,
    )


def tiny_train_cfg(ckpt_dir: Path) -> TrainConfig:
    return TrainConfig(
        total_tokens=16 * 2 * 40, batch_tokens=16 * 2, micro_batch=2, context=16,
        lr=1e-3, lr_min=1e-4, warmup_steps=2, weight_decay=0.1, beta1=0.9, beta2=0.95,
        grad_clip=1.0, seed=7, ckpt_dir=str(ckpt_dir), ckpt_every=1000, ckpt_keep=3,
        eval_every=1000, eval_batches=1,
    )


def build_cfg(tmp_path: Path) -> Config:
    return Config(
        name="tiny", model=tiny_model_cfg(),
        data=tiny_data_cfg(tmp_path / "data"),
        train=tiny_train_cfg(tmp_path / "checkpoints"),
    )


def _write_val_shards(cfg: Config, with_code: bool = True, seed: int = 0) -> None:
    shard_dir = Path(cfg.data.shard_dir)
    rng = np.random.RandomState(seed)
    write_shard(shard_dir / "val" / "shard_000.bin", rng.randint(0, VOCAB, 400).astype(np.uint16))
    if with_code:
        write_shard(
            shard_dir / "code_val" / "shard_000.bin",
            rng.randint(0, VOCAB, 400).astype(np.uint16),
        )


def _save_milestone(cfg: Config, step: int, model: Quipu) -> Path:
    """Replicates Trainer.save_milestone exactly: bf16 state_dict where the tied
    "embed.weight"/"lm_head.weight" tensor is written to disk once (both keys are
    still present, pointing at the same storage), atomic save."""
    path = Path(cfg.train.ckpt_dir) / "milestones" / f"step_{step:06d}.pt"
    state = train_mod._bf16_state_dict(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    train_mod._atomic_save(state, path)
    return path


def _save_final(cfg: Config, step: int, model: Quipu) -> Path:
    """Replicates Trainer.save_checkpoint's full (fp32) checkpoint + latest.pt pointer."""
    ckpt_dir = Path(cfg.train.ckpt_dir)
    path = ckpt_dir / f"step_{step:06d}.pt"
    train_mod._atomic_save(
        {
            "step": step, "model": model.state_dict(), "optimizer": {},
            "stream": {"position": 0, "wraps": 0}, "skipped_steps": 0,
            "torch_rng": torch.get_rng_state(), "cuda_rng": None,
        },
        path,
    )
    train_mod._atomic_save({"file": path.name}, ckpt_dir / LATEST)
    return path


def _fresh_model(cfg: Config, seed: int) -> Quipu:
    torch.manual_seed(seed)
    return Quipu(cfg.model)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------


def test_run_produces_metrics_and_samples_with_right_structure(tmp_path):
    cfg = build_cfg(tmp_path)
    _write_val_shards(cfg)
    _save_milestone(cfg, 10, _fresh_model(cfg, 1))
    _save_milestone(cfg, 20, _fresh_model(cfg, 2))
    _save_final(cfg, 30, _fresh_model(cfg, 3))

    out_dir = tmp_path / "out"
    milestone_eval.run(cfg, device="cpu", eval_batches=2, out_dir=out_dir)

    import json
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))["checkpoints"]
    assert [m["step"] for m in metrics] == [10, 20, 30]
    assert [m["label"] for m in metrics] == ["step_000010", "step_000020", "final"]
    for m in metrics:
        assert isinstance(m["text_val_loss"], float)
        assert isinstance(m["code_val_loss"], float)
        assert isinstance(m["seconds"], float)
        assert "note" not in m

    samples = (out_dir / "samples.md").read_text(encoding="utf-8")
    for prompt in milestone_eval.ALL_PROMPTS:
        assert f"`{prompt}`" in samples
    assert "step_000010" in samples and "step_000020" in samples and "final" in samples


def test_samples_are_ordered_by_prompt_then_by_checkpoint_step(tmp_path):
    cfg = build_cfg(tmp_path)
    _write_val_shards(cfg)
    _save_milestone(cfg, 10, _fresh_model(cfg, 1))
    _save_milestone(cfg, 20, _fresh_model(cfg, 2))
    _save_final(cfg, 30, _fresh_model(cfg, 3))

    out_dir = tmp_path / "out"
    milestone_eval.run(cfg, device="cpu", eval_batches=2, out_dir=out_dir)
    text = (out_dir / "samples.md").read_text(encoding="utf-8")

    prompt_positions = [text.index(f"`{p}`") for p in milestone_eval.ALL_PROMPTS]
    assert prompt_positions == sorted(prompt_positions), "prompts must appear in fixed order"

    for prompt in milestone_eval.ALL_PROMPTS:
        start = text.index(f"`{prompt}`")
        end = min(
            (text.index(f"`{other}`") for other in milestone_eval.ALL_PROMPTS
             if text.index(f"`{other}`") > start),
            default=len(text),
        )
        section = text[start:end]
        step_order = [section.index(label) for label in ("step_000010", "step_000020", "final")]
        assert step_order == sorted(step_order), f"checkpoints out of order for prompt {prompt!r}"


def test_greedy_sample_is_deterministic_across_two_runs(tmp_path):
    cfg = build_cfg(tmp_path)
    _write_val_shards(cfg)
    _save_milestone(cfg, 10, _fresh_model(cfg, 1))

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    milestone_eval.run(cfg, device="cpu", eval_batches=2, out_dir=out_a)
    milestone_eval.run(cfg, device="cpu", eval_batches=2, out_dir=out_b)

    text_a = (out_a / "samples.md").read_text(encoding="utf-8")
    text_b = (out_b / "samples.md").read_text(encoding="utf-8")
    assert text_a == text_b, "greedy decoding + seeded sampling must be reproducible"


def test_checkpoint_files_are_untouched(tmp_path):
    cfg = build_cfg(tmp_path)
    _write_val_shards(cfg)
    m1 = _save_milestone(cfg, 10, _fresh_model(cfg, 1))
    m2 = _save_milestone(cfg, 20, _fresh_model(cfg, 2))
    final = _save_final(cfg, 30, _fresh_model(cfg, 3))
    latest = Path(cfg.train.ckpt_dir) / LATEST

    before = {p: _hash(p) for p in (m1, m2, final, latest)}
    milestone_eval.run(cfg, device="cpu", eval_batches=2, out_dir=tmp_path / "out")
    after = {p: _hash(p) for p in (m1, m2, final, latest)}

    assert before == after, "evaluation must never modify a checkpoint file"


def test_missing_code_val_dir_is_noted_not_a_crash(tmp_path):
    cfg = build_cfg(tmp_path)
    _write_val_shards(cfg, with_code=False)
    _save_milestone(cfg, 10, _fresh_model(cfg, 1))

    out_dir = tmp_path / "out"
    milestone_eval.run(cfg, device="cpu", eval_batches=2, out_dir=out_dir)

    import json
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))["checkpoints"]
    assert metrics[0]["code_val_loss"] is None
    assert "code_val" in metrics[0]["note"]


def test_extra_unexpected_key_fails_loudly(tmp_path):
    cfg = build_cfg(tmp_path)
    model = _fresh_model(cfg, 1)
    path = _save_milestone(cfg, 10, model)
    state = torch.load(path, map_location="cpu", weights_only=True)
    state["totally.unexpected.key"] = torch.zeros(3)
    train_mod._atomic_save(state, path)

    with pytest.raises(AssertionError):
        milestone_eval.load_milestone_model(cfg, path, "cpu")


def test_missing_a_real_weight_fails_loudly(tmp_path):
    cfg = build_cfg(tmp_path)
    model = _fresh_model(cfg, 1)
    path = _save_milestone(cfg, 10, model)
    state = torch.load(path, map_location="cpu", weights_only=True)
    del state["blocks.0.attn.q.weight"]  # a real weight
    train_mod._atomic_save(state, path)

    with pytest.raises(AssertionError):
        milestone_eval.load_milestone_model(cfg, path, "cpu")


def test_no_checkpoints_found_exits_nonzero(tmp_path):
    cfg = build_cfg(tmp_path)
    _write_val_shards(cfg)
    with pytest.raises(milestone_eval.NoCheckpointsFound):
        milestone_eval.discover_checkpoints(Path(cfg.train.ckpt_dir))


def test_final_identical_to_last_milestone_is_skipped(tmp_path):
    cfg = build_cfg(tmp_path)
    model = _fresh_model(cfg, 1)
    _save_milestone(cfg, 20, model)
    _save_final(cfg, 20, model)  # same step, same weights

    checkpoints = milestone_eval.discover_checkpoints(Path(cfg.train.ckpt_dir))
    assert [label for label, _, _, _ in checkpoints] == ["step_000020"]


def test_final_at_same_step_with_different_weights_is_kept(tmp_path):
    cfg = build_cfg(tmp_path)
    _save_milestone(cfg, 20, _fresh_model(cfg, 1))
    _save_final(cfg, 20, _fresh_model(cfg, 99))  # same step, different weights

    checkpoints = milestone_eval.discover_checkpoints(Path(cfg.train.ckpt_dir))
    assert [label for label, _, _, _ in checkpoints] == ["step_000020", "final"]


def test_a_corrupt_milestone_does_not_lose_the_other_checkpoints_results(tmp_path):
    cfg = build_cfg(tmp_path)
    _write_val_shards(cfg)
    _save_milestone(cfg, 10, _fresh_model(cfg, 1))
    bad_path = _save_milestone(cfg, 20, _fresh_model(cfg, 2))
    _save_milestone(cfg, 30, _fresh_model(cfg, 3))
    # Corrupt the middle milestone: delete a real weight, matching how
    # load_milestone_model fails loudly on a real corruption.
    state = torch.load(bad_path, map_location="cpu", weights_only=True)
    del state["blocks.0.attn.q.weight"]
    train_mod._atomic_save(state, bad_path)

    out_dir = tmp_path / "out"
    ok = milestone_eval.run(cfg, device="cpu", eval_batches=2, out_dir=out_dir)
    assert ok is False, "a per-checkpoint failure must be reported back to the caller"

    import json
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))["checkpoints"]
    by_step = {m["step"]: m for m in metrics}
    assert set(by_step) == {10, 20, 30}
    assert "error" not in by_step[10] and isinstance(by_step[10]["text_val_loss"], float)
    assert "error" not in by_step[30] and isinstance(by_step[30]["text_val_loss"], float)
    assert "error" in by_step[20] and "text_val_loss" not in by_step[20]

    samples = (out_dir / "samples.md").read_text(encoding="utf-8")
    assert "step_000020" in samples and "error" in samples.lower()
    for prompt in milestone_eval.ALL_PROMPTS:
        section_start = samples.index(f"`{prompt}`")
        # The good checkpoints still produced samples for every prompt.
        assert "step_000010" in samples[section_start:]
        assert "step_000030" in samples[section_start:]
