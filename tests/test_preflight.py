"""scripts/preflight.py's check logic, driven directly against fakes -- CPU only,
no CUDA, no real shards, no subprocess. preflight.py itself needs a GPU and the
real 3B-token shards (it is a script, not part of this suite; see its own
docstring), but the pieces that decide PASS/FAIL/SKIPPED are pure functions over
files and JSON, and this file proves they behave: a correct milestone set passes
every check, and a missing milestone fails loudly (mirroring what would make the
real run's summary exit 1).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

import quipu.train as train_mod
from quipu.config import Config, DataConfig, ModelConfig, TrainConfig
from quipu.model import Quipu

_SPEC = importlib.util.spec_from_file_location(
    "preflight", Path(__file__).resolve().parent.parent / "scripts" / "preflight.py"
)
preflight = importlib.util.module_from_spec(_SPEC)
sys.modules["preflight"] = preflight
_SPEC.loader.exec_module(preflight)


VOCAB = 50257  # real gpt2 vocab, matching how milestone_eval decodes fixed prompts


def tiny_model_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=VOCAB, d_model=16, n_layer=1, n_head=2, n_kv_head=1,
        ffn_hidden=32, context=16, rope_base=10000.0, norm_eps=1e-6,
    )


def tiny_data_cfg(shard_dir: Path) -> DataConfig:
    return DataConfig(
        dataset="x", subset="x", shard_dir=str(shard_dir), shard_tokens=1000, val_tokens=1000,
        code_dataset="x", code_share=0.1, code_languages=("Python", "JavaScript"),
        code_licenses=("mit", "apache-2.0"), html_cap=0.1, code_max_doc_tokens=100_000,
        code_val_tokens=1000, code_heldout_first_file=1, code_files_total=2,
    )


def tiny_train_cfg(ckpt_dir: Path, milestones=(2, 4), ckpt_keep=2) -> TrainConfig:
    return TrainConfig(
        total_tokens=16 * 2 * 5, batch_tokens=16 * 2, micro_batch=2, context=16,
        lr=1e-3, lr_min=1e-4, warmup_steps=1, weight_decay=0.1, beta1=0.9, beta2=0.95,
        grad_clip=1.0, seed=7, ckpt_dir=str(ckpt_dir), ckpt_every=2, ckpt_keep=ckpt_keep,
        eval_every=2, eval_batches=1, milestones=tuple(milestones),
    )


def build_cfg(tmp_path: Path, **train_kwargs) -> Config:
    return Config(
        name="tiny", model=tiny_model_cfg(),
        data=tiny_data_cfg(tmp_path / "data"),
        train=tiny_train_cfg(tmp_path / "checkpoints", **train_kwargs),
    )


def _fresh_model(cfg: Config, seed: int) -> Quipu:
    torch.manual_seed(seed)
    return Quipu(cfg.model)


def _save_milestone(cfg: Config, step: int, model: Quipu) -> Path:
    """Replicates Trainer.save_milestone exactly: bf16 state_dict, atomic save."""
    path = Path(cfg.train.ckpt_dir) / "milestones" / f"step_{step:06d}.pt"
    state = train_mod._bf16_state_dict(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    train_mod._atomic_save(state, path)
    return path


def _save_resumable_ckpt(cfg: Config, step: int) -> Path:
    """A minimal ckpt_dir/step_NNNNNN.pt: only its existence matters to
    check_milestones (it checks whether the resumable copy was pruned away)."""
    path = Path(cfg.train.ckpt_dir) / f"step_{step:06d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real checkpoint, presence is all that's checked")
    return path


# ---------------------------------------------------------------------------
# expected_milestone_steps
# ---------------------------------------------------------------------------


def test_expected_milestone_steps_adds_final():
    cfg = build_cfg(Path("/unused"), milestones=(2, 4))
    assert preflight.expected_milestone_steps(cfg.train) == [2, 4, 5]


# ---------------------------------------------------------------------------
# check_milestones
# ---------------------------------------------------------------------------


def test_check_milestones_passes_when_everything_is_right(tmp_path):
    cfg = build_cfg(tmp_path, milestones=(2, 4), ckpt_keep=2)
    ckpt_dir = Path(cfg.train.ckpt_dir)
    for step in (2, 4, 5):
        _save_milestone(cfg, step, _fresh_model(cfg, step))
    # Only steps 4 and 5 kept resumable (ckpt_keep=2); step 2's resumable copy was
    # pruned, but its milestone must survive -- this is what the last check proves.
    _save_resumable_ckpt(cfg, 4)
    _save_resumable_ckpt(cfg, 5)

    check = preflight.Checks()
    preflight.check_milestones(check, cfg, ckpt_dir)

    assert check.failed == 0, "every milestone check should pass on a well-formed run"


def test_check_milestones_fails_on_a_missing_milestone(tmp_path):
    cfg = build_cfg(tmp_path, milestones=(2, 4), ckpt_keep=2)
    ckpt_dir = Path(cfg.train.ckpt_dir)
    # Step 4's milestone is simply never written (e.g. Ctrl+C between eval and the
    # milestone save).
    _save_milestone(cfg, 2, _fresh_model(cfg, 2))
    _save_milestone(cfg, 5, _fresh_model(cfg, 5))
    _save_resumable_ckpt(cfg, 4)
    _save_resumable_ckpt(cfg, 5)

    check = preflight.Checks()
    preflight.check_milestones(check, cfg, ckpt_dir)

    assert check.failed >= 1, "a missing milestone must fail loudly, not pass silently"
    # This mirrors main()'s summary: `failed = check.failed or check.skipped`, and
    # `sys.exit(1 if failed else 0)`.
    assert (check.failed or check.skipped) != 0


def test_check_milestones_fails_when_a_milestone_is_not_bf16(tmp_path):
    cfg = build_cfg(tmp_path, milestones=(2,), ckpt_keep=2)
    ckpt_dir = Path(cfg.train.ckpt_dir)
    model = _fresh_model(cfg, 1)
    # fp32 state dict where bf16 was required.
    path = ckpt_dir / "milestones" / "step_000002.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    train_mod._atomic_save(model.state_dict(), path)
    _save_milestone(cfg, 5, _fresh_model(cfg, 5))
    _save_resumable_ckpt(cfg, 5)

    check = preflight.Checks()
    preflight.check_milestones(check, cfg, ckpt_dir)

    assert check.failed >= 1


def test_check_milestones_fails_when_load_state_dict_would_break_the_tie(tmp_path):
    cfg = build_cfg(tmp_path, milestones=(2,), ckpt_keep=2)
    ckpt_dir = Path(cfg.train.ckpt_dir)
    model = _fresh_model(cfg, 1)
    state = train_mod._bf16_state_dict(model)
    del state["lm_head.weight"]  # simulate a corrupted/incompatible checkpoint
    path = ckpt_dir / "milestones" / "step_000002.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    train_mod._atomic_save(state, path)
    _save_milestone(cfg, 5, _fresh_model(cfg, 5))
    _save_resumable_ckpt(cfg, 5)

    check = preflight.Checks()
    preflight.check_milestones(check, cfg, ckpt_dir)

    assert check.failed >= 1


def test_check_milestones_fails_on_empty_milestones_dir(tmp_path, capsys):
    """No milestones/ directory at all (the loop over `found` never runs) must not
    leave bf16_ok/load_ok at their initial True -- both must fail. Asserted on the
    printed PASS/FAIL lines for those two specific checks (not just the aggregate
    failure count), since the "milestone files exist" and pruning checks also fail
    here and would otherwise mask a bf16_ok/load_ok mutation that stayed True."""
    cfg = build_cfg(tmp_path, milestones=(2, 4), ckpt_keep=2)
    ckpt_dir = Path(cfg.train.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)  # ckpt_dir exists; milestones/ does not

    check = preflight.Checks()
    preflight.check_milestones(check, cfg, ckpt_dir)
    out = capsys.readouterr().out

    assert "FAIL  milestones are bf16" in out
    assert "FAIL  milestones load into Quipu(cfg.model) strict=True with tie intact" in out


def test_check_milestones_fails_when_nothing_was_pruned():
    """If ckpt_keep is generous enough that every milestone step's resumable
    checkpoint is still on disk, the pruning-survival check can't demonstrate
    anything -- it must fail rather than pass vacuously."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cfg = build_cfg(tmp_path, milestones=(2, 4), ckpt_keep=10)
        ckpt_dir = Path(cfg.train.ckpt_dir)
        for step in (2, 4, 5):
            _save_milestone(cfg, step, _fresh_model(cfg, step))
            _save_resumable_ckpt(cfg, step)

        check = preflight.Checks()
        preflight.check_milestones(check, cfg, ckpt_dir)

        assert check.failed >= 1


# ---------------------------------------------------------------------------
# write_preflight_config_toml
# ---------------------------------------------------------------------------


def test_write_preflight_config_toml_round_trips(tmp_path):
    from quipu.config import load_config

    cfg = build_cfg(tmp_path, milestones=(2, 4))
    toml_path = tmp_path / "config.toml"
    preflight.write_preflight_config_toml(cfg, toml_path)

    reloaded = load_config(toml_path)
    assert reloaded == cfg


def test_write_preflight_config_toml_round_trips_language_weight_tables(tmp_path):
    # quipu-moe's weight tables are dicts, with keys like "C++" that TOML must quote.
    from quipu.config import load_config

    cfg = load_config(Path(__file__).resolve().parent.parent / "configs" / "quipu-moe.toml")
    toml_path = tmp_path / "config.toml"
    preflight.write_preflight_config_toml(cfg, toml_path)

    reloaded = load_config(toml_path)
    assert reloaded == cfg
    assert reloaded.data.code_language_weights["C++"] == 0.01


# ---------------------------------------------------------------------------
# check_milestone_eval_output
# ---------------------------------------------------------------------------


def _write_metrics(out_dir: Path, checkpoints: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps({"checkpoints": checkpoints}), encoding="utf-8"
    )


def _write_samples(out_dir: Path, prompts: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"## Prompt: `{p}`\n" for p in prompts)
    (out_dir / "samples.md").write_text(body, encoding="utf-8")


def test_check_milestone_eval_output_passes_on_good_fixtures(tmp_path):
    out_dir = tmp_path / "out"
    labels = ["step_000002", "step_000004", "final"]
    _write_metrics(out_dir, [
        {"label": l, "step": i, "text_val_loss": 3.1, "code_val_loss": 2.5}
        for i, l in enumerate(labels)
    ])
    prompts = ["def fibonacci(n):", "The history of the printing press"]
    _write_samples(out_dir, prompts)

    check = preflight.Checks()
    preflight.check_milestone_eval_output(check, out_dir, labels, prompts)

    assert check.failed == 0


def test_check_milestone_eval_output_fails_on_missing_prompt(tmp_path):
    out_dir = tmp_path / "out"
    labels = ["step_000002", "final"]
    _write_metrics(out_dir, [
        {"label": l, "step": i, "text_val_loss": 3.1, "code_val_loss": 2.5}
        for i, l in enumerate(labels)
    ])
    # samples.md is missing one of the two fixed prompts.
    _write_samples(out_dir, ["def fibonacci(n):"])

    check = preflight.Checks()
    preflight.check_milestone_eval_output(
        check, out_dir, labels, ["def fibonacci(n):", "The history of the printing press"],
    )

    assert check.failed >= 1


def test_check_milestone_eval_output_fails_on_non_finite_loss(tmp_path):
    out_dir = tmp_path / "out"
    labels = ["final"]
    _write_metrics(out_dir, [
        {"label": "final", "step": 5, "text_val_loss": float("nan"), "code_val_loss": 2.5}
    ])
    _write_samples(out_dir, ["def fibonacci(n):"])

    check = preflight.Checks()
    preflight.check_milestone_eval_output(check, out_dir, labels, ["def fibonacci(n):"])

    assert check.failed >= 1


def test_check_milestone_eval_output_fails_on_empty_checkpoints_list(tmp_path):
    """An empty but well-formed `checkpoints: []` must not leave the finite-loss
    checks unexercised and passing -- there is nothing to have verified."""
    out_dir = tmp_path / "out"
    _write_metrics(out_dir, [])
    _write_samples(out_dir, ["def fibonacci(n):"])

    check = preflight.Checks()
    preflight.check_milestone_eval_output(check, out_dir, [], ["def fibonacci(n):"])

    assert check.failed >= 2, "both finite-loss checks must fail on zero checkpoints"


def test_check_milestone_eval_output_skips_finite_check_for_a_failed_checkpoint(tmp_path):
    """A checkpoint that failed to load/evaluate has an "error" entry and no loss
    fields at all (see milestone_eval.run); that must not itself fail the finite
    checks -- it's already reflected by the checkpoint-count/label check."""
    out_dir = tmp_path / "out"
    labels = ["step_000002", "final"]
    _write_metrics(out_dir, [
        {"label": "step_000002", "step": 2, "error": "RuntimeError: boom"},
        {"label": "final", "step": 5, "text_val_loss": 3.0, "code_val_loss": 2.0},
    ])
    _write_samples(out_dir, ["def fibonacci(n):"])

    check = preflight.Checks()
    preflight.check_milestone_eval_output(check, out_dir, labels, ["def fibonacci(n):"])

    assert check.failed == 0


# ---------------------------------------------------------------------------
# _print_guard_section (pure formatting, no subprocess)
# ---------------------------------------------------------------------------


def test_print_guard_section_extracts_start_guards_block(capsys):
    stdout = (
        "real config stuff\n"
        "start guards:\n"
        "  [ok] gpu-vram: 0.10 GB held by other processes (limit 1.5 GB)\n"
        "  [FAILED] power: on battery\n"
        "         fix: plug the laptop into mains power, or pass --force\n"
        "\n"
        "--dry-run: nothing started. Plan:\n"
    )
    preflight._print_guard_section(stdout, returncode=1)
    out = capsys.readouterr().out
    assert "start guards:" in out
    assert "gpu-vram" in out and "power" in out
    assert "(weekend.py --dry-run exit 1)" in out
    # Only the guard block onward is echoed, not the ignored "real config stuff" line.
    assert "real config stuff" not in out


def test_print_guard_section_handles_missing_block(capsys):
    preflight._print_guard_section("nothing useful here\n", returncode=1)
    out = capsys.readouterr().out
    assert "no 'start guards:' section" in out
    assert "nothing useful here" in out
