import dataclasses
from pathlib import Path

import pytest

from quipu.config import load_config


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "quipu-114m.toml"


def test_loads_the_shipped_config():
    cfg = load_config(CONFIG)
    assert cfg.name == "quipu-114m"
    assert cfg.model.d_model == 768
    assert cfg.model.n_layer == 12
    assert cfg.model.n_kv_head == 4


def test_derives_step_count_from_token_budget():
    cfg = load_config(CONFIG)
    # 1,500,000,000 / 524,288 = 2861 (floor)
    assert cfg.train.steps == 2861


def test_derives_gradient_accumulation():
    cfg = load_config(CONFIG)
    assert cfg.train.grad_accum == 64
    cfg2 = load_config(CONFIG, overrides={"train": {"micro_batch": 16}})
    assert cfg2.train.grad_accum == 32
    assert cfg.train.context == cfg.model.context


def test_rejects_a_batch_that_does_not_divide_evenly():
    # A batch_tokens that is not a multiple of micro_batch x context would silently
    # train on a different number of tokens than the config claims.
    with pytest.raises(ValueError, match="batch_tokens"):
        load_config(CONFIG, overrides={"train": {"batch_tokens": 524_289}})


@pytest.mark.parametrize(
    "get_target, attr",
    [
        (lambda cfg: cfg.model, "d_model"),
        (lambda cfg: cfg.data, "dataset"),
        (lambda cfg: cfg.train, "lr"),
        (lambda cfg: cfg, "name"),
    ],
)
def test_config_is_frozen(get_target, attr):
    cfg = load_config(CONFIG)
    target = get_target(cfg)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(target, attr, "mutated")


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"model": {"n_head": 10}}, "d_model"),
        ({"model": {"n_kv_head": 5}}, "n_kv_head"),
        ({"model": {"d_model": 780, "n_head": 12}}, "head_dim"),
    ],
)
def test_model_guards(overrides, match):
    # Each override should only disturb the field it names, and the resulting
    # error must name the guard it actually trips (not a different guard tripped
    # first by the same override).
    with pytest.raises(ValueError, match=match):
        load_config(CONFIG, overrides=overrides)


def test_train_context_belongs_to_model():
    with pytest.raises(ValueError, match="context"):
        load_config(CONFIG, overrides={"train": {"context": 2048}})


def test_rejects_unknown_top_level_key():
    with pytest.raises(ValueError, match="trian"):
        load_config(CONFIG, overrides={"trian": {}})


def test_rejects_a_non_int_value_for_an_int_field():
    with pytest.raises(ValueError, match="micro_batch"):
        load_config(CONFIG, overrides={"train": {"micro_batch": 8.0}})


def test_rejects_zero_micro_batch_without_a_zero_division_error():
    # micro_batch feeds a modulo below (batch_tokens % (micro_batch * context)); the
    # positivity check must run before that division, not surface it as a crash.
    with pytest.raises(ValueError, match="micro_batch"):
        load_config(CONFIG, overrides={"train": {"micro_batch": 0}})


def test_rejects_a_run_with_zero_steps():
    # A run this short exits at step 0 and "succeeds" having trained nothing.
    with pytest.raises(ValueError, match="steps"):
        load_config(CONFIG, overrides={"train": {"total_tokens": 100}})


def test_rejects_warmup_longer_than_the_run():
    with pytest.raises(ValueError, match="warmup"):
        load_config(CONFIG, overrides={"train": {"warmup_steps": 10_000}})


def test_rejects_non_positive_lr():
    # match is anchored to "lr must be positive" (not just "lr") so this test can't
    # be satisfied by the lr_min guard firing instead of the lr guard.
    with pytest.raises(ValueError, match=r"^lr must be positive"):
        load_config(CONFIG, overrides={"train": {"lr": 0.0}})


def test_rejects_lr_min_above_lr():
    with pytest.raises(ValueError, match="lr_min"):
        load_config(CONFIG, overrides={"train": {"lr_min": 1e-3}})


def test_rejects_negative_lr_min():
    with pytest.raises(ValueError, match="lr_min"):
        load_config(CONFIG, overrides={"train": {"lr_min": -1e-5}})


def test_rejects_a_bool_for_an_int_field():
    # bool is an int subclass; without an explicit check, True would silently
    # become micro_batch=1.
    with pytest.raises(ValueError, match="micro_batch"):
        load_config(CONFIG, overrides={"train": {"micro_batch": True}})


def test_seed_zero_is_a_legitimate_seed():
    cfg = load_config(CONFIG, overrides={"train": {"seed": 0}})
    assert cfg.train.seed == 0


def test_ckpt_keep_is_loaded_and_must_be_positive():
    # ckpt_keep=0 would let retention delete every checkpoint it just wrote.
    assert load_config(CONFIG).train.ckpt_keep == 3
    with pytest.raises(ValueError, match="ckpt_keep"):
        load_config(CONFIG, overrides={"train": {"ckpt_keep": 0}})
