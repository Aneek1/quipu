import pytest

from quipu.config import load_config


CONFIG = "configs/quipu-114m.toml"


def test_loads_the_shipped_config():
    cfg = load_config(CONFIG)
    assert cfg.name == "quipu-114m"
    assert cfg.model.d_model == 768
    assert cfg.model.n_layer == 12
    assert cfg.model.n_kv_head == 4


def test_derives_step_count_from_token_budget():
    cfg = load_config(CONFIG)
    # 2,500,000,000 / 524,288 = 4768 (floor)
    assert cfg.train.steps == 4768


def test_derives_gradient_accumulation():
    cfg = load_config(CONFIG)
    # 524,288 tokens per step / (micro_batch x context)
    expected = cfg.train.batch_tokens // (cfg.train.micro_batch * cfg.model.context)
    assert cfg.train.grad_accum == expected
    assert cfg.train.grad_accum >= 1


def test_rejects_a_batch_that_does_not_divide_evenly():
    # A batch_tokens that is not a multiple of micro_batch x context would silently
    # train on a different number of tokens than the config claims.
    with pytest.raises(ValueError, match="batch_tokens"):
        load_config(CONFIG, overrides={"train": {"batch_tokens": 524_289}})


def test_config_is_frozen():
    cfg = load_config(CONFIG)
    with pytest.raises(Exception):
        cfg.model.d_model = 1024
