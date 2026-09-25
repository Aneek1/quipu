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
    # 3,000,000,000 / 524,288 = 5722 (floor); 5722 x 524,288 = 2,999,975,936
    assert cfg.train.steps == 5722
    assert cfg.train.total_tokens - cfg.train.steps * cfg.train.batch_tokens == 24_064


def test_derives_gradient_accumulation():
    cfg = load_config(CONFIG)
    assert cfg.train.grad_accum == 128   # micro_batch 4
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


def test_code_mix_fields_are_loaded():
    d = load_config(CONFIG).data
    assert d.code_dataset == "codeparrot/github-code-clean"
    assert d.code_share == 0.2
    assert d.html_cap == 0.1
    assert d.code_val_tokens == 5_000_000
    assert (d.code_heldout_first_file, d.code_files_total) == (840, 880)
    assert d.code_languages == ("Python", "JavaScript", "TypeScript", "HTML", "CSS",
                                "PHP", "Java", "GO", "SQL", "Shell", "Dockerfile")
    assert d.code_licenses == ("mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause",
                               "isc", "cc0-1.0", "unlicense")
    # Lists become tuples so the frozen config can't be mutated through them.
    assert isinstance(d.code_languages, tuple) and isinstance(d.code_licenses, tuple)


@pytest.mark.parametrize("value", [0, 0.0, 1, 1.0, -0.2, 1.5, True, "0.2"])
def test_code_share_must_be_strictly_between_0_and_1(value):
    with pytest.raises(ValueError, match="code_share"):
        load_config(CONFIG, overrides={"data": {"code_share": value}})


@pytest.mark.parametrize("value", [0, 0.0, -0.1, 1.01, False, "0.1"])
def test_html_cap_must_be_in_0_exclusive_1_inclusive(value):
    with pytest.raises(ValueError, match="html_cap"):
        load_config(CONFIG, overrides={"data": {"html_cap": value}})


def test_html_cap_of_one_means_uncapped_and_is_allowed():
    assert load_config(CONFIG, overrides={"data": {"html_cap": 1.0}}).data.html_cap == 1.0


@pytest.mark.parametrize("key", ["code_languages", "code_licenses"])
@pytest.mark.parametrize("value", [[], "Python", ["Python", ""], ["mit", 3], ["mit", "mit"]])
def test_code_filter_lists_must_be_non_empty_lists_of_strings(key, value):
    with pytest.raises(ValueError, match=key):
        load_config(CONFIG, overrides={"data": {key: value}})


@pytest.mark.parametrize("overrides, match", [
    ({"code_heldout_first_file": 880}, "code_heldout_first_file"),
    ({"code_heldout_first_file": 900}, "code_heldout_first_file"),
    ({"code_heldout_first_file": 0}, "code_heldout_first_file"),
    ({"code_files_total": 0}, "code_files_total"),
    ({"code_val_tokens": 0}, "code_val_tokens"),
    ({"code_val_tokens": 5e6}, "code_val_tokens"),
])
def test_held_out_range_and_code_val_budget_are_sane(overrides, match):
    with pytest.raises(ValueError, match=match):
        load_config(CONFIG, overrides={"data": overrides})


def test_a_missing_code_field_is_an_error(tmp_path):
    # Every run must state its mix; a missing field is not silently text-only.
    lines = CONFIG.read_text(encoding="utf-8").splitlines()
    cfg = tmp_path / "missing_code_share.toml"
    cfg.write_text("\n".join(l for l in lines if not l.startswith("code_share")),
                   encoding="utf-8")
    with pytest.raises(TypeError, match="code_share"):
        load_config(cfg)


def test_milestones_are_loaded_as_a_tuple():
    cfg = load_config(CONFIG)
    assert cfg.train.milestones == (100, 250, 500, 1000, 2000, 4000)
    assert all(m < cfg.train.steps for m in cfg.train.milestones)


def test_milestones_may_be_empty():
    assert load_config(CONFIG, overrides={"train": {"milestones": []}}).train.milestones == ()


@pytest.mark.parametrize(
    "milestones, match",
    [
        ([100, 100], "strictly increasing"),
        ([250, 100], "strictly increasing"),
        ([0, 100], "positive"),
        ([-5], "positive"),
        ([100.0], "ints"),
        ([True], "ints"),
        (["100"], "ints"),
        (100, "list"),
        ([100, 5722], "less than steps"),
        ([100, 9000], "less than steps"),
    ],
)
def test_bad_milestones_are_rejected(milestones, match):
    with pytest.raises(ValueError, match=match):
        load_config(CONFIG, overrides={"train": {"milestones": milestones}})
