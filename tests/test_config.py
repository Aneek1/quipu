import dataclasses
from pathlib import Path

import pytest

from quipu.config import load_config


CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CONFIG = CONFIGS / "quipu-114m.toml"
MOE = CONFIGS / "quipu-moe.toml"
MOE_AB = CONFIGS / "quipu-moe-ab.toml"
MOE_SMOKE = CONFIGS / "quipu-moe-smoke.toml"


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
    assert d.code_max_doc_tokens == 16_000
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


@pytest.mark.parametrize("value", [0, -1, 16_000.0, True, "16000"])
def test_code_max_doc_tokens_must_be_a_positive_int(value):
    with pytest.raises(ValueError, match="code_max_doc_tokens"):
        load_config(CONFIG, overrides={"data": {"code_max_doc_tokens": value}})


def test_code_max_doc_tokens_defaults_to_16000(tmp_path):
    lines = CONFIG.read_text(encoding="utf-8").splitlines()
    cfg = tmp_path / "no_max_doc.toml"
    cfg.write_text("\n".join(l for l in lines if not l.startswith("code_max_doc_tokens")),
                   encoding="utf-8")
    assert load_config(cfg).data.code_max_doc_tokens == 16_000


# ---------------------------------------------------------------------------
# quipu-moe settings (Task M2)
# ---------------------------------------------------------------------------


def test_quipu_114m_gets_the_dense_gpt2_adamw_defaults():
    # quipu-114m's file names none of the new fields; every one must default to
    # what quipu-114m already was, so nothing about that run changes.
    cfg = load_config(CONFIG)
    m, d, t = cfg.model, cfg.data, cfg.train
    assert (m.kind, m.n_experts, m.top_k, m.expert_hidden, m.shared_experts,
            m.shared_hidden, m.attnres_blocks) == ("dense", 0, 0, 0, 0, 0, 0)
    assert (m.activation, m.situ_beta_gate, m.situ_beta_up, m.balance_update_rate) == (
        "swiglu", 4.0, 25.0, 0.3)
    assert (m.moe_dispatch, m.capacity_factor) == ("loop", 1.5)
    assert m.attnres_checkpoint is True
    assert d.tokenizer == "gpt2"
    assert d.code_language_weights == {} and d.text_language_weights == {}
    assert (d.lid_model, d.lid_revision) == ("AneekC/lid-specialists-9plus1", "")
    assert (t.optimizer, t.muon_lr, t.muon_momentum, t.muon_ns_steps, t.muon_per_head,
            t.compile, t.budget_usd, t.usd_per_hour) == (
        "adamw", 0.02, 0.95, 5, True, False, 0.0, 0.0)
    # The shipped numbers themselves are untouched.
    assert (m.vocab_size, m.ffn_hidden, m.context) == (50257, 2048, 1024)
    assert (t.steps, t.grad_accum) == (5722, 128)


def test_quipu_moe_config_matches_the_spec():
    cfg = load_config(MOE)
    m, d, t = cfg.model, cfg.data, cfg.train
    assert cfg.name == "quipu-moe"
    assert (m.kind, m.vocab_size, m.d_model, m.n_layer, m.n_head, m.n_kv_head,
            m.context) == ("moe", 49152, 768, 16, 12, 4, 2048)
    assert (m.n_experts, m.top_k, m.expert_hidden, m.shared_experts, m.shared_hidden) == (
        64, 4, 384, 1, 768)
    assert d.tokenizer == "artifacts/tokenizer/tokenizer.json"
    assert d.code_share == 0.6
    assert d.code_language_weights["Python"] == 0.30
    assert d.code_language_weights["JavaScript"] == 0.25
    assert d.code_language_weights["TypeScript"] == 0.12
    assert d.code_language_weights["HTML"] == 0.08 == d.html_cap
    assert (d.code_language_weights["CSS"], d.code_language_weights["SQL"]) == (0.05, 0.05)
    others = set(d.code_language_weights) - {"Python", "JavaScript", "TypeScript", "HTML",
                                             "CSS", "SQL"}
    assert sum(d.code_language_weights[k] for k in others) == pytest.approx(0.15)
    # Of all tokens: 60% code, 28% English, 12% over the nine FineWeb-2 subsets.
    text = d.text_language_weights
    assert set(text) == {"eng_Latn", "ind_Latn", "zsm_Latn", "cmn_Hani", "jpn_Jpan",
                         "kor_Hang", "tam_Taml", "hin_Deva", "hin_Latn", "urd_Latn"}
    assert (1 - d.code_share) * text["eng_Latn"] == pytest.approx(0.28)
    for lang in set(text) - {"eng_Latn"}:
        assert (1 - d.code_share) * text[lang] == pytest.approx(0.12 / 9)
    assert (t.total_tokens, t.batch_tokens, t.budget_usd) == (10_200_000_000, 524_288, 20.0)
    assert t.usd_per_hour > 0


def test_quipu_moe_ab_is_the_full_config_at_8_layers_and_200m_tokens():
    full, ab = load_config(MOE), load_config(MOE_AB)
    assert ab.model == dataclasses.replace(full.model, n_layer=8)
    assert ab.data == full.data
    assert ab.train.total_tokens == 200_000_000
    # Only the documented train differences: the A/B orchestrator guards spend.
    assert ab.train == dataclasses.replace(
        full.train, total_tokens=200_000_000, warmup_steps=40,
        ckpt_dir="checkpoints-moe-ab", ckpt_every=100, ckpt_keep=1, eval_every=50,
        milestones=(), budget_usd=0.0)


def test_quipu_moe_smoke_is_tiny_and_switches_every_new_path_on():
    m, d, t = (lambda c: (c.model, c.data, c.train))(load_config(MOE_SMOKE))
    assert (m.kind, m.n_layer, m.d_model, m.n_experts, m.top_k, m.context) == (
        "moe", 2, 128, 8, 2, 256)
    assert m.vocab_size == 512
    assert (m.activation, m.attnres_blocks, t.optimizer) == ("situ_glu", 2, "muon")
    assert t.total_tokens == 2_000_000
    assert d.tokenizer != "gpt2"


@pytest.mark.parametrize("path", [MOE, MOE_AB, MOE_SMOKE])
def test_moe_configs_have_weights_summing_to_one(path):
    d = load_config(path).data
    assert sum(d.code_language_weights.values()) == pytest.approx(1.0, abs=1e-9)
    assert sum(d.text_language_weights.values()) == pytest.approx(1.0, abs=1e-9)
    assert set(d.code_language_weights) == set(d.code_languages)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"kind": "sparse"}, "kind"),
        ({"activation": "gelu"}, "activation"),
        ({"n_experts": 4, "top_k": 0}, "top_k"),
        ({"n_experts": 4, "top_k": 1}, "top_k"),
        ({"moe_dispatch": "scatter"}, "moe_dispatch"),
        ({"capacity_factor": 0.0}, "capacity_factor"),
        ({"capacity_factor": 0.9}, "capacity_factor"),
        ({"n_experts": 2, "top_k": 3}, "n_experts"),
        ({"expert_hidden": 0}, "expert_hidden"),
        ({"shared_experts": -1}, "shared_experts"),
        ({"shared_experts": 1, "shared_hidden": 0}, "shared_hidden"),
        ({"shared_experts": 0, "shared_hidden": 128}, "shared_experts"),
        ({"attnres_blocks": 3}, "attnres_blocks"),
        ({"attnres_checkpoint": 1}, "attnres_checkpoint"),
        ({"attnres_checkpoint": "yes"}, "attnres_checkpoint"),
        ({"top_k": 2.0}, "top_k"),
        ({"situ_beta_gate": 0.0}, "situ_beta_gate"),
        ({"situ_beta_up": -1.0}, "situ_beta_up"),
        ({"balance_update_rate": 0.0}, "balance_update_rate"),
        ({"balance_update_rate": 1.5}, "balance_update_rate"),
    ],
)
def test_moe_model_guards(overrides, match):
    with pytest.raises(ValueError, match=match):
        load_config(MOE_SMOKE, overrides={"model": overrides})


@pytest.mark.parametrize("field", ["n_experts", "top_k", "expert_hidden", "shared_experts",
                                   "shared_hidden", "attnres_blocks"])
def test_dense_refuses_moe_settings_it_would_ignore(field):
    with pytest.raises(ValueError, match=field):
        load_config(CONFIG, overrides={"model": {field: 1}})


def test_dense_refuses_padded_dispatch():
    with pytest.raises(ValueError, match="moe_dispatch"):
        load_config(CONFIG, overrides={"model": {"moe_dispatch": "padded"}})


def test_moe_configs_balance_at_0_3_and_accept_padded_dispatch():
    for path in (MOE, MOE_AB, MOE_SMOKE):
        assert load_config(path).model.balance_update_rate == 0.3
    m = load_config(MOE_SMOKE, overrides={"model": {"moe_dispatch": "padded",
                                                    "capacity_factor": 2.0}}).model
    assert (m.moe_dispatch, m.capacity_factor) == ("padded", 2.0)


def test_dense_refuses_situ_glu():
    with pytest.raises(ValueError, match="activation"):
        load_config(CONFIG, overrides={"model": {"activation": "situ_glu"}})


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"optimizer": "sgd"}, "optimizer"),
        ({"muon_lr": 0.0}, "muon_lr"),
        ({"muon_momentum": 1.0}, "muon_momentum"),
        ({"muon_momentum": -0.1}, "muon_momentum"),
        ({"muon_ns_steps": 0}, "muon_ns_steps"),
        ({"muon_per_head": 1}, "muon_per_head"),
        ({"compile": "yes"}, "compile"),
        ({"budget_usd": -1.0}, "budget_usd"),
        ({"usd_per_hour": -0.5}, "usd_per_hour"),
        ({"budget_usd": 20.0, "usd_per_hour": 0.0}, "usd_per_hour"),
    ],
)
def test_optimizer_and_budget_guards(overrides, match):
    with pytest.raises(ValueError, match=match):
        load_config(MOE_SMOKE, overrides={"train": overrides})


def _weights(key="code_language_weights"):
    return dict(getattr(load_config(MOE_SMOKE).data, key))


def _load_mix(**data):
    """The smoke config with these [data] overrides; weight tables replace whole."""
    return load_config(MOE_SMOKE, overrides={"data": data})


def test_a_weight_table_override_replaces_the_whole_table():
    cfg = load_config(MOE_SMOKE, overrides={"data": {
        "text_language_weights": {"eng_Latn": 1.0}}})
    assert cfg.data.text_language_weights == {"eng_Latn": 1.0}


def test_a_partial_code_weight_override_replaces_the_table_and_is_validated():
    # Not merged into the file's table: only Python is left, which then fails the
    # sum and the must-cover-code_languages rules instead of silently keeping the rest.
    with pytest.raises(ValueError, match="sum to 1"):
        load_config(MOE_SMOKE, overrides={"data": {"code_language_weights": {"Python": 0.35}}})
    with pytest.raises(ValueError, match="code_languages"):
        load_config(MOE_SMOKE, overrides={"data": {"code_language_weights": {"Python": 1.0}}})
    ok = load_config(MOE_SMOKE, overrides={"data": {
        "code_languages": ["Python", "Rust"],
        "code_language_weights": {"Python": 0.75, "Rust": 0.25}}})
    assert ok.data.code_language_weights == {"Python": 0.75, "Rust": 0.25}


def test_other_sections_still_merge_key_by_key():
    cfg = load_config(MOE_SMOKE, overrides={"model": {"top_k": 3}})
    assert (cfg.model.top_k, cfg.model.n_experts) == (3, 8)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda w: w.update(Python=0.31), "sum to 1"),               # sums to 1.01
        (lambda w: w.pop("Rust"), "sum to 1"),                       # sums to 0.99
        (lambda w: w.update(Python=0.0, JavaScript=0.55), "Python"),  # zero weight
        (lambda w: w.update(Python="0.3"), "Python"),
        (lambda w: w.update(Python=True), "Python"),
    ],
)
def test_code_language_weight_values_are_checked(mutate, match):
    w = _weights()
    mutate(w)
    with pytest.raises(ValueError, match=match):
        _load_mix(code_language_weights=w)


def test_the_unchanged_weight_tables_load_through_the_helper():
    cfg = _load_mix(code_language_weights=_weights(),
                    text_language_weights=_weights("text_language_weights"))
    assert cfg.data.code_language_weights == load_config(MOE).data.code_language_weights


def test_code_language_weights_must_cover_exactly_code_languages():
    w = _weights()
    w["Python"] -= 0.01
    w["Kotlin"] = 0.01           # not in code_languages: would never be read
    with pytest.raises(ValueError, match="code_languages"):
        _load_mix(code_language_weights=w)
    with pytest.raises(ValueError, match="code_languages"):   # Rust listed, not weighted
        _load_mix(code_language_weights={"Python": 1.0})


def test_html_weight_may_not_exceed_html_cap():
    with pytest.raises(ValueError, match="html_cap"):
        load_config(MOE_SMOKE, overrides={"data": {"html_cap": 0.05}})


def test_weights_within_tolerance_of_one_are_accepted():
    w = _weights()
    w["Python"] += 5e-7
    cfg = _load_mix(code_language_weights=w)
    assert cfg.data.code_language_weights["Python"] == pytest.approx(0.3 + 5e-7)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda w: w.update(eng_Latn=0.71), "sum to 1"),
        (lambda w: w.pop("urd_Latn"), "sum to 1"),
        (lambda w: w.update(eng_Latn=-0.1, ind_Latn=0.8333333333333333), "eng_Latn"),
        (lambda w: w.update(en=w.pop("eng_Latn")), "eng_Latn"),     # English renamed
    ],
)
def test_text_language_weights_are_checked(mutate, match):
    w = _weights("text_language_weights")
    mutate(w)
    with pytest.raises(ValueError, match=match):
        _load_mix(text_language_weights=w)


def test_language_weights_must_be_a_table():
    with pytest.raises(ValueError, match="text_language_weights"):
        load_config(CONFIG, overrides={"data": {"text_language_weights": ["eng_Latn"]}})


def test_weights_are_copied_not_shared_with_the_overrides():
    w = _weights()
    cfg = _load_mix(code_language_weights=w)
    w["Python"] = 0.9
    assert cfg.data.code_language_weights["Python"] == 0.30


@pytest.mark.parametrize("overrides, match", [
    ({"lid_model": ""}, "lid_model"),
    ({"lid_model": 3}, "lid_model"),
    ({"lid_revision": None}, "lid_revision"),
    ({"tokenizer": ""}, "tokenizer"),
])
def test_lid_and_tokenizer_strings_are_checked(overrides, match):
    with pytest.raises(ValueError, match=match):
        load_config(MOE_SMOKE, overrides={"data": overrides})


def test_a_tokenizer_path_that_does_not_exist_yet_is_not_checked(tmp_path):
    missing = tmp_path / "nope" / "tokenizer.json"
    cfg = load_config(MOE_SMOKE, overrides={"data": {"tokenizer": str(missing)}})
    assert cfg.data.tokenizer == str(missing)


def test_vocab_size_must_match_an_existing_tokenizer(tmp_path):
    from quipu.bpe import train_bpe

    text = ("def area(w, h):\n    return w * h\n" * 30
            + "The quipu was a recording device made of knotted cords.\n" * 30)
    tok = train_bpe([text] * 20, 512, tmp_path / "tokenizer.json")
    path = str(tmp_path / "tokenizer.json")
    ok = load_config(MOE_SMOKE, overrides={
        "model": {"vocab_size": tok.vocab_size}, "data": {"tokenizer": path}})
    assert ok.model.vocab_size == tok.vocab_size
    with pytest.raises(ValueError, match="vocab_size"):
        load_config(MOE_SMOKE, overrides={
            "model": {"vocab_size": tok.vocab_size + 1}, "data": {"tokenizer": path}})


def test_gpt2_tokenizer_is_not_checked_against_vocab_size():
    # quipu-114m's own tests build tiny-vocab gpt2 configs; "gpt2" is not a file.
    assert load_config(CONFIG, overrides={"model": {"vocab_size": 1000}}).data.tokenizer == "gpt2"
