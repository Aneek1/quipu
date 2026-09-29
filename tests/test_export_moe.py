"""M10: quipu-moe export (standalone loader, parity, int4) and the model card. CPU only."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from quipu import model_card
from quipu.config import load_config
from quipu.int4 import (dequantize_int4, dequantize_state, quantize_int4, quantize_state,
                        select_int4)
from tests.moe_fixtures import (fresh_model, save_final, save_milestone, tiny_moe,
                                write_eval_shards)

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


export_hf = _load("export_hf_m10", ROOT / "scripts" / "export_hf.py")


def _export(tmp_path, *, int4=True, chat=False, attnres=2, activation="situ_glu", **kw):
    cfg, tok = tiny_moe(tmp_path, attnres_blocks=attnres, activation=activation)
    write_eval_shards(cfg, tok, tokens=300)
    save_milestone(cfg, 10, fresh_model(cfg, 1))
    save_final(cfg, 20, fresh_model(cfg, 2))
    out = tmp_path / "hf"
    logs: list[str] = []
    e = export_hf.MoEExport(cfg=cfg, ckpt_dir=Path(cfg.train.ckpt_dir), out=out, int4=int4,
                            chat=chat, parity_tol=1e-5, bpb_batches=2,
                            results_dir=tmp_path / "results", log=logs.append,
                            card_inputs=model_card.CardInputs(), **kw)
    code = export_hf.export_moe(e)
    return code, cfg, tok, out, logs


# ---- int4 ---------------------------------------------------------------------------------

def test_int4_round_trip_is_within_half_a_step_of_every_value():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(3, 5, 200, generator=g) * 0.05          # 200: not a multiple of 128
    q, s, m = quantize_int4(w, 128)
    assert q.dtype == torch.uint8 and q.shape == (3, 5, 128) and s.shape == (3, 5, 2)
    back = dequantize_int4(q, s, m, w.shape, 128)
    step = s.float().repeat_interleave(128, -1)[..., :200]
    assert back.shape == w.shape
    assert ((back - w).abs() <= step / 2 + 1e-6 + w.abs() * 2 ** -10).all()


def test_int4_packs_known_codes_low_nibble_first():
    w = torch.arange(16.0).view(1, 16)                      # min 0, scale 1: codes 0..15
    q, s, m = quantize_int4(w, 16)
    assert float(s) == 1.0 and float(m) == 0.0
    assert q.tolist() == [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]]
    assert torch.equal(dequantize_int4(q, s, m, [1, 16], 16), w)


def test_int4_selects_the_expert_and_attention_matrices_only(tmp_path):
    cfg, _ = tiny_moe(tmp_path)
    state = fresh_model(cfg).state_dict()
    chosen = {k for k, v in state.items() if select_int4(k, v)}
    assert "blocks.0.moe.experts.gate" in chosen and "blocks.1.attn.q.weight" in chosen
    assert "blocks.0.moe.shared.0.down.weight" in chosen
    for k in ("embed.weight", "lm_head.weight", "blocks.0.moe.router.weight",
              "blocks.0.moe.balancer.bias", "norm.weight", "attnres.queries.0"):
        assert k not in chosen
    saved, meta = quantize_state(state)
    assert saved["blocks.0.moe.router.weight"].dtype == torch.float32
    assert saved["blocks.0.moe.balancer.bias"].dtype == torch.float32
    back = dequantize_state(saved, meta)
    assert set(back) == set(state) and back["blocks.0.moe.experts.up"].shape == state[
        "blocks.0.moe.experts.up"].shape


def test_the_standalone_dequant_equals_the_training_side_one():
    mq = _load("modeling_quipu_moe_dq", ROOT / "hf" / "modeling_quipu_moe.py")
    w = torch.randn(4, 300)
    q, s, m = quantize_int4(w)
    assert torch.equal(mq.dequantize_int4(q, s, m, list(w.shape), 128),
                       dequantize_int4(q, s, m, w.shape, 128))


# ---- export and parity --------------------------------------------------------------------

@pytest.mark.parametrize("attnres,activation", [(2, "situ_glu"), (0, "swiglu")])
def test_export_parity_fp32_bf16_milestone_and_int4(tmp_path, attnres, activation):
    code, cfg, _, out, logs = _export(tmp_path, attnres=attnres, activation=activation)
    assert code == 0, logs
    for f in ["config.json", "model.safetensors", "model-int4.safetensors", "tokenizer.json",
              "tokenizer_config.json", "special_tokens_map.json", "generation_config.json",
              "modeling_quipu_moe.py", "README.md", "milestones/step_000010.safetensors",
              "int4_report.json"]:
        assert (out / f).is_file(), f
    rep = json.loads((out / "int4_report.json").read_text(encoding="utf-8"))
    assert set(rep["parity"]) == {"final (fp32)", "milestone step_000010 (bf16)",
                                  "int4 (dequantized)"}
    assert all(v <= 1e-5 for v in rep["parity"].values()), rep["parity"]
    # int4 quality: measured per split, close to fp32 on a random tiny model.
    assert set(rep["bpb"]) == {"eng_Latn", "ind_Latn", "zho_Hans", "tam_Taml", "code"}
    for v in rep["bpb"].values():
        assert abs(v["int4"] / v["fp32"] - 1) < 0.05
    assert rep["int4_vs_fp32_max_logit_diff"] > 0
    assert rep["sizes"]["model-int4.safetensors"] < rep["sizes"]["model.safetensors"]
    assert (tmp_path / "results").glob("*-int4.json")
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["architecture"] == "quipu-moe" and config["active_params"] < config["total_params"]


def test_export_fails_when_the_standalone_model_disagrees(tmp_path, monkeypatch):
    real = export_hf._import_standalone

    def broken(path, name):
        mod = real(path, name)
        mod.glu = lambda g, u, cfg: torch.relu(g) * u   # wrong activation
        return mod

    monkeypatch.setattr(export_hf, "_import_standalone", broken)
    code, *_, logs = _export(tmp_path, int4=False)
    assert code == 1 and any("FAIL" in m for m in logs)


def test_standalone_loader_runs_without_quipu(tmp_path):
    code, cfg, tok, out, _ = _export(tmp_path, int4=True)
    assert code == 0
    x = export_hf.parity_input(cfg)
    ref = export_hf._logits(fresh_model(cfg, 2).eval(), x)
    torch.save(x, tmp_path / "x.pt")
    script = textwrap.dedent(f"""
        import importlib.abc, importlib.util, sys
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "quipu" or name.startswith("quipu."):
                    raise ImportError("quipu must not be imported")
        sys.meta_path.insert(0, Block())
        import torch
        spec = importlib.util.spec_from_file_location("mq", r"{out / 'modeling_quipu_moe.py'}")
        mq = importlib.util.module_from_spec(spec); sys.modules["mq"] = mq
        spec.loader.exec_module(mq)
        x = torch.load(r"{tmp_path / 'x.pt'}")
        m = mq.load(r"{out}")
        m4 = mq.load(r"{out}", weights="model-int4.safetensors")
        tok = mq.load_tokenizer(r"{out}")
        with torch.no_grad():
            torch.save({{"fp32": m(x), "int4": m4(x)}}, r"{tmp_path / 'y.pt'}")
        assert tok.decode(tok.encode("def area(w, h):")) == "def area(w, h):"
        print(repr(mq.generate(m, tok, "def area", max_new_tokens=3)))
        print(repr(mq.chat(m, tok, [{{"role": "user", "content": "hi"}}], max_new_tokens=3)))
        assert not any(k == "quipu" or k.startswith("quipu.") for k in sys.modules)
    """)
    r = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True,
                       cwd=tmp_path, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    y = torch.load(tmp_path / "y.pt")
    assert (y["fp32"] - ref).abs().max().item() <= 1e-5
    assert (y["int4"] - ref).abs().max().item() > 0      # int4 is not free


def test_chat_export_carries_the_chat_template(tmp_path):
    code, cfg, tok, out, _ = _export(tmp_path, int4=False, chat=True)
    assert code == 0
    tcfg = json.loads((out / "tokenizer_config.json").read_text(encoding="utf-8"))
    gen = json.loads((out / "generation_config.json").read_text(encoding="utf-8"))
    assert tcfg["eos_token"] == "<|end|>"
    assert gen["eos_token_id"] == [tok.special_id("<|end|>"), tok.eot]
    jinja2 = pytest.importorskip("jinja2")
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hai!"}]
    rendered = jinja2.Template(tcfg["chat_template"]).render(messages=msgs,
                                                             add_generation_prompt=True)
    mq = _load("modeling_quipu_moe_chat", out / "modeling_quipu_moe.py")
    assert rendered == mq.render_chat(msgs) == \
        "<|system|>Be brief.<|end|><|user|>Hai!<|end|><|assistant|>"
    card = (out / "README.md").read_text(encoding="utf-8")
    assert "chat model" in card and "base_model: AneekC/" in card


# ---- model card -------------------------------------------------------------------------

def test_release_name_of_the_full_config_is_1B_A149M():
    cfg = load_config(ROOT / "configs" / "quipu-moe.toml", {"data": {"tokenizer": "missing.json"}})
    total, active = model_card.count_params(cfg.model)
    assert model_card.release_name(total, active) == "quipu-moe-1B-A149M"
    assert "AneekC/" + model_card.release_name(total, active) == model_card.BASE_REPO


def test_card_with_missing_inputs_says_not_yet_measured(tmp_path):
    cfg, _ = tiny_moe(tmp_path)
    f = model_card.fields(cfg, model_card.CardInputs(run_dir=tmp_path / "nope"), variant="base")
    template = (ROOT / "hf" / "README_moe.md").read_text(encoding="utf-8")
    card = model_card.render(template, f)
    assert "{{" not in card
    for key in ("bpb", "code_eval", "int4", "experts", "samples", "ab_results", "decontamination"):
        assert f[key].startswith(model_card.NOT_MEASURED), key
    assert "| Tokens trained | _not yet measured_ |" in card
    assert "| Cost | _not yet measured_ |" in card
    assert "| Hardware | _not yet measured_ |" in card
    assert "total" in card and "active per token" in card
    assert "AneekC/lid-specialists-9plus1" in card          # the LID model is credited
    assert "FP8 training was not tried" in card
    assert "published, not re-run" in card                 # the reference table is shown


def test_card_is_filled_from_result_files(tmp_path):
    cfg, _ = tiny_moe(tmp_path)
    ms = tmp_path / "milestones"
    ms.mkdir()
    (ms / "metrics.json").write_text(json.dumps({"checkpoints": [
        {"label": "final", "step": 99, "bpb": {"eng_Latn": {"bpb": 1.234}, "tam_Taml": {"bpb": 0.5}}}]}))
    (ms / "samples.json").write_text(json.dumps({"Aaj mausam bahut": [
        {"label": "final", "step": 99, "greedy": "Aaj mausam bahut accha hai", "sampled": None}]}))
    ce = tmp_path / "ce"
    ce.mkdir()
    total, active = model_card.count_params(cfg.model)
    name = model_card.release_name(total, active)
    (ce / f"{name}.json").write_text(json.dumps({"results": {"humaneval": {
        "greedy_pass@1": 4.3, "pass@10": 9.1, "samples": 20}}, "benchmarks": {}}))
    ab = tmp_path / "ab"
    ab.mkdir()
    (ab / "summary.md").write_text("# A/B\n\n## Runs\n\n| run |\n|---|\n| muon |\n\n"
                                   "## Decisions\n\n| pair | kept |\n|---|---|\n| 1 optimizer | adamw |\n"
                                   "\n## winners.toml\n")
    run = tmp_path / "run"
    (run / "runs").mkdir(parents=True)
    (run / "runs" / "r.json").write_text(json.dumps({"steps": [{"step": 10, "tokens": 5_000_000}]}))
    (run / "plan.json").write_text(json.dumps({"tokens_per_s": 123456.0}))
    inp = model_card.CardInputs(run_dir=run, ab_dir=ab, milestones_dir=ms, code_eval_dir=ce,
                                hardware="1 x RTX 5090")
    card = model_card.render((ROOT / "hf" / "README_moe.md").read_text(encoding="utf-8"),
                             model_card.fields(cfg, inp, total=total, active=active))
    assert "| final | 99 | 1.234 | 0.500 |" in card
    assert "Aaj mausam bahut accha hai" in card
    assert f"| **{name}** |" in card and "| 4.3 | 9.1 |" in card
    assert "| 1 optimizer | adamw |" in card
    assert "5,000,000 (10 steps)" in card and "123,456 tokens/s" in card
    assert "1 x RTX 5090" in card
