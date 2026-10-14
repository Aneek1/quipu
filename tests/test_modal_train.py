"""scripts/remote/modal_train.py without Modal and without a GPU.

A stub `modal` module records how the app is declared and fails the test if anything
is launched; the budget guard's parsing and arithmetic run on fixture text; the smoke's
checks run on fixture rank records; and the two in-container procedures (the smoke's
two launches, the monthly run with its first start / resume / decay branch) run for
real on the CPU with a tiny model, two gloo processes started without torchrun (this
torch's torchrun cannot start its rendezvous store on Windows)."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import math
import sys
import tomllib
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts" / "remote" / "modal_train.py"
TOKENIZER = ROOT / "artifacts" / "tokenizer" / "tokenizer.json"
RUN_CONFIG = ROOT / "remote-runs" / "quipu-moe" / "results" / "moe" / "run_config.toml"

SUMMARY_TABLE = """\
+------------------------------------+
| Metered Cost:             |  25.04 |
|   Ephemeral Apps:         |  24.75 |
|   Deployed Apps:          |   0.28 |
| Credits:                  | -25.00 |
| Network Egress Allowance: |  -0.04 |
| Billed Cost:              |  $0.00 |
+------------------------------------+
"""
SUMMARY_JSON = """{
  "metered_cost": "25.03546599",
  "billed_cost": "0E-8",
  "adjustments": {"credits": "-25.00000000"},
  "metered_cost_breakdown": {"ephemeral_apps": "24.75277966", "deployed_apps": "0.28497620"}
}"""


# ---- loading the script (with or without a stub modal) ---------------------------------

def fake_modal():
    m = types.ModuleType("modal")
    m.declared, m.volumes, m.launched = {}, {}, []

    class Fn:
        def __init__(self, fn, kw):
            self.fn, self.kw = fn, kw

        def with_options(self, **kw):
            m.launched.append(("with_options", self.fn.__name__, kw))
            return self

        def remote(self, *a, **k):
            m.launched.append(("remote", self.fn.__name__, a))
            raise AssertionError("a test launched a Modal function")

    class App:
        def __init__(self, name, **kw):
            self.name = name

        def function(self, **kw):
            def deco(fn):
                f = Fn(fn, kw)
                m.declared[fn.__name__] = f
                return f
            return deco

        @contextlib.contextmanager
        def run(self):
            m.launched.append(("app.run",))
            raise AssertionError("a test started the Modal app")
            yield

    class Image:
        files: list = []
        dirs: list = []

        @classmethod
        def debian_slim(cls, **kw):
            cls.python = kw.get("python_version")
            return cls()

        def apt_install(self, *a):
            return self

        def pip_install(self, *a):
            Image.pip = a
            return self

        def env(self, d):
            return self

        def add_local_dir(self, src, dst, **k):
            assert Path(src).is_dir(), src
            Image.dirs.append(dst)
            return self

        def add_local_file(self, src, dst, **k):
            assert Path(src).exists(), src
            Image.files.append(dst)
            return self

    class Volume:
        @staticmethod
        def from_name(name, create_if_missing=False):
            v = types.SimpleNamespace(name=name, create=create_if_missing)
            m.volumes[name] = v
            return v

    m.App, m.Image, m.Volume = App, Image, Volume
    m.is_local = lambda: True
    m.enable_output = contextlib.nullcontext
    return m


def load(monkeypatch, modal_module=None):
    monkeypatch.setitem(sys.modules, "modal", modal_module)    # None: import -> ImportError
    spec = importlib.util.spec_from_file_location("modal_train_under_test", MODULE)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)           # dataclasses need it
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mt(monkeypatch):
    return load(monkeypatch)


# ---- the budget guard -------------------------------------------------------------------

def test_the_metered_cost_is_read_from_the_json_and_from_the_table(mt):
    assert mt.parse_metered_cost(SUMMARY_JSON) == pytest.approx(25.03546599)
    assert mt.parse_metered_cost(SUMMARY_TABLE) == pytest.approx(25.04)
    assert mt.parse_metered_cost("| Metered Cost: | $1,234.50 |") == pytest.approx(1234.5)


@pytest.mark.parametrize("text", ["", "{}", "| Billed Cost: | $0.00 |", "garbage",
                                  '{"metered_cost": "nan"}', '{"metered_cost": "-1"}'])
def test_a_summary_without_a_metered_cost_is_an_error_not_a_guess(mt, text):
    with pytest.raises(ValueError):
        mt.parse_metered_cost(text)


def test_read_metered_cost_asks_the_modal_cli_for_json(mt):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout=SUMMARY_JSON, stderr="")
    assert mt.read_metered_cost(run) == pytest.approx(25.035, abs=1e-3)
    assert calls[0][1:] == ["-m", "modal", "billing", "summary", "--json"]
    fail = lambda cmd, **kw: types.SimpleNamespace(returncode=1, stdout="", stderr="no token")
    with pytest.raises(RuntimeError, match="no token"):
        mt.read_metered_cost(fail)


def test_the_guard_refuses_only_above_the_cap(mt):
    assert mt.check_budget(25.0, 2.0)[0] is True                 # exactly $27: allowed
    ok, msg = mt.check_budget(25.04, 2.0)
    assert ok is False and "REFUSED" in msg and "0.04" in msg
    ok, msg = mt.check_budget(10.0, 5.0, cap=20.0)
    assert ok and "$15.00" in msg


def test_prices_and_the_smoke_cost(mt):
    assert mt.GPU_USD_PER_HOUR == {"B200": 6.25, "H100": 3.95, "L40S": 1.95, "L4": 0.80}
    cpu, mem = mt.resources(2)
    assert (cpu, mem) == (10.0, 64 * 1024)
    rate = mt.usd_per_hour("L40S", 2, cpu, mem)
    assert rate == pytest.approx(2 * 1.95 + 10 * 0.0473 + 64 * 0.008)
    p = mt.make_plan(mt.parse_args(["--smoke"]))
    assert p.worst_case_usd == pytest.approx(rate * 0.25)        # the 15 min timeout
    assert p.worst_case_usd < 1.25
    with pytest.raises(ValueError, match="no price"):
        mt.usd_per_hour("A100", 1, 1, 1024)


# ---- the plan ---------------------------------------------------------------------------

def test_smoke_defaults(mt):
    p = mt.make_plan(mt.parse_args(["--smoke"]))
    assert (p.mode, p.gpu_type, p.gpus, p.gpu_spec) == ("smoke", "L40S", 2, "L40S:2")
    assert (p.dispatch, p.micro_batch, p.compile) == ("loop", 2, False)
    assert p.timeout_s == 15 * 60 and (p.steps1, p.steps2) == (6, 4)
    p = mt.make_plan(mt.parse_args(["--smoke", "--gpu-type", "H100", "--gpus", "4",
                                    "--dispatch", "grouped", "--compile"]))
    assert (p.gpu_spec, p.dispatch, p.compile) == ("H100:4", "grouped", True)


def test_run_defaults_and_timeout(mt):
    p = mt.make_plan(mt.parse_args([]))
    assert (p.mode, p.gpu_spec, p.dispatch, p.micro_batch, p.compile) == (
        "run", "B200", "grouped", 32, True)
    assert p.hours == 3.75 and p.timeout_s == int(3.75 * 3600 + 20 * 60)
    assert p.trainer_budget_usd == pytest.approx(3.75 * p.usd_per_hour, abs=1e-3)
    d = mt.make_plan(mt.parse_args(["--gpus", "8", "--decay-steps", "600", "--hours", "1"]))
    assert (d.mode, d.gpu_spec, d.decay_steps) == ("decay", "B200:8", 600)


@pytest.mark.parametrize("argv, match", [
    (["--gpus", "3"], "whole multiple"),                  # grad_accum 8 over 3 ranks
    (["--gpus", "9"], "1..8"),
    (["--gpu-type", "A100"], "--gpu-type"),
    (["--smoke", "--decay-steps", "5"], "run-mode"),
    (["--hours", "0"], "--hours"),
    (["--dispatch", "fast"], "--dispatch"),
])
def test_bad_plans_are_refused(mt, argv, match):
    with pytest.raises(ValueError, match=match):
        mt.make_plan(mt.parse_args(argv))


def test_the_constants_match_the_configs_and_the_tokenizer(mt):
    from quipu.config import load_config

    cpt = load_config(ROOT / mt.CONFIG)
    assert (cpt.model.context, cpt.train.batch_tokens) == (mt.CONTEXT, mt.BATCH_TOKENS)
    pool_cfg = tomllib.loads((ROOT / "configs" / "data-v3.toml").read_text(encoding="utf-8"))
    assert pool_cfg["pool"]["tokenizer_sha256"] == mt.TOKENIZER_SHA256
    assert cpt.data.tokenizer == mt.TOKENIZER
    if TOKENIZER.exists():
        assert hashlib.sha256(TOKENIZER.read_bytes()).hexdigest() == mt.TOKENIZER_SHA256


@pytest.mark.skipif(not RUN_CONFIG.exists(), reason="the pretrain's run config is not here")
def test_the_cpt_config_is_the_published_model_with_its_optimizer():
    """results/ab/winners.toml says situ_glu, the published model is swiglu, and a
    strict state_dict load cannot tell them apart: the cpt config must not pick the
    winners up."""
    import dataclasses

    from quipu.config import load_config

    cpt = load_config(ROOT / "configs" / "quipu-moe-cpt.toml")
    run = load_config(RUN_CONFIG)
    assert cpt.model == run.model and cpt.model.activation == "swiglu"
    for f in ("optimizer", "muon_momentum", "muon_ns_steps", "muon_per_head",
              "muon_weight_decay", "weight_decay", "beta1", "beta2", "grad_clip",
              "precision", "batch_tokens"):
        assert getattr(cpt.train, f) == getattr(run.train, f), f
    assert cpt.train.warmup_start_lr == run.train.lr_min      # where the cosine ended
    assert dataclasses.asdict(cpt.data)["tokenizer"] == run.data.tokenizer


# ---- the app and the local driver (stub modal: nothing may launch) -----------------------

def test_the_app_declares_the_smoke_and_the_run_on_the_right_volumes(monkeypatch):
    fm = fake_modal()
    mod = load(monkeypatch, fm)
    s, t = fm.declared["smoke"].kw, fm.declared["train"].kw
    assert s["timeout"] == 15 * 60 and s["gpu"] == "L40S:2"
    assert set(s["volumes"]) == {"/pool", "/shards"}
    assert set(t["volumes"]) == {"/pool", "/shards", "/ckpt"}
    assert fm.volumes["quipu-pool"].create is False             # never create the pool
    assert fm.volumes["quipu-ckpt"].create is True
    assert {"/src/quipu"} <= set(fm.Image.dirs)
    for f in ("configs/quipu-moe-cpt.toml", "configs/quipu-moe.toml",
              "artifacts/tokenizer/tokenizer.json"):
        assert f"/src/{f}" in fm.Image.files
    assert fm.Image.python == "3.13" and any(p.startswith("torch==") for p in fm.Image.pip)
    assert fm.launched == []


def test_plan_only_prints_the_plan_and_launches_nothing(monkeypatch, capsys):
    fm = fake_modal()
    mod = load(monkeypatch, fm)
    rc = mod.main(["--smoke", "--plan-only"], metered=lambda: 25.04)
    out = capsys.readouterr().out
    assert rc == 0 and fm.launched == []
    assert "2 x L40S" in out and "budget guard" in out and "-> OK" in out
    assert "torch.distributed.run --standalone --nproc_per_node=2 -m quipu.rankrun" in out
    assert "--resume" in out and "plan only" in out


def test_over_the_cap_nothing_launches(monkeypatch, capsys):
    fm = fake_modal()
    mod = load(monkeypatch, fm)
    monkeypatch.setattr(mod, "volume_state", lambda: "stub")
    assert mod.main([], metered=lambda: 25.04) == 3              # a B200 run on top: refused
    assert "REFUSED" in capsys.readouterr().out and fm.launched == []
    # plan-only still answers (and still launches nothing)
    assert mod.main(["--plan-only"], metered=lambda: 25.04) == 0
    assert fm.launched == []


def test_an_unreadable_bill_refuses(monkeypatch, capsys):
    fm = fake_modal()
    mod = load(monkeypatch, fm)

    def broken():
        raise RuntimeError("offline")
    assert mod.main(["--smoke"], metered=broken) == 3
    assert "offline" in capsys.readouterr().err and fm.launched == []


def test_here_runs_the_same_procedures_locally_without_the_billing_guard(mt, monkeypatch,
                                                                      tmp_path):
    calls = {}
    monkeypatch.setattr(mt, "run_smoke", lambda *a, **k: calls.setdefault("smoke", k)
                        and {"passed": True})
    import quipu.shard_view as sv
    monkeypatch.setattr(sv, "make_shard_view", lambda *a, **k: calls.setdefault("view", a))

    def no_bill():
        raise AssertionError("--here must not read the Modal bill")
    if not TOKENIZER.exists():
        pytest.skip("no tokenizer")
    rc = mt.main(["--smoke", "--here", "--gpus", "8", "--pool", "/p", "--val", "/v",
                  "--work", str(tmp_path)], metered=no_bill)
    assert rc == 0 and calls["smoke"]["world"] == 8 and calls["smoke"]["device"] == "cuda"
    assert calls["view"][1:3] == (Path("/p"), Path("/v"))
    monkeypatch.setattr(mt, "run_training", lambda **k: calls.setdefault("run", k) and {"ok": True})
    rc = mt.main(["--here", "--gpus", "8", "--pool", "/p", "--val", "/v", "--ckpt-root", "/c",
                  "--hours", "2", "--work", str(tmp_path)], metered=no_bill)
    assert rc == 0 and calls["run"]["world"] == 8 and calls["run"]["hours"] == 2
    assert calls["run"]["usd_per_hour"] == 1.0                 # budget_usd == hours
    with pytest.raises(SystemExit):
        mt.parse_args(["--here", "--pool", "/p", "--val", "/v"])   # run needs --ckpt-root


def test_the_plan_logic_runs_without_modal(mt, capsys):
    assert mt.modal is None
    assert mt.main(["--smoke", "--plan-only"], metered=lambda: 0.0) == 0
    assert "plan only" in capsys.readouterr().out


# ---- the smoke's checks -----------------------------------------------------------------

def _rec(rank, start, end, batches, exit=0, pm=(True,), world=2):
    return {"rank": rank, "world": world, "exit": exit, "start_step": start,
            "final_step": end, "checkpoints": [end], "params_match": list(pm),
            "batches": list(batches), "step_seconds": [1.0, 0.5, 0.5],
            "tokens_per_step": 16384}


def good():
    m1 = {r: _rec(r, 0, 6, [f"a{r}-{i}" for i in range(12)]) for r in range(2)}
    m2 = {r: _rec(r, 6, 10, [f"b{r}-{i}" for i in range(8)]) for r in range(2)}
    log = {"steps": [{"step": s, "train_loss": 11.0 - 0.3 * s} for s in range(1, 11)],
           "resumes": [{"from_step": 6}]}
    return dict(world=2, steps1=6, steps2=4, accum_per_rank=2, rc1=0, rc2=0,
                markers1=m1, markers2=m2, runlog=log)


def test_a_clean_smoke_passes(mt):
    assert mt.check_smoke(**good()) == []
    assert mt.tokens_per_second(good()["markers1"][0]) == pytest.approx(16384 / 0.5)


@pytest.mark.parametrize("breaks, match", [
    (lambda g: g.update(rc2=1), "torchrun exit 1"),
    (lambda g: g["markers1"][1].update(exit=1), "rank 1 exit 1"),
    (lambda g: g["markers2"].pop(1), "rank records [0]"),
    (lambda g: g["markers1"][0].update(params_match=[True, False]), "diverged"),
    (lambda g: g["markers2"][0].update(params_match=[]), "never checked"),
    (lambda g: g["markers2"][1].update(start_step=0), "started at step 0"),
    (lambda g: g["markers2"][1].update(final_step=9), "stopped at step 9"),
    (lambda g: g["markers1"][1]["batches"].__setitem__(0, "a0-0"), "identical micro-batches"),
    (lambda g: g["markers2"][0]["batches"].__setitem__(0, "a0-3"), "re-read"),
    (lambda g: g["markers1"][0]["batches"].pop(), "read 11 micro-batches"),
    (lambda g: g["runlog"]["steps"][3].update(train_loss=float("nan")), "non-finite"),
    (lambda g: g["runlog"].update(steps=[{"step": s, "train_loss": 10.0 + 0.01 * s}
                                         for s in range(1, 11)]), "not decreasing"),
    (lambda g: g["runlog"]["steps"].pop(6), "run log steps"),
    (lambda g: g["runlog"].update(resumes=[]), "no resume"),
    (lambda g: g.update(runlog=None), "no run log"),
])
def test_each_broken_smoke_is_named(mt, breaks, match):
    g = good()
    breaks(g)
    failures = mt.check_smoke(**g)
    assert any(match in f for f in failures), failures


def test_the_commands(mt):
    cmd = mt.train_command(python="python", nproc=8, marker_dir="/m", config="c.toml",
                           run_id="r", run_dir="/runs", overrides={"train.compile": False,
                                                                   "train.micro_batch": 32},
                           flags=["--resume"])
    assert cmd[:6] == ["python", "-m", "torch.distributed.run", "--standalone",
                       "--nproc_per_node=8", "-m"]
    assert cmd[6:10] == ["quipu.rankrun", "--marker-dir", "/m", "--"]
    assert "train.compile=false" in cmd and "train.micro_batch=32" in cmd
    assert cmd[-1] == "--resume"


# ---- the run mode's start decision -------------------------------------------------------

def _ckpt(d: Path, step: int) -> Path:
    import torch

    d.mkdir(parents=True, exist_ok=True)
    f = d / f"step_{step:06d}.pt"
    f.write_bytes(b"x")
    torch.save({"file": f.name}, d / "latest.pt")
    return f


def test_the_first_launch_starts_from_init_with_the_optimizer_states(mt, tmp_path):
    with pytest.raises(FileNotFoundError, match="expected one step"):
        mt.plan_run(tmp_path)
    (tmp_path / "init").mkdir()
    (tmp_path / "init" / "step_008129.pt").write_bytes(b"x")
    s = mt.plan_run(tmp_path)
    assert s.run_id == "quipu-moe-cpt" and s.ckpt_dir == tmp_path / "stable"
    assert s.flags == ["--init-from", str(tmp_path / "init" / "step_008129.pt"),
                       "--init-optimizers"]


def test_later_launches_resume_the_stable_run(mt, tmp_path):
    _ckpt(tmp_path / "stable", 9000)
    s = mt.plan_run(tmp_path)
    assert s.flags == ["--resume"] and s.overrides == {}


def test_a_decay_branches_from_the_stable_checkpoint_then_resumes_itself(mt, tmp_path):
    with pytest.raises(FileNotFoundError, match="no stable checkpoint"):
        mt.plan_run(tmp_path, decay_steps=600)
    src = _ckpt(tmp_path / "stable", 9000)
    s = mt.plan_run(tmp_path, decay_steps=600, batch_tokens=1000)
    assert s.run_id == "quipu-moe-cpt-decay-009000"
    assert s.ckpt_dir == tmp_path / "decay" / "step_009000"
    assert s.flags == ["--branch-from", str(src)]
    assert s.overrides == {"train.decay_start_step": 9000, "train.decay_steps": 600,
                           "train.total_tokens": 9600 * 1000}
    _ckpt(tmp_path / "decay" / "step_009000", 9200)
    assert mt.plan_run(tmp_path, decay_steps=600).flags == ["--resume"]


def test_a_run_log_without_a_checkpoint_is_set_aside(mt, tmp_path):
    log = tmp_path / "quipu-moe-cpt.json"
    log.write_text("{}", encoding="utf-8")
    assert mt.set_aside_stale_log(tmp_path, "quipu-moe-cpt", resuming=True) is None
    aside = mt.set_aside_stale_log(tmp_path, "quipu-moe-cpt", resuming=False)
    assert aside is not None and aside.exists() and not log.exists()


# ---- end to end on the CPU: the real trainer, two processes ------------------------------

def _tiny_world(tmp_path: Path):
    """A pool of learnable token runs (each document counts up from a random start),
    the old-style val shard, a tiny tokenizer, and the overrides that shrink the cpt
    config to a CPU-sized model."""
    from tests.moe_fixtures import tiny_moe
    from quipu.data import write_shard

    _, tok = tiny_moe(tmp_path)
    sha = hashlib.sha256((tmp_path / "tokenizer.json").read_bytes()).hexdigest()
    rng = np.random.default_rng(0)
    V = tok.vocab_size

    def runs(n):
        out = []
        while len(out) < n:
            start, length = int(rng.integers(1, V)), int(rng.integers(20, 60))
            out += [(start + i) % (V - 1) + 1 for i in range(length)] + [0]
        return np.array(out[:n], dtype=np.uint16)
    pool = tmp_path / "pool"
    shards = []
    for i in range(2):
        write_shard(pool / "mixed" / f"shard_{i:05d}.bin", runs(30000))
        shards.append({"file": f"shard_{i:05d}.bin", "chunk": 0, "tokens": 30000})
    (pool / "manifest.json").write_text(json.dumps({
        "format": "data v3 pool (quipu/pool.py)", "dtype": "uint16 little-endian",
        "shard_dir": "mixed", "shards": shards,
        "provenance": {"tokenizer": {"sha256": sha, "eot": 0, "vocab_size": V}}}),
        encoding="utf-8")
    write_shard(tmp_path / "val" / "shard_000.bin", runs(4000))
    config = tmp_path / "tiny-cpt.toml"
    config.write_text(f'inherit = "{(ROOT / "configs" / "quipu-moe-cpt.toml").as_posix()}"\n',
                      encoding="utf-8")
    extra = {"model.vocab_size": V, "model.d_model": 64, "model.n_layer": 2,
             "model.n_head": 4, "model.n_kv_head": 2, "model.context": 64,
             "model.n_experts": 8, "model.top_k": 2, "model.expert_hidden": 32,
             "model.shared_hidden": 64, "data.tokenizer": str(tmp_path / "tokenizer.json"),
             "train.lr": 3e-3, "train.muon_lr": 0.02}
    return pool, sha, config, extra


def test_the_smoke_passes_end_to_end_on_two_cpu_processes(mt, tmp_path):
    from quipu.shard_view import make_shard_view

    pool, sha, config, extra = _tiny_world(tmp_path)
    make_shard_view(tmp_path / "view", pool, tmp_path / "val", sha)
    lines = []
    res = mt.run_smoke(tmp_path / "smoke", config=str(config), shard_dir=str(tmp_path / "view"),
                       world=2, steps1=6, steps2=4, micro_batch=2, dispatch="loop",
                       compile=False, device="cpu", usd_per_hour=1.0, budget_usd=1.0,
                       cwd=ROOT, timeout_s=600, extra_overrides=extra,
                       launcher="processes", echo=lines.append)
    assert res["passed"], "\n".join(res["failures"] + lines[-40:])
    assert len(res["losses"]) == 10 and res["tokens_per_s"]["run2"] > 0
    assert any("SMOKE PASS" in l for l in lines)


def test_the_monthly_run_starts_resumes_and_branches_a_decay_on_the_cpu(mt, tmp_path):
    import torch

    pool, sha, config, extra = _tiny_world(tmp_path)
    ckpt_root = tmp_path / "ckpt"
    small = {**extra, "train.batch_tokens": 256, "train.warmup_steps": 2,
             "train.ckpt_every": 4, "train.eval_every": 4, "train.eval_batches": 1}

    lines: list[str] = []

    def launch(decay_steps=0, **over):
        return mt.run_training(
            ckpt_root=ckpt_root, pool_root=pool, val_dir=tmp_path / "val",
            view_dir=tmp_path / "view", world=2, micro_batch=2, dispatch="loop",
            compile=False, hours=1.0, usd_per_hour=1.0, decay_steps=decay_steps, cwd=ROOT,
            timeout_s=600, config=str(config), extra_overrides={**small, **over}, device="cpu",
            local_run_dir=tmp_path / "runs-local", launcher="processes",
            echo=lines.append)

    # The pretrain's checkpoint in /init: a short single-process run of the same model.
    from quipu.shard_view import make_shard_view
    from quipu.train import run_main

    make_shard_view(tmp_path / "pre-view", pool, tmp_path / "val", sha)
    args = ["--config", str(config), "--run-id", "pre", "--run-dir", str(tmp_path / "pre"),
            "--device", "cpu"]
    for k, v in {**small, "train.total_tokens": 256 * 4, "train.compile": "false",
                 "model.moe_dispatch": "loop", "train.micro_batch": 2,
                 "data.shard_dir": str(tmp_path / "pre-view"),
                 "train.ckpt_dir": str(ckpt_root / "init")}.items():
        args += ["--override", f"{k}={v}"]
    assert run_main(args) == 0
    assert (ckpt_root / "init" / "step_000004.pt").exists()

    first = launch(**{"train.total_tokens": 256 * 6})
    assert first["ok"], lines[-40:]
    assert "first launch" in first["plan"]
    assert (first["start_step"], first["final_step"]) == (0, 6)
    assert first["params_match"] and all(first["params_match"])
    log = json.loads((ckpt_root / "runs" / "quipu-moe-cpt.json").read_text(encoding="utf-8"))
    assert log["init_from"]["optimizers"] is True
    assert log["steps"][0]["lr"] == pytest.approx(6e-5)            # the wsd re-warmup start

    second = launch(**{"train.total_tokens": 256 * 10})
    assert second["ok"] and second["plan"] == "resume the stable run"
    assert (second["start_step"], second["final_step"]) == (6, 10)

    decay = launch(decay_steps=4)
    assert decay["ok"] and (decay["start_step"], decay["final_step"]) == (10, 14)
    d = ckpt_root / "decay" / "step_000010"
    assert (d / "milestones" / "step_000014.pt").exists()
    dlog = json.loads((ckpt_root / "runs" / "quipu-moe-cpt-decay-000010.json")
                      .read_text(encoding="utf-8"))
    lrs = [s["lr"] for s in dlog["steps"]]
    assert lrs[0] == pytest.approx(3e-3) and lrs[-1] < lrs[0]
    assert all(a > b for a, b in zip(lrs, lrs[1:]))
    # the stable chain is untouched by the branch
    ptr = torch.load(ckpt_root / "stable" / "latest.pt", weights_only=True)
    assert ptr["file"] == "step_000010.pt"
