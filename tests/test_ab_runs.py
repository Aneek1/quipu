"""M8: the A/B orchestrator (scripts/ab_runs.py) and train.py's --override.

Most tests drive the orchestrator with an in-process fake runner and a fake clock;
the divergence and retry tests run a fake `quipu.train` subprocess (a small script
written to tmp_path); one end-to-end test trains two tiny smoke arms for real, on
the CPU (device forced: CUDA_VISIBLE_DEVICES="" does not hide this laptop's GPU).
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

from quipu.config import load_config, parse_overrides
from quipu.data import write_shard

ROOT = Path(__file__).resolve().parents[1]
AB = ROOT / "configs" / "quipu-moe-ab.toml"
SMOKE = ROOT / "configs" / "quipu-moe-smoke.toml"

_spec = importlib.util.spec_from_file_location("ab_runs", ROOT / "scripts" / "ab_runs.py")
ab = importlib.util.module_from_spec(_spec)
sys.modules["ab_runs"] = ab
_spec.loader.exec_module(ab)


# ---- helpers ---------------------------------------------------------------------

class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def res(val, tps=100_000.0, spikes=0, status="completed", losses=None, wall=None):
    """A hand-made RunResult."""
    return ab.RunResult(
        status=status, final_val_loss=val, bpb=None, tokens_per_s=tps,
        effective_tokens_per_s=tps, wall_s=wall or 1.0, spikes=spikes,
        train_losses=losses or [], reason="", attempts=1,
    )


class FakeRunner:
    """In-process stand-in for the training subprocess. `outcome(spec)` returns a
    RunResult (or a list of them, one per attempt); the fake clock advances by the
    run's tokens at `tps`."""

    def __init__(self, clock, outcome, tps=100_000.0):
        self.clock = clock
        self.outcome = outcome
        self.tps = tps
        self.calls: list[str] = []

    def __call__(self, spec, ctx):
        self.calls.append(spec.name)
        self.clock.t += spec.tokens / self.tps
        out = self.outcome(spec)
        if isinstance(out, list):
            out = out[sum(1 for c in self.calls if c == spec.name) - 1]
        return out


def val_by_settings(spec):
    """Deterministic fake losses: AdamW best at 6e-4, Muon best at 0.02 and better
    than AdamW by a lot; AttnRes a little better; SiTU-GLU worse. The seed moves the
    loss by 0.001 (the noise)."""
    s = spec.settings
    v = 3.0
    if s["train.optimizer"] == "adamw":
        v += abs(math.log(s["train.lr"] / 6e-4))
    else:
        v += abs(math.log(s["train.muon_lr"] / 0.02)) - 0.2
    if s["model.attnres_blocks"]:
        v -= 0.05
    if s["model.activation"] == "situ_glu":
        v += 0.05
    if s["train.precision"] == "fp8":
        v += 0.0005
    v += 0.001 * (s["train.seed"] - 1337)
    tps = 100_000.0 * (1.3 if s["train.precision"] == "fp8" else 1.0)
    return res(v, tps=tps)


def orch(tmp_path, runner, clock=None, **kw):
    kw.setdefault("budget_usd", 100.0)
    kw.setdefault("usd_per_hour", 0.55)
    return ab.Orchestrator(AB, tmp_path / "ab", runner, clock=clock or FakeClock(), **kw)


# ---- train.py --override ------------------------------------------------------------

def test_override_values_are_typed_by_the_dataclass_field():
    over = parse_overrides([
        "train.lr=1.2e-3", "train.seed=7", "model.attnres_blocks=4",
        "model.activation=situ_glu", "train.compile=false", "train.milestones=[]",
        "train.total_tokens=100_000_000", "data.shard_dir=C:\\x\\shards", "name=ab-x",
        "train.muon_lr=1",
    ])
    assert over == {
        "train": {"lr": 1.2e-3, "seed": 7, "compile": False, "milestones": [],
                  "total_tokens": 100_000_000, "muon_lr": 1.0},
        "model": {"attnres_blocks": 4, "activation": "situ_glu"},
        "data": {"shard_dir": "C:\\x\\shards"},
        "name": "ab-x",
    }
    assert isinstance(over["train"]["muon_lr"], float)


@pytest.mark.parametrize("bad", [
    "train.nope=1", "nosection.lr=1", "train.seed=1.5", "train.compile=yes",
    "train.lr=fast", "train.lr", "model.n_layer=true", "a.b.c=1",
])
def test_bad_overrides_are_refused(bad):
    with pytest.raises(ValueError):
        parse_overrides([bad])


def test_train_cli_applies_overrides(tmp_path, monkeypatch):
    import quipu.train as train_mod

    seen = {}

    class Stop(Exception):
        pass

    def fake_trainer(**kw):
        seen.update(kw)
        raise Stop

    monkeypatch.setattr(train_mod, "Trainer", fake_trainer)
    with pytest.raises(Stop):
        train_mod.main(["--config", str(SMOKE), "--device", "cpu", "--run-dir",
                        str(tmp_path / "runs"), "--override", "train.lr=0.002",
                        "--override", "model.attnres_blocks=0"])
    assert seen["train_cfg"].lr == 0.002
    assert seen["model_cfg"].attnres_blocks == 0
    assert Path(seen["run_dir"]) == tmp_path / "runs"


def test_train_cli_bad_override_is_a_usage_error():
    from quipu.train import EXIT_USAGE, run_main

    assert run_main(["--config", str(SMOKE), "--override", "train.nope=1"]) == EXIT_USAGE


# ---- dry run --------------------------------------------------------------------------

def test_dry_run_lists_the_exact_runs(capsys, tmp_path):
    assert ab.main(["--config", str(AB), "--out", str(tmp_path / "ab"), "--dry-run",
                    "--tokens-per-second", "150000"]) == 0
    out = capsys.readouterr().out
    assert "13 runs" in out and "2,000,000,000 tokens" in out
    lines = [ln for ln in out.splitlines() if ln.lstrip()[:3].strip().isdigit()]
    assert len(lines) == 13
    assert "sweep-adamw-lr6e-04" in lines[0] and "100,000,000" in lines[0]
    assert "train.lr=0.0006" in lines[0] and "train.optimizer=adamw" in lines[0]
    assert "sweep-adamw-lr3e-04" in lines[1] and "sweep-adamw-lr1.2e-03" in lines[2]
    assert "sweep-muon-lr2e-02" in lines[3] and "train.lr=<best AdamW lr>" in lines[3]
    assert "p1-adamw" in lines[6] and "200,000,000" in lines[6]
    assert "p1-muon" in lines[7] and "train.muon_lr=<best Muon lr>" in lines[7]
    assert "p2-attnres" in lines[8] and "model.attnres_blocks=4" in lines[8]
    assert "p3-situ_glu" in lines[9] and "model.activation=situ_glu" in lines[9]
    assert [ln.split()[1] for ln in lines] == ["sweep"] * 6 + ["pair"] * 4 + ["seed"] * 3
    assert "p1-winner-seed1338" in lines[10] and "p3-winner-seed1338" in lines[12]
    assert "train.seed=1338" in lines[10]
    assert not any("fp8" in ln for ln in lines)
    assert not (tmp_path / "ab").exists()          # nothing trained, nothing written


def test_dry_run_with_fp8_adds_pair_4_and_its_seed_rerun(capsys, tmp_path):
    assert ab.main(["--config", str(AB), "--out", str(tmp_path / "ab"), "--dry-run",
                    "--with-fp8", "--tokens-per-second", "150000"]) == 0
    out = capsys.readouterr().out
    assert "15 runs" in out and "2,400,000,000 tokens" in out
    lines = [ln for ln in out.splitlines() if ln.lstrip()[:3].strip().isdigit()]
    assert len(lines) == 15
    assert "p4-fp8" in lines[10] and "train.precision=fp8" in lines[10]
    assert [ln.split()[1] for ln in lines].count("seed") == 4
    assert "p4-winner-seed1338" in lines[14]


def test_dry_run_cost_follows_the_throughput_assumption(capsys, tmp_path):
    ab.main(["--config", str(AB), "--out", str(tmp_path), "--dry-run",
             "--tokens-per-second", "250000", "--usd-per-hour", "0.55"])
    out = capsys.readouterr().out
    # 2.0e9 tokens / 250k tok/s = 8000 s, + 13 runs x 180 s overhead = 10340 s.
    hours = (2e9 / 250_000 + 13 * ab.PRIOR_OVERHEAD_S) / 3600
    assert f"${hours * 0.55:.2f}" in out


# ---- decision rules ---------------------------------------------------------------------

def test_noise_rule_keeps_the_simpler_option_within_noise():
    # Muon is 0.01 better, but two seeds of Muon differ by 0.02: within noise.
    d = ab.decide_pair("optimizer", res(3.00), res(2.99), res(3.01))
    assert d.keep == "simple" and "noise" in d.reason
    # 0.1 better with the same noise: kept.
    d = ab.decide_pair("optimizer", res(3.00), res(2.90), res(2.92))
    assert d.keep == "complex"
    assert d.noise == pytest.approx(0.02)


def test_simpler_option_wins_when_it_has_lower_loss():
    d = ab.decide_pair("optimizer", res(2.9), res(3.0), res(2.9))
    assert d.keep == "simple" and d.prelim == "simple"


def test_throughput_rule_for_attnres():
    # 0.1 better, well beyond noise, but 15% slower: rejected.
    d = ab.decide_pair("attnres", res(3.0, tps=100.0), res(2.9, tps=85.0), res(2.9, tps=85.0))
    assert d.keep == "simple" and d.prelim == "simple" and "throughput" in d.reason
    # 5% slower: kept.
    d = ab.decide_pair("attnres", res(3.0, tps=100.0), res(2.9, tps=95.0), res(2.905))
    assert d.keep == "complex"


def test_spike_rule_for_situ_glu():
    d = ab.decide_pair("activation", res(3.0, spikes=1), res(2.8, spikes=2), res(2.8))
    assert d.keep == "simple" and "spike" in d.reason
    d = ab.decide_pair("activation", res(3.0, spikes=2), res(2.8, spikes=2), res(2.81))
    assert d.keep == "complex"


def test_fp8_rule():
    # 1.3x faster, loss 0.005 worse, noise 0.01: kept.
    d = ab.decide_pair("precision", res(3.000, tps=100.0), res(3.005, tps=130.0), res(3.015))
    assert d.keep == "complex"
    # Only 1.1x faster: rejected even with equal loss.
    d = ab.decide_pair("precision", res(3.0, tps=100.0), res(3.0, tps=110.0), res(3.0))
    assert d.keep == "simple" and "1.2" in d.reason
    # Fast but 0.05 worse with noise 0.01: rejected.
    d = ab.decide_pair("precision", res(3.00, tps=100.0), res(3.05, tps=130.0), res(3.06))
    assert d.keep == "simple" and "noise" in d.reason
    # Fast, within noise, but more spikes: rejected.
    d = ab.decide_pair("precision", res(3.0, tps=100.0), res(3.0, tps=130.0, spikes=1), res(3.0))
    assert d.keep == "simple" and "spike" in d.reason


def test_missing_arms_keep_the_simpler_option_and_say_so():
    failed = res(None, status="failed")
    for simple, complex_, reseed in [
        (res(3.0), None, None),
        (res(3.0), failed, None),
        (None, res(2.0), res(2.0)),
        (res(3.0), res(2.0), None),              # the winner's seed re-run is missing
    ]:
        d = ab.decide_pair("optimizer", simple, complex_, reseed)
        assert d.keep == "simple", (simple, complex_, reseed)
        assert "missing" in d.reason


def test_a_diverged_complex_arm_loses():
    d = ab.decide_pair("activation", res(3.0), res(None, status="diverged"), None)
    assert d.keep == "simple" and "diverged" in d.reason


def test_sweep_picks_the_lowest_final_val_loss():
    best, note = ab.decide_sweep({3e-4: res(3.2), 6e-4: res(3.1), 1.2e-3: res(3.05)}, 6e-4)
    assert best == 1.2e-3
    best, note = ab.decide_sweep({3e-4: res(None, status="failed"), 6e-4: None}, 6e-4)
    assert best == 6e-4 and "config" in note


def test_count_spikes():
    losses = [4.0] * 20 + [6.5] + [4.0] * 5 + [5.0]
    assert ab.count_spikes(losses) == 1        # 6.5 > 1.5 x 4.0; 5.0 is not


# ---- the whole procedure on fake results --------------------------------------------------

def test_full_procedure_on_fake_results_writes_summary_and_winners(tmp_path):
    clock = FakeClock()
    runner = FakeRunner(clock, val_by_settings)
    o = orch(tmp_path, runner, clock, with_fp8=True)
    o.run()
    # 6 sweeps + p1 x2 + p2..p4 x1 + seed re-runs: p1 winner (Muon), p2 winner
    # (AttnRes on), p3 winner (SwiGLU = the p1 winner: shared), p4 (fp8).
    assert len(runner.calls) == 6 + 2 + 3 + 3
    winners = tomllib.loads((tmp_path / "ab" / "winners.toml").read_text(encoding="utf-8"))
    assert winners["train"]["optimizer"] == "muon"
    assert winners["train"]["lr"] == 6e-4 and winners["train"]["muon_lr"] == 0.02
    assert winners["model"]["attnres_blocks"] == 4
    assert winners["model"]["activation"] == "swiglu"
    assert winners["train"]["precision"] == "fp8"
    summary = (tmp_path / "ab" / "summary.md").read_text(encoding="utf-8")
    for name in ("sweep-adamw-lr6e-04", "p1-muon", "p2-attnres", "p3-situ_glu", "p4-fp8"):
        assert name in summary
    assert "| decision" in summary.lower() or "decision |" in summary.lower()


def test_winners_toml_round_trips_into_load_config(tmp_path):
    clock = FakeClock()
    o = orch(tmp_path, FakeRunner(clock, val_by_settings), clock)
    o.run()
    over = tomllib.loads((tmp_path / "ab" / "winners.toml").read_text(encoding="utf-8"))
    cfg = load_config(ROOT / "configs" / "quipu-moe.toml", over)
    assert cfg.train.optimizer == "muon" and cfg.train.muon_lr == 0.02
    assert cfg.model.attnres_blocks == 4 and cfg.model.activation == "swiglu"
    assert cfg.train.precision == "bf16"          # pair 4 not run: bf16


def test_resume_skips_completed_runs(tmp_path):
    clock = FakeClock()
    first = FakeRunner(clock, val_by_settings)
    orch(tmp_path, first, clock).run()
    again = FakeRunner(clock, val_by_settings)
    orch(tmp_path, again, clock).run()
    assert first.calls and again.calls == []
    # A different config hash (another --override) misses the cache.
    third = FakeRunner(clock, val_by_settings)
    orch(tmp_path, third, clock, extra_overrides={"train.eval_batches": 5},
         skip_sweeps=True, pairs=("attnres",), seed_reruns=False).run()
    assert len(third.calls) == 2


def test_spend_guard_refuses_a_run_that_would_exceed_the_budget(tmp_path, capsys):
    clock = FakeClock()
    runner = FakeRunner(clock, val_by_settings, tps=100_000.0)
    # 100M tokens at 100k tok/s = 1000 s = $0.1528 at $0.55/h; budget $0.40 fits two
    # sweeps (measured rate after the first), not a third.
    o = orch(tmp_path, runner, clock, budget_usd=0.40)
    o.run()
    assert runner.calls == ["sweep-adamw-lr6e-04", "sweep-adamw-lr3e-04"]
    out = capsys.readouterr().out
    assert "refused" in out and "$0.31" in out     # running spend after run 2
    summary = (tmp_path / "ab" / "summary.md").read_text(encoding="utf-8")
    assert "budget" in summary
    winners = tomllib.loads((tmp_path / "ab" / "winners.toml").read_text(encoding="utf-8"))
    assert winners["train"]["optimizer"] == "adamw"       # missing arms: simpler option
    assert winners["model"]["attnres_blocks"] == 0


def test_spend_estimate_uses_the_prior_before_any_run(tmp_path):
    clock = FakeClock()
    o = orch(tmp_path, FakeRunner(clock, val_by_settings), clock, budget_usd=0.10,
             tokens_per_second=100_000.0)
    o.run()
    assert o.runner.calls == []                    # $0.18 estimated > $0.10


def test_spend_ledger_carries_over_between_invocations(tmp_path):
    clock = FakeClock()
    o = orch(tmp_path, FakeRunner(clock, val_by_settings), clock, budget_usd=0.40)
    o.run()
    clock2 = FakeClock()
    o2 = orch(tmp_path, FakeRunner(clock2, val_by_settings), clock2, budget_usd=0.40)
    assert o2.guard.spent() == pytest.approx(2000 / 3600 * 0.55)


def test_crashed_run_is_retried_once_then_marked_failed(tmp_path):
    clock = FakeClock()
    crash = res(None, status="crashed")

    def outcome(spec):
        if spec.name == "p2-attnres":
            return [crash, crash]
        if spec.name == "p3-situ_glu":
            return [crash, res(2.0)]
        return val_by_settings(spec)

    runner = FakeRunner(clock, outcome)
    o = orch(tmp_path, runner, clock, skip_sweeps=True)
    o.run()
    assert runner.calls.count("p2-attnres") == 2
    assert runner.calls.count("p3-situ_glu") == 2
    cached = [json.loads(p.read_text()) for p in (tmp_path / "ab" / "cache").glob("*.json")]
    by_name = {c["name"]: c["result"] for c in cached}
    assert by_name["p2-attnres"]["status"] == "failed"
    assert by_name["p3-situ_glu"]["status"] == "completed"
    summary = (tmp_path / "ab" / "summary.md").read_text(encoding="utf-8")
    assert "missing" in summary
    # Re-running does not retry the failed run again (unless asked).
    again = FakeRunner(clock, outcome)
    orch(tmp_path, again, clock, skip_sweeps=True).run()
    assert again.calls == []


# ---- subprocess runner: divergence early stop ------------------------------------------------

FAKE_TRAIN = r'''
import argparse, json, os, sys, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--config"); p.add_argument("--run-id"); p.add_argument("--run-dir")
p.add_argument("--device"); p.add_argument("--resume", action="store_true")
p.add_argument("--override", action="append", default=[])
a = p.parse_args()
behave = json.loads(Path(os.environ["FAKE_TRAIN_SPEC"]).read_text())
with open(os.environ["FAKE_TRAIN_CALLS"], "a") as f:
    f.write(a.run_id + (" resume" if a.resume else "") + "\n")
losses = behave["losses"]
log = {"steps": [], "evals": [], "status": "running"}
path = Path(a.run_dir) / (a.run_id + ".json")
path.parent.mkdir(parents=True, exist_ok=True)
for i, loss in enumerate(losses, 1):
    log["steps"].append({"step": i, "train_loss": loss})
    path.write_text(json.dumps(log))
    print(f"step {i}/{len(losses)}  loss {loss:.4f}  lr 1.00e-03  grad 1.000  1,000 tok/s", flush=True)
    time.sleep(behave.get("sleep", 0.0))
log["evals"].append({"step": len(losses), "val_loss": behave.get("val", 3.0)})
log["status"] = "completed"
path.write_text(json.dumps(log))
sys.exit(behave.get("exit", 0))
'''


def fake_subprocess_runner(tmp_path, monkeypatch, behave):
    script = tmp_path / "fake_train.py"
    script.write_text(FAKE_TRAIN, encoding="utf-8")
    spec_path = tmp_path / "behave.json"
    spec_path.write_text(json.dumps(behave), encoding="utf-8")
    monkeypatch.setenv("FAKE_TRAIN_SPEC", str(spec_path))
    monkeypatch.setenv("FAKE_TRAIN_CALLS", str(tmp_path / "calls.txt"))
    return ab.SubprocessRunner(cmd=[sys.executable, str(script)], device="cpu")


def one_spec(o, name="p2-attnres", tokens=None):
    settings = dict(o.base_settings(), **{"model.attnres_blocks": 4})
    return ab.RunSpec(name=name, phase="pair", settings=settings,
                      tokens=tokens or o.arm_tokens, baseline=None)


def test_divergence_stops_the_run_at_a_quarter(tmp_path, monkeypatch):
    # 100 steps; the baseline sits at 3.0; this run is 7.0 by step 25 -> killed.
    behave = {"losses": [3.0] * 10 + [7.0] * 90, "sleep": 0.05}
    runner = fake_subprocess_runner(tmp_path, monkeypatch, behave)
    o = orch(tmp_path, runner, arm_tokens=100 * 524_288)
    spec = one_spec(o)
    r = runner(spec, o.context_for(spec, baseline_losses=[3.0] * 100))
    assert r.status == "diverged" and "baseline" in r.reason
    logged = json.loads(next((tmp_path / "ab" / "runs").glob("*.json")).read_text())
    assert len(logged["steps"]) < 60          # stopped early, not run to the end


def test_nan_loss_stops_the_run(tmp_path, monkeypatch):
    behave = {"losses": [3.0] * 5 + [float("nan")] * 95, "sleep": 0.05}
    runner = fake_subprocess_runner(tmp_path, monkeypatch, behave)
    o = orch(tmp_path, runner, arm_tokens=100 * 524_288)
    spec = one_spec(o)
    r = runner(spec, o.context_for(spec, baseline_losses=None))
    assert r.status == "diverged" and "non-finite" in r.reason


def test_nonfinite_exit_code_is_diverged_and_crash_is_crashed(tmp_path, monkeypatch):
    runner = fake_subprocess_runner(tmp_path, monkeypatch, {"losses": [3.0] * 3, "exit": 3})
    o = orch(tmp_path, runner, arm_tokens=3 * 524_288)
    spec = one_spec(o)
    assert runner(spec, o.context_for(spec, None)).status == "diverged"
    runner = fake_subprocess_runner(tmp_path, monkeypatch, {"losses": [3.0] * 3, "exit": 1})
    assert runner(spec, o.context_for(spec, None)).status == "crashed"


def test_healthy_fake_run_reports_val_loss_and_throughput(tmp_path, monkeypatch):
    behave = {"losses": [3.0] * 12, "val": 2.5}
    runner = fake_subprocess_runner(tmp_path, monkeypatch, behave)
    o = orch(tmp_path, runner, arm_tokens=12 * 524_288)
    spec = one_spec(o)
    r = runner(spec, o.context_for(spec, baseline_losses=[3.0] * 12))
    assert r.status == "completed" and r.final_val_loss == 2.5
    assert r.tokens_per_s == 1000.0 and r.spikes == 0 and len(r.train_losses) == 12


# ---- end to end on the smoke config, for real, on the CPU ------------------------------------

def test_smoke_end_to_end_two_arms_on_cpu(tmp_path, capsys):
    shards = tmp_path / "shards"
    rng = np.random.RandomState(0)
    for split in ("train", "val"):
        write_shard(shards / split / "shard_000.bin",
                    rng.randint(0, 512, 40_000).astype(np.uint16))
    out = tmp_path / "ab"
    code = ab.main([
        "--config", str(SMOKE), "--out", str(out), "--device", "cpu",
        "--skip-sweeps", "--pairs", "attnres", "--no-seed-reruns",
        "--arm-tokens", str(8 * 4096), "--attnres-blocks", "2",
        "--budget-usd", "1", "--usd-per-hour", "0.5",
        "--override", f"data.shard_dir={shards}",
    ])
    assert code == 0, capsys.readouterr().out
    cached = [json.loads(p.read_text()) for p in (out / "cache").glob("*.json")]
    assert sorted(c["name"] for c in cached) == ["p1-base", "p2-attnres"]
    for c in cached:
        r = c["result"]
        assert r["status"] == "completed", r
        assert math.isfinite(r["final_val_loss"]) and len(r["train_losses"]) == 8
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "p2-attnres" in summary and "pair 2" in summary
    winners = tomllib.loads((out / "winners.toml").read_text(encoding="utf-8"))
    # No seed re-run: AttnRes either lost on loss or has no noise estimate; either
    # way the simpler option is kept.
    assert winners["model"]["attnres_blocks"] == 0
    assert not list((out / "ckpt").glob("*/latest.pt"))           # checkpoints cleaned
