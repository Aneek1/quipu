"""M8: the A/B orchestrator (scripts/ab_runs.py) and train.py's --override.

Most tests drive the orchestrator with an in-process fake runner and a fake clock;
the divergence and retry tests run a fake `quipu.train` subprocess (a small script
written to tmp_path); one end-to-end test trains two tiny smoke arms for real, on
the CPU (device forced: CUDA_VISIBLE_DEVICES="" does not hide this laptop's GPU).
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import re
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

from quipu.config import load_config, parse_overrides
from quipu.data import write_shard
from quipu.spend import ENV_VAR as LEDGER_ENV, Ledger

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


@pytest.fixture(autouse=True)
def box_ledger(tmp_path, monkeypatch):
    """Every test's box ledger is in tmp_path, never the repo's results/spend.json."""
    path = tmp_path / "box" / "spend.json"
    monkeypatch.setenv(LEDGER_ENV, str(path))
    return path


def orch(tmp_path, runner, clock=None, out="ab", **kw):
    clock = clock or FakeClock()
    kw.setdefault("budget_usd", 100.0)
    kw.setdefault("usd_per_hour", 0.55)
    kw.setdefault("ledger", Ledger.load(tmp_path / "box" / "spend.json", clock=clock))
    return ab.Orchestrator(AB, tmp_path / out, runner, clock=clock, **kw)


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


def test_an_ab_config_that_inherits_is_refused_with_a_clear_message(tmp_path, capsys):
    # The cache key hashes the config file alone and the runs are compared on it:
    # a base pulled in through `inherit` could change unseen under a cached result.
    cfg = tmp_path / "ab.toml"
    cfg.write_text(f'inherit = "{AB.as_posix()}"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="inherit"):
        ab.Orchestrator(cfg, tmp_path / "out", None, budget_usd=100.0, usd_per_hour=0.55)
    assert ab.main(["--config", str(cfg), "--out", str(tmp_path / "out"), "--dry-run",
                    "--tokens-per-second", "150000"]) == ab.EXIT_USAGE
    assert "inherit" in capsys.readouterr().err


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


def test_dry_run_with_shared_noise_lists_one_seed_rerun_and_costs_less(capsys, tmp_path):
    base = ["--config", str(AB), "--out", str(tmp_path / "ab"), "--dry-run",
            "--tokens-per-second", "150000", "--usd-per-hour", "0.55"]
    assert ab.main(base) == 0
    full = capsys.readouterr().out
    assert ab.main(base + ["--shared-noise"]) == 0
    out = capsys.readouterr().out
    assert "11 runs" in out and "1,600,000,000 tokens" in out     # 13 - 2 seed re-runs
    lines = [ln for ln in out.splitlines() if ln.lstrip()[:3].strip().isdigit()]
    assert [ln.split()[1] for ln in lines].count("seed") == 1
    assert "p1-winner-seed1338" in lines[-1]
    assert "--shared-noise" in out

    def cost(text):
        return float(re.search(r"h, \$([\d.]+)", text).group(1))
    hours = (1.6e9 / 150_000 + 11 * ab.PRIOR_OVERHEAD_S) / 3600
    assert cost(out) == pytest.approx(round(hours * 0.55, 2)) and cost(out) < cost(full)


def test_shared_noise_runs_only_pair_1s_seed_rerun_and_uses_it_for_every_pair(tmp_path):
    clock = FakeClock()
    runner = FakeRunner(clock, val_by_settings)
    o = orch(tmp_path, runner, clock, skip_sweeps=True, shared_noise=True)
    assert o.run() == 0
    seeds = [c for c in runner.calls if "seed" in c]
    assert len(seeds) == 1 and seeds[0].startswith("p1-")       # pair 1's winner only
    noises = {d.noise for _, d, _ in o.decisions}
    assert len(noises) == 1 and noises.pop() == pytest.approx(0.001)   # the seed step
    for pair, d, _ in o.decisions:
        if pair.number > 1:
            assert "shared noise" in d.reason
    summary = (tmp_path / "ab" / "summary.md").read_text(encoding="utf-8")
    assert "Noise estimate is shared" in summary and "0.0010" in summary
    # AttnRes (0.05 better, beyond the shared 0.001) is kept as without the flag.
    assert o.winners()["model"]["attnres_blocks"] == 4


def test_shared_noise_is_off_by_default(tmp_path):
    clock = FakeClock()
    runner = FakeRunner(clock, val_by_settings)
    o = orch(tmp_path, runner, clock, skip_sweeps=True)
    o.run()
    assert not o.shared_noise
    assert "Noise estimate is shared" not in (tmp_path / "ab" / "summary.md").read_text("utf-8")


def test_shared_noise_unknown_keeps_the_simpler_option():
    d = ab.decide_pair("attnres", res(3.0), res(2.9), None, shared=True, shared_noise=None)
    assert d.keep == "simple" and "shared noise estimate" in d.reason
    d = ab.decide_pair("attnres", res(3.0), res(2.9), None, shared=True, shared_noise=0.01)
    assert d.keep == "complex" and d.noise == 0.01


# ---- the trainer's own backstop ------------------------------------------------------------------

def test_each_attempt_passes_the_budget_left_as_the_trainers_backstop(tmp_path):
    clock = FakeClock()
    seen = []

    def outcome(spec):
        return val_by_settings(spec)

    class Spy(FakeRunner):
        def __call__(self, spec, ctx):
            flags = dict(f.split("=", 1) for f in ctx.overrides)
            seen.append((float(flags["train.budget_usd"]), float(flags["train.usd_per_hour"]),
                         o.guard.spent()))
            return super().__call__(spec, ctx)

    o = orch(tmp_path, Spy(clock, outcome), clock, skip_sweeps=True, pairs=("attnres",),
             seed_reruns=False, budget_usd=5.0, usd_per_hour=0.36)
    assert o.run() == 0
    assert len(seen) == 2
    # Less the grace reserve, plus half of it back as slack: the orchestrator's
    # deadline (spend + the grace reserve) always stops a trainer before its own
    # backstop, and an orphan still has half the grace for its checkpoint.
    grace_usd = ab.STOP_GRACE_S / 3600 * 0.36
    for budget, rate, spent in seen:
        assert rate == 0.36
        assert budget == pytest.approx(5.0 - spent - 0.5 * grace_usd, abs=1e-4)
    assert seen[1][0] < seen[0][0]                     # the second run has less left
    # The orchestrator's own overrides stay refused from the command line.
    assert "train.budget_usd" in ab.RESERVED_OVERRIDES


def test_the_trainers_backstop_exit_is_stopped_budget(tmp_path, monkeypatch):
    # The trainer stopped itself on its backstop (exit 4) with its checkpoint written:
    # a budget stop (resumed next time with more budget), not a crash to retry.
    behave = {"losses": [3.0] * 5, "exit": 4}
    runner = fake_subprocess_runner(tmp_path, monkeypatch, behave)
    o = orch(tmp_path, runner, arm_tokens=100 * 524_288)
    spec = one_spec(o)
    r = runner(spec, o.context_for(spec, None))
    assert r.status == "stopped_budget" and "backstop" in r.reason


def test_the_orchestrator_tags_its_ledger_ticks(tmp_path):
    o = orch(tmp_path, FakeRunner(FakeClock(), val_by_settings))
    assert o.ledger.tool == "ab_runs"


@pytest.mark.parametrize("how", ["returns", "raises"])
def test_spend_stop_succeeds_right_after_the_orchestrator_exits(
        tmp_path, box_ledger, monkeypatch, how):
    # run_until_stopped releases the orchestrator's tag on the way out (a finally),
    # so `spend stop` right after it needs no --force, however it ended.
    from quipu import spend

    monkeypatch.setattr(spend, "remove_ticker", lambda ledger, system=None: [])
    clock = FakeClock()
    clock.t = 1_000_000.0

    def outcome(spec):
        if how == "raises":
            raise RuntimeError("the orchestrator fell over")
        return val_by_settings(spec)

    o = orch(tmp_path, FakeRunner(clock, outcome), clock, skip_sweeps=True, pairs=("attnres",),
             seed_reruns=False)
    if how == "raises":
        with pytest.raises(RuntimeError):
            ab.run_until_stopped(o)
    else:
        assert ab.run_until_stopped(o) == 0
    assert Ledger.load(box_ledger, clock=clock).active_tool() is None
    assert spend.main(["--ledger", str(box_ledger), "stop"]) == 0
    assert Ledger.load(box_ledger, clock=clock).current["ended"]


def test_the_childs_output_reaches_its_log_file_line_by_line(tmp_path, monkeypatch):
    # Each line is flushed to the run's log as it arrives (a SIGKILLed orchestrator
    # loses none of it), and the trainer is told that file ($QUIPU_TRAIN_LOG) so an
    # orphaned trainer can go on writing there.
    env_out = tmp_path / "env.json"
    behave = {"losses": [3.0] * 400, "sleep": 0.05, "marker": str(tmp_path / "marker"),
              "env_out": str(env_out)}
    runner = fake_subprocess_runner(tmp_path, monkeypatch, behave, grace_s=20, kill_wait_s=5,
                                    poll_s=0.1)
    o = orch(tmp_path, runner, arm_tokens=400 * 524_288)
    spec = one_spec(o)
    ctx = o.context_for(spec, None)

    def budget_check():
        text = ctx.log_path.read_text(encoding="utf-8") if ctx.log_path.exists() else ""
        return "step 3/" in text

    ctx = dataclasses.replace(ctx, budget_check=budget_check)
    r = runner(spec, ctx)
    assert r.status == "stopped_budget"                  # the line was on disk mid-run
    steps = [ln for ln in ctx.log_path.read_text(encoding="utf-8").splitlines()
             if ln.startswith("step ")]
    assert len(steps) < 100                              # well before the 8 KB buffer
    import os
    assert json.loads(env_out.read_text())["QUIPU_TRAIN_LOG"] == os.path.abspath(ctx.log_path)


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
    clock.t += 300                                   # the box stays up between invocations
    o2 = orch(tmp_path, FakeRunner(clock, val_by_settings), clock, budget_usd=0.40)
    assert o2.guard.spent() == pytest.approx(2300 / 3600 * 0.55)
    assert o2.run() == ab.EXIT_BUDGET                # nothing more fits
    assert o2.runner.calls == []


def test_separate_out_dirs_share_one_box_ledger(tmp_path):
    clock = FakeClock()
    orch(tmp_path, FakeRunner(clock, val_by_settings), clock, out="ab1",
         skip_sweeps=True, pairs=("attnres",), seed_reruns=False).run()
    spent = 2 * 2000 / 3600 * 0.55                   # two 200M-token arms
    o2 = orch(tmp_path, FakeRunner(clock, val_by_settings), clock, out="ab2")
    assert o2.guard.spent() == pytest.approx(spent)
    assert not (tmp_path / "ab1" / "spend.json").exists()
    assert (tmp_path / "box" / "spend.json").exists()


def test_spent_usd_is_one_adjustment_however_often_it_is_passed(tmp_path):
    clock = FakeClock()
    for _ in range(2):
        orch(tmp_path, FakeRunner(clock, val_by_settings), clock, spent_usd=1.0,
             skip_sweeps=True, pairs=("attnres",), seed_reruns=False).run()
    led = Ledger.load(tmp_path / "box" / "spend.json", clock=clock)
    assert led.adjustments == {ab.SPENT_KEY: 1.0}
    assert led.spent_usd() == pytest.approx(1.0 + 2 * 2000 / 3600 * 0.55)


def test_a_corrupt_ledger_stops_the_orchestrator(tmp_path, box_ledger, capsys):
    box_ledger.parent.mkdir(parents=True)
    box_ledger.write_text("{not json", encoding="utf-8")
    code = ab.main(["--config", str(AB), "--out", str(tmp_path / "ab"),
                    "--budget-usd", "3", "--usd-per-hour", "0.55"])
    assert code == ab.EXIT_USAGE
    assert "spend ledger" in capsys.readouterr().err


def test_usd_per_hour_is_required_to_train_but_not_for_a_dry_run(tmp_path, capsys):
    assert ab.main(["--config", str(AB), "--out", str(tmp_path / "ab"),
                    "--budget-usd", "3"]) == ab.EXIT_USAGE
    assert "--usd-per-hour" in capsys.readouterr().err
    assert ab.main(["--config", str(AB), "--out", str(tmp_path / "ab"), "--dry-run"]) == 0


@pytest.mark.parametrize("key", sorted(ab.RESERVED_OVERRIDES))
def test_orchestrator_owned_overrides_are_refused(tmp_path, key, capsys):
    with pytest.raises(ValueError, match="orchestrator"):
        orch(tmp_path, FakeRunner(FakeClock(), val_by_settings),
             extra_overrides={key: "1"})
    assert ab.main(["--config", str(AB), "--out", str(tmp_path / "ab"), "--dry-run",
                    "--override", f"{key}=1"]) == ab.EXIT_USAGE
    assert key in capsys.readouterr().err


# ---- throughput measurement -----------------------------------------------------------------

def test_a_resumed_attempt_does_not_count_as_measured_throughput(tmp_path):
    clock = FakeClock()

    def outcome(spec):
        r = val_by_settings(spec)
        r.resumed = spec.name == "p1-base"
        return r

    o = orch(tmp_path, FakeRunner(clock, outcome), clock, skip_sweeps=True,
             pairs=("attnres",), seed_reruns=False)
    o.run()
    assert len(o.guard.measured) == 1                # p2-attnres only
    by_name = {json.loads(p.read_text())["name"]: json.loads(p.read_text())["result"]
               for p in (tmp_path / "ab" / "cache").glob("*.json")}
    assert by_name["p1-base"]["effective_tokens_per_s"] is None


def test_a_restarted_orchestrator_starts_from_the_measured_throughput(tmp_path):
    clock = FakeClock()
    runner = FakeRunner(clock, val_by_settings, tps=50_000.0)   # 4000 s per 200M arm

    def outcome(spec):
        r = val_by_settings(spec)
        if spec.name == "p2-attnres":
            return [res(None, status="crashed"), dataclasses_replace(r, resumed=True)]
        return r

    runner.outcome = outcome
    orch(tmp_path, runner, clock, skip_sweeps=True, pairs=("attnres",),
         seed_reruns=False).run()
    again = orch(tmp_path, FakeRunner(clock, val_by_settings), clock, skip_sweeps=True,
                 pairs=("attnres",), seed_reruns=False)
    # Only p1-base (first attempt, not resumed) seeds the estimate: 50k tok/s.
    assert again.guard.measured == [pytest.approx(50_000.0)]
    assert again.guard.estimate_s(200_000_000) == pytest.approx(4000.0)


def dataclasses_replace(r, **kw):
    import dataclasses
    return dataclasses.replace(r, **kw)


# ---- in-run budget stop -----------------------------------------------------------------------

def test_a_run_stopped_by_the_budget_keeps_its_checkpoint_and_exits_4(tmp_path):
    clock = FakeClock()

    class StopsSecond(FakeRunner):
        def __call__(self, spec, ctx):
            out = super().__call__(spec, ctx)
            if spec.name == "p2-attnres":
                ctx.ckpt_dir.mkdir(parents=True, exist_ok=True)
                (ctx.ckpt_dir / "latest.pt").write_text("x")
                return res(None, status="stopped_budget")
            return out

    runner = StopsSecond(clock, val_by_settings)
    o = orch(tmp_path, runner, clock, skip_sweeps=True, pairs=("attnres", "activation"))
    assert o.run() == ab.EXIT_BUDGET
    assert runner.calls == ["p1-base", "p2-attnres"]        # nothing after it starts
    assert list((tmp_path / "ab" / "ckpt").glob("*/latest.pt"))   # kept for the resume
    cached = {json.loads(p.read_text())["name"] for p in (tmp_path / "ab" / "cache").glob("*.json")}
    assert "p2-attnres" not in cached                         # re-run (resumed) next time
    summary = (tmp_path / "ab" / "summary.md").read_text(encoding="utf-8")
    assert "stopped_budget" in summary


def test_the_run_deadline_follows_the_box_ledger(tmp_path):
    clock = FakeClock()
    o = orch(tmp_path, FakeRunner(clock, val_by_settings), clock, budget_usd=1.0,
             usd_per_hour=0.36)                  # $1 = 10,000 s of box time
    o.begin()
    spec = one_spec(o)
    ctx = o.context_for(spec, None)
    assert ctx.budget_check() is False
    # The deadline keeps STOP_GRACE_S of budget for the interrupt checkpoint.
    clock.t = 10_000 - ab.STOP_GRACE_S - 1
    assert ctx.budget_check() is False
    clock.t = 10_000 - ab.STOP_GRACE_S + 0.5
    assert ctx.budget_check() is True


# ---- runs that must not be retried, and what the summary reports -----------------------------

def test_an_out_of_memory_run_is_not_retried(tmp_path):
    clock = FakeClock()

    def outcome(spec):
        if spec.name == "p2-attnres":
            return res(None, status="failed_oom")
        return val_by_settings(spec)

    runner = FakeRunner(clock, outcome)
    orch(tmp_path, runner, clock, skip_sweeps=True, pairs=("attnres",),
         seed_reruns=False).run()
    assert runner.calls.count("p2-attnres") == 1
    by_name = {json.loads(p.read_text())["name"]: json.loads(p.read_text())["result"]
               for p in (tmp_path / "ab" / "cache").glob("*.json")}
    assert by_name["p2-attnres"]["status"] == "failed_oom"


def test_eval_every_is_clamped_to_about_twenty_evals():
    assert ab.aligned_eval_every(1000, 10) == 50          # 20 evals, not 100
    assert 1000 // ab.aligned_eval_every(1000, 10) <= ab.MAX_EVALS
    assert ab.aligned_eval_every(381, 50) == 127          # unchanged when already few
    assert ab.aligned_eval_every(7, 1) == 1               # tiny runs: every step is fine
    for steps in (190, 381, 953, 1000, 4096):
        d = ab.aligned_eval_every(steps, 5)
        assert steps % d == 0 and steps // d <= ab.MAX_EVALS


def test_every_run_checkpoints_at_half_way(tmp_path):
    o = orch(tmp_path, FakeRunner(FakeClock(), val_by_settings))
    spec = one_spec(o)                                    # 200M tokens = 381 steps
    flags = dict(f.split("=", 1) for f in o.context_for(spec, None).overrides)
    assert flags["train.ckpt_every"] == "191"
    assert 381 // int(flags["train.eval_every"]) <= ab.MAX_EVALS


def test_keep_checkpoints_is_refused_when_they_would_not_fit_on_disk(tmp_path, monkeypatch, capsys):
    import collections
    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(ab.shutil, "disk_usage", lambda p: usage(10**12, 0, 20 * 10**9))
    code = ab.main(["--config", str(AB), "--out", str(tmp_path / "ab"), "--keep-checkpoints",
                    "--budget-usd", "3", "--usd-per-hour", "0.55"])
    assert code == ab.EXIT_USAGE
    assert "disk" in capsys.readouterr().err
    # Plenty of disk: the check passes (the dry run reports it and trains nothing).
    monkeypatch.setattr(ab.shutil, "disk_usage", lambda p: usage(10**13, 0, 10**13))
    assert ab.main(["--config", str(AB), "--out", str(tmp_path / "ab"), "--keep-checkpoints",
                    "--dry-run"]) == 0


def test_summary_reports_skipped_steps_overhead_and_useless_seed_reruns(tmp_path):
    clock = FakeClock()

    def outcome(spec):
        r = val_by_settings(spec)
        r.skipped = 3 if spec.name == "p3-situ_glu" else 0
        r.startup_s = 40.0
        r.overhead_s = 200.0
        return r

    o = orch(tmp_path, FakeRunner(clock, outcome), clock, skip_sweeps=True,
             pairs=("activation",))
    o.run()
    summary = (tmp_path / "ab" / "summary.md").read_text(encoding="utf-8")
    header = next(ln for ln in summary.splitlines() if ln.startswith("| # |"))
    assert "skipped" in header and "overhead s" in header
    row = next(ln for ln in summary.splitlines() if "| p3-situ_glu |" in ln)
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[header.strip("|").split("|").index(" skipped ")] == "3"
    assert "median 200 s" in summary and "prior 180 s" in summary
    # SiTU-GLU lost on loss (prelim simple): the seed re-run cannot change pair 3.
    assert "cannot change" in summary


# ---- subprocess runner: signals, budget, OOM, env ----------------------------------------------

class FakeProc:
    """Popen stand-in for _stop_child: `alive_for` wait calls time out first."""

    def __init__(self, alive_for):
        self.alive_for = alive_for
        self.calls = []

    def wait(self, timeout=None):
        self.calls.append(("wait", timeout))
        if self.alive_for:
            self.alive_for -= 1
            raise ab.subprocess.TimeoutExpired("x", timeout)
        return 130

    def poll(self):
        return None if self.alive_for else 130

    def terminate(self):
        self.calls.append(("terminate",))

    def kill(self):
        self.calls.append(("kill",))


def test_ctrl_c_waits_for_the_child_then_interrupts_then_terminates_then_kills(monkeypatch):
    runner = ab.SubprocessRunner(device="cpu", grace_s=300, kill_wait_s=30)
    sent = []
    monkeypatch.setattr(runner, "_interrupt", lambda proc: sent.append("int") or True)
    # The child got Ctrl+C too and finishes its checkpoint in time: nothing else sent.
    proc = FakeProc(alive_for=0)
    runner._stop_child(proc, child_got_it=True)
    assert proc.calls == [("wait", 300)] and sent == []
    # A child that never exits: wait 300 s, SIGINT, terminate, kill, in that order.
    proc = FakeProc(alive_for=3)
    runner._stop_child(proc, child_got_it=True)
    assert proc.calls == [("wait", 300), ("wait", 30), ("terminate",), ("wait", 30),
                          ("kill",), ("wait", None)]
    assert sent == ["int"]
    # A child that did not get it (its own process group): interrupt first, then wait.
    sent.clear()
    proc = FakeProc(alive_for=0)
    runner._stop_child(proc, child_got_it=False)
    assert sent == ["int"] and proc.calls == [("wait", 300)]



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
if behave.get("env_out"):
    Path(behave["env_out"]).write_text(json.dumps(dict(os.environ)))
if behave.get("sid_out"):
    Path(behave["sid_out"]).write_text(str(os.getsid(0)))
import signal
def on_stop(signum, frame):
    Path(behave["marker"]).write_text("interrupt checkpoint")
    sys.exit(130)
for name in ("SIGINT", "SIGBREAK"):
    if hasattr(signal, name):
        signal.signal(getattr(signal, name),
                      signal.SIG_IGN if behave.get("ignore_signals") else on_stop)
losses = behave["losses"]
log = {"steps": [], "evals": [], "status": "running"}
path = Path(a.run_dir) / (a.run_id + ".json")
path.parent.mkdir(parents=True, exist_ok=True)
for i, loss in enumerate(losses, 1):
    log["steps"].append({"step": i, "train_loss": loss, "skipped": behave.get("skipped", 0)})
    path.write_text(json.dumps(log))
    print(f"step {i}/{len(losses)}  loss {loss:.4f}  lr 1.00e-03  grad 1.000  1,000 tok/s", flush=True)
    time.sleep(behave.get("sleep", 0.0))
for s in behave.get("saves", []):
    print(f"checkpoint step {len(losses)} saved in {s:.1f} s (1.4 GB)", flush=True)
if behave.get("oom"):
    print("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB", flush=True)
    sys.exit(1)
log["evals"].append({"step": len(losses), "val_loss": behave.get("val", 3.0)})
log["status"] = "completed"
path.write_text(json.dumps(log))
sys.exit(behave.get("exit", 0))
'''


def fake_subprocess_runner(tmp_path, monkeypatch, behave, **runner_kw):
    script = tmp_path / "fake_train.py"
    script.write_text(FAKE_TRAIN, encoding="utf-8")
    spec_path = tmp_path / "behave.json"
    spec_path.write_text(json.dumps(behave), encoding="utf-8")
    monkeypatch.setenv("FAKE_TRAIN_SPEC", str(spec_path))
    monkeypatch.setenv("FAKE_TRAIN_CALLS", str(tmp_path / "calls.txt"))
    return ab.SubprocessRunner(cmd=[sys.executable, str(script)], device="cpu", **runner_kw)


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
    assert r.max_save_s is None


def test_parse_save_s_reads_the_trainers_checkpoint_line():
    assert ab.parse_save_s("checkpoint step 1200 saved in 41.7 s (1.4 GB)\n") == 41.7
    assert ab.parse_save_s("checkpoint step 5 saved in 0.0 s (0.0 GB)") == 0.0
    assert ab.parse_save_s("step 5/20  loss 3.0  1,000 tok/s") is None
    assert ab.parse_save_s("warning: checkpoint step 5 saved in 1.0 s") is None


def test_save_times_raise_the_stop_grace_for_later_stops(tmp_path, monkeypatch):
    behave = {"losses": [3.0] * 3, "val": 2.5, "saves": [40.0, 150.0, 90.0]}
    said = []
    runner = fake_subprocess_runner(tmp_path, monkeypatch, behave, grace_s=300,
                                    kill_wait_s=30, echo=said.append)
    assert runner.effective_grace_s == 300
    o = orch(tmp_path, runner, arm_tokens=3 * 524_288, stop_grace_s=300)
    spec = one_spec(o)
    r = runner(spec, o.context_for(spec, None))
    assert r.max_save_s == 150.0 and runner.max_save_s == 150.0
    assert runner.effective_grace_s == 450.0                     # 3 x 150 > 300
    assert any("stop grace now 450 s" in s for s in said)
    # The next stop waits the raised grace.
    runner._interrupt = lambda proc: True
    proc = FakeProc(alive_for=0)
    runner._stop_child(proc, child_got_it=False)
    assert proc.calls == [("wait", 450.0)]
    # A shorter save later never lowers it; a result without saves reports None.
    runner._saw_save(10.0)
    assert runner.effective_grace_s == 450.0
    assert ab.RunResult.from_dict({k: v for k, v in r.to_dict().items()
                                   if k != "max_save_s"}).max_save_s is None


def test_summary_reports_checkpoint_save_times(tmp_path):
    clock = FakeClock()

    def outcome(spec):
        r = val_by_settings(spec)
        return dataclasses.replace(r, max_save_s=120.0) if spec.name == "p2-attnres" else r

    o = orch(tmp_path, FakeRunner(clock, outcome), clock, skip_sweeps=True, stop_grace_s=300)
    o.run()
    summary = (tmp_path / "ab" / "summary.md").read_text(encoding="utf-8")
    assert "## Checkpoint saves" in summary
    assert "p2-attnres 120.0 s" in summary and "= 360 s" in summary


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


def test_divergence_is_checked_at_every_printed_step_after_a_quarter(tmp_path, monkeypatch):
    # Healthy at 25%, diverging from step 60: the old single check at 25% missed it.
    behave = {"losses": [3.0] * 59 + [7.0] * 41, "sleep": 0.02}
    runner = fake_subprocess_runner(tmp_path, monkeypatch, behave)
    o = orch(tmp_path, runner, arm_tokens=100 * 524_288)
    spec = one_spec(o)
    r = runner(spec, o.context_for(spec, baseline_losses=[3.0] * 100))
    assert r.status == "diverged" and "step 60/" in r.reason


def test_the_runner_interrupts_the_child_at_the_budget(tmp_path, monkeypatch):
    marker = tmp_path / "marker"
    behave = {"losses": [3.0] * 400, "sleep": 0.05, "marker": str(marker)}
    runner = fake_subprocess_runner(tmp_path, monkeypatch, behave, poll_s=0.05,
                                    grace_s=20, kill_wait_s=5)
    o = orch(tmp_path, runner, arm_tokens=400 * 524_288)
    spec = one_spec(o)
    ctx = o.context_for(spec, None)
    polls = []
    ctx.budget_check = lambda: polls.append(1) or len(polls) >= 5
    r = runner(spec, ctx)
    assert r.status == "stopped_budget" and "budget" in r.reason
    assert marker.read_text() == "interrupt checkpoint"      # it took the SIGINT path


def test_a_child_that_ignores_the_interrupt_is_terminated(tmp_path, monkeypatch):
    behave = {"losses": [3.0] * 400, "sleep": 0.05, "ignore_signals": True,
              "marker": str(tmp_path / "marker")}
    runner = fake_subprocess_runner(tmp_path, monkeypatch, behave, poll_s=0.05,
                                    grace_s=0.5, kill_wait_s=5)
    o = orch(tmp_path, runner, arm_tokens=400 * 524_288)
    spec = one_spec(o)
    ctx = o.context_for(spec, None)
    ctx.budget_check = lambda: True
    r = runner(spec, ctx)
    assert r.status == "stopped_budget"
    assert r.wall_s < 15                                     # not the whole 20 s run


def test_cuda_out_of_memory_is_failed_oom(tmp_path, monkeypatch):
    runner = fake_subprocess_runner(tmp_path, monkeypatch, {"losses": [3.0] * 3, "oom": True})
    o = orch(tmp_path, runner, arm_tokens=3 * 524_288)
    spec = one_spec(o)
    r = runner(spec, o.context_for(spec, None))
    assert r.status == "failed_oom" and "out of memory" in r.reason


def test_child_env_resume_flag_skipped_steps_and_startup(tmp_path, monkeypatch):
    env_out = tmp_path / "env.json"
    behave = {"losses": [3.0] * 12, "val": 2.5, "skipped": 2, "env_out": str(env_out)}
    runner = fake_subprocess_runner(tmp_path, monkeypatch, behave)
    o = orch(tmp_path, runner, arm_tokens=12 * 524_288)
    spec = one_spec(o)
    ctx = o.context_for(spec, None)
    r = runner(spec, ctx)
    env = json.loads(env_out.read_text())
    assert Path(env["TORCHINDUCTOR_CACHE_DIR"]) == tmp_path / "ab" / "inductor-cache"
    assert env["TORCHINDUCTOR_FX_GRAPH_CACHE"] == "1"
    assert env["TORCHINDUCTOR_AUTOGRAD_CACHE"] == "1"
    assert r.skipped == 2 and r.resumed is False
    assert r.startup_s is not None and 0 <= r.startup_s <= r.wall_s
    assert r.overhead_s is not None
    # A retry that finds the run's checkpoint resumes, and says so.
    ctx.ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ctx.ckpt_dir / "latest.pt").write_text("x")
    assert runner(spec, ctx).resumed is True


# ---- the budget's grace reserve, resumed estimates, --spent-usd 0, --stop-grace-s -------------

def test_allows_keeps_the_stop_grace_reserve(tmp_path):
    led = Ledger.load(tmp_path / "box" / "spend.json", clock=FakeClock())
    g = ab.SpendGuard(1.0, led, 0.36, grace_s=300)       # reserve: 300 s x $0.36/h = $0.03
    assert g.allows(0.96) is True
    assert g.allows(0.98) is False                       # fits the budget, not the reserve
    assert ab.SpendGuard(1.0, led, 0.36, grace_s=0).allows(0.98) is True


def test_a_resumed_runs_estimate_scales_by_its_remaining_steps(tmp_path):
    o = orch(tmp_path, FakeRunner(FakeClock(), val_by_settings))
    spec = one_spec(o)                                   # 200M tokens = 381 steps
    ctx = o.context_for(spec, None)
    full, done = o.estimate_for(spec, ctx)
    assert done == 0 and full == pytest.approx(o.guard.estimate_usd(spec.tokens))
    ctx.ckpt_dir.mkdir(parents=True)
    (ctx.ckpt_dir / "latest.pt").write_text("x")
    (ctx.ckpt_dir / "step_000100.pt").write_text("x")
    (ctx.ckpt_dir / "step_000191.pt").write_text("x")
    part, done = o.estimate_for(spec, ctx)
    assert done == 191
    left = spec.tokens * (381 - 191) / 381
    assert part == pytest.approx((left / ab.PRIOR_TOKENS_PER_S + ab.PRIOR_OVERHEAD_S)
                                 / 3600 * 0.55)


def test_a_resumed_run_that_fits_the_budget_is_not_refused(tmp_path):
    clock = FakeClock()
    runner = FakeRunner(clock, val_by_settings)
    # p1-base at 200M: $0.33 in full at the prior, ~$0.19 with 190 of 381 steps left.
    o = orch(tmp_path, runner, clock, budget_usd=0.28, skip_sweeps=True,
             pairs=("attnres",), seed_reruns=False)
    spec = ab.RunSpec("p1-base", "pair", o.base_settings(), o.arm_tokens)
    ckpt = o.context_for(spec, None).ckpt_dir
    ckpt.mkdir(parents=True)
    (ckpt / "latest.pt").write_text("x")
    (ckpt / "step_000191.pt").write_text("x")
    o.run()
    assert runner.calls[:1] == ["p1-base"]


def test_spent_usd_zero_clears_the_adjustment_and_none_keeps_it(tmp_path):
    clock = FakeClock()
    kw = dict(skip_sweeps=True, pairs=("attnres",), seed_reruns=False)
    orch(tmp_path, FakeRunner(clock, val_by_settings), clock, spent_usd=1.0, **kw).run()
    orch(tmp_path, FakeRunner(clock, val_by_settings), clock, **kw).run()     # not passed
    assert Ledger.load(tmp_path / "box" / "spend.json").adjustments == {ab.SPENT_KEY: 1.0}
    o = orch(tmp_path, FakeRunner(clock, val_by_settings), clock, spent_usd=0.0, **kw)
    assert o.projected_spent() == pytest.approx(o.ledger.spent_usd() - 1.0)
    o.run()
    assert Ledger.load(tmp_path / "box" / "spend.json").adjustments == {ab.SPENT_KEY: 0.0}


def test_stop_grace_s_reaches_the_runner_and_the_budget_reserve(tmp_path, monkeypatch):
    made = []

    class Spy(ab.Orchestrator):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            made.append(self)

    monkeypatch.setattr(ab, "Orchestrator", Spy)
    monkeypatch.setattr(ab, "checkpoint_disk_check", lambda o: (True, "ok"))
    assert ab.main(["--config", str(AB), "--out", str(tmp_path / "ab"), "--dry-run",
                    "--stop-grace-s", "120"]) == 0
    (o,) = made
    assert o.guard.grace_s == 120 and o.runner.grace_s == 120
    assert o.spent_usd is None                           # --spent-usd not passed


def test_the_budget_deadline_uses_the_configured_grace(tmp_path):
    clock = FakeClock()
    o = orch(tmp_path, FakeRunner(clock, val_by_settings), clock, budget_usd=1.0,
             usd_per_hour=0.36, stop_grace_s=60)          # $1 = 10,000 s of box time
    o.begin()
    ctx = o.context_for(one_spec(o), None)
    clock.t = 10_000 - 61
    assert ctx.budget_check() is False
    clock.t = 10_000 - 59
    assert ctx.budget_check() is True


# ---- stopping cleanly: signals, sessions, the interrupt forwarded -------------------------------

def test_stopping_a_child_reports_how_long_its_interrupt_checkpoint_took():
    said = []
    runner = ab.SubprocessRunner(device="cpu", grace_s=300, kill_wait_s=30, echo=said.append)
    runner._interrupt = lambda proc: True
    runner._stop_child(FakeProc(alive_for=0), child_got_it=False)
    assert any("exited" in s and "after the interrupt" in s for s in said)


def test_a_stop_signal_raises_keyboard_interrupt_once():
    said = []
    handler = ab._stop_signal_handler(said.append)
    with pytest.raises(KeyboardInterrupt):
        handler(15, None)
    handler(15, None)                                    # a repeat does not cut the grace short
    assert len(said) == 2 and "again" in said[1]


def test_children_run_in_their_own_session_on_posix():
    assert ab._popen_kwargs(posix=True) == {"start_new_session": True}
    assert ab._popen_kwargs(posix=False) == {
        "creationflags": getattr(ab.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)}


def test_ctrl_c_forwards_the_interrupt_to_the_child_explicitly(tmp_path, monkeypatch):
    """The child no longer shares the terminal's process group, so it only learns
    of a Ctrl+C from the orchestrator: it must be sent, and the child checkpoints."""
    marker = tmp_path / "marker"
    behave = {"losses": [3.0] * 400, "sleep": 0.05, "marker": str(marker)}

    def echo(line):
        if line.startswith("step 3/"):
            raise KeyboardInterrupt

    runner = fake_subprocess_runner(tmp_path, monkeypatch, behave, grace_s=20, kill_wait_s=5)
    runner.echo = echo
    o = orch(tmp_path, runner, arm_tokens=400 * 524_288)
    spec = one_spec(o)
    with pytest.raises(KeyboardInterrupt):
        runner(spec, o.context_for(spec, None))
    assert marker.read_text() == "interrupt checkpoint"


def test_echo_survives_a_closed_terminal(monkeypatch):
    def gone(*a, **kw):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr("builtins.print", gone)
    ab._echo("a line after the terminal hung up")        # no exception


SIGNAL_DRIVER = r'''
import importlib.util, json, os, sys
from pathlib import Path
root, out, script, sig_name = sys.argv[1:5]
sys.path.insert(0, root)
spec = importlib.util.spec_from_file_location("ab_runs", Path(root) / "scripts" / "ab_runs.py")
ab = importlib.util.module_from_spec(spec); sys.modules["ab_runs"] = ab; spec.loader.exec_module(ab)
runner = ab.SubprocessRunner(cmd=[sys.executable, script], device="cpu", grace_s=20, kill_wait_s=5)
o = ab.Orchestrator(Path(root) / "configs" / "quipu-moe-ab.toml", out, runner, budget_usd=100.0,
                    usd_per_hour=0.55, skip_sweeps=True, pairs=("attnres",), seed_reruns=False,
                    arm_tokens=400 * 524_288, bytes_per_token=None)
o.begin()
sys.exit(ab.run_until_stopped(o))
'''


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals and sessions")
@pytest.mark.parametrize("sig_name", ["SIGTERM", "SIGHUP"])
def test_sigterm_or_sighup_stops_the_orchestrator_through_the_checkpoint_path(
        tmp_path, monkeypatch, sig_name):
    import os
    import signal
    import subprocess

    marker, sid_out = tmp_path / "marker", tmp_path / "sid"
    behave = {"losses": [3.0] * 400, "sleep": 0.05, "marker": str(marker),
              "sid_out": str(sid_out)}
    fake_subprocess_runner(tmp_path, monkeypatch, behave)      # writes the script + env
    driver = tmp_path / "driver.py"
    driver.write_text(SIGNAL_DRIVER, encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(driver), str(ROOT), str(tmp_path / "ab"),
         str(tmp_path / "fake_train.py"), sig_name],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        if line.startswith("step 3/"):
            proc.send_signal(getattr(signal, sig_name))
            break
    rest = proc.stdout.read()
    assert proc.wait(timeout=60) == ab.EXIT_INTERRUPTED, rest
    assert marker.read_text() == "interrupt checkpoint"         # the child checkpointed
    # The child ran in its own session: a terminal hangup never reached it directly.
    assert int(sid_out.read_text()) != os.getsid(0)
    assert (tmp_path / "ab" / "summary.md").exists()
