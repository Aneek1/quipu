"""scripts/remote/run_moe.py (plan Task M9): the throughput gate, the plan and GO file,
the long run under the spend guard, and scripts/remote/sync.sh. Everything runs on
the CPU with a fake clock and a fake trainer (no torch model is built), except one
test that drives a real child process through SIGINT / CTRL_BREAK."""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from quipu.config import load_config
from quipu.spend import Ledger

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "quipu-moe-smoke.toml"
MOE = ROOT / "configs" / "quipu-moe.toml"
SYNC = ROOT / "scripts" / "remote" / "sync.sh"

_spec = importlib.util.spec_from_file_location("run_moe", ROOT / "scripts" / "remote" / "run_moe.py")
rm = importlib.util.module_from_spec(_spec)
sys.modules["run_moe"] = rm
_spec.loader.exec_module(rm)

# The smoke config: batch 4096 tokens, 2,000,000 tokens = 488 steps, warmup 20,
# milestones [100, 250], eval_every 100.
BT = 4096
CFG_STEPS = 2_000_000 // BT
RATE = 3.6          # $/h: $0.001 per second, so dollars read as kiloseconds


class FakeClock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class FakeTrainer:
    """Stands in for `python -m quipu.train` behind the launcher's run_child hook.
    Each step advances the fake clock by batch_tokens / tps (plus `startup_s` before
    the first step of every attempt), writes the run log and polls should_stop the
    way the real watcher does. A stop request is the SIGINT: the interrupt
    checkpoint is written at the current step and the child exits 130."""

    def __init__(self, clock: FakeClock, tps, startup_s: float = 30.0,
                 crash_at: int | None = None, backstop_at: int | None = None) -> None:
        self.clock = clock
        self.tps = tps                       # a number, or a function of the attempt number
        self.startup_s = startup_s
        self.crash_at = crash_at
        self.backstop_at = backstop_at       # the trainer's own budget stop: exit 4
        self.calls: list[dict] = []
        self.checkpoint_step = 0
        self.sigints = 0

    def __call__(self, cmd, log_path, should_stop):
        cfg_path = Path(cmd[cmd.index("--config") + 1])
        run_dir = Path(cmd[cmd.index("--run-dir") + 1])
        run_id = cmd[cmd.index("--run-id") + 1]
        resume = "--resume" in cmd
        raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
        total, bt = raw["train"]["total_tokens"], raw["train"]["batch_tokens"]
        steps = total // bt
        attempt = len(self.calls) + 1
        tps = self.tps(attempt) if callable(self.tps) else self.tps
        self.calls.append({"resume": resume, "total_tokens": total,
                           "milestones": raw["train"]["milestones"],
                           "budget_usd": raw["train"]["budget_usd"],
                           "usd_per_hour": raw["train"]["usd_per_hour"],
                           "start": self.checkpoint_step if resume else 0})
        run_log = run_dir / f"{run_id}.json"
        ckpt_dir = Path(raw["train"]["ckpt_dir"])
        if resume:
            log = json.loads(run_log.read_text(encoding="utf-8"))
            log["steps"] = [s for s in log["steps"] if s["step"] <= self.checkpoint_step]
            step = self.checkpoint_step
        else:
            assert not run_log.exists()
            log = {"steps": [], "evals": [], "status": "running"}
            step = 0
        run_dir.mkdir(parents=True, exist_ok=True)
        started = self.clock()
        self.clock.t += self.startup_s
        events = []
        while step < steps:
            self.clock.t += bt / tps
            step += 1
            log["steps"].append({"step": step, "train_loss": 3.0, "tokens": step * bt})
            run_log.write_text(json.dumps(log), encoding="utf-8")
            if step % 10 == 0:
                events.append((self.clock(), step))
            if self.crash_at is not None and step == self.crash_at:
                self.crash_at = None
                return rm.ChildOutcome(code=1, stop_reason=None, events=events,
                                       started_at=started)
            if self.backstop_at is not None and step == self.backstop_at:
                self._checkpoint(ckpt_dir, step)
                log["status"] = "stopped_budget"
                run_log.write_text(json.dumps(log), encoding="utf-8")
                return rm.ChildOutcome(code=4, stop_reason=None, events=events,
                                       started_at=started)
            reason = should_stop(list(events))
            if reason:
                self.sigints += 1
                self._checkpoint(ckpt_dir, step)
                log["status"] = "interrupted"
                run_log.write_text(json.dumps(log), encoding="utf-8")
                return rm.ChildOutcome(code=130, stop_reason=reason, events=events,
                                       started_at=started)
        self._checkpoint(ckpt_dir, step)
        log["evals"].append({"step": step, "val_loss": 2.5})
        log["status"] = "completed"
        run_log.write_text(json.dumps(log), encoding="utf-8")
        return rm.ChildOutcome(code=0, stop_reason=None, events=events, started_at=started)

    def _checkpoint(self, ckpt_dir: Path, step: int) -> None:
        self.checkpoint_step = step
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "latest.pt").write_text(f"step {step}", encoding="utf-8")


class GoAfter:
    """sleep() that advances the fake clock and creates GO after `n` sleeps."""

    def __init__(self, clock: FakeClock, go: Path, n: int | None) -> None:
        self.clock, self.go, self.n = clock, go, n
        self.calls = 0

    def __call__(self, seconds: float) -> None:
        self.calls += 1
        self.clock.t += seconds
        if self.n is not None and self.calls >= self.n:
            self.go.parent.mkdir(parents=True, exist_ok=True)
            self.go.write_text("", encoding="utf-8")


def make_launcher(tmp_path, clock, trainer, *, budget, reserve=0.5, go_after=3,
                  gate_minutes=10.0, winners=None, overrides=(), ledger=None, **kw):
    out = tmp_path / "results" / "moe"
    ledger = ledger or Ledger.load(tmp_path / "spend.json", clock=clock)
    evals: list[list[str]] = []

    def run_cmd(cmd, log_path):
        evals.append(cmd)
        return 0

    kw.setdefault("sleep", GoAfter(clock, out / "GO", go_after))
    launcher = rm.Launcher(
        config=SMOKE, winners=winners, budget_usd=budget, usd_per_hour=RATE,
        reserve_usd=reserve, gate_minutes=gate_minutes, out=out,
        ckpt_dir=tmp_path / "checkpoints" / "quipu-moe", run_id="quipu-moe",
        device="cpu", overrides=list(overrides), ledger=ledger, run_child=trainer,
        run_cmd=run_cmd, clock=clock, echo=lines.append, poll_s=60.0,
        wait_print_s=300.0, retry_wait_s=5.0, headroom=0.0, warmup_skip_steps=20, **kw)
    launcher.eval_calls = evals
    return launcher


lines: list[str] = []


@pytest.fixture(autouse=True)
def _clear_lines():
    lines.clear()
    yield


# ---- the pure parts --------------------------------------------------------------------------

def test_effective_tokens_per_s_skips_the_warmup_steps():
    # Step lines every 10 steps; the first 20 steps (compile, first steps) are left out.
    events = [(100.0, 10), (200.0, 20), (210.0, 30), (220.0, 40), (230.0, 50)]
    assert rm.effective_tokens_per_s(events, 1000, skip_steps=20) == pytest.approx(
        (50 - 20) * 1000 / (230 - 200))
    assert rm.effective_tokens_per_s(events[:2], 1000, skip_steps=20) is None   # too short


def test_projection_math():
    p = rm.project(tokens_total=1_000_000, tokens_done=100_000, tokens_per_s=100.0,
                   usd_per_hour=3.6, spent_usd=2.0, reserve_usd=0.5, startup_s=60.0,
                   budget_usd=20.0)
    assert p.tokens_left == 900_000
    assert p.hours == pytest.approx((9000 + 60) / 3600)
    assert p.train_usd == pytest.approx(9.06)
    assert p.total_usd == pytest.approx(2.0 + 9.06 + 0.5)
    assert p.remaining_after == pytest.approx(20.0 - 11.06)     # the reserve is still in it
    assert p.fits


def test_fit_trims_to_whole_batches_and_leaves_the_reserve():
    # $5 budget, $1 spent, $0.5 reserve, 60 s startup ($0.06): $3.44 = 3440 s of training
    # at 100 tok/s = 344,000 tokens after the 100,000 done -> 444,000 -> whole 4096 batches.
    total = rm.fit_total_tokens(budget_usd=5.0, spent_usd=1.0, reserve_usd=0.5,
                                usd_per_hour=3.6, tokens_per_s=100.0, startup_s=60.0,
                                tokens_done=100_000, batch_tokens=4096)
    assert total % 4096 == 0
    assert total == (444_000 // 4096) * 4096
    p = rm.project(tokens_total=total, tokens_done=100_000, tokens_per_s=100.0,
                   usd_per_hour=3.6, spent_usd=1.0, reserve_usd=0.5, startup_s=60.0,
                   budget_usd=5.0)
    assert p.fits and p.remaining_after >= 0.5 - 1e-9     # the reserve is still there


def test_milestones_rescale_by_fraction_dedupe_and_fold_into_the_gate_step():
    # 19,454 -> 15,000 steps: each milestone keeps its fraction of the run.
    ms = rm.rescale_milestones([389, 973, 1945, 3891, 7782, 13618], 19_454, 15_000)
    assert ms == tuple(round(m * 15_000 / 19_454) for m in (389, 973, 1945, 3891, 7782, 13618))
    # A tiny run: rounding collides, duplicates collapse, nothing reaches the end.
    assert rm.rescale_milestones([10, 11, 12, 90], 100, 20) == (2, 18)
    assert rm.rescale_milestones([50, 99], 100, 50) == (25,)
    # Milestones inside the gate move to the gate's checkpoint step (the resume writes it).
    assert rm.rescale_milestones([100, 250], 488, 400, gate_step=90) == (90, 205)
    assert rm.rescale_milestones([100, 250], 488, 400, gate_step=300) == (300,)


def test_guard_stops_when_spend_plus_the_next_interval_plus_reserve_exceed_the_budget():
    # next interval: min(eval_every, steps left) steps at the measured rate.
    m = rm.guard_margin_usd(steps_left=1000, eval_every=100, batch_tokens=1000,
                            tokens_per_s=100.0, usd_per_hour=3.6)
    assert m == pytest.approx(1.0)                # 100 steps x 10 s = 1000 s = $1
    assert rm.guard_margin_usd(steps_left=30, eval_every=100, batch_tokens=1000,
                               tokens_per_s=100.0, usd_per_hour=3.6) == pytest.approx(0.3)
    assert not rm.over_budget(spent_usd=18.5, margin_usd=1.0, reserve_usd=0.5, budget_usd=20.0)
    assert rm.over_budget(spent_usd=18.51, margin_usd=1.0, reserve_usd=0.5, budget_usd=20.0)


def test_dump_toml_round_trips_the_resolved_config(tmp_path):
    raw = rm.resolve_raw(MOE, None, [])
    text = rm.dump_toml(raw)
    assert tomllib.loads(text) == raw
    path = tmp_path / "run.toml"
    path.write_text(text, encoding="utf-8")
    # the same config; only `layers` (which files it was read from) differs
    assert dataclasses.replace(load_config(path), layers=()) == dataclasses.replace(
        load_config(MOE), layers=())


def test_winners_and_overrides_are_merged_and_launcher_keys_refused(tmp_path):
    winners = tmp_path / "winners.toml"
    winners.write_text('[model]\nactivation = "situ_glu"\n[train]\noptimizer = "muon"\n'
                       "muon_lr = 0.04\n", encoding="utf-8")
    raw = rm.resolve_raw(MOE, winners, ["train.micro_batch=16"])
    assert raw["model"]["activation"] == "situ_glu"
    assert raw["train"]["optimizer"] == "muon" and raw["train"]["muon_lr"] == 0.04
    assert raw["train"]["micro_batch"] == 16
    bad = tmp_path / "bad.toml"
    bad.write_text("[train]\ntotal_tokens = 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="total_tokens"):
        rm.resolve_raw(MOE, bad, [])
    with pytest.raises(ValueError, match="milestones"):
        rm.resolve_raw(MOE, None, ["train.milestones=[5]"])


@pytest.mark.parametrize("line", ['inherit = "quipu-moe.toml"',
                                  'inherit_if_present = ["x.toml"]'])
def test_a_config_that_inherits_is_refused_with_a_clear_message(tmp_path, line):
    # resolve_raw reads the TOML itself and writes the result next to the run: an
    # `inherit` there would be re-resolved against the wrong directory, or dropped.
    cfg = tmp_path / "child.toml"
    cfg.write_text(f'name = "child"\n{line}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="inherit"):
        rm.resolve_raw(cfg, None, [])
    winners = tmp_path / "winners.toml"
    winners.write_text(f"{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inherit"):
        rm.resolve_raw(MOE, winners, [])


# ---- the launcher, end to end with a fake trainer -------------------------------------------

def plan_json(tmp_path) -> dict:
    return json.loads((tmp_path / "results" / "moe" / "plan.json").read_text(encoding="utf-8"))


def run_config(tmp_path) -> dict:
    return tomllib.loads((tmp_path / "results" / "moe" / "run_config.toml").read_text(
        encoding="utf-8"))


def test_within_budget_the_plan_keeps_the_tokens_and_the_run_resumes_from_the_gate(tmp_path):
    clock = FakeClock()
    trainer = FakeTrainer(clock, tps=BT / 2.0)        # 2 s a step; 488 steps ~ 16 min
    launcher = make_launcher(tmp_path, clock, trainer, budget=20.0, gate_minutes=5.0)
    assert launcher.run() == 0
    gate, long = trainer.calls
    assert not gate["resume"] and gate["milestones"] == [] and gate["total_tokens"] == 2_000_000
    # The long run resumes from the gate's interrupt checkpoint: the gate's steps are kept.
    assert long["resume"] and long["start"] > 0
    assert trainer.checkpoint_step == CFG_STEPS
    plan = plan_json(tmp_path)
    assert not plan["trimmed"] and plan["total_tokens"] == 2_000_000
    assert plan["gate_step"] == long["start"] > 0
    assert plan["tokens_per_s"] == pytest.approx(BT / 2.0)
    assert plan["startup_s"] == pytest.approx(30.0)      # the fake's startup, steps excluded
    md = (tmp_path / "results" / "moe" / "plan.md").read_text(encoding="utf-8")
    assert "resumes from the gate" in md and "Trimmed: no" in md
    summary = (tmp_path / "results" / "moe" / "summary.md").read_text(encoding="utf-8")
    assert "completed" in summary
    assert launcher.eval_calls and "milestone_eval.py" in " ".join(launcher.eval_calls[0])


def test_over_budget_trims_to_whole_batches_rescales_milestones_and_waits_for_go(tmp_path):
    clock = FakeClock()
    trainer = FakeTrainer(clock, tps=BT / 10.0)       # 10 s a step: 488 steps = 4880 s = $4.88
    launcher = make_launcher(tmp_path, clock, trainer, budget=3.5, reserve=0.5,
                             gate_minutes=10.0, go_after=4)
    assert launcher.run() == 0
    plan = plan_json(tmp_path)
    assert plan["trimmed"]
    total = plan["total_tokens"]
    assert total % BT == 0 and total < 2_000_000
    steps = total // BT
    gate_step = plan["gate_step"]
    expected = rm.rescale_milestones([100, 250], CFG_STEPS, steps, gate_step=gate_step)
    assert tuple(plan["milestones"]) == expected and all(m < steps for m in expected)
    # The long run trained with the trimmed numbers, from the gate's checkpoint.
    long = trainer.calls[-1]
    assert long["total_tokens"] == total and tuple(long["milestones"]) == expected
    assert long["resume"]
    md = (tmp_path / "results" / "moe" / "plan.md").read_text(encoding="utf-8")
    assert "Trimmed: yes" in md and f"{total:,}" in md and "/min" in md
    # It waited for GO: four polls, each printed/ticked, and GO came before the long run.
    assert launcher.sleep.calls >= 4
    assert any("waiting for" in s and "/min" in s for s in lines)
    # The ledger kept ticking while it waited (last_seen is the time GO was seen or later).
    led = json.loads((tmp_path / "spend.json").read_text(encoding="utf-8"))
    # The reserve is still unspent at the end.
    spent = Ledger.load(tmp_path / "spend.json", clock=clock).spent_usd()
    assert spent + 0.5 <= 3.5 + 1e-6
    assert led["sessions"][-1]["last_seen"] == pytest.approx(clock.t)


def test_the_ledger_ticks_while_waiting_for_go(tmp_path):
    clock = FakeClock()
    trainer = FakeTrainer(clock, tps=BT / 2.0)
    ticks: list[float] = []

    class Spy(GoAfter):
        def __call__(self, seconds):
            super().__call__(seconds)
            data = json.loads((tmp_path / "spend.json").read_text(encoding="utf-8"))
            ticks.append(data["sessions"][-1]["last_seen"])

    out = tmp_path / "results" / "moe"
    launcher = make_launcher(tmp_path, clock, trainer, budget=20.0, gate_minutes=5.0,
                             sleep=Spy(clock, out / "GO", 6))
    assert launcher.run() == 0
    assert len(ticks) >= 6 and ticks == sorted(ticks) and ticks[-1] > ticks[0]
    assert sum(1 for s in lines if "waiting for" in s) >= 1


def test_a_stale_go_from_before_the_plan_is_removed(tmp_path):
    clock = FakeClock()
    trainer = FakeTrainer(clock, tps=BT / 2.0)
    go = tmp_path / "results" / "moe" / "GO"
    go.parent.mkdir(parents=True)
    go.write_text("", encoding="utf-8")
    launcher = make_launcher(tmp_path, clock, trainer, budget=20.0, gate_minutes=5.0,
                             go_after=2)
    assert launcher.run() == 0
    assert launcher.sleep.calls >= 2          # it waited for a new GO
    assert any("stale" in s for s in lines)


def test_spend_guard_stops_with_sigint_checkpoint_and_exit_4(tmp_path):
    clock = FakeClock()
    # The long run is 3x slower than the gate measured: the guard must stop it.
    trainer = FakeTrainer(clock, tps=lambda attempt: BT / 5.0 if attempt == 1 else BT / 15.0)
    budget, reserve = 3.0, 0.5
    launcher = make_launcher(tmp_path, clock, trainer, budget=budget, reserve=reserve,
                             gate_minutes=5.0, go_after=1)
    assert launcher.run() == rm.EXIT_BUDGET
    assert trainer.sigints == 2              # the gate's stop, then the guard's
    plan = plan_json(tmp_path)
    steps = plan["total_tokens"] // BT
    stopped_at = trainer.checkpoint_step
    assert plan["gate_step"] < stopped_at < steps
    # It stopped at the right moment: over now, and not over one step earlier.
    led = Ledger.load(tmp_path / "spend.json", clock=clock)
    spent = led.spent_usd()
    tps_live = BT / 15.0
    margin = rm.guard_margin_usd(steps - stopped_at, 100, BT, tps_live, RATE)
    assert rm.over_budget(spent, margin, reserve, budget)
    earlier = spent - (BT / tps_live) / 3600 * RATE
    margin_before = rm.guard_margin_usd(steps - stopped_at + 1, 100, BT, tps_live, RATE)
    assert not rm.over_budget(earlier, margin_before, reserve, budget)
    assert spent + reserve <= budget + 1e-9                 # the reserve survived
    summary = (tmp_path / "results" / "moe" / "summary.md").read_text(encoding="utf-8")
    assert "spend guard" in summary and f"step {stopped_at}" in summary
    assert launcher.eval_calls == []                        # no eval of an unfinished run


def test_nothing_affordable_after_the_gate_is_exit_4_without_a_long_run(tmp_path):
    clock = FakeClock()
    # The gate (5 min = $0.30) fits the $0.82 budget with the $0.50 reserve; after it
    # not one more step does.
    trainer = FakeTrainer(clock, tps=BT / 5.0)
    launcher = make_launcher(tmp_path, clock, trainer, budget=0.82, reserve=0.5,
                             gate_minutes=5.0, go_after=1)
    assert launcher.run() == rm.EXIT_BUDGET
    assert len(trainer.calls) == 1
    md = (tmp_path / "results" / "moe" / "plan.md").read_text(encoding="utf-8")
    assert "does not fit" in md


def test_a_crash_is_retried_with_resume(tmp_path):
    clock = FakeClock()
    trainer = FakeTrainer(clock, tps=BT / 2.0, crash_at=300)
    launcher = make_launcher(tmp_path, clock, trainer, budget=20.0, gate_minutes=5.0,
                             go_after=1)
    assert launcher.run() == 0
    assert len(trainer.calls) == 3 and all(c["resume"] for c in trainer.calls[1:])


def test_the_ledger_accumulates_across_launcher_restarts(tmp_path):
    clock = FakeClock()
    trainer = FakeTrainer(clock, tps=lambda a: BT / 5.0 if a == 1 else BT / 15.0)
    first = make_launcher(tmp_path, clock, trainer, budget=3.0, gate_minutes=5.0, go_after=1)
    assert first.run() == rm.EXIT_BUDGET
    spent_first = Ledger.load(tmp_path / "spend.json", clock=clock).spent_usd()
    clock.t += 600                                   # the box stays up for ten minutes
    # A restart with more budget: same box session (not a new one), spend keeps counting,
    # no second gate (plan.json is reused), and the run resumes from its checkpoint.
    ledger = Ledger.load(tmp_path / "spend.json", clock=clock)
    second = make_launcher(tmp_path, clock, trainer, budget=20.0, gate_minutes=5.0,
                           go_after=1, ledger=ledger)
    assert second.run() == 0
    data = json.loads((tmp_path / "spend.json").read_text(encoding="utf-8"))
    assert len(data["sessions"]) == 1
    total = Ledger.load(tmp_path / "spend.json", clock=clock).spent_usd()
    assert total > spent_first + 600 / 3600 * RATE - 1e-9
    assert trainer.calls[-1]["resume"] and len(trainer.calls) == 3


def test_ensure_session_is_called_at_start(tmp_path):
    clock = FakeClock()
    ledger = Ledger.load(tmp_path / "spend.json", clock=clock)
    trainer = FakeTrainer(clock, tps=BT / 2.0)
    launcher = make_launcher(tmp_path, clock, trainer, budget=20.0, gate_minutes=5.0,
                             go_after=1, ledger=ledger)
    assert launcher.run() == 0
    assert ledger.sessions and ledger.sessions[0]["usd_per_hour"] == RATE
    assert ledger.sessions[0]["box_start"] == pytest.approx(1_000_000.0)


def test_a_changed_config_after_the_gate_is_a_usage_error(tmp_path):
    clock = FakeClock()
    trainer = FakeTrainer(clock, tps=lambda a: BT / 5.0 if a == 1 else BT / 15.0)
    assert make_launcher(tmp_path, clock, trainer, budget=3.0, gate_minutes=5.0,
                         go_after=1).run() == rm.EXIT_BUDGET
    again = make_launcher(tmp_path, clock, trainer, budget=20.0, gate_minutes=5.0,
                          go_after=1, overrides=["train.lr=2e-3"])
    assert again.run() == rm.EXIT_USAGE
    # A mistyped restart is not the end of the run: the summary stays, so sync.sh goes on.
    assert "spend guard" in (tmp_path / "results" / "moe" / "summary.md").read_text(
        encoding="utf-8")
    assert len(trainer.calls) == 2


def test_a_run_that_finishes_inside_the_gate_is_evaluated_without_a_go(tmp_path):
    clock = FakeClock()
    trainer = FakeTrainer(clock, tps=BT / 0.1)           # 488 steps in under a minute
    launcher = make_launcher(tmp_path, clock, trainer, budget=20.0, gate_minutes=5.0,
                             go_after=None)
    assert launcher.run() == 0
    assert len(trainer.calls) == 1 and launcher.sleep.calls == 0
    assert plan_json(tmp_path)["completed"] and launcher.eval_calls


def test_a_gate_too_short_to_measure_is_exit_1(tmp_path):
    clock = FakeClock()
    trainer = FakeTrainer(clock, tps=BT / 20.0)          # 20 s a step: 13 steps in 5 min
    launcher = make_launcher(tmp_path, clock, trainer, budget=20.0, gate_minutes=5.0)
    assert launcher.run() == rm.EXIT_FAILED
    assert "--gate-minutes" in (tmp_path / "results" / "moe" / "summary.md").read_text(
        encoding="utf-8")


def test_main_needs_the_box_rate(tmp_path, capsys):
    assert rm.main(["--config", str(SMOKE), "--out", str(tmp_path / "moe")]) == rm.EXIT_USAGE
    assert "--usd-per-hour" in capsys.readouterr().err


# ---- the real child runner: SIGINT (Windows CTRL_BREAK), checkpoint, exit -----------------

FAKE_CHILD = r'''
import signal, sys, time
from pathlib import Path
marker = Path(sys.argv[1])
def stop(signum, frame):
    marker.write_text("interrupt checkpoint")
    print("checkpoint step 9 saved in 4.5 s (1.4 GB)", flush=True)
    sys.exit(130)
for name in ("SIGINT", "SIGBREAK"):
    if hasattr(signal, name):
        signal.signal(getattr(signal, name), stop)
for i in range(1, 2000):
    print(f"step {i}/2000  loss 3.0000  lr 1.00e-03  grad 1.000  1,000 tok/s", flush=True)
    if i == 1:
        print("checkpoint step 1 saved in 20.0 s (1.4 GB)", flush=True)
    time.sleep(0.02)
'''


def test_subprocess_child_interrupts_so_the_trainer_checkpoints(tmp_path):
    script = tmp_path / "child.py"
    script.write_text(FAKE_CHILD, encoding="utf-8")
    marker = tmp_path / "marker.txt"
    child = rm.SubprocessChild(poll_s=0.1, grace_s=30.0, kill_wait_s=5.0, echo=lambda s: None)
    n = {"polls": 0}

    def should_stop(events):
        n["polls"] += 1
        return "budget" if n["polls"] >= 3 else None

    out = child([sys.executable, str(script), str(marker)], tmp_path / "child.log", should_stop)
    assert out.stop_reason == "budget"
    assert marker.read_text() == "interrupt checkpoint"
    assert out.code in rm.TRAIN_INTERRUPT_CODES
    assert out.events and out.events[0][1] >= 1
    # The trainer's checkpoint lines: recorded, and the grace for later stops is
    # max(grace_s, 3 x the longest save) = 60 s.
    assert out.save_s == [20.0, 4.5]
    assert child.max_save_s == 20.0 and child.effective_grace_s == 60.0


def test_parse_save_s_reads_the_trainers_checkpoint_line():
    assert rm.parse_save_s("checkpoint step 1200 saved in 41.7 s (1.4 GB)\n") == 41.7
    assert rm.parse_save_s("step 5/20  loss 3.0000") is None
    assert rm.parse_save_s("note: checkpoint step 5 saved in 1.0 s") is None


def test_the_stop_grace_follows_the_longest_save_and_never_shrinks():
    said = []
    child = rm.SubprocessChild(grace_s=300.0, kill_wait_s=5.0, echo=said.append)
    assert child.effective_grace_s == 300.0
    child._saw_save(50.0)                       # 3 x 50 = 150 < 300: unchanged
    assert child.effective_grace_s == 300.0 and not said
    child._saw_save(140.0)
    assert child.effective_grace_s == 420.0 and "stop grace now 420 s" in said[-1]
    child._saw_save(10.0)
    assert child.effective_grace_s == 420.0
    waits = []

    class Proc:
        def wait(self, timeout=None):
            waits.append(timeout)
            return 130
    child._interrupt = lambda proc: True
    child._stop_child(Proc(), child_got_it=False)
    assert waits == [420.0]


def test_summary_reports_each_attempts_longest_checkpoint_save(tmp_path):
    import dataclasses

    clock = FakeClock()
    trainer = FakeTrainer(clock, tps=BT / 2.0)

    def with_saves(cmd, log_path, should_stop):
        return dataclasses.replace(trainer(cmd, log_path, should_stop), save_s=[12.5, 30.0])

    launcher = make_launcher(tmp_path, clock, with_saves, budget=20.0, gate_minutes=5.0)
    assert launcher.run() == 0
    summary = (tmp_path / "results" / "moe" / "summary.md").read_text(encoding="utf-8")
    assert summary.count("longest checkpoint save 30.0 s") == 2     # gate + long run


# ---- the trainer's backstop, the no-op rerun, an ended session -------------------------------

def test_every_trainer_gets_the_budget_left_at_its_launch_as_its_backstop(tmp_path):
    clock = FakeClock()
    ledger = Ledger.load(tmp_path / "spend.json", clock=clock)
    trainer = FakeTrainer(clock, tps=BT / 2.0, crash_at=300)
    spent_at_launch = []

    def spy(cmd, log_path, should_stop):
        spent_at_launch.append(ledger.spent_usd())
        return trainer(cmd, log_path, should_stop)

    launcher = make_launcher(tmp_path, clock, spy, budget=20.0, reserve=1.0, gate_minutes=5.0,
                             go_after=1, ledger=ledger)
    assert launcher.run() == 0
    gate, first, retry = trainer.calls
    assert all(c["usd_per_hour"] == RATE for c in trainer.calls)
    # The gate's trainer: at most 2 x the gate's cost (5 min = $0.30), so an orphaned
    # gate cannot train the whole configured run.
    assert gate["budget_usd"] == pytest.approx(rm.GATE_BACKSTOP_FACTOR * 0.30, abs=1e-4)
    # Each long-run attempt: the budget left at its launch, less the reserve, plus half
    # the stop grace as slack, so a live launcher's guard always stops it first (a
    # retry gets less: its trainer's clock starts again at zero).
    slack = 0.5 * rm.STOP_GRACE_S / 3600 * RATE
    for call, spent in zip((first, retry), spent_at_launch[1:]):
        assert call["budget_usd"] == pytest.approx(20.0 - spent - 1.0 + slack, abs=1e-3)
    assert retry["budget_usd"] < first["budget_usd"]


def test_the_backstop_slack_follows_the_childs_stop_grace(tmp_path):
    clock = FakeClock()
    trainer = FakeTrainer(clock, tps=BT / 2.0)

    ledger = Ledger.load(tmp_path / "spend.json", clock=clock)
    spent_at_launch = []

    def spy(cmd, log_path, should_stop):
        spent_at_launch.append(ledger.spent_usd())
        return trainer(cmd, log_path, should_stop)
    spy.effective_grace_s = 900.0                      # 3 x a 300 s checkpoint save

    launcher = make_launcher(tmp_path, clock, spy, budget=20.0, reserve=1.0, gate_minutes=5.0,
                             go_after=1, ledger=ledger)
    assert launcher.run() == 0
    first = trainer.calls[1]
    assert first["budget_usd"] == pytest.approx(
        20.0 - spent_at_launch[1] - 1.0 + 0.5 * 900.0 / 3600 * RATE, abs=1e-3)


def test_the_trainers_backstop_exit_is_a_budget_stop_and_not_retried(tmp_path):
    clock = FakeClock()
    trainer = FakeTrainer(clock, tps=BT / 2.0, backstop_at=300)
    launcher = make_launcher(tmp_path, clock, trainer, budget=20.0, gate_minutes=5.0,
                             go_after=1)
    assert launcher.run() == rm.EXIT_BUDGET
    assert len(trainer.calls) == 2                    # the gate, one long-run attempt
    summary = (tmp_path / "results" / "moe" / "summary.md").read_text(encoding="utf-8")
    assert "backstop" in summary and "step 300" in summary
    assert launcher.eval_calls == []


def test_a_completed_run_is_marked_so_the_same_command_again_is_a_no_op(tmp_path):
    clock = FakeClock()
    trainer = FakeTrainer(clock, tps=BT / 2.0)
    assert make_launcher(tmp_path, clock, trainer, budget=20.0, gate_minutes=5.0).run() == 0
    plan = plan_json(tmp_path)
    assert plan["completed"] and plan["evaluated"]
    summary = tmp_path / "results" / "moe" / "summary.md"
    before = summary.read_text(encoding="utf-8")
    calls = len(trainer.calls)
    again = make_launcher(tmp_path, clock, trainer, budget=20.0, gate_minutes=5.0)
    assert again.run() == 0
    assert len(trainer.calls) == calls and again.eval_calls == []     # nothing re-run
    assert summary.read_text(encoding="utf-8") == before              # the summary stands
    assert any("nothing to do" in s for s in lines)


def test_a_box_session_ended_under_the_long_run_is_a_budget_stop(tmp_path):
    clock = FakeClock()
    ledger = Ledger.load(tmp_path / "spend.json", clock=clock)
    trainer = FakeTrainer(clock, tps=BT / 2.0)

    def ends_the_session(cmd, log_path, should_stop):
        if not trainer.calls:                          # the gate runs normally
            return trainer(cmd, log_path, should_stop)
        n = {"checks": 0}

        def check(events):
            n["checks"] += 1
            if n["checks"] == 50:                      # `spend stop --force` mid-run
                Ledger.load(tmp_path / "spend.json", clock=clock).end_session()
            return should_stop(events)
        return trainer(cmd, log_path, check)

    launcher = make_launcher(tmp_path, clock, ends_the_session, budget=20.0, gate_minutes=5.0,
                             go_after=1, ledger=ledger)
    assert launcher.run() == rm.EXIT_BUDGET
    assert trainer.sigints == 2                        # the gate's stop, then this one
    assert trainer.checkpoint_step < CFG_STEPS
    summary = (tmp_path / "results" / "moe" / "summary.md").read_text(encoding="utf-8")
    assert "session was ended" in summary


def test_the_launcher_tags_its_ledger_ticks_so_spend_stop_refuses(tmp_path):
    clock = FakeClock()
    ledger = Ledger.load(tmp_path / "spend.json", clock=clock)
    trainer = FakeTrainer(clock, tps=BT / 2.0)
    assert make_launcher(tmp_path, clock, trainer, budget=20.0, gate_minutes=5.0,
                         ledger=ledger).run() == 0
    assert ledger.tool == "run_moe"
    assert Ledger.load(tmp_path / "spend.json", clock=clock).active_tool()[0] == "run_moe"


@pytest.mark.parametrize("how", ["returns", "raises"])
def test_spend_stop_succeeds_right_after_the_launcher_exits(tmp_path, monkeypatch, how):
    # run_until_stopped releases the launcher's tag on the way out (a finally), so
    # `spend stop` right after it needs no --force, however the launcher ended.
    from quipu import spend

    monkeypatch.setattr(spend, "remove_ticker", lambda ledger, system=None: [])
    clock = FakeClock()
    path = tmp_path / "spend.json"
    ledger = Ledger.load(path, clock=clock)
    trainer = FakeTrainer(clock, tps=BT / 2.0)
    launcher = make_launcher(tmp_path, clock, trainer, budget=20.0, gate_minutes=5.0,
                             ledger=ledger)
    if how == "raises":
        def boom():
            ledger.ensure_session(RATE)
            ledger.tick()                                  # tagged: run_moe is running
            raise RuntimeError("the launcher fell over")
        launcher.run = boom
        with pytest.raises(RuntimeError):
            rm.run_until_stopped(launcher, ledger)
    else:
        assert rm.run_until_stopped(launcher, ledger) == 0
    assert Ledger.load(path, clock=clock).active_tool() is None
    assert spend.main(["--ledger", str(path), "stop"]) == 0
    assert Ledger.load(path, clock=clock).current["ended"]


# ---- the child's output: teed to the attempt's log, line by line ---------------------------

TEE_CHILD = r'''
import os, sys, time
print("log is " + os.environ.get("QUIPU_TRAIN_LOG", "unset"), flush=True)
time.sleep(20)
'''


def test_the_childs_output_reaches_its_log_file_line_by_line(tmp_path):
    # The launcher flushes each line to the attempt's log (a SIGKILLed launcher loses
    # none of it), and tells the trainer that file ($QUIPU_TRAIN_LOG) so an orphaned
    # trainer can go on writing there.
    script = tmp_path / "child.py"
    script.write_text(TEE_CHILD, encoding="utf-8")
    log = tmp_path / "logs" / "train_1.log"
    seen = []

    def should_stop(events):
        text = log.read_text(encoding="utf-8") if log.exists() else ""
        seen.append(text)
        return "stop: the line is on disk" if "log is " in text else None

    child = rm.SubprocessChild(poll_s=0.1, grace_s=10.0, kill_wait_s=5.0, echo=lambda s: None)
    outcome = child([sys.executable, str(script)], log, should_stop)
    assert outcome.stop_reason == "stop: the line is on disk"   # seen while it ran
    assert f"log is {log}" in log.read_text(encoding="utf-8")


# ---- the GO wait is bounded ----------------------------------------------------------------

def test_no_go_within_the_limit_writes_the_note_and_exits_5(tmp_path):
    clock = FakeClock()
    trainer = FakeTrainer(clock, tps=BT / 2.0)
    launcher = make_launcher(tmp_path, clock, trainer, budget=20.0, gate_minutes=5.0,
                             go_after=None, go_max_wait_min=30.0)
    assert launcher.run() == rm.EXIT_NO_GO
    assert len(trainer.calls) == 1 and launcher.sleep.calls == 30      # 30 polls of 60 s
    out = tmp_path / "results" / "moe"
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "GO not given" in summary and f"${RATE:.2f}/h" in summary
    assert "Vast console" in summary and "disk is kept" in summary
    assert "same run_moe.py command" in summary and "quipu.spend stop" in summary
    assert "GO not given within 30 min" in (out / "plan.md").read_text(encoding="utf-8")
    assert launcher.eval_calls == []
    # Later, the same command: no second gate, a new wait, GO, the long run resumes.
    again = make_launcher(tmp_path, clock, trainer, budget=20.0, gate_minutes=5.0,
                          go_after=2, go_max_wait_min=30.0)
    assert again.run() == 0
    assert len(trainer.calls) == 2 and trainer.calls[-1]["resume"]


def test_the_wait_stops_once_less_than_80_percent_of_the_approved_run_fits(tmp_path):
    clock = FakeClock()
    trainer = FakeTrainer(clock, tps=BT / 10.0)        # a slow run the budget trims
    launcher = make_launcher(tmp_path, clock, trainer, budget=3.5, reserve=0.5,
                             gate_minutes=10.0, go_after=None)
    approved = None

    class Watch(GoAfter):
        def __call__(self, seconds):
            nonlocal approved
            if approved is None:
                approved = plan_json(tmp_path)["total_tokens"]
            super().__call__(seconds)

    launcher.sleep = Watch(clock, tmp_path / "results" / "moe" / "GO", None)
    assert launcher.run() == rm.EXIT_NO_GO
    assert len(trainer.calls) == 1
    waited_min = launcher.sleep.calls
    assert 0 < waited_min < rm.GO_MAX_WAIT_MIN          # well before the time limit
    plan = plan_json(tmp_path)
    # plan.md is re-fitted to what fits now (< 80% of the approved run) for a new approval.
    assert plan["total_tokens"] < 0.8 * approved and plan["total_tokens"] % BT == 0
    md = (tmp_path / "results" / "moe" / "plan.md").read_text(encoding="utf-8")
    assert "fresh approval" in md and f"{approved:,}" in md
    summary = (tmp_path / "results" / "moe" / "summary.md").read_text(encoding="utf-8")
    assert "GO not given" in summary


# ---- the gate's overhead and the one re-fit after the first interval ------------------------

def test_planning_rate_adds_the_eval_and_checkpoint_overhead_per_interval():
    # 1000 tok/s less 5% = 1052.6 s per 1M tokens; batch 1000 tokens: 1.0526 s a step,
    # + 30 s per 100 steps (eval) + 60 s per 200 steps (save) = 1.6526 s a step.
    tps = rm.planning_tokens_per_s(1000.0, headroom=0.05, batch_tokens=1000, eval_every=100,
                                   ckpt_every=200, eval_s=30.0, save_s=60.0)
    assert tps == pytest.approx(1000 / (1000 / 950 + 0.3 + 0.3))
    assert rm.planning_tokens_per_s(1000.0, headroom=0.0, batch_tokens=1000, eval_every=100,
                                    ckpt_every=100, eval_s=0.0, save_s=0.0) == pytest.approx(1000)
    assert rm.HEADROOM == 0.05 and rm.RESERVE_USD == 1.00


def test_the_plan_uses_the_gates_longest_save_or_the_default(tmp_path):
    import dataclasses

    clock = FakeClock()
    trainer = FakeTrainer(clock, tps=BT / 2.0)

    def with_saves(cmd, log_path, should_stop):
        return dataclasses.replace(trainer(cmd, log_path, should_stop), save_s=[12.5, 40.0])

    assert make_launcher(tmp_path / "a", clock, with_saves, budget=20.0,
                         gate_minutes=5.0).run() == 0
    plan = json.loads((tmp_path / "a" / "results" / "moe" / "plan.json").read_text("utf-8"))
    assert plan["save_s"] == 40.0 and plan["save_observed"] and plan["eval_s"] == 30.0
    md = (tmp_path / "a" / "results" / "moe" / "plan.md").read_text(encoding="utf-8")
    assert "longest save the gate reported" in md
    assert make_launcher(tmp_path / "b", clock, FakeTrainer(clock, tps=BT / 2.0), budget=20.0,
                         gate_minutes=5.0).run() == 0
    plan = json.loads((tmp_path / "b" / "results" / "moe" / "plan.json").read_text("utf-8"))
    assert plan["save_s"] == rm.SAVE_OVERHEAD_S and not plan["save_observed"]


def test_the_overhead_makes_the_plan_trim_more_than_the_bare_rate_would(tmp_path):
    clock = FakeClock()
    trainer = FakeTrainer(clock, tps=BT / 10.0)
    launcher = make_launcher(tmp_path, clock, trainer, budget=3.5, reserve=0.5,
                             gate_minutes=10.0, go_after=1)
    launcher.run()
    plan = plan_json(tmp_path)
    bare = rm.fit_total_tokens(budget_usd=3.5, spent_usd=plan["spent_at_plan"], reserve_usd=0.5,
                               usd_per_hour=RATE, tokens_per_s=BT / 10.0,
                               startup_s=plan["startup_s"],
                               tokens_done=plan["gate_step"] * BT, batch_tokens=BT)
    with_overhead = rm.fit_total_tokens(
        budget_usd=3.5, spent_usd=plan["spent_at_plan"], reserve_usd=0.5, usd_per_hour=RATE,
        tokens_per_s=BT / (10.0 + 30 / 100 + 60 / 100), startup_s=plan["startup_s"],
        tokens_done=plan["gate_step"] * BT, batch_tokens=BT)
    assert with_overhead < bare
    assert plan["total_tokens"] <= with_overhead


def test_a_long_run_slower_than_planned_is_refitted_once_after_its_first_interval(tmp_path):
    clock = FakeClock()
    # The gate measures 2 s a step (planned with the default overhead: 2.9 s); the long
    # run takes 4 s a step, so its first full interval is ~28% slower than planned.
    trainer = FakeTrainer(clock, tps=lambda a: BT / 2.0 if a == 1 else BT / 4.0)
    launcher = make_launcher(tmp_path, clock, trainer, budget=2.2, reserve=0.5,
                             gate_minutes=5.0, go_after=1)
    assert launcher.run() == 0
    plan = plan_json(tmp_path)
    refit = plan["refit"]
    assert refit["from_total"] == 2_000_000 and refit["to_total"] < 2_000_000
    assert refit["to_total"] % BT == 0 and plan["total_tokens"] == refit["to_total"]
    gate, interrupted, resumed = trainer.calls              # re-fitted once, then no more
    assert interrupted["total_tokens"] == 2_000_000 and trainer.sigints == 2
    assert resumed["resume"] and resumed["total_tokens"] == refit["to_total"]
    assert resumed["milestones"] == plan["milestones"]
    # Milestones already reached stay; none lands past the new end.
    assert all(m < refit["to_total"] // BT for m in plan["milestones"])
    at = refit["at_step"]
    assert at - plan["gate_step"] >= 100                    # a full interval after the gate
    assert tuple(plan["milestones"]) == rm.refit_milestones(
        [100, 250], CFG_STEPS, refit["from_total"] // BT, refit["to_total"] // BT,
        plan["gate_step"], at)
    md = (tmp_path / "results" / "moe" / "plan.md").read_text(encoding="utf-8")
    assert "re-fitted after the long run's first full interval" in md
    spent = Ledger.load(tmp_path / "spend.json", clock=clock).spent_usd()
    assert spent + 0.5 <= 2.2 + 1e-6                        # the reserve survived
    summary = (tmp_path / "results" / "moe" / "summary.md").read_text(encoding="utf-8")
    assert "Re-fitted at step" in summary


def test_a_long_run_as_fast_as_planned_is_never_refitted_or_extended(tmp_path):
    clock = FakeClock()
    trainer = FakeTrainer(clock, tps=BT / 2.0)            # faster than the planning rate
    launcher = make_launcher(tmp_path, clock, trainer, budget=20.0, gate_minutes=5.0)
    assert launcher.run() == 0
    plan = plan_json(tmp_path)
    assert "refit" not in plan and plan["refit_checked"]
    assert plan["total_tokens"] == 2_000_000 and len(trainer.calls) == 2
    assert any("no re-fit" in n for n in plan["notes"])


def test_refit_milestones_keep_the_reached_ones_and_rescale_the_rest():
    # 488 -> 400 steps at step 260 (gate at 135): 100 (at the gate step, 135) and 250 are
    # reached and stay; a later one is rescaled; one that would fall behind moves to 260.
    assert rm.refit_milestones([100, 250, 400], 488, 488, 400, 135, 260) == (135, 250, 328)
    assert rm.refit_milestones([100, 300], 488, 488, 300, 0, 260) == (100, 260)
    assert rm.refit_milestones([450], 488, 488, 300, 0, 260) == (277,)


# ---- the child runs in its own session; stop signals take the checkpoint path -----------------

def test_the_trainer_is_started_in_its_own_session_or_process_group(tmp_path, monkeypatch):
    seen = {}

    class NoPopen:
        def __init__(self, *a, **kw):
            seen.update(kw)
            raise OSError("not starting anything")

    monkeypatch.setattr(rm.subprocess, "Popen", NoPopen)
    with pytest.raises(OSError):
        rm.SubprocessChild(echo=lambda s: None)(["x"], tmp_path / "log", lambda e: None)
    expected = rm.childproc.popen_kwargs()
    assert {k: seen[k] for k in expected} == expected


def test_ctrl_c_is_forwarded_to_the_trainer_which_checkpoints(tmp_path):
    """The trainer no longer shares the terminal's process group: it only learns of a
    Ctrl+C (or SIGTERM / SIGHUP, which raise the same KeyboardInterrupt) from the
    launcher, which must send it."""
    script = tmp_path / "child.py"
    script.write_text(FAKE_CHILD, encoding="utf-8")
    marker = tmp_path / "marker.txt"

    def echo(line):
        if line.startswith("step 3/"):
            raise KeyboardInterrupt

    child = rm.SubprocessChild(poll_s=0.1, grace_s=30.0, kill_wait_s=5.0, echo=echo)
    with pytest.raises(KeyboardInterrupt):
        child([sys.executable, str(script), str(marker)], tmp_path / "child.log", lambda e: None)
    assert marker.read_text() == "interrupt checkpoint"


def test_a_stop_signal_raises_keyboard_interrupt_once():
    said = []
    handler = rm.childproc.stop_signal_handler(said.append, "[moe]")
    with pytest.raises(KeyboardInterrupt):
        handler(15, None)
    handler(1, None)                                    # a hangup after the SIGTERM: ignored
    assert len(said) == 2 and "again" in said[1] and said[0].startswith("[moe]")


SIGNAL_CHILD = r'''
import os, signal, sys, time
from pathlib import Path
marker, sid_out = Path(sys.argv[1]), Path(sys.argv[2])
sid_out.write_text(str(os.getsid(0)))
def stop(signum, frame):
    marker.write_text("interrupt checkpoint")
    print("checkpoint step 9 saved in 0.5 s (1.4 GB)", flush=True)
    sys.exit(130)
signal.signal(signal.SIGINT, stop)
for i in range(1, 2000):
    print(f"step {i}/2000  loss 3.0000  lr 1.00e-03  grad 1.000  1,000 tok/s", flush=True)
    time.sleep(0.05)
'''

SIGNAL_DRIVER = r'''
import importlib.util, sys
from pathlib import Path
root, child, marker, sid_out = sys.argv[1:5]
sys.path.insert(0, root)
spec = importlib.util.spec_from_file_location("run_moe", Path(root) / "scripts" / "remote" / "run_moe.py")
rm = importlib.util.module_from_spec(spec); sys.modules["run_moe"] = rm; spec.loader.exec_module(rm)
previous = rm.childproc.install_stop_signals(rm._echo, "[moe]")
runner = rm.SubprocessChild(grace_s=20.0, kill_wait_s=5.0, poll_s=0.2)
try:
    runner([sys.executable, child, marker, sid_out], Path(marker).with_suffix(".log"),
           lambda events: None)
except KeyboardInterrupt:
    print("[driver] stopped", flush=True)
    sys.exit(130)
sys.exit(0)
'''


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals and sessions")
@pytest.mark.parametrize("sig_name", ["SIGTERM", "SIGHUP"])
def test_sigterm_or_sighup_to_the_launcher_checkpoints_the_trainer(tmp_path, sig_name):
    import signal

    child, driver = tmp_path / "child.py", tmp_path / "driver.py"
    child.write_text(SIGNAL_CHILD, encoding="utf-8")
    driver.write_text(SIGNAL_DRIVER, encoding="utf-8")
    marker, sid_out = tmp_path / "marker", tmp_path / "sid"
    proc = subprocess.Popen([sys.executable, str(driver), str(ROOT), str(child), str(marker),
                             str(sid_out)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    for line in proc.stdout:
        if line.startswith("step 3/"):
            proc.send_signal(getattr(signal, sig_name))
            break
    rest = proc.stdout.read()
    assert proc.wait(timeout=60) == 130, rest
    assert marker.read_text() == "interrupt checkpoint"       # the trainer checkpointed
    assert "[driver] stopped" in rest
    assert int(sid_out.read_text()) != os.getsid(0)           # in a session of its own


# ---- sync.sh ---------------------------------------------------------------------------------

def _bash() -> str | None:
    """Git Bash on Windows (the one sync.sh is written for), else bash on PATH."""
    if os.name == "nt":
        found = shutil.which("bash")
        if found and "git" in found.lower():
            return found
        git = shutil.which("git")
        for parent in Path(git).parents if git else ():
            for cand in (parent / "bin" / "bash.exe", parent / "usr" / "bin" / "bash.exe"):
                if cand.is_file():
                    return str(cand)
        return None       # WSL's bash.exe cannot see Windows paths or this environment
    return shutil.which("bash")


BASH = _bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="needs bash (Git Bash on Windows)")


def _sync(tmp_path, *args, tool="rsync", keep=None):
    # A relative LOCAL_DIR reads the same in Git Bash and elsewhere (cygpath keeps it).
    env = {**os.environ, "SYNC_TOOL": tool, "LOCAL_DIR": "local-copy/quipu-moe",
           "HOME": tmp_path.as_posix()}
    env.pop("KEEP_LOCAL", None)
    if keep is not None:
        env["KEEP_LOCAL"] = keep
    return subprocess.run([BASH, "-s", "--", *args], input=SYNC.read_bytes().decode("utf-8"),
                          capture_output=True, text=True, encoding="utf-8", timeout=60,
                          env=env, cwd=tmp_path)


def test_sync_is_lf_and_strict():
    text = SYNC.read_bytes().decode("utf-8")
    assert "\r" not in text and "set -euo pipefail" in text
    assert "--delete" not in text                      # never deletes local files
    assert "ForwardAgent=no" in text and "ssh -A" not in text


@needs_bash
def test_sync_parses_with_bash_n():
    proc = subprocess.run([BASH, "-n"], input=SYNC.read_bytes().decode("utf-8"),
                          capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert proc.returncode == 0, proc.stderr


@needs_bash
def test_sync_dry_run_prints_the_rsync_commands(tmp_path):
    proc = _sync(tmp_path, "--dry-run", "1.2.3.4", "40022", "2")
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    local = "local-copy/quipu-moe"
    remote = "root@1.2.3.4:/workspace/quipu"
    # The key is ~/.ssh/id_ed25519 (Git Bash spells HOME /c/...), no agent forwarding.
    ssh = r"ssh -p 40022 -i \S+/\.ssh/id_ed25519 -o IdentitiesOnly=yes -o ForwardAgent=no"
    assert re.search(r'rsync -av --partial-dir=\.rsync-partial -e "' + ssh
                     + " -o BatchMode=yes", out), out
    assert f"{remote}/checkpoints/quipu-moe/latest.pt {local}/checkpoints/quipu-moe/latest.pt.incoming" in out
    assert f"{remote}/checkpoints/quipu-moe/step_NNNNNN.pt {local}/checkpoints/quipu-moe/" in out
    assert f"{remote}/checkpoints/quipu-moe/milestones/ {local}/checkpoints/quipu-moe/milestones/" in out
    assert "--exclude inductor-cache/" in out and f"{remote}/results/ {local}/results/" in out
    assert "test -f /workspace/quipu/results/moe/summary.md" in out
    assert "every 2 h" in out
    assert "--delete" not in out and not (tmp_path / "local-copy").exists()


@needs_bash
def test_sync_dry_run_falls_back_to_scp(tmp_path):
    proc = _sync(tmp_path, "--dry-run", "host.example", "22", tool="scp")
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "scp -P 22 -i" in out and "rsync" not in out
    assert all("BatchMode=yes" in ln for ln in out.splitlines()
               if ln.startswith(("scp ", "ssh ")))
    assert "root@host.example:/workspace/quipu/checkpoints/quipu-moe/step_NNNNNN.pt" in out
    assert "tar -C /workspace/quipu -cf - --exclude=inductor-cache" in out
    assert "every 3 h" in out                         # the default interval


@needs_bash
def test_sync_usage_errors(tmp_path):
    assert _sync(tmp_path, "--dry-run", "onlyhost").returncode == 2
    assert _sync(tmp_path, "--dry-run", "host", "notaport").returncode == 2
    assert _sync(tmp_path, "--dry-run", "host", "22", "0").returncode == 2
    assert _sync(tmp_path, "--prune-only").returncode == 2                  # needs KEEP_LOCAL
    assert _sync(tmp_path, "--dry-run", "host", "22", keep="0").returncode == 2
    assert _sync(tmp_path, "--dry-run", "host", "22", keep="two").returncode == 2


@needs_bash
def test_keep_local_prunes_old_copies_but_never_latest_or_milestones(tmp_path):
    ckpt = tmp_path / "local-copy" / "quipu-moe" / "checkpoints" / "quipu-moe"
    (ckpt / "milestones").mkdir(parents=True)
    for step in (100, 200, 300, 400, 500):
        (ckpt / f"step_{step:06d}.pt").write_bytes(b"x")
        (ckpt / "milestones" / f"step_{step:06d}.pt").write_bytes(b"m")
    (ckpt / "step_000600.pt.part").write_bytes(b"partial")
    # latest.pt points at step 200 (a pointer that lags the newest copies, e.g. a
    # restore): it must survive although it is not among the 2 newest.
    (ckpt / "latest.pt").write_bytes(b"\x80\x02}q\x00X\x04\x00\x00\x00fileq\x01X\x0e\x00\x00\x00"
                                     b"step_000200.ptq\x02s.")
    proc = _sync(tmp_path, "--prune-only", keep="2")
    assert proc.returncode == 0, proc.stderr
    left = sorted(p.name for p in ckpt.iterdir() if p.is_file())
    assert left == ["latest.pt", "step_000200.pt", "step_000400.pt", "step_000500.pt",
                    "step_000600.pt.part"]
    assert len(list((ckpt / "milestones").iterdir())) == 5          # untouched
    # Unset KEEP_LOCAL (the default) keeps everything.
    proc = _sync(tmp_path, "--dry-run", "1.2.3.4", "22")
    assert proc.returncode == 0 and "rm -f" not in proc.stdout
    assert len(list(ckpt.glob("step_*.pt"))) == 3
