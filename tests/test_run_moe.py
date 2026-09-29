"""scripts/remote/run_moe.py (plan Task M9): the throughput gate, the plan and GO file,
the long run under the spend guard, and scripts/remote/sync.sh. Everything runs on
the CPU with a fake clock and a fake trainer (no torch model is built), except one
test that drives a real child process through SIGINT / CTRL_BREAK."""
from __future__ import annotations

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
                 crash_at: int | None = None) -> None:
        self.clock = clock
        self.tps = tps                       # a number, or a function of the attempt number
        self.startup_s = startup_s
        self.crash_at = crash_at
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
    assert load_config(path) == load_config(MOE)


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


def _sync(tmp_path, *args, tool="rsync"):
    # A relative LOCAL_DIR reads the same in Git Bash and elsewhere (cygpath keeps it).
    env = {**os.environ, "SYNC_TOOL": tool, "LOCAL_DIR": "local-copy/quipu-moe",
           "HOME": tmp_path.as_posix()}
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
    assert re.search(r'rsync -av --partial -e "' + ssh, out), out
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
    assert "root@host.example:/workspace/quipu/checkpoints/quipu-moe/step_NNNNNN.pt" in out
    assert "tar -C /workspace/quipu -cf - --exclude=inductor-cache" in out
    assert "every 3 h" in out                         # the default interval


@needs_bash
def test_sync_usage_errors(tmp_path):
    assert _sync(tmp_path, "--dry-run", "onlyhost").returncode == 2
    assert _sync(tmp_path, "--dry-run", "host", "notaport").returncode == 2
    assert _sync(tmp_path, "--dry-run", "host", "22", "0").returncode == 2
