"""quipu/spend.py: the box-level spend ledger shared by the A/B orchestrator (M8)
and the rented-box launcher (M9). Every test uses a fake clock and a tmp ledger."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from quipu import spend
from quipu.spend import Ledger, LedgerError


class FakeClock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


RATE = 0.36   # $/h: 100 s = $0.01


class FakeSystem:
    """Stands in for crontab, pgrep-like cron detection and process spawning, so no
    test ever touches the real crontab or starts a real ticker process."""

    def __init__(self, crontab: str | None = "", cron_alive: bool = True,
                 posix: bool = True, nohup: bool = True) -> None:
        self.crontab = crontab          # None: no crontab binary
        self._cron_alive = cron_alive
        self.posix = posix
        self.nohup = nohup
        self.spawned: list[list[str]] = []
        self.runs: list[list[str]] = []

    def which(self, name):
        if name == "crontab":
            return None if self.crontab is None else "/usr/bin/crontab"
        if name == "nohup":
            return "/usr/bin/nohup" if self.nohup else None
        return None

    def run(self, args, input=None):
        import subprocess
        self.runs.append(list(args))
        if args == ["crontab", "-l"]:
            if self.crontab == "":
                return subprocess.CompletedProcess(args, 1, "", "no crontab for root\n")
            return subprocess.CompletedProcess(args, 0, self.crontab, "")
        if args == ["crontab", "-"]:
            self.crontab = input
            return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(f"unexpected command {args}")

    def cron_alive(self):
        return self._cron_alive

    def spawn(self, args, cwd):
        self.spawned.append(list(args))
        return 4242


@pytest.fixture(autouse=True)
def fake_system(monkeypatch):
    fake = FakeSystem()
    monkeypatch.setattr(spend, "SYSTEM", fake)
    return fake


def usd(seconds: float, rate: float = RATE) -> float:
    return seconds / 3600 * rate


def test_a_missing_ledger_is_empty_and_spends_nothing(tmp_path):
    led = Ledger.load(tmp_path / "spend.json", clock=FakeClock())
    assert led.spent_usd() == 0.0 and led.sessions == []
    assert not (tmp_path / "spend.json").exists()      # loading never writes


def test_spend_is_box_time_times_rate_and_is_written_at_once(tmp_path):
    clock = FakeClock()
    path = tmp_path / "spend.json"
    led = Ledger.load(path, clock=clock)
    led.start_session(RATE)
    assert path.exists()                                # box_start is on disk right away
    clock.t += 1000
    assert led.spent_usd() == pytest.approx(usd(1000))
    assert led.remaining(1.0) == pytest.approx(1.0 - usd(1000))


def test_a_kill_without_a_final_write_loses_at_most_one_tick(tmp_path):
    clock = FakeClock()
    path = tmp_path / "spend.json"
    led = Ledger.load(path, clock=clock)
    led.start_session(RATE)
    start = clock.t
    for _ in range(10):                  # ticking every 60 s, as the orchestrator does
        clock.t += 60
        led.tick()
    clock.t += 59                        # SIGKILL here: nothing more is written
    killed_at = clock.t
    del led
    # The box stopped with the process; a new box session starts much later.
    clock.t += 10_000
    again = Ledger.load(path, clock=clock)
    again.start_session(RATE)
    true_first = killed_at - start
    counted = again.spent_usd()
    assert counted <= usd(true_first)
    assert usd(true_first) - counted <= usd(60) + 1e-12


def test_two_box_sessions_accumulate(tmp_path):
    clock = FakeClock()
    path = tmp_path / "spend.json"
    led = Ledger.load(path, clock=clock)
    led.start_session(RATE)
    clock.t += 3600
    led.tick()
    clock.t += 5000                      # box off
    led2 = Ledger.load(path, clock=clock)
    led2.start_session(0.72)             # another box, another rate
    clock.t += 1800
    assert led2.spent_usd() == pytest.approx(usd(3600) + usd(1800, 0.72))
    assert len(json.loads(path.read_text("utf-8"))["sessions"]) == 2


def test_ensure_session_starts_one_only_on_first_use(tmp_path):
    clock = FakeClock()
    path = tmp_path / "spend.json"
    led = Ledger.load(path, clock=clock)
    assert led.ensure_session(RATE) is True
    clock.t += 100
    led.tick()
    other = Ledger.load(path, clock=clock)       # a second invocation on the same box
    assert other.ensure_session(RATE) is False
    clock.t += 100
    assert other.spent_usd() == pytest.approx(usd(200))
    assert len(other.sessions) == 1


def test_separate_processes_with_their_own_out_dirs_share_the_ledger(tmp_path, monkeypatch):
    shared = tmp_path / "box" / "spend.json"
    monkeypatch.setenv(spend.ENV_VAR, str(shared))
    assert spend.default_path() == shared
    clock = FakeClock()
    a = Ledger.load(clock=clock)
    a.start_session(RATE)
    clock.t += 600
    a.tick()
    b = Ledger.load(clock=clock)                  # same box, another tool
    clock.t += 600
    b.tick()
    assert b.spent_usd() == pytest.approx(usd(1200))
    assert a.spent_usd() == pytest.approx(usd(1200))


def test_the_default_ledger_is_results_spend_json_in_the_repo(monkeypatch):
    monkeypatch.delenv(spend.ENV_VAR, raising=False)
    assert spend.default_path() == Path(spend.__file__).resolve().parents[1] / "results" / "spend.json"


@pytest.mark.parametrize("text", [
    "", "{", "[]", '{"sessions": 3}', '{"sessions": [{"box_start": "x"}]}',
    '{"sessions": [{"box_start": 1, "usd_per_hour": 0.5, "last_seen": NaN}]}',
    '{"sessions": [{"box_start": 10, "usd_per_hour": 0.5, "last_seen": 5}]}',
    '{"sessions": [], "adjustments": {"k": "1"}}',
    '{"sessions": [{"box_start": 1, "usd_per_hour": -1, "last_seen": 2}]}',
])
def test_a_corrupt_ledger_is_a_hard_error(tmp_path, text):
    path = tmp_path / "spend.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(LedgerError):
        Ledger.load(path)


def test_an_unreadable_ledger_is_a_hard_error(tmp_path):
    (tmp_path / "spend.json").mkdir()                 # a directory: cannot be read
    with pytest.raises(LedgerError):
        Ledger.load(tmp_path / "spend.json")


def test_adjustments_are_idempotent_by_key(tmp_path):
    clock = FakeClock()
    path = tmp_path / "spend.json"
    led = Ledger.load(path, clock=clock)
    led.start_session(RATE)
    led.adjust("laptop-prep", 1.5)
    led.adjust("laptop-prep", 1.5)                    # passed again: not counted twice
    assert led.spent_usd() == pytest.approx(1.5)
    again = Ledger.load(path, clock=clock)
    again.adjust("laptop-prep", 1.5)
    assert again.spent_usd() == pytest.approx(1.5)
    again.adjust("laptop-prep", 2.0)                  # a correction replaces it
    assert again.spent_usd() == pytest.approx(2.0)
    again.adjust("earlier-box", 0.25)                 # another key adds
    assert again.spent_usd() == pytest.approx(2.25)
    with pytest.raises(ValueError):
        again.adjust("bad", float("nan"))


def test_tick_without_a_session_is_an_error(tmp_path):
    with pytest.raises(LedgerError):
        Ledger.load(tmp_path / "spend.json").tick()


def test_writes_are_atomic_and_merge_what_another_process_wrote(tmp_path):
    clock = FakeClock()
    path = tmp_path / "spend.json"
    a = Ledger.load(path, clock=clock)
    a.start_session(RATE)
    b = Ledger.load(path, clock=clock)
    b.adjust("from-b", 0.5)                           # b writes an adjustment
    clock.t += 120
    a.tick()                                          # a's stale copy must not drop it
    final = Ledger.load(path, clock=clock)
    assert final.adjustments == {"from-b": 0.5}
    assert final.sessions[-1]["last_seen"] == clock.t
    leftovers = [p for p in tmp_path.iterdir() if p.name not in ("spend.json", "spend.json.lock")]
    assert not leftovers                                                    # no temp files


def test_tick_if_due_writes_at_most_once_per_interval(tmp_path):
    clock = FakeClock()
    path = tmp_path / "spend.json"
    led = Ledger.load(path, clock=clock)
    led.start_session(RATE)
    clock.t += 30
    assert led.tick_if_due(60) is False
    clock.t += 31
    assert led.tick_if_due(60) is True


def test_cli_start_show_and_adjust(tmp_path, capsys, fake_system):
    path = tmp_path / "spend.json"
    assert spend.main(["--ledger", str(path), "start", "--usd-per-hour", "0.5",
                       "--box-start", "1000"]) == 0
    data = json.loads(path.read_text("utf-8"))
    assert data["sessions"][0]["box_start"] == 1000.0
    assert data["sessions"][0]["usd_per_hour"] == 0.5
    assert spend.main(["--ledger", str(path), "adjust", "--key", "x", "--usd", "0.1"]) == 0
    assert spend.main(["--ledger", str(path), "show"]) == 0
    assert "spent $" in capsys.readouterr().out
    path.write_text("garbage", encoding="utf-8")
    assert spend.main(["--ledger", str(path), "show"]) == 2



def test_cli_show_after_stop_does_not_claim_a_running_rate(tmp_path, capsys, fake_system):
    # After `spend stop` the box is not billed: "now $0.50/h" would say it still is.
    path = tmp_path / "spend.json"
    assert spend.main(["--ledger", str(path), "start", "--usd-per-hour", "0.5"]) == 0
    assert spend.main(["--ledger", str(path), "show"]) == 0
    assert "now $0.50/h" in capsys.readouterr().out
    assert spend.main(["--ledger", str(path), "stop"]) == 0
    capsys.readouterr()
    assert spend.main(["--ledger", str(path), "show"]) == 0
    out = capsys.readouterr().out
    assert "now $" not in out and "no running box session" in out

# ---- two writers: the lock and the merge ------------------------------------------------

def test_a_stale_in_memory_adjustment_never_overwrites_a_newer_disk_value(tmp_path):
    clock = FakeClock()
    path = tmp_path / "spend.json"
    a = Ledger.load(path, clock=clock)
    a.start_session(RATE)
    a.adjust("k", 1.0)
    b = Ledger.load(path, clock=clock)
    b.adjust("k", 2.0)                       # the newer value, from another process
    clock.t += 60
    a.tick()                                 # a still holds k = 1.0 in memory
    a.adjust("other", 0.5)
    assert Ledger.load(path).adjustments == {"k": 2.0, "other": 0.5}
    assert a.adjustments == {"k": 2.0, "other": 0.5}     # a now sees the disk's value


def test_a_start_is_not_lost_to_a_concurrent_tick(tmp_path, capsys):
    clock = FakeClock()
    path = tmp_path / "spend.json"
    a = Ledger.load(path, clock=clock)
    a.start_session(RATE)
    b = Ledger.load(path, clock=clock)
    clock.t += 120
    b.start_session(0.72)                    # a new session while a keeps ticking
    new_start = clock.t
    clock.t += 60
    a.tick()                                 # a's memory still has the old session last
    disk = Ledger.load(path, clock=clock)
    assert len(disk.sessions) == 2
    old, new = disk.sessions
    assert old["last_seen"] <= new_start     # the old session was never extended
    assert new["box_start"] == new_start and new["last_seen"] == clock.t
    assert "changed" in capsys.readouterr().err


def test_the_ledger_lock_excludes_a_second_writer(tmp_path):
    path = tmp_path / "spend.json"
    with spend._file_lock(spend.lock_path(path)):
        other = Ledger.load(path, clock=FakeClock())
        other.lock_timeout_s = 0.2
        with pytest.raises(LedgerError, match="lock"):
            other.start_session(RATE)
    other.start_session(RATE)                # released: it goes through
    assert len(Ledger.load(path).sessions) == 1


def test_a_read_during_a_windows_replace_is_retried(tmp_path, monkeypatch):
    path = tmp_path / "spend.json"
    led = Ledger.load(path, clock=FakeClock())
    led.start_session(RATE)
    real = Path.read_text
    fails = {"n": 2}

    def flaky(self, *a, **kw):
        if self == path and fails["n"]:
            fails["n"] -= 1
            raise PermissionError(13, "in use by a replace")
        return real(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", flaky)
    monkeypatch.setattr(spend.time, "sleep", lambda s: None)
    assert len(Ledger.load(path).sessions) == 1
    assert fails["n"] == 0


RACE_ADDER = """
import sys
from quipu.spend import Ledger
Ledger.load(sys.argv[1]).adjust("k" + sys.argv[2], 1.0)
"""
RACE_TICKER = """
import sys
from quipu.spend import Ledger
led = Ledger.load(sys.argv[1])
for _ in range(int(sys.argv[2])):
    led.tick()
"""


def test_concurrent_adjust_writers_and_a_ticker_lose_nothing(tmp_path):
    """The reviewer's race probe as a test: N processes each add one adjustment
    while another process ticks in a tight loop. Every adjustment survives."""
    import os
    import subprocess
    import sys

    root = Path(spend.__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root))
    n = 4
    for trial in range(3):
        path = tmp_path / f"race{trial}.json"
        Ledger.load(path).start_session(1.0)
        procs = [subprocess.Popen([sys.executable, "-c", RACE_TICKER, str(path), "150"],
                                  cwd=root, env=env)]
        procs += [subprocess.Popen([sys.executable, "-c", RACE_ADDER, str(path), str(i)],
                                   cwd=root, env=env) for i in range(n)]
        assert [p.wait(timeout=120) for p in procs] == [0] * (n + 1)
        data = json.loads(path.read_text("utf-8"))
        assert sorted(data["adjustments"]) == [f"k{i}" for i in range(n)], trial
        assert len(data["sessions"]) == 1


# ---- idle box time: the per-minute ticker ------------------------------------------------

def test_start_installs_a_tagged_crontab_line_once(tmp_path, fake_system, capsys):
    fake_system.crontab = "0 3 * * * /usr/bin/backup\n"
    path = tmp_path / "spend.json"
    assert spend.main(["--ledger", str(path), "start", "--usd-per-hour", "0.5"]) == 0
    assert spend.main(["--ledger", str(path), "start", "--usd-per-hour", "0.5"]) == 0
    lines = [ln for ln in fake_system.crontab.splitlines() if ln.strip()]
    assert lines[0] == "0 3 * * * /usr/bin/backup"            # other lines are kept
    ours = [ln for ln in lines if spend.CRON_TAG in ln]
    assert len(ours) == 1                                      # idempotent
    line = ours[0]
    assert line.startswith("* * * * * cd ")
    assert "-m quipu.spend --ledger" in line and str(path.resolve()) in line
    assert " tick >/dev/null 2>&1" in line
    assert fake_system.spawned == []
    assert "crontab" in capsys.readouterr().out                # says what it installed


def test_without_crontab_start_spawns_a_nohup_tick_loop(tmp_path, fake_system, capsys):
    fake_system.crontab = None
    path = tmp_path / "spend.json"
    assert spend.main(["--ledger", str(path), "start", "--usd-per-hour", "0.5"]) == 0
    (args,) = fake_system.spawned
    assert args[0] == "nohup" and args[2:5] == ["-m", "quipu.spend", "--ledger"]
    assert args[-3:] == ["tick", "--loop", "60"]
    assert spend.ticker_pid_path(path).read_text().strip() == "4242"
    assert "4242" in capsys.readouterr().out


def test_a_crontab_without_a_running_cron_daemon_falls_back_to_the_loop(tmp_path, fake_system):
    fake_system._cron_alive = False
    path = tmp_path / "spend.json"
    assert spend.main(["--ledger", str(path), "start", "--usd-per-hour", "0.5"]) == 0
    assert fake_system.spawned and spend.CRON_TAG not in (fake_system.crontab or "")


def test_stop_removes_the_ticker_and_ends_the_session(tmp_path, fake_system, monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(spend.time, "time", clock)
    fake_system.crontab = "0 3 * * * /usr/bin/backup\n"
    path = tmp_path / "spend.json"
    assert spend.main(["--ledger", str(path), "start", "--usd-per-hour", "0.36"]) == 0
    spend.ticker_pid_path(path).write_text("777")             # a loop from an earlier start
    clock.t += 1000
    assert spend.main(["--ledger", str(path), "stop"]) == 0
    assert spend.CRON_TAG not in fake_system.crontab
    assert "/usr/bin/backup" in fake_system.crontab
    assert not spend.ticker_pid_path(path).exists()
    clock.t += 5000                                            # the box is gone
    led = Ledger.load(path, clock=clock)
    assert led.spent_usd() == pytest.approx(usd(1000))         # not counted to now
    with pytest.raises(LedgerError, match="ended"):
        led.tick()                                             # a stray tick cannot revive it
    assert led.ensure_session(RATE) is True                    # the next tool starts a new one


def test_stop_refuses_while_a_tool_is_ticking_unless_forced(tmp_path, monkeypatch, capsys):
    clock = FakeClock()
    monkeypatch.setattr(spend.time, "time", clock)
    path = tmp_path / "spend.json"
    assert spend.main(["--ledger", str(path), "start", "--usd-per-hour", "0.36",
                       "--no-ticker"]) == 0
    tool = Ledger.load(path, clock=clock)
    tool.tool = "run_moe"
    clock.t += 30
    tool.tick()                                        # run_moe is running
    idle = Ledger.load(path, clock=clock)
    clock.t += 30
    idle.tick()                                        # the idle ticker: no tool tag
    capsys.readouterr()
    assert spend.main(["--ledger", str(path), "stop"]) == 2
    err = capsys.readouterr().err
    assert "run_moe" in err and "30 s ago" in err and "--force" in err
    assert not Ledger.load(path, clock=clock).current.get("ended")
    # Two minutes after the tool's last tick it is taken as gone.
    clock.t += spend.TOOL_ACTIVE_S
    assert spend.main(["--ledger", str(path), "stop"]) == 0
    assert Ledger.load(path, clock=clock).current["ended"]


def test_stop_force_ends_the_session_under_a_running_tool(tmp_path, monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(spend.time, "time", clock)
    path = tmp_path / "spend.json"
    assert spend.main(["--ledger", str(path), "start", "--usd-per-hour", "0.36",
                       "--no-ticker"]) == 0
    tool = Ledger.load(path, clock=clock)
    tool.tool = "ab_runs"
    tool.tick()
    assert spend.main(["--ledger", str(path), "stop", "--force"]) == 0
    assert Ledger.load(path, clock=clock).current["ended"]
    assert Ledger.load(path, clock=clock).active_tool() is None


def test_the_tool_tag_survives_a_round_trip_and_bad_tags_are_refused(tmp_path):
    clock = FakeClock()
    path = tmp_path / "spend.json"
    led = Ledger.load(path, clock=clock)
    led.tool = "ab_runs"
    led.start_session(RATE)
    led.tick()
    again = Ledger.load(path, clock=clock)
    assert again.current["tool"] == "ab_runs" and again.current["tool_seen"] == clock.t
    assert again.active_tool() == ("ab_runs", 0.0)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["sessions"][-1]["tool_seen"] = "soon"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(LedgerError):
        Ledger.load(path, clock=clock)


def test_the_tick_loop_exits_when_its_pid_file_is_replaced_or_the_session_ends(tmp_path):
    import os

    clock = FakeClock()
    path = tmp_path / "spend.json"
    Ledger.load(path, clock=clock).start_session(RATE)
    pid_file = spend.ticker_pid_path(path)
    pid_file.write_text(str(os.getpid()))
    naps = []

    def sleep(s):
        naps.append(s)
        clock.t += s
        if len(naps) == 3:
            pid_file.write_text("1")                           # a newer start took over

    assert spend.tick_loop(path, 60, clock=clock, sleep=sleep) == 0
    assert len(naps) == 3
    assert Ledger.load(path).sessions[-1]["last_seen"] == clock.t - 60
    # An ended session: the loop stops at once.
    pid_file.write_text(str(os.getpid()))
    Ledger.load(path, clock=clock).end_session()
    naps.clear()
    assert spend.tick_loop(path, 60, clock=clock, sleep=sleep) == 0
    assert naps == []


def test_a_new_start_after_an_unended_session_warns_about_the_uncounted_gap(tmp_path, capsys):
    clock = FakeClock()
    path = tmp_path / "spend.json"
    led = Ledger.load(path, clock=clock)
    led.start_session(RATE)
    clock.t += 600
    led.tick()                               # last seen here; the box died unticked
    clock.t += 7200
    again = Ledger.load(path, clock=clock)
    again.start_session(RATE)
    err = capsys.readouterr().err
    assert "WARNING" in err and "2:00" in err and "spend adjust" in err
    assert f"${usd(7200):.2f}" in err
    assert again.sessions[0]["last_seen"] == 1_000_600.0      # ended at its last tick
    assert again.spent_usd() == pytest.approx(usd(600))


def test_start_after_a_stopped_session_does_not_warn(tmp_path, capsys):
    clock = FakeClock()
    path = tmp_path / "spend.json"
    led = Ledger.load(path, clock=clock)
    led.start_session(RATE)
    clock.t += 600
    led.end_session()
    clock.t += 7200
    Ledger.load(path, clock=clock).start_session(RATE)
    assert "WARNING" not in capsys.readouterr().err


def test_cli_start_no_ticker_installs_nothing(tmp_path, fake_system):
    path = tmp_path / "spend.json"
    assert spend.main(["--ledger", str(path), "start", "--usd-per-hour", "0.5",
                       "--no-ticker"]) == 0
    assert fake_system.runs == [] and fake_system.spawned == []


# ---- show --plus / --usd-only fail closed (an A/B budget from a shell substitution) ------------

def test_show_plus_prints_spent_plus_the_amount(tmp_path, monkeypatch, capsys):
    clock = FakeClock()
    monkeypatch.setattr(spend.time, "time", clock)
    path = tmp_path / "spend.json"
    assert spend.main(["--ledger", str(path), "start", "--usd-per-hour", "0.36",
                       "--no-ticker"]) == 0
    assert spend.main(["--ledger", str(path), "adjust", "--key", "cpu-box", "--usd", "0.5"]) == 0
    clock.t += 1000                                           # $0.10 of box time
    capsys.readouterr()
    assert spend.main(["--ledger", str(path), "show", "--plus", "3"]) == 0
    assert capsys.readouterr().out.strip() == "3.6000"
    assert spend.main(["--ledger", str(path), "show", "--usd-only"]) == 0
    assert capsys.readouterr().out.strip() == "0.6000"


@pytest.mark.parametrize("flags", [["--plus", "3"], ["--usd-only"]])
def test_show_for_scripts_fails_closed_on_a_missing_ledger(tmp_path, capsys, flags):
    # `AB=$(python3 -m quipu.spend show --plus 3) || exit 1`: a missing ledger must not
    # read as $0 spent (the A/B would get the whole box budget), and print nothing.
    path = tmp_path / "spend.json"
    assert spend.main(["--ledger", str(path), "show", *flags]) != 0
    out = capsys.readouterr()
    assert out.out == "" and "no box session" in out.err


@pytest.mark.parametrize("flags", [["--plus", "3"], ["--usd-only"]])
def test_show_for_scripts_fails_closed_on_a_ledger_without_a_session(tmp_path, capsys, flags):
    path = tmp_path / "spend.json"
    assert spend.main(["--ledger", str(path), "adjust", "--key", "x", "--usd", "1"]) == 0
    capsys.readouterr()
    assert spend.main(["--ledger", str(path), "show", *flags]) != 0
    out = capsys.readouterr()
    assert out.out == "" and "no box session" in out.err


@pytest.mark.parametrize("flags", [["--plus", "3"], ["--usd-only"]])
def test_show_for_scripts_fails_closed_on_a_corrupt_ledger(tmp_path, capsys, flags):
    path = tmp_path / "spend.json"
    path.write_text("{ truncated", encoding="utf-8")
    assert spend.main(["--ledger", str(path), "show", *flags]) != 0
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("bad", ["nan", "inf", "-1"])
def test_show_plus_refuses_a_non_finite_or_negative_amount(tmp_path, capsys, bad):
    path = tmp_path / "spend.json"
    assert spend.main(["--ledger", str(path), "start", "--usd-per-hour", "0.36",
                       "--no-ticker"]) == 0
    capsys.readouterr()
    assert spend.main(["--ledger", str(path), "show", "--plus", bad]) != 0
    assert capsys.readouterr().out == ""


# ---- a tool that exits releases the ledger: `spend stop` right after it succeeds ---------------

def test_release_tool_clears_this_tools_tag_so_stop_succeeds_at_once(tmp_path, monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(spend.time, "time", clock)
    path = tmp_path / "spend.json"
    assert spend.main(["--ledger", str(path), "start", "--usd-per-hour", "0.36",
                       "--no-ticker"]) == 0
    tool = Ledger.load(path, clock=clock)
    tool.tool = "run_moe"
    tool.tick()
    assert Ledger.load(path, clock=clock).active_tool()[0] == "run_moe"
    assert tool.release_tool() is True
    cur = Ledger.load(path, clock=clock).current
    assert "tool" not in cur and "tool_seen" not in cur
    assert cur["last_seen"] == clock.t                        # the box time is kept
    assert spend.main(["--ledger", str(path), "stop"]) == 0   # no --force needed
    assert Ledger.load(path, clock=clock).current["ended"]


def test_release_tool_leaves_another_tools_tag_alone(tmp_path):
    clock = FakeClock()
    path = tmp_path / "spend.json"
    Ledger.load(path, clock=clock).start_session(RATE)
    ab = Ledger.load(path, clock=clock)
    ab.tool = "ab_runs"
    ab.tick()
    moe = Ledger.load(path, clock=clock)
    moe.tool = "run_moe"
    assert moe.release_tool() is False                        # ab_runs is still running
    assert Ledger.load(path, clock=clock).active_tool()[0] == "ab_runs"
    idle = Ledger.load(path, clock=clock)                     # no tool name: never releases
    assert idle.release_tool() is False
    assert Ledger.load(path, clock=clock).active_tool()[0] == "ab_runs"


def test_release_tool_on_a_missing_or_ended_ledger_is_a_no_op(tmp_path):
    clock = FakeClock()
    path = tmp_path / "spend.json"
    led = Ledger.load(path, clock=clock)
    led.tool = "run_moe"
    assert led.release_tool() is False
    assert not path.exists()                                  # nothing written
    led.start_session(RATE)
    led.tick()
    led.end_session()
    assert led.release_tool() is True                         # the tag goes; still ended
    assert Ledger.load(path, clock=clock).current["ended"]
