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
    assert not [p for p in tmp_path.iterdir() if p.name != "spend.json"]   # no temp files


def test_tick_if_due_writes_at_most_once_per_interval(tmp_path):
    clock = FakeClock()
    path = tmp_path / "spend.json"
    led = Ledger.load(path, clock=clock)
    led.start_session(RATE)
    clock.t += 30
    assert led.tick_if_due(60) is False
    clock.t += 31
    assert led.tick_if_due(60) is True


def test_cli_start_show_and_adjust(tmp_path, capsys):
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
