"""A scripted stand-in for subprocess.run that answers `gh api` calls from the
recorded JSON under fixtures/gh, so no test touches the network."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable

FIXTURES = Path(__file__).parent / "fixtures" / "gh"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def ok(body) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], 0, stdout=json.dumps(body), stderr="")


def http_error(status: int, message: str) -> subprocess.CompletedProcess:
    body = json.dumps({"message": message, "status": str(status)})
    return subprocess.CompletedProcess([], 1, stdout=body, stderr=f"gh: {message} (HTTP {status})\n")


class FakeGh:
    """`handler(args)` maps the arguments after `gh api` to a CompletedProcess.
    Every call is recorded in `calls`; `script` answers the first calls in order
    before the handler is used (for scripted failures)."""

    def __init__(self, handler: Callable[[list[str]], subprocess.CompletedProcess],
                 script: list[subprocess.CompletedProcess] | None = None) -> None:
        self.handler = handler
        self.script = list(script or [])
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        assert cmd[:2] == ["gh", "api"], cmd
        assert kwargs.get("timeout"), "every gh call needs a timeout"
        args = list(cmd[2:])
        self.calls.append(args)
        if self.script:
            return self.script.pop(0)
        return self.handler(args)


def query_of(args: list[str]) -> str | None:
    for a in args:
        if a.startswith("q="):
            return a[2:]
    return None


def page_of(args: list[str]) -> int:
    for a in args:
        if a.startswith("page="):
            return int(a[5:])
    return 1


def search_handler(pages: dict[str, dict[int, dict]]):
    """pages[query][page] -> response; a missing page answers HTTP 422."""
    def handler(args):
        q = query_of(args)
        resp = pages.get(q, {}).get(page_of(args))
        if resp is None:
            return http_error(422, "Cannot access beyond the first 1000 results")
        return ok(resp)
    return handler
