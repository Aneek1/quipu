"""Licence filter: only the SPDX allow-list passes; SOURCES records attribution."""
import json

import pytest

from stepbuild.dataset.licence import (
    ALLOWED, append_source, is_allowed, licence_of, source_record,
)
from tests.stepbuild.fakegh import FakeGh, fixture, http_error, ok

RESPONSES = {
    "repos/acme/mit/license": ok(fixture("licence_mit.json")),
    "repos/acme/gpl/license": ok(fixture("licence_gpl.json")),
    "repos/acme/other/license": ok(fixture("licence_noassertion.json")),
    "repos/acme/none/license": http_error(404, "Not Found"),
}


def _runner():
    return FakeGh(lambda args: RESPONSES[args[0]])


def test_allow_list_is_the_spec_set():
    assert ALLOWED == {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "0BSD",
                       "Unlicense"}


def test_keeps_mit_and_drops_noassertion_none_and_gpl():
    runner = _runner()
    got = {r: licence_of(f"acme/{r}", runner=runner) for r in ("mit", "gpl", "other", "none")}
    assert got == {"mit": "MIT", "gpl": "GPL-3.0", "other": "NOASSERTION", "none": None}
    kept = [r for r, spdx in got.items() if is_allowed(spdx)]
    assert kept == ["mit"]
    for spdx in ("NOASSERTION", None, "GPL-3.0", "AGPL-3.0", "LGPL-2.1", "MPL-2.0", "mit"):
        assert not is_allowed(spdx), spdx
    for spdx in ALLOWED:
        assert is_allowed(spdx)


def test_licence_lookups_are_cached(tmp_path):
    runner = _runner()
    assert licence_of("acme/mit", runner=runner, cache_dir=tmp_path) == "MIT"
    assert licence_of("acme/none", runner=runner, cache_dir=tmp_path) is None
    n = len(runner.calls)
    assert licence_of("acme/mit", runner=runner, cache_dir=tmp_path) == "MIT"
    assert licence_of("acme/none", runner=runner, cache_dir=tmp_path) is None
    assert len(runner.calls) == n


def test_bad_repo_names_are_refused():
    with pytest.raises(ValueError):
        licence_of("../etc", runner=_runner())


def test_sources_record_and_append(tmp_path):
    path = tmp_path / "SOURCES.jsonl"
    append_source(path, "acme/mit", "MIT", "a" * 40, "b" * 40, tag="fullstack")
    append_source(path, "acme/two", "ISC", "c" * 40, "d" * 40)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0] == {"repo": "acme/mit", "licence": "MIT", "tag": "fullstack",
                       "first_sha": "a" * 40, "last_sha": "b" * 40,
                       "url": "https://github.com/acme/mit"}
    assert rows[1]["repo"] == "acme/two"
    with pytest.raises(ValueError, match="allow-list"):
        source_record("acme/gpl", "GPL-3.0", "a", "b")
