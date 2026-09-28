"""Keep only permissively licensed repos, and record where the data came from
(spec §3.1, `licence.py`).

The licence is the one GitHub itself detects (`gh api repos/{repo}/license`,
`.license.spdx_id`). Only the SPDX ids in ALLOWED pass; everything else is
dropped, including "NOASSERTION" (GitHub found a licence file it could not
identify: it may say anything) and a repo with no licence at all (no licence
means all rights reserved, not public domain). Copyleft licences (GPL, AGPL,
LGPL, MPL, ...) are dropped because a model trained on them is a derived-work
question this project does not want to answer.

Responses are cached per repo under `cache_dir/licence/` like the search pages,
404 (no licence) included, so a rerun asks GitHub nothing it already asked.

SOURCES.jsonl holds one JSON object per repo that contributed examples: repo,
licence, tag, the first and last commit of the mined history, and the repo URL,
which is what an attribution notice needs.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Callable

from quipu.fsio import write_text_atomic
from stepbuild.dataset.discover import Runner, gh_json, valid_repo

ALLOWED = frozenset(
    {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "0BSD", "Unlicense"}
)


def is_allowed(spdx: str | None) -> bool:
    return spdx in ALLOWED


def licence_of(
    repo: str,
    runner: Runner = subprocess.run,
    cache_dir: Path | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> str | None:
    """The SPDX id GitHub detects for `repo` ("NOASSERTION" when it cannot tell),
    or None when the repo has no licence file."""
    if not valid_repo(repo):
        raise ValueError(f"repo must be 'owner/name', got {repo!r}")
    path = None
    if cache_dir is not None:
        key = hashlib.sha256(repo.casefold().encode("utf-8")).hexdigest()[:32]
        path = Path(cache_dir) / "licence" / f"{key}.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))["spdx_id"]
    response = gh_json([f"repos/{repo}/license"], runner=runner, sleep=sleep, not_found_ok=True)
    spdx = None
    if isinstance(response, dict):
        lic = response.get("license")
        if isinstance(lic, dict) and isinstance(lic.get("spdx_id"), str):
            spdx = lic["spdx_id"]
    if path is not None:
        write_text_atomic(path, json.dumps({"repo": repo, "spdx_id": spdx}) + "\n")
    return spdx


def source_record(
    repo: str, licence: str, first_sha: str, last_sha: str, tag: str | None = None
) -> dict:
    if not valid_repo(repo):
        raise ValueError(f"repo must be 'owner/name', got {repo!r}")
    if not is_allowed(licence):
        raise ValueError(f"{repo}: licence {licence!r} is not in the allow-list")
    return {
        "repo": repo,
        "licence": licence,
        "tag": tag,
        "first_sha": first_sha,
        "last_sha": last_sha,
        "url": f"https://github.com/{repo}",
    }


def append_source(
    path: Path, repo: str, licence: str, first_sha: str, last_sha: str, tag: str | None = None
) -> None:
    """Append one SOURCES.jsonl line (the build rewrites the file whole and
    atomically through source_record; this is for adding a single repo)."""
    line = json.dumps(source_record(repo, licence, first_sha, last_sha, tag)) + "\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line)
