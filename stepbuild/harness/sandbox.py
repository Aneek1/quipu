"""The sandbox: a throwaway copy of the project template that a model builds into.

Every app run starts from `template/` (a Flask backend and a Vite + React
frontend that already pass all three checks), so a failing check always points at
something the model wrote, never at the scaffolding.

node_modules is the expensive part (~100 MB, minutes to install), so it is
installed once into a cache outside the repo and linked into each sandbox:

- The cache lives in `%LOCALAPPDATA%/quipu/stepbuild-npm-cache` (or
  `~/.cache/quipu/stepbuild-npm-cache`), overridable with env `STEPBUILD_CACHE`.
- It is keyed by a hash of the template's package.json, so changing a pinned
  version installs afresh instead of silently reusing stale packages.
- The install happens in a staging directory that is renamed into place only
  when `npm install` succeeds, and a `.complete` marker is written last, so an
  interrupted or failed install never leaves a half cache that later runs trust.
- Linking uses a directory junction on Windows (`mklink /J`, which needs no
  admin rights or developer mode, unlike a symlink) and a symlink elsewhere, and
  falls back to a full copy if linking fails. Vite writes `frontend/dist` inside
  the sandbox, never into the cache.

`write_blocks` is the second safety boundary after FileBlock validation: it
resolves every target (following links such as the node_modules junction) and
refuses anything that lands outside the sandbox root, checking all blocks before
writing any, so a rejected reply changes nothing on disk.

This is not a security boundary for untrusted code: checks run the model's code
with the user's rights. It is meant for our own models on our own specs.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from quipu.fsio import replace_with_retry
from stepbuild.harness.blocks import FileBlock
from stepbuild.harness.checks import run_with_timeout

TEMPLATE_DIR = Path(__file__).resolve().parent / "template"
NPM_INSTALL_TIMEOUT_S = 600
_COMPLETE = ".complete"
# Never copied from the template, in case someone built it in place.
_TEMPLATE_IGNORE = shutil.ignore_patterns("node_modules", "dist", "__pycache__", ".pytest_cache")


class SandboxError(RuntimeError):
    """The sandbox could not be prepared (for example, npm install failed)."""


@dataclasses.dataclass(frozen=True)
class Sandbox:
    root: Path   # project root: contains backend/ and frontend/


def default_cache_dir() -> Path:
    env = os.environ.get("STEPBUILD_CACHE")
    if env:
        return Path(env)
    local = os.environ.get("LOCALAPPDATA")
    base = Path(local) if local else Path.home() / ".cache"
    return base / "quipu" / "stepbuild-npm-cache"


def _cache_key(package_json: Path) -> str:
    """Hash of package.json with line endings normalised, so a CRLF checkout of the
    same file (git autocrlf) shares the cache with an LF one."""
    data = package_json.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()[:16]


def _npm_install(workdir: Path, timeout_s: int) -> None:
    npm = shutil.which("npm")
    if npm is None:
        raise SandboxError("npm not found on PATH: install Node.js to build the frontend")
    code, output = run_with_timeout(
        [npm, "install", "--no-audit", "--no-fund", "--loglevel=error"],
        cwd=workdir,
        env=os.environ.copy(),
        timeout_s=timeout_s,
    )
    if code is None:
        raise SandboxError(f"npm install timed out after {timeout_s}s in {workdir}")
    if code != 0:
        raise SandboxError(f"npm install failed (exit {code}) in {workdir}:\n{output}")


def ensure_node_modules(cache_dir: Path, timeout_s: int = NPM_INSTALL_TIMEOUT_S) -> Path:
    """Return the cached node_modules for the current template, installing it once."""
    package_json = TEMPLATE_DIR / "frontend" / "package.json"
    key_dir = Path(cache_dir) / _cache_key(package_json)
    node_modules = key_dir / "node_modules"
    if (key_dir / _COMPLETE).is_file() and node_modules.is_dir():
        return node_modules
    key_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{key_dir.name}.staging-", dir=key_dir.parent))
    try:
        shutil.copy2(package_json, staging / "package.json")
        _npm_install(staging, timeout_s)
        if not (staging / "node_modules").is_dir():
            raise SandboxError(f"npm install produced no node_modules in {staging}")
        (staging / _COMPLETE).write_text("ok\n", encoding="utf-8")
        if key_dir.exists():
            if (key_dir / _COMPLETE).is_file():
                return node_modules  # another process finished first; keep theirs
            shutil.rmtree(key_dir)   # leftover of an older, broken attempt
        replace_with_retry(staging, key_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return node_modules


def _make_link(target: Path, link: Path) -> None:
    """Create a directory link at `link` pointing to `target`; raise OSError if the
    platform refuses."""
    if sys.platform == "win32":
        proc = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0 or not link.is_dir():
            raise OSError(f"mklink /J failed: {(proc.stdout + proc.stderr).strip()}")
    else:
        os.symlink(target, link, target_is_directory=True)


def _link_dir(target: Path, link: Path) -> None:
    try:
        _make_link(target, link)
    except (OSError, subprocess.SubprocessError):
        shutil.copytree(target, link, symlinks=True)


def create_sandbox(dest_parent: Path, cache_dir: Path) -> Sandbox:
    """Copy the template into a new directory under `dest_parent` and link the cached
    node_modules into its frontend/. Deleting the sandbox later with shutil.rmtree
    removes the junction/symlink itself, not the cache behind it."""
    dest_parent = Path(dest_parent)
    dest_parent.mkdir(parents=True, exist_ok=True)
    node_modules = ensure_node_modules(Path(cache_dir))
    root = Path(tempfile.mkdtemp(prefix="app-", dir=dest_parent))
    shutil.copytree(TEMPLATE_DIR, root, ignore=_TEMPLATE_IGNORE, dirs_exist_ok=True)
    _link_dir(node_modules, root / "frontend" / "node_modules")
    return Sandbox(root)


def _target(root: Path, block: FileBlock) -> Path:
    target = (root / block.path).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"refusing to write {block.path!r}: it resolves outside the sandbox")
    return target


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    """Like quipu.fsio.write_text_atomic, but writes bytes so content keeps its LF
    line endings on Windows (text mode would turn them into CRLF)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(data)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    replace_with_retry(tmp, path)


def write_blocks(sandbox: Sandbox, blocks: Sequence[FileBlock]) -> None:
    """Write each block's content to its path under the sandbox root, atomically per
    file. Every path is checked before any file is written."""
    root = Path(sandbox.root).resolve()
    targets = [(_target(root, b), b) for b in blocks]
    for target, block in targets:
        _write_bytes_atomic(target, block.content.encode("utf-8"))
