"""The sandbox: a throwaway copy of the project template that a model builds into.

Every app run starts from `template/` (a Flask backend and a Vite + React
frontend that already pass all three checks), so a failing check always points at
something the model wrote, never at the scaffolding.

node_modules is the expensive part, so it is installed once into a cache outside
the repo and linked into each sandbox:

- The cache lives in `%LOCALAPPDATA%/quipu/stepbuild-npm-cache` (or
  `~/.cache/quipu/stepbuild-npm-cache`), overridable with env `STEPBUILD_CACHE`.
- It is filled with `npm ci` from the committed package-lock.json, so every
  machine gets the same package tree, and it is keyed by a hash of package.json
  plus package-lock.json, so changing either installs afresh instead of silently
  reusing stale packages.
- The install happens in a staging directory that is renamed into place only
  when `npm ci` succeeds, and a `.complete` marker is written last, so an
  interrupted or failed install never leaves a half cache that later runs trust.
  The marker alone is not trusted either: node_modules/vite/package.json must
  exist too, or the cache is rebuilt (it may have been emptied by a delete that
  followed a sandbox's link into it). Two processes doing the first install at
  once both install; the loser uses the winner's cache.
- Linking uses a directory junction on Windows (`mklink /J`, which needs no
  admin rights or developer mode, unlike a symlink) and a symlink elsewhere, and
  falls back to a full copy if linking fails. Vite writes `frontend/dist` inside
  the sandbox, never into the cache.

Delete sandboxes with `remove_sandbox()`, never with a hand-written recursive
delete: frontend/node_modules is a link into the shared cache, and a delete that
walks into it (for example one that trusts Path.is_dir(), which follows
junctions) empties the cache for every other sandbox. remove_sandbox removes the
link itself first, then the rest. Every caller (bench acceptance, the runner,
the benchmark run) must use it. shutil.rmtree on Python 3.13 also leaves the
cache alone, and a test pins that, but remove_sandbox does not rely on it.

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
_LOCKFILE = "package-lock.json"
# A file every real install has; without it the cache was emptied or is broken.
_VITE_MARKER = Path("vite") / "package.json"
# Never copied from the template, in case someone built it in place.
_TEMPLATE_IGNORE = shutil.ignore_patterns("node_modules", "dist", "__pycache__", ".pytest_cache")


class SandboxError(RuntimeError):
    """The sandbox could not be prepared (for example, npm ci failed)."""


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


def _cache_key(frontend_dir: Path) -> str:
    """Hash of package.json and package-lock.json with line endings normalised, so
    a CRLF checkout of the same files (git autocrlf) shares the cache with LF."""
    digest = hashlib.sha256()
    for name in ("package.json", _LOCKFILE):
        data = (Path(frontend_dir) / name).read_bytes().replace(b"\r\n", b"\n")
        digest.update(name.encode() + b"\0" + data + b"\0")
    return digest.hexdigest()[:16]


def _cache_valid(key_dir: Path) -> bool:
    return (key_dir / _COMPLETE).is_file() and (key_dir / "node_modules" / _VITE_MARKER).is_file()


def _npm_install(workdir: Path, timeout_s: int) -> None:
    npm = shutil.which("npm")
    if npm is None:
        raise SandboxError("npm not found on PATH: install Node.js to build the frontend")
    code, output = run_with_timeout(
        [npm, "ci", "--no-audit", "--no-fund", "--loglevel=error"],
        cwd=workdir,
        env=os.environ.copy(),
        timeout_s=timeout_s,
    )
    if code is None:
        raise SandboxError(f"npm ci timed out after {timeout_s}s in {workdir}")
    if code != 0:
        raise SandboxError(f"npm ci failed (exit {code}) in {workdir}:\n{output}")


def ensure_node_modules(cache_dir: Path, timeout_s: int = NPM_INSTALL_TIMEOUT_S) -> Path:
    """Return the cached node_modules for the current template, installing it once."""
    frontend = TEMPLATE_DIR / "frontend"
    key_dir = Path(cache_dir) / _cache_key(frontend)
    node_modules = key_dir / "node_modules"
    if _cache_valid(key_dir):
        return node_modules
    key_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{key_dir.name}.staging-", dir=key_dir.parent))
    try:
        for name in ("package.json", _LOCKFILE):
            shutil.copy2(frontend / name, staging / name)
        _npm_install(staging, timeout_s)
        if not (staging / "node_modules" / _VITE_MARKER).is_file():
            raise SandboxError(f"npm ci produced no node_modules/vite in {staging}")
        (staging / _COMPLETE).write_text("ok\n", encoding="utf-8")
        if key_dir.exists():
            if _cache_valid(key_dir):
                return node_modules  # another process finished first; keep theirs
            shutil.rmtree(key_dir)   # leftover of a broken or emptied cache
        try:
            replace_with_retry(staging, key_dir)
        except OSError:
            # Two first installs racing: the loser's rename fails because the
            # winner's directory is already there. Use it if it is complete.
            if _cache_valid(key_dir):
                return node_modules
            raise
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
    node_modules into its frontend/. Delete it with remove_sandbox()."""
    dest_parent = Path(dest_parent)
    dest_parent.mkdir(parents=True, exist_ok=True)
    node_modules = ensure_node_modules(Path(cache_dir))
    root = Path(tempfile.mkdtemp(prefix="app-", dir=dest_parent))
    shutil.copytree(TEMPLATE_DIR, root, ignore=_TEMPLATE_IGNORE, dirs_exist_ok=True)
    _link_dir(node_modules, root / "frontend" / "node_modules")
    return Sandbox(root)


def remove_sandbox(sandbox: Sandbox) -> None:
    """Delete a sandbox without touching the shared node_modules cache: remove the
    frontend/node_modules link itself first (never following it), then the tree.
    A node_modules that is a real directory (the copy fallback) is just deleted."""
    root = Path(sandbox.root)
    link = root / "frontend" / "node_modules"
    if link.is_junction():
        os.rmdir(link)       # removes the junction, not its target
    elif link.is_symlink():
        os.unlink(link)
    elif link.exists():
        shutil.rmtree(link)  # a private copy, safe to delete
    if root.exists():
        shutil.rmtree(root)


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
