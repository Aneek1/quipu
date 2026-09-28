# Stepbuild Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared core of the stepwise app builder: a harness that drives any `StepModel` through a fixed 6-step Flask + React build with real checks and retries, a 10-app benchmark with hidden acceptance tests and reference solutions, and a dataset builder that turns permissively licensed git history into full-file step examples.

**Architecture:** New top-level package `stepbuild/` in the quipu repo with three sub-packages: `harness/` (blocks, plan, sandbox, checks, retrieve, model, runner), `bench/` (app specs + acceptance + reference, runner, scores) and `dataset/` (discover, licence, mine, filters, format, split, build). The harness is model-agnostic; the benchmark proves itself with reference solutions replayed through a scripted model; the dataset builder is pure-function core plus thin `gh`/`git` adapters tested against fixtures.

**Tech Stack:** Python 3.13 (uv project), pytest, Flask, pyflakes, tiktoken (for BM25 token ids, reusing `quipu.memory.bm25`), Node 24 / npm 11 + Vite + React for the frontend template, `gh` CLI (logged in as Aneek1) and `git`.

**Spec:** `docs/superpowers/specs/2026-09-28-stepbuild-core-design.md` — read it before starting any task.

---

## Standing rules (every task)

- Branch `pipeline-114m`. Test command, always exactly: `CUDA_VISIBLE_DEVICES= python -m uv run python -m pytest -m "not gpu_gate" -q` (plain `uv run pytest` falls through to the global Python on this machine). Run only the new test file while iterating (`... -m pytest tests/stepbuild/test_x.py -q`), full suite before committing.
- TDD: write the failing test, see it fail for the right reason, implement, see it pass.
- Dependencies via `python -m uv add <pkg>` (or `--dev`); commit `uv.lock`. Never pip-install globally. After adding anything, confirm torch still reports `+cu128` and `cuda.is_available() True`.
- While iterating, `-m "not gpu_gate and not npm"` skips the slow npm gate; the full command must still pass before each commit.
- Commits: plain sentences, **no trailers of any kind**, never push. Never commit `data/`, `results/`, `checkpoints/`, `hf_export/`, `node_modules/`, or the npm/template cache.
- Mutation checks are required where a task lists them: back the file up to the session scratchpad, mutate, show the named test fails, restore **by copying the backup** (never `git checkout -- <file>`), confirm with `cmp`.
- Style: follow `quipu/` — explanatory docstrings that say *why*, frozen dataclasses, strict validation with clear errors, atomic writes via `quipu.fsio.write_text_atomic` / `replace_with_retry`.
- Tests that need npm or git carry markers: `@pytest.mark.npm` (skipped when `npm` is not on PATH) and `@pytest.mark.git` (skipped when `git` is not). Register them in `tests/conftest.py` the same way `cuda` is registered. Nothing in the unit suite touches the network.

## File structure

```
stepbuild/
  __init__.py
  harness/
    __init__.py
    blocks.py        FILE-block parse/render + path validation
    plan.py          Step/AppPlan dataclasses, the fixed 6-step build order
    template/        project skeleton copied into every sandbox
      backend/app.py, backend/models.py, backend/requirements.txt, backend/tests/conftest.py
      frontend/package.json, frontend/vite.config.js, frontend/index.html, frontend/src/main.jsx, frontend/src/App.jsx
    sandbox.py       materialise template + node_modules cache into a temp dir
    checks.py        pytest / pyflakes / npm build runners with timeouts
    model.py         StepModel protocol + ScriptedModel
    retrieve.py      BM25 example retrieval over the step dataset
    prompt.py        build the messages for a step
    runner.py        plan → step → parse → write → check → retry loop, traces
  bench/
    __init__.py
    apps/<name>/spec.md
    apps/<name>/acceptance/test_acceptance.py
    apps/<name>/reference/step_1 … step_5/  (FILE-block replies, one .txt per step)
    acceptance.py    run an app's hidden acceptance tests against a finished project
    run.py           run a StepModel over all apps, write results + summary
  dataset/
    __init__.py
    filters.py       commit/file/message filters (pure)
    format.py        example → chat JSONL (pure)
    split.py         repo-level split (pure, deterministic)
    discover.py      gh search → tagged candidate repos (cached)
    licence.py       SPDX allow-list + SOURCES.jsonl
    mine.py          git history → examples
    build.py         CLI orchestration + report
tests/stepbuild/
  test_blocks.py test_plan.py test_sandbox.py test_checks.py test_model.py
  test_retrieve.py test_prompt.py test_runner.py test_bench_reference.py test_bench_run.py
  test_filters.py test_format.py test_split.py test_discover.py test_licence.py test_mine.py test_build.py
  fixtures/…
```

Add `"stepbuild"` to `[tool.hatch.build.targets.wheel] packages` in `pyproject.toml` (Task 1) so `import stepbuild` works under `uv run`.

---

### Task 1: Package skeleton and FILE blocks

**Files:** Create `stepbuild/__init__.py`, `stepbuild/harness/__init__.py`, `stepbuild/harness/blocks.py`, `tests/stepbuild/__init__.py`, `tests/stepbuild/test_blocks.py`. Modify `pyproject.toml` (packages), `tests/conftest.py` (markers `npm`, `git`).

Interface:
```python
@dataclass(frozen=True)
class FileBlock:
    path: str      # POSIX-style relative path, validated
    content: str   # exact file contents, trailing newline normalised to exactly one

class BlockError(ValueError): ...

def render_blocks(blocks: Sequence[FileBlock]) -> str
def parse_blocks(text: str, allowed: Collection[str] | None = None) -> list[FileBlock]
```
Format: a line `=== FILE: <path> ===`, the contents, a line `=== END FILE ===`. Text outside blocks is ignored (models chat around their output).

- [ ] Write failing tests:
  - round trip: `parse_blocks(render_blocks(bs)) == bs` for two files incl. one with blank lines and one with `===` inside content that is not a marker line;
  - chatter before/between/after blocks is ignored;
  - raises `BlockError` for: no blocks at all; unterminated block; duplicate path; absolute path (`/etc/x`, `C:\x`); `..` segment; backslashes (normalise? No — reject, message says use `/`); empty path; path not in `allowed` when `allowed` is given (message lists allowed files).
- [ ] Run, see failures. Implement. Run, see passes.
- [ ] Register `npm`/`git` markers in `tests/conftest.py` with skip logic: `shutil.which("npm") is None` / `shutil.which("git") is None`.
- [ ] Mutation: accept `..` paths → the traversal test fails. Restore.
- [ ] Full suite green. Commit: "Add the stepbuild package with FILE-block parsing and path validation".

### Task 2: The build plan

**Files:** Create `stepbuild/harness/plan.py`, `tests/stepbuild/test_plan.py`.

```python
@dataclass(frozen=True)
class Step:
    number: int                     # 1..6
    key: str                        # "model" | "routes" | "api_tests" | "components" | "wiring" | "run"
    title: str                      # instruction shown to the model, includes the app name
    allowed_files: tuple[str, ...]  # files the model may write in this step
    checks: tuple[str, ...]         # subset of ("pyflakes", "pytest", "npm_build")
    model_step: bool                # False for "run" (checks only, no model call)

@dataclass(frozen=True)
class AppPlan:
    app: str
    spec: str
    steps: tuple[Step, ...]

def make_plan(app: str, spec: str) -> AppPlan
```
Fixed order and files (these are the only files the template leaves for the model; see Task 3):
1 `model`: `backend/models.py` — checks `pyflakes`
2 `routes`: `backend/app.py` — checks `pyflakes`, `pytest` (changed in review: the template's `test_smoke.py` makes pytest meaningful here, see the contract below)
3 `api_tests`: `backend/tests/test_api.py` — checks `pyflakes`, `pytest`
4 `components`: `frontend/src/components/List.jsx`, `frontend/src/components/Form.jsx` — checks `npm_build`
5 `wiring`: `frontend/src/api.js`, `frontend/src/App.jsx` — checks `npm_build`
6 `run`: no files, `model_step=False` — checks `pyflakes`, `pytest`, `npm_build`

- [ ] Tests: six steps in order; numbers 1..6; step 6 has no files and `model_step False`; every title contains the app name; `make_plan` rejects empty app/spec; plans are equal for equal inputs (frozen).
- [ ] Implement, pass, commit: "Add the fixed six-step Flask + React build plan".

### Task 3: Template, sandbox and checks

**Files:** Create `stepbuild/harness/template/**` (listed above), `stepbuild/harness/sandbox.py`, `stepbuild/harness/checks.py`, `tests/stepbuild/test_sandbox.py`, `tests/stepbuild/test_checks.py`. Add dependencies: `python -m uv add flask pyflakes`.

Template contents (keep minimal, all must pass checks as-is):
- `backend/app.py`: `from flask import Flask; app = Flask(__name__)` and a `create_app()` returning it — **placeholder the model replaces in step 2**; `backend/models.py`: empty module with a docstring — replaced in step 1; `backend/requirements.txt`: `flask`, `pytest`; `backend/tests/conftest.py`: fixture `client` built from `create_app()` with `app.config["TESTING"] = True`.
- `frontend/`: `package.json` (react, react-dom, vite, @vitejs/plugin-react; script `"build": "vite build"`), `vite.config.js`, `index.html`, `src/main.jsx` rendering `<App/>`, `src/App.jsx` placeholder, `src/components/.gitkeep`.

```python
@dataclass(frozen=True)
class Sandbox:
    root: Path
def create_sandbox(dest_parent: Path, cache_dir: Path) -> Sandbox
    # copy template; ensure node_modules exists in cache_dir (npm install once, 10-min timeout);
    # link it into frontend/ (directory junction on Windows via `mklink /J`, symlink elsewhere; fall back to copy)
def write_blocks(sandbox: Sandbox, blocks: Sequence[FileBlock]) -> None   # atomic per file, creates parents

@dataclass(frozen=True)
class CheckResult:
    name: str; passed: bool; output: str; seconds: float   # output = last 60 lines of stdout+stderr
def run_checks(sandbox: Sandbox, names: Sequence[str], timeout_s: int = 180) -> list[CheckResult]
```
Checks: `pyflakes` → `python -m pyflakes backend` (fails only on reported problems); `pytest` → `python -m pytest -q backend/tests` with cwd `backend`; `npm_build` → `npm run build` in `frontend`. All via `subprocess.run` with timeout; a timeout is a failed check with output `"timed out after Ns"`. Checks run with `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` set so the network is unreachable (best effort, documented in the module docstring). Use the **current interpreter** (`sys.executable`) for Python checks.

- [ ] Tests (`npm`-marked where npm is needed): template passes `pyflakes` and `pytest` (0 tests collected must count as pass — treat pytest exit code 5 as pass for the template only via `allow_no_tests=True`); template passes `npm_build`; `write_blocks` creates nested files and refuses paths escaping `root`; a syntax error file fails `pyflakes` with the file name in output; a failing test fails `pytest`; a JSX syntax error fails `npm_build`; timeout path returns failed with the timeout message (use `timeout_s=1` and a test that sleeps).
- [ ] The npm cache lives outside the repo: default `%LOCALAPPDATA%/quipu/stepbuild-npm-cache` (or `~/.cache/quipu/stepbuild-npm-cache`), overridable by env `STEPBUILD_CACHE`.
- [ ] Commit: "Add the Flask + React sandbox template and the pytest, pyflakes and npm build checks".

### Task 4: Benchmark apps — infrastructure and first two apps

**Files:** Create `stepbuild/bench/__init__.py`, `stepbuild/bench/acceptance.py`, `stepbuild/bench/apps/todo/**`, `stepbuild/bench/apps/notes/**`, `tests/stepbuild/test_bench_reference.py`.

Per app:
- `spec.md`: one paragraph describing entities, fields, validation and endpoints in plain words (not a schema dump).
- `reference/step_1.txt … step_5.txt`: the exact FILE-block reply a perfect model would give for steps 1–5 (render with `render_blocks`). They must satisfy each step's `allowed_files`.
- `acceptance/test_acceptance.py`: pytest file importing `create_app` from the finished project (the acceptance runner puts the sandbox `backend/` on `sys.path`). Tests the REST contract stated in the spec: create (201 + body), list, get one, update, delete (then 404), validation error (400) for a missing required field. Plus nothing frontend-specific (frontend is covered by `npm_build`).

```python
def run_acceptance(app: str, project_root: Path, timeout_s: int = 180) -> CheckResult
    # copies acceptance/ into a temp dir, runs pytest with rootdir there and backend on PYTHONPATH
```
- [ ] Gate test (`npm`-marked because it builds the frontend): for each app present, replay `reference/step_N.txt` into a fresh sandbox, run every step's checks (all pass) and then `run_acceptance` (passes). Also: acceptance **fails** on the bare template (proves the tests test something).
- [ ] Commit: "Add benchmark infrastructure with the todo and notes apps and their reference solutions".

### Task 5: Remaining eight benchmark apps

**Files:** `stepbuild/bench/apps/{bookmarks,recipes,expenses,reading_list,habits,contacts,inventory,todo_auth}/**`.

Same structure as Task 4. Specific requirements beyond CRUD: `expenses` rejects negative amounts (400); `habits` has a check-in endpoint that is idempotent per day; `inventory` rejects stock below zero on decrement (409); `todo_auth` uses Flask session login (register, login, logout; todos are per user; 401 when logged out). Keep every app small enough that each reference file is under 150 lines.
Lessons from the Task 4 review (binding for Task 5):
- Apply the D1–D5 wording (see the contract below): every spec states the partial-PUT rule, "ids are never reused, even after a delete", "every new app instance starts with no items", and names each of its 400 rules (required, type, range) explicitly.
- `isinstance(True, int)` is True in Python. Acceptance tests for `expenses`/`inventory` (or any numeric field) must not send a bool and expect 400 unless that app's spec.md says booleans are rejected as numbers.
- Acceptance tests must include the partial-PUT, PUT-validation and id-reuse cases, and must fail on the breakage mutations in `test_bench_reference.py`.
- [ ] The Task 4 gate test must now cover all 10 apps and pass. Commit: "Add the remaining eight benchmark apps with acceptance tests and reference solutions".

### Task 6: Model interface, prompt and retrieval

**Files:** Create `stepbuild/harness/model.py`, `stepbuild/harness/prompt.py`, `stepbuild/harness/retrieve.py`, tests `test_model.py`, `test_prompt.py`, `test_retrieve.py`.

```python
class StepModel(Protocol):
    name: str
    def complete(self, messages: list[dict[str, str]]) -> str: ...

class ScriptedModel:
    """Returns canned replies in order; raises RuntimeError if asked more times than scripted.
    Records every messages list it was given (for assertions)."""
    def __init__(self, replies: Sequence[str], name: str = "scripted") -> None
    calls: list[list[dict[str, str]]]

@dataclass(frozen=True)
class Example:
    step: str       # the STEP instruction
    reply: str      # FILE blocks
class ExampleLibrary:
    def __init__(self, examples: Sequence[Example]) -> None   # builds quipu.memory.bm25 over tiktoken gpt2 ids of step+reply
    @classmethod
    def from_jsonl(cls, paths: Sequence[Path], split: str = "train") -> "ExampleLibrary"
    def top(self, query: str, k: int = 2) -> list[Example]    # empty library → []

SYSTEM_PROMPT: str   # the spec's system message, verbatim
def build_messages(plan: AppPlan, step: Step, files: dict[str, str], examples: Sequence[Example]) -> list[dict[str, str]]
    # user message: APP SPEC, STEP (title), ALLOWED FILES, CURRENT FILES (FILE blocks of the step's allowed files
    # that already exist plus backend/models.py and backend/app.py once written), EXAMPLES (retrieved), PROJECT TREE
def append_feedback(messages, reply: str, failures: Sequence[CheckResult]) -> list[dict[str, str]]
    # adds the assistant reply and a user turn "The checks failed:" + each failed check's name and output
```
- [ ] Tests: ScriptedModel order + over-call error + recorded calls; `top` returns the example sharing a rare identifier first, `[]` for empty library, deterministic ties (earlier example wins); `build_messages` puts the system prompt first, includes allowed files and existing current files, never includes acceptance tests; `append_feedback` includes only failed checks.
- [ ] Commit: "Add the step model interface, prompt builder and BM25 example retrieval".

### Task 7: The runner

**Files:** Create `stepbuild/harness/runner.py`, `tests/stepbuild/test_runner.py`.

```python
@dataclass(frozen=True)
class Attempt:
    reply: str; parse_error: str | None; checks: tuple[CheckResult, ...]
@dataclass(frozen=True)
class StepTrace:
    number: int; key: str; attempts: tuple[Attempt, ...]; passed: bool
@dataclass(frozen=True)
class AppResult:
    app: str; model: str; status: str   # "passed_steps" | f"failed_at_step_{n}"
    steps: tuple[StepTrace, ...]; seconds: float
def run_app(model: StepModel, plan: AppPlan, sandbox: Sandbox, library: ExampleLibrary,
            max_retries: int = 3) -> AppResult
def write_trace(result: AppResult, path: Path) -> None   # atomic JSON
```
Loop per model step: build messages → `model.complete` → `parse_blocks(reply, allowed=step.allowed_files)`; on `BlockError` record the attempt with `parse_error` and feed the error back as a failed pseudo-check `"format"`; else write blocks, run the step's checks; all pass → next step; else feed failures back. Up to `1 + max_retries` attempts. Step 6 runs checks only. Stop at the first step that exhausts its attempts.

- [ ] Tests (`npm`-marked): (1) ScriptedModel replaying `todo` reference → status `passed_steps`, every step one attempt; (2) step 2 reply with a Python syntax error four times → `failed_at_step_2`, 4 attempts, pyflakes output recorded, model called exactly 4 times for that step and never for step 3; (3) bad reply then the reference reply → step passes with 2 attempts and the second call's messages contain the pyflakes error; (4) a reply writing a disallowed file → parse_error recorded and fed back. Fast non-npm variants of (2)–(4) using steps 1–3 only are allowed and preferred for the default suite.
- [ ] Mutation: `max_retries` ignored (single attempt) → test (3) fails. Restore.
- [ ] Commit: "Add the step runner with retries, feedback and per-app traces".

### Task 8: Benchmark runner and scores

**Files:** Create `stepbuild/bench/run.py`, `tests/stepbuild/test_bench_run.py`.

```python
def run_bench(model: StepModel, apps: Sequence[str] | None, out_dir: Path, library: ExampleLibrary) -> list[dict]
    # per app: fresh sandbox → run_app → if passed_steps run_acceptance → write out_dir/<model>/<app>.json
def summarise(records: Sequence[dict]) -> str
    # markdown: apps fully passing acceptance %, first-try step pass %, mean retries per model step,
    # failing-step histogram, wall time per app; one row per app + totals
def main(argv=None) -> int   # CLI: --model scripted-reference (built in: replays reference solutions), --apps, --out
```
A built-in `ReferenceModel` (per app, returns that app's reference replies) makes `python -m stepbuild.bench.run --model reference` produce a 100% table — the end-to-end proof the spec asks for.
- [ ] Tests: summary numbers on hand-made records (including a failed app and retries); CLI with `--model reference --apps todo` (npm-marked) writes JSON + `summary.md` with 100% for that app.
- [ ] Commit: "Add the benchmark runner, scores and a reference model".

### Task 9: Dataset — filters, format, split (pure)

**Files:** Create `stepbuild/dataset/__init__.py`, `filters.py`, `format.py`, `split.py`, tests `test_filters.py`, `test_format.py`, `test_split.py`.

```python
@dataclass(frozen=True)
class FileChange:
    path: str; before: str | None; after: str | None; added: int; removed: int
@dataclass(frozen=True)
class Commit:
    sha: str; message: str; parents: int; changes: tuple[FileChange, ...]
def drop_reason(commit: Commit, max_files=3, max_lines=200, max_file_lines=400) -> str | None
    # None = keep; otherwise a short reason key used in the report:
    # "merge", "root", "too_many_files", "too_many_lines", "file_type", "excluded_path", "too_long_file",
    # "low_info_message", "deleted_file"
def format_example(repo: str, licence: str, tag: str, commit: Commit, context: dict[str, str],
                   tree: Sequence[str], max_user_tokens: int = 6000) -> dict | None   # None if over the cap
def assign_split(repo: str, ratios=(0.90, 0.05, 0.05)) -> str   # stable hash of repo name → train/validation/test
```
All rules exactly as spec §3.1 (extensions, excluded paths, low-information messages list, message ≥ 12 chars after stripping, reverts). A deletion of a whole file drops the commit (`deleted_file`) — the full-file format cannot express it. Token count: `len(text) / 4` estimate (the spec allows it; no Qwen tokenizer dependency now).
- [ ] Tests: one keep and one drop case for **every** reason key; format output has system/user/assistant with FILE blocks, the assistant turn contains post-commit contents of every changed file, over-cap returns None; split is deterministic, roughly matches ratios over 2,000 synthetic repo names (±2%), and a repo always maps to the same split.
- [ ] Mutation: split by commit instead of repo (e.g. hash `repo+sha`) → a test asserting all commits of one repo share a split fails. Restore.
- [ ] Commit: "Add the step dataset filters, chat formatting and repo-level split".

### Task 10: Dataset — discovery, licences, mining, build CLI

**Files:** Create `discover.py`, `licence.py`, `mine.py`, `build.py` and tests `test_discover.py`, `test_licence.py`, `test_mine.py`, `test_build.py`, fixtures under `tests/stepbuild/fixtures/`.

- `discover.py`: uses `gh api` via `subprocess` (no token handling in our code) against GitHub code search to find repos whose `requirements.txt`/`pyproject.toml` contain `flask` and repos whose `package.json` contains `"react"`; intersect for `fullstack`, remainder tagged `flask`/`react`. Responses cached as JSON under `data/stepbuild/cache/` keyed by query+page; on HTTP 403/429 sleep with exponential backoff (max 5 tries, max 60 s). `def discover(limit: int, cache_dir: Path, runner=subprocess.run) -> list[Candidate]` with `Candidate(repo, tag, stars)`; injectable `runner` for tests.
- `licence.py`: `ALLOWED = {"MIT","Apache-2.0","BSD-2-Clause","BSD-3-Clause","ISC","0BSD","Unlicense"}`; `licence_of(repo, runner) -> str | None` via `gh api repos/{repo}/license` (`.license.spdx_id`); `append_source(path, repo, licence, first_sha, last_sha)`.
- `mine.py`: `mine_repo(repo_dir: Path) -> Iterator[Commit]` using `git log --first-parent --reverse` and `git show --numstat` / `git show <sha>:<path>`; context = pre-commit versions of changed files + top-2 BM25 files from `git ls-tree` at the parent (text files under 400 lines only); tree = depth-2 paths.
- `build.py`: CLI `python -m stepbuild.dataset.build --limit N --out data/stepbuild` → discover → licence filter → clone (full, into `data/stepbuild/repos/`, skip if present) → mine → filter → format → split → JSONL shards `train/validation/test-*.jsonl` (atomic) + `SOURCES.jsonl` + `report.md` (repos found/licensed/mined per tag, examples kept, drops per reason key). Rerun skips repos listed in `manifest.json`.
- [ ] Tests (no network): discover parses recorded search JSON fixtures, tags correctly, retries on a scripted 429 then succeeds, uses the cache on a second call; licence keeps MIT, drops `NOASSERTION`/None/GPL; mine over a tiny repo built in `tmp_path` with `git init` (git-marked) yields the expected commits and skips the root commit; build end-to-end with injected runner + tiny repo writes shards, SOURCES, report with correct counts.
- [ ] Mutation: licence check bypassed → a GPL fixture repo appears in SOURCES and the licence test fails. Restore.
- [ ] Commit: "Add repo discovery, licence filtering, git mining and the dataset build CLI".
- [ ] **Real run (network, owner's gh account):** `python -m uv run python -m stepbuild.dataset.build --limit 30 --out data/stepbuild` as a smoke test; report `report.md` verbatim. Then, if the smoke run is clean, `--limit 300`. `data/` stays uncommitted.

### Task 11: Final review

- [ ] Whole-feature review against the spec (dispatch a reviewer): every §3 unit exists; §5 tests and mutations present; `python -m stepbuild.bench.run --model reference` prints 100% for all 10 apps; dataset report produced.
- [ ] Record results and any follow-ups in `docs/superpowers/plans/2026-09-28-stepbuild-core.md` under a "Results" heading. Commit.

## Contract for Tasks 3–5

Fixed in the Task 2 review. The step titles in `stepbuild/harness/plan.py` tell the model this contract, and `tests/stepbuild/test_plan.py` pins the wording. The template (Task 3), the benchmark specs, acceptance tests and reference solutions (Tasks 4–5) must all agree with it.

**`backend/models.py` (step 1).**
- A `Store` class keeps items in a dict and gives each new item an integer `id`.
- Its methods are `create(data)`, `list_items()`, `get(id)`, `update(id, data)` and `delete(id)`. Items are plain dicts that include their `id`.
- `get` and `update` return None and `delete` returns False when the id does not exist; `delete` returns True when it removed the item. (D2, Task 4 review.)
- There is a `validate_<name>(data)` function for each JSON body the API accepts: one per entity created through the API, plus one for the body of each extra action endpoint the spec names (such as a check-in or a decrement). Each returns a list of error strings, one for each rule the spec states: a missing or empty required field, a field of the wrong type, and any range rule (for example a negative amount). The list is empty when the data is valid. (D5.)
- Standard library only, no Flask imports.

**`backend/app.py` (step 2).**
- `create_app()` creates a new `Store()` on every call and uses it in the routes, so each app and each test starts empty.
- It imports with `from models import Store, ...`.
- Every endpoint is under `/api/` and returns JSON. The list endpoint returns a JSON array of items.
- When creating, it fills in the spec's default values for optional fields that were left out. (D4.)
- For PUT, it merges the sent fields into the existing item and validates the merged item with the same validate function, so PUT is a partial update. (D1.)
- Status codes: 201 create, 200 read/update, 200 or 204 delete, 400 with an `{"error": ...}` body on validation errors, 404 missing, plus any code the spec names (401, 409).
- Apps with login set `app.secret_key` inside `create_app()`.
- Step 2 runs the `pyflakes` and `pytest` checks.

**Template (Task 3).**
- `backend/tests/conftest.py` provides the `client` fixture, built from `create_app()` with `TESTING = True`. Step 3 uses this fixture and never redefines it.
- `backend/tests/test_smoke.py` calls `create_app()` and asserts the app object exists, so a backend bug shows up in step 2, while the model can still fix it.
- `frontend/src/main.jsx` must eagerly `import.meta.glob('./components/*.jsx', { eager: true })`. Otherwise the step-4 build would not compile the components, because nothing imports them until step 5.

**Sandbox lifecycle (Tasks 4, 7, 8; fixed in the Task 3 review).**
- Callers delete sandboxes only with `stepbuild.harness.sandbox.remove_sandbox()`. `frontend/node_modules` is a junction (symlink off Windows) into the shared npm cache, and a hand-written recursive delete that follows it (anything trusting `Path.is_dir()`) empties the cache for every sandbox.
- `run_acceptance` (Task 4) must write its own `pytest.ini` (or pass `-c`) in its temp dir, so a stray config in a parent directory (there is one in `%TEMP%` on the dev machine) cannot change the run. It must also set `PYTHONPATH` to the sandbox's `backend/` itself, because the checks' environment (`checks._check_env`) strips `PYTHONPATH` and the acceptance tests do not use the template's conftest.

**`backend/tests/test_api.py` (step 3).**
- Tests use the `client` fixture and import with `from app import create_app` only if they need it.
- If the spec includes login, register and log in with `client` first; the fixture starts logged out.

**Frontend (steps 4–5).**
- `List({items, onDelete})` renders the items, calls `onDelete(item.id)` and uses `item.id` as the React key.
- `Form({onSubmit})` has controlled inputs for the fields of the app's main entity and clears itself after submit. Neither component makes network calls.
- `api.js` exports async functions to list, create and delete items (it may export others). They call relative URLs under `/api/`, send JSON and throw on a non-2xx response. (D3.)
- `App.jsx` has a default export, loads the items on mount, and imports `List` and `Form` by explicit paths: `import List from './components/List.jsx'` and `import Form from './components/Form.jsx'`.
- If the spec includes login, api.js also exports register, login and logout, and App.jsx shows a login form with a register button when logged out (a 401 on load means logged out) and a logout button when logged in.

**Benchmark `spec.md` (Tasks 4–5).**
- Each spec names every endpoint (method plus `/api/` path), the required fields, the id field, the shape of the list response and any extra status codes.
- Each spec also states, in these words: "ids are never reused, even after a delete"; "every new app instance starts with no items"; and "PUT accepts any subset of the fields; fields left out keep their values; a field that is sent follows the same rules as on create (so an empty `title` is 400)" (adapt the example field for apps without a `title`); and "a field sent as `null` counts as sent, so `null` for a field is 400" (Task 5 review). `tests/stepbuild/test_bench_reference.py` checks for these phrases.
- Each spec lists every 400 rule explicitly (missing/empty required fields, wrong types, range rules) with its default values, e.g. "a string that is not empty after trimming whitespace".
- App-specific rules:
  - expenses: 400 on a negative amount.
  - inventory: 409 when stock would go below zero.
  - habits: a check-in is idempotent per day.
  - todo_auth: register, login and logout endpoints; 401 when logged out; todos are per user; users are kept in memory.

**Acceptance tests (Tasks 4–5).**
- They define their own client and do not rely on the template's conftest.
- They accept 200 or 204 on delete, and assert only that an `error` key exists (not its text).
- They compare only the fields the spec names (a `view()` helper), because no spec says "no other keys": a response with an extra key such as `created_at` is correct. The gate test `test_acceptance_tolerates_spec_correct_variations` adds such a key to every reference backend and requires acceptance to pass. (Task 5 review.)
- They rely on the integer `id` and on the list endpoint returning a JSON array.
- They cover, besides CRUD: a partial PUT keeps the fields left out; a PUT that breaks a create rule is 400; ids are not reused after a delete; each new app starts empty. The gate test `test_acceptance_catches_plausible_breakages` mutates each reference backend (full-replace PUT, unvalidated PUT, `len + 1` ids) and requires acceptance to fail; its patterns match the todo/notes reference code, so later references should keep the same shapes (`changes = {key: data[key] for key in FIELDS if key in data}`, `errors = validate_<entity>({**item, **changes})`, `item["id"] = self._next_id`) or extend the table.

**Reference solutions.**
- `reference/step_N.txt` writes exactly that step's allowed files: no more, no fewer.
