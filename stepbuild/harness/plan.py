"""The fixed six-step build plan every benchmark app is built against.

Every app is a Flask API plus a Vite React frontend, built in the same order:

    1 model       backend/models.py                          pyflakes
    2 routes      backend/app.py                             pyflakes, pytest
    3 api_tests   backend/tests/test_api.py                  pyflakes, pytest
    4 components  frontend/src/components/{List,Form}.jsx    npm_build
    5 wiring      frontend/src/{api.js,App.jsx}              npm_build
    6 run         (no files, no model call)                  pyflakes, pytest, npm_build

The order is fixed rather than planned by the model because the thing being
measured is whether a small model can write each file well, not whether it can plan;
a fixed order also makes step N comparable across apps and models. The files are
exactly the ones the sandbox template leaves for the model, and the titles state the
contract the template depends on (a `create_app()` factory in backend/app.py, a
`client` fixture in backend/tests/conftest.py, a smoke test that makes step 2's
pytest check meaningful before step 3 writes any tests), so a model that follows its
instructions produces a project the template's fixtures and the hidden acceptance
tests can import.

Titles are the instruction the model sees for the step, so each one names the app,
every file to write, and what "done" means in concrete terms: small models follow
explicit instructions far better than implied ones.

Validation is strict, as in blocks.py: allowed files are validated by building a
FileBlock (one definition of a safe path), checks must come from CHECKS, and
AppPlan asserts the steps are numbered 1..6 in order so a later edit to the table
cannot produce a plan the runner would misread.
"""
from __future__ import annotations

import dataclasses
import re

from stepbuild.harness.blocks import FileBlock

CHECKS = ("pyflakes", "pytest", "npm_build")
N_STEPS = 6

# Letters, digits, '_' and '-', starting with a letter: the name ends up in
# directory names and results paths, so it must be safe and unambiguous there.
_APP_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")


@dataclasses.dataclass(frozen=True)
class Step:
    number: int                     # 1..6
    key: str                        # "model" | "routes" | "api_tests" | "components" | "wiring" | "run"
    title: str                      # instruction shown to the model, includes the app name
    allowed_files: tuple[str, ...]  # files the model may write in this step
    checks: tuple[str, ...]         # subset of CHECKS, in the order they run
    model_step: bool                # False for "run" (checks only, no model call)

    def __post_init__(self) -> None:
        if not isinstance(self.allowed_files, tuple) or not isinstance(self.checks, tuple):
            # A list would make the frozen dataclass mutable through the back door.
            raise TypeError("allowed_files and checks must be tuples")
        if not self.key or not self.title.strip():
            raise ValueError(f"step {self.number}: key and title must be non-empty")
        for path in self.allowed_files:
            FileBlock(path, "")  # raises BlockError (a ValueError) for any unsafe path
        folded = [p.casefold() for p in self.allowed_files]
        if len(set(folded)) != len(folded):
            raise ValueError(f"step {self.key!r}: duplicate allowed file (paths ignore case)")
        unknown = [c for c in self.checks if c not in CHECKS]
        if unknown:
            raise ValueError(f"step {self.key!r}: unknown checks {unknown}; known: {CHECKS}")
        if len(set(self.checks)) != len(self.checks):
            raise ValueError(f"step {self.key!r}: duplicate check")
        if self.model_step and not self.allowed_files:
            raise ValueError(f"step {self.key!r}: a model step must allow at least one file")


@dataclasses.dataclass(frozen=True)
class AppPlan:
    app: str
    spec: str
    steps: tuple[Step, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.steps, tuple):
            raise TypeError("steps must be a tuple")
        numbers = [s.number for s in self.steps]
        if numbers != list(range(1, N_STEPS + 1)):
            raise ValueError(f"steps must be numbered 1..{N_STEPS} in order, got {numbers}")


# (key, allowed files, checks, title template). The order of this table is the
# build order; step numbers are assigned from it. The literal text `{app}` is
# replaced with the app name by str.replace (not str.format), so braces in a title,
# e.g. a JSX or dict example added later, can never be misread as fields.
_STEPS: tuple[tuple[str, tuple[str, ...], tuple[str, ...], str], ...] = (
    (
        "model",
        ("backend/models.py",),
        ("pyflakes",),
        "Write backend/models.py for the {app} app: a `Store` class that keeps items "
        "in a dict and assigns each new item an integer `id`, with methods "
        "create(data), list_items(), get(id), update(id, data) and delete(id); items "
        "are plain dicts that include their `id`. `get` and `update` return None and "
        "`delete` returns False when the id does not exist; `delete` returns True when "
        "it removed the item. Also write a validate_<entity>(data) function for each "
        "entity created through the API: validate_<entity>(data) returns a list of "
        "error strings, one for each rule the spec states: a missing or empty required "
        "field, a field of the wrong type, and any range rule (for example a negative "
        "amount); the list is empty when the data is valid. "
        "Standard library only; no Flask imports in this file.",
    ),
    (
        "routes",
        ("backend/app.py",),
        ("pyflakes", "pytest"),
        "Write backend/app.py for the {app} app: a create_app() Flask factory that "
        "creates a new Store() for each call and uses it in the routes; import from "
        "models with `from models import Store, ...`. Expose the REST endpoints described "
        "in the spec, all under /api/. Every item has the integer `id` assigned by the "
        "store, and the list endpoint returns a JSON array of items. When creating, "
        "fill in the spec's default values for optional fields that were left out. "
        "For PUT, merge the sent fields into the existing item and validate the merged "
        "item with the same validate function. Return JSON and "
        "proper status codes: 201 create, 200 read/update, 200 or 204 delete, 400 with "
        "an {\"error\": ...} body on validation errors, 404 missing, plus any other "
        "status codes the spec names (e.g. 401, 409). If the spec includes login, set "
        "app.secret_key inside create_app().",
    ),
    (
        "api_tests",
        ("backend/tests/test_api.py",),
        ("pyflakes", "pytest"),
        "Write backend/tests/test_api.py for the {app} app: pytest tests that use the "
        "`client` fixture from backend/tests/conftest.py (do not redefine it); import "
        "with `from app import create_app` only if needed. Check every /api/ endpoint in "
        "the spec: create returns 201 and the new item with its `id`, list returns a "
        "JSON array, get one, update, delete and then 404, and 400 when a required "
        "field is missing. The tests must pass against backend/app.py as written.",
    ),
    (
        "components",
        ("frontend/src/components/List.jsx", "frontend/src/components/Form.jsx"),
        ("npm_build",),
        "Write the React components for the {app} app: "
        "frontend/src/components/List.jsx (default export List, props `items` and "
        "`onDelete`; renders each item's fields with a delete button that calls "
        "onDelete(item.id), and uses item.id as the React key) and "
        "frontend/src/components/Form.jsx (default export Form, prop `onSubmit`; "
        "controlled inputs for the fields of the app's main entity, calls onSubmit with "
        "the values and clears itself). Function components with hooks, no extra "
        "packages, no network calls in these files.",
    ),
    (
        "wiring",
        ("frontend/src/api.js", "frontend/src/App.jsx"),
        ("npm_build",),
        "Wire up the {app} app frontend: frontend/src/api.js exports async functions "
        "to list, create and delete items (it may export others) that call the "
        "backend endpoints from the spec with fetch (api.js calls "
        "relative URLs under /api/, sends JSON bodies, throws on a non-2xx response), "
        "and frontend/src/App.jsx (default export App) does "
        "import List from './components/List.jsx' and "
        "import Form from './components/Form.jsx', loads the items on mount, and "
        "creates and deletes items through api.js, updating state.",
    ),
    (
        "run",
        (),
        ("pyflakes", "pytest", "npm_build"),
        "Final check of the {app} app: lint the backend, run the API tests and build "
        "the frontend. No files are written in this step.",
    ),
)


def make_plan(app: str, spec: str) -> AppPlan:
    """The six concrete steps for building `app` from `spec`.

    Raises ValueError for an empty spec or an app name that is not a simple slug
    (letters, digits, '_' and '-', starting with a letter).
    """
    if not isinstance(app, str) or not app.strip():
        raise ValueError("app name must be a non-empty string")
    if not _APP_NAME.fullmatch(app):
        raise ValueError(
            f"app name {app!r} must start with a letter and use only letters, digits, '_' and '-'"
        )
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError(f"spec for app {app!r} must be a non-empty string")
    steps = tuple(
        Step(
            number=i,
            key=key,
            title=title.replace("{app}", app),
            allowed_files=files,
            checks=checks,
            model_step=bool(files),
        )
        for i, (key, files, checks, title) in enumerate(_STEPS, start=1)
    )
    return AppPlan(app=app, spec=spec, steps=steps)
