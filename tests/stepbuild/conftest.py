"""Fixtures shared by the stepbuild tests.

`npm_cache` is session-scoped and uses the real default cache (or STEPBUILD_CACHE),
so the one slow `npm install` happens at most once per machine per template
version, not once per test run.
"""
import pytest


@pytest.fixture(scope="session")
def npm_cache():
    from stepbuild.harness.sandbox import default_cache_dir

    return default_cache_dir()
