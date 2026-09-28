"""Shared fixtures for the backend tests.

The backend directory is put on sys.path here, so `from app import create_app`
works however pytest is started (from backend/ or from the project root) and
without an __init__.py or a pytest config file in the project.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app  # noqa: E402


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client
