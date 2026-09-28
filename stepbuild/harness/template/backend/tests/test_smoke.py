"""create_app() must build a Flask app. This keeps pytest meaningful in step 2,
before the model has written any API tests of its own."""
from flask import Flask

from app import create_app


def test_create_app_returns_a_flask_app():
    assert isinstance(create_app(), Flask)
