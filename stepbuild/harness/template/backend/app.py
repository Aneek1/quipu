"""Placeholder Flask app. The model replaces this whole file in step 2 (routes).

It only has to satisfy the template's own checks: pyflakes is clean and
tests/test_smoke.py can build an app from create_app().
"""
from flask import Flask


def create_app():
    app = Flask(__name__)
    return app
