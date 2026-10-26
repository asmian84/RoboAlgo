"""The Postgres driver must be named explicitly, not left to SQLAlchemy's default.

Regression for a failure that was invisible locally: SQLAlchemy 2.0.x resolves a
bare ``postgresql://`` URL to psycopg2, SQLAlchemy 2.1.x to psycopg (v3).
requirements.txt ships only psycopg2-binary and is unpinned, so a fresh install
(which is what Render builds) got 2.1 and every connection died with
``ModuleNotFoundError: No module named 'psycopg'``. Nothing caught it because the
suite never opens a connection and callers log the failure as a warning.

These tests assert the contract (the URL names the driver we ship), not the
installed SQLAlchemy's default, so they fail on any version that would break.
"""
import importlib.util
import pathlib

import pytest
from sqlalchemy.engine import make_url

from config.settings import _pin_postgres_driver

REPO = pathlib.Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("bare", [
    "postgresql://user:pw@host:5432/db",
    "postgres://user:pw@host/db",          # Render and Heroku legacy scheme
])
def test_bare_postgres_urls_pin_psycopg2(bare):
    url = make_url(_pin_postgres_driver(bare))
    assert url.get_backend_name() == "postgresql"
    assert url.get_driver_name() == "psycopg2"


@pytest.mark.parametrize("explicit", [
    "postgresql+psycopg://u@h/d",
    "postgresql+psycopg2://u@h/d",
    "sqlite:///x.db",
])
def test_urls_that_already_choose_a_driver_are_untouched(explicit):
    assert _pin_postgres_driver(explicit) == explicit


def test_credentials_host_and_database_survive_the_rewrite():
    out = make_url(_pin_postgres_driver("postgres://alice:s3cret@db.internal:6543/trading"))
    assert (out.username, out.password, out.host, out.port, out.database) == (
        "alice", "s3cret", "db.internal", 6543, "trading")


def test_the_pinned_driver_is_installed_and_declared():
    # The original bug was a mismatch between the driver SQLAlchemy chose and the
    # driver requirements.txt ships. Check both ends of that.
    assert importlib.util.find_spec("psycopg2") is not None, "psycopg2 not installed"
    declared = (REPO / "requirements.txt").read_text().lower()
    assert "psycopg2" in declared, "psycopg2 is pinned in the URL but not declared in requirements.txt"
