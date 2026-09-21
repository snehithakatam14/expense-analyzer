"""
tests/conftest.py — Shared pytest fixtures for the expense analyzer test suite.

Provides:
  - ``test_settings``  — overrides OPENAI_API_KEY and DATABASE_URL for tests
  - ``app``            — Flask test app with in-memory SQLite
  - ``client``         — Flask test client
  - ``db_session``     — Fresh SQLAlchemy session per test
  - Sample transaction fixtures
"""
from __future__ import annotations

import os
import pytest
from datetime import date, datetime
from decimal import Decimal
from unittest.mock import patch

# ─────────────────────────────────────────────────────────────
# Override environment before any module-level settings are loaded
# ─────────────────────────────────────────────────────────────

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("FLASK_ENV", "testing")

from config import get_settings
from database import SessionLocal, engine, init_db
from models import Base, NormalizedTransaction, TransactionORM, CategoryORM


# ─────────────────────────────────────────────────────────────
# DB Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Create all tables once per test session on the in-memory DB."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def db_session():
    """
    Provide a transactional DB session that rolls back after each test.
    This ensures test isolation — no test can pollute another's state.
    """
    connection = engine.connect()
    transaction = connection.begin()
    session = SessionLocal(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


# ─────────────────────────────────────────────────────────────
# Flask Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def app():
    """Create Flask test app (DB init mocked to avoid file-based SQLite)."""
    with patch("app.init_db"):
        from app import create_app
        application = create_app()
    application.config["TESTING"] = True
    return application


@pytest.fixture()
def client(app):
    """Flask test client scoped per test."""
    return app.test_client()


# ─────────────────────────────────────────────────────────────
# Sample Data Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_normalized_transaction() -> NormalizedTransaction:
    return NormalizedTransaction(
        id="txn-test-001",
        date=date(2024, 1, 15),
        description="Starbucks Coffee",
        amount=Decimal("-5.75"),
        currency="USD",
        merchant_name="Starbucks",
        source_file="test.csv",
        ingested_at=datetime(2024, 1, 15, 12, 0, 0),
        content_hash="a" * 64,
    )


@pytest.fixture()
def sample_csv_bytes() -> bytes:
    return (
        "date,description,amount,currency,merchant_name\n"
        "2024-01-15,Starbucks Coffee,-5.75,USD,Starbucks\n"
        "2024-01-16,Uber Ride,-12.50,USD,Uber\n"
        "2024-01-20,Salary Deposit,3000.00,USD,Employer Inc\n"
    ).encode("utf-8")


@pytest.fixture()
def sample_json_bytes() -> bytes:
    import json
    return json.dumps([
        {
            "date": "2024-01-15",
            "description": "Starbucks Coffee",
            "amount": -5.75,
            "currency": "USD",
            "merchant_name": "Starbucks",
        },
        {
            "date": "2024-01-16",
            "description": "Netflix Subscription",
            "amount": -15.99,
            "currency": "USD",
            "merchant_name": "Netflix",
        },
    ]).encode("utf-8")
