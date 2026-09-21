# SQLAlchemy engine, session factory, and init helpers. WAL mode enabled for SQLite.
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from config import get_settings
from models import Base

logger = logging.getLogger(__name__)

settings = get_settings()

# ─────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────

_engine_kwargs: dict = {
    "echo": settings.flask_env == "development",
}

# SQLite-specific: allow the same connection across threads (Flask dev server)
if "sqlite" in settings.database_url:
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(settings.database_url, **_engine_kwargs)


# ─────────────────────────────────────────────────────────────
# SQLite Pragmas — applied once per physical connection
# ─────────────────────────────────────────────────────────────


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_connection: object, connection_record: object) -> None:
    """Configure SQLite for production-safe behaviour."""
    if "sqlite" not in settings.database_url:
        return
    cursor = dbapi_connection.cursor()  # type: ignore[union-attr]
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA cache_size=-64000")   # 64 MB page cache
    cursor.execute("PRAGMA temp_store=MEMORY")
    cursor.close()
    logger.debug("SQLite pragmas applied.")


# ─────────────────────────────────────────────────────────────
# Session Factory
# ─────────────────────────────────────────────────────────────

SessionLocal: sessionmaker[Session] = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


# ─────────────────────────────────────────────────────────────
# Public Helpers
# ─────────────────────────────────────────────────────────────


def init_db() -> None:
    """
    Create all ORM tables that don't yet exist.

    Safe to call on every startup — uses CREATE TABLE IF NOT EXISTS semantics
    via SQLAlchemy's ``checkfirst=True`` default.
    """
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialised (or already exist).")


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Yield a transactional SQLAlchemy Session.

    Usage::

        with get_db() as db:
            db.add(some_orm_object)
            # commit is automatic on clean exit

    Guarantees:
      - Commits on clean exit
      - Rolls back on any exception
      - Always closes the session
    """
    db: Session = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
