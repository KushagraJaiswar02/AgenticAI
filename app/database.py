"""Minimal SQLAlchemy/SQLite foundation for V0.1."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.errors import DatabaseError


class Database:
    """Own the SQLAlchemy engine and session factory."""

    def __init__(self, database_url: str = "sqlite:///jarvis.db") -> None:
        try:
            self.engine = create_engine(database_url, future=True)
            self.session_factory = sessionmaker(
                bind=self.engine,
                autoflush=False,
                autocommit=False,
                expire_on_commit=False,
            )
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except SQLAlchemyError as exc:
            raise DatabaseError("Database initialization failed") from exc

    def session(self) -> Session:
        """Create a SQLAlchemy session for the caller."""
        try:
            return self.session_factory()
        except SQLAlchemyError as exc:
            raise DatabaseError("Database session creation failed") from exc

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Yield a session and close it after use."""
        session = self.session()
        try:
            yield session
        except SQLAlchemyError as exc:
            session.rollback()
            raise DatabaseError("Database operation failed") from exc
        finally:
            session.close()
