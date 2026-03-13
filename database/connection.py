"""
Cycle Trading Intelligence System - Database Connection
Manages SQLAlchemy engine, sessions, and database initialization.
"""

import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from config.settings import DATABASE_URL
from database.models import Base

logger = logging.getLogger(__name__)

_engine = None
_SessionFactory = None


def get_engine():
    """Get or create the SQLAlchemy engine (singleton)."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False,
        )
    return _engine


def get_session() -> Session:
    """Create a new database session."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine())
    return _SessionFactory()


def init_db():
    """Create all tables defined in models."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Database tables created successfully.")


def reset_db():
    """Drop and recreate all tables. USE WITH CAUTION."""
    engine = get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    logger.info("Database reset: all tables dropped and recreated.")


def check_connection() -> bool:
    """Verify database connectivity."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False
