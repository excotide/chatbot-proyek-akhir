"""Koneksi database. Endpoint FastAPI memakai `get_db()` sebagai dependency."""
from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings

if not settings.database_url:
    raise RuntimeError("DATABASE_URL belum diisi di backend/.env (lihat .env.example)")

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
