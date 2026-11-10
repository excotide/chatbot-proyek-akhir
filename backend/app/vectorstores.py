"""Tiga database vektor dengan antarmuka yang sama, untuk komparasi dan untuk chatbot.

Koleksi (collection) = satu kumpulan vektor berdimensi sama, mis. 'kb_multilingual_e5_small' (KB) atau
'bench_10000' (data sintetis benchmark). Id vektor = chunks.id untuk KB, sehingga ketiga database memakai
chunk dan id yang sama (Desain-Database.md bagian 1).

- pgvector  : KB dibaca dari tabel chunk_embeddings (sumber utama); koleksi lain disimpan di tabel vs_<nama>.
- chroma    : server ChromaDB (HTTP), satu collection per koleksi, jarak cosine.
- surrealdb : server SurrealDB (HTTP /sql), satu tabel per koleksi.

search(..., exact=True) = pencarian brute-force (tanpa indeks); exact=False = memakai indeks HNSW.
ChromaDB selalu memakai HNSW (tidak punya mode exact), jadi exact diabaikan di sana.
Semua skor yang dikembalikan = cosine similarity (1 - jarak cosine).
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

import httpx
import numpy as np
from sqlalchemy import text

from app.config import settings

KB_PREFIX = "kb_"


def kb_collection(model_name: str) -> str:
    """'intfloat/multilingual-e5-small' -> 'kb_multilingual_e5_small'."""
    return KB_PREFIX + re.sub(r"[^a-z0-9]+", "_", model_name.split("/")[-1].lower()).strip("_")


def _safe(name: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,60}", name):
        raise ValueError(f"Nama koleksi tidak valid: {name!r}")
    return name


class VectorStore(ABC):
    name: str

    @abstractmethod
    def upsert(self, collection: str, ids: list[int], vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, collection: str, query: np.ndarray, k: int, exact: bool = False) -> list[tuple[int, float]]: ...

    @abstractmethod
    def build_index(self, collection: str, dim: int) -> None:
        """Bangun indeks HNSW (cosine)."""

    @abstractmethod
    def count(self, collection: str) -> int: ...

    @abstractmethod
    def drop(self, collection: str) -> None: ...

    def ping(self) -> bool:
        try:
            return self.count("__ping__") >= 0
        except Exception:  # noqa: BLE001
            return False


# ---------------------------------------------------------------- pgvector
class PgVectorStore(VectorStore):
    name = "pgvector"

    def __init__(self):
        from app.db.session import engine
        self.engine = engine

    @staticmethod
    def _vec(v: np.ndarray) -> str:
        return "[" + ",".join(f"{x:.7g}" for x in v) + "]"

    def _is_kb(self, collection: str) -> bool:
        return collection.startswith(KB_PREFIX)

    def _kb_model(self, collection: str) -> str:
        from app.rag.embedding import MODELS
        for m in MODELS:
            if kb_collection(m) == collection:
                return m
        raise ValueError(f"Koleksi KB tidak dikenal: {collection}")

    def upsert(self, collection, ids, vectors):
        if self._is_kb(collection):
            raise ValueError("KB pgvector diisi oleh scripts/embed_kb.py (tabel chunk_embeddings)")
        t = "vs_" + _safe(collection)
        dim = vectors.shape[1]
        with self.engine.begin() as c:
            c.execute(text(f"CREATE TABLE IF NOT EXISTS {t} (id bigint PRIMARY KEY, embedding vector({dim}) NOT NULL)"))
            for i in range(0, len(ids), 1000):
                rows = [{"id": int(a), "e": self._vec(v)} for a, v in zip(ids[i:i + 1000], vectors[i:i + 1000])]
                c.execute(text(f"INSERT INTO {t} (id, embedding) VALUES (:id, CAST(:e AS vector)) "
                               f"ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding"), rows)

    def search(self, collection, query, k, exact=False):
        q = self._vec(query)
        with self.engine.begin() as c:
            if exact:  # paksa scan penuh walaupun ada indeks
                c.execute(text("SET LOCAL enable_indexscan = off"))
            else:
                c.execute(text("SET LOCAL hnsw.ef_search = 40"))
            if self._is_kb(collection):
                rows = c.execute(text(
                    "SELECT chunk_id, embedding <=> CAST(:q AS vector) AS d FROM chunk_embeddings "
                    "WHERE model_name = :m ORDER BY d LIMIT :k"), {"q": q, "m": self._kb_model(collection), "k": k}).all()
            else:
                rows = c.execute(text(f"SELECT id, embedding <=> CAST(:q AS vector) AS d FROM vs_{_safe(collection)} "
                                      f"ORDER BY d LIMIT :k"), {"q": q, "k": k}).all()
        return [(int(i), 1.0 - float(d)) for i, d in rows]

    def build_index(self, collection, dim):
        t = "vs_" + _safe(collection)
        with self.engine.begin() as c:
            c.execute(text(f"CREATE INDEX IF NOT EXISTS {t}_hnsw ON {t} USING hnsw (embedding vector_cosine_ops)"))

    def count(self, collection):
        with self.engine.connect() as c:
            if collection == "__ping__":
                return c.execute(text("SELECT 0")).scalar()
            if self._is_kb(collection):
                return c.execute(text("SELECT count(*) FROM chunk_embeddings WHERE model_name = :m"),
                                 {"m": self._kb_model(collection)}).scalar()
            exists = c.execute(text("SELECT to_regclass(:t)"), {"t": "vs_" + _safe(collection)}).scalar()
            return c.execute(text(f"SELECT count(*) FROM vs_{collection}")).scalar() if exists else 0

    def drop(self, collection):
        if self._is_kb(collection):
            raise ValueError("KB pgvector tidak dihapus dari sini")
        with self.engine.begin() as c:
            c.execute(text(f"DROP TABLE IF EXISTS vs_{_safe(collection)}"))


# ---------------------------------------------------------------- ChromaDB
class ChromaStore(VectorStore):
    name = "chroma"

    def __init__(self):
        import chromadb
        from urllib.parse import urlparse
        u = urlparse(settings.chroma_url)
        self.client = chromadb.HttpClient(host=u.hostname, port=u.port or 8000)

    def _col(self, collection, create=True):
        name = _safe(collection)
        if create:
            return self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        return self.client.get_collection(name)

    def upsert(self, collection, ids, vectors):
        col = self._col(collection)
        for i in range(0, len(ids), 1000):
            col.upsert(ids=[str(a) for a in ids[i:i + 1000]], embeddings=vectors[i:i + 1000].tolist())

    def search(self, collection, query, k, exact=False):
        r = self._col(collection, create=False).query(query_embeddings=[query.tolist()], n_results=k, include=["distances"])
        return [(int(i), 1.0 - float(d)) for i, d in zip(r["ids"][0], r["distances"][0])]

    def build_index(self, collection, dim):
        pass  # ChromaDB membangun indeks HNSW otomatis saat data dimasukkan

    def count(self, collection):
        if collection == "__ping__":
            self.client.heartbeat()
            return 0
        try:
            return self._col(collection, create=False).count()
        except Exception:  # noqa: BLE001  (koleksi belum ada)
            return 0

    def drop(self, collection):
        try:
            self.client.delete_collection(_safe(collection))
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------- SurrealDB
class SurrealStore(VectorStore):
    name = "surrealdb"

    def __init__(self):
        self.http = httpx.Client(base_url=settings.surreal_url, timeout=600,
                                 auth=(settings.surreal_user, settings.surreal_pass),
                                 headers={"Accept": "application/json", "surreal-ns": settings.surreal_ns,
                                          "surreal-db": settings.surreal_db})
        self._ready = False

    def sql(self, query: str) -> list:
        if not self._ready:
            self._ready = True
            self.sql(f"DEFINE NAMESPACE IF NOT EXISTS {settings.surreal_ns}; USE NS {settings.surreal_ns}; "
                     f"DEFINE DATABASE IF NOT EXISTS {settings.surreal_db};")
        r = self.http.post("/sql", content=query.encode())
        if r.status_code != 200:
            raise RuntimeError(f"SurrealDB HTTP {r.status_code}: {r.text[:300]}")
        out = r.json()
        for stmt in out:
            if stmt.get("status") != "OK":
                raise RuntimeError(f"SurrealDB: {str(stmt.get('result'))[:300]}")
        return [stmt.get("result") for stmt in out]

    @staticmethod
    def _vec(v: np.ndarray) -> str:
        return "[" + ",".join(f"{x:.7g}" for x in v) + "]"

    # HTTP /sql SurrealDB membatasi ukuran badan permintaan (±1 MB, ditolak dengan HTTP 413),
    # jadi baris dikirim per batch yang ukurannya di bawah batas ini.
    MAX_BODY = 900_000

    def upsert(self, collection, ids, vectors):
        t = _safe(collection)
        batch, size = [], 0

        def flush():
            if batch:
                self.sql(f"INSERT INTO {t} [{','.join(batch)}] ON DUPLICATE KEY UPDATE embedding = $input.embedding;")

        for a, v in zip(ids, vectors):
            row = f"{{id: {t}:{int(a)}, embedding: {self._vec(v)}}}"
            if batch and size + len(row) > self.MAX_BODY:
                flush()
                batch, size = [], 0
            batch.append(row)
            size += len(row) + 1
        flush()

    def search(self, collection, query, k, exact=False):
        t = _safe(collection)
        op = f"<|{k},COSINE|>" if exact else f"<|{k},40|>"  # brute force vs indeks HNSW (ef=40)
        res = self.sql(f"SELECT record::id(id) AS cid, vector::distance::knn() AS d FROM {t} "
                       f"WHERE embedding {op} {self._vec(query)} ORDER BY d;")[0]
        return [(int(r["cid"]), 1.0 - float(r["d"])) for r in res]

    def build_index(self, collection, dim):
        t = _safe(collection)
        self.sql(f"DEFINE INDEX IF NOT EXISTS {t}_hnsw ON {t} FIELDS embedding HNSW DIMENSION {dim} DIST COSINE;")

    def count(self, collection):
        if collection == "__ping__":
            self.sql("RETURN 1;")
            return 0
        res = self.sql(f"SELECT count() AS n FROM {_safe(collection)} GROUP ALL;")[0]
        return int(res[0]["n"]) if res else 0

    def drop(self, collection):
        self.sql(f"REMOVE TABLE IF EXISTS {_safe(collection)};")


STORES = {"pgvector": PgVectorStore, "chroma": ChromaStore, "surrealdb": SurrealStore}
_cache: dict[str, VectorStore] = {}


def get_store(name: str | None = None) -> VectorStore:
    name = name or settings.vector_db
    if name not in STORES:
        raise ValueError(f"Database vektor tidak dikenal: {name} (pilihan: {', '.join(STORES)})")
    if name not in _cache:
        _cache[name] = STORES[name]()
    return _cache[name]
