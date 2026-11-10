"""Tiga metode pencarian dengan antarmuka yang sama: `search(query, k) -> list[Hit]`.

- bm25     : pencocokan kata (bm25.py)
- semantic : kemiripan makna, cosine similarity antara embedding pertanyaan dan chunk.
             Vektor chunk disimpan di PostgreSQL + pgvector (tabel chunk_embeddings).
- hybrid   : gabungan peringkat BM25 dan semantic dengan Reciprocal Rank Fusion (RRF):
             skor = 1/(k + peringkat_bm25) + 1/(k + peringkat_semantic), k = RRF_K (default 60).
"""
import logging

from sqlalchemy import select

from app.config import settings
from app.db.models import ChunkEmbedding
from app.db.session import SessionLocal
from app.rag.bm25 import BM25Retriever
from app.rag.chunking import Chunk, Hit
from app.rag.embedding import embed_query
from app.vectorstores import get_store, kb_collection

log = logging.getLogger("retrievers")


class SemanticRetriever:
    """Cari berdasarkan kemiripan embedding. vector_db: pgvector (tabel chunk_embeddings, default),
    chroma, atau surrealdb (koleksi hasil scripts/sync_vectors.py)."""
    name = "semantic"

    def __init__(self, chunks: list[Chunk], model_name: str, vector_db: str = "pgvector"):
        self.chunks = chunks
        self.model_name = model_name
        self.vector_db = vector_db
        self._by_id = {c.db_id: c for c in chunks}
        if vector_db != "pgvector":
            store = get_store(vector_db)
            n = store.count(kb_collection(model_name))
            if n < len(self._by_id):
                log.warning("%s hanya punya %d dari %d vektor KB. Jalankan scripts/sync_vectors.py.",
                            vector_db, n, len(self._by_id))
            return
        with SessionLocal() as db:
            have = set(db.scalars(select(ChunkEmbedding.chunk_id).where(
                ChunkEmbedding.model_name == model_name, ChunkEmbedding.chunk_id.in_(self._by_id))))
        missing = len(self._by_id) - len(have)
        if missing:
            log.warning("%d chunk belum punya embedding model %s. Jalankan scripts/embed_kb.py.", missing, model_name)

    def search(self, query: str, k: int = 3) -> list[Hit]:
        q = embed_query(query, self.model_name)
        if self.vector_db != "pgvector":
            rows = get_store(self.vector_db).search(kb_collection(self.model_name), q, k)
            return [Hit(self._by_id[cid], s, similarity=s) for cid, s in rows if cid in self._by_id]
        dist = ChunkEmbedding.embedding.cosine_distance(q)  # operator <=> pgvector
        stmt = (select(ChunkEmbedding.chunk_id, dist.label("d"))
                .where(ChunkEmbedding.model_name == self.model_name, ChunkEmbedding.chunk_id.in_(self._by_id))
                .order_by(dist).limit(k))
        with SessionLocal() as db:
            rows = db.execute(stmt).all()
        return [Hit(self._by_id[cid], 1 - d, similarity=1 - d) for cid, d in rows]


class HybridRetriever:
    name = "hybrid"

    def __init__(self, chunks: list[Chunk], model_name: str, rrf_k: int = 60, vector_db: str = "pgvector"):
        self.chunks = chunks
        self.vector_db = vector_db
        self.bm25 = BM25Retriever(chunks)
        self.semantic = SemanticRetriever(chunks, model_name, vector_db)
        self.rrf_k = rrf_k

    def search(self, query: str, k: int = 3) -> list[Hit]:
        n = len(self.chunks)  # peringkat penuh dari kedua metode (KB kecil, murah)
        lexical = self.bm25.search(query, n)
        semantic = self.semantic.search(query, n)
        sim = {h.chunk.db_id: h.similarity for h in semantic}
        fused: dict[int, float] = {}
        for ranking in (lexical, semantic):
            for rank, h in enumerate(ranking, 1):
                fused[h.chunk.db_id] = fused.get(h.chunk.db_id, 0.0) + 1 / (self.rrf_k + rank)
        by_id = {c.db_id: c for c in self.chunks}
        best = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
        return [Hit(by_id[cid], score, similarity=sim.get(cid)) for cid, score in best]


def build_retriever(method: str, chunks: list[Chunk], model_name: str | None = None, vector_db: str | None = None):
    model_name = model_name or settings.embedding_model
    vector_db = vector_db or settings.vector_db
    if method == "bm25":
        return BM25Retriever(chunks)
    if method == "semantic":
        return SemanticRetriever(chunks, model_name, vector_db)
    if method == "hybrid":
        return HybridRetriever(chunks, model_name, settings.rrf_k, vector_db)
    raise ValueError(f"RETRIEVER tidak dikenal: {method} (pilihan: bm25, semantic, hybrid)")


def has_context(method: str, hits: list[Hit]) -> bool:
    """Ambang 'tidak ada konteks'. BM25 memakai skor BM25; semantic/hybrid memakai cosine tertinggi."""
    if not hits:
        return False
    if method == "bm25":
        return hits[0].score >= settings.min_score
    return max(h.similarity or 0.0 for h in hits) >= settings.min_similarity
