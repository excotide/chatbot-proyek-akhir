"""Buat embedding untuk chunk aktif dan simpan ke PostgreSQL + pgvector (tabel chunk_embeddings).

Jalankan dari folder backend (setelah scripts/ingest_kb.py):
    .venv\\Scripts\\python scripts\\embed_kb.py                 # model utama (EMBEDDING_MODEL di .env)
    .venv\\Scripts\\python scripts\\embed_kb.py --all           # semua model pembanding
    .venv\\Scripts\\python scripts\\embed_kb.py --model intfloat/multilingual-e5-small

Chunk yang sudah punya embedding untuk model tersebut dilewati.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.db.models import Chunk, ChunkEmbedding, Document  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.rag.chunking import Chunk as RChunk  # noqa: E402
from app.rag.embedding import MODELS, embed_passages, load_model, model_available  # noqa: E402


def embed_model(model_name: str) -> None:
    if not model_available(model_name):
        print(f"{model_name}: DILEWATI, model belum terunduh (jalankan scripts/download_models.py)")
        return
    with SessionLocal() as db, db.begin():
        done = select(ChunkEmbedding.chunk_id).where(ChunkEmbedding.model_name == model_name)
        rows = db.execute(
            select(Chunk, Document.title).join(Document, Chunk.document_id == Document.id)
            .where(Document.is_active, Chunk.id.not_in(done)).order_by(Chunk.id)
        ).all()
        if not rows:
            print(f"{model_name}: semua chunk aktif sudah punya embedding")
            return
        t0 = time.perf_counter()
        load_model(model_name)
        t_load = time.perf_counter() - t0
        # teks yang di-embed sama dengan teks yang diindeks BM25: judul dokumen + heading + isi
        texts = [RChunk(str(c.id), title, c.section or "", c.text).index_text for c, title in rows]
        t1 = time.perf_counter()
        vecs = embed_passages(texts, model_name)
        t_embed = time.perf_counter() - t1
        db.add_all(ChunkEmbedding(chunk_id=c.id, model_name=model_name, embedding=v) for (c, _), v in zip(rows, vecs))
    print(f"{model_name}: {len(rows)} chunk di-embed, dimensi {vecs.shape[1]}, "
          f"muat model {t_load:.1f} dtk, embedding {t_embed:.1f} dtk ({t_embed / len(rows) * 1000:.0f} ms/chunk)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=settings.embedding_model)
    ap.add_argument("--all", action="store_true", help="semua model di app/rag/embedding.py")
    args = ap.parse_args()
    for name in (MODELS if args.all else [args.model]):
        embed_model(name)


if __name__ == "__main__":
    main()
