"""Salin vektor KB dari pgvector (tabel chunk_embeddings, sumber utama) ke ChromaDB dan SurrealDB.

Hanya chunk aktif dari KB yang dipakai chatbot (KB_DATA_SOURCE, chunk sesuai CHUNK_ROWS/CHUNK_OVERLAP) yang
disalin, sehingga ketiga database mencari di kumpulan chunk yang sama persis. Koleksi lama dihapus lalu dibuat
ulang (KB kecil), jadi jalankan lagi setiap kali KB diimpor ulang atau di-embed ulang.

    .venv\\Scripts\\python scripts\\sync_vectors.py                 # model EMBEDDING_MODEL
    .venv\\Scripts\\python scripts\\sync_vectors.py --all           # semua model yang sudah di-embed
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.db.models import ChunkEmbedding  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.pipeline import load_chunks_from_db  # noqa: E402
from app.rag.embedding import MODELS  # noqa: E402
from app.vectorstores import get_store, kb_collection  # noqa: E402


def kb_vectors(model_name: str) -> tuple[list[int], np.ndarray]:
    active = {c.db_id for c in load_chunks_from_db()}
    with SessionLocal() as db:
        rows = db.execute(select(ChunkEmbedding.chunk_id, ChunkEmbedding.embedding)
                          .where(ChunkEmbedding.model_name == model_name).order_by(ChunkEmbedding.chunk_id)).all()
    rows = [(cid, emb) for cid, emb in rows if cid in active]
    return [cid for cid, _ in rows], np.array([np.asarray(e, dtype=np.float32) for _, e in rows])


def sync(model_name: str) -> None:
    ids, vecs = kb_vectors(model_name)
    if not ids:
        print(f"{model_name}: belum ada embedding, jalankan scripts/embed_kb.py dulu")
        return
    col = kb_collection(model_name)
    for name in ("chroma", "surrealdb"):
        store = get_store(name)
        t0 = time.perf_counter()
        store.drop(col)
        store.upsert(col, ids, vecs)
        store.build_index(col, vecs.shape[1])
        print(f"{name:10} {col}: {store.count(col)} vektor (dimensi {vecs.shape[1]}) dalam {time.perf_counter() - t0:.1f} dtk")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    print(f"KB: data_source={settings.kb_data_source}")
    for m in (MODELS if args.all else [settings.embedding_model]):
        sync(m)


if __name__ == "__main__":
    main()
