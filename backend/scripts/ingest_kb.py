"""Impor knowledge base dari data/kb/{asli,dummy} ke tabel documents dan chunks.

Jalankan dari folder backend:
    .venv\\Scripts\\python scripts\\ingest_kb.py

Aturan:
- Nama folder menentukan data_source (asli / dummy).
- File yang isinya tidak berubah (hash sama) tidak diimpor ulang.
- File yang berubah: versi lama dinonaktifkan (tidak dihapus), versi baru dibuat dengan chunk baru.
- File yang dihapus dari folder: versinya di database dinonaktifkan.
- CSV dipotong sesuai CHUNK_ROWS / CHUNK_OVERLAP di .env. Bila nilainya diubah, chunk dengan ukuran baru
  ditambahkan berdampingan dengan yang lama (untuk perbandingan strategi chunking).
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.db.models import Chunk, Document  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.rag.chunking import chunk_csv, chunk_markdown, title_from_filename  # noqa: E402

BACKEND_DIR = Path(__file__).resolve().parent.parent
SOURCES = ("asli", "dummy")


def kb_files(kb_root: Path):
    for source in SOURCES:
        folder = kb_root / source
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() in (".md", ".csv") and path.name.lower() != "readme.md":
                yield source, path


def build_chunks(path: Path, text: str) -> tuple[str, list, int, int]:
    if path.suffix.lower() == ".md":
        title, raw = chunk_markdown(text, title_from_filename(path.stem))
        return title, raw, 0, 0
    raw = chunk_csv(text, settings.chunk_rows, settings.chunk_overlap)
    return title_from_filename(path.stem), raw, settings.chunk_rows, settings.chunk_overlap


def main() -> None:
    kb_root = BACKEND_DIR / settings.kb_root
    report, seen = [], set()
    with SessionLocal() as db, db.begin():
        for source, path in kb_files(kb_root):
            rel = path.relative_to(BACKEND_DIR).as_posix()
            seen.add(rel)
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            title, raw, size, overlap = build_chunks(path, data.decode("utf-8-sig"))

            active = db.scalar(select(Document).where(Document.source_path == rel, Document.is_active))
            if active and active.content_hash == digest:
                doc, status = active, "tetap"
            else:
                if active:
                    active.is_active = False
                    db.flush()
                # versi dengan isi yang sama pernah ada (mis. file dikembalikan) -> aktifkan lagi
                doc = db.scalar(select(Document).where(Document.source_path == rel, Document.content_hash == digest))
                if doc:
                    doc.is_active, doc.title, status = True, title, "diaktifkan lagi"
                else:
                    doc = Document(title=title, source_path=rel, data_source=source, content_hash=digest)
                    db.add(doc)
                    status = "baru" if not active else "diperbarui"
                db.flush()

            have = db.scalar(select(Chunk.id).where(
                Chunk.document_id == doc.id, Chunk.chunk_size == size, Chunk.chunk_overlap == overlap).limit(1))
            if not have:
                db.add_all(Chunk(document_id=doc.id, chunk_index=r.index, section=r.section[:200], text=r.text,
                                 chunk_size=size, chunk_overlap=overlap) for r in raw)
                if status == "tetap":
                    status = f"tetap, chunk {size}/{overlap} ditambahkan"
            report.append((source, path.name, status, len(raw), "per heading" if size == 0 else f"{size} baris, overlap {overlap}"))

        for doc in db.scalars(select(Document).where(Document.is_active)):
            if doc.source_path not in seen:
                doc.is_active = False
                report.append((doc.data_source, Path(doc.source_path).name, "dinonaktifkan (file hilang)", 0, "-"))

    print(f"{'sumber':7} {'file':32} {'status':34} {'chunk':>5}  strategi")
    for source, name, status, n, strat in report:
        print(f"{source:7} {name:32} {status:34} {n:>5}  {strat}")
    for source in SOURCES:
        total = sum(n for s, _, st, n, _ in report if s == source and not st.startswith("dinonaktifkan"))
        print(f"total chunk aktif {source}: {total}")


if __name__ == "__main__":
    main()
