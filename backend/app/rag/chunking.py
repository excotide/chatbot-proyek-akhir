"""Chunking knowledge base.

Strategi (lihat Desain-Database.md dan keputusan tahap KB):
- Markdown (FAQ, profil, fitur, ketentuan): satu chunk per heading (## / ###). Satu topik = satu chunk,
  jadi jawaban tidak terpotong di tengah. Disimpan dengan chunk_size = chunk_overlap = 0.
- CSV (katalog produk, daftar harga): dipotong per `size` baris dengan `overlap` baris yang diulang.
  Baris header kolom diulang di setiap chunk supaya tiap chunk bisa dibaca sendiri.
"""
import csv
import io
import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,3})\s+(.*)$")


@dataclass
class RawChunk:
    """Hasil chunking sebuah file, sebelum disimpan ke database."""
    index: int
    section: str
    text: str


@dataclass
class Chunk:
    """Chunk yang dipakai retriever (sudah ada di database)."""
    id: str        # id tampilan
    doc: str       # judul dokumen
    section: str
    text: str
    db_id: int | None = None
    source: str = ""   # path file sumber, mis. data/kb/asli/daftar-harga.csv

    @property
    def index_text(self) -> str:
        """Teks yang diindeks/di-embed. Judul dan heading ikut agar pertanyaan 'harga paket' cocok."""
        return f"{self.doc} {self.section} {self.text}"


@dataclass
class Hit:
    """Satu hasil retrieval. `score` = skor utama metode (BM25, cosine, atau RRF);
    `similarity` = cosine similarity bila metode memakai embedding (dipakai untuk ambang 'tidak ada konteks')."""
    chunk: Chunk
    score: float
    similarity: float | None = None


def title_from_filename(stem: str) -> str:
    """'daftar-harga' -> 'Daftar Harga'. Dipakai untuk CSV yang tidak punya judul di dalam file."""
    return " ".join(w.capitalize() for w in re.split(r"[-_]+", stem) if w)


def _windows(items: list, size: int, overlap: int) -> list[list]:
    if size <= overlap:
        raise ValueError("ukuran chunk harus lebih besar dari overlap")
    step = size - overlap
    out, i = [], 0
    while i < len(items):
        out.append(items[i : i + size])
        if i + size >= len(items):
            break
        i += step
    return out


def chunk_markdown(md: str, fallback_title: str) -> tuple[str, list[RawChunk]]:
    """Satu chunk per heading. Teks sebelum heading pertama (selain judul) ikut sebagai bagian 'Pengantar'."""
    title, heading, lines = "", "Pengantar", []
    sections: list[tuple[str, list[str]]] = []
    for raw in md.splitlines():
        line = raw.strip()
        m = _HEADING.match(line)
        if m:
            if len(m.group(1)) == 1 and not title:
                title = m.group(2).strip()
                continue
            if lines:
                sections.append((heading, lines))
            heading, lines = m.group(2).strip(), []
        elif line:
            lines.append(line)
    if lines:
        sections.append((heading, lines))
    chunks = [RawChunk(i, h, "\n".join(ls)) for i, (h, ls) in enumerate(sections, 1)]
    return title or fallback_title, chunks


def chunk_csv(text: str, size: int, overlap: int) -> list[RawChunk]:
    rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
    if len(rows) < 2:
        return []
    header, body = rows[0], rows[1:]

    def fmt(r: list[str]) -> str:
        return "| " + " | ".join(c.strip() for c in r) + " |"

    chunks = []
    for n, window in enumerate(_windows(body, size, overlap), 1):
        first = 1 + (n - 1) * (size - overlap)
        last = first + len(window) - 1
        chunks.append(RawChunk(n, f"Baris {first}-{last}", "\n".join([fmt(header), *map(fmt, window)])))
    return chunks
