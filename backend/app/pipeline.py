"""Pipeline chatbot: pertanyaan -> retrieval -> LLM -> jawaban.

Dipakai simulator web sekarang. Untuk WhatsApp nanti cukup panggil `generate_reply()`
dari `handle_message` di main.py, tanpa mengubah isinya.
"""
import logging
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache

from sqlalchemy import and_, or_, select

from app.config import settings
from app.db.models import Chunk as ChunkRow
from app.db.models import Document
from app.db.session import SessionLocal
from app.rag import llm
from app.rag.chunking import Chunk
from app.rag.retrievers import build_retriever, has_context

log = logging.getLogger("pipeline")

# Penanda yang wajib dibalas LLM bila jawabannya tidak ada di konteks; pipeline mengubahnya jadi eskalasi ke sales.
NO_INFO = "TIDAK_ADA_INFO"
# Penanda bila pelanggan minta dihubungkan dengan sales/manusia (atau menyetujui tawaran itu).
HANDOFF = "ESKALASI_SALES"
HANDOFF_REPLY = ("Baik, aku teruskan ke tim sales ya. Nanti kamu dihubungi langsung untuk membahas kebutuhanmu "
                 "lebih lanjut.")

SYSTEM_PROMPT = (
    "Kamu asisten penjualan AutoAudit dari PT Ordo Teknologi Karya yang membalas chat WhatsApp calon pelanggan.\n"
    "Aturan:\n"
    "1. Gunakan HANYA fakta yang tertulis di KONTEKS. Jangan menambah nama orang, angka, harga, fitur, atau kebijakan "
    "yang tidak tertulis di KONTEKS.\n"
    "2. Pelanggan sering memakai bahasa santai, singkatan, atau sinonim, dan bisa menyebut nama produk berbahasa "
    "Inggris dalam terjemahan Indonesia (misalnya 'paket merah' untuk 'Red Package'). Satuan waktu juga bisa berbeda "
    "(misalnya 'setahun' = '12 bulan'). Pahami maksudnya, cocokkan dengan nama produk dan varian di KONTEKS, lalu "
    "jawab. Jika KONTEKS hanya menjawab sebagian, jawab bagian itu.\n"
    f"3. Jika jawaban sama sekali tidak tertulis di KONTEKS, balas hanya dengan satu kata: {NO_INFO}\n"
    "4. Jika ada beberapa paket atau varian yang cocok dengan pertanyaan, sebutkan semuanya beserta harganya.\n"
    "5. Saat membandingkan paket, sebutkan hanya perbedaan yang tertulis (misalnya fitur yang termasuk atau tidak "
    "termasuk). Jangan menyimpulkan perbedaan yang tidak tertulis.\n"
    "5a. Jika pelanggan meminta saran paket untuk suatu kebutuhan, rekomendasikan paket yang 'fitur termasuk'-nya di "
    "KONTEKS memuat kebutuhan itu, sebutkan alasannya dari fitur tersebut beserta harganya. Paket yang mencantumkan "
    "kebutuhan itu di 'fitur tidak termasuk' bukan rekomendasi. Ini penalaran dari fakta di KONTEKS, bukan menambah fakta.\n"
    f"5b. Jika pelanggan meminta dihubungkan dengan tim sales atau manusia, atau menyetujui tawaran untuk dihubungi, "
    f"balas hanya dengan satu kata: {HANDOFF}\n"
    "6. Tulis harga beserta satuannya persis seperti di KONTEKS (misalnya 'per akun'). Jangan menambah satuan lain "
    "seperti 'per bulan' jika tidak tertulis.\n"
    "7. Jika pelanggan meminta potongan harga di luar harga paket, jangan menjanjikan apa pun dan katakan permintaannya "
    "akan diteruskan ke tim sales.\n"
    "8. Sebut nama produk atau varian (misalnya 'Paket Reguler 12 Bulan'), bukan kode internal seperti AA-REGULER.\n"
    "9. Gunakan bahasa Indonesia yang ramah dan sopan, ringkas (maksimal sekitar 10 baris). Jangan menyebut "
    "istilah internal seperti 'KONTEKS', 'dokumen', 'katalog', atau 'tabel'. Jika suatu informasi (misalnya harga) "
    "tidak tersedia, jangan dibahas.\n"
    "10. Format untuk WhatsApp: jika jawaban menyebut 3 hal atau lebih (fitur, varian, harga), WAJIB ditulis sebagai "
    "daftar: buka dengan satu kalimat singkat, lalu satu butir per baris diawali '- '. Jangan gabungkan banyak butir "
    "dalam satu kalimat panjang. Jangan tambahkan kalimat penutup yang menjelaskan alasan secara kaku. Tebalkan nama produk dan harga dengan SATU bintang, "
    "misalnya *Red Package* dan *Rp5.000.000 per bulan*. Jangan memakai dua bintang, tanda #, tabel, atau tautan Markdown."
)

NO_CONTEXT_REPLY = (
    "Maaf, aku belum punya informasi soal itu. Pertanyaanmu aku teruskan ke tim sales ya, "
    "nanti dibantu langsung."
)

# Nama kolom CSV -> label yang mudah dibaca LLM
_COLUMN_LABELS = {
    "kode_produk": "kode produk", "nama_produk": "nama produk", "fitur_termasuk": "fitur termasuk",
    "fitur_tidak_termasuk": "fitur tidak termasuk", "minimal_langganan": "minimal langganan", "harga_rp": "harga",
}


def _rupiah(value: str) -> str:
    """'333000' -> 'Rp333.000', '470-730' -> 'Rp470-Rp730'."""
    parts = value.split("-")
    if all(p.isdigit() for p in parts):
        return "-".join("Rp" + f"{int(p):,}".replace(",", ".") for p in parts)
    return value


def product_names(chunks) -> dict[str, str]:
    """kode_produk -> nama_produk dari chunk katalog. Dipakai untuk melengkapi baris harga di prompt,
    karena tabel harga dan katalog disimpan terpisah."""
    names: dict[str, str] = {}
    for c in chunks:
        lines = c.text.splitlines()
        if not lines or not lines[0].startswith("|"):
            continue
        header = [h.strip() for h in lines[0].strip("|").split("|")]
        if "kode_produk" in header and "nama_produk" in header:
            ik, iname = header.index("kode_produk"), header.index("nama_produk")
            for row in lines[1:]:
                cells = [x.strip() for x in row.strip("|").split("|")]
                if len(cells) > max(ik, iname):
                    names[cells[ik]] = cells[iname]
    return names


def _table_to_lines(text: str, names: dict[str, str] | None = None) -> str:
    """Tabel '| a | b |' (chunk CSV) diubah jadi baris 'kolom: nilai' agar mudah dibaca model kecil.
    Di baris harga, kode produk dilengkapi nama produk dari katalog (data di database tetap terpisah)."""
    lines = text.splitlines()
    if not lines or not lines[0].startswith("|"):
        return text
    header = [h.strip() for h in lines[0].strip("|").split("|")]
    out = []
    for row in lines[1:]:
        cells = [c.strip() for c in row.strip("|").split("|")]
        pairs = []
        for h, v in zip(header, cells):
            if v in ("", "-"):
                continue
            if h == "harga_rp":
                v = _rupiah(v)
            elif h == "kode_produk" and names and v in names and "nama_produk" not in header:
                v = f"{v} ({names[v]})"
            pairs.append(f"{_COLUMN_LABELS.get(h, h)}: {v}")
        out.append("- " + "; ".join(pairs))
    return "\n".join(out)


_GREETING = re.compile(r"^\s*(halo+|hai+|hi+|hello|selamat\s+(pagi|siang|sore|malam)|permisi|assalamu.*|p)\b[\s!.,?]*$", re.I)
_THANKS = re.compile(r"^\s*(terima\s*kasih|makasih|thanks?|thx|oke|ok|siap)\b.*$", re.I)
GREETING_REPLY = "Halo! Aku asisten AutoAudit. Ada yang bisa kubantu soal harga paket, token, atau cara pembayaran?"
THANKS_REPLY = "Sama-sama! Kalau ada pertanyaan lain, tinggal kabari ya."


@dataclass
class Source:
    doc: str
    section: str
    score: float
    text: str
    chunk_id: int | None = None   # id di tabel chunks, untuk log retrieval_results
    similarity: float | None = None  # cosine, bila metode memakai embedding


@dataclass
class BotReply:
    text: str
    mode: str                       # "llm" | "retrieval_only" | "no_context" | "smalltalk"
    sources: list[Source] = field(default_factory=list)
    escalate: bool = False          # perlu diteruskan ke sales
    latency_ms: dict = field(default_factory=dict)
    note: str = ""
    retriever: str = ""
    search_query: str = ""          # pertanyaan hasil tulis ulang (kosong bila tidak diubah)
    llm_model: str = ""             # model yang menyusun jawaban, mis. groq:openai/gpt-oss-120b (kosong bila tidak dipanggil)
    vector_db: str = ""             # database vektor yang dipakai (kosong untuk BM25 / sapaan)


def load_chunks_from_db() -> list[Chunk]:
    """Chunk aktif dari database: Markdown per heading (0/0) + CSV dengan ukuran sesuai .env."""
    stmt = (
        select(ChunkRow, Document.title, Document.source_path)
        .join(Document, ChunkRow.document_id == Document.id)
        .where(
            Document.is_active,
            Document.data_source == settings.kb_data_source,
            or_(
                and_(ChunkRow.chunk_size == 0, ChunkRow.chunk_overlap == 0),
                and_(ChunkRow.chunk_size == settings.chunk_rows, ChunkRow.chunk_overlap == settings.chunk_overlap),
            ),
        )
        .order_by(Document.id, ChunkRow.chunk_index)
    )
    with SessionLocal() as db:
        return [Chunk(id=str(c.id), doc=title, section=c.section or "", text=c.text, db_id=c.id, source=path)
                for c, title, path in db.execute(stmt)]


METHODS = ("semantic", "hybrid", "bm25")


@lru_cache(maxsize=1)
def _chunks() -> tuple:
    return tuple(load_chunks_from_db())


VECTOR_DBS = ("pgvector", "chroma", "surrealdb")


@lru_cache(maxsize=16)
def get_retriever(method: str | None = None, vector_db: str | None = None):
    """Retriever per (metode, database vektor), dibuat sekali lalu disimpan. Default: RETRIEVER dan VECTOR_DB di .env.
    BM25 tidak memakai database vektor."""
    method = method or settings.retriever
    vector_db = None if method == "bm25" else (vector_db or settings.vector_db)
    chunks = list(_chunks())
    if not chunks:
        log.warning("KB kosong untuk data_source=%s. Jalankan scripts/ingest_kb.py dulu.", settings.kb_data_source)
    log.info("KB dimuat dari database: %d chunk (data_source=%s, CSV %d baris/overlap %d, retriever=%s)",
             len(chunks), settings.kb_data_source, settings.chunk_rows, settings.chunk_overlap, method)
    return build_retriever(method, chunks, vector_db=vector_db)


def reload_kb() -> int:
    """Kosongkan cache chunk dan retriever (setelah scripts/ingest_kb.py atau embed_kb.py)."""
    _chunks.cache_clear()
    get_retriever.cache_clear()
    return len(_chunks())


REWRITE_PROMPT = (
    "Tugasmu menulis ulang pesan terakhir pelanggan menjadi pertanyaan lengkap dalam bahasa Indonesia baku yang bisa "
    "dipahami tanpa membaca riwayat. Hasilnya dipakai untuk mencari dokumen, bukan untuk dikirim ke pelanggan.\n"
    "Aturan:\n"
    "1. Jika pesan terakhir memakai kata rujukan eksplisit ke hal di riwayat (misalnya 'keduanya', 'yang itu', "
    "'tersebut', 'yang tadi', 'kalau yang 6 bulan?', 'fiturnya apa'), lengkapi rujukannya dengan nama produk atau "
    "topik dari riwayat. Jangan mengganti kata umum seperti 'paket' atau 'harga' dengan produk tertentu dari riwayat; "
    "misalnya 'Harga paket 12 bulan berapa?' tetap ditulis tentang paket 12 bulan secara umum.\n"
    "2. Ubah bahasa santai dan singkatan chat menjadi bahasa baku (misalnya 'brp' menjadi 'berapa', 'gmn' menjadi "
    "'bagaimana'), tanpa mengubah maksudnya.\n"
    "3. Jika pelanggan menyebut produk dengan terjemahan atau sebutan lain (misalnya 'paket merah', 'paket biru', "
    "'storage tambahan'), ganti dengan nama yang sesuai dari DAFTAR NAMA PRODUK RESMI. Nyatakan durasi langganan "
    "dalam bulan seperti di katalog (misalnya 'setahun' menjadi '12 bulan', 'setengah tahun' menjadi '6 bulan').\n"
    "4. Jika pesan terakhir adalah jawaban singkat ('ya', 'boleh', 'oke', 'tidak') atas tawaran atau pertanyaan bot, "
    "tulis ulang sebagai pernyataan pelanggan yang lengkap dari sudut pandang pelanggan (misalnya 'Ya, saya ingin "
    "dihubungkan dengan tim untuk membahas integrasi ERP').\n"
    "5. Jika pesan terakhir membahas topik baru, jangan bawa topik dari riwayat.\n"
    "6. Jangan menjawab pertanyaan dan jangan menambah informasi yang tidak ada di pesan atau riwayat.\n"
    "7. Balas hanya dengan satu kalimat, tanpa penjelasan."
)


_REFERENCE = re.compile(
    r"\b(keduanya|ketiganya|semuanya|itu|tersebut|tadi|sebelumnya|barusan|di atas|\w{2,}nya)\b"
    r"|^\s*(ya|iya|yap|boleh|oke|ok|siap|mau|gak|nggak|ngga|tidak|enggak|lanjut)\b",
    re.I,
)


def refers_to_history(text: str) -> bool:
    """True bila pesan kemungkinan merujuk ke percakapan sebelumnya: ada kata rujukan ('keduanya', 'itu',
    'tersebut', akhiran '-nya'), jawaban singkat ('ya', 'boleh', 'gak'), atau pesan sangat pendek
    ('kalau yang 6 bulan?'). Aturan pasti ini menggantikan keputusan LLM yang terbukti terlalu sering
    menyambungkan pertanyaan baru ke topik lama."""
    return bool(_REFERENCE.search(text)) or len(text.split()) <= 4


def to_whatsapp(text: str) -> str:
    """Ubah sisa Markdown dari LLM ke format yang didukung WhatsApp.
    WhatsApp: *tebal*, _miring_, ~coret~, daftar '- ' / '1. '. Tidak mengenal **, #, tabel, atau [teks](tautan)."""
    t = text.replace(" ", " ").replace(" ", " ")
    t = re.sub(r"\*\*(.+?)\*\*", r"*\1*", t)                         # **tebal** -> *tebal*
    t = re.sub(r"__(.+?)__", r"_\1_", t)                              # __miring__ -> _miring_
    t = re.sub(r"^\s{0,3}#{1,6}\s*(.+)$", r"*\1*", t, flags=re.M)     # ### Judul -> *Judul*
    t = re.sub(r"^(\s*)[*•]\s+", r"\1- ", t, flags=re.M)         # '* butir' / '• butir' -> '- butir'
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", t)             # [teks](url) -> teks (url)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _fuse(rankings: list[list], k: int) -> list:
    """Gabungkan beberapa daftar Hit dengan Reciprocal Rank Fusion; tiap chunk memakai Hit dengan skor tertinggi."""
    fused: dict[int, float] = {}
    best: dict[int, object] = {}
    for ranking in rankings:
        for rank, h in enumerate(ranking, 1):
            cid = h.chunk.db_id
            fused[cid] = fused.get(cid, 0.0) + 1 / (settings.rrf_k + rank)
            if cid not in best or h.score > best[cid].score:
                best[cid] = h
    order = sorted(fused, key=lambda c: fused[c], reverse=True)[:k]
    return [best[c] for c in order]


async def rewrite_query(text: str, history: list[dict], product_list: list[str] | None = None,
                        model: str | None = None) -> str:
    """Pesan pelanggan -> pertanyaan lengkap berbahasa baku untuk retrieval dan LLM.
    product_list: nama produk resmi dari katalog, agar sebutan pelanggan ('paket merah') diganti nama resmi
    ('Red Package'). Bila gagal atau hasilnya aneh, teks asli yang dipakai."""
    lines = [f"{'Pelanggan' if m['role'] == 'user' else 'Bot'}: {m['content'][:300]}"
             for m in history[-settings.history_messages:]]
    riwayat = "\n".join(lines) if lines else "(belum ada)"
    produk = ", ".join(product_list) if product_list else "(tidak ada)"
    msg = f"DAFTAR NAMA PRODUK RESMI:\n{produk}\n\nRIWAYAT:\n{riwayat}\n\nPESAN TERAKHIR PELANGGAN:\n{text}"
    out = await llm.chat([{"role": "system", "content": REWRITE_PROMPT}, {"role": "user", "content": msg}], model)
    out = out.strip().strip('"').strip("'").splitlines()[0].strip() if out.strip() else ""
    if not out or NO_INFO in out or HANDOFF in out or len(out) > 3 * len(text) + 200:
        return text
    return out


def _table_rows(text: str) -> tuple[list[str], list[list[str]]]:
    lines = text.splitlines()
    if not lines or not lines[0].startswith("|"):
        return [], []
    header = [h.strip() for h in lines[0].strip("|").split("|")]
    return header, [[c.strip() for c in row.strip("|").split("|")] for row in lines[1:]]


def price_index(chunks) -> dict[str, list[str]]:
    """kode_produk -> baris harga (sudah diformat), dari chunk daftar harga. Baris ganda akibat overlap dibuang."""
    index: dict[str, list[str]] = {}
    for c in chunks:
        header, rows = _table_rows(c.text)
        if "harga_rp" not in header or "kode_produk" not in header:
            continue
        ik = header.index("kode_produk")
        for cells in rows:
            line = _table_to_lines("| " + " | ".join(header) + " |\n| " + " | ".join(cells) + " |")
            if line not in index.setdefault(cells[ik], []):
                index[cells[ik]].append(line)
    return index


def _linked_prices(sources: list[Source], prices: dict[str, list[str]]) -> str:
    """Tabel katalog dan harga disimpan terpisah. Bila baris katalog suatu produk ada di konteks tetapi baris
    harganya tidak, lampirkan harga produk itu (seperti JOIN lewat kode_produk) saat prompt disusun."""
    in_catalog, in_prices = [], set()
    for s in sources:
        header, rows = _table_rows(s.text)
        if "kode_produk" not in header:
            continue
        ik = header.index("kode_produk")
        for cells in rows:
            if "harga_rp" in header:
                in_prices.add(cells[ik])
            elif "nama_produk" in header and cells[ik] not in in_catalog:
                in_catalog.append(cells[ik])
    lines = [line for code in in_catalog if code not in in_prices for line in prices.get(code, [])]
    return "\n".join(lines)


def _build_messages(text: str, history: list[dict], sources: list[Source], names: dict[str, str],
                    search_query: str = "", prices: dict[str, list[str]] | None = None) -> list[dict]:
    context = "\n\n".join(f"[{i}] {s.doc} - {s.section}\n{_table_to_lines(s.text, names)}" for i, s in enumerate(sources, 1))
    linked = _linked_prices(sources, prices or {})
    if linked:
        context += f"\n\n[{len(sources) + 1}] Daftar Harga - harga produk di atas\n{linked}"
    question = f"PERTANYAAN PELANGGAN:\n{text}"
    if search_query and search_query != text:
        question += (f"\nMAKSUD PERTANYAAN (versi lengkap dan baku, hasil tulis ulang otomatis): {search_query}\n"
                     "Jika MAKSUD PERTANYAAN tidak cocok dengan KONTEKS atau tidak sesuai pesan asli, utamakan "
                     "PERTANYAAN PELANGGAN yang asli.")
    # Aturan inti diulang di akhir: model kecil lebih patuh pada instruksi yang paling dekat dengan pertanyaan.
    user = (f"KONTEKS:\n{context}\n\n{question}\n\n"
            f"Jawab hanya dari KONTEKS di atas. Jika pelanggan minta saran paket, rekomendasikan paket yang fiturnya "
            f"di KONTEKS memenuhi kebutuhannya. Jika pelanggan minta dihubungkan dengan sales atau menyetujui tawaran "
            f"dihubungi, balas: {HANDOFF}. Jika jawabannya tidak tertulis di KONTEKS, balas: {NO_INFO}")
    return [{"role": "system", "content": SYSTEM_PROMPT}, *history[-settings.history_messages:],
            {"role": "user", "content": user}]


def _fallback_text(sources: list[Source]) -> str:
    best = sources[0]
    return f"(Mode tanpa LLM) Bagian dokumen paling relevan, {best.doc} - {best.section}:\n{_table_to_lines(best.text)}"


async def generate_reply(text: str, history: list[dict] | None = None, method: str | None = None,
                         llm_model: str | None = None, vector_db: str | None = None) -> BotReply:
    """history: daftar {"role": "user"|"assistant", "content": str} dari percakapan sebelumnya.
    method: metode pencarian untuk pesan ini (bm25 | semantic | hybrid); kosong = default .env.
    llm_model: model LLM untuk pesan ini, mis. 'groq:openai/gpt-oss-20b'; kosong = default .env."""
    history = history or []
    t0 = time.perf_counter()

    for pattern, canned in ((_GREETING, GREETING_REPLY), (_THANKS, THANKS_REPLY)):
        if pattern.match(text):
            return BotReply(canned, "smalltalk", latency_ms={"total": round((time.perf_counter() - t0) * 1000, 1)},
                            note="Sapaan/basa-basi, retrieval dan LLM dilewati.")

    if method and method not in METHODS:
        raise ValueError(f"Metode tidak dikenal: {method}")
    if vector_db and vector_db not in VECTOR_DBS:
        raise ValueError(f"Database vektor tidak dikenal: {vector_db}")
    retriever = get_retriever(method, vector_db)
    latency: dict = {}

    # Setiap pesan ditulis ulang oleh LLM jadi pertanyaan lengkap berbahasa baku sebelum dicari:
    # - pertanyaan lanjutan ("keduanya", "kalau yang 6 bulan?") dilengkapi dari riwayat,
    # - bahasa santai ("yg setaun brp") dibakukan; tanpa ini LLM penjawab menolak menyamakan 'setaun' dengan '12 bulan'.
    # Topik baru tidak mewarisi topik lama.
    search_query = text
    if settings.query_rewrite:
        t_rw = time.perf_counter()
        try:
            names = sorted(set(product_names(retriever.chunks).values()))
            # Riwayat hanya diberikan bila pesan memang merujuk ke belakang. Tanpa batas ini, LLM penulis ulang
            # mengubah pertanyaan umum ("Harga paket 12 bulan berapa?") jadi tentang produk yang baru dibahas.
            search_query = await rewrite_query(text, history if refers_to_history(text) else [], names, llm_model)
        except llm.LLMUnavailable as e:
            log.warning("Tulis ulang pertanyaan dilewati: %s", e)
        latency["rewrite"] = round((time.perf_counter() - t_rw) * 1000, 1)
    rewritten = search_query if search_query != text else ""

    t_ret = time.perf_counter()
    hits = retriever.search(search_query, settings.top_k)
    if rewritten:
        # Tulis ulang kadang membuat pencarian meleset (mis. 'aman' jadi cocok ke 'Aturan aman'),
        # jadi pesan asli juga dicari dan kedua peringkat digabung (multi-query, RRF).
        hits = _fuse([hits, retriever.search(text, settings.top_k)], settings.top_k)
    t_retrieval = time.perf_counter()

    sources = [Source(h.chunk.doc, h.chunk.section, round(h.score, 4), h.chunk.text, h.chunk.db_id,
                      None if h.similarity is None else round(h.similarity, 4)) for h in hits]
    latency["retrieval"] = round((t_retrieval - t_ret) * 1000, 1)

    if not has_context(retriever.name, hits):
        latency["total"] = round((time.perf_counter() - t0) * 1000, 1)
        return BotReply(NO_CONTEXT_REPLY, "no_context", sources, escalate=True, latency_ms=latency,
                        note="Skor retrieval di bawah ambang batas, LLM tidak dipanggil.", retriever=retriever.name, vector_db=getattr(retriever, 'vector_db', ''),
                        search_query=rewritten)

    try:
        answer = await llm.chat(_build_messages(text, history, sources, product_names(retriever.chunks), search_query,
                                                price_index(retriever.chunks)), llm_model)
        answer = to_whatsapp(answer)
        mode, note = "llm", ""
        if HANDOFF in answer:
            now = time.perf_counter()
            latency["llm"] = round((now - t_retrieval) * 1000, 1)
            latency["total"] = round((now - t0) * 1000, 1)
            return BotReply(HANDOFF_REPLY, "llm", sources, escalate=True, latency_ms=latency,
                            note="Pelanggan minta dihubungkan dengan sales.", retriever=retriever.name, vector_db=getattr(retriever, 'vector_db', ''),
                            search_query=rewritten, llm_model=llm.model_label(llm_model))
        if NO_INFO in answer:
            now = time.perf_counter()
            latency["llm"] = round((now - t_retrieval) * 1000, 1)
            latency["total"] = round((now - t0) * 1000, 1)
            return BotReply(NO_CONTEXT_REPLY, "no_context", sources, escalate=True, latency_ms=latency,
                            note="LLM menyatakan konteks tidak memuat jawaban.", retriever=retriever.name, vector_db=getattr(retriever, 'vector_db', ''),
                            search_query=rewritten, llm_model=llm.model_label(llm_model))
    except llm.LLMUnavailable as e:
        log.warning("LLM tidak tersedia: %s", e)
        answer, mode, note = _fallback_text(sources), "retrieval_only", str(e)

    now = time.perf_counter()
    latency["llm"] = round((now - t_retrieval) * 1000, 1)
    latency["total"] = round((now - t0) * 1000, 1)
    return BotReply(answer, mode, sources, latency_ms=latency, note=note, retriever=retriever.name, vector_db=getattr(retriever, 'vector_db', ''),
                    search_query=rewritten, llm_model=llm.model_label(llm_model) if mode == "llm" else "")
