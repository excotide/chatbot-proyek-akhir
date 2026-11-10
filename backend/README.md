# Backend Orchestrator + WhatsApp (OpenWA)

Prototipe koneksi WhatsApp untuk PA. Pesan masuk lewat **OpenWA** (gateway tidak resmi, engine Baileys),
diterima backend FastAPI, lalu dibalas. Saat ini balasannya masih *echo*; nanti diganti pipeline RAG + klasifikasi.

```
HP pelanggan ──▶ WhatsApp ──▶ OpenWA (Docker, :2785) ──webhook──▶ FastAPI (:8000) /webhook/openwa
HP pelanggan ◀── WhatsApp ◀── OpenWA  ◀──── POST send-text ─────── FastAPI
```

Sumber pesan bisa diganti ke Meta Cloud API resmi lewat `MESSAGING_PROVIDER=cloud_api` di `.env`
(endpoint `/webhook/meta` sudah ada), tanpa mengubah kode RAG/klasifikasi.

## Database

PostgreSQL 16 + pgvector di Docker (folder `../database`). Rancangan tabel: `Desain-Database.md` di folder PA.

1. Pertama kali saja: salin `../database/.env.example` ke `../database/.env`, isi password, lalu isi `DATABASE_URL` di `backend/.env` dengan password yang sama.
2. Nyalakan database (Docker Desktop harus jalan):
   ```
   cd "D:\TUGAS DAN MATERI\SMT5\PA\Proyek PA\database"
   docker compose up -d
   ```
3. Buat atau perbarui tabel (dari folder `backend`):
   ```
   .venv\Scripts\alembic upgrade head
   ```

Port database `127.0.0.1:5434` (5433 sudah dipakai PostgreSQL lain di laptop ini). Model tabel ada di `app/db/models.py`. Setelah mengubah model, buat migrasi baru:
```
.venv\Scripts\alembic revision --autogenerate -m "keterangan perubahan"
.venv\Scripts\alembic upgrade head
```

## Simulator chatbot (tanpa WhatsApp)

Untuk mengembangkan RAG tanpa menyentuh WhatsApp. Cukup backend, tidak perlu Docker/OpenWA:

```
cd "D:\TUGAS DAN MATERI\SMT5\PA\Proyek PA\backend"
.venv\Scripts\python -m uvicorn app.main:app --port 8000
```

Buka http://localhost:8000/simulator. Panel kanan menampilkan chunk yang dipakai beserta skor, mode jawaban, dan waktu per tahap.

- **Knowledge base:** dibaca dari database (tabel `documents` dan `chunks`), hasil impor `data/kb/asli` lewat `.venv\Scripts\python scripts\ingest_kb.py`. Detail format dan chunking ada di `data/kb/README.md`. Pengaturan di `.env`: `KB_DATA_SOURCE` (asli/dummy), `CHUNK_ROWS`, `CHUNK_OVERLAP`, `TOP_K`, `MIN_SCORE`.
- **Retrieval:** `semantic` (default, embedding + cosine di pgvector), `hybrid` (BM25 + semantic, RRF), atau `bm25` (kata kunci). Default diatur `RETRIEVER` di `.env`; di simulator metode bisa dipilih per pesan lewat dropdown di header. Hasil evaluasi ada di `data/eval/hasil-retrieval.md`. Kode di `app/rag/retrievers.py`.
- **Tulis ulang pertanyaan:** setiap pesan ditulis ulang oleh LLM menjadi pertanyaan lengkap berbahasa baku ("keduanya" dilengkapi dari riwayat, "yg setaun brp" dibakukan). Pencarian memakai versi asli dan versi tulis ulang, lalu peringkatnya digabung (RRF). Matikan dengan `QUERY_REWRITE=false`.
- **Embedding:** model `EMBEDDING_MODEL` (default `intfloat/multilingual-e5-small`, terbaik dari 3 model yang dibandingkan), dijalankan di CPU. Model diunduh sekali ke `Proyek PA/.cache/huggingface`. Setelah impor KB, buat vektornya: `.venv\Scripts\python scripts\embed_kb.py` (tambah `--all` untuk ketiga model pembanding).
- **LLM:** dipilih lewat `LLM_PROVIDER` di `.env`: `ollama` (lokal, `OLLAMA_MODEL`, mis. `qwen2.5:3b`) atau `groq` (cloud, `GROQ_MODEL`, default `openai/gpt-oss-120b`; `GROQ_API_KEY` di `Proyek PA/.env`). Dengan Groq, isi chat dan potongan KB dikirim ke server Groq. Bila LLM tidak tersedia, simulator menampilkan potongan dokumen paling relevan saja.
- **Pertanyaan di luar dokumen:** ambang skor dimatikan (`MIN_SCORE=0`, `MIN_SIMILARITY=0`) karena evaluasi menunjukkan tidak ada nilai yang memisahkan pertanyaan KB dan luar KB. LLM yang memutuskan: bila jawabannya tidak ada di konteks, LLM membalas penanda `TIDAK_ADA_INFO` dan bot meneruskan ke sales.
- **Perilaku bot:** `TOP_K=5` chunk dikirim ke LLM. Bila baris katalog suatu produk terambil tanpa baris harganya, harga produk itu dilampirkan otomatis lewat `kode_produk` (tabel tetap terpisah di database). Bot boleh merekomendasikan paket berdasarkan fitur yang tertulis, dan permintaan dihubungkan ke sales (termasuk jawaban 'ya' atas tawaran itu) dibalas dengan serah terima dan ditandai eskalasi.
- **Pilihan di simulator:** panel kiri berisi daftar percakapan simulator dari database (klik untuk membuka dan melanjutkan; *+ Baru* membuat lead baru). Header berisi dropdown metode pencarian dan model LLM per pesan: model Groq dari `GROQ_MODELS` di `.env` dan model Ollama yang sudah di-pull. Model yang dipakai tercatat di `bot_responses.llm_model`.
- **Riwayat percakapan:** disimpan di database (`leads`, `messages`, `bot_responses`, `retrieval_results`). Satu sesi browser = satu lead (`channel=simulator`, `data_source=dummy`). `HISTORY_MESSAGES` pesan terakhir dibawa ke LLM, dan pertanyaan lanjutan ditulis ulang sebelum retrieval (`QUERY_REWRITE`). Tombol *Percakapan baru* membuat lead baru; percakapan lama tidak dihapus. Kode: `app/conversation.py`.
- **Evaluasi retrieval:** `.venv\Scripts\python scripts\eval_retrieval.py` memakai set uji `data/eval/testset_retrieval.csv` dan menulis hasil ke `data/eval/hasil-retrieval.md`.
- Pipeline ada di `app/pipeline.py` (`generate_reply`). Jalur WhatsApp di `main.py` masih echo; tinggal memanggil fungsi itu dari `handle_message` bila sudah siap.

## Menjalankan WhatsApp

1. **Nyalakan Docker Desktop**, tunggu sampai statusnya *Engine running*.
2. **Jalankan OpenWA** (folder `wa-gateway`):
   ```
   cd "D:\TUGAS DAN MATERI\SMT5\PA\Proyek PA\wa-gateway"
   docker compose up -d
   ```
3. **Jalankan backend** (folder `backend`, terminal terpisah):
   ```
   cd "D:\TUGAS DAN MATERI\SMT5\PA\Proyek PA\backend"
   .venv\Scripts\activate
   uvicorn app.main:app --reload --port 8000
   ```
4. **Siapkan sesi WhatsApp** (terminal lain, folder `backend`):
   ```
   .venv\Scripts\python scripts\setup_openwa.py
   ```
   Skrip membuat sesi `pa-bot`, mendaftarkan webhook, lalu membuka `qr.png`.
   Scan dari HP **nomor khusus bot**: WhatsApp → Perangkat tertaut → Tautkan perangkat.
5. **Uji**: dari HP lain, kirim pesan ke nomor bot. Log uvicorn menampilkan `PESAN MASUK` dan `BALASAN TERKIRIM`.

Dashboard OpenWA: http://localhost:2785 (login pakai `API_MASTER_KEY` di `wa-gateway/.env`).

## Catatan penting

- OpenWA **tidak resmi** (meniru WhatsApp Web). Ada risiko nomor dibatasi/diblokir → pakai nomor cadangan,
  jangan kirim pesan massal. Untuk produksi tetap Meta Cloud API.
- File `.env` dan folder `wa-gateway/data` berisi kunci dan sesi login WhatsApp. Jangan dibagikan atau di-commit.
- Sesi tersimpan di `wa-gateway/data`, jadi setelah restart tidak perlu scan ulang.

## Struktur

```
app/
  main.py              endpoint webhook + handle_message (tempat pipeline RAG nanti)
  config.py            baca .env
  messaging/
    base.py            IncomingMessage + interface MessagingProvider
    openwa.py          adapter OpenWA (HMAC X-OpenWA-Signature, send-text)
    cloud_api.py       adapter Meta Cloud API (X-Hub-Signature-256, Graph API)
scripts/setup_openwa.py  buat sesi, daftar webhook, tampilkan QR
```
