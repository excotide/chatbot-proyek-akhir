# Knowledge base chatbot

Pemisahan sumber data sesuai arahan dosen (bimbingan 1):

- `asli/` — **data asli**, disusun dari *Compro 29082026 - Bungkus.pdf* (company profile AutoAudit, PT Ordo Teknologi Karya). Dipakai chatbot secara default (`KB_DATA_SOURCE=asli`).
- `dummy/` — **data buatan sendiri** dari tahap simulator awal. Angkanya mengikuti mockup dashboard dan **tidak sama** dengan harga asli. Dipakai hanya bila `KB_DATA_SOURCE=dummy`.

Compro bertanda *strictly confidential*. Yang sengaja **tidak** dimasukkan: nomor rekening, nomor HP kontak sales, dan nomor telepon pelanggan pada contoh tabel di compro.

## Format

| Jenis isi | Format | Alasan |
|---|---|---|
| Profil, fitur, platform, keamanan, ketentuan | Markdown (`.md`) | teks bebas per topik |
| Katalog produk, daftar harga | CSV (`.csv`) | data berbentuk baris dan kolom |

Tabel produk dan harga **dipisah**: `katalog-produk.csv` berisi nama, kategori, deskripsi, dan fitur, sedangkan `daftar-harga.csv` berisi harga per varian. Keduanya dihubungkan dengan kolom `kode_produk`.

## Strategi chunking

- **Markdown:** satu chunk per heading (`##`). Tulis satu topik per heading. Disimpan dengan `chunk_size = 0`.
- **CSV:** per `CHUNK_ROWS` baris (default 5) dengan `CHUNK_OVERLAP` baris diulang (default 1). Baris header kolom diulang di setiap chunk.

Jumlah chunk saat ini (asli): daftar-harga 6, katalog-produk 4, fitur 11, platform 8, profil 7, keamanan 5, **total 41**.

## Memperbarui knowledge base

1. Ubah atau tambah file di `asli/` (baris pertama Markdown: `# Judul dokumen`).
2. Dari folder `backend`: `.venv\Scripts\python scripts\ingest_kb.py`
3. Muat ulang tanpa restart server: `POST /simulator/reload-kb`, atau restart server.

File yang berubah membuat versi dokumen baru di database. Versi lama dinonaktifkan, tidak dihapus, karena chunk-nya mungkin sudah tercatat sebagai sumber jawaban bot.
