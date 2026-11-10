# Hasil Evaluasi Retrieval

- **Tanggal:** 09 November 2026 16:27
- **Perangkat:** 13th Gen Intel(R) Core(TM) i5-1335U, CPU saja, Python 3.12.7
- **KB:** 41 chunk (data_source=asli; Markdown per heading, CSV 5 baris/overlap 1)
- **Set uji:** 40 pertanyaan (32 ada di KB, 8 di luar KB), `data/eval/testset_retrieval.csv`. **Status: draf, belum ditinjau.**
- **Vektor:** PostgreSQL + pgvector, pencarian exact (tanpa indeks). Hybrid = RRF k=60.

## Ringkasan

| Konfigurasi | Hit@1 | Hit@3 | MRR@5 | Hit@3 persis | Hit@3 parafrase | Hit@3 santai | Ambang terbaik | Akurasi seimbang luar KB | Waktu rata-rata (ms) |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 78% | 91% | 0.84 | 100% | 83% | 90% | 5.17 | 86% (5 KB ditolak, 1 luar KB lolos) | 0 |
| semantic · all-indo-e5-small-v4 | 50% | 81% | 0.67 | 60% | 92% | 90% | cos 0.274 | 69% (0 KB ditolak, 5 luar KB lolos) | 24 |
| hybrid · all-indo-e5-small-v4 | 78% | 84% | 0.82 | 80% | 83% | 90% | cos 0.274 | 69% (0 KB ditolak, 5 luar KB lolos) | 24 |
| semantic · indo-sentence-bert-base | 53% | 81% | 0.66 | 80% | 75% | 90% | cos 0.384 | 70% (3 KB ditolak, 4 luar KB lolos) | 36 |
| hybrid · indo-sentence-bert-base | 72% | 91% | 0.81 | 100% | 83% | 90% | cos 0.377 | 70% (3 KB ditolak, 4 luar KB lolos) | 36 |
| semantic · multilingual-e5-small | 75% | 94% | 0.85 | 90% | 92% | 100% | cos 0.832 | 78% (2 KB ditolak, 3 luar KB lolos) | 24 |
| hybrid · multilingual-e5-small | 81% | 91% | 0.88 | 100% | 83% | 90% | cos 0.832 | 77% (3 KB ditolak, 3 luar KB lolos) | 25 |

Catatan:
- Ambang dicari dari set uji yang sama, jadi angkanya optimistis. Gunakan sebagai titik awal, bukan hasil akhir.
- Waktu semantic/hybrid termasuk membuat embedding pertanyaan di CPU.
- Detail per pertanyaan: `data/eval/detail-retrieval.csv`.
