"""Evaluasi metode retrieval dan model embedding dengan set uji data/eval/testset_retrieval.csv.

Jalankan dari folder backend (setelah ingest_kb.py dan embed_kb.py --all):
    .venv\\Scripts\\python scripts\\eval_retrieval.py

Metrik (pertanyaan yang jawabannya ada di KB):
- Hit@1 / Hit@3 : chunk relevan ada di peringkat 1 / tiga teratas
- MRR@5         : rata-rata 1/peringkat chunk relevan pertama di lima teratas (0 bila tidak ada)
Pertanyaan di luar KB:
- Ambang terbaik dicari dari data (sweep). Skor ambang: BM25 = skor teratas; semantic/hybrid = cosine tertinggi.
- Akurasi seimbang = rata-rata (porsi pertanyaan KB yang lolos, porsi pertanyaan luar KB yang ditolak).
Waktu = waktu retrieval saja per pertanyaan (termasuk embedding pertanyaan), CPU.

Hasil: data/eval/hasil-retrieval.md (ringkasan) dan data/eval/detail-retrieval.csv (per pertanyaan).
"""
import csv
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.pipeline import load_chunks_from_db  # noqa: E402
from app.rag.embedding import MODELS, model_available  # noqa: E402
from app.rag.retrievers import build_retriever  # noqa: E402
from app.db.models import ChunkEmbedding  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from sqlalchemy import select  # noqa: E402

BACKEND = Path(__file__).resolve().parent.parent
TESTSET = BACKEND / "data" / "eval" / "testset_retrieval.csv"
OUT_MD = BACKEND / "data" / "eval" / "hasil-retrieval.md"
OUT_CSV = BACKEND / "data" / "eval" / "detail-retrieval.csv"
K = 5
JENIS = ("persis", "parafrase", "santai")


def cpu_name() -> str:
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=15).stdout.strip()
        return out or platform.processor()
    except Exception:  # noqa: BLE001
        return platform.processor()


def relevant(chunk, row) -> bool:
    docs, keys = row["dokumen"].split("|"), row["kunci"].split("|")
    return chunk.source.rsplit("/", 1)[-1] in docs and any(k in chunk.section or k in chunk.text for k in keys)


def best_threshold(pos: list[float], neg: list[float]) -> tuple[float, float, int, int]:
    """Ambang t: lolos bila skor >= t. Pilih t dengan akurasi seimbang tertinggi."""
    best = (0.0, 0.0, 0, 0)
    for t in sorted(set(pos + neg)):
        tpr = sum(s >= t for s in pos) / len(pos)
        tnr = sum(s < t for s in neg) / len(neg)
        bal = (tpr + tnr) / 2
        if bal > best[1]:
            best = (t, bal, sum(s < t for s in pos), sum(s >= t for s in neg))
    return best


def run(method: str, model: str | None, chunks, rows):
    r = build_retriever(method, chunks, model)
    r.search("halo", 1)  # pemanasan: muat model, tidak dihitung
    detail = []
    for row in rows:
        t0 = time.perf_counter()
        hits = r.search(row["pertanyaan"], K)
        ms = (time.perf_counter() - t0) * 1000
        rank = next((i for i, h in enumerate(hits, 1) if relevant(h.chunk, row)), None) if row["jenis"] != "luar_kb" else None
        top3 = hits[:3]
        gate = (top3[0].score if top3 else 0.0) if method == "bm25" else max((h.similarity or 0.0 for h in top3), default=0.0)
        detail.append({"id": row["id"], "jenis": row["jenis"], "rank": rank, "gate": gate, "ms": ms,
                       "top1": f"{hits[0].chunk.source.rsplit('/', 1)[-1]} :: {hits[0].chunk.section}" if hits else "-"})
    ans = [d for d in detail if d["jenis"] != "luar_kb"]
    neg = [d["gate"] for d in detail if d["jenis"] == "luar_kb"]
    t, bal, fn, fp = best_threshold([d["gate"] for d in ans], neg)
    summary = {
        "hit1": sum(d["rank"] == 1 for d in ans) / len(ans),
        "hit3": sum(d["rank"] is not None and d["rank"] <= 3 for d in ans) / len(ans),
        "mrr": sum(1 / d["rank"] for d in ans if d["rank"]) / len(ans),
        "per_jenis": {j: sum(d["rank"] is not None and d["rank"] <= 3 for d in ans if d["jenis"] == j)
                      / max(1, sum(d["jenis"] == j for d in ans)) for j in JENIS},
        "thr": t, "bal": bal, "fn": fn, "fp": fp, "n_ans": len(ans), "n_neg": len(neg),
        "ms_mean": statistics.mean(d["ms"] for d in detail), "ms_median": statistics.median(d["ms"] for d in detail),
    }
    return summary, detail


def main() -> None:
    rows = list(csv.DictReader(open(TESTSET, encoding="utf-8")))
    chunks = load_chunks_from_db()
    with SessionLocal() as db:
        embedded = set(db.scalars(select(ChunkEmbedding.model_name).distinct()))
    skipped = [name for name in MODELS if not model_available(name) or name not in embedded]
    configs = [("bm25", None)] + [(m, name) for name in MODELS if name not in skipped for m in ("semantic", "hybrid")]
    results, details = [], []
    for method, model in configs:
        label = method if model is None else f"{method} · {model.split('/')[-1]}"
        print(f"menjalankan {label} ...", flush=True)
        s, d = run(method, model, chunks, rows)
        results.append((label, s))
        details += [{"konfigurasi": label, **x} for x in d]

    n_ans, n_neg = results[0][1]["n_ans"], results[0][1]["n_neg"]
    lines = [
        "# Hasil Evaluasi Retrieval",
        "",
        f"- **Tanggal:** {datetime.now():%d %B %Y %H:%M}",
        f"- **Perangkat:** {cpu_name()}, CPU saja, Python {platform.python_version()}",
        f"- **KB:** {len(chunks)} chunk (data_source={settings.kb_data_source}; Markdown per heading, CSV {settings.chunk_rows} baris/overlap {settings.chunk_overlap})",
        f"- **Set uji:** {len(rows)} pertanyaan ({n_ans} ada di KB, {n_neg} di luar KB), `data/eval/testset_retrieval.csv`. "
        "**Status: draf, belum ditinjau.**",
        f"- **Vektor:** PostgreSQL + pgvector, pencarian exact (tanpa indeks). Hybrid = RRF k={settings.rrf_k}.",
        "",
        "## Ringkasan",
        "",
        "| Konfigurasi | Hit@1 | Hit@3 | MRR@5 | Hit@3 persis | Hit@3 parafrase | Hit@3 santai | Ambang terbaik | Akurasi seimbang luar KB | Waktu rata-rata (ms) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, s in results:
        pj = s["per_jenis"]
        thr = f"{s['thr']:.2f}" if label == "bm25" else f"cos {s['thr']:.3f}"
        lines.append(
            f"| {label} | {s['hit1']:.0%} | {s['hit3']:.0%} | {s['mrr']:.2f} | {pj['persis']:.0%} | {pj['parafrase']:.0%} | "
            f"{pj['santai']:.0%} | {thr} | {s['bal']:.0%} ({s['fn']} KB ditolak, {s['fp']} luar KB lolos) | {s['ms_mean']:.0f} |")
    lines += [
        "",
        "Catatan:",
        "- Ambang dicari dari set uji yang sama, jadi angkanya optimistis. Gunakan sebagai titik awal, bukan hasil akhir.",
        "- Waktu semantic/hybrid termasuk membuat embedding pertanyaan di CPU.",
        "- Detail per pertanyaan: `data/eval/detail-retrieval.csv`.",
    ] + ([f"- **Belum dievaluasi** (model belum terunduh atau belum di-embed): {', '.join(skipped)}."] if skipped else [])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["konfigurasi", "id", "jenis", "rank", "gate", "ms", "top1"])
        w.writeheader()
        for d in details:
            w.writerow({**d, "gate": f"{d['gate']:.4f}", "ms": f"{d['ms']:.1f}"})
    print("\n".join(lines[10:12 + len(results)]))
    print(f"\nditulis: {OUT_MD.relative_to(BACKEND)} dan {OUT_CSV.relative_to(BACKEND)}")


if __name__ == "__main__":
    main()
