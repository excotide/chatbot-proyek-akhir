"""Komparasi database vektor: PostgreSQL + pgvector vs ChromaDB vs SurrealDB.

Prinsip: hanya database yang berubah. Vektor, pertanyaan, k, dan perangkat sama untuk ketiganya.

Bagian A - KB asli: vektor chunk KB (model EMBEDDING_MODEL) dimasukkan ke koleksi 'realkb' di ketiga DB,
           lalu 40 pertanyaan set uji dicari. Hit@k seharusnya sama (kualitas ditentukan embedding).
Bagian B - data sintetis: vektor 384 dimensi berbentuk klaster (seed tetap, bisa diulang) dengan ukuran
           bertingkat. Diukur: waktu insert, waktu bangun indeks HNSW, latensi cari exact dan HNSW
           (median, p95), recall@10 HNSW terhadap jawaban exact (numpy), dan RAM container.

    .venv\\Scripts\\python scripts\\bench_vectordb.py                       # ukuran 1000 dan 10000
    .venv\\Scripts\\python scripts\\bench_vectordb.py --sizes 1000 10000 100000

Hasil: data/eval/hasil-vectordb.md dan data/eval/detail-vectordb.csv
Catatan: latensi diukur dari klien Python (termasuk HTTP/koneksi dan serialisasi), satu query per waktu.
"""
import argparse
import csv
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app.config import settings  # noqa: E402
from app.pipeline import load_chunks_from_db  # noqa: E402
from app.rag.embedding import embed_query  # noqa: E402
from app.vectorstores import get_store  # noqa: E402
from scripts.sync_vectors import kb_vectors  # noqa: E402

BACKEND = Path(__file__).resolve().parent.parent
OUT_MD = BACKEND / "data" / "eval" / "hasil-vectordb.md"
OUT_CSV = BACKEND / "data" / "eval" / "detail-vectordb.csv"
TESTSET = BACKEND / "data" / "eval" / "testset_retrieval.csv"
STORES = ("pgvector", "chroma", "surrealdb")
CONTAINERS = {"pgvector": "pa-postgres", "chroma": "pa-chroma", "surrealdb": "pa-surrealdb"}
DIM, K, N_QUERIES, SEED = 384, 10, 100, 42
# SurrealDB 3.2.6: indeks HNSW macet (server berhenti merespons) pada 10.000 vektor 384 dimensi, baik indeks
# dibangun sesudah maupun sebelum insert; 2.000 dan 5.000 berhasil (uji 9 Nov 2026). Di atas batas ini SurrealDB
# hanya diukur dengan pencarian exact agar benchmark tidak menggantung.
SURREAL_HNSW_MAX = 5000


def mem_mb(container: str) -> float | None:
    try:
        out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
                             capture_output=True, text=True, timeout=30).stdout.strip().split("/")[0].strip()
        num = float("".join(ch for ch in out if ch.isdigit() or ch == "."))
        return num * {"KiB": 1 / 1024, "MiB": 1, "GiB": 1024}.get(out[-3:], 1)
    except Exception:  # noqa: BLE001
        return None


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def timed_searches(store, col, queries, k, exact, repeat=1):
    lat, results = [], []
    store.search(col, queries[0], k, exact)  # pemanasan
    for _ in range(repeat):
        results = []
        for q in queries:
            t0 = time.perf_counter()
            results.append(store.search(col, q, k, exact))
            lat.append((time.perf_counter() - t0) * 1000)
    return lat, results


# ------------------------------------------------------------------ Bagian A
def part_a(rows_out: list[dict]) -> list[dict]:
    chunks = {c.db_id: c for c in load_chunks_from_db()}
    ids, vecs = kb_vectors(settings.embedding_model)
    tests = [r for r in csv.DictReader(open(TESTSET, encoding="utf-8")) if r["jenis"] != "luar_kb"]
    queries = [embed_query(r["pertanyaan"], settings.embedding_model) for r in tests]

    def relevant(cid, r):
        c = chunks.get(cid)
        return c is not None and c.source.rsplit("/", 1)[-1] in r["dokumen"].split("|") and \
            any(k in c.section or k in c.text for k in r["kunci"].split("|"))

    summary = []
    for name in STORES:
        store = get_store(name)
        store.drop("realkb")
        store.upsert("realkb", ids, vecs)
        store.build_index("realkb", vecs.shape[1])
        exact = name != "chroma"
        lat, res = timed_searches(store, "realkb", queries, 5, exact, repeat=3)
        ranks = [next((i for i, (cid, _) in enumerate(hits, 1) if relevant(cid, r)), None) for hits, r in zip(res, tests)]
        n = len(tests)
        s = {"db": name, "n_vec": len(ids), "hit1": sum(x == 1 for x in ranks) / n,
             "hit3": sum(x is not None and x <= 3 for x in ranks) / n,
             "mrr": sum(1 / x for x in ranks if x) / n, "p50": statistics.median(lat), "p95": pct(lat, 95),
             "top_ids": [[cid for cid, _ in h] for h in res]}
        summary.append(s)
        rows_out.append({"bagian": "A", "db": name, "ukuran": len(ids), "metrik": "hit1/hit3/mrr",
                         "nilai": f"{s['hit1']:.3f}/{s['hit3']:.3f}/{s['mrr']:.3f}"})
        print(f"A {name:10} Hit@1 {s['hit1']:.0%} Hit@3 {s['hit3']:.0%} MRR {s['mrr']:.2f} | p50 {s['p50']:.1f} ms p95 {s['p95']:.1f} ms", flush=True)
        store.drop("realkb")
    same = all(s["top_ids"] == summary[0]["top_ids"] for s in summary)
    for s in summary:
        s["same_as_pg"] = s["top_ids"] == summary[0]["top_ids"]
    print("A urutan top-5 identik di ketiga DB:", same, flush=True)
    return summary


# ------------------------------------------------------------------ Bagian B
def synthetic(n: int, rng: np.random.Generator, centers: np.ndarray) -> np.ndarray:
    lab = rng.integers(0, len(centers), n)
    v = centers[lab] + rng.normal(0, 0.6 / np.sqrt(DIM), (n, DIM)) * 3
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def part_b(sizes: list[int], rows_out: list[dict]) -> list[dict]:
    rng = np.random.default_rng(SEED)
    centers = rng.normal(size=(64, DIM))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    results = []
    for n in sizes:
        data = synthetic(n, rng, centers)
        queries = synthetic(N_QUERIES, rng, centers)
        truth = np.argsort(-(queries @ data.T), axis=1)[:, :K]  # jawaban exact
        ids = list(range(1, n + 1))
        col = f"bench_{n}"
        for name in STORES:
            store = get_store(name)
            store.drop(col)
            mem0 = mem_mb(CONTAINERS[name])
            t0 = time.perf_counter(); store.upsert(col, ids, data); t_ins = time.perf_counter() - t0
            hnsw_ok = not (name == "surrealdb" and n > SURREAL_HNSW_MAX)
            t_idx = None
            if hnsw_ok:
                t0 = time.perf_counter(); store.build_index(col, DIM); t_idx = time.perf_counter() - t0
            r = {"db": name, "n": n, "insert_s": t_ins, "index_s": t_idx, "hnsw_ok": hnsw_ok}
            if name != "chroma":
                lat, res = timed_searches(store, col, queries, K, exact=True)
                r |= {"exact_p50": statistics.median(lat), "exact_p95": pct(lat, 95),
                      "exact_recall": float(np.mean([len({c - 1 for c, _ in h} & set(t)) / K for h, t in zip(res, truth)]))}
            if hnsw_ok:
                lat, res = timed_searches(store, col, queries, K, exact=False)
                r |= {"hnsw_p50": statistics.median(lat), "hnsw_p95": pct(lat, 95),
                      "hnsw_recall": float(np.mean([len({c - 1 for c, _ in h} & set(t)) / K for h, t in zip(res, truth)]))}
            else:
                r |= {"hnsw_p50": None, "hnsw_p95": None, "hnsw_recall": None}
            mem1 = mem_mb(CONTAINERS[name])
            r["mem_mb"] = mem1
            r["mem_delta_mb"] = (mem1 - mem0) if mem0 is not None and mem1 is not None else None
            results.append(r)
            for key, val in r.items():
                if key not in ("db", "n", "hnsw_ok"):
                    rows_out.append({"bagian": "B", "db": name, "ukuran": n, "metrik": key,
                                     "nilai": "" if val is None else f"{val:.4f}"})
            nf = lambda v, d=1: "  -  " if v is None else f"{v:.{d}f}"  # noqa: E731
            print(f"B n={n:<7} {name:10} insert {t_ins:6.1f}s indeks {nf(t_idx)}s | exact p50 {nf(r.get('exact_p50'))}ms | "
                  f"HNSW p50 {nf(r['hnsw_p50'])}ms recall {nf(r['hnsw_recall'], 3)} | RAM {mem1 or 0:.0f} MB", flush=True)
            store.drop(col)
    return results


def cpu_name() -> str:
    try:
        return subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                              capture_output=True, text=True, timeout=15).stdout.strip() or platform.processor()
    except Exception:  # noqa: BLE001
        return platform.processor()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000])
    args = ap.parse_args()
    rows: list[dict] = []
    a = part_a(rows)
    b = part_b(args.sizes, rows)

    f = lambda v, d=1: "-" if v is None else f"{v:.{d}f}"  # noqa: E731
    md = [
        "# Hasil Komparasi Database Vektor", "",
        f"- **Tanggal:** {datetime.now():%d %B %Y %H:%M}",
        f"- **Perangkat:** {cpu_name()}, semua database di Docker Desktop (WSL2) pada laptop yang sama",
        "- **Versi:** PostgreSQL 16 + pgvector 0.8.0 · ChromaDB 1.5.9 · SurrealDB 3.2.6",
        f"- **Embedding (bagian A):** {settings.embedding_model}, cosine. Bagian B: vektor sintetis {DIM} dimensi, "
        f"64 klaster, seed {SEED}, {N_QUERIES} query, k={K}.",
        "- **Latensi** diukur dari klien Python per query (termasuk koneksi/HTTP dan serialisasi).", "",
        "## Bagian A: KB asli (kualitas retrieval)", "",
        "| Database | Vektor | Hit@1 | Hit@3 | MRR@5 | Urutan top-5 sama dengan pgvector | Latensi p50 (ms) | p95 (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in a:
        md.append(f"| {s['db']} | {s['n_vec']} | {s['hit1']:.0%} | {s['hit3']:.0%} | {s['mrr']:.2f} | "
                  f"{'ya' if s['same_as_pg'] else 'tidak'} | {s['p50']:.1f} | {s['p95']:.1f} |")
    md += ["", "Pencarian exact untuk pgvector dan SurrealDB; ChromaDB selalu memakai HNSW.", "",
           "## Bagian B: data sintetis (performa)", "",
           "| Ukuran | Database | Insert (dtk) | Bangun indeks (dtk) | Exact p50 / p95 (ms) | HNSW p50 / p95 (ms) | Recall@10 HNSW | RAM container (MB) |",
           "|---|---|---|---|---|---|---|---|"]
    for r in b:
        ex = f"{f(r.get('exact_p50'))} / {f(r.get('exact_p95'))}" if "exact_p50" in r else "tidak tersedia"
        if r["hnsw_ok"]:
            idx = "otomatis saat insert" if r["db"] == "chroma" else f"{r['index_s']:.1f}"
            hn, rc = f"{r['hnsw_p50']:.1f} / {r['hnsw_p95']:.1f}", f"{r['hnsw_recall']:.3f}"
        else:
            idx, hn, rc = "macet (lihat catatan)", "-", "-"
        md.append(f"| {r['n']:,} | {r['db']} | {r['insert_s']:.1f} | {idx} | {ex} | {hn} | {rc} | {f(r['mem_mb'], 0)} |")
    md += ["", "Catatan:",
           "- ChromaDB tidak punya mode pencarian exact; indeks HNSW dibangun otomatis saat insert, jadi waktu "
           "indeksnya sudah termasuk di waktu insert.",
           "- RAM adalah pemakaian total container saat itu (termasuk data lain di database tersebut). Docker Desktop "
           "di laptop ini dibatasi ±1,86 GB untuk semua container.",
           f"- SurrealDB 3.2.6: indeks HNSW berhasil untuk 2.000 dan 5.000 vektor, tetapi pada 10.000 vektor server "
           f"berhenti merespons (tanpa error, tanpa kehabisan memori) sampai container dinyalakan ulang, baik indeks "
           f"dibangun sesudah maupun sebelum insert. Di atas {SURREAL_HNSW_MAX:,} vektor SurrealDB hanya diukur dengan "
           f"pencarian exact.",
           "- Recall@10 = porsi 10 tetangga terdekat sebenarnya (dihitung exact dengan numpy) yang ditemukan.",
           "- Detail mentah: `data/eval/detail-vectordb.csv`."]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["bagian", "db", "ukuran", "metrik", "nilai"])
        w.writeheader(); w.writerows(rows)
    print(f"\nditulis: {OUT_MD.relative_to(BACKEND)} dan {OUT_CSV.relative_to(BACKEND)}")


if __name__ == "__main__":
    main()
