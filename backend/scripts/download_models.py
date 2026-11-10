"""Unduh model embedding ke Proyek PA/.cache/models/<org>__<nama>/ dengan resume dan verifikasi.

Pengunduh bawaan Hugging Face sempat menggantung di jaringan ini, jadi skrip ini mengunduh sendiri:
- melanjutkan unduhan yang terputus (HTTP Range),
- batas waktu baca 60 detik: koneksi yang menggantung diputus lalu diulang,
- bobot model (file LFS) diverifikasi dengan SHA-256 dari Hugging Face,
- bila file yang sama sudah ada di cache Hugging Face (termasuk .incomplete), dipakai ulang.

    .venv\\Scripts\\python scripts\\download_models.py            # semua model di app/rag/embedding.py
    .venv\\Scripts\\python scripts\\download_models.py --model LazarusNLP/all-indo-e5-small-v4
"""
import argparse
import fnmatch
import hashlib
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from app.config import settings  # noqa: E402
from app.rag.embedding import MODELS, local_model_dir  # noqa: E402

PATTERNS = ["*.json", "*.txt", "*.model", "tokenizer*", "sentencepiece*", "1_Pooling/*", "2_Normalize/*"]
TIMEOUT = httpx.Timeout(connect=20, read=60, write=60, pool=20)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def list_files(client: httpx.Client, repo: str) -> list[dict]:
    r = client.get(f"https://huggingface.co/api/models/{repo}/tree/main", params={"recursive": "true"})
    r.raise_for_status()
    items = [i for i in r.json() if i["type"] == "file"]
    names = {i["path"] for i in items}
    weight = "model.safetensors" if "model.safetensors" in names else "pytorch_model.bin"
    return [i for i in items if i["path"] == weight or any(fnmatch.fnmatch(i["path"], p) for p in PATTERNS)]


def seed_from_hf_cache(repo: str, sha: str, dest: Path) -> None:
    """Pakai ulang blob yang sudah ada di cache Hugging Face (lengkap atau .incomplete)."""
    blobs = Path(settings.hf_home) / "hub" / f"models--{repo.replace('/', '--')}" / "blobs"
    for cand in (blobs / sha, blobs / f"{sha}.incomplete"):
        if cand.exists() and cand.stat().st_size > (dest.stat().st_size if dest.exists() else 0):
            shutil.copyfile(cand, dest)
            print(f"   memakai ulang {cand.stat().st_size / 1e6:.0f} MB dari cache Hugging Face", flush=True)
            return


def fetch(client: httpx.Client, repo: str, item: dict, dest: Path, attempts: int = 10) -> None:
    """Unduh ke <nama>.part, verifikasi, lalu ganti nama. Jadi file final selalu utuh.
    attempts = jumlah kegagalan berturut-turut TANPA kemajuan sebelum menyerah."""
    size, lfs = item["size"], item.get("lfs")
    sha = lfs["oid"] if lfs else None
    if dest.exists() and dest.stat().st_size == size and (not sha or sha256(dest) == sha):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if dest.exists():  # sisa unduhan lama yang belum terverifikasi
        dest.replace(part)
    if sha:
        seed_from_hf_cache(repo, sha, part)
    url = f"https://huggingface.co/{repo}/resolve/main/{item['path']}"
    fails, t_start = 0, time.perf_counter()
    start_bytes = part.stat().st_size if part.exists() else 0
    while fails < attempts:
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink()
            have = 0
        try:
            if have < size:
                headers = {"Range": f"bytes={have}-"} if have else {}
                with client.stream("GET", url, headers=headers) as r:
                    if r.status_code not in (200, 206):
                        raise RuntimeError(f"HTTP {r.status_code}")
                    with open(part, "ab" if r.status_code == 206 else "wb") as f:
                        for block in r.iter_bytes(1 << 20):
                            f.write(block)
            if part.stat().st_size != size:
                raise RuntimeError(f"ukuran {part.stat().st_size} != {size}")
            if sha and sha256(part) != sha:
                part.unlink()
                raise RuntimeError("SHA-256 tidak cocok, file dihapus")
            part.replace(dest)
            return
        except Exception as e:  # noqa: BLE001
            got = part.stat().st_size if part.exists() else 0
            fails = 0 if got > have else fails + 1  # ada kemajuan -> hitungan gagal direset
            wait = 2 if fails == 0 else min(30, 3 * fails)
            rate = (got - start_bytes) / max(1.0, time.perf_counter() - t_start) / 1e3
            print(f"   {item['path']}: {type(e).__name__} {str(e)[:60]} ({got / 1e6:.0f}/{size / 1e6:.0f} MB, ~{rate:.0f} KB/dtk), "
                  f"lanjut dalam {wait} dtk", flush=True)
            time.sleep(wait)
    raise SystemExit(f"Gagal mengunduh {repo}/{item['path']} (koneksi tidak maju). Jalankan lagi untuk melanjutkan.")


def download(repo: str) -> None:
    t0 = time.perf_counter()
    root = local_model_dir(repo)
    with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
        items = list_files(client, repo)
        for item in items:
            fetch(client, repo, item, root / item["path"])
    total = sum(i["size"] for i in items) / 1e6
    print(f"OK  {repo}: {len(items)} file, {total:.0f} MB, terverifikasi, {time.perf_counter() - t0:.0f} dtk -> {root}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    args = ap.parse_args()
    for repo in ([args.model] if args.model else MODELS):
        download(repo)


if __name__ == "__main__":
    main()
