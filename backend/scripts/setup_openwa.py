"""Siapkan sesi OpenWA: buat sesi, tampilkan QR, daftarkan webhook ke backend.

Jalankan dari folder backend:  python scripts/setup_openwa.py
"""
import base64
import os
import re
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from app.config import settings  # noqa: E402

API = settings.openwa_base_url.rstrip("/")
H = {"X-API-Key": settings.openwa_api_key}
client = httpx.Client(headers=H, timeout=20)


def unwrap(j):
    return j.get("data", j) if isinstance(j, dict) else j


def save_session_id(session_id: str) -> None:
    env = ROOT / ".env"
    text = env.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^OPENWA_SESSION_ID=.*$", f"OPENWA_SESSION_ID={session_id}", text)
    env.write_text(text, encoding="utf-8")


def find_or_create_session() -> dict:
    r = client.get(f"{API}/sessions")
    if r.status_code == 200:
        items = unwrap(r.json())
        if isinstance(items, dict):
            items = items.get("items") or items.get("sessions") or []
        for s in items:
            if s.get("name") == settings.openwa_session_name:
                print(f"Sesi '{s['name']}' sudah ada (id={s['id']}, status={s.get('status')})")
                return s
    r = client.post(f"{API}/sessions", json={"name": settings.openwa_session_name})
    r.raise_for_status()
    s = unwrap(r.json())
    print(f"Sesi baru dibuat: id={s['id']}")
    return s


def wait_for_qr_or_ready(sid: str) -> None:
    client.post(f"{API}/sessions/{sid}/start")
    shown = None
    for _ in range(90):  # ~3 menit
        st = unwrap(client.get(f"{API}/sessions/{sid}").json())
        status = str(st.get("status", "")).lower()
        if status in ("ready", "connected", "authenticated", "working"):
            print(f"\nWhatsApp TERHUBUNG (status={status})")
            return
        r = client.get(f"{API}/sessions/{sid}/qr")
        if r.status_code == 200:
            qr = unwrap(r.json()).get("qrCode") or ""
            if qr.startswith("data:image") and qr != shown:
                png = base64.b64decode(qr.split(",", 1)[1])
                (ROOT / "qr.png").write_bytes(png)
                if shown is None:
                    print("\nQR disimpan di backend/qr.png dan dibuka.")
                    print("Scan dari HP: WhatsApp > Perangkat tertaut > Tautkan perangkat")
                    if os.name == "nt":
                        os.startfile(ROOT / "qr.png")
                else:
                    print("(QR diperbarui — buka ulang qr.png kalau yang lama kedaluwarsa)")
                shown = qr
        time.sleep(2)
    print("\nBelum terhubung setelah 3 menit. Jalankan ulang skrip ini untuk QR baru.")


def register_webhook(sid: str) -> None:
    url = settings.backend_webhook_url
    r = client.get(f"{API}/sessions/{sid}/webhooks")
    if r.status_code == 200:
        items = unwrap(r.json())
        if isinstance(items, dict):
            items = items.get("items") or items.get("webhooks") or []
        if any(w.get("url") == url for w in items):
            print(f"Webhook ke {url} sudah terdaftar")
            return
    body = {"url": url, "events": ["message.received", "session.status"],
            "secret": settings.openwa_webhook_secret, "retryCount": 3}
    r = client.post(f"{API}/sessions/{sid}/webhooks", json=body)
    if r.status_code >= 400:
        print(f"Gagal daftar webhook ({r.status_code}): {r.text[:500]}")
        return
    print(f"Webhook terdaftar -> {url}")


if __name__ == "__main__":
    try:
        client.get(f"{API}/health").raise_for_status()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"OpenWA belum jalan di {API} ({e}). Jalankan dulu: docker compose up -d di folder wa-gateway")
    s = find_or_create_session()
    save_session_id(s["id"])
    register_webhook(s["id"])
    wait_for_qr_or_ready(s["id"])
