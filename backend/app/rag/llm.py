"""Klien LLM: Ollama (lokal) atau Groq (cloud).

Model ditulis sebagai '<penyedia>:<nama>', mis. 'groq:openai/gpt-oss-120b' atau 'ollama:qwen2.5:3b'.
Default diambil dari LLM_PROVIDER + GROQ_MODEL / OLLAMA_MODEL di .env; simulator bisa memilih model lain per pesan.
Keduanya menerima daftar pesan format chat yang sama ({"role", "content"}), jadi pipeline tidak berubah.
Catatan privasi: dengan Groq, isi chat pelanggan dan potongan knowledge base dikirim ke server Groq.
"""
import asyncio
import re
import time

import httpx

from app.config import settings

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
PROVIDERS = ("groq", "ollama")


class LLMUnavailable(Exception):
    """LLM tidak bisa dipakai: Ollama mati / model belum diunduh, atau Groq gagal (key, kuota, jaringan)."""


# Setelah gagal, model yang sama tidak dicoba lagi selama jeda ini supaya tiap pesan tidak ikut lambat
# (mis. Windows butuh ~2 detik untuk menolak koneksi ke Ollama yang mati). Dicatat PER MODEL, jadi kuota
# satu model Groq yang habis tidak ikut memblokir model lain.
_RETRY_AFTER = 20.0
_down: dict[str, tuple[float, str]] = {}


def default_model() -> str:
    if settings.llm_provider == "groq":
        return f"groq:{settings.groq_model}"
    return f"ollama:{settings.ollama_model}"


def parse_model(spec: str) -> tuple[str, str]:
    provider, _, name = spec.partition(":")
    if provider not in PROVIDERS or not name:
        raise ValueError(f"Model LLM tidak dikenal: {spec!r} (format: groq:<nama> atau ollama:<nama>)")
    return provider, name


def model_label(model: str | None = None) -> str:
    """Nama model untuk log, UI, dan kolom bot_responses.llm_model (mis. 'groq:openai/gpt-oss-120b')."""
    return model or default_model()


async def chat(messages: list[dict], model: str | None = None) -> str:
    spec = model_label(model)
    until, reason = _down.get(spec, (0.0, ""))
    if time.monotonic() < until:
        raise LLMUnavailable(reason)
    provider, name = parse_model(spec)
    try:
        if provider == "groq":
            return await _chat_groq(messages, name)
        return await _chat_ollama(messages, name)
    except LLMUnavailable as e:
        _down[spec] = (time.monotonic() + _RETRY_AFTER, str(e))
        raise


async def _chat_ollama(messages: list[dict], name: str) -> str:
    payload = {
        "model": name,
        "messages": messages,
        "stream": False,
        "keep_alive": "30m",  # model tetap di memori 30 menit, pesan berikutnya tidak perlu memuat ulang
        # temperature rendah = jawaban konsisten; num_predict membatasi panjang jawaban (dan waktu di CPU)
        "options": {"temperature": 0.1, "num_ctx": 4096, "num_predict": 300},
    }
    try:
        async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client:
            r = await client.post(f"{settings.ollama_base_url}/api/chat", json=payload)
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        raise LLMUnavailable(f"Ollama tidak terjangkau di {settings.ollama_base_url}") from e
    except httpx.TimeoutException as e:
        raise LLMUnavailable("Ollama terlalu lama menjawab (timeout)") from e
    if r.status_code == 404:
        raise LLMUnavailable(f"Model '{name}' belum diunduh (ollama pull {name})")
    if r.status_code != 200:
        raise LLMUnavailable(f"Ollama error {r.status_code}: {r.text[:200]}")
    return r.json()["message"]["content"].strip()


def _groq_limit_message(r: httpx.Response, name: str) -> str:
    msg = (r.json().get("error") or {}).get("message", "") if r.headers.get("content-type", "").startswith("application/json") else ""
    wait = re.search(r"try again in ([0-9hms.]+)", msg)
    when = f", pulih dalam ±{wait.group(1).split('.')[0]}s" if wait else ""
    if "per day" in msg:
        return f"Kuota HARIAN Groq untuk {name} habis{when}. Pilih model lain di simulator."
    return f"Kuota per menit Groq untuk {name} habis{when}, coba lagi sebentar."


async def _chat_groq(messages: list[dict], name: str) -> str:
    if not settings.groq_api_key:
        raise LLMUnavailable("GROQ_API_KEY belum diisi")
    payload: dict = {"model": name, "messages": messages, "temperature": 0.1, "max_completion_tokens": 1024}
    # Model penalaran: proses berpikir tidak ikut dikirim sebagai jawaban
    if name.startswith("openai/gpt-oss"):
        payload |= {"reasoning_effort": "low", "include_reasoning": False}
    elif name.startswith("qwen/"):
        payload["reasoning_format"] = "hidden"
    headers = {"Authorization": f"Bearer {settings.groq_api_key}"}
    try:
        async with httpx.AsyncClient(timeout=settings.groq_timeout) as client:
            r = await client.post(GROQ_URL, json=payload, headers=headers)
            # Akun gratis dibatasi token per menit. Bila kena 429, tunggu sesuai 'retry-after' (maks 20 dtk) lalu
            # coba sekali lagi, daripada langsung jatuh ke mode tanpa LLM.
            if r.status_code == 429:
                try:
                    wait = float(r.headers.get("retry-after", "5"))
                except ValueError:
                    wait = 5.0
                if wait <= 20:
                    await asyncio.sleep(wait + 0.5)
                    r = await client.post(GROQ_URL, json=payload, headers=headers)
    except httpx.TimeoutException as e:
        raise LLMUnavailable("Groq terlalu lama menjawab (timeout)") from e
    except httpx.HTTPError as e:
        raise LLMUnavailable(f"Groq tidak terjangkau: {type(e).__name__}") from e
    if r.status_code == 401:
        raise LLMUnavailable("GROQ_API_KEY ditolak (tidak valid atau sudah dicabut)")
    if r.status_code == 429:
        raise LLMUnavailable(_groq_limit_message(r, name))
    if r.status_code != 200:
        raise LLMUnavailable(f"Groq error {r.status_code}: {r.text[:200]}")
    content = r.json()["choices"][0]["message"].get("content") or ""
    return re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()


async def _ollama_models() -> list[str] | None:
    """Model yang sudah di-pull di Ollama; None bila Ollama tidak jalan."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:  # noqa: BLE001
        return None


async def available_models() -> list[dict]:
    """Pilihan model untuk simulator: model Groq dari GROQ_MODELS (bila key ada) + model Ollama yang terpasang."""
    out = []
    groq_state = "ready" if settings.groq_api_key else "no_key"
    for name in dict.fromkeys([settings.groq_model, *settings.groq_models]):
        spec = f"groq:{name}"
        until, _ = _down.get(spec, (0.0, ""))
        out.append({"id": spec, "provider": "groq", "name": name,
                    "state": "cooldown" if time.monotonic() < until else groq_state})
    ollama = await _ollama_models()
    if ollama is None:
        out.append({"id": f"ollama:{settings.ollama_model}", "provider": "ollama", "name": settings.ollama_model,
                    "state": "offline"})
    else:
        for name in ollama:
            out.append({"id": f"ollama:{name}", "provider": "ollama", "name": name, "state": "ready"})
    return out


async def status(model: str | None = None) -> dict:
    """Status satu model untuk badge di UI."""
    spec = model_label(model)
    provider, name = parse_model(spec)
    if provider == "groq":
        return {"state": "ready" if settings.groq_api_key else "no_key", "model": name, "provider": "groq"}
    names = await _ollama_models()
    if names is None:
        return {"state": "offline", "model": name, "provider": "ollama"}
    ready = any(n == name or n.split(":")[0] == name for n in names)
    return {"state": "ready" if ready else "model_missing", "model": name, "provider": "ollama"}
