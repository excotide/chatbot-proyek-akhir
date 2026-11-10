"""Backend API Orchestrator — tahap 1: koneksi WhatsApp.

Alur: webhook masuk -> cek tanda tangan -> parse -> balas 200 cepat
      -> proses di background (sekarang: balasan echo; nanti: RAG + klasifikasi).
"""
import json
import logging
from collections import deque

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

from app.config import settings
from app.messaging import IncomingMessage, get_provider
from app.messaging.cloud_api import CloudAPIProvider
from app.messaging.openwa import OpenWAProvider
from app.simulator import router as simulator_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("orchestrator")

app = FastAPI(title="PA Chatbot Orchestrator")
app.include_router(simulator_router)  # simulator chat web di /simulator


@app.on_event("startup")
async def _warmup() -> None:
    """Muat chunk dan model embedding di latar belakang agar pesan pertama tidak menunggu ~7 detik."""
    import asyncio

    from app.pipeline import get_retriever
    from app.rag.embedding import load_model

    def warm() -> None:
        try:
            get_retriever()
            if settings.retriever != "bm25":
                load_model(settings.embedding_model)
            log.info("Pemanasan selesai: retriever=%s, model=%s", settings.retriever, settings.embedding_model)
        except Exception as e:  # noqa: BLE001
            log.warning("Pemanasan gagal (server tetap jalan): %s", e)

    asyncio.get_running_loop().run_in_executor(None, warm)
provider = get_provider()

# Pencegah proses ganda saat webhook dikirim ulang. Nanti diganti cek di database.
_seen_ids: deque[str] = deque(maxlen=1000)


async def handle_message(msg: IncomingMessage) -> None:
    log.info("PESAN MASUK [%s] dari %s: %s", msg.provider, msg.sender, msg.text)
    # TODO: ganti dengan pipeline RAG -> LLM, lalu klasifikasi tahap lead
    reply = f"Halo! Pesanmu sudah diterima bot: \"{msg.text}\""
    try:
        sent_id = await provider.send_text(msg.sender, reply)
        log.info("BALASAN TERKIRIM ke %s (id=%s)", msg.sender, sent_id)
    except Exception as e:  # noqa: BLE001
        log.error("Gagal mengirim balasan ke %s: %s", msg.sender, e)


async def _process(request: Request, bg: BackgroundTasks, source) -> dict:
    raw = await request.body()
    headers = {k.lower(): v for k, v in request.headers.items()}
    if not source.verify_signature(raw, headers):
        log.warning("Tanda tangan webhook tidak valid, ditolak")
        raise HTTPException(401, "invalid signature")
    payload = json.loads(raw or b"{}")
    log.debug("payload: %s", json.dumps(payload)[:2000])
    for msg in source.parse_webhook(payload):
        if msg.message_id and msg.message_id in _seen_ids:
            continue
        _seen_ids.append(msg.message_id)
        bg.add_task(handle_message, msg)
    return {"status": "ok"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "provider": provider.name}


# ---------- OpenWA ----------
@app.post("/webhook/openwa")
async def webhook_openwa(request: Request, bg: BackgroundTasks):
    log.info("webhook OpenWA: event=%s", request.headers.get("x-openwa-event"))
    return await _process(request, bg, OpenWAProvider())


# ---------- Meta Cloud API ----------
@app.get("/webhook/meta")
def webhook_meta_verify(request: Request):
    p = request.query_params
    if p.get("hub.mode") == "subscribe" and p.get("hub.verify_token") == settings.wa_verify_token:
        return PlainTextResponse(p.get("hub.challenge", ""))
    raise HTTPException(403)


@app.post("/webhook/meta")
async def webhook_meta(request: Request, bg: BackgroundTasks):
    return await _process(request, bg, CloudAPIProvider())
