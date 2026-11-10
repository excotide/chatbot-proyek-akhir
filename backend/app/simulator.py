"""Simulator chat berbasis web. Memakai pipeline yang sama dengan jalur WhatsApp, tanpa WhatsApp.

Percakapan disimpan ke database: satu percakapan = satu lead (channel=simulator, data_source=dummy).
Percakapan baru dibuat dengan id sesi dari browser; percakapan lama dibuka dan dilanjutkan lewat lead_id.
Riwayat untuk LLM dibaca dari tabel messages, jadi tetap ada walau server dinyalakan ulang.
"""
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.conversation import (bot_reply_detail, find_lead, get_or_create_lead, history_for_llm,
                              list_simulator_conversations, load_messages, save_bot_reply, save_customer_message,
                              simulator_lead, simulator_phone)
from app.db.session import SessionLocal
from app.pipeline import METHODS, VECTOR_DBS, BotReply, generate_reply, get_retriever, reload_kb as reload_chunks
from app.vectorstores import get_store, kb_collection
from app.rag import llm

router = APIRouter(prefix="/simulator", tags=["simulator"])

_HTML = Path(__file__).parent / "static" / "simulator.html"


class ChatIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)       # dipakai untuk percakapan baru
    lead_id: int | None = None                                # diisi bila melanjutkan percakapan dari daftar
    text: str = Field(min_length=1, max_length=2000)
    method: Literal["semantic", "hybrid", "bm25"] | None = None  # metode pencarian; kosong = default .env
    llm_model: str | None = Field(default=None, max_length=120)  # mis. 'groq:openai/gpt-oss-20b'; kosong = default
    vector_db: Literal["pgvector", "chroma", "surrealdb"] | None = None  # database vektor; kosong = default .env


class ResetIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)


@router.get("")
def page() -> FileResponse:
    return FileResponse(_HTML, media_type="text/html")


@router.get("/info")
async def info() -> dict:
    r = get_retriever()
    docs = sorted({c.doc for c in r.chunks})
    return {
        "retriever": r.name,
        "methods": list(METHODS),
        "chunks": len(r.chunks),
        "docs": docs,
        "data_source": settings.kb_data_source,
        "chunk_rows": settings.chunk_rows,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "min_score": settings.min_score,
        "min_similarity": settings.min_similarity,
        "embedding_model": settings.embedding_model,
        "llm": await llm.status(),
        "llm_default": llm.default_model(),
        "llm_models": await llm.available_models(),
        "vector_db_default": settings.vector_db,
        "vector_dbs": vector_db_status(),
    }


def vector_db_status() -> list[dict]:
    """Status tiap database vektor: server hidup dan jumlah vektor KB untuk model embedding aktif."""
    col = kb_collection(settings.embedding_model)
    out = []
    for name in VECTOR_DBS:
        try:
            n = get_store(name).count(col)
            state = "ready" if n > 0 else "empty"
        except Exception:  # noqa: BLE001
            n, state = 0, "offline"
        out.append({"id": name, "state": state, "vectors": n})
    return out


@router.post("/chat")
async def chat(body: ChatIn) -> dict:
    if body.llm_model:
        allowed = {m["id"] for m in await llm.available_models()}
        if body.llm_model not in allowed:
            raise HTTPException(400, f"Model LLM tidak tersedia: {body.llm_model}")
    with SessionLocal() as db:
        # 1) pesan pelanggan disimpan dulu (transaksi sendiri), riwayat diambil SEBELUM pesan ini
        with db.begin():
            if body.lead_id is not None:
                lead = simulator_lead(db, body.lead_id)
                if lead is None:
                    raise HTTPException(404, "Percakapan tidak ditemukan")
            else:
                lead = get_or_create_lead(db, simulator_phone(body.session_id), "simulator", "dummy",
                                          "Pelanggan simulator")
            history = history_for_llm(db, lead.id)
            customer_msg = save_customer_message(db, lead, body.text)
        if not lead.bot_active:
            reply = BotReply("(Bot nonaktif: percakapan ini sudah diambil alih sales.)", "taken_over")
            return asdict(reply) | {"lead_id": lead.id}
        # 2) jawab, lalu simpan balasan bot + metadata + chunk sumber
        reply = await generate_reply(body.text, history, body.method, body.llm_model, body.vector_db)
        with db.begin():
            bot_msg = save_bot_reply(db, lead, customer_msg, reply)
        emb = settings.embedding_model if reply.retriever in ("semantic", "hybrid") else None
        return asdict(reply) | {"lead_id": lead.id, "message_id": bot_msg.id, "embedding_model": emb}


@router.get("/conversations")
def conversations() -> dict:
    """Daftar percakapan simulator yang tersimpan, terbaru dulu."""
    with SessionLocal() as db:
        return {"conversations": list_simulator_conversations(db)}


@router.get("/history")
def history(session_id: str | None = None, lead_id: int | None = None) -> dict:
    """Riwayat satu percakapan, berdasarkan lead_id (dari daftar) atau id sesi browser."""
    with SessionLocal() as db:
        if lead_id is not None:
            lead = simulator_lead(db, lead_id)
        elif session_id:
            lead = find_lead(db, simulator_phone(session_id))
        else:
            raise HTTPException(400, "Isi session_id atau lead_id")
        if lead is None:
            return {"lead_id": None, "messages": []}
        msgs = load_messages(db, lead.id)
        out = []
        for i, m in enumerate(msgs):
            item = {"id": m.id, "sender": m.sender_type, "text": m.text, "time": m.created_at.isoformat()}
            if m.sender_type == "customer":
                # pesan pelanggan yang tidak diikuti balasan (mis. server dinyalakan ulang saat memproses)
                item["answered"] = i + 1 < len(msgs) and msgs[i + 1].sender_type != "customer"
            out.append(item)
        return {"lead_id": lead.id, "messages": out}


@router.get("/messages/{message_id}")
def message_detail(message_id: int) -> dict:
    """Status sistem yang dipakai bot saat menyusun satu balasan (dibaca dari bot_responses + retrieval_results)."""
    with SessionLocal() as db:
        d = bot_reply_detail(db, message_id)
        if d is None:
            raise HTTPException(404, "Balasan bot tidak ditemukan")
        return d


@router.post("/reload-kb")
def reload_kb() -> dict:
    """Muat ulang chunk dari database setelah menjalankan scripts/ingest_kb.py, tanpa restart server."""
    return {"chunks": reload_chunks()}


@router.post("/reset")
def reset(body: ResetIn) -> dict:
    """'Percakapan baru' dibuat di browser dengan id sesi baru (= lead baru).
    Percakapan lama tidak dihapus: tetap tersimpan di database sebagai data."""
    return {"status": "ok"}
