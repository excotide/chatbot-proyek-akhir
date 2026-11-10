"""Penyimpanan percakapan ke database (tabel leads, messages, bot_responses, retrieval_results).

Dipakai simulator sekarang; jalur WhatsApp nanti memakai fungsi yang sama dengan channel='whatsapp'.
Aturan dari Desain-Database.md:
- satu pelanggan (nomor WA / sesi simulator) = satu baris leads
- setiap pesan pelanggan, bot, dan sales = satu baris messages
- setiap balasan bot punya baris bot_responses (1:1) dan chunk sumbernya di retrieval_results
- lead dengan bot_active = false tidak dibalas bot (sudah diambil alih sales)
"""
import hashlib
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import BotResponse, Chunk, Document, Lead, Message, RetrievalResult
from app.pipeline import BotReply


def simulator_phone(session_id: str) -> str:
    """Id sesi browser -> nilai kolom leads.phone (maks 30 karakter), mis. 'sim-3f2a...'."""
    return "sim-" + hashlib.sha1(session_id.encode()).hexdigest()[:20]


def get_or_create_lead(db: Session, phone: str, channel: str, data_source: str, name: str | None = None) -> Lead:
    lead = db.scalar(select(Lead).where(Lead.phone == phone))
    if lead is None:
        lead = Lead(phone=phone, channel=channel, data_source=data_source, name=name)
        db.add(lead)
        db.flush()
    return lead


def find_lead(db: Session, phone: str) -> Lead | None:
    return db.scalar(select(Lead).where(Lead.phone == phone))


def simulator_lead(db: Session, lead_id: int) -> Lead | None:
    """Lead simulator berdasarkan id (lead WhatsApp asli tidak bisa dibuka dari simulator)."""
    lead = db.get(Lead, lead_id)
    return lead if lead is not None and lead.channel == "simulator" else None


def list_simulator_conversations(db: Session, limit: int = 50) -> list[dict]:
    """Daftar percakapan simulator, terbaru dulu: judul = pesan pertama pelanggan."""
    first_msg = (select(Message.text).where(Message.lead_id == Lead.id, Message.sender_type == "customer")
                 .order_by(Message.created_at, Message.id).limit(1).correlate(Lead).scalar_subquery())
    n_msg = select(func.count(Message.id)).where(Message.lead_id == Lead.id).correlate(Lead).scalar_subquery()
    rows = db.execute(
        select(Lead.id, Lead.created_at, Lead.last_message_at, first_msg.label("judul"), n_msg.label("n"))
        .where(Lead.channel == "simulator")
        .order_by(Lead.last_message_at.desc().nulls_last(), Lead.id.desc())
        .limit(limit)
    ).all()
    return [{"lead_id": r.id, "title": (r.judul or "(belum ada pesan)")[:80], "messages": r.n,
             "last_message_at": (r.last_message_at or r.created_at).isoformat()} for r in rows]


def load_messages(db: Session, lead_id: int, limit: int | None = None) -> list[Message]:
    """Pesan terakhir lead, urut dari yang terlama."""
    stmt = select(Message).where(Message.lead_id == lead_id).order_by(Message.created_at.desc(), Message.id.desc())
    if limit:
        stmt = stmt.limit(limit)
    return list(reversed(db.scalars(stmt).all()))


def history_for_llm(db: Session, lead_id: int) -> list[dict]:
    """Riwayat format chat untuk pipeline: pesan pelanggan = user, pesan bot/sales = assistant."""
    return [{"role": "user" if m.sender_type == "customer" else "assistant", "content": m.text}
            for m in load_messages(db, lead_id, settings.history_messages)]


def save_customer_message(db: Session, lead: Lead, text: str, wa_message_id: str | None = None) -> Message:
    msg = Message(lead_id=lead.id, sender_type="customer", text=text, wa_message_id=wa_message_id)
    db.add(msg)
    db.flush()
    lead.last_message_at = msg.created_at or datetime.now().astimezone()
    return msg


def save_bot_reply(db: Session, lead: Lead, reply_to: Message, reply: BotReply) -> Message:
    msg = Message(lead_id=lead.id, sender_type="bot", text=reply.text)
    db.add(msg)
    db.flush()
    lat = reply.latency_ms
    db.add(BotResponse(
        message_id=msg.id,
        reply_to_message_id=reply_to.id,
        mode=reply.mode,
        escalate=reply.escalate,
        retriever=reply.retriever or None,
        embedding_model=settings.embedding_model if reply.retriever in ("semantic", "hybrid") else None,
        vector_db=reply.vector_db or None,
        search_query=reply.search_query or None,
        llm_model=reply.llm_model or None,
        latency_retrieval_ms=round(lat["retrieval"]) if "retrieval" in lat else None,
        latency_llm_ms=round(lat["llm"]) if "llm" in lat else None,
        latency_total_ms=round(lat.get("total", 0)),
    ))
    db.flush()
    for rank, s in enumerate(reply.sources, 1):
        if s.chunk_id is not None:
            db.add(RetrievalResult(bot_response_id=msg.id, chunk_id=s.chunk_id, rank=rank, score=s.score))
    lead.last_message_at = datetime.now().astimezone()
    return msg


def bot_reply_detail(db: Session, message_id: int) -> dict | None:
    """Detail satu balasan bot (simulator): metode, model, waktu proses, pertanyaan yang dicari, dan chunk sumber."""
    msg = db.get(Message, message_id)
    if msg is None or msg.sender_type != "bot":
        return None
    lead = db.get(Lead, msg.lead_id)
    if lead is None or lead.channel != "simulator":
        return None
    br = db.get(BotResponse, message_id)
    if br is None:
        return {"message_id": message_id, "has_detail": False}
    question = db.get(Message, br.reply_to_message_id)
    rows = db.execute(
        select(RetrievalResult.rank, RetrievalResult.score, Chunk.section, Chunk.text, Document.title, Document.source_path)
        .join(Chunk, RetrievalResult.chunk_id == Chunk.id).join(Document, Chunk.document_id == Document.id)
        .where(RetrievalResult.bot_response_id == message_id).order_by(RetrievalResult.rank)
    ).all()
    return {
        "message_id": message_id, "has_detail": True, "lead_id": lead.id,
        "time": msg.created_at.isoformat(),
        "question": question.text if question else None,
        "mode": br.mode, "escalate": br.escalate, "retriever": br.retriever,
        "embedding_model": br.embedding_model, "vector_db": br.vector_db, "llm_model": br.llm_model,
        "search_query": br.search_query,
        "latency_ms": {"retrieval": br.latency_retrieval_ms, "llm": br.latency_llm_ms, "total": br.latency_total_ms},
        "sources": [{"rank": r.rank, "score": r.score, "doc": r.title, "section": r.section or "", "text": r.text,
                     "source": r.source_path} for r in rows],
    }
