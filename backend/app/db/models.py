"""Model tabel sesuai Desain-Database.md (kamus data, bagian 4).

Perubahan struktur tabel dilakukan lewat migrasi Alembic:
    alembic revision --autogenerate -m "pesan"
    alembic upgrade head
"""
from datetime import datetime
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CHAR,
    REAL,
    BigInteger,
    Boolean,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    false,
    func,
    text as sql_text,
    true,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Kode tahap funnel. Urutan dan label tampilan diatur di kode, bukan di tabel (lihat bagian 8).
STAGES = ("tanya_tanya", "diskusi_spesifik", "nego", "invoice", "quotation", "deal")
REPLY_MODES = ("llm", "retrieval_only", "no_context", "smalltalk")


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


def _ts(**kw) -> Mapped[datetime]:
    return mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False, **kw)


class Base(DeclarativeBase):
    # Nama constraint konsisten, supaya migrasi Alembic berikutnya rapi.
    metadata = MetaData(naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    })


# ---------- CRM & percakapan ----------

class User(Base):
    __tablename__ = "users"
    __table_args__ = (CheckConstraint(_in("role", ("admin", "sales")), name="role"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(150), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(10))
    created_at: Mapped[datetime] = _ts()


class Lead(Base):
    __tablename__ = "leads"
    __table_args__ = (
        CheckConstraint(_in("channel", ("whatsapp", "simulator")), name="channel"),
        CheckConstraint(_in("data_source", ("asli", "dummy")), name="data_source"),
        CheckConstraint(f"current_stage IS NULL OR {_in('current_stage', STAGES)}", name="current_stage"),
        CheckConstraint("current_confidence IS NULL OR current_confidence BETWEEN 0 AND 1", name="current_confidence"),
        CheckConstraint(f"stage_source IS NULL OR {_in('stage_source', ('model', 'manual'))}", name="stage_source"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    phone: Mapped[str] = mapped_column(String(30), unique=True)
    name: Mapped[str | None] = mapped_column(String(100))
    channel: Mapped[str] = mapped_column(String(10))
    data_source: Mapped[str] = mapped_column(String(10))
    needs: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    current_stage: Mapped[str | None] = mapped_column(String(20))
    current_confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    stage_source: Mapped[str | None] = mapped_column(String(10))
    bot_active: Mapped[bool] = mapped_column(Boolean, server_default=true())
    last_message_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    created_at: Mapped[datetime] = _ts()

    messages: Mapped[list["Message"]] = relationship(back_populates="lead", order_by="Message.created_at")
    classifications: Mapped[list["Classification"]] = relationship(back_populates="lead")


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint(_in("sender_type", ("customer", "bot", "sales")), name="sender_type"),
        # pesan sales wajib mencatat siapa sales-nya
        CheckConstraint("sender_type <> 'sales' OR sender_user_id IS NOT NULL", name="sales_has_user"),
        Index("ix_messages_lead_id_created_at", "lead_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    lead_id: Mapped[int] = mapped_column(ForeignKey("leads.id", ondelete="CASCADE"))
    sender_type: Mapped[str] = mapped_column(String(10))
    sender_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"))
    text: Mapped[str] = mapped_column(Text)
    # id pesan dari WhatsApp; UNIQUE mencegah pesan diproses dua kali saat webhook dikirim ulang
    wa_message_id: Mapped[str | None] = mapped_column(String(100), unique=True)
    created_at: Mapped[datetime] = _ts()

    lead: Mapped[Lead] = relationship(back_populates="messages")
    bot_response: Mapped["BotResponse | None"] = relationship(
        back_populates="message", foreign_keys="BotResponse.message_id", uselist=False
    )


# ---------- Chatbot (RAG) ----------

class BotResponse(Base):
    __tablename__ = "bot_responses"
    __table_args__ = (CheckConstraint(_in("mode", REPLY_MODES), name="mode"),)

    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id", ondelete="CASCADE"), primary_key=True)
    reply_to_message_id: Mapped[int] = mapped_column(ForeignKey("messages.id", ondelete="CASCADE"))
    mode: Mapped[str] = mapped_column(String(20))
    escalate: Mapped[bool] = mapped_column(Boolean, server_default=false())
    retriever: Mapped[str | None] = mapped_column(String(20))
    embedding_model: Mapped[str | None] = mapped_column(String(100))   # untuk semantic/hybrid
    vector_db: Mapped[str | None] = mapped_column(String(20))          # pgvector | chroma | surrealdb
    search_query: Mapped[str | None] = mapped_column(Text)            # pertanyaan hasil tulis ulang, bila berbeda
    llm_model: Mapped[str | None] = mapped_column(String(100))
    latency_retrieval_ms: Mapped[int | None] = mapped_column(Integer)
    latency_llm_ms: Mapped[int | None] = mapped_column(Integer)
    latency_total_ms: Mapped[int] = mapped_column(Integer)

    message: Mapped[Message] = relationship(back_populates="bot_response", foreign_keys=[message_id])
    results: Mapped[list["RetrievalResult"]] = relationship(back_populates="bot_response", order_by="RetrievalResult.rank")


class RetrievalResult(Base):
    __tablename__ = "retrieval_results"
    __table_args__ = (UniqueConstraint("bot_response_id", "rank", name="uq_retrieval_results_response_rank"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    bot_response_id: Mapped[int] = mapped_column(ForeignKey("bot_responses.message_id", ondelete="CASCADE"))
    # RESTRICT: chunk yang pernah dipakai menjawab tidak boleh terhapus, supaya log tetap bisa dilacak
    chunk_id: Mapped[int] = mapped_column(ForeignKey("chunks.id", ondelete="RESTRICT"))
    rank: Mapped[int] = mapped_column(SmallInteger)
    score: Mapped[float] = mapped_column(REAL)

    bot_response: Mapped[BotResponse] = relationship(back_populates="results")
    chunk: Mapped["Chunk"] = relationship()


# ---------- Knowledge base ----------

class Document(Base):
    """Satu baris = satu versi isi file. Bila file berubah, versi lama dinonaktifkan (is_active = false)
    tetapi tidak dihapus, karena chunk-nya mungkin sudah dipakai bot menjawab (retrieval_results)."""
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(_in("data_source", ("asli", "dummy")), name="data_source"),
        UniqueConstraint("source_path", "content_hash", name="uq_documents_path_hash"),
        # hanya boleh ada satu versi aktif per file
        Index("uq_documents_active_path", "source_path", unique=True, postgresql_where=sql_text("is_active")),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    source_path: Mapped[str] = mapped_column(String(255))
    data_source: Mapped[str] = mapped_column(String(10))
    content_hash: Mapped[str] = mapped_column(CHAR(64))
    is_active: Mapped[bool] = mapped_column(Boolean, server_default=true())
    created_at: Mapped[datetime] = _ts()

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document")


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_size", "chunk_overlap", "chunk_index", name="uq_chunks_strategy_index"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    chunk_index: Mapped[int] = mapped_column(Integer)
    section: Mapped[str | None] = mapped_column(String(200))
    text: Mapped[str] = mapped_column(Text)
    # Markdown: 0/0 = satu chunk per heading. CSV: jumlah baris per chunk dan baris yang diulang.
    chunk_size: Mapped[int] = mapped_column(SmallInteger)
    chunk_overlap: Mapped[int] = mapped_column(SmallInteger)

    document: Mapped[Document] = relationship(back_populates="chunks")


class ChunkEmbedding(Base):
    """Khusus varian pgvector di komparasi database vektor."""
    __tablename__ = "chunk_embeddings"
    __table_args__ = (UniqueConstraint("chunk_id", "model_name", name="uq_chunk_embeddings_chunk_model"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    chunk_id: Mapped[int] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"))
    model_name: Mapped[str] = mapped_column(String(100))
    # dimensi belum dikunci; ditetapkan (dan indeks HNSW dibuat) setelah model embedding dipilih
    embedding: Mapped[list[float]] = mapped_column(Vector())


# ---------- Klasifikasi tahap ----------

class Classification(Base):
    __tablename__ = "classifications"
    __table_args__ = (
        CheckConstraint(_in("predicted_stage", STAGES), name="predicted_stage"),
        CheckConstraint("confidence BETWEEN 0 AND 1", name="confidence"),
        Index("ix_classifications_lead_id_created_at", "lead_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    lead_id: Mapped[int] = mapped_column(ForeignKey("leads.id", ondelete="CASCADE"))
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id", ondelete="CASCADE"))
    model_name: Mapped[str] = mapped_column(String(100))
    predicted_stage: Mapped[str] = mapped_column(String(20))
    confidence: Mapped[Decimal] = mapped_column(Numeric(5, 4))
    probabilities: Mapped[dict] = mapped_column(JSONB)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = _ts()

    lead: Mapped[Lead] = relationship(back_populates="classifications")
    reviews: Mapped[list["StageReview"]] = relationship(back_populates="classification")


class StageReview(Base):
    __tablename__ = "stage_reviews"
    __table_args__ = (
        CheckConstraint(_in("action", ("confirm", "correct")), name="action"),
        CheckConstraint(_in("final_stage", STAGES), name="final_stage"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    classification_id: Mapped[int] = mapped_column(ForeignKey("classifications.id", ondelete="CASCADE"))
    reviewer_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"))
    action: Mapped[str] = mapped_column(String(10))
    final_stage: Mapped[str] = mapped_column(String(20))
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _ts()

    classification: Mapped[Classification] = relationship(back_populates="reviews")
