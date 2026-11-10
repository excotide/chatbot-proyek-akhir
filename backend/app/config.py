from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# .env dibaca dari folder backend, dari mana pun perintah dijalankan (uvicorn, alembic, skrip).
# Proyek PA/.env (berisi GROQ_API_KEY) ikut dibaca lebih dulu; nilai di backend/.env menimpa bila sama.
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
ROOT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(ROOT_ENV_FILE, ENV_FILE), env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = ""  # postgresql+psycopg://user:pass@127.0.0.1:5434/chatbot_pa

    messaging_provider: str = "openwa"  # "openwa" | "cloud_api"

    # OpenWA
    openwa_base_url: str = "http://localhost:2785/api"
    openwa_api_key: str = ""
    openwa_webhook_secret: str = ""
    openwa_session_name: str = "pa-bot"
    openwa_session_id: str = ""
    backend_webhook_url: str = "http://host.docker.internal:8000/webhook/openwa"

    # Meta Cloud API
    wa_verify_token: str = ""
    wa_app_secret: str = ""
    wa_access_token: str = ""
    wa_phone_number_id: str = ""
    wa_graph_version: str = "v21.0"

    # RAG (simulator)
    kb_root: str = "data/kb"           # berisi asli/ dan dummy/, relatif terhadap folder backend
    kb_data_source: str = "asli"       # sumber KB yang dipakai chatbot: asli | dummy
    chunk_rows: int = 5                # CSV: jumlah baris data per chunk (Markdown: per heading)
    chunk_overlap: int = 1             # CSV: baris yang diulang antar chunk
    top_k: int = 5                     # chunk yang dikirim ke LLM; 5 agar baris katalog per paket ikut terambil (uji 9 Nov)
    # Pilihan berdasarkan evaluasi 9 Nov 2026 (data/eval/hasil-retrieval.md): semantic + multilingual-e5-small
    # memberi Hit@3 tertinggi (94%). Di simulator metode bisa diganti per pesan.
    retriever: str = "semantic"        # bm25 | semantic | hybrid
    # Ambang dimatikan (0): evaluasi menunjukkan tidak ada nilai yang memisahkan pertanyaan KB dan luar KB.
    # Keputusan "tidak ada info" diserahkan ke LLM (penanda TIDAK_ADA_INFO).
    min_score: float = 0.0             # BM25: skor minimum
    min_similarity: float = 0.0        # semantic/hybrid: cosine minimum
    rrf_k: int = 60                    # konstanta Reciprocal Rank Fusion untuk hybrid

    # Database vektor: pgvector (utama, tabel chunk_embeddings) | chroma | surrealdb (pembanding)
    vector_db: str = "pgvector"
    chroma_url: str = "http://127.0.0.1:8001"
    surreal_url: str = "http://127.0.0.1:8002"
    surreal_user: str = "root"
    surreal_pass: str = ""
    surreal_ns: str = "pa"
    surreal_db: str = "vektor"

    # Embedding (sentence-transformers, CPU)
    embedding_model: str = "intfloat/multilingual-e5-small"  # terbaik dari 3 model yang dibandingkan
    hf_home: str = str(Path(__file__).resolve().parents[2] / ".cache" / "huggingface")  # cache model di drive D

    # Riwayat percakapan
    history_messages: int = 10         # jumlah pesan terakhir (pelanggan + bot) yang dibawa ke LLM
    query_rewrite: bool = True         # tulis ulang tiap pesan jadi pertanyaan lengkap & baku ("keduanya", "yg setaun brp") sebelum retrieval

    # Penyedia LLM: ollama (lokal) | groq (cloud, data chat dikirim ke Groq)
    llm_provider: str = "ollama"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    # Model Groq yang bisa dipilih di simulator (dicek tersedia di akun 9 Nov 2026)
    groq_models: list[str] = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
    groq_timeout: float = 60.0

    # LLM lokal (Ollama)
    ollama_base_url: str = "http://127.0.0.1:11434"  # 127.0.0.1, bukan localhost: di Windows localhost mencoba IPv6 dulu dan lambat gagal
    ollama_model: str = "qwen2.5:3b"
    ollama_timeout: float = 120.0


settings = Settings()
