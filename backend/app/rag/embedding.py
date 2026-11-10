"""Model embedding (sentence-transformers, CPU).

Teks chunk dan teks pertanyaan HARUS di-embed dengan model yang sama. Vektor dinormalisasi
(panjang 1), sehingga cosine similarity = hasil kali titik (dot product).
"""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import settings

# Cache model di drive D (C: hampir penuh). Harus diset sebelum sentence_transformers diimpor.
os.environ.setdefault("HF_HOME", settings.hf_home)

# Awalan teks per model, sesuai kartu model masing-masing di Hugging Face.
MODELS: dict[str, dict[str, str]] = {
    "intfloat/multilingual-e5-small": {"query": "query: ", "passage": "passage: "},  # dipakai chatbot (hasil evaluasi terbaik)
    "LazarusNLP/all-indo-e5-small-v4": {"query": "", "passage": ""},       # pembanding, khusus Indonesia
    "firqaaa/indo-sentence-bert-base": {"query": "", "passage": ""},       # pembanding berbasis IndoBERT
}


def _prefix(model_name: str, kind: str) -> str:
    if model_name not in MODELS:
        raise ValueError(f"Model embedding tidak dikenal: {model_name}. Pilihan: {', '.join(MODELS)}")
    return MODELS[model_name][kind]


def local_model_dir(model_name: str) -> Path:
    """Folder hasil scripts/download_models.py, mis. .cache/models/LazarusNLP__all-indo-e5-small-v4."""
    return Path(settings.hf_home).parent / "models" / model_name.replace("/", "__")


def model_available(model_name: str) -> bool:
    """True bila model sudah lengkap di folder lokal (hasil scripts/download_models.py)."""
    d = local_model_dir(model_name)
    return (d / "modules.json").exists() and any((d / w).exists() for w in ("model.safetensors", "pytorch_model.bin"))


@lru_cache(maxsize=4)
def load_model(model_name: str):
    from sentence_transformers import SentenceTransformer  # impor lambat: torch berat

    local = local_model_dir(model_name)
    if (local / "modules.json").exists():
        return SentenceTransformer(str(local), device="cpu", local_files_only=True)
    return SentenceTransformer(model_name, device="cpu")


def embed_passages(texts: list[str], model_name: str, batch_size: int = 16) -> np.ndarray:
    p = _prefix(model_name, "passage")
    return load_model(model_name).encode(
        [p + t for t in texts], batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True
    )


def embed_query(text: str, model_name: str) -> np.ndarray:
    p = _prefix(model_name, "query")
    return load_model(model_name).encode([p + text], normalize_embeddings=True, convert_to_numpy=True)[0]
