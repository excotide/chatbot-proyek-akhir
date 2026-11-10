"""Lapisan adapter pesan.

Backend (RAG, klasifikasi, CRM) hanya mengenal IncomingMessage dan MessagingProvider.
Sumber pesan (OpenWA atau Meta Cloud API) bisa diganti lewat MESSAGING_PROVIDER di .env.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class IncomingMessage:
    message_id: str   # id unik dari provider, dipakai untuk mencegah proses ganda
    sender: str       # nomor pengirim, format 628xxxx (tanpa +, tanpa @c.us)
    text: str
    timestamp: int    # unix detik
    provider: str     # "openwa" | "cloud_api"


class MessagingProvider(ABC):
    name: str

    @abstractmethod
    def verify_signature(self, raw_body: bytes, headers: dict[str, str]) -> bool:
        """True jika webhook benar-benar dari provider (cek HMAC)."""

    @abstractmethod
    def parse_webhook(self, payload: dict) -> list[IncomingMessage]:
        """Ambil pesan teks masuk dari payload webhook. Event lain diabaikan."""

    @abstractmethod
    async def send_text(self, to: str, text: str) -> str:
        """Kirim pesan teks ke nomor 628xxxx. Mengembalikan message id."""
