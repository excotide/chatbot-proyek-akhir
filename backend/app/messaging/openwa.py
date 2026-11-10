import hashlib
import hmac
import time

import httpx

from app.config import settings
from app.messaging.base import IncomingMessage, MessagingProvider


def _strip_jid(jid: str) -> str:
    """'628123@c.us' / '628123@s.whatsapp.net' -> '628123'."""
    return jid.split("@", 1)[0].split(":", 1)[0]


class OpenWAProvider(MessagingProvider):
    name = "openwa"

    def __init__(self) -> None:
        self.base_url = settings.openwa_base_url.rstrip("/")
        self.headers = {"X-API-Key": settings.openwa_api_key}

    def verify_signature(self, raw_body: bytes, headers: dict[str, str]) -> bool:
        # Header: X-OpenWA-Signature: sha256=<hex HMAC-SHA256(raw body, secret)>
        secret = settings.openwa_webhook_secret
        if not secret:
            return True  # webhook didaftarkan tanpa secret
        expected = "sha256=" + hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, headers.get("x-openwa-signature", ""))

    def parse_webhook(self, payload: dict) -> list[IncomingMessage]:
        if payload.get("event") != "message.received":
            return []
        data = payload.get("data") or {}
        if data.get("fromMe") or data.get("isGroup"):
            return []
        chat_id = data.get("chatId") or data.get("from") or ""
        if chat_id.endswith("@g.us") or chat_id == "status@broadcast":
            return []
        body = data.get("body") or ""
        # nama tipe bisa beda antar engine (text/chat/conversation); yang penting ada teks & bukan media
        if data.get("hasMedia") or not body.strip():
            return []
        sender = data.get("author") or data.get("from") or chat_id
        return [IncomingMessage(
            message_id=str(data.get("id") or payload.get("idempotencyKey") or ""),
            sender=_strip_jid(sender),
            text=body,
            timestamp=int(data.get("timestamp") or time.time()),
            provider=self.name,
        )]

    async def send_text(self, to: str, text: str) -> str:
        url = f"{self.base_url}/sessions/{settings.openwa_session_id}/messages/send-text"
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(url, headers=self.headers, json={"chatId": f"{to}@c.us", "text": text})
            r.raise_for_status()
            return r.json().get("messageId", "")
