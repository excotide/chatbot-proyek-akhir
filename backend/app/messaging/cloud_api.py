"""Meta WhatsApp Cloud API (resmi). Dipakai setelah akun bisnis terverifikasi."""
import hashlib
import hmac

import httpx

from app.config import settings
from app.messaging.base import IncomingMessage, MessagingProvider


class CloudAPIProvider(MessagingProvider):
    name = "cloud_api"

    def verify_signature(self, raw_body: bytes, headers: dict[str, str]) -> bool:
        expected = "sha256=" + hmac.new(settings.wa_app_secret.encode(), raw_body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, headers.get("x-hub-signature-256", ""))

    def parse_webhook(self, payload: dict) -> list[IncomingMessage]:
        out = []
        for entry in payload.get("entry", []):
            for change in entry.get("changes", []):
                for msg in change.get("value", {}).get("messages", []):  # status sent/read diabaikan
                    if msg.get("type") == "text":
                        out.append(IncomingMessage(
                            message_id=msg["id"],
                            sender=msg["from"],
                            text=msg["text"]["body"],
                            timestamp=int(msg.get("timestamp", 0)),
                            provider=self.name,
                        ))
        return out

    async def send_text(self, to: str, text: str) -> str:
        url = f"https://graph.facebook.com/{settings.wa_graph_version}/{settings.wa_phone_number_id}/messages"
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                url,
                headers={"Authorization": f"Bearer {settings.wa_access_token}"},
                json={"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": text}},
            )
            r.raise_for_status()
            return r.json()["messages"][0]["id"]
