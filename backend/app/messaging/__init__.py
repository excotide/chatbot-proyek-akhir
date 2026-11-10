from app.config import settings
from app.messaging.base import IncomingMessage, MessagingProvider


def get_provider() -> MessagingProvider:
    if settings.messaging_provider == "cloud_api":
        from app.messaging.cloud_api import CloudAPIProvider
        return CloudAPIProvider()
    from app.messaging.openwa import OpenWAProvider
    return OpenWAProvider()


__all__ = ["IncomingMessage", "MessagingProvider", "get_provider"]
