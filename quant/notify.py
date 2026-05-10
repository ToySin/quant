"""Discord webhook notifications.

Discord webhooks are URLs that accept JSON POST requests. The
content can be plain text, or richer "embeds" with title, color,
fields, and a footer. We use embeds for daily PnL and rebalance
summaries so they stand out in a feed.

Setup: Discord channel → Edit Channel → Integrations → Webhooks
→ New Webhook → Copy URL. Paste into the workspace .env as
DISCORD_WEBHOOK_URL.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests


# Discord embed colors are 24-bit RGB integers
COLOR_GREEN = 0x2ECC71
COLOR_RED = 0xE74C3C
COLOR_BLUE = 0x3498DB
COLOR_GREY = 0x95A5A6


@dataclass(frozen=True)
class Field:
    name: str
    value: str
    inline: bool = True


class DiscordWebhook:
    """Thin wrapper around a Discord webhook URL.

    Failures are raised as RuntimeError so callers can decide whether
    to swallow them — a notifier shouldn't block a rebalance run on
    transient network issues.
    """

    def __init__(self, url: str, *, username: str = "Quant Bot",
                 timeout: float = 10.0) -> None:
        if not url.startswith("https://"):
            raise ValueError("webhook URL must be HTTPS")
        self._url = url
        self._username = username
        self._timeout = timeout

    @classmethod
    def from_env(cls) -> "DiscordWebhook":
        url = os.environ.get("DISCORD_WEBHOOK_URL")
        if not url:
            raise RuntimeError(
                "Set DISCORD_WEBHOOK_URL in the workspace .env"
            )
        return cls(url)

    def post(self, content: str) -> None:
        """Plain-text post."""
        self._send({"username": self._username, "content": content})

    def post_embed(self, *, title: str, description: str = "",
                   color: int = COLOR_BLUE,
                   fields: list[Field] | None = None,
                   footer: str = "") -> None:
        """Embed-style post with title, color bar on the side, and fields."""
        embed: dict = {"title": title, "color": color}
        if description:
            embed["description"] = description
        if fields:
            embed["fields"] = [
                {"name": f.name, "value": f.value, "inline": f.inline}
                for f in fields
            ]
        if footer:
            embed["footer"] = {"text": footer}

        self._send({
            "username": self._username,
            "embeds": [embed],
        })

    def _send(self, payload: dict) -> None:
        try:
            response = requests.post(self._url, json=payload,
                                     timeout=self._timeout)
        except requests.RequestException as exc:
            raise RuntimeError(f"Discord webhook POST failed: {exc}") from exc
        if response.status_code >= 300:
            raise RuntimeError(
                f"Discord webhook returned {response.status_code}: {response.text}"
            )


def color_for_pnl(pnl: float) -> int:
    """Green if positive, red if negative, grey if zero."""
    if pnl > 0:
        return COLOR_GREEN
    if pnl < 0:
        return COLOR_RED
    return COLOR_GREY
