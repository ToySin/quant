"""Discord webhook tests — mocked HTTP."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from quant.notify import (
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_RED,
    DiscordWebhook,
    Field,
    color_for_pnl,
)

URL = "https://discord.com/api/webhooks/123/abc"


def test_constructor_requires_https():
    with pytest.raises(ValueError, match="HTTPS"):
        DiscordWebhook("http://example.com/hook")


def test_color_for_pnl():
    assert color_for_pnl(100) == COLOR_GREEN
    assert color_for_pnl(-50) == COLOR_RED
    assert color_for_pnl(0) != COLOR_GREEN
    assert color_for_pnl(0) != COLOR_RED


@patch("quant.notify.requests.post")
def test_post_sends_plain_content(mock_post):
    mock_post.return_value = MagicMock(status_code=204, text="")
    hook = DiscordWebhook(URL, username="Test")
    hook.post("hello world")

    args, kwargs = mock_post.call_args
    assert args[0] == URL
    assert kwargs["json"]["content"] == "hello world"
    assert kwargs["json"]["username"] == "Test"


@patch("quant.notify.requests.post")
def test_post_embed_sends_structured_payload(mock_post):
    mock_post.return_value = MagicMock(status_code=204, text="")
    hook = DiscordWebhook(URL)
    hook.post_embed(
        title="Daily snapshot",
        description="Equity unchanged",
        color=COLOR_BLUE,
        fields=[
            Field("Equity", "$100,000", inline=True),
            Field("PnL", "+0.0%", inline=True),
        ],
        footer="2026-05-10",
    )

    embed = mock_post.call_args.kwargs["json"]["embeds"][0]
    assert embed["title"] == "Daily snapshot"
    assert embed["color"] == COLOR_BLUE
    assert embed["description"] == "Equity unchanged"
    assert len(embed["fields"]) == 2
    assert embed["fields"][0]["name"] == "Equity"
    assert embed["footer"]["text"] == "2026-05-10"


@patch("quant.notify.requests.post")
def test_failure_raises_runtime_error(mock_post):
    mock_post.return_value = MagicMock(status_code=500, text="server error")
    hook = DiscordWebhook(URL)
    with pytest.raises(RuntimeError, match="500"):
        hook.post("test")


@patch("quant.notify.requests.post")
def test_network_exception_raises_runtime_error(mock_post):
    import requests
    mock_post.side_effect = requests.ConnectionError("nope")
    hook = DiscordWebhook(URL)
    with pytest.raises(RuntimeError, match="Discord webhook POST failed"):
        hook.post("test")


def test_from_env_missing_raises(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    with pytest.raises(RuntimeError, match="DISCORD_WEBHOOK_URL"):
        DiscordWebhook.from_env()


def test_from_env_loads_url(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", URL)
    hook = DiscordWebhook.from_env()
    assert hook._url == URL
