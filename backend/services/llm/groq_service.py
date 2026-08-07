"""
Talks to the Groq API (OpenAI-compatible /chat/completions endpoint).
Groq hosts full-size open models (Llama 3.3 70B, Llama 3.1 8B, etc.) on their
own fast inference hardware, so you get real quality without needing a GPU
or downloading multi-gigabyte weights locally. This is the app's only LLM
provider — no local model server to run.

Setup:
    1. Get a free API key: https://console.groq.com/keys
    2. In .env:  GROQ_API_KEY=gsk_...
                 GROQ_MODEL=openai/gpt-oss-120b   (or openai/gpt-oss-20b for speed)
                 LLM_PROVIDER=groq

NOTE ON MODEL CHOICE: Groq deprecates/decommissions models over time (see
https://console.groq.com/docs/deprecations) — llama-3.3-70b-versatile and
llama-3.1-8b-instant were deprecated June 2026 in favor of the openai/gpt-oss-*
models. If chat requests start failing with a "model_decommissioned" error,
check that page and update GROQ_MODEL in .env accordingly; no code change
should be needed since the model id is just a string.
"""
import json
import os
import re
from typing import AsyncIterator

import httpx

from services.prompt_builder import BASE_INSTRUCTIONS

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Module-level, lazily-created client shared across requests instead of a
# fresh httpx.AsyncClient() per chat message — reuses the underlying
# TCP/TLS connection (keep-alive) to api.groq.com instead of paying a full
# connection + TLS handshake on every single message.
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=60)
    return _client

DEFAULT_SYSTEM_PROMPT = BASE_INSTRUCTIONS + (
    "\n\nRespond in the same language the user writes in (Arabic or English); "
    "never mix languages within one reply."
)

TITLE_SYSTEM_PROMPT = (
    "You name chat conversations. Given the user's first message, reply with "
    "ONLY a very short title of 3-6 words in the same language as the message. "
    "No quotes, no punctuation at the end, no explanation."
)


def _clean_title(raw: str, fallback: str) -> str:
    title = raw.strip().strip('"').strip("'").strip("«»").strip()
    title = re.sub(r"\s+", " ", title)
    if len(title) > 60:
        title = title[:60].rstrip()
    return title or fallback


async def generate_chat_title(first_message: str) -> str:
    """Short model-written title for a brand-new chat, derived once from its
    very first message. Non-streaming (max_tokens=30) so it's a cheap call;
    falls back to a prefix of the message if the provider is unreachable."""
    fallback = first_message.strip()[:60] or "New conversation"
    if not GROQ_API_KEY:
        return fallback
    try:
        client = _get_client()
        resp = await client.post(
            GROQ_URL,
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": TITLE_SYSTEM_PROMPT},
                    {"role": "user", "content": first_message[:500]},
                ],
                "max_tokens": 30,
                "temperature": 0.3,
            },
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        return _clean_title(raw, fallback)
    except Exception:
        return fallback


async def stream_chat(
    history: list[dict], user_message: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> AsyncIterator[str]:
    """Streams the assistant's reply as plain text chunks, given the running
    conversation history, the new user message, and the system prompt
    (typically the output of services/prompt_builder.py, already grounded
    in retrieved document context)."""
    if not GROQ_API_KEY:
        yield (
            "\n\n[GROQ_API_KEY is not set. Get a free key at "
            "https://console.groq.com/keys and add GROQ_API_KEY=... to your .env]"
        )
        return

    messages = [{"role": "system", "content": system_prompt}]
    messages += history
    messages.append({"role": "user", "content": user_message})

    payload = {"model": GROQ_MODEL, "messages": messages, "stream": True}
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    try:
        client = _get_client()
        async with client.stream("POST", GROQ_URL, json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                yield (
                    f"\n\n[Groq returned {resp.status_code}: {body.decode(errors='ignore')[:300]}. "
                    f"Check GROQ_API_KEY and GROQ_MODEL in your .env.]"
                )
                return
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                piece = delta.get("content", "")
                if piece:
                    yield piece
    except httpx.ConnectError:
        yield "\n\n[Couldn't reach Groq (api.groq.com). Check your internet connection.]"
    except httpx.TimeoutException:
        yield "\n\n[Groq request timed out. Try again.]"
