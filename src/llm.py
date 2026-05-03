"""Thin wrapper around the Groq SDK with streaming."""
from __future__ import annotations

import os
from typing import Iterator

from dotenv import load_dotenv

from src.config import GROQ_DEFAULT_MODEL

load_dotenv()


class GroqKeyMissing(RuntimeError):
    pass


def _client():
    """Lazy import + create the Groq client. Raises if key missing."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise GroqKeyMissing(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "(get one free at https://console.groq.com/keys)."
        )
    from groq import Groq

    return Groq(api_key=api_key)


def stream_chat(
    messages: list[dict[str, str]],
    model: str = GROQ_DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> Iterator[str]:
    """Yield text deltas from a streaming Groq chat completion."""
    client = _client()
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for event in completion:
        delta = event.choices[0].delta
        if delta and delta.content:
            yield delta.content


if __name__ == "__main__":
    # Smoke test: python -m src.llm
    msgs = [{"role": "user", "content": "Say hello in one short sentence."}]
    for tok in stream_chat(msgs):
        print(tok, end="", flush=True)
    print()
