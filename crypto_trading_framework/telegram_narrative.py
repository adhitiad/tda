"""OpenRouter-based narrative generator for Telegram signal messages."""

from __future__ import annotations

import asyncio
import os

from openai import AsyncOpenAI

from crypto_trading_framework.core.logging import get_logger

logger = get_logger("telegram_narrative")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL_ID = "nvidia/nemotron-3-ultra-550b-a55b:free"
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 2
REQUEST_TIMEOUT_SECONDS = 15

SYSTEM_PROMPT = (
    "You are a professional crypto trading analyst. Generate concise, "
    "convincing trading narratives in Indonesian Bahasa Indonesia."
)

PROMPT_TEMPLATE = (
    "Berdasarkan data sinyal trading berikut:\n"
    "Simbol: {symbol}\n"
    "Arah: {direction}\n"
    "Entry: {entry}\n"
    "TP: {tp}\n"
    "SL: {sl}\n"
    "RSI: {rsi}\n"
    "MACD: {macd}\n"
    "Sentimen Pasar: {sentiment_score}\n"
    "Buatlah 2 kalimat penjelasan profesional, padat, dan meyakinkan "
    "dalam bahasa Indonesia mengapa ini adalah peluang trading yang valid. "
    "Jangan gunakan kata-kata seperti 'pasti' atau 'jaminan'."
)


def _build_prompt(signal_data: dict) -> str:
    return PROMPT_TEMPLATE.format(
        symbol=signal_data.get("symbol", "UNKNOWN"),
        direction=signal_data.get("direction", "UNKNOWN"),
        entry=signal_data.get("entry", "UNKNOWN"),
        tp=signal_data.get("tp", "UNKNOWN"),
        sl=signal_data.get("sl", "UNKNOWN"),
        rsi=signal_data.get("rsi", "UNKNOWN"),
        macd=signal_data.get("macd", "UNKNOWN"),
        sentiment_score=signal_data.get("sentiment_score", "UNKNOWN"),
    )


async def generate_signal_narrative(signal_data: dict) -> str:
    """Generate a professional trading narrative in Indonesian via OpenRouter.

    Retries up to MAX_RETRIES times on HTTP 429 (rate limit) errors.
    Returns a fallback message if all attempts fail.
    """
    client = AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_prompt(signal_data)},
    ]

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=MODEL_ID,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=512,
                ),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            narrative = response.choices[0].message.content.strip()
            if narrative:
                logger.info("[Narrative] Successfully generated signal narrative")
                return narrative
            logger.warning("[Narrative] Empty response from OpenRouter, retrying")

        except asyncio.TimeoutError:
            logger.warning(f"[Narrative] Request timed out (attempt {attempt + 1})")
        except Exception as exc:
            status_code: int = 0
            if hasattr(exc, "response") and exc.response is not None:
                status_code = getattr(exc.response, "status_code", 0)
            if status_code == 429 and attempt < MAX_RETRIES:
                logger.warning(
                    f"[Narrative] Rate limited (429), retrying in "
                    f"{RETRY_DELAY_SECONDS}s (attempt {attempt + 1}/{MAX_RETRIES})"
                )
                await asyncio.sleep(RETRY_DELAY_SECONDS)
                continue
            logger.error(f"[Narrative] OpenRouter error: {exc}")

    logger.error("[Narrative] All attempts failed, returning fallback narrative")
    return (
        "Sinyal trading telah dihasilkan. Silakan evaluasi peluang ini "
        "dengan analisis teknikal dan manajemen risiko yang sesuai."
    )