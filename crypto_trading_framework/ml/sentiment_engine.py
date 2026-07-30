"""Local sentiment engine using groq for low-latency LLM-based sentiment analysis."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from groq import Groq

from crypto_trading_framework.core.logging import get_logger

logger = get_logger("sentiment_engine")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_ID = "openai/gpt-oss-20b"
TEMPERATURE = 0.32
MAX_COMPLETION_TOKENS = 2048
TOP_P = 1
REASONING_EFFORT = "high"
REQUEST_TIMEOUT_SECONDS = 3

SYSTEM_PROMPT = (
    "You are a financial sentiment analyzer. Analyze the following crypto headlines. "
    "Output ONLY a valid JSON object with a single key 'sentiment_score' ranging from "
    "-1.0 (extreme bearish) to 1.0 (extreme bullish). No markdown, no explanation."
)

TOOLS: list[dict[str, Any]] = [
    {"type": "browser_search"},
    {"type": "code_interpreter"},
]


def _parse_sentiment_score(response_text: str) -> float:
    try:
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```", 2)[-1].strip()
        payload = json.loads(cleaned)
        score = float(payload.get("sentiment_score", 0.0))
        return max(-1.0, min(1.0, score))
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as exc:
        logger.warning(f"[Sentiment] Failed to parse response: {exc}")
        return 0.0


async def analyze_sentiment_ollama(headlines: list[str]) -> float:
    """Analyze crypto headlines and return an aggregate sentiment score.

    Uses groq with openai/gpt-oss-20b for low-latency inference.
    Falls back to 0.0 on timeout or any error.
    """
    if not headlines:
        return 0.0

    combined = "\n".join(f"- {h}" for h in headlines)
    user_message = f"Headlines:\n{combined}"

    client = Groq(api_key=GROQ_API_KEY)

    try:
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=MODEL_ID,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=TEMPERATURE,
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                    top_p=TOP_P,
                    reasoning_effort=REASONING_EFFORT,
                    stream=True,
                    stop=None,
                    tools=TOOLS,
                ),
            ),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        chunks: list[str] = []
        for chunk in response:
            delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
            if delta:
                chunks.append(delta)

        full_text = "".join(chunks)
        score = _parse_sentiment_score(full_text)
        logger.info(f"[Sentiment] Score={score:.4f} from {len(headlines)} headlines")
        return score

    except asyncio.TimeoutError:
        logger.error("[Sentiment] groq request timed out after 3s, returning fallback 0.0")
        return 0.0
    except Exception as exc:
        logger.error(f"[Sentiment] groq analysis failed: {exc}, returning fallback 0.0")
        return 0.0