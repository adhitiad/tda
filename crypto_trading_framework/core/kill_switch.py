"""
Kill Switch (Dead Man's Switch) for graceful shutdown.

Pengecekan dilakukan melalui:
1. File lokal `KILL_SWITCH.txt` di direktori proyek.
2. Redis key `kill_switch` (jika Redis tersedia).

Setiap iterasi loop utama harus memanggil `is_active()`.
Jika kill switch aktif, panggil `trigger()` untuk memulai
graceful shutdown.
"""

from __future__ import annotations

from pathlib import Path

from crypto_trading_framework.core.logging import get_logger

logger = get_logger("kill_switch")

try:
    from crypto_trading_framework.db.redis_cache import get_redis_cache
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

_KILL_FILE = "KILL_SWITCH.txt"
_KILL_REDIS_KEY = "kill_switch"


def _check_file_kill_switch() -> bool:
    path = Path(_KILL_FILE)
    if not path.exists():
        return False
    try:
        content = path.read_text(encoding="utf-8").strip().lower()
        if content in ("1", "true", "yes", "on", "active"):
            return True
    except OSError as exc:
        logger.debug("[KillSwitch] Gagal baca file kill switch: %s", exc)
    return False


def _check_redis_kill_switch() -> bool:
    if not REDIS_AVAILABLE:
        return False
    try:
        cache = get_redis_cache()
        value = cache.get(_KILL_REDIS_KEY)
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on", "active")
        if isinstance(value, bytes):
            return value.decode("utf-8").strip().lower() in ("1", "true", "yes", "on", "active")
        return bool(value)
    except Exception as exc:
        logger.debug("[KillSwitch] Gagal baca Redis kill switch: %s", exc)
        return False


def is_active() -> bool:
    if _check_file_kill_switch():
        return True
    return _check_redis_kill_switch()


def trigger(reason: str = "manual") -> None:
    try:
        Path(_KILL_FILE).write_text(reason, encoding="utf-8")
        logger.warning("[KillSwitch] Kill switch diaktifkan via file: %s", reason)
    except OSError as exc:
        logger.error("[KillSwitch] Gagal tulis file kill switch: %s", exc)

    if REDIS_AVAILABLE:
        try:
            cache = get_redis_cache()
            cache.set(_KILL_REDIS_KEY, reason, ttl=3600)
            logger.warning("[KillSwitch] Kill switch diaktifkan via Redis: %s", reason)
        except Exception as exc:
            logger.error("[KillSwitch] Gagal set Redis kill switch: %s", exc)


def clear() -> None:
    try:
        path = Path(_KILL_FILE)
        if path.exists():
            path.unlink()
            logger.info("[KillSwitch] File kill switch dihapus")
    except OSError as exc:
        logger.error("[KillSwitch] Gagal hapus file kill switch: %s", exc)

    if REDIS_AVAILABLE:
        try:
            cache = get_redis_cache()
            cache.delete(_KILL_REDIS_KEY)
            logger.info("[KillSwitch] Redis kill switch dihapus")
        except Exception as exc:
            logger.error("[KillSwitch] Gagal hapus Redis kill switch: %s", exc)
