import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = logging.INFO


def setup_logger(
    name: str = "crypto_trading",
    log_file: str | None = None,
    level: int = LOG_LEVEL,
    fmt: str = LOG_FORMAT,
    datefmt: str = LOG_DATE_FORMAT,
) -> logging.Logger:
    """
    Membuat dan mengkonfigurasi logger.

    :param name: Nama logger
    :param log_file: Path file log (opsional). Jika None, hanya log ke console.
    :param level: Level log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    :param fmt: Format string log
    :param datefmt: Format tanggal
    :return: Instance logger yang sudah dikonfigurasi
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "crypto_trading") -> logging.Logger:
    """
    Mendapatkan instance logger. Jika belum ada, buat baru dengan konfigurasi default.

    :param name: Nama logger
    :return: Instance logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        setup_logger(name)
    return logger
