import json
import smtplib
import ssl
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tenacity import retry, stop_after_attempt, wait_exponential

from crypto_trading_framework.core.logging import get_logger

logger = get_logger("alerting")


@dataclass
class Alert:
    level: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        ctx = ", ".join(f"{k}={v}" for k, v in self.context.items())
        if ctx:
            return f"[{self.level}] {self.message} ({ctx})"
        return f"[{self.level}] {self.message}"


class AlertChannel(ABC):
    @abstractmethod
    def send(self, alert: Alert) -> bool:
        ...


class TelegramChannel(AlertChannel):
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def send(self, alert: Alert) -> bool:
        payload = {
            "chat_id": self.chat_id,
            "text": str(alert),
            "parse_mode": "HTML",
        }
        try:
            req = Request(
                self.url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    logger.info(f"[Telegram] Alert terkirim: {alert.message}")
                    return True
                logger.warning(f"[Telegram] Gagal kirim alert: HTTP {resp.status}")
                return False
        except (HTTPError, URLError, OSError) as e:
            logger.error(f"[Telegram] Gagal kirim alert: {e}")
            return False


class DiscordChannel(AlertChannel):
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def send(self, alert: Alert) -> bool:
        color_map = {
            "CRITICAL": 15158332,
            "ERROR": 15158332,
            "WARNING": 16776960,
            "INFO": 3447003,
        }
        payload = {
            "embeds": [
                {
                    "title": f"Alert: {alert.level}",
                    "description": str(alert),
                    "color": color_map.get(alert.level.upper(), 3447003),
                }
            ]
        }
        try:
            req = Request(
                self.webhook_url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=10) as resp:
                if resp.status in (200, 204):
                    logger.info(f"[Discord] Alert terkirim: {alert.message}")
                    return True
                logger.warning(f"[Discord] Gagal kirim alert: HTTP {resp.status}")
                return False
        except (HTTPError, URLError, OSError) as e:
            logger.error(f"[Discord] Gagal kirim alert: {e}")
            return False


class EmailChannel(AlertChannel):
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: list[str],
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs

    def send(self, alert: Alert) -> bool:
        subject = f"[{alert.level}] Trading Bot Alert"
        body = str(alert)
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = self.from_addr
        msg["To"] = ", ".join(self.to_addrs)

        try:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, context=context) as server:
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())
            logger.info(f"[Email] Alert terkirim: {alert.message}")
            return True
        except (smtplib.SMTPException, OSError) as e:
            logger.error(f"[Email] Gagal kirim alert: {e}")
            return False


class AlertManager:
    _instance = None

    def __init__(self):
        self.channels: list[AlertChannel] = []
        self.min_level = "INFO"

    @classmethod
    def get_instance(cls) -> "AlertManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def configure(self, config: dict):
        self.channels = []
        self.min_level = config.get("min_level", "INFO")

        telegram = config.get("telegram", {})
        if telegram.get("enabled", False):
            bot_token = telegram.get("bot_token", "")
            chat_id = telegram.get("chat_id", "")
            if bot_token and chat_id:
                self.channels.append(TelegramChannel(bot_token=bot_token, chat_id=chat_id))

        discord = config.get("discord", {})
        if discord.get("enabled", False):
            webhook_url = discord.get("webhook_url", "")
            if webhook_url:
                self.channels.append(DiscordChannel(webhook_url=webhook_url))

        email = config.get("email", {})
        if email.get("enabled", False):
            required = ["smtp_host", "smtp_port", "from_addr", "to_addrs"]
            if all(email.get(k) for k in required):
                self.channels.append(EmailChannel(
                    smtp_host=email["smtp_host"],
                    smtp_port=int(email["smtp_port"]),
                    username=email.get("username", ""),
                    password=email.get("password", ""),
                    from_addr=email["from_addr"],
                    to_addrs=email["to_addrs"] if isinstance(email["to_addrs"], list) else [email["to_addrs"]],
                ))

        logger.info(f"[AlertManager] {len(self.channels)} channel aktif (min_level={self.min_level})")

    def _should_alert(self, level: str) -> bool:
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        try:
            return levels.index(level.upper()) >= levels.index(self.min_level.upper())
        except ValueError:
            return True

    def send(self, alert: Alert) -> bool:
        if not self._should_alert(alert.level):
            return False

        if not self.channels:
            logger.debug(f"[AlertManager] Tidak ada channel aktif, alert di-log saja: {alert}")
            return False

        success = True
        for channel in self.channels:
            try:
                ok = channel.send(alert)
                success = success and ok
            except Exception as e:
                logger.error(f"[AlertManager] Channel {type(channel).__name__} error: {e}")
                success = False
        return success


def alert_training_failure(symbol: str, error: str, context: dict | None = None):
    manager = AlertManager.get_instance()
    alert = Alert(
        level="ERROR",
        message=f"Training gagal untuk {symbol}: {error}",
        context=context or {},
    )
    manager.send(alert)


def alert_execution_error(symbol: str, error: str, context: dict | None = None):
    manager = AlertManager.get_instance()
    alert = Alert(
        level="ERROR",
        message=f"Execution error untuk {symbol}: {error}",
        context=context or {},
    )
    manager.send(alert)


def alert_drawdown_breach(symbol: str, drawdown_pct: float, threshold: float, context: dict | None = None):
    manager = AlertManager.get_instance()
    alert = Alert(
        level="CRITICAL",
        message=f"Drawdown breach {symbol}: {drawdown_pct:.2%} >= threshold {threshold:.2%}",
        context={"drawdown_pct": f"{drawdown_pct:.4f}", "threshold": f"{threshold:.4f}"},
    )
    manager.send(alert)


def alert_model_drift(symbol: str, drift_score: float, threshold: float, context: dict | None = None):
    manager = AlertManager.get_instance()
    alert = Alert(
        level="WARNING",
        message=f"Model drift terdeteksi {symbol}: score={drift_score:.4f} >= threshold={threshold:.4f}",
        context={"drift_score": f"{drift_score:.4f}", "threshold": f"{threshold:.4f}"},
    )
    manager.send(alert)
