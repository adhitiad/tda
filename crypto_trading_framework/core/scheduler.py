import threading
import time
from datetime import datetime, timedelta
from typing import Callable, Optional

from crypto_trading_framework.core.logging import get_logger

logger = get_logger("scheduler")


class TrainingScheduler:
    def __init__(self, training_callback: Callable, stop_callback: Optional[Callable] = None):
        self.training_callback = training_callback
        self.stop_callback = stop_callback
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.training_started = False
        self.training_completed = False

    def _is_training_window(self) -> bool:
        now = datetime.now()
        if now.weekday() != 6:
            return False
        current_time = now.time()
        return current_time >= datetime.strptime("02:00", "%H:%M").time()

    def _wait_until_training_window(self):
        while self.running and not self.training_started:
            now = datetime.now()
            if now.weekday() == 6:
                next_sunday_2am = now.replace(hour=2, minute=0, second=0, microsecond=0)
                if now >= next_sunday_2am:
                    break
                wait_seconds = (next_sunday_2am - now).total_seconds()
                if wait_seconds > 0:
                    logger.info(f"[SCHEDULER] Menunggu {wait_seconds/3600:.1f} jam sampai window pelatihan Minggu 02:00")
                    time.sleep(min(wait_seconds, 60))
                    continue
            time.sleep(60)

    def _run_training_loop(self):
        logger.info("[SCHEDULER] Memulai siklus pelatihan Minggu 02:00")
        self.training_started = True
        try:
            self.training_callback()
        except Exception as e:
            logger.error(f"[SCHEDULER] Error saat pelatihan: {e}")
        self.training_completed = True
        logger.info("[SCHEDULER] Pelatihan selesai. Bot akan berhenti.")
        if self.stop_callback:
            self.stop_callback()
        self.running = False

    def start(self):
        if self.running:
            logger.warning("[SCHEDULER] Scheduler sudah berjalan")
            return
        self.running = True
        self.training_started = False
        self.training_completed = False
        self.thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.thread.start()
        logger.info("[SCHEDULER] Scheduler dimulai")

    def _run_scheduler(self):
        while self.running and not self.training_completed:
            if self._is_training_window():
                self._run_training_loop()
            else:
                time.sleep(60)

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            try:
                self.thread.join(timeout=5)
            except KeyboardInterrupt:
                logger.warning("[SCHEDULER] KeyboardInterrupt during shutdown")
        logger.info("[SCHEDULER] Scheduler dihentikan")

    def is_training_active(self) -> bool:
        return self.training_started and not self.training_completed

    def get_next_training_time(self) -> datetime:
        now = datetime.now()
        days_until_sunday = (6 - now.weekday()) % 7
        if days_until_sunday == 0 and now.time() >= datetime.strptime("02:00", "%H:%M").time():
            days_until_sunday = 7
        next_sunday = now + timedelta(days=days_until_sunday)
        return next_sunday.replace(hour=2, minute=0, second=0, microsecond=0)
