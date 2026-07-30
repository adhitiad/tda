"""Celery worker for PyTorch inference."""
import logging
import os
from typing import Any

from celery import Celery

logger = logging.getLogger("inference_worker")

_model_cache: dict[str, Any] = {}


def create_celery_app(
    broker_url: str = "redis://localhost:6379/0",
    result_backend: str = "redis://localhost:6379/1",
) -> Celery:
    """Create and configure Celery app."""
    app = Celery(
        "inference",
        broker=broker_url,
        backend=result_backend,
    )
    app.conf.task_serializer = "json"
    app.conf.result_serializer = "json"
    app.conf.accept_content = ["json"]
    app.conf.task_acks_late = True
    app.conf.worker_prefetch_multiplier = 1
    app.conf.result_expires = 3600

    @app.task(bind=True, name="inference.run", max_retries=2, default_retry_delay=5)
    def run_inference(
        self,
        model_path: str,
        model_type: str,
        input_size: int,
        input_data: list,
        device: str = "cpu",
    ) -> list:
        """Run PyTorch model inference."""
        try:
            import torch

            from crypto_trading_framework.ml.model import create_model

            cache_key = f"{model_path}:{model_type}:{input_size}:{device}"
            if cache_key not in _model_cache:
                model = create_model(model_type, input_size=input_size).to(device)
                state_dict = torch.load(model_path, map_location=device, weights_only=True)
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                _model_cache[cache_key] = model
                logger.info(f"[Inference] Loaded model from {model_path}")

            model = _model_cache[cache_key]
            x = torch.tensor(input_data, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits = model(x)
                preds = torch.sigmoid(logits).cpu().numpy().flatten()
            return preds.tolist()

        except Exception as exc:
            logger.exception("[Inference] Task failed")
            raise self.retry(exc=exc)

    return app


def get_celery_app(config: dict | None = None) -> Celery:
    """Get or create Celery app from config."""
    if config is None:
        broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
        result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
    else:
        tq = config.get("task_queue", {})
        celery_cfg = tq.get("celery", {})
        broker_url = celery_cfg.get("broker_url", "redis://localhost:6379/0")
        result_backend = celery_cfg.get("result_backend", "redis://localhost:6379/1")

    return create_celery_app(broker_url, result_backend)