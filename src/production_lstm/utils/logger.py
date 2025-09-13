import logging
import json
import sys
from datetime import datetime
from typing import Dict, Optional


class StructuredJSONLogger:
    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger("StructuredLogger")
        self.logger.setLevel(log_level)

        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            self.logger.addHandler(handler)

    def _log(self, level: str, event: str, data: Dict):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "event": event,
            **data,
        }
        self.logger.info(json.dumps(log_entry, default=str))

    def log_prediction(
        self,
        prediction: float,
        actual: Optional[float] = None,
        features: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        trace_id: Optional[str] = None,
    ):
        data = {
            "prediction": prediction,
            "actual": actual,
            "error": actual - prediction if actual is not None else None,
            "features_summary": features,
            "metadata": metadata,
            "trace_id": trace_id,
        }
        self._log("INFO", "model_prediction", data)

    def log_event(self, event: str, level: str = "INFO", **kwargs):
        self._log(level.upper(), event, kwargs)

    def log_error(self, event: str, error_message: str, exc_info=None, **kwargs):
        data = {
            "error_message": error_message,
            "details": kwargs,
        }
        if exc_info:
            data["exception"] = str(exc_info)
        self._log("ERROR", event, data)
