import numpy as np
import smtplib
import json
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, Any, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

from production_lstm.config import settings
from production_lstm.utils.logger import StructuredJSONLogger


class AlertManager:
    def __init__(self, logger: StructuredJSONLogger):
        self.config = settings
        self.logger = logger
        self.alert_history = deque(maxlen=100)

    def check_model_performance(
        self, predictions: np.ndarray, actuals: np.ndarray
    ) -> Dict[str, Any]:
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        r2 = r2_score(actuals, predictions)

        alerts = []
        if mae > self.config.max_mae_threshold:
            alerts.append(
                {
                    "type": "PERFORMANCE_DEGRADATION",
                    "metric": "MAE",
                    "value": mae,
                    "threshold": self.config.max_mae_threshold,
                    "severity": "HIGH",
                }
            )
        if rmse > self.config.max_rmse_threshold:
            alerts.append(
                {
                    "type": "PERFORMANCE_DEGRADATION",
                    "metric": "RMSE",
                    "value": rmse,
                    "threshold": self.config.max_rmse_threshold,
                    "severity": "HIGH",
                }
            )
        if r2 < self.config.min_r2_threshold:
            alerts.append(
                {
                    "type": "PERFORMANCE_DEGRADATION",
                    "metric": "R2",
                    "value": r2,
                    "threshold": self.config.min_r2_threshold,
                    "severity": "MEDIUM",
                }
            )

        for alert in alerts:
            self._process_alert(alert)

        return {"mae": mae, "rmse": rmse, "r2": r2, "alerts_triggered": alerts}

    def check_data_drift(
        self, current_data: np.ndarray, reference_data: np.ndarray
    ) -> Dict[str, Any]:
        drift_alerts = []
        for i in range(current_data.shape[1]):
            statistic, _ = stats.ks_2samp(reference_data[:, i], current_data[:, i])

            if statistic > self.config.drift_threshold:
                drift_alerts.append(
                    {
                        "type": "DATA_DRIFT",
                        "feature_index": i,
                        "ks_statistic": statistic,
                        "threshold": self.config.drift_threshold,
                        "severity": "HIGH" if statistic > 0.3 else "MEDIUM",
                    }
                )

        for alert in drift_alerts:
            self._process_alert(alert)

        return {
            "drift_detected": len(drift_alerts) > 0,
            "alerts_triggered": drift_alerts,
        }

    def _process_alert(self, alert: Dict[str, Any]):
        alert["timestamp"] = datetime.now().isoformat()
        self.alert_history.append(alert)
        self.logger.log_event("alert_triggered", level="WARNING", alert_details=alert)

        if (
            alert["severity"] == "HIGH"
            and self.config.smtp_user
            and self.config.smtp_password
        ):
            self._send_email_alert(alert)

    def _send_email_alert(self, alert: Dict[str, Any]):
        try:
            msg = MimeMultipart()
            msg["From"] = self.config.smtp_user
            msg["To"] = self.config.alert_email
            msg["Subject"] = f"Alerta do Modelo LSTM: {alert['type']}"

            body = f"Alerta do Sistema de Predição LSTM\n\nDetalhes:\n{json.dumps(alert, indent=2, default=str)}"
            msg.attach(MimeText(body, "plain"))

            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_user, self.config.smtp_password)
                server.send_message(msg)
            self.logger.log_event("alert_email_sent", recipient=self.config.alert_email)
        except Exception as e:
            self.logger.log_error(
                "alert_email_failed", "Falha ao enviar email de alerta.", exc_info=e
            )


class ContinuousTrainingManager:
    def __init__(self, logger: StructuredJSONLogger):
        self.config = settings
        self.logger = logger
        self.last_retrain_date = datetime.now()
        self.baseline_performance: Dict[str, float] = {}
        self.new_data_buffer = deque()

    def update_baseline(self, performance: Dict[str, float]):
        self.baseline_performance = {
            "mae": performance.get("mae", 0.0),
            "r2": performance.get("r2", 0.0),
        }
        self.last_retrain_date = datetime.now()
        self.new_data_buffer.clear()
        self.logger.log_event(
            "baseline_performance_updated", new_baseline=self.baseline_performance
        )

    def add_new_data(self, features: np.ndarray, target: float):
        self.new_data_buffer.append({"features": features, "target": target})

    def should_retrain(self, current_performance: Dict[str, float]) -> Tuple[bool, str]:
        reasons = []

        if (datetime.now() - self.last_retrain_date) >= timedelta(
            days=self.config.retrain_frequency_days
        ):
            reasons.append("Tempo")

        if self.baseline_performance:
            mae_baseline = self.baseline_performance.get("mae", float("inf"))
            degradation = (current_performance["mae"] - mae_baseline) / max(
                mae_baseline, 1e-6
            )
            if degradation > self.config.performance_degradation_threshold:
                reasons.append(f"Degradação_MAE({degradation:.1%})")

        if len(self.new_data_buffer) >= self.config.min_new_samples_for_retrain:
            reasons.append("Novos_Dados")

        return len(reasons) > 0, " | ".join(reasons) or "Nenhum critério atendido"
