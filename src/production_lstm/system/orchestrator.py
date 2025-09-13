import numpy as np
import json
from datetime import datetime
from collections import deque
from threading import Lock
from typing import Dict, Any, Tuple, List, Optional
import torch

from production_lstm.config import settings
from production_lstm.utils.logger import StructuredJSONLogger
from production_lstm.system.monitoring import AlertManager, ContinuousTrainingManager
from production_lstm.system.business import BusinessMetricsCalculator
from production_lstm.system.interpretability import ModelInterpreter
from production_lstm.models.predictor import LSTMPredictor


class ProductionLSTMSystem:
    def __init__(self, device: str = "cpu"):
        self.config = settings
        self.device = torch.device(device)
        self.logger = StructuredJSONLogger(log_level=self.config.log_level)
        self.alert_manager = AlertManager(self.logger)
        self.training_manager = ContinuousTrainingManager(self.logger)
        self.business_metrics = BusinessMetricsCalculator()
        self.interpreter = ModelInterpreter(device=device)

        self.model: Optional[LSTMPredictor] = None
        self.scaler_target = None
        self.scaler_features = None
        self.feature_names: Optional[List[str]] = None

        self.buffer_lock = Lock()
        self.prediction_buffer = deque(maxlen=self.config.monitoring_window_size)
        self.reference_data_features: Optional[np.ndarray] = None

    def initialize_model(
        self,
        model: LSTMPredictor,
        scalers: Tuple,
        feature_names: List[str],
        reference_data_train: np.ndarray,
    ):
        self.model = model.to(self.device)
        self.model.eval()
        self.scaler_target, self.scaler_features = scalers
        self.feature_names = feature_names

        num_samples, seq_len, num_features = reference_data_train.shape
        self.reference_data_features = reference_data_train.reshape(-1, num_features)

        self.interpreter.initialize_explainer(
            model, reference_data_train, feature_names
        )
        self.training_manager.update_baseline({"mae": 0, "r2": 1})
        self.logger.log_event("production_system_initialized")

    def predict_with_monitoring(
        self,
        features: np.ndarray,
        actual_value: Optional[float] = None,
        volume: float = 1.0,
        explain: bool = False,
        trace_id: str = "",
    ) -> Dict[str, Any]:
        if not self.model or not self.feature_names:
            raise RuntimeError("Sistema não foi inicializado.")

        with torch.no_grad():
            features_scaled = self.scaler_features.transform(features)
            input_tensor = (
                torch.FloatTensor(features_scaled).unsqueeze(0).to(self.device)
            )
            prediction_scaled = self.model(input_tensor).cpu().numpy().item()
            prediction = self.scaler_target.inverse_transform([[prediction_scaled]])[0][
                0
            ]

        result = {"prediction": float(prediction)}
        if explain:
            result["explanation"] = self.interpreter.explain_prediction(features)

        with self.buffer_lock:
            self.prediction_buffer.append(
                {
                    "prediction": float(prediction),
                    "actual": actual_value,
                    "features": features,
                    "volume": volume,
                    "timestamp": datetime.now(),
                }
            )

        self.logger.log_prediction(
            prediction=float(prediction),
            actual=actual_value,
            features={
                name: float(val)
                for name, val in zip(self.feature_names, features[-1, :])
            },
            metadata={"volume": volume},
            trace_id=trace_id,
        )

        if actual_value is not None:
            self.training_manager.add_new_data(features, actual_value)

        return result

    def run_monitoring_checks(self) -> Optional[Dict[str, Any]]:
        with self.buffer_lock:
            if len(self.prediction_buffer) < 50:
                return None
            buffer_copy = list(self.prediction_buffer)

        valid_data = [d for d in buffer_copy if d["actual"] is not None]
        if len(valid_data) < 20:
            return None

        predictions = np.array([d["prediction"] for d in valid_data])
        actuals = np.array([d["actual"] for d in valid_data])
        volumes = np.array([d["volume"] for d in valid_data])

        results: Dict[str, Any] = {"timestamp": datetime.now().isoformat()}

        performance = self.alert_manager.check_model_performance(predictions, actuals)
        results["performance"] = performance

        if self.reference_data_features is not None:
            recent_features = np.array([d["features"] for d in buffer_copy]).reshape(
                -1, self.reference_data_features.shape[1]
            )
            results["data_drift"] = self.alert_manager.check_data_drift(
                recent_features, self.reference_data_features
            )

        results["business_metrics"] = self.business_metrics.calculate_business_impact(
            predictions, actuals, volumes
        )
        should_retrain, reason = self.training_manager.should_retrain(performance)
        results["retrain_recommendation"] = {
            "should_retrain": should_retrain,
            "reason": reason,
        }

        self.logger.log_event("monitoring_check_completed", monitoring_results=results)
        return results

    def generate_monitoring_dashboard(
        self, monitoring_results: Optional[Dict[str, Any]]
    ) -> str:
        if not monitoring_results:
            return "DASHBOARD DE MONITORAMENTO: Dados insuficientes para gerar o relatório."

        header = f"DASHBOARD DE MONITORAMENTO - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        perf = monitoring_results.get("performance", {})
        drift = monitoring_results.get("data_drift", {})
        retrain = monitoring_results.get("retrain_recommendation", {})
        business_metrics = monitoring_results.get("business_metrics", {})

        perf_report = f"""
        PERFORMANCE DO MODELO:
        - MAE:  {perf.get('mae', 0):.4f} (Alerta > {self.config.max_mae_threshold})
        - R²:   {perf.get('r2', 0):.4f} (Alerta < {self.config.min_r2_threshold})
        - Alertas: {len(perf.get('alerts_triggered', []))}"""

        drift_report = f"""
        DATA DRIFT:
        - Drift Detectado: {'Sim' if drift.get('drift_detected') else 'Não'}
        - Alertas: {len(drift.get('alerts_triggered', []))}"""

        retrain_report = f"""
        RETREINAMENTO:
        - Recomendação: {'SIM' if retrain.get('should_retrain') else 'NÃO'}
        - Motivo: {retrain.get('reason')}"""

        business_report = self.business_metrics.generate_business_report(
            business_metrics
        )

        return "\n\n".join(
            [header, business_report, perf_report, drift_report, retrain_report]
        )
