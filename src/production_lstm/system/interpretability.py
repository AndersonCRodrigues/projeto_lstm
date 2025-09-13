import numpy as np
import shap
import torch
from typing import List, Dict, Any

from production_lstm.config import settings
from production_lstm.models.predictor import LSTMPredictor


class ModelInterpreter:
    def __init__(self, device: str = "cpu"):
        self.config = settings
        self.device = torch.device(device)
        self.explainer = None
        self.feature_names = None

    def initialize_explainer(
        self,
        model: LSTMPredictor,
        background_data: np.ndarray,
        feature_names: List[str],
    ):
        if not self.config.enable_shap_explanations:
            return

        try:
            self.feature_names = feature_names

            if len(background_data.shape) != 3:
                raise ValueError("Background data must have 3D format")

            background_sample = background_data[: self.config.shap_sample_size]
            background_tensor = (
                torch.from_numpy(background_sample).float().to(self.device)
            )

            self.explainer = shap.DeepExplainer(model, background_tensor)

        except Exception as e:
            self.explainer = None

    def explain_prediction(self, input_data: np.ndarray) -> Dict[str, Any]:
        if not self.explainer:
            return {
                "explanation": "Interpretability not available (explainer not initialized)"
            }

        try:
            input_tensor = (
                torch.from_numpy(input_data).unsqueeze(0).float().to(self.device)
            )
            shap_values = self.explainer.shap_values(input_tensor)
            agg_shap_values = np.abs(shap_values[0]).sum(axis=0)

            feature_importance = [
                {"feature": name, "shap_importance": float(importance), "rank": i + 1}
                for i, (name, importance) in enumerate(
                    sorted(
                        zip(self.feature_names, agg_shap_values),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                )
            ]

            return {"feature_importance": feature_importance}

        except Exception as e:
            return {"explanation": f"Error generating SHAP explanation: {str(e)}"}
