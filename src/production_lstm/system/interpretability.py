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
                raise ValueError("Dados de fundo para SHAP devem ter formato 3D.")

            background_sample = background_data[: self.config.shap_sample_size]
            background_tensor = (
                torch.from_numpy(background_sample).float().to(self.device)
            )

            self.explainer = shap.DeepExplainer(model, background_tensor)
        except Exception as e:
            print(f"Erro ao inicializar SHAP: {e}")
            self.explainer = None

    def explain_prediction(self, input_data: np.ndarray) -> Dict[str, Any]:
        if not self.explainer:
            return {"explanation": "Explicador SHAP não inicializado."}

        try:
            return self._extracted_from_explain_prediction_6(input_data)
        except Exception as e:
            return {"explanation": f"Erro na geração da explicação SHAP: {str(e)}"}

    # TODO Rename this here and in `explain_prediction`
    def _extracted_from_explain_prediction_6(self, input_data):
        input_tensor = torch.from_numpy(input_data).unsqueeze(0).float().to(self.device)
        shap_values = self.explainer.shap_values(input_tensor)
        agg_shap_values = np.abs(shap_values[0]).sum(axis=0)

        feature_importance = [
            {"feature": name, "shap_importance": float(importance)}
            for name, importance in zip(self.feature_names, agg_shap_values)
        ]

        feature_importance.sort(key=lambda x: x["shap_importance"], reverse=True)
        for i, item in enumerate(feature_importance):
            item["rank"] = i + 1

        return {"feature_importance": feature_importance}
