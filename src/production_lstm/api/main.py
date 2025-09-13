from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel, Field
import numpy as np
import mlflow
import joblib
import torch
import os
from typing import List

from production_lstm.config import settings
from production_lstm.models.predictor import LSTMPredictor


class FeatureSet(BaseModel):
    dia_semana_sin: float
    dia_semana_cos: float
    mes_sin: float
    mes_cos: float
    media_movel_7: float
    media_movel_30: float
    desvio_movel_7: float
    lag_1: float
    lag_7: float
    fator_externo: float
    evento_especial: float


class PredictionInput(BaseModel):
    sequence: List[FeatureSet] = Field(min_length=60, max_length=60)


class PredictionOutput(BaseModel):
    prediction: float


class HealthStatus(BaseModel):
    status: str
    model_version: str


class ModelManager:
    def __init__(self):
        self.model: LSTMPredictor | None = None
        self.scaler_features = None
        self.scaler_target = None
        self.model_version: str = "N/A"
        self._feature_order = list(FeatureSet.model_fields.keys())

    def load_model(self):
        try:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            model_uri = f"models:/{settings.mlflow_model_name}/Production"

            model_info = mlflow.models.get_model_info(model_uri)
            self.model_version = model_info.version
            run_id = model_info.run_id

            self.model = mlflow.pytorch.load_model(model_uri)
            self.model.eval()

            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="processors"
            )

            self.scaler_target = joblib.load(
                os.path.join(local_path, "scaler_target.pkl")
            )
            self.scaler_features = joblib.load(
                os.path.join(local_path, "scaler_features.pkl")
            )

        except Exception as e:
            self.model = None
            self.model_version = "Load Failed"
            raise RuntimeError(f"Failed to load model: {e}")

    def predict(self, data: PredictionInput) -> float:
        feature_array = np.array(
            [
                [getattr(step, feature) for feature in self._feature_order]
                for step in data.sequence
            ]
        )

        features_scaled = self.scaler_features.transform(feature_array)
        input_tensor = torch.FloatTensor(features_scaled).unsqueeze(0)

        with torch.no_grad():
            prediction_scaled = self.model(input_tensor).cpu().numpy().item()

        prediction = self.scaler_target.inverse_transform([[prediction_scaled]])[0][0]
        return float(prediction)


model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    if not model_manager.model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded or failed to initialize",
        )
    return model_manager


app = FastAPI(
    title="LSTM Prediction API",
    description="Time series prediction service using trained LSTM model",
    version="2.0.0",
)


@app.on_event("startup")
def startup_event():
    try:
        model_manager.load_model()
    except RuntimeError:
        pass


@app.get("/health", response_model=HealthStatus)
def health_check():
    return {
        "status": "ok" if model_manager.model else "unhealthy",
        "model_version": model_manager.model_version,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput, manager: ModelManager = Depends(get_model_manager)):
    try:
        prediction = manager.predict(data)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {e}",
        )
