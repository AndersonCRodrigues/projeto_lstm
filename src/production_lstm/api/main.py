from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field
import numpy as np
import mlflow
import joblib
import torch
import os
import time
import uuid
from typing import List

from production_lstm.config import settings
from production_lstm.models.predictor import LSTMPredictor


class FeatureSet(BaseModel):
    dia_semana_sin: float = Field(..., ge=-1, le=1)
    dia_semana_cos: float = Field(..., ge=-1, le=1)
    mes_sin: float = Field(..., ge=-1, le=1)
    mes_cos: float = Field(..., ge=-1, le=1)
    media_movel_7: float = Field(..., gt=0)
    media_movel_30: float = Field(..., gt=0)
    desvio_movel_7: float = Field(..., ge=0)
    lag_1: float
    lag_7: float
    fator_externo: float
    evento_especial: float = Field(..., ge=0, le=1)


class PredictionInput(BaseModel):
    sequence: List[FeatureSet] = Field(min_length=60, max_length=60)


class PredictionOutput(BaseModel):
    prediction: float
    trace_id: str


class ExplanationRequestOutput(BaseModel):
    status: str
    explanation_id: str


class HealthStatus(BaseModel):
    status: str
    model_version: str


class ModelManager:
    def __init__(self):
        self.model: LSTMPredictor | None = None
        self.scaler_features = None
        self.scaler_target = None
        self.model_version: str = "N/A"
        self.is_ready = False

    def load_model_with_retries(self, retries=3, delay=5):
        for attempt in range(retries):
            try:
                mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
                for stage in ["Production", "Staging"]:
                    try:
                        return self._extracted_from_load_model_with_retries_7(stage)
                    except mlflow.exceptions.MlflowException:
                        continue

                raise RuntimeError(
                    "Nenhum modelo encontrado nos estágios 'Production' ou 'Staging'."
                )
            except Exception as e:
                print(f"Falha na tentativa {attempt + 1}: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)

        self.is_ready = False
        self.model_version = "Falha no carregamento"
        print(self.model_version)

    # TODO Rename this here and in `load_model_with_retries`
    def _extracted_from_load_model_with_retries_7(self, stage):
        model_uri = f"models:/{settings.mlflow_model_name}/{stage}"
        model_info = mlflow.models.get_model_info(model_uri)

        self.model = mlflow.pytorch.load_model(model_uri)
        self.model.eval()
        self.model_version = f"{stage} v{model_info.version}"

        run_id = model_info.run_id
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="processors"
        )
        self.scaler_target = joblib.load(
            os.path.join(local_path, "scaler_target.pkl")
        )
        self.scaler_features = joblib.load(
            os.path.join(local_path, "scaler_features.pkl")
        )

        self.is_ready = True
        print(f"Modelo {self.model_version} carregado.")
        return


model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    if not model_manager.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo não está pronto.",
        )
    return model_manager


app = FastAPI(title="API de Previsão LSTM (Robusta)", version="2.0.0")


@app.on_event("startup")
def startup_event():
    model_manager.load_model_with_retries()


@app.get("/health", response_model=HealthStatus)
def health_check():
    return {
        "status": "ok" if model_manager.is_ready else "unhealthy",
        "model_version": model_manager.model_version,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput, manager: ModelManager = Depends(get_model_manager)):
    trace_id = str(uuid.uuid4())
    try:
        feature_order = list(FeatureSet.model_fields.keys())
        feature_array = np.array(
            [[getattr(step, f) for f in feature_order] for step in data.sequence]
        )

        features_scaled = manager.scaler_features.transform(feature_array)
        input_tensor = torch.FloatTensor(features_scaled).unsqueeze(0)

        with torch.no_grad():
            prediction_scaled = manager.model(input_tensor).cpu().numpy().item()

        prediction = manager.scaler_target.inverse_transform([[prediction_scaled]])[0][
            0
        ]

        return {"prediction": float(prediction), "trace_id": trace_id}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro na predição (Trace ID: {trace_id}): {e}",
        ) from e


@app.post("/request-explanation", response_model=ExplanationRequestOutput)
def request_explanation(data: PredictionInput, background_tasks: BackgroundTasks):
    explanation_id = str(uuid.uuid4())
    print(f"Adicionando tarefa de explicação em segundo plano: {explanation_id}")
    return {"status": "explanation_queued", "explanation_id": explanation_id}
