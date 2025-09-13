import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import optuna
import mlflow
import joblib
from pathlib import Path

from production_lstm.config import settings
from production_lstm.data.processing import (
    gerar_serie_multivariada_realista,
    preprocessar_dados_robusto,
    TimeSeriesDataset,
)
from production_lstm.models.predictor import LSTMPredictor
from production_lstm.utils.logger import StructuredJSONLogger


class ModelTrainer:
    def __init__(
        self, model: LSTMPredictor, training_params: dict, device: str = "cpu"
    ):
        self.model = model.to(device)
        self.training_params = training_params
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_params.get("learning_rate", 0.001),
            weight_decay=training_params.get("weight_decay", 1e-5),
        )
        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min", patience=10, factor=0.5
        )

    def _evaluate_loss(self, data_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs.squeeze(), y_batch)
                total_loss += loss.item()
        return total_loss / len(data_loader)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        trial: optuna.Trial | None = None,
    ) -> float:
        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        max_patience = 25

        for epoch in range(epochs):
            self.model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs.squeeze(), y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            avg_val_loss = self._evaluate_loss(val_loader)
            self.scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if trial:
                trial.report(avg_val_loss, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            if patience_counter >= max_patience:
                break

        if best_model_state:
            self.model.load_state_dict(best_model_state)
        return best_val_loss

    def get_final_metrics(self, data_loader: DataLoader, scaler_target) -> dict:
        self.model.eval()
        predictions_scaled, targets_scaled = [], []
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                outputs = self.model(X_batch.to(self.device))
                predictions_scaled.extend(outputs.squeeze().cpu().numpy())
                targets_scaled.extend(y_batch.cpu().numpy())

        predictions = scaler_target.inverse_transform(
            np.array(predictions_scaled).reshape(-1, 1)
        ).flatten()
        targets = scaler_target.inverse_transform(
            np.array(targets_scaled).reshape(-1, 1)
        ).flatten()

        mae = np.mean(np.abs(targets - predictions))
        r2 = 1 - (
            np.sum((targets - predictions) ** 2)
            / np.sum((targets - np.mean(targets)) ** 2)
        )
        return {"test_mae": mae, "test_r2": r2}


def run_training_pipeline(optimize: bool = True, n_trials: int = 30):
    logger = StructuredJSONLogger(settings.log_level)
    logger.log_event("training_pipeline_started", optimize=optimize, n_trials=n_trials)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = gerar_serie_multivariada_realista()
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        scaler_target,
        scaler_features,
        feature_cols,
    ) = preprocessar_dados_robusto(df)
    logger.log_event(
        "data_processing_completed",
        train_size=len(X_train),
        val_size=len(X_val),
        test_size=len(X_test),
    )

    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)

    def objective(trial: optuna.Trial) -> float:
        model_params = {
            "input_size": X_train.shape[2],
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
            "num_lstm_layers": trial.suggest_int("num_lstm_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "use_bidirectional": trial.suggest_categorical(
                "use_bidirectional", [True, False]
            ),
        }
        training_params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
        }
        model = LSTMPredictor(**model_params)
        trainer = ModelTrainer(model, training_params, device)
        train_loader = DataLoader(
            train_dataset, batch_size=training_params["batch_size"], shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=training_params["batch_size"])
        return trainer.train(train_loader, val_loader, epochs=75, trial=trial)

    if optimize:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        best_params = study.best_params
        logger.log_event(
            "hyperparameter_optimization_completed", best_params=best_params
        )
    else:
        best_params = {
            "hidden_size": 64,
            "num_lstm_layers": 2,
            "dropout": 0.3,
            "use_bidirectional": False,
            "learning_rate": 0.001,
            "batch_size": 64,
        }

    final_model_params = {
        k: v
        for k, v in best_params.items()
        if k in ["hidden_size", "num_lstm_layers", "dropout", "use_bidirectional"]
    }
    final_model_params["input_size"] = X_train.shape[2]
    final_training_params = {
        k: v for k, v in best_params.items() if k in ["learning_rate", "batch_size"]
    }

    final_model = LSTMPredictor(**final_model_params)
    final_trainer = ModelTrainer(final_model, final_training_params, device)

    full_train_dataset = TimeSeriesDataset(
        np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val])
    )
    full_train_loader = DataLoader(
        full_train_dataset,
        batch_size=final_training_params["batch_size"],
        shuffle=True,
    )
    test_loader = DataLoader(
        TimeSeriesDataset(X_test, y_test),
        batch_size=final_training_params["batch_size"],
    )

    final_trainer.train(full_train_loader, test_loader, epochs=150)
    final_metrics = final_trainer.get_final_metrics(test_loader, scaler_target)
    logger.log_event("final_model_training_completed", final_metrics=final_metrics)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    with mlflow.start_run() as run:
        mlflow.log_params(final_model_params)
        mlflow.log_params(final_training_params)
        mlflow.log_metrics(final_metrics)

        artifact_path = Path("artifacts")
        artifact_path.mkdir(exist_ok=True)
        joblib.dump(scaler_target, artifact_path / "scaler_target.pkl")
        joblib.dump(scaler_features, artifact_path / "scaler_features.pkl")

        mlflow.pytorch.log_model(final_trainer.model, "model", signature=False)
        mlflow.log_artifacts(artifact_path, artifact_path="processors")

        mlflow.register_model(
            f"runs:/{run.info.run_id}/model", settings.mlflow_model_name
        )

    logger.log_event("mlflow_versioning_completed", run_id=run.info.run_id)
    logger.log_event("training_pipeline_finished")


if __name__ == "__main__":
    logger = StructuredJSONLogger()
    try:
        run_training_pipeline(optimize=True, n_trials=15)
    except Exception as e:
        logger.log_error(
            "pipeline_failed",
            "O pipeline de treinamento falhou catastroficamente.",
            exc_info=e,
        )
        raise e
