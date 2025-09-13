from pydantic_settings import BaseSettings, SettingsConfigDict


class ProductionConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    alert_email: str = "admin@company.com"
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""

    max_mae_threshold: float = 5.0
    max_rmse_threshold: float = 8.0
    min_r2_threshold: float = 0.7
    drift_threshold: float = 0.15

    retrain_frequency_days: int = 7
    min_new_samples_for_retrain: int = 100
    performance_degradation_threshold: float = 0.1

    monitoring_window_size: int = 1000
    log_level: str = "INFO"

    enable_shap_explanations: bool = True
    shap_sample_size: int = 100

    cost_per_unit_error: float = 10.0
    revenue_per_accurate_prediction: float = 5.0
    inventory_cost_reduction_factor: float = 0.02

    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    mlflow_experiment_name: str = "Previsao_Demanda_LSTM"
    mlflow_model_name: str = "ProductionLSTM-Forecast"


settings = ProductionConfig()
