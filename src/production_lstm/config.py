from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator, HttpUrl
from pathlib import Path
from typing import Optional
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"

if not ENV_FILE_PATH.exists():
    ENV_FILE_PATH = PROJECT_ROOT / ".env.example"


class ProductionConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH) if ENV_FILE_PATH.exists() else None,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    alert_email: str = Field(
        default="admin@company.com", description="Email para alertas"
    )
    smtp_server: str = Field(default="smtp.gmail.com", description="Servidor SMTP")
    smtp_port: int = Field(default=587, ge=1, le=65535, description="Porta SMTP")
    smtp_user: str = Field(default="", description="Usuário SMTP")
    smtp_password: str = Field(default="", description="Senha SMTP")

    max_mae_threshold: float = Field(
        default=5.0, gt=0, description="Threshold máximo MAE"
    )
    max_rmse_threshold: float = Field(
        default=8.0, gt=0, description="Threshold máximo RMSE"
    )
    min_r2_threshold: float = Field(
        default=0.7, ge=0, le=1, description="Threshold mínimo R²"
    )
    drift_threshold: float = Field(
        default=0.15, gt=0, le=1, description="Threshold de drift"
    )

    retrain_frequency_days: int = Field(
        default=7, gt=0, description="Frequência de retreinamento em dias"
    )
    min_new_samples_for_retrain: int = Field(
        default=100, gt=0, description="Mínimo de amostras para retreinar"
    )
    performance_degradation_threshold: float = Field(
        default=0.1, gt=0, le=1, description="Threshold de degradação"
    )

    monitoring_window_size: int = Field(
        default=1000, gt=0, description="Tamanho da janela de monitoramento"
    )
    log_level: str = Field(default="INFO", description="Nível de log")

    enable_shap_explanations: bool = Field(
        default=True, description="Habilitar explicações SHAP"
    )
    shap_sample_size: int = Field(
        default=100, gt=0, description="Tamanho da amostra SHAP"
    )

    cost_per_unit_error: float = Field(
        default=10.0, ge=0, description="Custo por erro unitário"
    )
    revenue_per_accurate_prediction: float = Field(
        default=5.0, ge=0, description="Receita por predição precisa"
    )
    inventory_cost_reduction_factor: float = Field(
        default=0.02, ge=0, le=1, description="Fator de redução de custo"
    )

    mlflow_tracking_uri: str = Field(
        default="http://127.0.0.1:5000", description="URI do MLflow tracking server"
    )
    mlflow_experiment_name: str = Field(
        default="Previsao_Demanda_LSTM", description="Nome do experimento MLflow"
    )
    mlflow_model_name: str = Field(
        default="ProductionLSTM-Forecast", description="Nome do modelo no MLflow"
    )

    api_host: str = Field(default="0.0.0.0", description="Host da API")
    api_port: int = Field(default=8000, ge=1, le=65535, description="Porta da API")
    workers: int = Field(default=1, gt=0, description="Número de workers")

    # Timeout configurations
    model_timeout: int = Field(
        default=30, gt=0, description="Timeout do modelo em segundos"
    )
    api_timeout: int = Field(default=60, gt=0, description="Timeout da API em segundos")

    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level deve ser um de: {valid_levels}")
        return v.upper()

    @validator("mlflow_tracking_uri")
    def validate_mlflow_uri(cls, v):
        if not v.startswith(("http://", "https://", "file://", "sqlite://")):
            raise ValueError(
                "mlflow_tracking_uri deve começar com http://, https://, file:// ou sqlite://"
            )
        return v

    @validator("smtp_port", "api_port")
    def validate_ports(cls, v, field):
        if not (1 <= v <= 65535):
            raise ValueError(f"{field.name} deve estar entre 1 e 65535")
        return v

    def is_email_configured(self) -> bool:
        """Verifica se as configurações de email estão válidas"""
        return bool(self.smtp_user and self.smtp_password and "@" in self.alert_email)

    def get_mlflow_config(self) -> dict:
        """Retorna configuração do MLflow"""
        return {
            "tracking_uri": self.mlflow_tracking_uri,
            "experiment_name": self.mlflow_experiment_name,
            "model_name": self.mlflow_model_name,
        }

    def validate_production_readiness(self) -> list[str]:
        """Valida se a configuração está pronta para produção"""
        issues = []

        if self.mlflow_tracking_uri.startswith("http://127.0.0.1"):
            issues.append("MLflow URI usa localhost - não funcionará em produção")

        if not self.is_email_configured():
            issues.append("Configurações de email incompletas")

        if self.log_level == "DEBUG":
            issues.append("Log level DEBUG não recomendado para produção")

        return issues


settings = ProductionConfig()

if __name__ == "__main__":
    print("Configurações carregadas:")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"ENV_FILE_PATH: {ENV_FILE_PATH}")
    print(f"Arquivo .env existe: {ENV_FILE_PATH.exists()}")

    if issues := settings.validate_production_readiness():
        print("\n⚠️  Problemas encontrados:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ Configuração pronta para produção")
