import os
from production_lstm.config import ProductionConfig


def test_config_loading_from_env():
    """Testa se a configuração Pydantic carrega corretamente as variáveis de ambiente."""
    # Define uma variável de ambiente temporária para o teste
    os.environ["ALERT_EMAIL"] = "test@example.com"

    # Cria uma nova instância para forçar a releitura das variáveis
    settings = ProductionConfig()

    assert settings.alert_email == "test@example.com"

    # Limpa a variável de ambiente
    del os.environ["ALERT_EMAIL"]
