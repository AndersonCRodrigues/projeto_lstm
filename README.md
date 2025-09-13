# Sistema LSTM de Previsão para Produção

Este projeto implementa um pipeline completo e robusto de MLOps para um modelo de previsão de séries temporais com LSTM, seguindo as melhores práticas de engenharia de software e deployment.

## Funcionalidades
- **Estrutura de Pacote Python**: Código organizado no layout `src/` e instalável via `setup.py`.
- **Modelo LSTM Configurável**: Arquitetura flexível para experimentação.
- **Gestão de Configuração Centralizada**: Uso de `pydantic-settings` para carregar configurações do arquivo `.env`.
- **Pipeline de Treinamento Automatizado**: Script dedicado (`training/pipeline.py`) que inclui otimização de hiperparâmetros (Optuna) e versionamento de modelos (MLflow).
- **API de Inferência em Tempo Real**: Serviço com FastAPI (`api/main.py`) para predições sob demanda, com carregamento de modelo otimizado e health check.
- **Containerização**: `Dockerfile` multi-stage para criar imagens otimizadas e consistentes.
- **Automação com `Makefile`**: Comandos simples para instalar, testar, treinar, rodar a API e construir a imagem Docker.

## Como Usar

Este projeto é gerenciado via `Makefile`.

1.  **Configurar o Ambiente Virtual e Instalar Dependências:**
    ```bash
    make install
    ```

2.  **Configurar Variáveis de Ambiente:**
    - Copie o conteúdo de `.env.example` para um novo arquivo chamado `.env`.
    - Preencha os valores (credenciais de SMTP, etc.).

3.  **Iniciar Serviços Auxiliares:**
    - Para treinar ou rodar a API, você precisa do servidor MLflow. Em um terminal, execute:
    ```bash
    make run-mlflow
    ```

4.  **Treinar o Modelo:**
    - Em outro terminal, execute o pipeline de treinamento:
    ```bash
    make train
    ```
    - O modelo treinado será versionado no MLflow.

5.  **Executar a API Localmente:**
    - Após treinar um modelo e promovê-lo para "Production" no MLflow UI, inicie a API:
    ```bash
    make run-api
    ```
    - A API estará disponível em `http://localhost:8000`.

6.  **Construir e Rodar com Docker:**
    ```bash
    make docker-build
    make docker-run
    ```