# Sistema de Previsão de Séries Temporais com LSTM

Este projeto implementa um pipeline de MLOps completo e robusto para um modelo de previsão de séries temporais com LSTM. O foco vai além do treinamento de um modelo, abrangendo versionamento, deployment como API, monitoramento e resiliência, seguindo as melhores práticas de engenharia de software.

---

## 📂 Arquitetura do Projeto

O projeto utiliza uma estrutura `src-layout` para garantir que o código seja modular, testável e empacotável.

```

production\_lstm\_project/
├── .env                  # Segredos e configurações de ambiente (local)
├── .env.example          # Template para o arquivo .env
├── Makefile              # Ponto de entrada para automação de tarefas
├── Dockerfile            # Receita para construir a imagem da aplicação
├── requirements.txt      # Dependências do projeto
├── setup.py              # Torna o projeto um pacote Python instalável
├── src/
│   └── production\_lstm/  # Pacote principal da aplicação
│       ├── api/          # Lógica da API com FastAPI
│       ├── training/     # Pipeline de treinamento e versionamento
│       ├── system/       # Módulos de MLOps (monitoramento, orquestração, etc.)
│       ├── data/         # Geração e processamento de dados
│       ├── models/       # Definição do modelo PyTorch
│       └── utils/        # Utilitários (ex: logger)
└── tests/
└── ...               # Testes automatizados

````

---

## ✨ Principais Funcionalidades

- **Pipeline de Treinamento Automatizado**: Script que utiliza Optuna para otimização de hiperparâmetros e MLflow para registrar experimentos e versionar modelos.
- **API de Inferência em Tempo Real**: Serviço de alta performance com FastAPI para servir predições, com carregamento de modelo resiliente (fallback e retentativas) e endpoint de saúde.
- **Geração de Dados Sintéticos Realistas**: Cria séries temporais com complexidades do mundo real, como mudanças de regime, anomalias e dados faltantes, para treinar um modelo mais robusto.
- **Arquitetura de Explicação Assíncrona**: Desacopla o cálculo pesado do SHAP do endpoint de predição, garantindo baixa latência.
- **Containerização com Docker**: Imagem Docker multi-stage otimizada para deployment consistente e portátil.
- **Automação com Makefile**: Comandos simples para gerenciar todo o ciclo de vida do projeto: instalar, testar, treinar e rodar.

---

## 🚀 Guia Rápido de Uso

### 1. Pré-requisitos

- Python 3.9+
- Docker
- Git

### 2. Configuração Inicial

```bash
# 1. Clone o repositório
git clone git@github.com:AndersonCRodrigues/projeto_lstm.git
cd production_lstm_project

# 2. Configure as variáveis de ambiente
cp .env.example .env
nano .env  # Edite com seus próprios valores (ex: credenciais)
````

### 3. Instalação do Ambiente

```bash
make install
```

### 4. Executando o Ciclo de Vida do ML

Você precisará de 2 ou 3 terminais abertos na pasta do projeto.

**Terminal 1: Iniciar o Servidor MLflow**

```bash
make run-mlflow
```

Acesse a interface em [http://127.0.0.1:5000](http://127.0.0.1:5000).

**Terminal 2: Treinar o Modelo**

```bash
make train
```

Após o treinamento, vá até a interface do MLflow, encontre o modelo `ProductionLSTM-Forecast` e promova a última versão para o estágio `Production`.

**Terminal 3: Iniciar a API de Predição**

```bash
make run-api
```

A API estará disponível em [http://localhost:8000](http://localhost:8000). Acesse [http://localhost:8000/docs](http://localhost:8000/docs) para a documentação interativa.

---

### 5. Testando a API

Exemplo de requisição usando `curl`:

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "sequence": [
      {
        "dia_semana_sin": 0.433,
        "dia_semana_cos": -0.9,
        "mes_sin": 0.5,
        "mes_cos": -0.866,
        "media_movel_7": 150.5,
        "media_movel_30": 145.2,
        "desvio_movel_7": 10.3,
        "lag_1": 152.1,
        "lag_7": 148.9,
        "fator_externo": 1.2,
        "evento_especial": 0.0
      }
    ]
  }'
```

> **Nota:** O exemplo acima precisa ser preenchido com 60 objetos FeatureSet para funcionar.

---

## 🧰 Comandos do Makefile

* `make install` : Cria o ambiente virtual e instala dependências.
* `make lint` : Formata e verifica a qualidade do código com Ruff.
* `make test` : Executa os testes automatizados com Pytest.
* `make run-mlflow` : Inicia a interface de usuário do MLflow.
* `make train` : Executa o pipeline de treinamento completo.
* `make run-api` : Inicia a API localmente em modo de desenvolvimento.
* `make docker-build` : Constrói a imagem Docker para a aplicação.
* `make docker-run` : Executa a API dentro de um contêiner Docker.
* `make clean` : Remove arquivos temporários, cache e o ambiente virtual.

---

## 🤝 Contribuição

Contribuições são bem-vindas! Abra uma issue para discutir novas funcionalidades ou envie um pull request com melhorias.

---

## 📄 Licença

Este projeto está licenciado sob a MIT License.

