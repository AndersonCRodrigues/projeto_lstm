import requests

api_url = "http://127.0.0.1:8000/predict"

sample_step = {
    "dia_semana_sin": 0.433,
    "dia_semana_cos": -0.900,
    "mes_sin": 0.5,
    "mes_cos": -0.866,
    "media_movel_7": 150.5,
    "media_movel_30": 145.2,
    "desvio_movel_7": 10.3,
    "lag_1": 152.1,
    "lag_7": 148.9,
    "fator_externo": 1.2,
    "evento_especial": 0.0,
}

payload = {"sequence": [sample_step] * 60}

response = requests.post(api_url, json=payload)

if response.ok:
    print(f"Predição recebida do modelo: {response.json()['prediction']:.2f}")
else:
    print(f"Erro {response.status_code}: {response.text}")
