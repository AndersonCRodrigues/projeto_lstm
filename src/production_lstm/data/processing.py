import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import List, Tuple, Any


class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def gerar_serie_multivariada_realista(n_pontos: int = 3000) -> pd.DataFrame:
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_pontos)]
    t = np.arange(n_pontos)

    break_point = int(n_pontos * 0.6)
    tendencia = np.concatenate(
        [
            0.01 * t[:break_point] ** 1.2,
            0.01 * t[break_point - 1] ** 1.2
            + 0.2 * (t[break_point:] - t[break_point - 1]),
        ]
    )

    sazonalidade = 15 * np.sin(2 * np.pi * t / 365.25) + 5 * np.sin(2 * np.pi * t / 7)

    vol_break = int(n_pontos * 0.75)
    ruido = np.concatenate(
        [
            np.random.normal(0, 2, vol_break),
            np.random.normal(0, 4, n_pontos - vol_break),
        ]
    )

    serie_principal = tendencia + sazonalidade + ruido + 50

    for _ in range(int(n_pontos * 0.01)):
        idx = np.random.randint(0, n_pontos)
        serie_principal[idx] *= np.random.choice([0.5, 1.5, 2.0])

    df = pd.DataFrame({"data": dates, "valor_principal": serie_principal})

    for _ in range(5):
        start_idx = np.random.randint(0, n_pontos - 20)
        end_idx = start_idx + np.random.randint(5, 15)
        df.loc[start_idx:end_idx, "valor_principal"] = np.nan

    day_of_week = df["data"].dt.dayofweek
    month = df["data"].dt.month
    df["dia_semana_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    df["dia_semana_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    df["mes_sin"] = np.sin(2 * np.pi * month / 12)
    df["mes_cos"] = np.cos(2 * np.pi * month / 12)

    df["media_movel_7"] = df["valor_principal"].rolling(window=7).mean()
    df["media_movel_30"] = df["valor_principal"].rolling(window=30).mean()
    df["desvio_movel_7"] = df["valor_principal"].rolling(window=7).std()
    df["lag_1"] = df["valor_principal"].shift(1)
    df["lag_7"] = df["valor_principal"].shift(7)

    df["fator_externo"] = 3 * np.cos(0.05 * t) + np.random.normal(0, 0.5, n_pontos)
    df["evento_especial"] = np.random.choice([0, 1], n_pontos, p=[0.97, 0.03])

    return df.drop(columns=["data"])
