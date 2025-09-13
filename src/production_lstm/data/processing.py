import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
    ruido1 = np.random.normal(0, 2, vol_break)
    ruido2 = np.random.normal(0, 4, n_pontos - vol_break)
    ruido = np.concatenate([ruido1, ruido2])

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

    df = df.drop(columns=["data"])

    return df


def preprocessar_dados_robusto(
    df: pd.DataFrame,
    target_col: str = "valor_principal",
    seq_length: int = 60,
    val_size: float = 0.15,
    test_size: float = 0.15,
    scaler_type: str = "standard",
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Any,
    Any,
    List[str],
]:
    feature_cols = [col for col in df.columns if col != target_col]

    imputer = SimpleImputer(strategy="mean")
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    scaler_target = StandardScaler()
    scaler_features = StandardScaler()

    target_scaled = scaler_target.fit_transform(df_imputed[[target_col]]).flatten()
    features_scaled = scaler_features.fit_transform(df_imputed[feature_cols])

    X, y = [], []
    for i in range(len(df_imputed) - seq_length):
        X.append(features_scaled[i : i + seq_length])
        y.append(target_scaled[i + seq_length])

    X, y = np.array(X), np.array(y)

    n_total = len(X)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)
    n_train = n_total - n_test - n_val

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        scaler_target,
        scaler_features,
        feature_cols,
    )
