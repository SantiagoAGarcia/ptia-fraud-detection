"""Preprocesamiento compartido por entrenamiento, evaluación y API.

Reglas de oro de este módulo:
  * La ingeniería de variables vive en UN solo lugar (`build_features`).
  * El split se hace ANTES de escalar; el scaler se ajusta solo con train
    (evita fuga de datos hacia validación/test).
  * El split de TEST es idéntico al del Hito 2
    (test_size=0.2, random_state=42, stratify=y), así las transacciones de
    `data/muestras.json` siguen siendo datos que el modelo nunca vio.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / 'models'
DATA_PATH = ROOT_DIR / 'data' / 'creditcard.csv'
SEED = 42

BASE_COLS = [f'V{i}' for i in range(1, 29)]
FEATURES = BASE_COLS + ['Hour', 'DayOfWeek', 'LogAmount']   # 31 variables


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Dataset crudo de Kaggle (Time, Amount, V1..V28) -> las 31 variables del modelo."""
    out = df[BASE_COLS].copy()
    out['Hour'] = (df['Time'] // 3600) % 24
    out['DayOfWeek'] = (df['Time'] // 86400) % 7
    out['LogAmount'] = np.log1p(df['Amount'])
    return out[FEATURES]


def load_splits(filepath=DATA_PATH):
    """Devuelve X/y de train (60 %), val (20 %) y test (20 %), SIN escalar.

    train -> entrena RF e IF y ajusta el scaler
    val   -> elegir pesos y umbral del ensemble
    test  -> reporte final (se usa una sola vez)
    """
    df = pd.read_csv(filepath)
    X, y = build_features(df), df['Class']
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, random_state=SEED, stratify=y_tv)
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Ajusta el scaler SOLO con train."""
    return StandardScaler().fit(X_train)


def scale(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(scaler.transform(X[FEATURES]), columns=FEATURES, index=X.index)
