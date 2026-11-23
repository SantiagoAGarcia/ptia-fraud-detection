import json
import joblib
import os
import sklearn
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import f1_score

from src.preprocessing import MODELS_DIR, fit_scaler, scale

DEFAULT_CONFIG = {'w_rf': 0.7, 'w_if': 0.3, 'threshold': 0.85}


class Trainer:
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """X_* llegan SIN escalar; aquí se ajusta el scaler solo con train."""
        os.makedirs(MODELS_DIR, exist_ok=True)

        # 0. Scaler ajustado únicamente con train (sin fuga de datos)
        scaler = fit_scaler(X_train)
        joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
        Xs_train = scale(scaler, X_train)

        # 1. Random Forest con class_weight (evita SMOTE sobre variables PCA)
        print("Entrenando Random Forest (Conocimiento)...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                    class_weight='balanced_subsample',
                                    random_state=42, n_jobs=-1)
        rf.fit(Xs_train, y_train)
        joblib.dump(rf, MODELS_DIR / 'random_forest.pkl')
        if X_val is not None:
            f1 = f1_score(y_val, rf.predict(scale(scaler, X_val)))
            print(f" -> RF F1-Score (validación): {f1:.4f}")

        # 2. Isolation Forest (Razonamiento no supervisado)
        print("Entrenando Isolation Forest...")
        contamination = float(y_train.mean())
        if_model = IsolationForest(n_estimators=100, contamination=contamination,
                                   random_state=42, n_jobs=-1)
        if_model.fit(Xs_train)
        joblib.dump(if_model, MODELS_DIR / 'isolation_forest.pkl')

        # 3. Rango del score del IF para normalizarlo a [0, 1] en inferencia
        print("Calculando rangos de normalización para Isolation Forest...")
        if_scores_train = -if_model.decision_function(Xs_train)
        params = {'if_min': float(if_scores_train.min()),
                  'if_max': float(if_scores_train.max())}
        joblib.dump(params, MODELS_DIR / 'if_normalization.pkl')
        print(f" -> IF Score range: [{params['if_min']:.4f}, {params['if_max']:.4f}]")

        # 4. Pesos y umbral por defecto; `python evaluate.py` los ajusta con validación
        with open(MODELS_DIR / 'config.json', 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        print(f" -> config.json por defecto: {DEFAULT_CONFIG}")

        # 5. Huella del entrenamiento: evaluate.py verifica que el split coincida
        with open(MODELS_DIR / 'train_info.json', 'w') as f:
            json.dump({'n_train': int(len(X_train)),
                       'n_frauds_train': int(y_train.sum()),
                       'sklearn_version': sklearn.__version__}, f, indent=2)
        print(" -> Modelos guardados exitosamente en /models")
