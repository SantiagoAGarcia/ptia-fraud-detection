import json
import joblib
import numpy as np
import pandas as pd

from src.preprocessing import MODELS_DIR

DEFAULT_CONFIG = {'w_rf': 0.7, 'w_if': 0.3, 'threshold': 0.85}


class FraudEnsemble:
    """Ensemble ponderado RF (supervisado) + IF (no supervisado).

    Pesos y umbral se leen de models/config.json (los escribe evaluate.py);
    si no existe se usan los valores por defecto 0.7 / 0.3 / 0.85.
    """

    def __init__(self, models_dir=MODELS_DIR):
        self.rf = joblib.load(models_dir / 'random_forest.pkl')
        self.if_model = joblib.load(models_dir / 'isolation_forest.pkl')
        self.scaler = joblib.load(models_dir / 'scaler.pkl')
        self.features = list(self.rf.feature_names_in_)

        # Una sola transacción por llamada: paralelizar solo agrega latencia
        self.rf.n_jobs = 1
        self.if_model.n_jobs = 1

        norm = joblib.load(models_dir / 'if_normalization.pkl')
        self.if_min, self.if_max = norm['if_min'], norm['if_max']

        cfg = dict(DEFAULT_CONFIG)
        cfg_path = models_dir / 'config.json'
        if cfg_path.exists():
            cfg.update(json.load(open(cfg_path)))
        self.w_rf, self.w_if, self.thresh = cfg['w_rf'], cfg['w_if'], cfg['threshold']

    # --- scores por componente (sirve para una fila o para todo un lote) -------------
    def component_scores(self, df: pd.DataFrame):
        """df con las 31 variables SIN escalar -> (score_rf, score_if) en [0, 1]."""
        X = pd.DataFrame(self.scaler.transform(df[self.features]),
                         columns=self.features, index=df.index)
        score_rf = self.rf.predict_proba(X)[:, 1]
        raw_if = -self.if_model.decision_function(X)
        score_if = np.clip((raw_if - self.if_min) / (self.if_max - self.if_min), 0, 1)
        return score_rf, score_if

    def combine(self, score_rf, score_if, w_rf=None, w_if=None):
        w_rf = self.w_rf if w_rf is None else w_rf
        w_if = self.w_if if w_if is None else w_if
        return w_rf * np.asarray(score_rf) + w_if * np.asarray(score_if)

    # --- API pública que usan Streamlit y FastAPI (misma firma de siempre) -----------
    def predict(self, transaction_dict):
        df = pd.DataFrame([transaction_dict])
        score_rf, score_if = self.component_scores(df)
        final_score = float(self.combine(score_rf, score_if)[0])
        return {
            'prediction': 'FRAUDE' if final_score > self.thresh else 'LEGITIMA',
            'final_score': final_score,
            'rf_score': float(score_rf[0]),
            'if_score': float(score_if[0]),
            'w_rf': self.w_rf,
            'w_if': self.w_if,
        }
