import joblib, pandas as pd, numpy as np

class FraudEnsemble:
    def __init__(self):
        self.rf = joblib.load('models/random_forest.pkl')
        self.if_model = joblib.load('models/isolation_forest.pkl')
        self.scaler = joblib.load('models/scaler.pkl')
        
        # Cargar parámetros de normalización del IF
        norm_params = joblib.load('models/if_normalization.pkl')
        self.if_min = norm_params['if_min']
        self.if_max = norm_params['if_max']
        
        self.w_rf, self.w_if, self.thresh = 0.7, 0.3, 0.85

    def predict(self, transaction_dict):
        df = pd.DataFrame([transaction_dict])
        X = df[self.rf.feature_names_in_]
        X_scaled = self.scaler.transform(X)
        
        # 1. Conocimiento (Supervisado)
        score_rf = self.rf.predict_proba(X_scaled)[0, 1]
        
        # 2. Razonamiento (No Supervisado) - NORMALIZACIÓN CORRECTA
        score_if_raw = -self.if_model.decision_function(X_scaled)[0]
        score_if = (score_if_raw - self.if_min) / (self.if_max - self.if_min)
        score_if = min(max(score_if, 0), 1)  # Asegurar que esté en [0, 1]
        
        # 3. Fusión del Ensemble
        final_score = (self.w_rf * score_rf) + (self.w_if * score_if)
        
        return {
            'prediction': 'FRAUDE' if final_score > self.thresh else 'LEGITIMA',
            'final_score': float(final_score),
            'rf_score': float(score_rf),
            'if_score': float(score_if)
        }