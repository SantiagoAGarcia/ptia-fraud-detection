import joblib, os, numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, f1_score

class Trainer:
    def train(self, X_train, y_train, X_test, y_test):
        os.makedirs('models', exist_ok=True)
        
        # 1. Random Forest con class_weight (EVITA SMOTE en variables PCA)
        print("Entrenando Random Forest (Conocimiento)...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced_subsample', random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        joblib.dump(rf, 'models/random_forest.pkl')
        print(f" -> RF F1-Score: {f1_score(y_test, rf.predict(X_test)):.4f}")
        
        # 2. Isolation Forest (Razonamiento no supervisado)
        print("Entrenando Isolation Forest...")
        contamination = float(y_train.mean())
        if_model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42, n_jobs=-1)
        if_model.fit(X_train)
        joblib.dump(if_model, 'models/isolation_forest.pkl')
        
        # 3. CALCULAR Y GUARDAR VALORES MÍN/MÁX DEL IF PARA NORMALIZACIÓN CORRECTA
        print("Calculando rangos de normalización para Isolation Forest...")
        if_scores_train = -if_model.decision_function(X_train)
        if_min = float(if_scores_train.min())
        if_max = float(if_scores_train.max())
        
        # Guardar estos valores para usarlos en inferencia
        normalization_params = {'if_min': if_min, 'if_max': if_max}
        joblib.dump(normalization_params, 'models/if_normalization.pkl')
        
        print(f" -> IF Score range: [{if_min:.4f}, {if_max:.4f}]")
        print(" -> Modelos guardados exitosamente en /models")