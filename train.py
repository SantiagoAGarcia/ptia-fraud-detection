from src.preprocessing import load_splits
from src.model_training import Trainer

print("🚀 Iniciando pipeline de entrenamiento del Ensemble...")
try:
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()
    print(f" -> train={len(X_train)}  val={len(X_val)}  test={len(X_test)} "
          f"(fraudes: {int(y_train.sum())}/{int(y_val.sum())}/{int(y_test.sum())})")

    Trainer().train(X_train, y_train, X_val, y_val)
    print("✅ ¡Entrenamiento completado exitosamente!")
    print("👉 Siguiente paso: python evaluate.py  (ajusta pesos/umbral y genera reportes)")
except FileNotFoundError:
    print("❌ ERROR: No se encontró 'data/creditcard.csv'. Descárgalo de Kaggle y ponlo en la carpeta data/ primero.")
