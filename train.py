from src.preprocessing import Preprocessor
from src.model_training import Trainer

print("🚀 Iniciando pipeline de entrenamiento del Ensemble...")
try:
    prep = Preprocessor()
    X_train, X_test, y_train, y_test = prep.process('data/creditcard.csv')

    trainer = Trainer()
    trainer.train(X_train, y_train, X_test, y_test)
    print("✅ ¡Entrenamiento completado exitosamente!")
except FileNotFoundError:
    print("❌ ERROR: No se encontró 'data/creditcard.csv'. Descárgalo de Kaggle y ponlo en la carpeta data/ primero.")
