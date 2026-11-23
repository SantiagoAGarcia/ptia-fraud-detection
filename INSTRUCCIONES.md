# Parte de Santiago — cómo aplicar los cambios

## 1. Copiar los archivos sobre tu repo (misma estructura de carpetas)
.gitignore · README.md · requirements.txt · train.py · evaluate.py · latency.py ·
src/preprocessing.py · src/model_training.py · src/ensemble.py

Antes, trae lo último de Nicolás:  `git pull`
Luego quita los .pyc del repo:      `git rm -r --cached src/__pycache__`

## 2. Correr el pipeline (con data/creditcard.csv de Kaggle en la carpeta data/)
```bash
pip install -r requirements.txt     # scikit-learn queda fijo en 1.9.0 (el de Nicolás)
python train.py                     # reentrena con split 60/20/20, scaler solo con train
python evaluate.py                  # ajusta pesos/umbral en validación + reportes en reports/
python latency.py                   # latencia por transacción
```
- `train.py` SOBRESCRIBE models/*.pkl (ahora entrenados sin fuga de datos).
- `evaluate.py` escribe models/config.json con pesos y umbral elegidos
  (usa `--no-save-config` para no tocarlo, `--objective f2` para priorizar Recall).
- Si `evaluate.py` dice que los modelos no coinciden con el split, falta correr train.py.

## 3. Revisar antes de subir
- `reports/resumen.md`: pesos elegidos, tabla de métricas, veredicto del «30 % menos falsos positivos».
  Si NO se cumple, hay que cambiar esa promesa en el documento (no forzar los números).
- `reports/casos_de_prueba.md`: los 3 casos para la sección 7 + 2 errores para la sección 8.

## 4. Subir y avisar
```bash
git add -A && git commit -m "Evaluación, split sin fuga de datos, reportes y limpieza" && git push
```
- Nicolás: redesplegar en Render (cambian los .pkl y models/config.json) y actualizar en
  prototipo/index.html los textos fijos «× 0,7», «× 0,3», «70 % / 30 %»: ahora /predict
  devuelve `w_rf` y `w_if` (y `umbral`), que puede leer de ahí.
- Felipe: carpeta `reports/` (metricas.tex, matriz_confusion.tex, PNG, casos de prueba, latencia.json).

## Qué cambió en el código (sin romper la API)
- `FraudEnsemble.predict(dict)` conserva firma y claves; ahora también devuelve w_rf y w_if.
  Verificado: con los .pkl actuales da exactamente los mismos scores que el código anterior.
- El split de TEST es idéntico al del Hito 2, así data/muestras.json sigue siendo test real.
- Pesos y umbral salen de models/config.json (por defecto 0.7 / 0.3 / 0.85).
