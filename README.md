# PTIA - Detección de Fraude (Ensemble Híbrido)

Sistema inteligente que combina Random Forest (conocimiento supervisado) e Isolation Forest (razonamiento no supervisado) para detectar fraude.

## Instalación
```bash
pip install -r requirements.txt
```

## Uso
1. Descarga `creditcard.csv` de Kaggle en la carpeta `data/`
2. Entrena los modelos: `python train.py`
3. Evalúa y ajusta pesos/umbral: `python evaluate.py` (genera `reports/` y `models/config.json`)
4. Mide la latencia: `python latency.py`
5. Ejecuta la interfaz: `streamlit run app/streamlit_app.py`
6. Ejecuta la API y el prototipo: `uvicorn api.main:app` y abre http://127.0.0.1:8000

## Metodología de evaluación
- Split estratificado 60 % train / 20 % validación / 20 % test (semilla 42). El test es el mismo del Hito 2.
- El scaler se ajusta solo con train. Pesos y umbral del ensemble se eligen con validación; las métricas finales se calculan una vez sobre test.
- Pesos y umbral viven en `models/config.json` (los lee `FraudEnsemble`); si cambian, hay que redesplegar la API.
- `reports/` contiene métricas (CSV/LaTeX), gráficas PNG y casos de prueba para el documento.

## Prototipo
https://ptia-fraud-detection.onrender.com

## Video Hito 2
https://youtu.be/_z1KGsW4crA?si=h9U9kXff_YFIHGVO
