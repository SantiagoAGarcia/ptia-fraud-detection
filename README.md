# PTIA - Detección de Fraude (Ensemble Híbrido)

Sistema inteligente que combina Random Forest (conocimiento supervisado) e Isolation Forest (razonamiento no supervisado) para detectar fraude.

## Instalación
```bash
pip install -r requirements.txt
```

## Uso
1. Descarga `creditcard.csv` de Kaggle en la carpeta `data/`
2. Entrena los modelos: `python train.py`
3. Ejecuta la interfaz: `streamlit run app/streamlit_app.py`
4. Ejecuta la API y el prototipo: `uvicorn api.main:app` y abre http://127.0.0.1:8000

## Video Hito 2
https://youtu.be/_z1KGsW4crA?si=h9U9kXff_YFIHGVO
