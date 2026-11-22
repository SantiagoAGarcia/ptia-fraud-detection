import json, random, time, sys, os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ensemble import FraudEnsemble

app = FastAPI(title="PTIA - Detección de Fraude")
ensemble = FraudEnsemble()
muestras = json.load(open('data/muestras.json'))


@app.get("/")
def prototipo():
    return FileResponse('prototipo/index.html')


@app.get("/muestra")
def muestra_aleatoria(clase: int | None = None):
    pool = [m for m in muestras if clase is None or m['clase'] == clase]
    return random.choice(pool)


@app.get("/muestra/{i}")
def muestra(i: int):
    if not 0 <= i < len(muestras):
        raise HTTPException(404, "Muestra no encontrada")
    return muestras[i]


@app.post("/predict")
def predict(transaccion: dict[str, float]):
    faltantes = [f for f in ensemble.rf.feature_names_in_ if f not in transaccion]
    if faltantes:
        raise HTTPException(422, f"Faltan variables: {', '.join(faltantes)}")
    inicio = time.perf_counter()
    resultado = ensemble.predict(transaccion)
    resultado['latencia_ms'] = round((time.perf_counter() - inicio) * 1000, 1)
    resultado['umbral'] = ensemble.thresh
    return resultado
