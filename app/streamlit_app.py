import streamlit as st, pandas as pd, numpy as np, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.ensemble import FraudEnsemble
    ensemble = FraudEnsemble()
except Exception as e:
    ensemble = None
    st.error(f"Modelos no encontrados. Ejecuta `python train.py` primero. Error: {e}")

st.set_page_config(page_title="PTIA - Detección de Fraude", layout="wide", page_icon="️")
st.title("🛡️ Sistema de Detección de Fraude (Ensemble Híbrido)")
st.markdown("Simula una transacción para ver cómo el modelo combina el **conocimiento** (RF) y el **razonamiento** (IF).")

# Cargar dataset para obtener ejemplos reales
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/creditcard.csv')
        df['Hour'] = (df['Time'] // 3600) % 24
        df['DayOfWeek'] = (df['Time'] // 86400) % 7
        df['LogAmount'] = np.log1p(df['Amount'])
        return df
    except:
        return None

df = load_data()

st.sidebar.header("️ Pruebas Rápidas (Dataset Real)")
if df is not None:
    st.sidebar.write("Usa ejemplos reales para probar el modelo:")
    if st.sidebar.button("🔴 Cargar Ejemplo de FRAUDE Real"):
        fraud_sample = df[df['Class'] == 1].sample(1).iloc[0]
        st.session_state['test_features'] = fraud_sample.to_dict()
        st.rerun()

    if st.sidebar.button("🟢 Cargar Ejemplo LEGÍTIMO Real"):
        legit_sample = df[df['Class'] == 0].sample(1).iloc[0]
        st.session_state['test_features'] = legit_sample.to_dict()
        st.rerun()

st.sidebar.header("📊 Configuración del Ensemble")
st.sidebar.write("Peso RF (Conocimiento): 70%")
st.sidebar.write("Peso IF (Razonamiento): 30%")
st.sidebar.write("Umbral de Alerta: 85%")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Datos de la Transacción")
    current_features = st.session_state.get('test_features', {})

    v1 = st.number_input("V1 (PCA)", value=float(current_features.get('V1', -1.35)))
    v2 = st.number_input("V2 (PCA)", value=float(current_features.get('V2', 1.20)))
    amount = st.number_input("Monto ($)", value=float(current_features.get('Amount', 150.0)))
    hour = st.slider("Hora del día", 0, 23, int(current_features.get('Hour', 14)))
    
    # Reconstruir el diccionario completo con las 31 features
    features = {f"V{i}": float(current_features.get(f"V{i}", 0.0)) for i in range(1, 29)}
    features.update({"V1": float(v1), "V2": float(v2), "Amount": float(amount), 
                     "Hour": float(hour), "DayOfWeek": float(current_features.get('DayOfWeek', 3.0)), 
                     "LogAmount": float(np.log1p(amount))})

with col2:
    st.subheader("Resultado del Análisis")
    if st.button("🔍 Analizar con IA", type="primary", use_container_width=True):
        if ensemble:
            res = ensemble.predict(features)
            
            if res['prediction'] == 'FRAUDE':
                st.error(f"🚨 ALERTA: FRAUDE DETECTADO (Confianza: {res['final_score']*100:.1f}%)")
            else:
                st.success(f"✅ Transacción Legítima (Riesgo: {res['final_score']*100:.1f}%)")
            
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("Score Final", f"{res['final_score']*100:.1f}%")
            c2.metric("Aporte RF", f"{res['rf_score']*100:.1f}%")
            c3.metric("Aporte IF", f"{res['if_score']*100:.1f}%")
        else:
            st.warning("El modelo no está cargado.")