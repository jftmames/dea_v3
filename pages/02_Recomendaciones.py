# ── pages/02_Recomendaciones.py ──
import streamlit as st
import pandas as pd
import json

# ------------------------------------------------------------------------------------------------
# Aquí reutilizaremos las funciones y datos que guardamos en st.session_state desde main.py
# ------------------------------------------------------------------------------------------------

st.set_page_config(page_title="Recomendaciones DEA", layout="wide")
st.title("Recomendaciones y Diagnóstico DEA")

# 1) Recuperar DataFrame original y parámetros DEA de session_state
if "original_df" not in st.session_state:
    st.warning("Primero sube un CSV en la página principal para poder ver recomendaciones.")
    st.stop()

df = st.session_state["original_df"]
inputs = st.session_state.get("dea_inputs", [])
outputs = st.session_state.get("dea_outputs", [])
model = st.session_state.get("dea_model", None)
orientation = st.session_state.get("dea_orientation", None)
super_eff = st.session_state.get("dea_super_eff", None)

# 2) Mostrar la explicación de la orientación (si existe)
if "orientation_feedback" in st.session_state:
    st.subheader("¿Es apropiada la orientación seleccionada?")
    st.write(st.session_state["orientation_feedback"])
else:
    st.info("Vuelve a ejecutar el DEA en la página principal para generar feedback de orientación.")

st.markdown("---")

# 3) Mostrar las recomendaciones de inputs/outputs (si existen)
if "last_reco" in st.session_state and st.session_state["last_reco"]:
    reco = st.session_state["last_reco"]
    st.subheader("Recomendaciones automáticas de Inputs/Outputs")
    if reco.get("recommend_inputs") is not None:
        st.write("• **Inputs sugeridos:**", reco["recommend_inputs"])
        st.write("• **Outputs sugeridos:**", reco["recommend_outputs"])
    else:
        st.write("📝 Texto libre de recomendación:")
        st.write(reco.get("text", "Sin recomendaciones adicionales"))
else:
    st.info("Vuelve a generar el árbol en la página principal para obtener recomendaciones.")

st.markdown("---")

# 4) (Opcional) Si quieres ver un diagnóstico tabular:
if "df_diag" in st.session_state:
    st.subheader("Diagnóstico técnico (tabla)")
    st.dataframe(st.session_state["df_diag"], use_container_width=True)
else:
    st.info("No existe diagnóstico generado. Genera un árbol en la página principal para producir diagnóstico.")
