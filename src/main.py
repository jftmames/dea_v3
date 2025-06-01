import streamlit as st
import pandas as pd 

from data_validator import validate
from dea_analyzer import run_dea
from inquiry_engine import generate_inquiry, to_plotly_tree

st.set_page_config(page_title="DEA Deliberativo MVP", layout="wide")
st.title("DEA Deliberativo – MVP")

# ------------------------------------------------------------------
# 1. Cargar CSV
# ------------------------------------------------------------------
upload = st.file_uploader("Sube tu CSV", type="csv")

if upload:
    df = pd.read_csv(upload)
    st.subheader("Vista previa")
    st.dataframe(df.head(), use_container_width=True)

    # solo columnas numéricas para selección
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.error("⚠️ El archivo no contiene columnas numéricas.")
        st.stop()

    st.markdown("### Selecciona columnas de **inputs** y **outputs**")
    st.info("⚠️ Solo columnas numéricas funcionarán en DEA.")

    inputs = st.multiselect("Inputs", numeric_cols, default=numeric_cols[:-1])
    outputs = st.multiselect("Outputs", numeric_cols, default=[numeric_cols[-1]])

    # ------------------------------------------------------------------
    # 2. Validación de datos
    # ------------------------------------------------------------------
    if st.button("Validar datos"):
        result = validate(df, inputs, outputs)
        st.subheader("Resultado del validador")
        st.json(result)

    # ------------------------------------------------------------------
    # 3. Ejecutar DEA
    # ------------------------------------------------------------------
        
if st.button("Ejecutar DEA (CCR input)"):
    with st.spinner("Optimizando…"):
        try:
            res = run_dea(df, inputs, outputs, model="CCR")
            if res["efficiency"].isna().all():
                st.error("⚠️ El solver no devolvió soluciones válidas.")
                st.stop()
        except (ValueError, KeyError) as e:
            st.error(f"❌ {e}")
            st.stop()

    st.session_state["res_df"] = res
    st.subheader("Eficiencias CCR")
    st.dataframe(res, use_container_width=True)


# ------------------------------------------------------------------
# 4. Complejos de Indagación
# ------------------------------------------------------------------
if "res_df" in st.session_state:
    ineff = st.session_state["res_df"].query("efficiency < 1")
    if len(ineff) == 0:
        st.info("Todas las DMU son eficientes; no hay árbol de indagación que generar.")
    else:
        st.subheader("Generar Complejo de Indagación")
        dmu = st.selectbox("Selecciona una DMU ineficiente", ineff["DMU"].tolist())

        if st.button("Crear árbol"):
            with st.spinner("Generando árbol con IA..."):
                root_q = f"¿Por qué la DMU {dmu} es ineficiente?"
                tree = generate_inquiry(root_q)
                fig = to_plotly_tree(tree)

            st.plotly_chart(fig, use_container_width=True)
            with st.expander("JSON del árbol"):
                st.json(tree)
