import streamlit as st
import pandas as pd
from data_validator import validate
from dea_analyzer import run_dea

st.title("DEA Deliberativo – MVP")

upload = st.file_uploader("Sube tu CSV", type="csv")
if upload:
    df = pd.read_csv(upload)
    st.dataframe(df.head())

    cols = df.columns.tolist()
    st.write("Elige columnas de *inputs* y *outputs*")
    inputs = st.multiselect("Inputs", cols, default=cols[:-1])
    outputs = st.multiselect("Outputs", cols, default=[cols[-1]])

    if st.button("Validar datos"):
        result = validate(df, inputs, outputs)
        st.json(result)

    if st.button("Ejecutar DEA (CCR input)"):
        res = run_dea(df, inputs, outputs, model="CCR")
        st.dataframe(res)
else:
    st.write("Carga un CSV para comenzar")

from inquiry_engine import generate_inquiry, to_plotly_tree

if 'res_df' not in st.session_state and 'res' in locals():
    st.session_state['res_df'] = res

if 'res_df' in st.session_state:
    st.subheader("Generar Complejo de Indagación")
    dmu = st.selectbox(
        "Selecciona una DMU ineficiente",
        st.session_state['res_df'].query("efficiency < 1")["DMU"].tolist()
    )
    if st.button("Crear árbol"):
        root_q = f"¿Por qué la DMU {dmu} es ineficiente?"
        tree = generate_inquiry(root_q)
        fig = to_plotly_tree(tree)
        st.plotly_chart(fig, use_container_width=True)
        st.json(tree)

