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
