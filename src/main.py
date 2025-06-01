import streamlit as st
import pandas as pd
from data_validator import validate

st.title("DEA Deliberativo – Demo Validador")

upload = st.file_uploader("Carga tu CSV", type="csv")
if upload:
    df = pd.read_csv(upload)
    st.subheader("Vista previa")
    st.dataframe(df.head())

    default_inputs = df.columns[:-1].tolist()
    default_output = [df.columns[-1]]

    if st.button("Validar"):
        result = validate(df, default_inputs, default_output)
        st.subheader("Resultado")
        st.json(result)
else:
    st.write("Hello DEA!")
