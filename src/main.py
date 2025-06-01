import streamlit as st
import pandas as pd
from data_validator import validate
st.write("Hello DEA!")
st.subheader("Validator demo")
uploaded = st.file_uploader("CSV", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())
    if st.button("Run validation"):
        result = validate(df, df.columns[:-1].tolist(), [df.columns[-1]])
        st.json(result)
