# pages/02_Reajustar_Modelo.py
import streamlit as st
import pandas as pd

from dea_models.radial import run_ccr, run_bcc

def main():
    st.title("Reajustar Modelo DEA")

    archivo = st.file_uploader("Sube tu CSV", type=["csv"])
    if archivo is not None:
        df = pd.read_csv(archivo)

        # 1) Seleccionar columna que identifica a cada DMU
        dmu_col = st.selectbox("Columna DMU", df.columns)

        # 2) Seleccionar inputs y outputs
        input_cols = st.multiselect("Seleccione columnas de Inputs", df.columns)
        output_cols = st.multiselect("Seleccione columnas de Outputs", df.columns)

        if st.button("Calcular CCR y BCC"):
            try:
                # Validar positividad de datos (usa utilidades)
                from dea_models.utils import validate_positive_dataframe
                validate_positive_dataframe(df, input_cols + output_cols)

                # Ejecutar CCR
                df_ccr = run_ccr(
                    df=df,
                    dmu_column=dmu_col,
                    input_cols=input_cols,
                    output_cols=output_cols,
                    orientation="input",
                    super_eff=False
                )
                st.subheader("Resultados CCR")
                st.dataframe(df_ccr)

                # Ejecutar BCC
                df_bcc = run_bcc(
                    df=df,
                    dmu_column=dmu_col,
                    input_cols=input_cols,
                    output_cols=output_cols,
                    orientation="input",
                    super_eff=False
                )
                st.subheader("Resultados BCC")
                st.dataframe(df_bcc)

            except ValueError as e:
                st.error(f"Error en datos: {e}")
