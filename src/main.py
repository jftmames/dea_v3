# src/main.py

import streamlit as st
import pandas as pd

from data_validator import validate
from results import mostrar_resultados
from report_generator import generate_html_report

st.set_page_config(page_title="Simulador Econométrico-Deliberativo", layout="wide")

st.title("Simulador Econométrico-Deliberativo")

# 1) Subida de datos
archivo = st.file_uploader("Sube tu archivo CSV con datos (DMU, inputs, outputs)", type=["csv"])
if archivo is None:
    st.info("Por favor, sube un archivo CSV para continuar.")
    st.stop()

df = pd.read_csv(archivo)

# 2) Seleccionar columna DMU
dmu_col = st.selectbox("Columna DMU (identificador)", df.columns)

# 3) Seleccionar inputs y outputs
cols = list(df.columns)
cols.remove(dmu_col)
input_cols = st.multiselect("Seleccione columnas de Inputs", cols)
output_cols = st.multiselect("Seleccione columnas de Outputs", cols)

if len(input_cols) < 1 or len(output_cols) < 1:
    st.info("Debes seleccionar al menos una columna de input y una de output.")
    st.stop()

# 4) Validar datos
validacion = validate(df, input_cols, output_cols)
if validacion["formal_issues"]:
    st.error("Problemas formales detectados:\n- " + "\n- ".join(validacion["formal_issues"]))
else:
    st.success("Validación formal OK")

# Mostrar sugerencias del LLM si las hay
if "issues" in validacion["llm"] and validacion["llm"]["issues"]:
    st.warning("Sugerencias del LLM:")
    st.json(validacion["llm"])

# 5) Ejecución de DEA y visualizaciones
if st.button("Ejecutar DEA (CCR y BCC)"):
    with st.spinner("Calculando eficiencias…"):
        resultados = mostrar_resultados(df, dmu_col, input_cols, output_cols)

    # Mostrar tablas de eficiencia
    st.subheader("Tabla de Eficiencias CCR")
    st.dataframe(resultados["df_ccr"])

    st.subheader("Tabla de Eficiencias BCC")
    st.dataframe(resultados["df_bcc"])

    # Mostrar histogramas
    st.subheader("Histograma de Eficiencias CCR")
    st.plotly_chart(resultados["hist_ccr"], use_container_width=True)

    st.subheader("Histograma de Eficiencias BCC")
    st.plotly_chart(resultados["hist_bcc"], use_container_width=True)

    # Mostrar scatter 3D
    st.subheader("Scatter 3D CCR")
    st.plotly_chart(resultados["scatter3d_ccr"], use_container_width=True)

    st.subheader("Scatter 3D BCC")
    st.plotly_chart(resultados["scatter3d_bcc"], use_container_width=True)

    # Benchmark Spider: elegir una DMU eficiente para comparar
    st.subheader("Benchmark Spider (CCR)")
    # Solo DMUs con eficiencia==1
    peers = resultados["merged_ccr"].query("efficiency == 1")["DMU"].tolist()
    if peers:
        sel_dmu = st.selectbox("Seleccione una DMU eficiente para benchmark (CCR)", peers)
        fig_spider = resultados["merged_ccr"].copy()  # DataFrame con columns 'DMU', inputs, outputs, 'efficiency'
        spider_fig = None
        try:
            from results import plot_benchmark_spider
            spider_fig = plot_benchmark_spider(fig_spider, sel_dmu, input_cols, output_cols)
            st.plotly_chart(spider_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error al generar spider: {e}")
    else:
        st.info("No hay DMUs eficientes CCR para benchmarking.")

    # 6) Generar link de descarga de reporte HTML
    if st.button("Generar Reporte HTML"):
        html_content = generate_html_report(
            df_dea=resultados["df_ccr"],
            df_tree=pd.DataFrame(),    # Aquí pondrías tu DataFrame real de árbol
            df_eee=pd.DataFrame()      # Aquí tu DataFrame real de metadatos EEE
        )
        # Mostrar link de descarga
        st.download_button(
            label="Descargar Reporte HTML",
            data=html_content,
            file_name="reporte_dea_deliberativo.html",
            mime="text/html"
        )

