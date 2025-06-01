# src/main.py

import streamlit as st
import pandas as pd

from data_validator import validate
from results import mostrar_resultados, plot_benchmark_spider, plot_slack_waterfall
from report_generator import generate_html_report, generate_excel_report, generate_pptx_report
from session_manager import init_db, save_session, load_sessions

st.set_page_config(page_title="Simulador Econométrico-Deliberativo", layout="wide")

init_db()
st.title("Simulador Econométrico-Deliberativo")

# Modo Tutorial en la barra lateral
tutorial_mode = st.sidebar.checkbox("Modo Tutorial")
if tutorial_mode:
    st.sidebar.markdown("### Paso 1/5: Carga de datos")
    st.sidebar.markdown("- Suba un CSV con columnas: DMU, inputs, outputs.")
    st.sidebar.markdown("### Paso 2/5: Seleccionar columnas")
    st.sidebar.markdown("- Elija la columna DMU, luego inputs y outputs.")
    st.sidebar.markdown("### Paso 3/5: Validación")
    st.sidebar.markdown("- La app validará que no haya nulos, ceros o negativos.")
    st.sidebar.markdown("### Paso 4/5: Ejecución DEA")
    st.sidebar.markdown("- Haga clic en 'Ejecutar DEA (CCR y BCC)'.")
    st.sidebar.markdown("### Paso 5/5: Reportes y Descargas")
    st.sidebar.markdown("- Puede descargar reporte HTML, Excel o PPTX.")

# Botón para reindexar RAG
if st.sidebar.button("Reindexar RAG"):
    try:
        from rag_indexer import reindex_rag
        reindex_rag("path/to/corpus")
        st.sidebar.success("Reindexación completada.")
    except Exception as e:
        st.sidebar.error(f"Error al reindexar: {e}")

# 1) Subida de datos
archivo = st.file_uploader("Sube tu archivo CSV con datos (DMU, inputs, outputs)", type=["csv"])
if archivo is None:
    st.info("Por favor, sube un archivo CSV para continuar.")
    st.stop()

df = pd.read_csv(archivo)

# 2) Seleccionar columna DMU
dmu_col = st.selectbox(
    "Columna DMU (identificador)",
    df.columns,
    help="Seleccione la columna que identifica cada unidad (DMU)."
)

# 3) Seleccionar inputs y outputs
input_cols = st.multiselect(
    "Seleccione columnas de Inputs",
    [c for c in df.columns if c != dmu_col],
    help="Lista de columnas usadas como insumos. Deben ser numéricas y > 0."
)

output_cols = st.multiselect(
    "Seleccione columnas de Outputs",
    [c for c in df.columns if c != dmu_col],
    help="Lista de columnas usadas como productos. Deben ser numéricas y > 0."
)

if len(input_cols) < 1 or len(output_cols) < 1:
    st.info("Debes seleccionar al menos una columna de input y una de output.")
    st.stop()

# 4) Validar datos
validacion = validate(df, input_cols, output_cols)
if validacion["formal_issues"]:
    st.error("Problemas formales detectados:\n- " + "\n- ".join(validacion["formal_issues"]))
else:
    st.success("Validación formal OK")

# Mostrar sugerencias del LLM (o error)
llm_json = validacion["llm"]
if "error" in llm_json:
    st.warning(f"Error al consultar LLM: {llm_json['message']}")
else:
    st.json(llm_json)

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

    # Dentro de main.py, después de haber construido merged_ccr:
    merged_ccr = resultados["merged_ccr"]

    st.subheader("Benchmark Spider (CCR)")
    peers = merged_ccr.query("efficiency == 1")["DMU"].tolist()
    if peers:
        sel_dmu = st.selectbox("Seleccione una DMU eficiente para benchmark (CCR)", peers)
        # ← Aquí valida que exista:
        if sel_dmu not in merged_ccr["DMU"].astype(str).tolist():
            st.error(f"La DMU '{sel_dmu}' no existe en los datos mostrados.")
        else:
            fig_spider = plot_benchmark_spider(merged_ccr, sel_dmu, input_cols, output_cols)
            st.plotly_chart(fig_spider, use_container_width=True)
    else:
        st.info("No hay DMUs eficientes CCR para benchmarking.")

    # Segunda sección: Diagnóstico de slacks CCR
    st.subheader("Diagnóstico de Slacks CCR")
    sel_dmu_ccr = st.selectbox("Seleccione DMU", merged_ccr["DMU"].tolist())
    # ← Validación:
    if sel_dmu_ccr not in merged_ccr["DMU"].astype(str).tolist():
        st.error(f"La DMU '{sel_dmu_ccr}' no existe.")
    else:
        slack_in = merged_ccr.loc[merged_ccr["DMU"] == sel_dmu_ccr, "slacks_inputs"].iloc[0]
        slack_out = merged_ccr.loc[merged_ccr["DMU"] == sel_dmu_ccr, "slacks_outputs"].iloc[0]
        slack_dict = {**slack_in, **slack_out}
        fig_water = plot_slack_waterfall(slack_dict, sel_dmu_ccr)
        st.pyplot(fig_water)

    # 6) Generar link de descarga de reporte HTML
    if st.button("Generar Reporte HTML"):
        html_content = generate_html_report(
            df_dea=resultados["df_ccr"],
            df_tree=pd.DataFrame(),    # Aquí iría el DataFrame real del árbol
            df_eee=pd.DataFrame()      # Aquí iría el DataFrame real de métricas EEE
        )
        st.download_button(
            label="Descargar Reporte HTML",
            data=html_content,
            file_name="reporte_dea_deliberativo.html",
            mime="text/html"
        )

    # 7) Generar link de descarga de reporte Excel
    if st.button("Generar Reporte Excel"):
        excel_io = generate_excel_report(
            df_dea=resultados["df_ccr"],
            df_tree=pd.DataFrame(),    # Aquí iría el DataFrame real del árbol
            df_eee=pd.DataFrame()      # Aquí iría el DataFrame real de métricas EEE
        )
        st.download_button(
            label="Descargar Reporte Excel",
            data=excel_io,
            file_name="reporte_dea_deliberativo.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # 8) Generar link de descarga de reporte PPTX
    if st.button("Generar Reporte PPTX"):
        pptx_io = generate_pptx_report(
            df_dea=resultados["df_ccr"],
            df_tree=pd.DataFrame(),    # Aquí iría el DataFrame real del árbol
            df_eee=pd.DataFrame()      # Aquí iría el DataFrame real de métricas EEE
        )
        st.download_button(
            label="Descargar Reporte PPTX",
            data=pptx_io,
            file_name="reporte_dea_deliberativo.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
        )

    # 9) Guardar sesión de resultados
    save_session(
        dmu_column=dmu_col,
        input_cols=input_cols,
        output_cols=output_cols,
        results=resultados
    )

# 10) Cargar sesiones previas
st.sidebar.subheader("Sesiones Guardadas")
sessions = load_sessions()
if sessions:
    for i, sess in enumerate(sessions, start=1):
        st.sidebar.markdown(
            f"**Sesión {i}:** DMU={sess['dmu_column']}, Inputs={sess['input_cols']}, Outputs={sess['output_cols']}"
        )
else:
    st.sidebar.info("No hay sesiones guardadas.")
