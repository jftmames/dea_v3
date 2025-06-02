# src/main.py

import streamlit as st
import pandas as pd
import uuid
import json
import datetime

from data_validator import validate
from results import mostrar_resultados, plot_benchmark_spider
from report_generator import generate_html_report, generate_excel_report
from session_manager import init_db, save_session, load_sessions

# -------------------------------------------------------
# 1) Inicialización de la base de datos de sesiones
# -------------------------------------------------------
init_db()
default_user_id = "user_1"  # Identificador estático; puedes cambiarlo según tu lógica de usuarios

# -------------------------------------------------------
# 2) Sidebar: cargar sesiones previas
# -------------------------------------------------------
st.sidebar.header("Sesiones Guardadas")
sessions = load_sessions(user_id=default_user_id)
selected_session_id = None

if sessions:
    # Mostrar lista de session_id en un selectbox
    ids = [s["session_id"] for s in sessions]
    selected_session_id = st.sidebar.selectbox("Seleccionar sesión para recargar", ids)

    if selected_session_id:
        sess = next(s for s in sessions if s["session_id"] == selected_session_id)
        st.sidebar.markdown(f"**Sesión:** {sess['session_id']}")
        st.sidebar.markdown(f"- Timestamp: {sess['timestamp']}")
        st.sidebar.markdown(f"- EEE Score: {sess['eee_score']}")
        st.sidebar.markdown(f"- Notas: {sess['notes']}")

# -------------------------------------------------------
# 3) Título principal y carga de archivo CSV
# -------------------------------------------------------
st.title("Simulador Econométrico-Deliberativo – DEA")
uploaded_file = st.file_uploader("Cargar archivo CSV (con DMUs)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Datos cargados")
    st.dataframe(df)

    # -------------------------------------------------------
    # 4) Selección de columnas: DMU, inputs y outputs
    # -------------------------------------------------------
    all_columns = df.columns.tolist()
    dmu_col = st.selectbox("Columna que identifica cada DMU", all_columns)

    candidate_inputs = [c for c in all_columns if c != dmu_col]
    input_cols = st.multiselect("Seleccionar columnas de inputs", candidate_inputs)

    candidate_outputs = [c for c in all_columns if c not in input_cols + [dmu_col]]
    output_cols = st.multiselect("Seleccionar columnas de outputs", candidate_outputs)

    # -------------------------------------------------------
    # 5) Botón para ejecutar DEA (CCR y BCC)
    # -------------------------------------------------------
    if st.button("Ejecutar DEA (CCR y BCC)"):
        # 5.1) Validar datos formales y sugerencias LLM
        errors = validate(df, input_cols, output_cols)
        llm_ready = errors.get("llm", {}).get("ready", True)
        if errors["formal_issues"] or not llm_ready:
            st.error("Se encontraron problemas en los datos o sugerencias del LLM:")
            if errors["formal_issues"]:
                st.write("– Formal issues:")
                for issue in errors["formal_issues"]:
                    st.write(f"  • {issue}")
            if "issues" in errors["llm"]:
                st.write("– LLM issues:")
                for issue in errors["llm"]["issues"]:
                    st.write(f"  • {issue}")
        else:
            # 5.2) Ejecutar modelos CCR y BCC
            with st.spinner("Calculando eficiencias…"):
                resultados = mostrar_resultados(df, dmu_col, input_cols, output_cols)

            if not resultados:
                st.error("Ocurrió un error al calcular DEA.")
            else:
                # 5.3) Mostrar tablas de resultados
                df_ccr = resultados["df_ccr"]
                df_bcc = resultados["df_bcc"]

                st.subheader("Resultados CCR")
                st.dataframe(df_ccr)

                st.subheader("Resultados BCC")
                st.dataframe(df_bcc)

                # 5.4) Mostrar histogramas de eficiencia
                st.subheader("Histograma de eficiencias CCR")
                st.plotly_chart(resultados["hist_ccr"], use_container_width=True)

                st.subheader("Histograma de eficiencias BCC")
                st.plotly_chart(resultados["hist_bcc"], use_container_width=True)

                # 5.5) Mostrar scatter 3D para CCR
                st.subheader("Scatter 3D Inputs vs Output (CCR)")
                st.plotly_chart(resultados["scatter3d_ccr"], use_container_width=True)

                # 5.6) Benchmark spider: seleccionar una DMU
                st.subheader("Benchmark Spider CCR")
                dmu_options = df_ccr["DMU"].astype(str).tolist()
                selected_dmu = st.selectbox("Seleccionar DMU para comparar contra peers eficientes", dmu_options)

                if selected_dmu:
                    merged_ccr = df_ccr.merge(df, on="DMU", how="left")
                    spider_fig = plot_benchmark_spider(merged_ccr, selected_dmu, input_cols, output_cols)
                    st.plotly_chart(spider_fig, use_container_width=True)

                # -------------------------------------------------------
                # 5.7) Guardar sesión
                # -------------------------------------------------------
                st.subheader("Guardar esta sesión")
                inquiry_tree = sess["inquiry_tree"] if (selected_session_id and "inquiry_tree" in sess) else {}
                eee_score     = sess["eee_score"]     if (selected_session_id and "eee_score" in sess) else 0.0
                notes         = st.text_area(
                                   "Notas sobre la sesión",
                                   value=sess["notes"] if (selected_session_id and "notes" in sess) else ""
                               )

                if st.button("Guardar sesión actual"):
                    save_session(
                        default_user_id,
                        inquiry_tree,
                        eee_score,
                        notes
                    )
                    st.success("Sesión guardada correctamente en la base de datos.")

                # -------------------------------------------------------
                # 5.8) Generar reporte HTML y Excel
                # -------------------------------------------------------
                st.subheader("Generar reportes")

                if st.button("Generar y descargar reporte HTML"):
                    html_str = generate_html_report(
                        df_dea=df_ccr,
                        df_tree=pd.DataFrame(),   # Placeholder: tu DataFrame de árbol
                        df_eee=pd.DataFrame()     # Placeholder: tu DataFrame de EEE
                    )
                    st.download_button(
                        label="Descargar Reporte HTML",
                        data=html_str,
                        file_name="reporte_dea.html",
                        mime="text/html"
                    )

                if st.button("Generar y descargar reporte Excel"):
                    excel_io = generate_excel_report(
                        df_dea=df_ccr,
                        df_tree=pd.DataFrame(),
                        df_eee=pd.DataFrame()
                    )
                    st.download_button(
                        label="Descargar Reporte Excel",
                        data=excel_io,
                        file_name="reporte_dea.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
