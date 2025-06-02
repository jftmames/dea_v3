# src/main.py

import streamlit as st
import pandas as pd
import datetime

from data_validator import validate
from results import mostrar_resultados, plot_benchmark_spider
from report_generator import generate_html_report, generate_excel_report
from session_manager import init_db, save_session, load_sessions

# -------------------------------------------------------
# 0) Configuración inicial de Streamlit
# -------------------------------------------------------
st.set_page_config(layout="wide")  # Usa ancho completo para que el sidebar permanezca visible

# -------------------------------------------------------
# 1) Inicialización de la base de datos de sesiones
# -------------------------------------------------------
init_db()
default_user_id = "user_1"  # Identificador estático; ajústalo según tu lógica

# -------------------------------------------------------
# 2) Sidebar: cargar sesiones previas
#    (Siempre se dibuja, no dentro de if uploaded_file)
# -------------------------------------------------------
st.sidebar.header("Simulador DEA - Sesiones Guardadas")

sessions = load_sessions(user_id=default_user_id)
selected_session_id = None
sess = {}

if sessions:
    ids = [s["session_id"] for s in sessions]
    selected_session_id = st.sidebar.selectbox("Seleccionar sesión para recargar", ids)

    if selected_session_id:
        sess = next(s for s in sessions if s["session_id"] == selected_session_id)
        st.sidebar.markdown(f"**Sesión:** {sess['session_id']}")
        st.sidebar.markdown(f"- Timestamp: {sess['timestamp']}")
        st.sidebar.markdown(f"- EEE Score: {sess['eee_score']}")
        st.sidebar.markdown(f"- Notas: {sess['notes']}")
else:
    st.sidebar.write("No hay sesiones guardadas.")

st.sidebar.markdown("---")
st.sidebar.write("**Instrucciones:**")
st.sidebar.write("- Carga un CSV, selecciona columnas y ejecuta DEA.")
st.sidebar.write("- Luego podrás guardar y generar reportes.")
st.sidebar.markdown("---")

# -------------------------------------------------------
# 3) Área principal: Título y carga de archivo CSV
# -------------------------------------------------------
st.title("Simulador Econométrico-Deliberativo – DEA")

# Inicializar en session_state las variables necesarias
if "df" not in st.session_state:
    st.session_state.df = None
if "dmu_col" not in st.session_state:
    st.session_state.dmu_col = None
if "input_cols" not in st.session_state:
    st.session_state.input_cols = []
if "output_cols" not in st.session_state:
    st.session_state.output_cols = []
if "dea_results" not in st.session_state:
    st.session_state.dea_results = None
if "selected_dmu" not in st.session_state:
    st.session_state.selected_dmu = None

uploaded_file = st.file_uploader("Cargar archivo CSV (con DMUs)", type=["csv"])

if uploaded_file is not None:
    df_uploaded = pd.read_csv(uploaded_file)
    st.session_state.df = df_uploaded

# Mostrar DataFrame si ya existe
if st.session_state.df is not None:
    df = st.session_state.df.copy()
    st.subheader("Datos cargados")
    st.dataframe(df)

    # -------------------------------------------------------
    # 4) Selección de columnas: DMU, inputs y outputs
    # -------------------------------------------------------
    all_columns = df.columns.tolist()
    st.session_state.dmu_col = st.selectbox(
        "Columna que identifica cada DMU",
        all_columns,
        index=all_columns.index(st.session_state.dmu_col) if st.session_state.dmu_col in all_columns else 0
    )

    candidate_inputs = [c for c in all_columns if c != st.session_state.dmu_col]
    st.session_state.input_cols = st.multiselect(
        "Seleccionar columnas de inputs",
        candidate_inputs,
        default=st.session_state.input_cols
    )

    candidate_outputs = [c for c in all_columns if c not in st.session_state.input_cols + [st.session_state.dmu_col]]
    st.session_state.output_cols = st.multiselect(
        "Seleccionar columnas de outputs",
        candidate_outputs,
        default=st.session_state.output_cols
    )

    # -------------------------------------------------------
    # 5) Botón para ejecutar DEA (CCR y BCC)
    # -------------------------------------------------------
    if st.button("Ejecutar DEA (CCR y BCC)"):
        errors = validate(df, st.session_state.input_cols, st.session_state.output_cols)
        llm_ready = errors.get("llm", {}).get("ready", True)

        if errors["formal_issues"] or not llm_ready:
            st.error("Se encontraron problemas en los datos o sugerencias del LLM:")
            if errors["formal_issues"]:
                st.write("– Formal issues:")
                for issue in errors["formal_issues"]:
                    st.write(f"  • {issue}")
            if "issues" in errors["llm"] and errors["llm"]["issues"]:
                st.write("– LLM issues:")
                for issue in errors["llm"]["issues"]:
                    st.write(f"  • {issue}")
        else:
            with st.spinner("Calculando eficiencias…"):
                resultados = mostrar_resultados(
                    df.copy(),
                    st.session_state.dmu_col,
                    st.session_state.input_cols,
                    st.session_state.output_cols
                )
            st.session_state.dea_results = resultados

    # -------------------------------------------------------
    # 6) Mostrar resultados si ya se calcularon
    # -------------------------------------------------------
    if st.session_state.dea_results is not None:
        resultados = st.session_state.dea_results
        df_ccr = resultados["df_ccr"]
        df_bcc = resultados["df_bcc"]

        st.subheader("Resultados CCR")
        st.dataframe(df_ccr)

        st.subheader("Resultados BCC")
        st.dataframe(df_bcc)

        st.subheader("Histograma de eficiencias CCR")
        st.plotly_chart(resultados["hist_ccr"], use_container_width=True)

        st.subheader("Histograma de eficiencias BCC")
        st.plotly_chart(resultados["hist_bcc"], use_container_width=True)

        st.subheader("Scatter 3D Inputs vs Output (CCR)")
        st.plotly_chart(resultados["scatter3d_ccr"], use_container_width=True)

        # -------------------------------------------------------
        # 7) Benchmark Spider CCR
        # -------------------------------------------------------
        st.subheader("Benchmark Spider CCR")
        dmu_options = df_ccr["DMU"].astype(str).tolist()
        st.session_state.selected_dmu = st.selectbox(
            "Seleccionar DMU para comparar contra peers eficientes",
            dmu_options,
            index=dmu_options.index(st.session_state.selected_dmu) if st.session_state.selected_dmu in dmu_options else 0
        )

        if st.session_state.selected_dmu:
            merged_ccr = df_ccr.merge(df, on="DMU", how="left")
            spider_fig = plot_benchmark_spider(
                merged_ccr,
                st.session_state.selected_dmu,
                st.session_state.input_cols,
                st.session_state.output_cols
            )
            st.plotly_chart(spider_fig, use_container_width=True)

        # -------------------------------------------------------
        # 8) Guardar sesión
        # -------------------------------------------------------
        st.subheader("Guardar esta sesión")
        inquiry_tree = sess.get("inquiry_tree", {}) if selected_session_id else {}
        eee_score = sess.get("eee_score", 0.0) if selected_session_id else 0.0
        notes = st.text_area(
            "Notas sobre la sesión",
            value=sess.get("notes", "") if selected_session_id else ""
        )

        if st.button("Guardar sesión actual"):
            save_session(
                default_user_id,
                inquiry_tree,
                eee_score,
                notes
            )
            st.success("Sesión guardada correctamente.")

        # -------------------------------------------------------
        # 9) Generar reportes
        # -------------------------------------------------------
        st.subheader("Generar reportes")

        # 9.1) Reporte HTML
        html_str = generate_html_report(
            df_dea=df_ccr,
            df_tree=pd.DataFrame(),   # Placeholder: tu DataFrame de árbol si lo tienes
            df_eee=pd.DataFrame()     # Placeholder: tu DataFrame de EEE si lo tienes
        )
        st.download_button(
            label="Descargar Reporte HTML",
            data=html_str,
            file_name=f"reporte_dea_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html"
        )

        # 9.2) Reporte Excel
        excel_io = generate_excel_report(
            df_dea=df_ccr,
            df_tree=pd.DataFrame(),
            df_eee=pd.DataFrame()
        )
        st.download_button(
            label="Descargar Reporte Excel",
            data=excel_io,
            file_name=f"reporte_dea_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
