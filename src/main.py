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

        # Recargar los datos y configuraciones si se selecciona una sesión
        if st.sidebar.button("Cargar datos de esta sesión"):
            # Aquí asumimos que tienes el DataFrame original guardado o puedes reconstruirlo
            # Si df_ccr y df_bcc se guardaron como JSON, se pueden cargar:
            st.session_state.df_ccr_loaded = sess.get("df_ccr") # Guardar en session_state para acceso futuro
            st.session_state.df_bcc_loaded = sess.get("df_bcc")
            st.session_state.dmu_col_loaded = sess.get("dmu_column")
            st.session_state.input_cols_loaded = sess.get("input_cols")
            st.session_state.output_cols_loaded = sess.get("output_cols")
            
            st.experimental_rerun() # Recargar la app para mostrar los cambios

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
    
    # Pre-seleccionar valores si se cargó una sesión
    dmu_col_default = st.session_state.get("dmu_col_loaded", all_columns[0] if all_columns else None)
    input_cols_default = st.session_state.get("input_cols_loaded", [])
    output_cols_default = st.session_state.get("output_cols_loaded", [])

    dmu_col = st.selectbox("Columna que identifica cada DMU", all_columns, index=all_columns.index(dmu_col_default) if dmu_col_default in all_columns else 0)

    candidate_inputs = [c for c in all_columns if c != dmu_col]
    input_cols = st.multiselect("Seleccionar columnas de inputs", candidate_inputs, default=input_cols_default)

    candidate_outputs = [c for c in all_columns if c not in input_cols + [dmu_col]]
    output_cols = st.multiselect("Seleccionar columnas de outputs", candidate_outputs, default=output_cols_default)

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
                # Almacenar resultados en st.session_state para que persistan
                st.session_state.resultados_dea = resultados
                st.session_state.df_original = df
                st.session_state.dmu_col = dmu_col
                st.session_state.input_cols = input_cols
                st.session_state.output_cols = output_cols

# -------------------------------------------------------
# 6) Mostrar resultados si ya se calcularon o se cargaron de una sesión
# -------------------------------------------------------
if "resultados_dea" in st.session_state:
    resultados = st.session_state.resultados_dea
    df = st.session_state.df_original
    dmu_col = st.session_state.dmu_col
    input_cols = st.session_state.input_cols
    output_cols = st.session_state.output_cols

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

    st.subheader("Benchmark Spider CCR")
    # Asegúrate de que df_ccr tiene la columna 'DMU' (renombrada en results.py)
    dmu_options = df_ccr[dmu_col].astype(str).tolist() # Usa dmu_col aquí
    
    # Intenta pre-seleccionar la primera DMU si no hay una seleccionada
    selected_dmu_index = 0
    if selected_session_id and "dmu_column" in sess: # Si se cargó una sesión, podría haber una DMU específica
        try:
            # Si el DataFrame completo fue cargado, busca la DMU allí
            if 'df_ccr_loaded' in st.session_state:
                if dmu_col in st.session_state.df_ccr_loaded.columns:
                    # Busca la DMU de la sesión cargada en las opciones actuales
                    dmu_from_session = sess.get("dmu_column")
                    if dmu_from_session and dmu_from_session in dmu_options:
                        selected_dmu_index = dmu_options.index(dmu_from_session)
        except Exception:
            pass # Si falla, usa la primera opción

    selected_dmu = st.selectbox("Seleccionar DMU para comparar contra peers eficientes", dmu_options, index=selected_dmu_index)

    if selected_dmu:
        # merged_ccr ya está en resultados, así que lo usamos directamente
        merged_ccr = resultados["merged_ccr"]
        spider_fig = plot_benchmark_spider(merged_ccr, selected_dmu, input_cols, output_cols)
        st.plotly_chart(spider_fig, use_container_width=True)

    # -------------------------------------------------------
    # 5.7) Guardar sesión
    # -------------------------------------------------------
    st.subheader("Guardar esta sesión")
    session_id = str(uuid.uuid4()) # Generar un nuevo ID de sesión
    inquiry_tree = {}              # Ajusta según tu lógica real de árbol
    eee_score = 0.0                # Ajusta según tu cálculo real de EEE
    notes = st.text_area("Notas sobre la sesión", value=sess.get("notes", "") if selected_session_id else "")

    if st.button("Guardar sesión actual"):
        save_session(
            session_id,
            default_user_id,
            inquiry_tree,
            eee_score,
            notes,
            dmu_col,        # Pasa dmu_col
            input_cols,     # Pasa input_cols
            output_cols,    # Pasa output_cols
            df_ccr,         # Pasa df_ccr
            df_bcc          # Pasa df_bcc
        )
        st.success(f"Sesión {session_id} guardada correctamente en la base de datos.")
        # Opcional: recargar sidebar para mostrar la nueva sesión
        st.experimental_rerun()

    # -------------------------------------------------------
    # 5.8) Generar reporte HTML y Excel
    # -------------------------------------------------------
    st.subheader("Generar reportes")

    # Placeholder para df_tree y df_eee. Debes reemplazarlos con tus DataFrames reales.
    df_tree_placeholder = pd.DataFrame({"Node": ["Root"], "Parent": ["None"], "Description": ["Initial Inquiry"]})
    df_eee_placeholder = pd.DataFrame({"Metric": ["EEE Score"], "Value": [eee_score]})

    if st.button("Generar y descargar reporte HTML"):
        html_str = generate_html_report(
            df_dea=df_ccr, # Usamos df_ccr como ejemplo de resultados DEA
            df_tree=df_tree_placeholder,
            df_eee=df_eee_placeholder
        )
        st.download_button(
            label="Descargar Reporte HTML",
            data=html_str,
            file_name=f"reporte_dea_{session_id}.html", # Nombre con ID de sesión
            mime="text/html"
        )

    if st.button("Generar y descargar reporte Excel"):
        excel_io = generate_excel_report(
            df_dea=df_ccr, # Usamos df_ccr como ejemplo de resultados DEA
            df_tree=df_tree_placeholder,
            df_eee=df_eee_placeholder
        )
        st.download_button(
            label="Descargar Reporte Excel",
            data=excel_io,
            file_name=f"reporte_dea_{session_id}.xlsx", # Nombre con ID de sesión
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
