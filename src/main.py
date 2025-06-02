# src/main.py

import streamlit as st
import pandas as pd
import datetime
import uuid # Para generar IDs de sesión únicos

from data_validator import validate
from results import mostrar_resultados, plot_benchmark_spider
from report_generator import generate_html_report, generate_excel_report, generate_pptx_report # Importar generate_pptx_report
from session_manager import init_db, save_session, load_sessions

# -------------------------------------------------------
# 0) Configuración inicial de Streamlit
# -------------------------------------------------------
st.set_page_config(layout="wide")  # Usar ancho completo para que sidebar sea fijo

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

# Cargar sesiones para este user_id
sessions = load_sessions(user_id=default_user_id)
selected_session_id = None
sess = {}  # Diccionario con datos de la sesión seleccionada

if sessions:
    ids_timestamps = [f"{s['session_id']} ({s['timestamp']})" for s in sessions]
    selected_session_display = st.sidebar.selectbox("Seleccionar sesión para recargar", ids_timestamps)
    
    if selected_session_display:
        selected_session_id = selected_session_display.split(" ")[0] # Extraer solo el ID
        sess = next(s for s in sessions if s["session_id"] == selected_session_id)
        
        st.sidebar.markdown(f"**Sesión seleccionada:** `{sess['session_id']}`")
        st.sidebar.markdown(f"- **Timestamp:** `{sess['timestamp']}`")
        st.sidebar.markdown(f"- **Notas:** `{sess['notes']}`")

        if st.sidebar.button("Cargar datos de esta sesión"):
            st.session_state.df = pd.DataFrame() # Resetear DF actual para evitar conflictos
            st.session_state.df = sess.get("original_dataframe", None) # Asumiendo que el DF original se guarda
            if st.session_state.df is None and "df_ccr" in sess and "df_bcc" in sess:
                st.warning("No se encontró el DataFrame original en la sesión. Solo se cargarán los resultados CCR/BCC.")
                # Aquí podrías intentar reconstruir algo mínimo si es viable, o manejarlo como un caso especial.
            
            st.session_state.dmu_col = sess.get("dmu_column", None)
            st.session_state.input_cols = sess.get("input_cols", [])
            st.session_state.output_cols = sess.get("output_cols", [])
            st.session_state.dea_results = {
                "df_ccr": sess.get("df_ccr", pd.DataFrame()),
                "df_bcc": sess.get("df_bcc", pd.DataFrame()),
                # Recuperar otras figuras si se guardaron sus datos serializables
                "hist_ccr": None, # Las figuras no se guardan directamente, se regeneran
                "hist_bcc": None,
                "scatter3d_ccr": None,
                "scatter3d_bcc": None,
            }
            # Regenerar figuras si los DataFrames de resultados están presentes
            if not sess.get("df_ccr", pd.DataFrame()).empty and not sess.get("df_bcc", pd.DataFrame()).empty:
                # Necesitaríamos el DataFrame original para plot_3d_inputs_outputs
                # Por simplicidad, si no se guardó el df original, las figuras 3D no se regenerarán
                # Puedes adaptar esto para guardar también el df original o reconstruirlo.
                temp_res = mostrar_resultados(
                    st.session_state.df,
                    st.session_state.dmu_col,
                    st.session_state.input_cols,
                    st.session_state.output_cols
                )
                st.session_state.dea_results["hist_ccr"] = temp_res["hist_ccr"]
                st.session_state.dea_results["hist_bcc"] = temp_res["hist_bcc"]
                st.session_state.dea_results["scatter3d_ccr"] = temp_res["scatter3d_ccr"]
                st.session_state.dea_results["scatter3d_bcc"] = temp_res["scatter3d_bcc"]

            st.sidebar.success(f"Sesión `{selected_session_id}` cargada.")
            st.rerun() # Recargar la página para mostrar los datos cargados
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

# Si aún no existe en estado el DataFrame cargado, guardamos None
if "df" not in st.session_state:
    st.session_state.df = None

uploaded_file = st.file_uploader("Cargar archivo CSV (con DMUs)", type=["csv"])

# Si se subió un archivo, lo guardamos en session_state
if uploaded_file is not None:
    # Si el archivo subido es diferente al que ya está en sesión, lo actualizamos
    if st.session_state.df is None or not df.equals(pd.read_csv(uploaded_file)): # Pequeña optimización
        df_uploaded = pd.read_csv(uploaded_file)
        st.session_state.df = df_uploaded
        # Resetear selecciones de columnas y resultados DEA al cargar nuevo archivo
        st.session_state.dmu_col = None
        st.session_state.input_cols = []
        st.session_state.output_cols = []
        st.session_state.dea_results = None
        st.rerun() # Recargar para que los selectbox se reinicien

# Mostrar DataFrame solo si ya está en session_state
if st.session_state.df is not None:
    df = st.session_state.df.copy() # Trabajar con una copia
    st.subheader("Datos cargados")
    st.dataframe(df)

    # -------------------------------------------------------
    # 4) Selección de columnas: DMU, inputs y outputs
    # -------------------------------------------------------
    all_columns = df.columns.tolist()
    
    # DMU selectbox
    # Asegurarse de que el valor por defecto sea el cargado de la sesión o el primero
    initial_dmu_index = 0
    if st.session_state.dmu_col and st.session_state.dmu_col in all_columns:
        initial_dmu_index = all_columns.index(st.session_state.dmu_col)
    st.session_state.dmu_col = st.selectbox(
        "Columna que identifica cada DMU",
        all_columns,
        index=initial_dmu_index,
        key="dmu_selector" # Añadir una key para estabilidad
    )

    # Inputs multiselect
    candidate_inputs = [c for c in all_columns if c != st.session_state.dmu_col]
    # Asegurarse de que los valores por defecto sean los cargados de la sesión
    default_inputs_selected = [col for col in st.session_state.input_cols if col in candidate_inputs]
    st.session_state.input_cols = st.multiselect(
        "Seleccionar columnas de inputs",
        candidate_inputs,
        default=default_inputs_selected,
        key="inputs_selector" # Añadir una key
    )

    # Outputs multiselect
    candidate_outputs = [c for c in all_columns if c not in st.session_state.input_cols + [st.session_state.dmu_col]]
    # Asegurarse de que los valores por defecto sean los cargados de la sesión
    default_outputs_selected = [col for col in st.session_state.output_cols if col in candidate_outputs]
    st.session_state.output_cols = st.multiselect(
        "Seleccionar columnas de outputs",
        candidate_outputs,
        default=default_outputs_selected,
        key="outputs_selector" # Añadir una key
    )

    # -------------------------------------------------------
    # 5) Botón para ejecutar DEA (CCR y BCC)
    # -------------------------------------------------------
    if st.button("Ejecutar DEA (CCR y BCC)", key="run_dea_button"):
        # Validar formalmente
        errors = validate(df, st.session_state.input_cols, st.session_state.output_cols)
        llm_ready = errors.get("llm", {}).get("ready", True)

        if errors["formal_issues"] or not llm_ready:
            st.error("Se encontraron problemas en los datos o sugerencias del LLM:")
            if errors["formal_issues"]:
                st.write("– **Problemas formales:**")
                for issue in errors["formal_issues"]:
                    st.write(f"  • {issue}")
            if "issues" in errors["llm"] and errors["llm"]["issues"]:
                st.write("– **Sugerencias del LLM:**")
                for issue in errors["llm"]["issues"]:
                    st.write(f"  • {issue}")
            if "raw" in errors["llm"]:
                st.info(f"Respuesta cruda del LLM: {errors['llm']['raw']}")
            if "suggested_fixes" in errors["llm"] and errors["llm"]["suggested_fixes"]:
                st.write("– **Posibles soluciones del LLM:**")
                for fix in errors["llm"]["suggested_fixes"]:
                    st.write(f"  • {fix}")
        else:
            # Ejecutar DEA y guardarlo en session_state
            with st.spinner("Calculando eficiencias…"):
                res = mostrar_resultados(
                    df.copy(), # Pasar una copia para evitar modificar el original
                    st.session_state.dmu_col,
                    st.session_state.input_cols,
                    st.session_state.output_cols
                )
            # Guardar resultados en session_state
            st.session_state.dea_results = res
            st.success("Cálculo DEA completado.")

    # -------------------------------------------------------
    # 6) Si ya está dea_results en estado, mostrar resultados
    # -------------------------------------------------------
    if "dea_results" in st.session_state and st.session_state.dea_results is not None:
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

        # Verificar si hay al menos dos inputs y un output para el scatter 3D
        if len(st.session_state.input_cols) >= 2 and len(st.session_state.output_cols) >= 1:
            st.subheader("Scatter 3D Inputs vs Output (CCR)")
            st.plotly_chart(resultados["scatter3d_ccr"], use_container_width=True)
            st.subheader("Scatter 3D Inputs vs Output (BCC)")
            st.plotly_chart(resultados["scatter3d_bcc"], use_container_width=True)
        else:
            st.info("Para visualizar el gráfico 3D, se requieren al menos dos inputs y un output.")


        # -------------------------------------------------------
        # 7) Benchmark Spider CCR
        # -------------------------------------------------------
        st.subheader("Benchmark Spider CCR")
        dmu_options = df_ccr[st.session_state.dmu_col].astype(str).tolist() # Usar dmu_col del estado
        
        # Inicializar selected_dmu si no existe o si la DMU previamente seleccionada no está en las opciones actuales
        if "selected_dmu" not in st.session_state or st.session_state.selected_dmu not in dmu_options:
            st.session_state.selected_dmu = dmu_options[0] if dmu_options else None

        selected_dmu_for_spider = st.selectbox(
            "Seleccionar DMU para comparar contra peers eficientes (CCR)",
            dmu_options,
            index=dmu_options.index(st.session_state.selected_dmu) if st.session_state.selected_dmu in dmu_options else 0,
            key="spider_dmu_selector" # Key para estabilidad
        )
        st.session_state.selected_dmu = selected_dmu_for_spider # Actualizar el estado

        if st.session_state.selected_dmu:
            # Generar figura sin recalcular todo el DEA
            # merged_ccr ya está en resultados si lo guardaste ahí, o recálculalo aquí
            if "merged_ccr" in resultados and resultados["merged_ccr"] is not None:
                merged_ccr = resultados["merged_ccr"]
            else:
                merged_ccr = df_ccr.merge(df, on=st.session_state.dmu_col, how="left")
            
            spider_fig = plot_benchmark_spider(
                merged_ccr,
                st.session_state.selected_dmu,
                st.session_state.input_cols,
                st.session_state.output_cols
            )
            st.plotly_chart(spider_fig, use_container_width=True)
        else:
            st.info("No hay DMUs disponibles para el gráfico Spider.")


        # -------------------------------------------------------
        # 8) Guardar sesión (si hay resultados de DEA)
        # -------------------------------------------------------
        st.subheader("Guardar esta sesión")
        
        # Generar un ID de sesión único si no hay una sesión seleccionada para recargar
        current_session_id = selected_session_id if selected_session_id else str(uuid.uuid4())
        st.write(f"ID de sesión actual: `{current_session_id}`")

        # inquiry_tree y eee_score son placeholders, asumir valores por defecto o de la sesión cargada
        inquiry_tree_data = sess.get("inquiry_tree", {"root": "initial_inquiry"}) if selected_session_id else {"root": "new_session"}
        eee_score_data = sess.get("eee_score", 0.0) if selected_session_id else 0.0
        
        notes_text = st.text_area(
            "Notas sobre esta sesión",
            value=sess.get("notes", "") if selected_session_id else "",
            key="session_notes" # Key para estabilidad
        )

        if st.button("Guardar sesión actual", key="save_session_button"):
            try:
                # Asegurarse de pasar el DataFrame original completo a save_session
                # Añadir 'original_dataframe' a la tabla en session_manager si no está
                save_session(
                    session_id=current_session_id,
                    user_id=default_user_id,
                    inquiry_tree=inquiry_tree_data,
                    eee_score=eee_score_data,
                    notes=notes_text,
                    dmu_column=st.session_state.dmu_col,
                    input_cols=st.session_state.input_cols,
                    output_cols=st.session_state.output_cols,
                    df_ccr=st.session_state.dea_results["df_ccr"],
                    df_bcc=st.session_state.dea_results["df_bcc"],
                    # Si quieres guardar el DF original para recarga completa:
                    # original_dataframe=st.session_state.df.to_json(orient='records')
                )
                st.success(f"Sesión `{current_session_id}` guardada correctamente.")
                st.rerun() # Recargar la página para que la sesión guardada aparezca en el sidebar
            except Exception as e:
                st.error(f"Error al guardar la sesión: {e}. Asegúrate de que todos los campos estén correctamente serializables.")


        # -------------------------------------------------------
        # 9) Generar y mostrar enlaces de descarga de reportes
        # -------------------------------------------------------
        st.subheader("Generar reportes")

        # Generar contenido HTML y enlace de descarga
        html_str = generate_html_report(
            df_dea=df_ccr, # Usamos df_ccr como base para el reporte DEA general
            df_tree=pd.DataFrame(),   # Placeholder: Tu DataFrame de árbol si lo tienes (ej. desde inquiry_tree_data)
            df_eee=pd.DataFrame([{"EEE Score": eee_score_data, "Notes": notes_text}]) # Placeholder: DataFrame de EEE
        )
        st.download_button(
            label="Descargar Reporte HTML",
            data=html_str,
            file_name=f"reporte_dea_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            key="download_html"
        )

        # Generar contenido Excel y enlace de descarga
        excel_io = generate_excel_report(
            df_dea=df_ccr, # Usamos df_ccr como base para el reporte DEA general
            df_tree=pd.DataFrame(), # Placeholder: Tu DataFrame de árbol si lo tienes
            df_eee=pd.DataFrame([{"EEE Score": eee_score_data, "Notes": notes_text}]) # Placeholder: DataFrame de EEE
        )
        st.download_button(
            label="Descargar Reporte Excel",
            data=excel_io,
            file_name=f"reporte_dea_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel"
        )

        # Generar contenido PPTX y enlace de descarga (similar a Excel/HTML)
        # Asegúrate de que generate_pptx_report esté implementado y funcione
        # con los mismos placeholders de DataFrames.
        pptx_io = generate_pptx_report(
            df_dea=df_ccr,
            df_tree=pd.DataFrame(),
            df_eee=pd.DataFrame([{"EEE Score": eee_score_data, "Notes": notes_text}])
        )
        st.download_button(
            label="Descargar Reporte PowerPoint",
            data=pptx_io,
            file_name=f"reporte_dea_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            key="download_pptx"
        )
