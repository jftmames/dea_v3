import streamlit as st
import pandas as pd
import json # Necesario para json.loads/dumps si el LLM devuelve diccionarios/listas

from data_validator import validate
from report_generator import generate_html_report, generate_excel_report, generate_pptx_report
from session_manager import init_db, save_session, load_sessions
from results import mostrar_resultados, plot_benchmark_spider
from dea_models.visualizations import plot_slack_waterfall


st.set_page_config(page_title="Simulador Econométrico-Deliberativo", layout="wide")

init_db()
st.title("Simulador Econométrico-Deliberativo")

# 1) Definimos un user_id por defecto. Si tú tuvieras un sistema de login,
#    aquí vendría el user real. Por ahora, usamos un literal:
default_user_id = "default_user" # Defined early to be used by save and load

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
        # Asegúrate de que esta ruta es configurable o correcta para tu corpus
        reindex_rag("path/to/corpus")
        st.sidebar.success("Reindexación completada.")
    except ImportError:
        st.sidebar.error("Módulo 'rag_indexer' no encontrado. Reindexación no disponible.")
    except Exception as e:
        st.sidebar.error(f"Error al reindexar: {e}")

# Inicializar o recuperar estado de la sesión si no existe
if 'df_loaded' not in st.session_state:
    st.session_state.df_loaded = None
    st.session_state.dmu_col_selected = None
    st.session_state.input_cols_selected = []
    st.session_state.output_cols_selected = []
    st.session_state.resultados_dea = None
    st.session_state.llm_response = None # Para guardar la respuesta del LLM
    st.session_state.inquiry_tree = {"root": "Start"} # Placeholder para inquiry_tree
    st.session_state.eee_score = 0.0 # Placeholder para eee_score
    st.session_state.notes = "Notas de la sesión." # Placeholder para notas


# 1) Subida de datos
archivo = st.file_uploader("Sube tu archivo CSV con datos (DMU, inputs, outputs)", type=["csv"])
if archivo is not None:
    st.session_state.df_loaded = pd.read_csv(archivo)

if st.session_state.df_loaded is None:
    st.info("Por favor, sube un archivo CSV para continuar.")
    st.stop()

df = st.session_state.df_loaded


# 2) Seleccionar columna DMU
dmu_col_options = df.columns.tolist()
if st.session_state.dmu_col_selected and st.session_state.dmu_col_selected in dmu_col_options:
    default_dmu_index = dmu_col_options.index(st.session_state.dmu_col_selected)
else:
    default_dmu_index = 0 if dmu_col_options else 0

dmu_col = st.selectbox(
    "Columna DMU (identificador)",
    dmu_col_options,
    index=default_dmu_index,
    help="Seleccione la columna que identifica cada unidad (DMU)."
)
st.session_state.dmu_col_selected = dmu_col


# 3) Seleccionar inputs y outputs
available_cols = [c for c in df.columns if c != dmu_col]

input_cols = st.multiselect(
    "Seleccione columnas de Inputs",
    available_cols,
    default=st.session_state.input_cols_selected if st.session_state.input_cols_selected else [],
    help="Lista de columnas usadas como insumos. Deben ser numéricas y > 0."
)
st.session_state.input_cols_selected = input_cols

output_cols = st.multiselect(
    "Seleccione columnas de Outputs",
    available_cols,
    default=st.session_state.output_cols_selected if st.session_state.output_cols_selected else [],
    help="Lista de columnas usadas como productos. Deben ser numéricas y > 0."
)
st.session_state.output_cols_selected = output_cols


if not dmu_col or len(input_cols) < 1 or len(output_cols) < 1:
    st.info("Debes seleccionar la columna DMU y al menos una columna de input y una de output.")
    st.stop()


# 4) Validar datos
st.subheader("Validación de Datos")
validacion = validate(df, input_cols, output_cols)

if validacion["formal_issues"]:
    st.error("Problemas formales detectados:\n- " + "\n- ".join(validacion["formal_issues"]))
else:
    st.success("Validación formal OK.")

# Mostrar sugerencias del LLM (o error)
llm_json = validacion["llm"]
st.session_state.llm_response = llm_json # Guardar la respuesta del LLM

if "error" in llm_json:
    st.warning(f"Error al consultar LLM: {llm_json['message']}")
else:
    with st.expander("Ver sugerencias del LLM"):
        st.json(llm_json)


# 5) Ejecución de DEA y visualizaciones
if st.button("Ejecutar DEA (CCR y BCC)"):
    with st.spinner("Calculando eficiencias…"):
        resultados = mostrar_resultados(df, dmu_col, input_cols, output_cols)
        st.session_state.resultados_dea = resultados

    st.success("Cálculos DEA completados.")
    st.experimental_rerun() # Disparar un re-run para mostrar los resultados y botones de descarga


if st.session_state.resultados_dea: # Solo mostrar si ya se ejecutó DEA
    resultados = st.session_state.resultados_dea

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

    # Benchmark Spider (CCR)
    merged_ccr = resultados["merged_ccr"]
    st.subheader("Benchmark Spider (CCR)")
    
    # Asegurarse de que la columna 'DMU' exista y manejar problemas de conversión de cadena
    if dmu_col not in merged_ccr.columns: # Usar dmu_col directamente
        st.error(f"La columna '{dmu_col}' no se encontró en los resultados de CCR para el spider plot.")
    else:
        # Peers eficientes para el selectbox
        # Es importante usar el nombre de la columna real (dmu_col) en lugar de un literal "DMU"
        efficient_dmus_ccr = merged_ccr[merged_ccr["efficiency"] == 1][dmu_col].astype(str).tolist()
        if efficient_dmus_ccr:
            selected_dmu_spider_ccr = st.selectbox(
                "Seleccione una DMU para benchmark (CCR)",
                options=efficient_dmus_ccr,
                key="spider_ccr_dmu_select"
            )
            if selected_dmu_spider_ccr: # Verificar si se seleccionó una DMU
                # plot_benchmark_spider espera la columna DMU en merged_ccr
                fig_spider_ccr = plot_benchmark_spider(merged_ccr, selected_dmu_spider_ccr, input_cols, output_cols)
                st.plotly_chart(fig_spider_ccr, use_container_width=True)
        else:
            st.info("No hay DMUs eficientes CCR para benchmarking con el spider plot.")

    # Diagnóstico de slacks CCR
    st.subheader("Diagnóstico de Slacks CCR")
    if dmu_col not in merged_ccr.columns:
        st.error(f"La columna '{dmu_col}' no se encontró en los resultados de CCR para el waterfall plot.")
    else:
        all_dmus_ccr = merged_ccr[dmu_col].astype(str).tolist()
        if all_dmus_ccr:
            sel_dmu_waterfall_ccr = st.selectbox(
                "Seleccione DMU para ver slacks (CCR)",
                options=all_dmus_ccr,
                key="waterfall_ccr_dmu_select"
            )
            if sel_dmu_waterfall_ccr:
                # Filtrar usando el nombre de la columna DMU
                dmu_data_ccr = merged_ccr[merged_ccr[dmu_col].astype(str) == sel_dmu_waterfall_ccr]
                if not dmu_data_ccr.empty:
                    # Asegurar que slacks_inputs y slacks_outputs sean diccionarios
                    slacks_in_ccr = dmu_data_ccr["slacks_inputs"].iloc[0]
                    slacks_out_ccr = dmu_data_ccr["slacks_outputs"].iloc[0]
                    
                    if not isinstance(slacks_in_ccr, dict): 
                        try:
                            slacks_in_ccr = json.loads(slacks_in_ccr)
                        except (json.JSONDecodeError, TypeError):
                            slacks_in_ccr = {}
                    if not isinstance(slacks_out_ccr, dict): 
                        try:
                            slacks_out_ccr = json.loads(slacks_out_ccr)
                        except (json.JSONDecodeError, TypeError):
                            slacks_out_ccr = {}

                    # Llamar a plot_slack_waterfall con diccionarios separados
                    fig_water_ccr = plot_slack_waterfall(slacks_in_ccr, slacks_out_ccr, sel_dmu_waterfall_ccr)
                    st.pyplot(fig_water_ccr)
                else:
                    st.error(f"No se encontraron datos de slacks para la DMU '{sel_dmu_waterfall_ccr}'.")
        else:
            st.info("No hay DMUs disponibles en los resultados CCR para el diagnóstico de slacks.")

    st.markdown("---")
    st.subheader("Guardar y Cargar Sesión")

    # Campos adicionales para guardar la sesión
    st.session_state.inquiry_tree = st.text_area("Árbol de Indagación (JSON)", json.dumps(st.session_state.inquiry_tree, indent=2), help="Representación JSON del árbol de indagación.")
    st.session_state.eee_score = st.number_input("Puntaje EEE", value=st.session_state.eee_score, help="Puntaje de Eficiencia, Efectividad y Equidad.")
    st.session_state.notes = st.text_area("Notas de la Sesión", st.session_state.notes, help="Notas adicionales sobre la sesión o el análisis.")


    if st.button("Guardar Sesión Actual"):
        try:
            # Generar un session_id único (simple timestamp)
            session_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_" + default_user_id
            
            # Asegúrate de que inquiry_tree sea un diccionario
            inquiry_tree_dict = json.loads(st.session_state.inquiry_tree) if isinstance(st.session_state.inquiry_tree, str) else st.session_state.inquiry_tree

            save_session(
                session_id=session_id,
                user_id=default_user_id,
                inquiry_tree=inquiry_tree_dict,
                eee_score=float(st.session_state.eee_score),
                notes=st.session_state.notes,
                dmu_column=dmu_col,
                input_cols=input_cols,
                output_cols=output_cols,
                df_ccr=resultados["df_ccr"],
                df_bcc=resultados["df_bcc"]
            )
            st.success(f"Sesión '{session_id}' guardada para el usuario {default_user_id}.")
        except Exception as e:
            st.error(f"Error al guardar la sesión: {e}")


# Report generation buttons - moved outside the main DEA execution button for better UX
# They will use st.session_state.resultados_dea if available
if 'resultados_dea' in st.session_state and st.session_state.resultados_dea is not None:
    st.markdown("---")
    st.subheader("Descargar Reportes")
    
    # Retrieve data from session state
    resultados_report = st.session_state.resultados_dea
    df_original_report = st.session_state.df_loaded
    dmu_col_report = st.session_state.dmu_col_selected
    input_cols_report = st.session_state.input_cols_selected
    output_cols_report = st.session_state.output_cols_selected

    cols1, cols2, cols3 = st.columns(3)

    # Preparar DataFrames para el reporte (si son placeholders, rellenarlos si es necesario)
    # df_tree y df_eee deberían venir de algún lugar del flujo de la app o ser placeholders.
    # Por ahora, usamos placeholders o la información de la sesión.
    df_tree_report = pd.DataFrame([{"key": k, "value": v} for k,v in st.session_state.inquiry_tree.items()])
    df_eee_report = pd.DataFrame([{"Métrico": "EEE Score", "Valor": st.session_state.eee_score},
                                  {"Métrico": "Notas", "Valor": st.session_state.notes}])

    with cols1:
        if st.button("Generar Reporte HTML", key="gen_html_btn"):
            with st.spinner("Generando Reporte HTML..."):
                html_content = generate_html_report(
                    df_dea=resultados_report["df_ccr"], # Puedes elegir entre df_ccr o df_bcc o ambos
                    df_tree=df_tree_report,
                    df_eee=df_eee_report
                )
                st.download_button(
                    label="Descargar Reporte HTML",
                    data=html_content,
                    file_name="reporte_dea_deliberativo.html",
                    mime="text/html",
                    key="download_html_btn"
                )
    with cols2:
        if st.button("Generar Reporte Excel", key="gen_excel_btn"):
            with st.spinner("Generando Reporte Excel..."):
                excel_io = generate_excel_report(
                    df_dea=resultados_report["df_ccr"], # Puedes elegir entre df_ccr o df_bcc o ambos
                    df_tree=df_tree_report,
                    df_eee=df_eee_report
                )
                st.download_button(
                    label="Descargar Reporte Excel",
                    data=excel_io,
                    file_name="reporte_dea_deliberativo.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel_btn"
                )
    with cols3:
        if st.button("Generar Reporte PPTX", key="gen_pptx_btn"):
            with st.spinner("Generando Reporte PPTX..."):
                pptx_io = generate_pptx_report(
                    df_dea=resultados_report["df_ccr"], # Puedes elegir entre df_ccr o df_bcc o ambos
                    df_tree=df_tree_report,
                    df_eee=df_eee_report
                )
                st.download_button(
                    label="Descargar Reporte PPTX",
                    data=pptx_io,
                    file_name="reporte_dea_deliberativo.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    key="download_pptx_btn"
                )


# MODIFIED SECTION FOR LOADING SESSIONS:
st.sidebar.subheader("Sesiones Guardadas")

# 2) Llamamos a load_sessions pasándole ese user_id:
sessions = load_sessions(user_id=default_user_id) 

if sessions:
    st.sidebar.markdown("Haz clic en una sesión para cargarla (funcionalidad no implementada en este ejemplo).")
    for i, sess in enumerate(sessions, start=1):
        # Displaying session_id and timestamp as per new requirement
        session_display_name = sess.get('session_id', f"Sesión {i}") # Usamos session_id como nombre si no hay
        timestamp = sess.get('timestamp', 'N/A')
        
        # Make it a button to allow reloading (basic example)
        if st.sidebar.button(f"Cargar: {session_display_name} ({timestamp})", key=f"load_sess_{session_display_name}"):
            st.sidebar.info(f"Cargando sesión '{session_display_name}'...")
            
            # Actualizar st.session_state con los datos de la sesión cargada
            st.session_state.df_loaded = st.session_state.get('df_loaded', pd.DataFrame()) # Mantener el original si no se sobreescribe
            # Aquí asumimos que dmu_column, input_cols, output_cols vienen directamente de la sesión
            st.session_state.dmu_col_selected = sess.get('dmu_column', None)
            st.session_state.input_cols_selected = sess.get('input_cols', [])
            st.session_state.output_cols_selected = sess.get('output_cols', [])
            st.session_state.resultados_dea = {
                "df_ccr": sess.get('df_ccr', pd.DataFrame()),
                "df_bcc": sess.get('df_bcc', pd.DataFrame()),
                # Nota: Las figuras (hist, scatter) no se guardan directamente.
                # Tendrías que recrearlas al cargar la sesión si quieres verlas de nuevo.
                # Aquí las generamos para la carga simulada
                "hist_ccr": plot_efficiency_histogram(sess['df_ccr']) if not sess['df_ccr'].empty else None,
                "hist_bcc": plot_efficiency_histogram(sess['df_bcc']) if not sess['df_bcc'].empty else None,
                "scatter3d_ccr": plot_3d_inputs_outputs(df, sess['input_cols'], sess['output_cols'], sess['df_ccr'], sess['dmu_column']) if not sess['df_ccr'].empty else None,
                "scatter3d_bcc": plot_3d_inputs_outputs(df, sess['input_cols'], sess['output_cols'], sess['df_bcc'], sess['dmu_column']) if not sess['df_bcc'].empty else None,
                # También necesitas merged_ccr y merged_bcc para el spider/waterfall si se recalculan
                "merged_ccr": sess['df_ccr'].merge(df, on=sess['dmu_column'], how="left") if not sess['df_ccr'].empty and sess['dmu_column'] in df.columns else pd.DataFrame(),
                "merged_bcc": sess['df_bcc'].merge(df, on=sess['dmu_column'], how="left") if not sess['df_bcc'].empty and sess['dmu_column'] in df.columns else pd.DataFrame(),
            }
            # Cargar los metadatos de la sesión
            st.session_state.inquiry_tree = sess.get('inquiry_tree', {})
            st.session_state.eee_score = sess.get('eee_score', 0.0)
            st.session_state.notes = sess.get('notes', "")

            st.sidebar.success(f"Sesión '{session_display_name}' cargada. Por favor, revisa las selecciones y ejecuta DEA si es necesario.")
            # Un rerun es necesario para que los selectbox y multiselectbox muestren los valores cargados.
            st.experimental_rerun() 

else:
    st.sidebar.info(f"No hay sesiones guardadas para el usuario {default_user_id}.")
