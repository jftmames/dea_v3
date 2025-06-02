import streamlit as st
import pandas as pd

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
        reindex_rag("path/to/corpus") # Asegúrate que esta ruta es configurable o correcta
        st.sidebar.success("Reindexación completada.")
    except ImportError:
        st.sidebar.error("Módulo 'rag_indexer' no encontrado. Reindexación no disponible.")
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

if not dmu_col or len(input_cols) < 1 or len(output_cols) < 1:
    st.info("Debes seleccionar la columna DMU y al menos una columna de input y una de output.")
    st.stop()

# 4) Validar datos
# Asumiendo que 'validate' no necesita user_id
validacion = validate(df, input_cols, output_cols) # Pasar dmu_col si es necesario para validate
if validacion["formal_issues"]:
    st.error("Problemas formales detectados:\n- " + "\n- ".join(validacion["formal_issues"]))
else:
    st.success("Validación formal OK")

# Mostrar sugerencias del LLM (o error)
llm_json = validacion["llm"]
if "error" in llm_json:
    st.warning(f"Error al consultar LLM: {llm_json['message']}")
else:
    st.json(llm_json) # Considerar st.expander para no ocupar mucho espacio

# 5) Ejecución de DEA y visualizaciones
if st.button("Ejecutar DEA (CCR y BCC)"):
    with st.spinner("Calculando eficiencias…"):
        # Asumiendo que mostrar_resultados no necesita user_id directamente
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

    # Benchmark Spider (CCR)
    merged_ccr = resultados["merged_ccr"]
    st.subheader("Benchmark Spider (CCR)")
    
    # Ensure 'DMU' column exists and handle potential string conversion issues
    if "DMU" not in merged_ccr.columns:
        st.error("La columna 'DMU' no se encontró en los resultados de CCR para el spider plot.")
    else:
        # Efficient peers for selectbox
        efficient_dmus_ccr = merged_ccr[merged_ccr["efficiency"] == 1]["DMU"].astype(str).tolist()
        if efficient_dmus_ccr:
            selected_dmu_spider_ccr = st.selectbox(
                "Seleccione una DMU para benchmark (CCR)",
                options=efficient_dmus_ccr,
                key="spider_ccr_dmu_select"
            )
            if selected_dmu_spider_ccr: # Check if a DMU is selected
                # plot_benchmark_spider expects 'DMU' column in merged_ccr
                fig_spider_ccr = plot_benchmark_spider(merged_ccr, selected_dmu_spider_ccr, input_cols, output_cols)
                st.plotly_chart(fig_spider_ccr, use_container_width=True)
        else:
            st.info("No hay DMUs eficientes CCR para benchmarking con el spider plot.")


    # Diagnóstico de slacks CCR
    st.subheader("Diagnóstico de Slacks CCR")
    if "DMU" not in merged_ccr.columns:
        st.error("La columna 'DMU' no se encontró en los resultados de CCR para el waterfall plot.")
    else:
        all_dmus_ccr = merged_ccr["DMU"].astype(str).tolist()
        if all_dmus_ccr:
            sel_dmu_waterfall_ccr = st.selectbox(
                "Seleccione DMU para ver slacks (CCR)",
                options=all_dmus_ccr,
                key="waterfall_ccr_dmu_select"
            )
            if sel_dmu_waterfall_ccr:
                dmu_data_ccr = merged_ccr[merged_ccr["DMU"].astype(str) == sel_dmu_waterfall_ccr]
                if not dmu_data_ccr.empty:
                    # Ensure slacks_inputs and slacks_outputs are dictionaries
                    slacks_in_ccr = dmu_data_ccr["slacks_inputs"].iloc[0]
                    slacks_out_ccr = dmu_data_ccr["slacks_outputs"].iloc[0]
                    
                    if not isinstance(slacks_in_ccr, dict): slacks_in_ccr = {}
                    if not isinstance(slacks_out_ccr, dict): slacks_out_ccr = {}

                    # Call plot_slack_waterfall with separate dicts as per its definition
                    fig_water_ccr = plot_slack_waterfall(slacks_in_ccr, slacks_out_ccr, sel_dmu_waterfall_ccr)
                    st.pyplot(fig_water_ccr)
                else:
                    st.error(f"No se encontraron datos de slacks para la DMU '{sel_dmu_waterfall_ccr}'.")
        else:
            st.info("No hay DMUs disponibles en los resultados CCR para el diagnóstico de slacks.")


    # 6) Generar link de descarga de reporte HTML
    # Buttons for reports should ideally be outside the main "Ejecutar DEA" button's conditional block
    # if they depend on 'resultados', or 'resultados' should be stored in session state.
    # For now, keeping them inside as per original structure.

    # Store results in session state to make them accessible for report generation
    st.session_state.resultados_dea = resultados 
    st.session_state.df_original = df 
    st.session_state.dmu_col = dmu_col
    st.session_state.input_cols = input_cols
    st.session_state.output_cols = output_cols

    # 9) Guardar sesión de resultados
    # save_session likely needs user_id if load_sessions uses it.
    save_session(
        user_id=default_user_id, # Added user_id
        dmu_column=dmu_col,
        input_cols=input_cols,
        output_cols=output_cols,
        results=resultados # Ensure 'resultados' is serializable (e.g., DataFrames to dicts if needed by backend)
    )
    st.success(f"Resultados de la sesión guardados para el usuario {default_user_id}.")


# Report generation buttons - moved outside the main DEA execution button for better UX
# They will use st.session_state.resultados_dea if available
if 'resultados_dea' in st.session_state:
    st.markdown("---")
    st.subheader("Descargar Reportes")
    
    # Retrieve data from session state
    resultados_report = st.session_state.resultados_dea
    # df_report = st.session_state.df_original # If needed by reports
    # dmu_col_report = st.session_state.dmu_col
    # input_cols_report = st.session_state.input_cols
    # output_cols_report = st.session_state.output_cols

    cols1, cols2, cols3 = st.columns(3)

    with cols1:
        if st.button("Generar Reporte HTML"):
            with st.spinner("Generando Reporte HTML..."):
                html_content = generate_html_report(
                    df_dea=resultados_report["df_ccr"], # Example: using CCR results
                    # Pass other necessary dataframes if your report generator uses them
                    df_tree=pd.DataFrame(), # Placeholder
                    df_eee=pd.DataFrame()   # Placeholder
                )
                st.download_button(
                    label="Descargar Reporte HTML",
                    data=html_content,
                    file_name="reporte_dea_deliberativo.html",
                    mime="text/html"
                )
    with cols2:
        if st.button("Generar Reporte Excel"):
            with st.spinner("Generando Reporte Excel..."):
                excel_io = generate_excel_report(
                    df_dea=resultados_report["df_ccr"], # Example: using CCR results
                    df_tree=pd.DataFrame(),  # Placeholder
                    df_eee=pd.DataFrame()    # Placeholder
                )
                st.download_button(
                    label="Descargar Reporte Excel",
                    data=excel_io,
                    file_name="reporte_dea_deliberativo.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    with cols3:
        if st.button("Generar Reporte PPTX"):
            with st.spinner("Generando Reporte PPTX..."):
                pptx_io = generate_pptx_report(
                    df_dea=resultados_report["df_ccr"], # Example: using CCR results
                    df_tree=pd.DataFrame(),  # Placeholder
                    df_eee=pd.DataFrame()    # Placeholder
                )
                st.download_button(
                    label="Descargar Reporte PPTX",
                    data=pptx_io,
                    file_name="reporte_dea_deliberativo.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                )

# MODIFIED SECTION FOR LOADING SESSIONS:
# Después (usamos "default_user" como ejemplo; reemplaza según tu lógica de autenticación):
st.sidebar.subheader("Sesiones Guardadas")

# 1) Definimos un user_id por defecto. Si tú tuvieras un sistema de login,
#    aquí vendría el user real. Por ahora, usamos un literal (defined at the top):
# default_user_id = "default_user" # Already defined

# 2) Llamamos a load_sessions pasándole ese user_id:
# Assuming load_sessions is adapted to take user_id
sessions = load_sessions(user_id=default_user_id) 

if sessions:
    st.sidebar.markdown("Haz clic en una sesión para cargarla (funcionalidad no implementada en este ejemplo).")
    for i, sess in enumerate(sessions, start=1):
        # Displaying session_id and timestamp as per new requirement
        # The actual loading logic upon clicking would need to be implemented
        # e.g., using a button or making the markdown a link that sets session state.
        
        # For simplicity, just displaying info. A button could trigger loading.
        session_display_name = sess.get('session_name', sess.get('session_id', f"Sesión {sess.get('id', i)}"))
        timestamp = sess.get('timestamp', 'N/A')
        
        # Make it a button to allow reloading (basic example)
        if st.sidebar.button(f"Cargar: {session_display_name} ({timestamp})", key=f"load_sess_{sess.get('id', i)}"):
            st.info(f"Funcionalidad de recargar sesión '{session_display_name}' no implementada completamente.")
            # Here you would typically:
            # 1. Fetch the full session data using session_id (sess.get('id') or sess.get('session_id'))
            # 2. Populate st.session_state with the data from the loaded session
            #    e.g., st.session_state.df_original = pd.DataFrame(sess['data']['df_original_dict'])
            #    st.session_state.dmu_col = sess['data']['dmu_col']
            #    ... and so on for input_cols, output_cols, results_dea
            # 3. Potentially re-run parts of the DEA display or allow user to proceed
            # st.experimental_rerun()
            st.sidebar.info(f"Simulando carga de {session_display_name}. DMU: {sess.get('dmu_column', 'N/A')}, Inputs: {sess.get('input_cols', 'N/A')}, Outputs: {sess.get('output_cols', 'N/A')}")
            # To truly load, you'd need to restore df, dmu_col, input_cols, output_cols, and re-display results.

else:
    st.sidebar.info(f"No hay sesiones guardadas para el usuario {default_user_id}.")
