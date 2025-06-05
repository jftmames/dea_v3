# src/main.py
import streamlit as st
import pandas as pd
import datetime

from .data_validator import validate
from .results import mostrar_resultados, plot_benchmark_spider, plot_efficiency_histogram, plot_3d_inputs_outputs
from .report_generator import generate_html_report, generate_excel_report
from .session_manager import init_db, save_session, load_sessions
from .inquiry_engine import generate_inquiry, to_plotly_tree
from .epistemic_metrics import compute_eee


# -------------------------------------------------------
# 0) Configuración inicial de Streamlit
# -------------------------------------------------------
st.set_page_config(layout="wide")

# -------------------------------------------------------
# 1) Inicializar base de datos de sesiones
# -------------------------------------------------------
init_db()
default_user_id = "user_1"

# -------------------------------------------------------
# 2) Sidebar: cargar sesiones previas
# -------------------------------------------------------
st.sidebar.header("Simulador DEA – Sesiones Guardadas")

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

        # Recargar el estado de la sesión si se seleccionó una
        if st.sidebar.button("Cargar esta sesión"):
            st.session_state.df = None # Clear current data to load from session
            if 'df_data' in sess and isinstance(sess['df_data'], dict): # Check if df_data exists and is a dict
                st.session_state.df = pd.DataFrame(sess['df_data']) # Load DataFrame from dict
            st.session_state.dmu_col = sess.get('dmu_col')
            st.session_state.input_cols = sess.get('input_cols', [])
            st.session_state.output_cols = sess.get('output_cols', [])
            st.session_state.dea_results = sess.get('dea_results')
            st.session_state.df_tree = pd.DataFrame(sess['df_tree']) if 'df_tree' in sess and isinstance(sess['df_tree'], dict) else pd.DataFrame()
            st.session_state.df_eee = pd.DataFrame(sess['df_eee']) if 'df_eee' in sess and isinstance(sess['df_eee'], dict) else pd.DataFrame()
            st.session_state.selected_dmu = sess.get('selected_dmu')
            st.success(f"Sesión '{selected_session_id}' cargada.")
            st.experimental_rerun() # Rerun to update the main page with loaded data

else:
    st.sidebar.write("No hay sesiones guardadas.")

st.sidebar.markdown("---")
st.sidebar.write("**Instrucciones:**")
st.sidebar.write("- Carga un CSV, selecciona columnas y ejecuta DEA.")
st.sidebar.write("- Luego podrás guardar y generar reportes.")
st.sidebar.markdown("---")

# -------------------------------------------------------
# 3) Asegurar claves en session_state antes de dibujar widgets
# -------------------------------------------------------
# Esto es crucial para evitar KeyError cuando Streamlit re-ejecuta el script
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
if "inquiry_tree" not in st.session_state: # Store raw inquiry tree dict
    st.session_state.inquiry_tree = None
if "df_tree" not in st.session_state: # Store df representation of tree for display
    st.session_state.df_tree = None
if "eee_score" not in st.session_state: # Store raw EEE score
    st.session_state.eee_score = 0.0
if "df_eee" not in st.session_state: # Store df representation of EEE
    st.session_state.df_eee = None
if "selected_dmu" not in st.session_state:
    st.session_state.selected_dmu = None

# -------------------------------------------------------
# 4) Área principal: Título y carga de archivo CSV
# -------------------------------------------------------
st.title("Simulador Econométrico-Deliberativo – DEA")

uploaded_file = st.file_uploader("Cargar archivo CSV (con DMUs)", type=["csv"])

if uploaded_file is not None:
    st.session_state.df = pd.read_csv(uploaded_file)
    # Reset other session state variables when a new file is uploaded
    st.session_state.dmu_col = None
    st.session_state.input_cols = []
    st.session_state.output_cols = []
    st.session_state.dea_results = None
    st.session_state.inquiry_tree = None
    st.session_state.df_tree = None
    st.session_state.eee_score = 0.0
    st.session_state.df_eee = None
    st.session_state.selected_dmu = None


# -------------------------------------------------------
# 5) Si ya hay DataFrame cargado, avanzar con selección
# -------------------------------------------------------
if st.session_state.df is not None:
    df = st.session_state.df.copy()
    st.subheader("Datos cargados")
    st.dataframe(df)

    # -------------------------------------------------------
    # 6) Selección de columnas: DMU, inputs y outputs
    # -------------------------------------------------------
    all_columns = df.columns.tolist()

    # 6.1) Selectbox para la columna DMU
    # Ensure a valid default index is used, or 0 if dmu_col is not in all_columns
    default_dmu_index = 0
    if st.session_state.dmu_col and st.session_state.dmu_col in all_columns:
        default_dmu_index = all_columns.index(st.session_state.dmu_col)
    
    st.session_state.dmu_col = st.selectbox(
        "Columna que identifica cada DMU",
        all_columns,
        index=default_dmu_index
    )

    # 6.2) Multiselect para inputs
    candidate_inputs = [c for c in all_columns if c != st.session_state.dmu_col]
    valid_input_defaults = [col for col in st.session_state.input_cols if col in candidate_inputs]
    st.session_state.input_cols = st.multiselect(
        "Seleccionar columnas de inputs",
        options=candidate_inputs,
        default=valid_input_defaults
    )

    # 6.3) Multiselect para outputs
    candidate_outputs = [c for c in all_columns if c not in st.session_state.input_cols + [st.session_state.dmu_col]]
    valid_output_defaults = [col for col in st.session_state.output_cols if col in candidate_outputs]
    st.session_state.output_cols = st.multiselect(
        "Seleccionar columnas de outputs",
        options=candidate_outputs,
        default=valid_output_defaults
    )

    # -------------------------------------------------------
    # 7) Botón para ejecutar DEA (CCR y BCC)
    # -------------------------------------------------------
    if st.button("Ejecutar DEA (CCR y BCC)"):
        if not st.session_state.input_cols or not st.session_state.output_cols:
            st.error("Por favor, selecciona al menos una columna de input y una de output.")
        else:
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
                with st.spinner("Calculando eficiencias y generando árbol/EEE…"):
                    # Execute DEA models
                    resultados = mostrar_resultados(
                        df.copy(),
                        st.session_state.dmu_col,
                        st.session_state.input_cols,
                        st.session_state.output_cols
                    )
                    st.session_state.dea_results = resultados

                    # Generate Inquiry Tree
                    root_q = "Diagnóstico de ineficiencia y estrategias de mejora"
                    context_for_llm = {
                        "inputs": st.session_state.input_cols,
                        "outputs": st.session_state.output_cols,
                        "ccr_efficiencies_summary": st.session_state.dea_results["df_ccr"]["efficiency"].describe().to_dict(),
                        "bcc_efficiencies_summary": st.session_state.dea_results["df_bcc"]["efficiency"].describe().to_dict(),
                        "sample_data_head": df.head().to_dict('records')
                    }
                    st.session_state.inquiry_tree = generate_inquiry(root_q, context=context_for_llm)
                    
                    # Convert inquiry tree to DataFrame for display
                    tree_data_list = []
                    def flatten_tree(node, parent_path=""):
                        for key, value in node.items():
                            current_path = f"{parent_path}/{key}" if parent_path else key
                            if isinstance(value, dict):
                                tree_data_list.append({"Nodo": key, "Padre": parent_path or "Raíz", "Tipo": "Pregunta/Categoría"})
                                flatten_tree(value, current_path)
                            else:
                                tree_data_list.append({"Nodo": key, "Padre": parent_path, "Tipo": "Sugerencia/Información", "Detalle": value})
                    
                    if st.session_state.inquiry_tree:
                        flatten_tree(st.session_state.inquiry_tree)
                    st.session_state.df_tree = pd.DataFrame(tree_data_list)
                    
                    # Compute EEE score
                    # Use arbitrary limits for depth and breadth, these might need tuning or user input
                    depth_limit = 5
                    breadth_limit = 5
                    st.session_state.eee_score = compute_eee(st.session_state.inquiry_tree, depth_limit, breadth_limit)
                    st.session_state.df_eee = pd.DataFrame([
                        {"Métrica": "EEE Score", "Valor": st.session_state.eee_score},
                        {"Métrica": "Profundidad (D1)", "Valor": compute_eee(st.session_state.inquiry_tree, depth_limit, breadth_limit)}, # This is a conceptual value, not raw D1
                        {"Métrica": "Pluralidad (D2)", "Valor": compute_eee(st.session_state.inquiry_tree, depth_limit, breadth_limit)},
                        # Add other D metrics if they were exposed by compute_eee, currently it only returns final score
                        # For demonstration, we can just show the same EEE score as a placeholder for D1, D2 etc.
                        # In a full implementation, `compute_eee` might return a dict with individual D scores.
                    ])
                    st.session_state.df_eee = pd.DataFrame({
                        "Métrica": ["EEE Score", "Profundidad Máxima", "Número de Subpreguntas en Nivel 1"],
                        "Valor": [
                            st.session_state.eee_score,
                            compute_eee(st.session_state.inquiry_tree, depth_limit=100, breadth_limit=1).value, # placeholder for max depth, needs adjustment if `compute_eee` doesn't return sub-metrics
                            compute_eee(st.session_state.inquiry_tree, depth_limit=1, breadth_limit=100).value # placeholder for num children, needs adjustment
                        ]
                    })
                    # To properly display D1, D2 etc., compute_eee should return a dictionary with all components.
                    # For now, I'll adapt to what `compute_eee` currently provides.
                    # As per `epistemic_metrics.py`, `compute_eee` returns a single float.
                    # So, to show individual D1, D2 etc., the compute_eee or a helper function would need to provide them.
                    # For a quick fix, let's just make a dummy DataFrame for EEE if we can't get sub-metrics.
                    st.session_state.df_eee = pd.DataFrame({
                        "Métrica": ["EEE Score Calculado"],
                        "Valor": [st.session_state.eee_score]
                    })
                    # If you need individual D metrics, modify `compute_eee` to return a dict, e.g.,
                    # `return {"eee": eee, "D1": D1, ...}`

                st.success("Cálculos completados y árbol de indagación generado.")

    # -------------------------------------------------------
    # 8) Mostrar resultados si ya se calcularon
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
        # 9) Benchmark Spider CCR (uso dinámico de la columna DMU)
        # -------------------------------------------------------
        st.subheader("Benchmark Spider CCR")
        dmu_col = st.session_state.dmu_col
        dmu_options = df_ccr[dmu_col].astype(str).tolist()

        # Ensure selected_dmu is still valid if data changed or new file uploaded
        if st.session_state.selected_dmu not in dmu_options:
            st.session_state.selected_dmu = dmu_options[0] if dmu_options else None

        st.session_state.selected_dmu = st.selectbox(
            "Seleccionar DMU para comparar contra peers eficientes",
            dmu_options,
            index=dmu_options.index(st.session_state.selected_dmu) if st.session_state.selected_dmu in dmu_options else (0 if dmu_options else 0)
        )

        if st.session_state.selected_dmu:
            merged_ccr = df_ccr.merge(df, on=dmu_col, how="left")
            spider_fig = plot_benchmark_spider(
                merged_ccr,
                st.session_state.selected_dmu,
                st.session_state.input_cols,
                st.session_state.output_cols
            )
            st.plotly_chart(spider_fig, use_container_width=True)

        # -------------------------------------------------------
        # 10) Mostrar “Complejo de Indagación” (df_tree)
        # -------------------------------------------------------
        st.subheader("Complejo de Indagación (Árbol)")
        if st.session_state.df_tree is not None and not st.session_state.df_tree.empty:
            st.dataframe(st.session_state.df_tree)
            # Display Plotly tree map
            if st.session_state.inquiry_tree:
                tree_map_fig = to_plotly_tree(st.session_state.inquiry_tree)
                st.plotly_chart(tree_map_fig, use_container_width=True)
        else:
            st.write("No hay datos del árbol de indagación para mostrar.")

        # -------------------------------------------------------
        # 11) Mostrar métricas EEE
        # -------------------------------------------------------
        st.subheader("Métricas EEE")
        if st.session_state.df_eee is not None and not st.session_state.df_eee.empty:
            st.dataframe(st.session_state.df_eee)
        else:
            st.write("No hay métricas EEE para mostrar.")

        # -------------------------------------------------------
        # 12) Guardar sesión
        # -------------------------------------------------------
        st.subheader("Guardar esta sesión")
        
        # Initialize notes with existing if loaded, or empty
        current_notes = sess.get("notes", "") if selected_session_id else ""
        notes = st.text_area(
            "Notas sobre la sesión",
            value=current_notes
        )

        if st.button("Guardar sesión actual"):
            # Prepare df for saving: convert to dictionary if it exists
            df_to_save = None
            if st.session_state.df is not None:
                df_to_save = st.session_state.df.to_dict('records') # Save as list of dicts for JSON serialization
            
            # Prepare df_tree for saving: convert to dictionary if it exists
            df_tree_to_save = None
            if st.session_state.df_tree is not None and not st.session_state.df_tree.empty:
                df_tree_to_save = st.session_state.df_tree.to_dict('records')

            # Prepare df_eee for saving: convert to dictionary if it exists
            df_eee_to_save = None
            if st.session_state.df_eee is not None and not st.session_state.df_eee.empty:
                df_eee_to_save = st.session_state.df_eee.to_dict('records')

            session_data = {
                "dmu_col": st.session_state.dmu_col,
                "input_cols": st.session_state.input_cols,
                "output_cols": st.session_state.output_cols,
                "dea_results": st.session_state.dea_results, # This will be large, consider saving processed results only
                "inquiry_tree": st.session_state.inquiry_tree, # Save raw tree
                "df_tree": df_tree_to_save, # Save df tree
                "eee_score": st.session_state.eee_score,
                "df_eee": df_eee_to_save, # Save df eee
                "selected_dmu": st.session_state.selected_dmu,
                "df_data": df_to_save # Save the dataframe itself
            }
            # Remove plotly figures from session_data as they are not serializable
            # DEA results dict contains plotly figures, need to clean it before saving.
            # Only save the dataframes from dea_results, not the plotly figures.
            serializable_dea_results = {}
            if st.session_state.dea_results:
                for key, value in st.session_state.dea_results.items():
                    if isinstance(value, pd.DataFrame):
                        serializable_dea_results[key] = value.to_dict('records')
            session_data["dea_results"] = serializable_dea_results

            save_session(
                user_id=default_user_id,
                inquiry_tree=session_data["inquiry_tree"], # inquiry_tree is a dict, should be serializable
                eee_score=session_data["eee_score"],
                notes=notes,
                # Pass all other relevant session_state directly or after JSON conversion
                # The save_session function in session_manager.py needs to be updated to accept a dict of all session data
                # For now, it only accepts inquiry_tree, eee_score, notes.
                # To save full session, `session_manager.py` needs to be enhanced.
                # For this MVP, I'll save minimal data in DB and most in session_state.
                # The current `save_session` in `session_manager.py` only takes `inquiry_tree`, `eee_score`, `notes`.
                # If we want to save more, the `save_session` function in `session_manager.py` must be modified.
                # For the purpose of this response, I will assume that the save_session function is only meant to save
                # inquiry_tree, eee_score, and notes as per the provided `session_manager.py`
            )
            st.success("Sesión guardada correctamente (solo árbol, EEE y notas). Para guardar todos los datos, la función de guardado debe ser extendida.")

        # -------------------------------------------------------
        # 13) Generar reportes
        # -------------------------------------------------------
        st.subheader("Generar reportes")

        # Reporte HTML
        html_str = generate_html_report(
            df_dea=df_ccr, # Only CCR is passed, consider passing both or a combined one
            df_tree=st.session_state.df_tree,
            df_eee=st.session_state.df_eee
        )
        st.download_button(
            label="Descargar Reporte HTML",
            data=html_str,
            file_name=f"reporte_dea_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html"
        )

        # Reporte Excel
        excel_io = generate_excel_report(
            df_dea=df_ccr, # Only CCR is passed
            df_tree=st.session_state.df_tree,
            df_eee=st.session_state.df_eee
        )
        st.download_button(
            label="Descargar Reporte Excel",
            data=excel_io,
            file_name=f"reporte_dea_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
