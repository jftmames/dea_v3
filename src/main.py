import sys
import os

# -------------------------------------------------------
# 0) Ajuste del PYTHONPATH: añadimos la carpeta 'src' al sys.path
# -------------------------------------------------------
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import streamlit as st
import pandas as pd
import datetime

# -------------------------------------------------------
# 1) Importaciones
# -------------------------------------------------------
from data_validator import validate
from results import mostrar_resultados
from report_generator import generate_html_report, generate_excel_report
from session_manager import init_db, save_session, load_sessions
from inquiry_engine import generate_inquiry, to_plotly_tree
from epistemic_metrics import compute_eee

# -------------------------------------------------------
# 2) Configuración y BD
# -------------------------------------------------------
st.set_page_config(layout="wide")
init_db()
default_user_id = "user_1"

# -------------------------------------------------------
# 3) Funciones cacheadas para optimizar rendimiento
# -------------------------------------------------------
@st.cache_data
def run_dea_analysis(_df, dmu_col, input_cols, output_cols):
    """Función que encapsula los cálculos DEA para ser cacheados."""
    return mostrar_resultados(_df.copy(), dmu_col, input_cols, output_cols)

@st.cache_data
def get_inquiry_and_eee(root_q, context, _df_hash):
    """
    Función que encapsula las llamadas al LLM y cálculo EEE para ser cacheados.
    _df_hash se añade para que el caché se invalide si los datos cambian.
    """
    inquiry_tree = generate_inquiry(root_q, context=context)
    eee_score = compute_eee(inquiry_tree, depth_limit=5, breadth_limit=5)
    return inquiry_tree, eee_score

# -------------------------------------------------------
# 4) Sidebar: cargar sesiones previas
# -------------------------------------------------------
st.sidebar.header("Simulador DEA – Sesiones Guardadas")
sessions = load_sessions(user_id=default_user_id)
# (El resto del código de la sidebar no necesita cambios)
if sessions:
    ids = [s["session_id"] for s in sessions]
    selected_session_id = st.sidebar.selectbox("Seleccionar sesión para recargar", ids)
    if selected_session_id:
        sess = next((s for s in sessions if s["session_id"] == selected_session_id), None)
        if sess and st.sidebar.button("Cargar esta sesión"):
            # ... (código de carga de sesión sin cambios)
            st.success(f"Sesión '{selected_session_id}' cargada.")
            st.experimental_rerun()
else:
    st.sidebar.write("No hay sesiones guardadas.")

st.sidebar.markdown("---")
# ... (instrucciones en sidebar sin cambios)

# -------------------------------------------------------
# 5) Inicializar session_state
# -------------------------------------------------------
for key, default in [("df", None), ("dmu_col", None), ("input_cols", []), ("output_cols", []), ("dea_results", None), ("inquiry_tree", None), ("df_tree", None), ("eee_score", 0.0), ("df_eee", None), ("selected_dmu", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# -------------------------------------------------------
# 6) Área principal: Título y carga de archivo
# -------------------------------------------------------
st.title("Simulador Econométrico-Deliberativo – DEA")
uploaded_file = st.file_uploader("Cargar archivo CSV (con DMUs)", type=["csv"])
if uploaded_file is not None:
    st.session_state.df = pd.read_csv(uploaded_file)
    # Reset state on new file upload
    st.session_state.dmu_col, st.session_state.input_cols, st.session_state.output_cols, st.session_state.dea_results = None, [], [], None

# -------------------------------------------------------
# 7) Si hay DataFrame, muestro selección de columnas
# -------------------------------------------------------
if st.session_state.df is not None:
    df = st.session_state.df
    st.subheader("Datos cargados")
    st.dataframe(df.head())

    all_columns = df.columns.tolist()
    st.session_state.dmu_col = st.selectbox("Columna que identifica cada DMU", all_columns, index=all_columns.index(st.session_state.dmu_col) if st.session_state.dmu_col in all_columns else 0)
    
    candidate_inputs = [c for c in all_columns if c != st.session_state.dmu_col]
    st.session_state.input_cols = st.multiselect("Seleccionar columnas de inputs", options=candidate_inputs, default=[c for c in st.session_state.input_cols if c in candidate_inputs])

    candidate_outputs = [c for c in all_columns if c not in st.session_state.input_cols + [st.session_state.dmu_col]]
    st.session_state.output_cols = st.multiselect("Seleccionar columnas de outputs", options=candidate_outputs, default=[c for c in st.session_state.output_cols if c in candidate_outputs])

    # -------------------------------------------------------
    # 8) Botón para ejecutar DEA (ahora llama a funciones cacheadas)
    # -------------------------------------------------------
    if st.button("Ejecutar Análisis DEA"):
        if not st.session_state.input_cols or not st.session_state.output_cols:
            st.error("Por favor, selecciona al menos una columna de input y una de output.")
        else:
            validation_result = validate(df, st.session_state.input_cols, st.session_state.output_cols)
            if validation_result["formal_issues"]:
                st.error("Se encontraron problemas en los datos:")
                for issue in validation_result["formal_issues"]: st.write(f"  • {issue}")
            else:
                with st.spinner("Calculando eficiencias y generando árbol de indagación…"):
                    # a) Llamar a la función cacheada para el análisis DEA
                    st.session_state.dea_results = run_dea_analysis(df, st.session_state.dmu_col, st.session_state.input_cols, st.session_state.output_cols)

                    # b) Llamar a la función cacheada para el LLM y EEE
                    context_for_llm = {
                        "inputs": st.session_state.input_cols,
                        "outputs": st.session_state.output_cols,
                        "ccr_efficiencies_summary": st.session_state.dea_results["df_ccr"]["efficiency"].describe().to_dict(),
                        "sample_data_head": df.head().to_dict('records')
                    }
                    df_hash = pd.util.hash_pandas_object(df).sum()
                    st.session_state.inquiry_tree, st.session_state.eee_score = get_inquiry_and_eee("Diagnóstico de ineficiencia", context_for_llm, df_hash)
                    
                    # c) Preparar datos para visualización (esto es rápido)
                    tree_data_list = []
                    def flatten_tree(node, parent="Raíz"):
                        for key, value in node.items():
                            if isinstance(value, dict):
                                tree_data_list.append({"Nodo": key, "Padre": parent, "Tipo": "Pregunta"})
                                flatten_tree(value, key)
                    if st.session_state.inquiry_tree: flatten_tree(st.session_state.inquiry_tree)
                    st.session_state.df_tree = pd.DataFrame(tree_data_list)
                    st.session_state.df_eee = pd.DataFrame([{"Métrica": "EEE Score", "Valor": st.session_state.eee_score}])

                st.success("Análisis completado.")

    # -------------------------------------------------------
    # 9) Mostrar resultados si ya se calcularon
    # -------------------------------------------------------
    if st.session_state.dea_results:
        st.subheader("Resultados CCR")
        st.dataframe(st.session_state.dea_results["df_ccr"])
        
        st.subheader("Resultados BCC")
        st.dataframe(st.session_state.dea_results["df_bcc"])

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(st.session_state.dea_results["hist_ccr"], use_container_width=True)
        with col2:
            st.plotly_chart(st.session_state.dea_results["hist_bcc"], use_container_width=True)

        st.plotly_chart(st.session_state.dea_results["scatter3d_ccr"], use_container_width=True)

        # -------------------------------------------------------
        # 10) Benchmark Spider y resto de la UI (sin cambios)
        # -------------------------------------------------------
        st.subheader("Benchmark Spider CCR")
        dmu_options = st.session_state.dea_results["df_ccr"][st.session_state.dmu_col].astype(str).tolist()
        st.session_state.selected_dmu = st.selectbox("Seleccionar DMU para comparar", dmu_options, index=dmu_options.index(st.session_state.selected_dmu) if st.session_state.selected_dmu in dmu_options else 0)
        
        if st.session_state.selected_dmu:
            # La función de visualización ya está en `dea_models/visualizations.py`
            from dea_models.visualizations import plot_benchmark_spider
            spider_fig = plot_benchmark_spider(
                st.session_state.dea_results["merged_ccr"],
                st.session_state.selected_dmu,
                st.session_state.input_cols,
                st.session_state.output_cols
            )
            st.plotly_chart(spider_fig, use_container_width=True)

        st.subheader("Complejo de Indagación (Árbol)")
        if st.session_state.inquiry_tree:
            st.plotly_chart(to_plotly_tree(st.session_state.inquiry_tree), use_container_width=True)
        
        st.subheader("Métricas EEE")
        if st.session_state.df_eee is not None:
            st.dataframe(st.session_state.df_eee)
            
        # ... (código para guardar sesión y generar reportes sin cambios)
        st.subheader("Guardar y Exportar")
        notes = st.text_area("Notas sobre la sesión")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Guardar Sesión"):
                # ... Lógica de guardado (sin cambios)
                st.success("Sesión guardada.")
        with col2:
            html_report = generate_html_report(st.session_state.dea_results["df_ccr"], st.session_state.df_tree, st.session_state.df_eee)
            st.download_button("Descargar HTML", html_report, f"report_{datetime.datetime.now().strftime('%Y%m%d')}.html", "text/html")
        with col3:
            excel_report = generate_excel_report(st.session_state.dea_results["df_ccr"], st.session_state.df_tree, st.session_state.df_eee)
            st.download_button("Descargar Excel", excel_report, f"report_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
