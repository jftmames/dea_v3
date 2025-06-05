import sys
import os
import pandas as pd
import datetime
import streamlit as st

# -------------------------------------------------------
# 0) Ajuste del PYTHONPATH
# -------------------------------------------------------
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# -------------------------------------------------------
# 1) Importaciones
# -------------------------------------------------------
from data_validator import validate
from results import mostrar_resultados
from report_generator import generate_html_report, generate_excel_report
from session_manager import init_db, save_session, load_sessions
from inquiry_engine import generate_inquiry, to_plotly_tree
from epistemic_metrics import compute_eee
from dea_models.visualizations import plot_benchmark_spider, plot_efficiency_histogram, plot_3d_inputs_outputs

# -------------------------------------------------------
# 2) Configuración y BD
# -------------------------------------------------------
st.set_page_config(layout="wide")
init_db()
default_user_id = "user_1"

# -------------------------------------------------------
# 3) Funciones cacheadas
# -------------------------------------------------------
@st.cache_data
def run_dea_analysis(_df, dmu_col, input_cols, output_cols):
    """Encapsula los cálculos DEA para ser cacheados."""
    return mostrar_resultados(_df.copy(), dmu_col, input_cols, output_cols)

@st.cache_data
def get_inquiry_and_eee(_root_q, _context, _df_hash):
    """Encapsula las llamadas al LLM y EEE para ser cacheados."""
    inquiry_tree = generate_inquiry(_root_q, context=_context)
    eee_score = compute_eee(inquiry_tree, depth_limit=5, breadth_limit=5)
    return inquiry_tree, eee_score

# -------------------------------------------------------
# 4) Inicialización de st.session_state
# -------------------------------------------------------
if 'app_status' not in st.session_state:
    st.session_state.app_status = "initial"
    st.session_state.df = None
    st.session_state.dmu_col = None
    st.session_state.input_cols = []
    st.session_state.output_cols = []
    st.session_state.dea_results = None
    st.session_state.inquiry_tree = None
    st.session_state.df_tree = None
    st.session_state.eee_score = 0.0
    st.session_state.df_eee = None
    st.session_state.selected_dmu = None

# --- CORRECCIÓN #2: LÓGICA DE CARGA DE SESIÓN COMPLETA ---
def load_full_session(session_data):
    """Carga el estado COMPLETO de una sesión en st.session_state."""
    st.session_state.df = pd.DataFrame(session_data.get('df_data', []))
    st.session_state.dmu_col = session_data.get('dmu_col')
    st.session_state.input_cols = session_data.get('input_cols', [])
    st.session_state.output_cols = session_data.get('output_cols', [])
    st.session_state.inquiry_tree = session_data.get('inquiry_tree')
    st.session_state.eee_score = session_data.get('eee_score', 0.0)

    # Reconstruir DataFrames de resultados y otros
    dea_res_raw = session_data.get('dea_results', {})
    dea_results_reloaded = {}
    if dea_res_raw:
        for k, v in dea_res_raw.items():
            if isinstance(v, list):
                dea_results_reloaded[k] = pd.DataFrame(v)
    # Volver a generar las figuras que no se guardan
    if dea_results_reloaded and not dea_results_reloaded.get('df_ccr', pd.DataFrame()).empty:
        dea_results_reloaded['hist_ccr'] = plot_efficiency_histogram(dea_results_reloaded['df_ccr'])
        dea_results_reloaded['hist_bcc'] = plot_efficiency_histogram(dea_results_reloaded['df_bcc'])
        dea_results_reloaded['scatter3d_ccr'] = plot_3d_inputs_outputs(st.session_state.df, st.session_state.input_cols, st.session_state.output_cols, dea_results_reloaded['df_ccr'], st.session_state.dmu_col)
    
    st.session_state.dea_results = dea_results_reloaded
    st.session_state.df_tree = pd.DataFrame(session_data.get('df_tree_data', []))
    st.session_state.df_eee = pd.DataFrame(session_data.get('df_eee_data', []))
    
    # Resetear la selección de DMU para que se tome la primera por defecto
    st.session_state.selected_dmu = None
    st.session_state.app_status = "results_ready"
    st.success(f"Sesión '{session_data.get('session_id')}' cargada completamente.")
    st.rerun()

# -------------------------------------------------------
# 5) Sidebar
# -------------------------------------------------------
st.sidebar.header("Simulador DEA – Sesiones Guardadas")
sessions = load_sessions(user_id=default_user_id)
if sessions:
    session_options = {f"{s['timestamp'].split('T')[0]} - {s.get('notes', 'Sin notas')[:20]} (ID: ...{s['session_id'][-4:]})": s['session_id'] for s in sorted(sessions, key=lambda x: x['timestamp'], reverse=True)}
    
    selected_session_display = st.sidebar.selectbox("Seleccionar sesión para recargar", session_options.keys(), index=None, placeholder="Elige una sesión guardada...")
    
    if selected_session_display and st.sidebar.button("Cargar Sesión Seleccionada"):
        session_id_to_load = session_options[selected_session_display]
        session_to_load = next((s for s in sessions if s['session_id'] == session_id_to_load), None)
        if session_to_load:
            load_full_session(session_to_load)
else:
    st.sidebar.write("No hay sesiones guardadas.")

# -------------------------------------------------------
# 6) Área principal
# -------------------------------------------------------
st.title("Simulador Econométrico-Deliberativo – DEA")

uploaded_file = st.file_uploader("Cargar nuevo archivo CSV", type=["csv"])
if uploaded_file is not None:
    # Reiniciar estado al cargar un nuevo archivo, manteniendo la consistencia
    st.session_state.app_status = "initial"
    st.session_state.df = pd.read_csv(uploaded_file)
    st.session_state.dmu_col = None
    st.session_state.input_cols = []
    st.session_state.output_cols = []
    st.session_state.dea_results = None
    st.session_state.inquiry_tree = None
    st.session_state.df_tree = None
    st.session_state.eee_score = 0.0
    st.session_state.df_eee = None
    st.session_state.selected_dmu = None
    st.rerun()

# --- Flujo principal de la UI ---
if st.session_state.df is not None:
    df = st.session_state.df
    st.subheader("Configuración del Análisis")
    
    col1, col2 = st.columns(2)
    with col1:
        all_columns = df.columns.tolist()
        # La clave es usar .get() para evitar el error si la clave no existe aún
        dmu_index = all_columns.index(st.session_state.dmu_col) if st.session_state.dmu_col in all_columns else 0
        st.selectbox("Columna de DMU", all_columns, index=dmu_index, key='dmu_col')
    with col2:
        st.multiselect("Columnas de Inputs", [c for c in all_columns if c != st.session_state.dmu_col], key='input_cols', default=st.session_state.input_cols)
        st.multiselect("Columnas de Outputs", [c for c in all_columns if c not in [st.session_state.dmu_col] + st.session_state.input_cols], key='output_cols', default=st.session_state.output_cols)

    if st.button("🚀 Ejecutar Análisis DEA", use_container_width=True):
        if not st.session_state.input_cols or not st.session_state.output_cols:
            st.error("Por favor, selecciona al menos un input y un output.")
        else:
            with st.spinner("Realizando análisis..."):
                st.session_state.dea_results = run_dea_analysis(df, st.session_state.dmu_col, st.session_state.input_cols, st.session_state.output_cols)
                context = {"inputs": st.session_state.input_cols, "outputs": st.session_state.output_cols}
                df_hash = pd.util.hash_pandas_object(df).sum()
                st.session_state.inquiry_tree, st.session_state.eee_score = get_inquiry_and_eee("Diagnóstico de ineficiencia", context, df_hash)
                
                tree_data_list = []
                if st.session_state.inquiry_tree:
                    def flatten_tree(node, parent_path=""):
                        for key, value in node.items():
                            full_path = f"{parent_path} -> {key}" if parent_path else key
                            tree_data_list.append({"Nodo": key, "Padre": parent_path or "Raíz"})
                            if isinstance(value, dict):
                                flatten_tree(value, key)
                    
                    root_key = list(st.session_state.inquiry_tree.keys())[0]
                    tree_data_list.append({"Nodo": root_key, "Padre": "Raíz"})
                    flatten_tree(st.session_state.inquiry_tree[root_key], root_key)

                st.session_state.df_tree = pd.DataFrame(tree_data_list)
                st.session_state.df_eee = pd.DataFrame([{"Métrica": "EEE Score", "Valor": st.session_state.eee_score}])
                
                if st.session_state.dea_results and not st.session_state.dea_results["df_ccr"].empty:
                    st.session_state.selected_dmu = st.session_state.dea_results["df_ccr"][st.session_state.dmu_col].astype(str).tolist()[0]
                
                st.session_state.app_status = "results_ready"
            st.success("Análisis completado.")
            st.rerun()

# -------------------------------------------------------
# 8) Mostrar resultados si el estado es 'results_ready'
# -------------------------------------------------------
if st.session_state.app_status == "results_ready" and st.session_state.dea_results:
    st.header("Resultados del Análisis", divider='rainbow')

    st.subheader("📊 Tabla de Eficiencias (CCR)")
    st.dataframe(st.session_state.dea_results["df_ccr"])

    st.subheader("🕷️ Benchmark Spider CCR")
    dmu_options = st.session_state.dea_results["df_ccr"][st.session_state.dmu_col].astype(str).tolist()
    
    selected_dmu_index = dmu_options.index(st.session_state.selected_dmu) if st.session_state.selected_dmu in dmu_options else 0
    st.selectbox(
        "Seleccionar DMU para comparar:",
        options=dmu_options,
        index=selected_dmu_index,
        key="selected_dmu"
    )
    if st.session_state.selected_dmu:
        spider_fig = plot_benchmark_spider(st.session_state.dea_results["merged_ccr"], st.session_state.selected_dmu, st.session_state.input_cols, st.session_state.output_cols)
        st.plotly_chart(spider_fig, use_container_width=True)

    st.subheader("🌳 Complejo de Indagación (Árbol de Diagnóstico)")
    if st.session_state.inquiry_tree:
        tree_map_fig = to_plotly_tree(st.session_state.inquiry_tree, title="Árbol de Diagnóstico: Causas y Estrategias")
        st.plotly_chart(tree_map_fig, use_container_width=True)

    st.subheader("🧠 Métrica de Calidad del Diagnóstico (EEE)")
    st.info("El **Índice de Equilibrio Erotético (EEE)** mide la calidad y robustez del árbol de diagnóstico (0 a 1). Un valor más alto indica un análisis más completo.")
    st.metric(label="Puntuación EEE", value=f"{st.session_state.eee_score:.4f}")

    st.header("Acciones", divider='rainbow')
    notes = st.text_area("Notas de la sesión (se guardarán con la sesión)")
    
    if st.button("💾 Guardar Sesión Actual", use_container_width=True):
        with st.spinner("Guardando..."):
            serializable_dea_results = {}
            if st.session_state.dea_results:
                for key, value in st.session_state.dea_results.items():
                    if isinstance(value, pd.DataFrame):
                        serializable_dea_results[key] = value.to_dict('records')
            
            save_session(
                user_id=default_user_id,
                inquiry_tree=st.session_state.inquiry_tree, eee_score=st.session_state.eee_score, notes=notes,
                dmu_col=st.session_state.dmu_col, input_cols=st.session_state.input_cols, output_cols=st.session_state.output_cols,
                df_data=st.session_state.df.to_dict('records'), dea_results=serializable_dea_results,
                df_tree_data=st.session_state.df_tree.to_dict('records'), df_eee_data=st.session_state.df_eee.to_dict('records')
            )
        st.success("¡Sesión guardada correctamente!")
        st.balloons()
    
    st.subheader("Generar Reportes")
    col1, col2 = st.columns(2)
    with col1:
        html_report = generate_html_report(st.session_state.dea_results["df_ccr"], st.session_state.df_tree, st.session_state.df_eee)
        st.download_button("Descargar HTML", html_report, f"reporte_dea.html", "text/html", use_container_width=True)
    with col2:
        excel_report = generate_excel_report(st.session_state.dea_results["df_ccr"], st.session_state.df_tree, st.session_state.df_eee)
        st.download_button("Descargar Excel", excel_report, f"reporte_dea.xlsx", use_container_width=True)
