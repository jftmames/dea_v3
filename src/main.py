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
# 3) Funciones de inicialización y carga
# -------------------------------------------------------
def initialize_state():
    """Inicializa o resetea el estado de la sesión para prevenir errores."""
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

# Ejecutar la inicialización solo una vez al principio de una nueva sesión
if 'app_status' not in st.session_state:
    initialize_state()

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

def load_full_session(session_data):
    """Carga de forma segura el estado COMPLETO de una sesión."""
    initialize_state() # Resetea para un estado limpio antes de cargar
    st.session_state.df = pd.DataFrame(session_data.get('df_data', []))
    st.session_state.dmu_col = session_data.get('dmu_col')
    st.session_state.input_cols = session_data.get('input_cols', [])
    st.session_state.output_cols = session_data.get('output_cols', [])
    st.session_state.inquiry_tree = session_data.get('inquiry_tree')
    st.session_state.eee_score = session_data.get('eee_score', 0.0)
    st.session_state.df_tree = pd.DataFrame(session_data.get('df_tree_data', []))
    st.session_state.df_eee = pd.DataFrame(session_data.get('df_eee_data', []))
    
    dea_res_raw = session_data.get('dea_results', {})
    st.session_state.dea_results = {k: pd.DataFrame(v) for k, v in dea_res_raw.items() if isinstance(v, list)}
    
    st.session_state.app_status = "results_ready"
    st.success(f"Sesión '{session_data.get('session_id')}' cargada.")
    st.rerun()

# -------------------------------------------------------
# 4) Sidebar
# -------------------------------------------------------
with st.sidebar:
    st.header("Simulador DEA – Sesiones Guardadas")
    sessions = load_sessions(user_id=default_user_id)
    if not sessions:
        st.write("No hay sesiones guardadas.")
    else:
        session_options = {f"{s['timestamp'].split('T')[0]} - {s.get('notes', 'Sin notas')[:20]}": s['session_id'] for s in sorted(sessions, key=lambda x: x['timestamp'], reverse=True)}
        selected_session_display = st.selectbox("Seleccionar sesión para recargar", session_options.keys(), index=None, placeholder="Elige una sesión guardada...")
        
        if st.button("Cargar Sesión Seleccionada") and selected_session_display:
            session_id_to_load = session_options[selected_session_display]
            session_to_load = next((s for s in sessions if s['session_id'] == session_id_to_load), None)
            if session_to_load:
                load_full_session(session_to_load)

# -------------------------------------------------------
# 5) Área principal
# -------------------------------------------------------
st.title("Simulador Econométrico-Deliberativo – DEA")

uploaded_file = st.file_uploader("Cargar nuevo archivo CSV", type=["csv"])
if uploaded_file is not None:
    initialize_state() # Reseteo limpio al cargar nuevo fichero
    st.session_state.df = pd.read_csv(uploaded_file)
    st.rerun()

# --- Flujo principal de la UI ---
if 'df' in st.session_state and st.session_state.df is not None:
    df = st.session_state.df
    st.subheader("Configuración del Análisis")
    
    col1, col2 = st.columns(2)
    with col1:
        all_columns = df.columns.tolist()
        current_dmu = st.session_state.get('dmu_col')
        dmu_index = all_columns.index(current_dmu) if current_dmu and current_dmu in all_columns else 0
        st.selectbox("Columna de DMU", all_columns, index=dmu_index, key='dmu_col')
    with col2:
        st.multiselect("Columnas de Inputs", [c for c in all_columns if c != st.session_state.get('dmu_col')], key='input_cols', default=st.session_state.get('input_cols', []))
        st.multiselect("Columnas de Outputs", [c for c in all_columns if c not in [st.session_state.get('dmu_col')] + st.session_state.get('input_cols', [])], key='output_cols', default=st.session_state.get('output_cols', []))

    if st.button("🚀 Ejecutar Análisis DEA", use_container_width=True):
        if not st.session_state.get('input_cols') or not st.session_state.get('output_cols'):
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
                            tree_data_list.append({"Nodo": key, "Padre": parent_path or "Raíz"})
                            if isinstance(value, dict):
                                flatten_tree(value, key)
                    flatten_tree(st.session_state.inquiry_tree, "")

                st.session_state.df_tree = pd.DataFrame(tree_data_list)
                st.session_state.df_eee = pd.DataFrame([{"Métrica": "EEE Score", "Valor": st.session_state.eee_score}])
                
                if st.session_state.dea_results and not st.session_state.dea_results["df_ccr"].empty:
                    st.session_state.selected_dmu = st.session_state.dea_results["df_ccr"][st.session_state.dmu_col].astype(str).tolist()[0]
                
                st.session_state.app_status = "results_ready"
            st.success("Análisis completado.")
            # La línea st.rerun() que causaba el bucle ha sido eliminada.

# --- Mostrar resultados ---
if st.session_state.get('app_status') == "results_ready" and st.session_state.get('dea_results'):
    st.header("Resultados del Análisis", divider='rainbow')

    st.subheader("📊 Tabla de Eficiencias (CCR)")
    st.dataframe(st.session_state.dea_results["df_ccr"])

    st.subheader("🕷️ Benchmark Spider CCR")
    dmu_options = st.session_state.dea_results["df_ccr"][st.session_state.dmu_col].astype(str).tolist()
    
    selected_dmu_index = dmu_options.index(st.session_state.selected_dmu) if st.session_state.get('selected_dmu') in dmu_options else 0
    st.selectbox("Seleccionar DMU para comparar:", options=dmu_options, index=selected_dmu_index, key="selected_dmu")
    
    if st.session_state.selected_dmu:
        spider_fig = plot_benchmark_spider(st.session_state.dea_results["merged_ccr"], st.session_state.selected_dmu, st.session_state.input_cols, st.session_state.output_cols)
        st.plotly_chart(spider_fig, use_container_width=True)

    st.subheader("🌳 Complejo de Indagación (Árbol de Diagnóstico)")
    if st.session_state.get('inquiry_tree'):
        tree_map_fig = to_plotly_tree(st.session_state.inquiry_tree, title="Árbol de Diagnóstico: Causas y Estrategias")
        st.plotly_chart(tree_map_fig, use_container_width=True)

    st.subheader("🧠 Métrica de Calidad del Diagnóstico (EEE)")
    st.info("El **Índice de Equilibrio Erotético (EEE)** mide la calidad y robustez del árbol de diagnóstico (0 a 1).")
    if st.session_state.get('eee_score') is not None:
        st.metric(label="Puntuación EEE", value=f"{st.session_state.eee_score:.4f}")

    st.header("Acciones", divider='rainbow')
    notes = st.text_area("Notas de la sesión (se guardarán con la sesión)")
    
    if st.button("💾 Guardar Sesión Actual", use_container_width=True):
        with st.spinner("Guardando..."):
            serializable_dea_results = {k: v.to_dict('records') for k, v in st.session_state.dea_results.items() if isinstance(v, pd.DataFrame)}
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
