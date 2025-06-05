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
# Se eliminan las importaciones de session_manager
from inquiry_engine import generate_inquiry, to_plotly_tree
from epistemic_metrics import compute_eee
from dea_models.visualizations import plot_benchmark_spider, plot_efficiency_histogram, plot_3d_inputs_outputs

# -------------------------------------------------------
# 2) Configuración
# -------------------------------------------------------
st.set_page_config(layout="wide")

# -------------------------------------------------------
# 3) Funciones de inicialización y carga
# -------------------------------------------------------
def initialize_state():
    """Inicializa o resetea el estado de la sesión."""
    st.session_state.app_status = "initial"
    st.session_state.df = None
    st.session_state.dmu_col = None
    st.session_state.input_cols = []
    st.session_state.output_cols = []
    st.session_state.dea_results = None
    st.session_state.inquiry_tree = None
    st.session_state.eee_metrics = None
    st.session_state.model_selection = 'CCR (Constantes)'
    st.session_state.orientation_selection = 'Input (Minimizar)'

if 'app_status' not in st.session_state:
    initialize_state()

@st.cache_data
def run_dea_analysis(_df, dmu_col, input_cols, output_cols, model_type, orientation):
    """Encapsula los cálculos DEA para ser cacheados."""
    if not input_cols or not output_cols:
        return None
    return mostrar_resultados(_df.copy(), dmu_col, input_cols, output_cols, model_type, orientation)

@st.cache_data
def get_inquiry_and_eee(_root_q, _context, _df_hash):
    """Encapsula las llamadas al LLM y EEE para ser cacheados."""
    if not os.getenv("OPENAI_API_KEY"):
        return None, {"score": 0, "D1": 0, "D2": 0, "D3": 0, "D4": 0, "D5": 0}, "OPENAI_API_KEY no configurada."
    inquiry_tree, error_msg = generate_inquiry(_root_q, context=_context)
    if error_msg and not inquiry_tree:
        return None, None, error_msg
    eee_metrics = compute_eee(inquiry_tree, depth_limit=5, breadth_limit=5)
    return inquiry_tree, eee_metrics, error_msg

# -------------------------------------------------------
# 4) Sidebar
# -------------------------------------------------------
st.sidebar.header("Configuración")
st.sidebar.info("La funcionalidad de guardar/cargar sesiones ha sido desactivada para simplificar la aplicación.")

# -------------------------------------------------------
# 5) Área principal
# -------------------------------------------------------
st.title("Simulador Econométrico-Deliberativo – DEA")
uploaded_file = st.file_uploader("Cargar nuevo archivo CSV", type=["csv"])
if uploaded_file is not None:
    if st.session_state.df is None:
        initialize_state()
        try:
            st.session_state.df = pd.read_csv(uploaded_file, sep=',')
        except Exception:
            try:
                uploaded_file.seek(0)
                st.session_state.df = pd.read_csv(uploaded_file, sep=';')
            except Exception as e:
                st.error(f"Error al leer el fichero CSV. Detalle: {e}")
                st.session_state.df = None
        if st.session_state.df is not None:
            st.rerun()

if 'df' in st.session_state and st.session_state.df is not None:
    df = st.session_state.df
    
    # --- FUNCIÓN DE CALLBACK PARA RESETEAR EL ESTADO ---
    def reset_analysis_state():
        st.session_state.app_status = "initial"
        st.session_state.dea_results = None
        st.session_state.inquiry_tree = None
        st.session_state.eee_metrics = None
    
    st.subheader("Configuración del Análisis")
    
    col_config, col_inputs, col_outputs = st.columns(3)
    with col_config:
        dmu_col_index = df.columns.tolist().index(st.session_state.get('dmu_col')) if st.session_state.get('dmu_col') in df.columns else 0
        st.selectbox("Columna de DMU (Unidad de Análisis)", df.columns.tolist(), key='dmu_col', index=dmu_col_index, on_change=reset_analysis_state)
        st.radio("Tipo de Modelo", ['CCR (Constantes)', 'BCC (Variables)'], key='model_selection', horizontal=True, on_change=reset_analysis_state)
        st.radio("Orientación del Modelo", ['Input (Minimizar)', 'Output (Maximizar)'], key='orientation_selection', horizontal=True, on_change=reset_analysis_state)
        
    with col_inputs:
        st.multiselect("Columnas de Inputs", [c for c in df.columns.tolist() if c != st.session_state.dmu_col], key='input_cols', on_change=reset_analysis_state)
    with col_outputs:
        st.multiselect("Columnas de Outputs", [c for c in df.columns.tolist() if c not in [st.session_state.dmu_col] + st.session_state.input_cols], key='output_cols', on_change=reset_analysis_state)

    if st.button("🚀 Ejecutar Análisis DEA", use_container_width=True):
        if not st.session_state.input_cols or not st.session_state.output_cols:
            st.error("Por favor, selecciona al menos un input y un output.")
        else:
            model_map = {'CCR (Constantes)': 'CCR', 'BCC (Variables)': 'BCC'}
            orientation_map = {'Input (Minimizar)': 'input', 'Output (Maximizar)': 'output'}
            
            selected_model = model_map[st.session_state.model_selection]
            selected_orientation = orientation_map[st.session_state.orientation_selection]

            with st.spinner("Realizando análisis..."):
                st.session_state.dea_results = run_dea_analysis(
                    df, st.session_state.dmu_col, st.session_state.input_cols, st.session_state.output_cols,
                    selected_model, selected_orientation
                )
                st.session_state.app_status = "results_ready"
            st.success("Análisis completado.")

# --- Mostrar resultados ---
if st.session_state.get('app_status') == "results_ready" and st.session_state.get('dea_results'):
    results = st.session_state.dea_results
    model_ran = results['model_type']
    
    st.header(f"Resultados del Análisis {model_ran}", divider='rainbow')
    
    st.subheader(f"📊 Tabla de Eficiencias ({model_ran})")
    st.dataframe(results["df_results"])
    
    st.subheader(f"Visualizaciones de Eficiencia ({model_ran})")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(results['histogram'], use_container_width=True)
    with col2:
        st.plotly_chart(results['scatter_3d'], use_container_width=True)
        
    st.subheader(f"🕷️ Benchmark Spider ({model_ran})")
    # Se usa .get() para acceder de forma segura a la columna, aunque el error de estado ya está resuelto.
    dmu_list = results["df_results"].get(st.session_state.dmu_col, pd.Series([])).astype(str).tolist()
    if dmu_list:
        selected_dmu = st.selectbox("Seleccionar DMU para comparar:", options=dmu_list, key=f"dmu_{model_ran.lower()}")
        if selected_dmu:
            spider_fig = plot_benchmark_spider(results["merged_df"], selected_dmu, st.session_state.input_cols, st.session_state.output_cols)
            st.plotly_chart(spider_fig, use_container_width=True)
    else:
        st.warning("No se pudo generar la lista de DMUs para el gráfico de araña.")
