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
from results import mostrar_resultados
from inquiry_engine import generate_inquiry, to_plotly_tree
from epistemic_metrics import compute_eee

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
    st.session_state.openai_error = None
    st.session_state.model_selection = 'CCR (Constantes)'
    st.session_state.orientation_selection = 'Input (Minimizar)'

if 'app_status' not in st.session_state:
    initialize_state()

@st.cache_data
def run_dea_analysis(_df, dmu_col, input_cols, output_cols, model_type, orientation):
    """Encapsula los cálculos DEA para ser cacheados."""
    return mostrar_resultados(_df.copy(), dmu_col, input_cols, output_cols, model_type, orientation)

@st.cache_data
def get_inquiry_and_eee(_root_q, _context, _df_hash):
    """Encapsula las llamadas al LLM y EEE, y devuelve el error si lo hay."""
    if not os.getenv("OPENAI_API_KEY"):
        return None, None, "La clave API de OpenAI no está configurada en los Secrets de la aplicación."
    
    inquiry_tree, error_msg = generate_inquiry(_root_q, context=_context)
    
    if error_msg and not inquiry_tree:
        return None, None, error_msg
    
    eee_metrics = compute_eee(inquiry_tree, depth_limit=5, breadth_limit=5)
    return inquiry_tree, eee_metrics, error_msg

# -------------------------------------------------------
# 4) Sidebar
# -------------------------------------------------------
st.sidebar.header("Acerca de")
st.sidebar.info("Simulador Econométrico-Deliberativo para Análisis Envolvente de Datos (DEA).")
st.sidebar.info("La funcionalidad de guardar/cargar sesiones ha sido desactivada.")

# -------------------------------------------------------
# 5) Área principal
# -------------------------------------------------------
st.title("Simulador Econométrico-Deliberativo – DEA")
uploaded_file = st.file_uploader("Cargar nuevo archivo CSV", type=["csv"])
if uploaded_file is not None:
    if not hasattr(st.session_state, '_file_id') or st.session_state._file_id != uploaded_file.id:
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
        
        st.session_state._file_id = uploaded_file.id
        if st.session_state.df is not None:
            st.rerun()

if 'df' in st.session_state and st.session_state.df is not None:
    df = st.session_state.df
    
    def reset_analysis_state():
        st.session_state.app_status = "initial"
        st.session_state.dea_results = None

    st.subheader("Configuración del Análisis")
    
    col_config, col_inputs, col_outputs = st.columns(3)
    with col_config:
        st.selectbox("Columna de DMU", df.columns.tolist(), key='dmu_col', on_change=reset_analysis_state)
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
                context = {"inputs": st.session_state.input_cols, "outputs": st.session_state.output_cols}
                df_hash = pd.util.hash_pandas_object(df).sum()
                tree, eee, error = get_inquiry_and_eee("Diagnóstico de ineficiencia", context, df_hash)
                
                st.session_state.inquiry_tree = tree
                st.session_state.eee_metrics = eee
                st.session_state.openai_error = error
                
                st.session_state.app_status = "results_ready"
            st.success("Análisis completado.")

# --- Mostrar resultados ---
if st.session_state.get('app_status') == "results_ready" and st.session_state.get('dea_results'):
    results = st.session_state.dea_results
    model_ran = results.get('model_type', 'Desconocido')
    
    st.header(f"Resultados del Análisis {model_ran}", divider='rainbow')
    st.dataframe(results["df_results"])
    # ... (código para los gráficos DEA)

    st.header("Análisis Deliberativo Asistido por IA", divider='rainbow')
    
    # --- Interfaz de error transparente ---
    if st.session_state.get('openai_error'):
        st.error(f"**Error en el Análisis Deliberativo:** {st.session_state.openai_error}")
    
    if st.session_state.get('inquiry_tree'):
        st.subheader("Árbol de Indagación")
        st.plotly_chart(to_plotly_tree(st.session_state.inquiry_tree), use_container_width=True)
    else:
        st.warning("No se pudo generar el Árbol de Indagación.")
