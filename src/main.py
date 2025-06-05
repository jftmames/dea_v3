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
    st.session_state.openai_error = None # Nuevo estado para guardar errores

if 'app_status' not in st.session_state:
    initialize_state()

@st.cache_data
def run_dea_analysis(_df, dmu_col, input_cols, output_cols):
    """Encapsula los cálculos DEA para ser cacheados."""
    return mostrar_resultados(_df.copy(), dmu_col, input_cols, output_cols)

@st.cache_data
def get_inquiry_and_eee(_root_q, _context, _df_hash):
    """Encapsula las llamadas al LLM y EEE, y devuelve el error si lo hay."""
    if not os.getenv("OPENAI_API_KEY"):
        return None, None, "La clave API de OpenAI no está configurada en los Secrets de la aplicación."
    
    inquiry_tree, error_msg = generate_inquiry(_root_q, context=_context)
    
    if error_msg and not inquiry_tree:
        # Si hay un error fatal que impide generar cualquier árbol
        return None, None, error_msg
    
    # Si hay un árbol (incluso el de respaldo), calcula EEE y pasa el mensaje de advertencia si existe
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
    st.subheader("Configuración del Análisis")
    
    def apply_scenario(new_inputs, new_outputs):
        st.session_state.input_cols = new_inputs
        st.session_state.output_cols = new_outputs

    col1, col2 = st.columns(2)
    with col1:
        dmu_col_index = df.columns.tolist().index(st.session_state.get('dmu_col')) if st.session_state.get('dmu_col') in df.columns else 0
        st.selectbox("Columna de DMU", df.columns.tolist(), key='dmu_col', index=dmu_col_index)
    with col2:
        st.multiselect("Columnas de Inputs", [c for c in df.columns.tolist() if c != st.session_state.dmu_col], key='input_cols')
        st.multiselect("Columnas de Outputs", [c for c in df.columns.tolist() if c not in [st.session_state.dmu_col] + st.session_state.input_cols], key='output_cols')

    if st.button("🚀 Ejecutar Análisis DEA", use_container_width=True):
        if not st.session_state.input_cols or not st.session_state.output_cols:
            st.error("Por favor, selecciona al menos un input y un output.")
        else:
            with st.spinner("Realizando análisis completo..."):
                st.session_state.dea_results = run_dea_analysis(df, st.session_state.dmu_col, st.session_state.input_cols, st.session_state.output_cols)
                context = {"inputs": st.session_state.input_cols, "outputs": st.session_state.output_cols}
                df_hash = pd.util.hash_pandas_object(df).sum()
                
                # Capturar los 3 valores de retorno
                tree, eee, error = get_inquiry_and_eee("Diagnóstico de ineficiencia", context, df_hash)
                st.session_state.inquiry_tree = tree
                st.session_state.eee_metrics = eee
                st.session_state.openai_error = error # Guardar el mensaje de error/advertencia

                st.session_state.app_status = "results_ready"
            st.success("Análisis completado.")

# --- Mostrar resultados ---
if st.session_state.get('app_status') == "results_ready" and st.session_state.get('dea_results'):
    results = st.session_state.dea_results
    st.header("Resultados del Análisis DEA", divider='rainbow')
    
    # ... (código para mostrar las pestañas CCR/BCC y los gráficos DEA se mantiene igual) ...

    # --- SECCIÓN DE ANÁLISIS DELIBERATIVO MEJORADA ---
    st.header("Análisis Deliberativo Asistido por IA", divider='rainbow')

    # Mostrar error de OpenAI si existe
    if st.session_state.get('openai_error'):
        # Distinguir entre un error fatal y una advertencia
        if "usando árbol de respaldo" in st.session_state.openai_error:
            st.warning(f"**Advertencia en el Análisis Deliberativo:** {st.session_state.openai_error}")
        else:
            st.error(f"**Error en el Análisis Deliberativo:** {st.session_state.openai_error}")
        st.info("Si el error persiste, comprueba tu API Key de OpenAI, que tenga crédito y que el servicio de OpenAI esté operativo.")

    # Mostrar el resto solo si el árbol existe (incluyendo el de respaldo)
    if st.session_state.get('inquiry_tree'):
        st.subheader("🔬 Escenarios Interactivos del Complejo de Indagación")
        # ... (El código de los escenarios se mantiene igual)
        
        st.subheader("🧠 Métrica de Calidad del Diagnóstico (EEE)")
        # ... (El código de la métrica EEE se mantiene igual)
