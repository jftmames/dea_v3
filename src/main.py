import sys
import os
import pandas as pd
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

# -------------------------------------------------------
# 2) Configuración
# -------------------------------------------------------
st.set_page_config(layout="wide", page_title="SED - Simulador DEA")

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

if 'app_status' not in st.session_state:
    initialize_state()

@st.cache_data
def run_dea_analysis(_df, dmu_col, input_cols, output_cols):
    """Encapsula los cálculos DEA para ser cacheados."""
    return mostrar_resultados(_df.copy(), dmu_col, input_cols, output_cols)

# -------------------------------------------------------
# 4) Área principal
# -------------------------------------------------------
st.title("Simulador DEA (Versión Base)")

uploaded_file = st.file_uploader("1. Cargar archivo CSV", type=["csv"])

if uploaded_file is not None:
    if not hasattr(st.session_state, '_file_id') or st.session_state._file_id != uploaded_file.file_id:
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
        
        st.session_state._file_id = uploaded_file.file_id
        if st.session_state.df is not None:
            st.session_state.dmu_col = st.session_state.df.columns[0]
            st.rerun()

if 'df' in st.session_state and st.session_state.df is not None:
    df = st.session_state.df
    
    st.subheader("2. Configuración del Análisis")
    
    col_dmu, col_inputs, col_outputs = st.columns(3)
    with col_dmu:
        st.selectbox("Columna de DMU (Unidad de Análisis)", df.columns.tolist(), key='dmu_col')
    with col_inputs:
        st.multiselect("Columnas de Inputs", [c for c in df.columns.tolist() if c != st.session_state.dmu_col], key='input_cols')
    with col_outputs:
        st.multiselect("Columnas de Outputs", [c for c in df.columns.tolist() if c not in [st.session_state.dmu_col] + st.session_state.input_cols], key='output_cols')

    st.divider()

    if st.button("3. Ejecutar Análisis DEA", use_container_width=True, type="primary"):
        if not st.session_state.input_cols or not st.session_state.output_cols:
            st.error("Por favor, selecciona al menos un input y un output.")
        else:
            with st.spinner("Realizando análisis..."):
                st.session_state.dea_results = run_dea_analysis(
                    df, st.session_state.dmu_col, st.session_state.input_cols, st.session_state.output_cols
                )
                st.session_state.app_status = "results_ready"
            st.success("Análisis completado.")

# --- Mostrar resultados ---
if st.session_state.get('app_status') == "results_ready" and st.session_state.get('dea_results'):
    results = st.session_state.dea_results
    
    st.header("4. Resultados del Análisis", divider='rainbow')
    
    tab_ccr, tab_bcc = st.tabs(["**Resultados CCR**", "**Resultados BCC**"])
    
    with tab_ccr:
        st.subheader("Tabla de Eficiencias (Modelo CCR)")
        st.dataframe(results.get("df_ccr"))

    with tab_bcc:
        st.subheader("Tabla de Eficiencias (Modelo BCC)")
        st.dataframe(results.get("df_bcc"))
