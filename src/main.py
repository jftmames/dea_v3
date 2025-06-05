# jftmames/-dea-deliberativo-mvp/-dea-deliberativo-mvp-b44b8238c978ae0314af30717b9399634d28f8f9/src/main.py
import sys
import os
import pandas as pd
import streamlit as st

# --- 0) Ajuste del PYTHONPATH ---
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# --- 1) Importaciones ---
from results import mostrar_resultados
from inquiry_engine import generate_inquiry, to_plotly_tree
from epistemic_metrics import compute_eee
from openai_helpers import generate_analysis_proposals

# --- 2) Configuración de la página ---
st.set_page_config(layout="wide", page_title="DEA Deliberativo con IA")

# --- 3) Funciones de inicialización y caché ---
def initialize_state():
    """Inicializa el estado de la sesión para un nuevo fichero."""
    for key in list(st.session_state.keys()):
        if not key.startswith('_'):
            del st.session_state[key]
    st.session_state.app_status = "initial"

if 'app_status' not in st.session_state:
    initialize_state()

@st.cache_data
def run_dea_analysis(_df, dmu_col, input_cols, output_cols):
    return mostrar_resultados(_df.copy(), dmu_col, input_cols, output_cols)

@st.cache_data
def run_inquiry_engine(root_question, _context):
    return generate_inquiry(root_question, context=_context)

@st.cache_data
def get_analysis_proposals(_df):
    return generate_analysis_proposals(_df.columns.tolist(), _df.head())

# --- 4) Flujo principal de la aplicación ---
st.title("💡 DEA Deliberativo con IA")
st.markdown("Una herramienta para analizar la eficiencia y razonar sobre sus causas con ayuda de Inteligencia Artificial.")

# --- ETAPA 1: Carga de Datos ---
st.header("Paso 1: Carga tus Datos", divider="blue")
uploaded_file = st.file_uploader("Sube un fichero CSV con tus datos", type=["csv"])

if uploaded_file:
    if st.session_state.get('_file_id') != uploaded_file.file_id:
        initialize_state()
        st.session_state._file_id = uploaded_file.file_id
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            st.session_state.df = pd.read_csv(uploaded_file, sep=';')
        st.session_state.app_status = "file_loaded"
        st.rerun()

# A partir de aquí, la app funciona si hay un DF cargado
if st.session_state.app_status != "initial":
    df = st.session_state.df

    # --- ETAPA 2: Propuestas de Análisis por la IA ---
    st.header("Paso 2: Elige un Enfoque de Análisis", divider="blue")

    if 'proposals' not in st.session_state:
        with st.spinner("La IA está analizando tus datos y generando propuestas..."):
            proposals_data = get_analysis_proposals(df)
            st.session_state.proposals = proposals_data.get("proposals", [])
            if not st.session_state.proposals:
                st.error("La IA no pudo generar propuestas de análisis para estos datos. Por favor, revisa el formato del fichero.")
                st.stop()

    if 'selected_proposal' not in st.session_state:
        st.session_state.selected_proposal = None

    if not st.session_state.selected_proposal:
        st.info("La IA ha preparado varios enfoques para analizar tus datos. Elige el que mejor se adapte a tu objetivo.")
        for i, proposal in enumerate(st.session_state.get("proposals", [])):
            with st.expander(f"**Propuesta {i+1}: {proposal['title']}**", expanded=i==0):
                st.markdown(f"**Razonamiento:** *{proposal['reasoning']}*")
                st.markdown(f"**Inputs sugeridos:** `{proposal['inputs']}`")
                st.markdown(f"**Outputs sugeridos:** `{proposal['outputs']}`")
                if st.button(f"Seleccionar este análisis", key=f"select_{i}"):
                    st.session_state.selected_proposal = proposal
                    st.session_state.app_status = "proposal_selected"
                    st.rerun()
    
    # --- ETAPA 3: Ejecución del Análisis y Resultados ---
    if st.session_state.get("selected_proposal"):
        proposal = st.session_state.selected_proposal
        st.header(f"Paso 3: Analizando bajo el enfoque '{proposal['title']}'", divider="blue")
        
        # Mostramos la configuración elegida
        st.success(f"**Análisis seleccionado:** {proposal['title']}. {proposal['reasoning']}")
        
        # Ejecución del análisis
        if 'dea_results' not in st.session_state or st.session_state.dea_results is None:
            with st.spinner("Realizando análisis DEA..."):
                # Asumimos que la primera columna es la DMU si no está especificado
                dmu_col = df.columns[0] 
                st.session_state.dea_results = run_dea_analysis(df, dmu_col, proposal['inputs'], proposal['outputs'])

        results = st.session_state.dea_results
        
        # --- ETAPA 4: Razonamiento sobre los resultados ---
        st.header("Paso 4: Razona sobre la Ineficiencia con IA", divider="blue")
        
        if st.button("Generar Hipótesis de Ineficiencia con IA", use_container_width=True):
             with st.spinner("La IA está razonando sobre los resultados..."):
                avg_eff = results["df_ccr"]["tec_efficiency_ccr"].mean()
                inefficient_count = (results["df_ccr"]["tec_efficiency_ccr"] < 0.999).sum()
                
                context = {"inputs": proposal['inputs'], "outputs": proposal['outputs'], "avg_efficiency_ccr": avg_eff, "inefficient_units_count": int(inefficient_count), "total_units_count": len(df)}
                root_question = f"Bajo el enfoque '{proposal['title']}', ¿cuáles son las principales causas de la ineficiencia detectada?"
                
                tree, error = run_inquiry_engine(root_question, context)
                if error: st.error(f"Error en el motor de indagación: {error}")
                st.session_state.inquiry_tree = tree

        if st.session_state.get("inquiry_tree"):
            col_tree, col_eee = st.columns([2, 1])
            with col_tree:
                st.subheader("Árbol de Indagación", anchor=False)
                st.plotly_chart(to_plotly_tree(st.session_state.inquiry_tree), use_container_width=True)
            with col_eee:
                st.subheader("Calidad del Razonamiento (EEE)", anchor=False)
                eee_metrics = compute_eee(st.session_state.inquiry_tree, depth_limit=3, breadth_limit=5)
                st.metric(label="Índice de Equilibrio Erotético (EEE)", value=f"{eee_metrics['score']:.2%}")
                st.caption("Mide la calidad y balance del árbol de preguntas generado por la IA.")
                with st.expander("Ver desglose del EEE"):
                    st.markdown(f"- **D1: Profundidad ({eee_metrics['D1']:.2f})**")
                    st.markdown(f"- **D2: Pluralidad ({eee_metrics['D2']:.2f})**")

        # --- ETAPA 5: Resultados Detallados ---
        st.header("Paso 5: Explora los Resultados Detallados", divider="blue")
        tab_ccr, tab_bcc = st.tabs(["**Resultados CCR**", "**Resultados BCC**"])
        
        with tab_ccr:
            st.dataframe(results.get("df_ccr"))
        with tab_bcc:
            st.dataframe(results.get("df_bcc"))
