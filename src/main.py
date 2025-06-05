# jftmames/-dea-deliberativo-mvp/-dea-deliberativo-mvp-b44b8238c978ae0314af30717b9399634d28f8f9/src/main.py
import sys
import os
import pandas as pd
import streamlit as st
import re

# --- 0) Ajuste del PYTHONPATH ---
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# --- 1) Importaciones ---
from results import mostrar_resultados
from inquiry_engine import suggest_actionable_variables # <-- CAMBIO DE NOMBRE
from openai_helpers import generate_analysis_proposals
from dea_models.visualizations import plot_hypothesis_distribution, plot_benchmark_spider

# --- 2) Configuración ---
st.set_page_config(layout="wide", page_title="DEA Deliberativo con IA")

# --- 3) Funciones de estado y caché ---
def initialize_state():
    for key in list(st.session_state.keys()):
        if not key.startswith('_'): del st.session_state[key]
    st.session_state.app_status = "initial"

def reset_analysis_state():
    st.session_state.app_status = "proposal_selected"
    st.session_state.dea_results = None
    st.session_state.actionable_suggestions = None
    st.session_state.plot_variable_name = None

if 'app_status' not in st.session_state:
    initialize_state()

@st.cache_data
def run_dea_analysis(_df, dmu_col, input_cols, output_cols):
    return mostrar_resultados(_df.copy(), dmu_col, input_cols, output_cols)

@st.cache_data
def get_actionable_suggestions(_context): # <-- CAMBIO DE NOMBRE
    return suggest_actionable_variables(context=_context)

@st.cache_data
def get_analysis_proposals(_df):
    return generate_analysis_proposals(_df.columns.tolist(), _df.head())

# --- 4) Flujo principal ---
st.title("💡 DEA Deliberativo con IA")
st.markdown("Una herramienta para analizar la eficiencia y razonar sobre sus causas con ayuda de Inteligencia Artificial.")

# ETAPA 1: Carga de Datos
st.header("Paso 1: Carga tus Datos", divider="blue")
uploaded_file = st.file_uploader("Sube un fichero CSV", type=["csv"])

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

if st.session_state.app_status != "initial":
    df = st.session_state.df

    # ETAPA 2: Propuestas de Análisis por la IA
    if st.session_state.app_status in ["file_loaded", "proposal_selected"]:
        st.header("Paso 2: Elige un Enfoque de Análisis", divider="blue")
        if 'proposals' not in st.session_state:
            with st.spinner("La IA está analizando tus datos y generando propuestas..."):
                proposals_data = get_analysis_proposals(df)
                st.session_state.proposals = proposals_data.get("proposals", [])
        
        if not st.session_state.get("proposals"):
            st.error("La IA no pudo generar propuestas. Por favor, revisa el formato del fichero o la clave de API.")
            st.stop()

        if 'selected_proposal' not in st.session_state: st.session_state.selected_proposal = None

        if not st.session_state.selected_proposal:
            st.info("La IA ha preparado varios enfoques para analizar tus datos. Elige el que mejor se adapte a tu objetivo.")
            for i, proposal in enumerate(st.session_state.get("proposals", [])):
                with st.expander(f"**Propuesta {i+1}: {proposal['title']}**", expanded=i==0):
                    st.markdown(f"**Razonamiento:** *{proposal['reasoning']}*")
                    st.markdown(f"**Inputs sugeridos:** `{proposal['inputs']}`")
                    st.markdown(f"**Outputs sugeridos:** `{proposal['outputs']}`")
                    if st.button(f"Seleccionar este análisis", key=f"select_{i}"):
                        st.session_state.selected_proposal = proposal
                        reset_analysis_state()
                        st.rerun()
    
    # ETAPAS 3, 4 y 5: se muestran tras seleccionar una propuesta
    if st.session_state.get("selected_proposal"):
        proposal = st.session_state.selected_proposal
        
        if st.session_state.app_status == "proposal_selected":
             st.header(f"Paso 3: Analizando bajo el enfoque '{proposal['title']}'", divider="blue")
             st.success(f"**Análisis seleccionado:** {proposal['title']}. {proposal['reasoning']}")

        if 'dea_results' not in st.session_state or st.session_state.dea_results is None:
            with st.spinner("Realizando análisis DEA..."):
                dmu_col = df.columns[0] 
                st.session_state.dea_results = run_dea_analysis(df, dmu_col, proposal['inputs'], proposal['outputs'])
                st.session_state.app_status = "results_ready"

        results = st.session_state.dea_results
        
        # ETAPA 4: Sugerencias de la IA y Exploración Interactiva
        if st.session_state.app_status in ["results_ready", "suggestions_done"]:
            st.header("Paso 4: Explora las Causas de la Ineficiencia", divider="blue")
            
            if 'actionable_suggestions' not in st.session_state or st.session_state.actionable_suggestions is None:
                with st.spinner("La IA está identificando las variables clave para investigar la ineficiencia..."):
                    context = {"inputs": proposal['inputs'], "outputs": proposal['outputs'], "avg_efficiency_ccr": results["df_ccr"]["tec_efficiency_ccr"].mean()}
                    suggestions, error = get_actionable_suggestions(context)
                    if error: st.error(f"Error: {error}")
                    st.session_state.actionable_suggestions = suggestions
                    st.session_state.app_status = "suggestions_done"

            if st.session_state.get("actionable_suggestions"):
                st.info("La IA sugiere que las siguientes variables son las más relevantes para entender la ineficiencia. Haz clic en una para analizarla visualmente.")
                
                placeholder = st.container() # Contenedor para el gráfico
                
                # Crear los botones de exploración
                cols = st.columns(len(st.session_state.actionable_suggestions))
                for i, suggestion in enumerate(st.session_state.actionable_suggestions):
                    with cols[i]:
                        if st.button(f"Analizar '{suggestion['variable']}'", key=suggestion['variable'], use_container_width=True):
                            st.session_state.plot_variable_name = suggestion['variable']
                            st.session_state.plot_variable_reasoning = suggestion['reasoning']
                
                # Si hay una variable seleccionada, mostrar el gráfico
                if st.session_state.get("plot_variable_name"):
                    var_to_plot = st.session_state.plot_variable_name
                    reasoning = st.session_state.plot_variable_reasoning
                    with placeholder:
                        st.markdown(f"#### Análisis de la variable: `{var_to_plot}`")
                        st.caption(f"Razonamiento de la IA: *{reasoning}*")
                        with st.spinner(f"Generando gráfico para '{var_to_plot}'..."):
                            fig = plot_hypothesis_distribution(
                                df_results=results['df_ccr'],
                                df_original=df,
                                variable=var_to_plot,
                                dmu_col=df.columns[0]
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            if st.button("Cerrar exploración"):
                                st.session_state.plot_variable_name = None
                                st.rerun()

        # ETAPA 5: Resultados Detallados
        st.header("Paso 5: Resultados Numéricos y Gráficos Detallados", divider="blue")
        tab_ccr, tab_bcc = st.tabs(["**Resultados CCR**", "**Resultados BCC**"])
        
        with tab_ccr:
            st.subheader("Tabla de Eficiencias y Slacks (Modelo CCR)")
            st.dataframe(results.get("df_ccr"))
            st.subheader("Visualizaciones de Eficiencia (CCR)")
            if "hist_ccr" in results and "scatter3d_ccr" in results:
                col1, col2 = st.columns(2)
                with col1: st.plotly_chart(results["hist_ccr"], use_container_width=True)
                with col2: st.plotly_chart(results["scatter3d_ccr"], use_container_width=True)
            st.subheader("Análisis de Benchmarking (CCR)")
            dmu_options_ccr = results.get("df_ccr", pd.DataFrame()).get(df.columns[0], []).astype(str).tolist()
            if dmu_options_ccr:
                selected_dmu_ccr = st.selectbox("Seleccionar DMU para comparar con sus benchmarks:", options=dmu_options_ccr, key="dmu_ccr_spider")
                if selected_dmu_ccr and "merged_ccr" in results:
                    spider_fig_ccr = plot_benchmark_spider(results["merged_ccr"], selected_dmu_ccr, proposal['inputs'], proposal['outputs'])
                    st.plotly_chart(spider_fig_ccr, use_container_width=True)

        with tab_bcc:
            st.subheader("Tabla de Eficiencias y Slacks (Modelo BCC)")
            st.dataframe(results.get("df_bcc"))
            st.subheader("Visualización de Eficiencia (BCC)")
            if "hist_bcc" in results:
                st.plotly_chart(results["hist_bcc"], use_container_width=True)
