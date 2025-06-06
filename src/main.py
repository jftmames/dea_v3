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
from dea_models.visualizations import plot_hypothesis_distribution, plot_benchmark_spider, plot_correlation

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
    st.session_state.inquiry_tree = None
    st.session_state.chart_to_show = None

if 'app_status' not in st.session_state:
    initialize_state()

# ... (Las funciones de caché no cambian) ...
@st.cache_data
def run_dea_analysis(_df, dmu_col, input_cols, output_cols):
    return mostrar_resultados(_df.copy(), dmu_col, input_cols, output_cols)

@st.cache_data
def run_inquiry_engine(root_question, _context):
    return generate_inquiry(root_question, context=_context)

@st.cache_data
def get_analysis_proposals(_df):
    return generate_analysis_proposals(_df.columns.tolist(), _df.head())

# --- 4) Flujo principal ---
st.title("💡 DEA Deliberativo con IA")
st.markdown("Una herramienta para analizar la eficiencia y razonar sobre sus causas con ayuda de Inteligencia Artificial.")

# ETAPA 1 y 2 ... (sin cambios)
st.header("Paso 1: Carga tus Datos", divider="blue")
uploaded_file = st.file_uploader("Sube un fichero CSV", type=["csv"])
if uploaded_file:
    #... (código de carga de fichero sin cambios)
    if st.session_state.get('_file_id') != uploaded_file.file_id:
        initialize_state()
        st.session_state._file_id = uploaded_file.file_id
        try: st.session_state.df = pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            st.session_state.df = pd.read_csv(uploaded_file, sep=';')
        st.session_state.app_status = "file_loaded"
        st.rerun()

if st.session_state.app_status != "initial":
    df = st.session_state.df
    if st.session_state.app_status in ["file_loaded", "proposal_selected"]:
        st.header("Paso 2: Elige un Enfoque de Análisis", divider="blue")
        # ... (código de propuestas de la IA sin cambios) ...
        if 'proposals' not in st.session_state:
            with st.spinner("La IA está analizando tus datos..."):
                proposals_data = get_analysis_proposals(df)
                st.session_state.proposals = proposals_data.get("proposals", [])
        if not st.session_state.get("proposals"):
            st.error("La IA no pudo generar propuestas.")
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
        
        # --- ETAPA 4: TALLER DE HIPÓTESIS DIRIGIDO POR EL USUARIO ---
        st.header("Paso 4: Razona y Explora las Causas con IA", divider="blue")
        
        col_map, col_workbench = st.columns([2,1])
        with col_map:
            st.subheader("Mapa de Razonamiento (IA)", anchor=False)
            if 'inquiry_tree' not in st.session_state: st.session_state.inquiry_tree = None

            if st.button("Generar/Inspirar con nuevo Mapa de Razonamiento", use_container_width=True):
                 with st.spinner("La IA está generando un mapa de ideas..."):
                    context = {"inputs": proposal['inputs'], "outputs": proposal['outputs'], "avg_efficiency_ccr": results["df_ccr"]["tec_efficiency_ccr"].mean()}
                    root_question = f"Bajo el enfoque '{proposal['title']}', ¿cuáles son las posibles causas de la ineficiencia?"
                    tree, error = run_inquiry_engine(root_question, context)
                    if error: st.error(f"Error: {error}")
                    st.session_state.inquiry_tree = tree
            
            if st.session_state.get("inquiry_tree"):
                st.plotly_chart(to_plotly_tree(st.session_state.inquiry_tree), use_container_width=True)
                eee_metrics = compute_eee(st.session_state.inquiry_tree, depth_limit=3, breadth_limit=5)
                st.caption(f"Calidad del Razonamiento (EEE): {eee_metrics['score']:.2%}")

        with col_workbench:
            st.subheader("Taller de Hipótesis (Usuario)", anchor=False)
            st.info("Usa este taller para explorar tus propias hipótesis, inspirado por el mapa de la IA.")
            
            all_vars = proposal['inputs'] + proposal['outputs']
            chart_type = st.selectbox("1. Elige un tipo de análisis:", ["Análisis de Distribución", "Análisis de Correlación"], key="wb_chart_type")

            if chart_type == "Análisis de Distribución":
                var_dist = st.selectbox("2. Elige la variable a analizar:", all_vars, key="wb_var_dist")
                if st.button("Generar Gráfico", key="gen_dist"):
                    st.session_state.chart_to_show = {"type": "distribution", "var": var_dist}
            
            elif chart_type == "Análisis de Correlación":
                var_x = st.selectbox("2. Elige la variable para el eje X:", all_vars, key="wb_var_x")
                var_y = st.selectbox("3. Elige la variable para el eje Y:", all_vars, key="wb_var_y")
                if st.button("Generar Gráfico", key="gen_corr"):
                     st.session_state.chart_to_show = {"type": "correlation", "var_x": var_x, "var_y": var_y}

        placeholder = st.container()
        if st.session_state.get("chart_to_show"):
            chart_info = st.session_state.chart_to_show
            with placeholder:
                st.subheader("Resultado de tu Hipótesis", anchor=False)
                if chart_info["type"] == "distribution":
                    fig = plot_hypothesis_distribution(results['df_ccr'], df, chart_info["var"], df.columns[0])
                elif chart_info["type"] == "correlation":
                    fig = plot_correlation(results['df_ccr'], df, chart_info["var_x"], chart_info["var_y"], df.columns[0])
                st.plotly_chart(fig, use_container_width=True)
                if st.button("Limpiar gráfico"):
                    st.session_state.chart_to_show = None
                    st.rerun()

        # ETAPA 5: Resultados Detallados
        st.header("Paso 5: Resultados Numéricos y Gráficos Detallados", divider="blue")
        # ... (código de la etapa 5 sin cambios) ...
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
