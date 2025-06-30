# main.py - VERSIÓN FINAL CON WORKFLOW DELIBERATIVO RESTAURADO
import sys
import os
import pandas as pd
import streamlit as st
import io
import json
import uuid
import openai
import plotly.express as px

# --- 0) AJUSTE DEL PYTHONPATH Y CONFIGURACIÓN INICIAL ---
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Configuración de la página de Streamlit
st.set_page_config(layout="wide", page_title="DEA Deliberative Modeler")

# --- 1) IMPORTACIONES DE MÓDULOS DEL PROYECTO ---
from analysis_dispatcher import execute_analysis
from inquiry_engine import generate_inquiry, to_plotly_tree
from epistemic_metrics import compute_eee
from data_validator import validate as validate_data
from report_generator import generate_html_report, generate_excel_report
from dea_models.visualizations import plot_hypothesis_distribution, plot_correlation
from dea_models.auto_tuner import generate_candidates, evaluate_candidates
from openai_helpers import explain_inquiry_tree

# --- 2) GESTIÓN DE ESTADO MULTI-ESCENARIO ---

def create_new_scenario(name: str = "Modelo Base", source_scenario_id: str = None):
    new_id = str(uuid.uuid4())
    if source_scenario_id and source_scenario_id in st.session_state.scenarios:
        st.session_state.scenarios[new_id] = st.session_state.scenarios[source_scenario_id].copy()
        st.session_state.scenarios[new_id]['name'] = f"Copia de {st.session_state.scenarios[source_scenario_id]['name']}"
        st.session_state.scenarios[new_id]['dea_results'] = None
        st.session_state.scenarios[new_id]['inquiry_tree'] = None
        st.session_state.scenarios[new_id]['user_justifications'] = {}
        st.session_state.scenarios[new_id]['checklist_responses'] = {}
        st.session_state.scenarios[new_id]['app_status'] = "data_loaded"
    else:
        st.session_state.scenarios[new_id] = {
            "name": name, "df": st.session_state.get("global_df"), "app_status": "initial",
            "proposals_data": None, "selected_proposal": None, "dea_config": {},
            "dea_results": None, "inquiry_tree": None, "tree_explanation": None,
            "chart_to_show": None, "user_justifications": {}, "data_overview": {},
            "checklist_responses": {}
        }
    st.session_state.active_scenario_id = new_id

def get_active_scenario():
    active_id = st.session_state.get('active_scenario_id')
    return st.session_state.scenarios.get(active_id)

def initialize_global_state():
    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = {}
        st.session_state.active_scenario_id = None
        st.session_state.global_df = None

def reset_all():
    st.cache_data.clear()
    st.cache_resource.clear()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_global_state()

# --- 3) FUNCIONES DE CACHÉ Y LÓGICA DE IA ---
@st.cache_data
def cached_run_dea_analysis(_df, dmu_col, input_cols, output_cols, model_key, period_col):
    return execute_analysis(_df.copy(), dmu_col, input_cols, output_cols, model_key, period_column=period_col)

@st.cache_data
def cached_run_inquiry_engine(root_question, _context):
    return generate_inquiry(root_question, context=_context)

@st.cache_data
def generate_analysis_proposals(df_columns: list[str], df_head: pd.DataFrame):
    # Esta es una función simulada. Reemplazar con tu llamada real a la IA
    proposals = [
        {"title": "Eficiencia Operativa", "reasoning": "Mide la eficiencia en el uso de recursos básicos.", "inputs": [df_columns[1]] if len(df_columns) > 1 else [], "outputs": [df_columns[-1]] if len(df_columns) > 1 else []},
        {"title": "Productividad General", "reasoning": "Analiza la capacidad de generar múltiples outputs desde múltiples inputs.", "inputs": df_columns[1:3], "outputs": df_columns[3:]}
    ]
    return {"proposals": proposals}


# --- CLASE ENCAPSULADORA DE LA UI ---
class AppRenderer:
    def __init__(self):
        pass

    def render_upload_step(self):
        st.header("Paso 1: Carga tus Datos", divider="blue")
        source_option = st.radio("Elige una fuente de datos:", ('Usar un caso de estudio', 'Subir un archivo CSV'), key='data_source_radio')
        df_to_load = None
        file_name = None
        if source_option == 'Usar un caso de estudio':
            datasets_path = os.path.join(script_dir, 'datasets')
            try:
                available_datasets = [f for f in os.listdir(datasets_path) if f.endswith('.csv')]
                if not available_datasets:
                    st.warning("No se encontraron datasets.")
                    return
                selected_dataset = st.selectbox('Selecciona un caso de estudio:', available_datasets)
                if selected_dataset:
                    df_to_load = pd.read_csv(os.path.join(datasets_path, selected_dataset))
                    file_name = selected_dataset
            except FileNotFoundError:
                st.error(f"Error: La carpeta `datasets` no se encuentra en `src`.")
                return
        else:
            uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"], label_visibility="collapsed")
            if uploaded_file:
                df_to_load = pd.read_csv(uploaded_file)
                file_name = uploaded_file.name
        
        if st.button("Cargar y Analizar Datos", type="primary", use_container_width=True):
            if df_to_load is not None:
                st.session_state.global_df = df_to_load
                create_new_scenario()
                active_scenario = get_active_scenario()
                active_scenario['app_status'] = "data_loaded"
                st.success(f"Datos de '{file_name}' cargados.")
                st.rerun()
            else:
                st.error("Por favor, selecciona un archivo válido.")

    def render_scenario_navigator(self):
        st.sidebar.title("Navegador de Escenarios")
        if not st.session_state.scenarios:
            st.sidebar.info("Carga datos para empezar.")
            return
        # ... (código del navegador sin cambios)

    def render_preliminary_analysis_step(self, active_scenario):
        st.header(f"Paso 1b: Exploración Preliminar de Datos", divider="blue")
        st.info("Revisa tus datos para identificar problemas antes del análisis.")
        df = active_scenario['df']
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        if not numerical_cols:
            st.warning("No se encontraron columnas numéricas.")
            return
        st.dataframe(df[numerical_cols].describe().T)
        for col in numerical_cols:
            st.plotly_chart(px.histogram(df, x=col, title=f"Distribución de {col}"), use_container_width=True)
        if st.button("Proceder al Paso 2: Elegir Enfoque", type="primary"):
            active_scenario['app_status'] = "file_loaded"
            st.rerun()

    def render_proposal_step(self, active_scenario):
        st.header(f"Paso 2: Elige un Enfoque de Análisis", divider="blue")
        if not active_scenario.get('proposals_data'):
            with st.spinner("La IA está sugiriendo enfoques..."):
                active_scenario['proposals_data'] = generate_analysis_proposals(active_scenario['df'].columns.tolist(), active_scenario['df'].head())
        proposals = active_scenario['proposals_data'].get("proposals", [])
        options = ["Configuración Manual"] + [p['title'] for p in proposals]
        selected_option = st.selectbox("Selecciona una opción:", options)
        # ... (resto de la lógica de selección de inputs/outputs)
        if st.button("Confirmar Configuración", type="primary"):
            # ... (lógica de confirmación)
            active_scenario['app_status'] = "proposal_selected"
            st.rerun()

    def render_validation_step(self, active_scenario):
        st.header(f"Paso 2b: Validación del Modelo", divider="gray")
        # ... (código de validación)
        if st.button("Proceder al Análisis", type="primary"):
            active_scenario['app_status'] = "validated"
            st.rerun()

    def render_main_dashboard(self, active_scenario):
        st.header(f"Paso 3: Configuración y Análisis del Modelo", divider="blue")
        model_options = {"Radial (CCR/BCC)": "CCR_BCC", "No Radial (SBM)": "SBM"}
        model_name = st.selectbox("1. Selecciona el tipo de modelo DEA:", list(model_options.keys()))
        model_key = model_options[model_name]
        active_scenario['dea_config'] = {'model': model_key}

        # --- CHECKLIST DELIBERATIVO ---
        with st.expander("Checklist de Buenas Prácticas Metodológicas"):
            # ... (código del checklist sin cambios)

        if st.button("Ejecutar Análisis DEA", type="primary", use_container_width=True):
            with st.spinner("Ejecutando análisis..."):
                results = cached_run_dea_analysis(active_scenario['df'], active_scenario['df'].columns[0], active_scenario['selected_proposal']['inputs'], active_scenario['selected_proposal']['outputs'], model_key, None)
                active_scenario['dea_results'] = results
                active_scenario['app_status'] = "results_ready"
                st.rerun()

        if active_scenario.get("dea_results"):
            st.header("Resultados del Análisis", divider="blue")
            st.dataframe(active_scenario["dea_results"]['main_df'])
            # --- LLAMADAS RESTAURADAS ---
            self.render_deliberation_workshop(active_scenario)
            self.render_download_section(active_scenario)

    def render_deliberation_workshop(self, active_scenario):
        st.header("Paso 4: Taller de Auditoría Metodológica", divider="blue")
        st.info("Usa la IA para generar un mapa de preguntas que auditen la robustez de tu análisis.")
        
        if st.button("Generar Mapa Metodológico", use_container_width=True):
            with st.spinner("La IA está generando el árbol de auditoría..."):
                context = {"model": active_scenario['dea_config']['model'], "inputs": active_scenario['selected_proposal']['inputs'], "outputs": active_scenario['selected_proposal']['outputs']}
                tree, error = cached_run_inquiry_engine("Generar un árbol de auditoría metodológica para este análisis DEA", context)
                if error:
                    st.error(f"Error al generar el mapa: {error}")
                else:
                    active_scenario['inquiry_tree'] = tree
                    active_scenario['user_justifications'] = {}
        
        if active_scenario.get("inquiry_tree"):
            eee_metrics = compute_eee(active_scenario['inquiry_tree'])
            st.metric("Calidad del Juicio (EEE)", f"{eee_metrics['score']:.2%}")
            self.render_interactive_inquiry_tree(active_scenario)
    
    def render_interactive_inquiry_tree(self, active_scenario):
        tree = active_scenario.get("inquiry_tree", {})
        # ... (código para renderizar el árbol y las justificaciones)

    def render_download_section(self, active_scenario):
        st.subheader("Exportar Análisis", divider="gray")
        col1, col2 = st.columns(2)
        with col1:
            html_report = generate_html_report(active_scenario.get('dea_results', {}), checklist_responses=active_scenario.get("checklist_responses", {}))
            st.download_button("Descargar Informe HTML", html_report, "report.html", "text/html", use_container_width=True)
        with col2:
            excel_report = generate_excel_report(active_scenario.get('dea_results', {}), checklist_responses=active_scenario.get("checklist_responses", {}))
            st.download_button("Descargar Informe Excel", excel_report, "report.xlsx", use_container_width=True)

# --- FUNCIÓN PRINCIPAL DE LA APLICACIÓN ---
def main():
    initialize_global_state()
    logo_path = os.path.join(script_dir, 'assets', 'logo.png')
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=200)
    else:
        st.sidebar.warning("Logo no encontrado.")
    st.sidebar.title("DEA Deliberative Modeler")
    if st.sidebar.button("🔴 Empezar Nueva Sesión"):
        reset_all()
        st.rerun()
    st.sidebar.divider()
    renderer = AppRenderer()
    renderer.render_scenario_navigator()
    active_scenario = get_active_scenario()
    
    analysis_tab, comparison_tab, challenges_tab = st.tabs(["Análisis Activo", "Comparar", "Retos DEA"])
    with analysis_tab:
        if not active_scenario:
            renderer.render_upload_step()
            return
            
        app_status = active_scenario.get('app_status', 'initial')
        if app_status == "initial": renderer.render_upload_step()
        elif app_status == "data_loaded": renderer.render_preliminary_analysis_step(active_scenario)
        elif app_status == "file_loaded": renderer.render_proposal_step(active_scenario)
        elif app_status == "proposal_selected": renderer.render_validation_step(active_scenario)
        elif app_status in ["validated", "results_ready"]: renderer.render_main_dashboard(active_scenario)

if __name__ == "__main__":
    main()
