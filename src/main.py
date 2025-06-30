# main.py - VERSIÓN FINAL COMPLETA, CORREGIDA Y CON TODAS LAS MEJORAS
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

st.set_page_config(layout="wide", page_title="DEA Deliberative Modeler")

# --- 1) IMPORTACIONES DE MÓDULOS DEL PROYECTO ---
from analysis_dispatcher import execute_analysis
from inquiry_engine import generate_inquiry, to_plotly_tree
from epistemic_metrics import compute_eee
from data_validator import validate as validate_data
from report_generator import generate_html_report, generate_excel_report
from dea_models.auto_tuner import generate_candidates, evaluate_candidates
from openai_helpers import explain_inquiry_tree

# --- 2) GESTIÓN DE ESTADO ---

def create_new_scenario(name: str = "Modelo Base", source_scenario_id: str = None):
    new_id = str(uuid.uuid4())
    if source_scenario_id and source_scenario_id in st.session_state.scenarios:
        st.session_state.scenarios[new_id] = st.session_state.scenarios[source_scenario_id].copy()
        st.session_state.scenarios[new_id]['name'] = f"Copia de {st.session_state.scenarios[source_scenario_id]['name']}"
    else:
        st.session_state.scenarios[new_id] = {
            "name": name, "df": st.session_state.get("global_df"), "app_status": "initial",
            "proposals_data": None, "selected_proposal": None, "dea_config": {},
            "dea_results": None, "inquiry_tree": None, "tree_explanation": None,
            "user_justifications": {}, "data_overview": {}, "checklist_responses": {}
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
    keys_to_delete = list(st.session_state.keys())
    for key in keys_to_delete:
        del st.session_state[key]
    initialize_global_state()

# --- 3) FUNCIONES DE LÓGICA DE IA (SIMULADAS PARA DESARROLLO) ---

@st.cache_data
def cached_run_dea_analysis(_df, dmu_col, input_cols, output_cols, model_key, period_col):
    return execute_analysis(_df.copy(), dmu_col, input_cols, output_cols, model_key, period_column=period_col)

@st.cache_data
def cached_run_inquiry_engine(root_question, _context):
    return generate_inquiry(root_question, context=_context)

@st.cache_data
def generate_analysis_proposals(df_columns: list[str], df_head: pd.DataFrame):
    # Función simulada. Reemplazar por la llamada real a la IA.
    proposals = []
    if len(df_columns) > 1:
        proposals.append({"title": "Eficiencia Operativa", "reasoning": "Mide la eficiencia en el uso de recursos básicos.", "inputs": [df_columns[1]], "outputs": [df_columns[-1]]})
    if len(df_columns) > 4:
        proposals.append({"title": "Productividad General", "reasoning": "Analiza la capacidad de generar múltiples outputs desde múltiples inputs.", "inputs": df_columns[1:3], "outputs": df_columns[3:5]})
    return {"proposals": proposals}


# --- CLASE ENCAPSULADORA DE LA UI ---
class AppRenderer:
    def __init__(self):
        pass

    def render_upload_step(self):
        st.header("Paso 1: Cargar Datos", divider="blue")
        source_option = st.radio("Elige una fuente:", ('Usar caso de estudio', 'Subir CSV'), key='data_source', horizontal=True)
        df_to_load, file_name = None, None

        if source_option == 'Usar caso de estudio':
            path = os.path.join(script_dir, 'datasets')
            try:
                datasets = [f for f in os.listdir(path) if f.endswith('.csv')]
                if not datasets:
                    st.warning("La carpeta `datasets` está vacía o no se encuentra en `src`."); return
                selected = st.selectbox('Selecciona un caso:', datasets)
                if selected:
                    df_to_load, file_name = pd.read_csv(os.path.join(path, selected)), selected
            except FileNotFoundError:
                st.error("Error: La carpeta `datasets` no se encuentra en `src`."); return
        else:
            uploaded_file = st.file_uploader("Sube tu archivo CSV", type="csv")
            if uploaded_file:
                df_to_load, file_name = pd.read_csv(uploaded_file), uploaded_file.name

        if st.button("Cargar y Continuar", type="primary", disabled=(df_to_load is None)):
            st.session_state.global_df = df_to_load
            create_new_scenario()
            get_active_scenario()['app_status'] = "data_loaded"
            st.rerun()

    def render_preliminary_analysis_step(self, scenario):
        st.header(f"Paso 1b: Exploración de Datos para '{scenario['name']}'", divider="blue")
        df = scenario['df']
        st.dataframe(df.describe())
        if st.button("Proceder al Paso 2", type="primary"):
            scenario['app_status'] = "file_loaded"
            st.rerun()

    def render_proposal_step(self, scenario):
        st.header(f"Paso 2: Elige un Enfoque de Análisis para '{scenario['name']}'", divider="blue")
        if not scenario.get('proposals_data'):
            with st.spinner("IA sugiriendo enfoques..."):
                scenario['proposals_data'] = generate_analysis_proposals(scenario['df'].columns.tolist(), scenario['df'].head())
        
        proposals = scenario['proposals_data'].get("proposals", [])
        options = ["Configuración Manual"] + [p.get('title') for p in proposals]
        choice = st.selectbox("Selecciona una opción:", options)
        
        inputs, outputs = [], []
        cols = [c for c in scenario['df'].columns if scenario['df'][c].dtype in ['int64', 'float64']]
        
        if choice == "Configuración Manual":
            inputs = st.multiselect("Selecciona Inputs:", options=cols)
            outputs = st.multiselect("Selecciona Outputs:", options=[c for c in cols if c not in inputs])
        else:
            proposal = next((p for p in proposals if p['title'] == choice), None)
            if proposal:
                st.info(f"Razonamiento IA: {proposal['reasoning']}")
                inputs = st.multiselect("Inputs:", options=cols, default=proposal.get('inputs', []))
                outputs = st.multiselect("Outputs:", options=[c for c in cols if c not in inputs], default=proposal.get('outputs', []))

        if st.button("Confirmar Enfoque", type="primary", disabled=(not inputs or not outputs)):
            scenario['selected_proposal'] = {'title': choice, 'inputs': inputs, 'outputs': outputs}
            scenario['app_status'] = "proposal_selected"
            st.rerun()
            
    def render_main_dashboard(self, scenario):
        st.header(f"Paso 3: Configuración y Análisis para '{scenario['name']}'", divider="blue")
        model_key = st.selectbox("Tipo de Modelo DEA:", ["CCR_BCC", "SBM"], key=f"model_{scenario['name']}")
        scenario['dea_config']['model'] = model_key

        # --- CHECKLIST DELIBERATIVO INTEGRADO ---
        st.markdown("---")
        with st.expander("Checklist de Buenas Prácticas Metodológicas"):
            st.info("Este checklist fomenta la autocrítica antes de ejecutar el modelo. Tus respuestas se guardarán en el informe final.")
            if 'checklist_responses' not in scenario:
                scenario['checklist_responses'] = {}
            
            scenario['checklist_responses']['homogeneity'] = st.checkbox("¿He verificado que las DMUs son suficientemente homogéneas?", key=f"check_homo_{scenario['name']}")
            
            num_dmus = len(scenario['df'])
            num_inputs = len(scenario['selected_proposal'].get('inputs', []))
            num_outputs = len(scenario['selected_proposal'].get('outputs', []))
            rule_value = 3 * (num_inputs + num_outputs)
            rule_text = f"¿He comprobado la regla empírica? (Nº DMUs ≥ 3 * (Inputs+Outputs)) --- En tu caso: **{num_dmus} ≥ {rule_value}**"
            scenario['checklist_responses']['rule_of_thumb'] = st.checkbox(rule_text, key=f"check_rule_{scenario['name']}")
            
            scenario['checklist_responses']['isotonicity'] = st.checkbox("¿He considerado la isotocidad? (A más inputs, no menos outputs)", key=f"check_iso_{scenario['name']}")
        st.markdown("---")

        if st.button("Ejecutar Análisis DEA", type="primary", use_container_width=True):
            with st.spinner("Calculando..."):
                scenario['dea_results'] = cached_run_dea_analysis(
                    scenario['df'], scenario['df'].columns[0],
                    scenario['selected_proposal']['inputs'], scenario['selected_proposal']['outputs'],
                    model_key, None
                )
            scenario['app_status'] = "results_ready"
            st.rerun()

        if scenario.get("dea_results"):
            st.header("Resultados del Análisis", divider="gray")
            st.dataframe(scenario["dea_results"]['main_df'])
            self.render_deliberation_workshop(scenario)
            self.render_download_section(scenario)

    def render_deliberation_workshop(self, scenario):
        st.header("Paso 4: Taller de Auditoría Metodológica", divider="blue")
        if st.button("Generar Mapa Metodológico con IA", key=f"gen_map_{scenario['name']}"):
            with st.spinner("IA generando árbol de preguntas..."):
                context = {"model": scenario['dea_config']['model'], "inputs": scenario['selected_proposal']['inputs'], "outputs": scenario['selected_proposal']['outputs']}
                tree, error = cached_run_inquiry_engine("Generar un árbol de auditoría metodológica para este análisis DEA", context)
                if error: st.error(f"Error al generar mapa: {error}")
                else: scenario['inquiry_tree'] = tree
        
        if scenario.get("inquiry_tree"):
            eee_metrics = compute_eee(scenario['inquiry_tree'])
            st.metric("Calidad del Juicio (EEE)", f"{eee_metrics['score']:.2%}")

    def render_download_section(self, scenario):
        st.subheader("Exportar", divider="gray")
        html_report = generate_html_report(scenario)
        st.download_button("Descargar Informe HTML", html_report, "reporte.html", "text/html", use_container_width=True)

# --- FUNCIÓN PRINCIPAL DE LA APLICACIÓN ---
def main():
    initialize_global_state()
    logo_path = os.path.join(script_dir, 'assets', 'logo.png')
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=150)
    
    st.sidebar.title("DEA Deliberative Modeler")
    if st.sidebar.button("🔴 Empezar Nueva Sesión"):
        reset_all()
        st.rerun()
    st.sidebar.divider()
    
    renderer = AppRenderer()
    # No es necesario llamar a render_scenario_navigator aquí si no se usa

    active_scenario = get_active_scenario()
    
    analysis_tab, comparison_tab, challenges_tab = st.tabs(["Análisis Activo", "Comparar", "Retos DEA"])
    with analysis_tab:
        if not active_scenario:
            renderer.render_upload_step()
        else:
            app_status = active_scenario.get('app_status', 'initial')
            if app_status == "data_loaded": renderer.render_preliminary_analysis_step(active_scenario)
            elif app_status == "file_loaded": renderer.render_proposal_step(active_scenario)
            elif app_status in ["proposal_selected", "validated", "results_ready"]: renderer.render_main_dashboard(active_scenario)
            else: renderer.render_upload_step()

if __name__ == "__main__":
    main()
