# main.py - VERSIÓN CON INDENTACIÓN CORREGIDA
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

# ... (resto de funciones cacheadas)

@st.cache_data
def generate_analysis_proposals(df_columns: list[str], df_head: pd.DataFrame):
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

    def render_preliminary_analysis_step(self, active_scenario):
        st.header(f"Paso 1b: Exploración Preliminar de Datos", divider="blue")
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
        
        # Lógica de selección de inputs/outputs
        if st.button("Confirmar Configuración", type="primary"):
            # ... tu lógica aquí ...
            active_scenario['app_status'] = "proposal_selected"
            st.rerun()

    def render_main_dashboard(self, active_scenario):
        st.header(f"Paso 3: Configuración y Análisis del Modelo", divider="blue")
        model_options = {"Radial (CCR/BCC)": "CCR_BCC", "No Radial (SBM)": "SBM"}
        model_name = st.selectbox("1. Selecciona el tipo de modelo DEA:", list(model_options.keys()))
        model_key = model_options[model_name]
        active_scenario['dea_config'] = {'model': model_key}

        # --- BLOQUE CON INDENTACIÓN CORREGIDA ---
        st.markdown("---")
        with st.expander("Checklist de Buenas Prácticas Metodológicas (Recomendado)"):
            # Este bloque ahora está correctamente indentado
            st.info("Este checklist fomenta la autocrítica antes de ejecutar el modelo.")
            
            if 'checklist_responses' not in active_scenario:
                active_scenario['checklist_responses'] = {}

            active_scenario['checklist_responses']['homogeneity'] = st.checkbox("¿He verificado que las unidades (DMUs) son suficientemente homogéneas?", key=f"check_homogeneity_{st.session_state.active_scenario_id}")
            
            # ... (resto de los checkboxes) ...
        st.markdown("---")
        
        if st.button("Ejecutar Análisis DEA", type="primary", use_container_width=True):
            with st.spinner("Ejecutando análisis..."):
                # ... Lógica de ejecución del análisis ...
                st.rerun()

        if active_scenario.get("dea_results"):
            st.header("Resultados del Análisis", divider="blue")
            st.dataframe(active_scenario["dea_results"]['main_df'])
            self.render_deliberation_workshop(active_scenario)

    def render_deliberation_workshop(self, active_scenario):
        st.header("Paso 4: Taller de Auditoría Metodológica", divider="blue")
        # ... (código completo para el taller deliberativo) ...

# --- FUNCIÓN PRINCIPAL ---
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
    active_scenario = get_active_scenario()

    if not active_scenario:
        renderer.render_upload_step()
    else:
        app_status = active_scenario.get('app_status', 'initial')
        if app_status == "data_loaded":
            renderer.render_preliminary_analysis_step(active_scenario)
        elif app_status == "file_loaded":
            renderer.render_proposal_step(active_scenario)
        elif app_status == "proposal_selected":
            # Suponiendo que tienes una función para esto
            # renderer.render_validation_step(active_scenario)
            pass
        elif app_status in ["validated", "results_ready"]:
            renderer.render_main_dashboard(active_scenario)
        else: # 'initial' state but with an active scenario (e.g., after loading)
             renderer.render_upload_step()


if __name__ == "__main__":
    main()
