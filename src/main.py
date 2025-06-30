# main.py - VERSIÓN CORREGIDA Y CON CHECKLIST INTEGRADO
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
        st.session_state.scenarios[new_id]['app_status'] = "data_loaded"
    else:
        st.session_state.scenarios[new_id] = {
            "name": name, "df": st.session_state.get("global_df"), "app_status": "initial",
            "proposals_data": None, "selected_proposal": None, "dea_config": {},
            "dea_results": None, "inquiry_tree": None, "tree_explanation": None,
            "chart_to_show": None, "user_justifications": {}, "data_overview": {},
            "checklist_responses": {} # Inicializar checklist
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
# ... (Sin cambios aquí, se mantiene tu código)

# --- CLASE ENCAPSULADORA DE LA UI (VERSIÓN COMPLETA Y CORREGIDA) ---
class AppRenderer:
    def __init__(self):
        pass

    def render_upload_step(self):
        st.header("Paso 1: Carga tus Datos", divider="blue")
        st.info("Para comenzar, selecciona una fuente de datos. Puedes subir tu propio archivo CSV o utilizar uno de nuestros casos de estudio para empezar rápidamente.")

        source_option = st.radio(
            "Elige una fuente de datos:",
            ('Usar un caso de estudio', 'Subir un archivo CSV'),
            key='data_source_radio'
        )

        df_to_load = None
        file_name = None

        if source_option == 'Usar un caso de estudio':
            datasets_path = os.path.join(script_dir, 'datasets')
            try:
                available_datasets = [f for f in os.listdir(datasets_path) if f.endswith('.csv')]
                if not available_datasets:
                    st.warning("No se encontraron datasets en la carpeta `datasets`.")
                    return
                selected_dataset = st.selectbox('Selecciona un caso de estudio:', available_datasets)
                if selected_dataset:
                    file_path = os.path.join(datasets_path, selected_dataset)
                    df_to_load = pd.read_csv(file_path)
                    file_name = selected_dataset
            except FileNotFoundError:
                st.error(f"Error: La carpeta `datasets` no se encuentra. Asegúrate de que esté en la misma ubicación que `main.py` (dentro de `src`).")
                return

        elif source_option == 'Subir un archivo CSV':
            uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"], label_visibility="collapsed")
            if uploaded_file:
                try:
                    df_to_load = pd.read_csv(uploaded_file)
                    file_name = uploaded_file.name
                except Exception as e:
                    st.error(f"No se pudo leer el archivo. Error: {e}")
                    return
        
        if st.button("Cargar y Analizar Datos", type="primary", use_container_width=True):
            if df_to_load is not None:
                st.session_state.global_df = df_to_load
                create_new_scenario(name="Modelo Base")
                active_scenario = get_active_scenario()
                active_scenario['app_status'] = "data_loaded"
                st.success(f"Datos de '{file_name}' cargados. El análisis preliminar está listo.")
                st.rerun()
            else:
                st.error("Por favor, selecciona un archivo válido antes de cargar los datos.")

    def render_scenario_navigator(self):
        st.sidebar.title("Navegador de Escenarios")
        st.sidebar.markdown("Gestiona y compara tus modelos.")
        st.sidebar.divider()

        if not st.session_state.scenarios:
            st.sidebar.info("Carga datos para empezar.")
            return

        scenario_names = {sid: s['name'] for sid, s in st.session_state.scenarios.items()}
        active_id = st.session_state.get('active_scenario_id')
        if active_id not in scenario_names:
            active_id = next(iter(scenario_names), None)

        st.session_state.active_scenario_id = st.sidebar.selectbox(
            "Escenario Activo", options=list(st.session_state.scenarios.keys()),
            format_func=lambda sid: scenario_names.get(sid, "N/A"),
            index=list(st.session_state.scenarios.keys()).index(active_id) if active_id in st.session_state.scenarios else 0,
            key='scenario_selector'
        )
        # ... (código para clonar, renombrar, eliminar sin cambios)
        if st.sidebar.button("➕ Nuevo Escenario", help="Crea un nuevo modelo desde cero."):
            st.session_state._new_scenario_requested = True 
            st.rerun()

    def render_preliminary_analysis_step(self, active_scenario):
        st.header(f"Paso 1b: Exploración Preliminar de Datos para '{active_scenario['name']}'", divider="blue")
        # ... (código de esta función sin cambios)
        if st.button("Proceder al Paso 2: Elegir Enfoque de Análisis", type="primary", use_container_width=True):
            active_scenario['app_status'] = "file_loaded"
            st.rerun()

    def render_proposal_step(self, active_scenario):
        st.header(f"Paso 2: Elige un Enfoque de Análisis para '{active_scenario['name']}'", divider="blue")
        # ... (código de esta función sin cambios)
        if st.button("Confirmar y Validar Configuración", type="primary", use_container_width=True):
            # ... (lógica del botón)
            pass

    def render_validation_step(self, active_scenario):
        st.header(f"Paso 2b: Validación del Modelo para '{active_scenario['name']}'", divider="gray")
        # ... (código de esta función sin cambios)
        if st.button("Proceder al Análisis", key=f"validate_{st.session_state.active_scenario_id}", type="primary"):
            active_scenario['app_status'] = "validated"
            st.rerun()

    def render_main_dashboard(self, active_scenario):
        st.header(f"Paso 3: Configuración y Análisis para '{active_scenario['name']}'", divider="blue")
        st.markdown("Configura y ejecuta el modelo DEA. Asegúrate de que los inputs/outputs y el tipo de modelo sean correctos.")
        
        # ... (código de selección de inputs, outputs y modelo sin cambios)
        
        # --- INICIO DEL CÓDIGO DEL CHECKLIST ---
        st.markdown("---")
        with st.expander("Checklist de Buenas Prácticas Metodológicas (Recomendado)"):
            st.info("Este checklist es un 'guardarraíl cognitivo' para fomentar la autocrítica antes de ejecutar el modelo. Tus respuestas se guardarán en el informe final.")
            
            if 'checklist_responses' not in active_scenario:
                active_scenario['checklist_responses'] = {}

            active_scenario['checklist_responses']['homogeneity'] = st.checkbox(
                "¿He verificado que las unidades (DMUs) son suficientemente homogéneas y comparables entre sí?",
                key=f"check_homogeneity_{st.session_state.active_scenario_id}"
            )

            num_dmus = len(active_scenario['df'])
            num_inputs = len(active_scenario['selected_proposal'].get('inputs', []))
            num_outputs = len(active_scenario['selected_proposal'].get('outputs', []))
            rule_of_thumb_value = 3 * (num_inputs + num_outputs)
            
            rule_text = (
                f"¿He comprobado la regla empírica? (Nº de DMUs ≥ 3 * (Inputs + Outputs))"
                f" --- En tu caso: **{num_dmus} ≥ {rule_of_thumb_value}**"
            )
            active_scenario['checklist_responses']['rule_of_thumb'] = st.checkbox(
                rule_text,
                key=f"check_rule_thumb_{st.session_state.active_scenario_id}"
            )

            active_scenario['checklist_responses']['isotonicity'] = st.checkbox(
                "¿He considerado la isotocidad? (A más inputs, no debería haber menos outputs).",
                key=f"check_isotonicity_{st.session_state.active_scenario_id}"
            )
        st.markdown("---")
        # --- FIN DEL CÓDIGO DEL CHECKLIST ---

        if st.button(f"Ejecutar Análisis DEA para '{active_scenario['name']}'", type="primary", use_container_width=True):
            # ... (código del botón sin cambios)
            pass

        if active_scenario.get("dea_results"):
            # ... (código para mostrar resultados sin cambios)
            self.render_deliberation_workshop(active_scenario)
            self.render_download_section(active_scenario)

    def render_deliberation_workshop(self, active_scenario):
        st.header("Paso 4: Deliberación y Justificación Metodológica", divider="blue")
        # ... (código de esta función sin cambios)

    def render_download_section(self, active_scenario):
        if not active_scenario.get('dea_results'): return
        st.subheader("Exportar Análisis del Escenario", divider="gray")
        col1, col2 = st.columns(2)
        with col1:
            html_report = generate_html_report(
                analysis_results=active_scenario.get('dea_results', {}),
                inquiry_tree=active_scenario.get("inquiry_tree"),
                user_justifications=active_scenario.get("user_justifications", {}),
                data_overview_info=active_scenario.get("data_overview", {}),
                # --- LÍNEA AÑADIDA ---
                checklist_responses=active_scenario.get("checklist_responses", {})
            )
            st.download_button("Descargar Informe HTML", html_report, f"report.html", "text/html", use_container_width=True)
        with col2:
            excel_report = generate_excel_report(
                analysis_results=active_scenario.get('dea_results', {}),
                inquiry_tree=active_scenario.get("inquiry_tree"),
                user_justifications=active_scenario.get("user_justifications", {}),
                data_overview_info=active_scenario.get("data_overview", {}),
                # --- LÍNEA AÑADIDA ---
                checklist_responses=active_scenario.get("checklist_responses", {})
            )
            st.download_button("Descargar Informe Excel", excel_report, f"report.xlsx", use_container_width=True)
            
    # ... (resto de funciones de renderizado: render_comparison_view, etc.)
    
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
    st.sidebar.divider()

    active_scenario = get_active_scenario()
    
    analysis_tab, comparison_tab, challenges_tab = st.tabs([
        "Análisis del Escenario Activo", "Comparar Escenarios", "Retos del DEA"
    ])

    with analysis_tab:
        app_status = 'initial'
        if active_scenario:
            app_status = active_scenario.get('app_status', 'initial')

        if app_status == "initial":
            renderer.render_upload_step()
        elif app_status == "data_loaded":
            renderer.render_preliminary_analysis_step(active_scenario)
        elif app_status == "file_loaded":
            renderer.render_proposal_step(active_scenario)
        elif app_status == "proposal_selected":
            renderer.render_validation_step(active_scenario)
        elif app_status in ["validated", "results_ready"]:
            renderer.render_main_dashboard(active_scenario)
            
if __name__ == "__main__":
    main()
