# /src/main.py
# --- VERSIÓN COMPLETA Y REFACTORIZADA ---

import sys
import os
import pandas as pd
import streamlit as st
import io
import uuid
import datetime

# --- 1) CONFIGURACIÓN INICIAL ---
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
st.set_page_config(layout="wide", page_title="DEA Deliberative Modeler")

# --- 2) IMPORTACIONES DE MÓDULOS DEL PROYECTO ---
from analysis_dispatcher import execute_analysis
from inquiry_engine import InquiryEngine, InquiryNode, to_plotly_tree
from epistemic_metrics import compute_eee
from data_validator import validate as validate_data
from report_generator import generate_html_report, generate_excel_report
from dea_models.auto_tuner import generate_candidates, evaluate_candidates
from openai_helpers import explain_inquiry_tree, generate_analysis_proposals, chat_completion

# --- NUEVO: Importación selectiva y clara desde ui_components ---
from ui_components import (
    render_scenario_navigator,
    render_comparison_view,
    render_preliminary_analysis_step,
    render_proposal_step,
    render_validation_step,
    render_main_dashboard,
    render_dynamic_inquiry_workshop, # <-- La nueva función clave
    render_optimization_workshop,
    render_download_section,
    render_dea_challenges_tab
)

# --- 3) GESTIÓN DE ESTADO Y TRACKER EPISTÉMICO ---

def initialize_global_state():
    """Inicializa el estado global, incluyendo el motor y el tracker."""
    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = {}
        st.session_state.active_scenario_id = None
        st.session_state.global_df = None
        st.session_state.inquiry_engine = InquiryEngine()
        st.session_state.epistemic_events = []

def log_epistemic_event(event_type: str, data: dict):
    """Registra un evento deliberativo en el log de la sesión."""
    event = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event_type": event_type,
        "data": data
    }
    st.session_state.epistemic_events.append(event)

def get_event_log() -> list:
    """Devuelve el registro completo de eventos epistémicos."""
    return st.session_state.get('epistemic_events', [])

def create_new_scenario(name: str = "Modelo Base", source_scenario_id: str = None):
    """Crea un nuevo escenario, actualizado para InquiryNode."""
    new_id = str(uuid.uuid4())
    if source_scenario_id and source_scenario_id in st.session_state.scenarios:
        source_scenario = st.session_state.scenarios[source_scenario_id]
        new_scenario = source_scenario.copy()
        new_scenario['name'] = f"Copia de {source_scenario['name']}"
        new_scenario['dea_results'] = None
        new_scenario['inquiry_tree_node'] = None # Clave actualizada
        new_scenario['user_justifications'] = {}
        st.session_state.scenarios[new_id] = new_scenario
    else:
        st.session_state.scenarios[new_id] = {
            "name": name, "df": st.session_state.get("global_df"), "app_status": "initial",
            "proposals_data": None, "selected_proposal": None, "dea_config": {},
            "dea_results": None, "inquiry_tree_node": None, # Clave actualizada
            "tree_explanation": None, "user_justifications": {}, "data_overview": {}
        }
    st.session_state.active_scenario_id = new_id

def get_active_scenario():
    """Devuelve el diccionario del escenario activo."""
    active_id = st.session_state.get('active_scenario_id')
    return st.session_state.scenarios.get(active_id)

def reset_all():
    """Reinicia la aplicación."""
    st.cache_data.clear()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_global_state()

# --- 4) FUNCIONES DE CACHÉ ---
@st.cache_data
def cached_run_dea_analysis(_df, dmu_col, inputs, outputs, model, period):
    return execute_analysis(_df.copy(), dmu_col, inputs, outputs, model, period_column=period)

@st.cache_data
def cached_explain_tree(_tree_node):
    # La función debe ser adaptada para recibir el InquiryNode
    return explain_inquiry_tree(_tree_node)

# ... (otras funciones de caché como cached_generate_candidates, etc.)

# ---
# --- FUNCIÓN PRINCIPAL DE LA APLICACIÓN ---
# ---

def main():
    """Función principal que orquesta la aplicación multi-escenario."""
    initialize_global_state()

    st.sidebar.image("https://i.imgur.com/8y0N5c5.png", width=200)
    st.sidebar.title("DEA Deliberative Modeler")
    if st.sidebar.button("🔴 Empezar Nueva Sesión"):
        reset_all()
        st.rerun()
    st.sidebar.divider()

    render_scenario_navigator()

    # Mover la lógica de creación/clonación aquí para que se ejecute antes de renderizar el contenido
    if st.session_state.pop('_new_scenario_requested', False):
        create_new_scenario(name=f"Nuevo Modelo {len(st.session_state.scenarios) + 1}")
        st.rerun()
    if clone_id := st.session_state.pop('_clone_scenario_requested', None):
        create_new_scenario(source_scenario_id=clone_id)
        st.rerun()

    active_scenario = get_active_scenario()

    if not active_scenario:
        # Lógica de carga de datos (reemplaza a render_upload_step)
        st.header("Paso 1: Carga tus Datos", divider="blue")
        uploaded_file = st.file_uploader("Sube un fichero CSV", type=["csv"])
        if uploaded_file:
            try:
                df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')))
            except:
                df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('latin-1')), sep=';')
            st.session_state.global_df = df
            create_new_scenario()
            st.rerun()
        return

    analysis_tab, comparison_tab, challenges_tab = st.tabs([
        "Análisis del Escenario Activo", "Comparar Escenarios", "Retos del DEA"
    ])

    with analysis_tab:
        app_status = active_scenario.get('app_status', 'initial')

        if app_status == "data_loaded":
            render_preliminary_analysis_step(active_scenario)
        elif app_status == "file_loaded":
            render_proposal_step(active_scenario)
        elif app_status == "proposal_selected":
            render_validation_step(active_scenario)
        elif app_status in ["validated", "results_ready"]:
            render_main_dashboard(active_scenario, cached_run_dea_analysis, validate_data)
            if active_scenario.get('dea_results'):
                # --- LLAMADA A LA NUEVA FUNCIÓN DINÁMICA ---
                render_dynamic_inquiry_workshop(
                    active_scenario,
                    st.session_state.inquiry_engine, # Pasamos la instancia del motor
                    compute_eee # Pasamos la función de cálculo
                )
                # render_optimization_workshop(...) # Llamadas a otros pasos
                # render_download_section(...)

    with comparison_tab:
        render_comparison_view(st.session_state.scenarios, get_active_scenario, compute_eee)

    with challenges_tab:
        render_dea_challenges_tab()

if __name__ == "__main__":
    main()
