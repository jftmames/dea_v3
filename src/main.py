# /src/main.py
# --- VERSIÓN ACTUALIZADA Y COHERENTE ---

import sys
import os
import pandas as pd
import streamlit as st
import io
import uuid

# --- 1) CONFIGURACIÓN INICIAL (Sin cambios) ---
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
st.set_page_config(layout="wide", page_title="DEA Deliberative Modeler")

# --- 2) IMPORTACIONES DE MÓDULOS DEL PROYECTO (MODIFICADO) ---
from analysis_dispatcher import execute_analysis
# Se importa el motor completo y la nueva clase de nodo
from inquiry_engine import InquiryEngine, InquiryNode, to_plotly_tree
from epistemic_metrics import compute_eee
from data_validator import validate as validate_data
from report_generator import generate_html_report, generate_excel_report
from dea_models.auto_tuner import generate_candidates, evaluate_candidates
from openai_helpers import explain_inquiry_tree, generate_analysis_proposals # Se asume que generate_analysis_proposals está aquí

# --- NUEVO: Se importan todos los componentes de UI ---
from ui_components import (
    render_scenario_navigator,
    render_comparison_view,
    render_eee_explanation,
    render_dynamic_inquiry_workshop, # La nueva función dinámica
    render_optimization_workshop,
    render_download_section,
    render_main_dashboard,
    render_preliminary_analysis_step,
    render_proposal_step,
    render_validation_step,
    render_dea_challenges_tab
)

# --- 3) GESTIÓN DE ESTADO Y TRACKER EPISTÉMICO (MODIFICADO) ---

def initialize_global_state():
    """Inicializa el estado global, incluyendo el motor de indagación y el tracker."""
    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = {}
        st.session_state.active_scenario_id = None
        st.session_state.global_df = None
        # --- NUEVO: Inicialización del motor y el tracker ---
        st.session_state.inquiry_engine = InquiryEngine()
        st.session_state.epistemic_events = []

def log_epistemic_event(event_type: str, data: dict):
    """NUEVO: Registra un evento deliberativo en el log de la sesión."""
    import datetime
    event = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event_type": event_type,
        "data": data
    }
    st.session_state.epistemic_events.append(event)
    print(f"EVENT LOGGED: {event_type}") # Para depuración en consola

def get_event_log() -> list:
    """NUEVO: Devuelve el registro completo de eventos epistémicos."""
    return st.session_state.get('epistemic_events', [])

def create_new_scenario(name: str = "Modelo Base", source_scenario_id: str = None):
    """Crea un nuevo escenario, en blanco o clonado. Actualizado para InquiryNode."""
    new_id = str(uuid.uuid4())
    if source_scenario_id and source_scenario_id in st.session_state.scenarios:
        # Lógica de clonación (adaptada para el nuevo árbol)
        source_scenario = st.session_state.scenarios[source_scenario_id]
        new_scenario = source_scenario.copy()
        new_scenario['name'] = f"Copia de {source_scenario['name']}"
        # Se resetean los resultados y el árbol al clonar
        new_scenario['dea_results'] = None
        new_scenario['inquiry_tree_node'] = None # Usamos la nueva clave
        new_scenario['user_justifications'] = {} # Mantenemos esto para compatibilidad con reportes
        st.session_state.scenarios[new_id] = new_scenario
    else:
        # Lógica para escenario nuevo
        st.session_state.scenarios[new_id] = {
            "name": name,
            "df": st.session_state.get("global_df"),
            "app_status": "initial",
            "proposals_data": None,
            "selected_proposal": None,
            "dea_config": {},
            "dea_results": None,
            "inquiry_tree_node": None, # Nueva clave para el objeto InquiryNode
            "tree_explanation": None,
            "user_justifications": {}, # Se puede ir eliminando, pero lo dejamos por ahora
            "data_overview": {}
        }
    st.session_state.active_scenario_id = new_id

def get_active_scenario():
    """Devuelve el diccionario del escenario activo. Sin cambios."""
    active_id = st.session_state.get('active_scenario_id')
    return st.session_state.scenarios.get(active_id) if active_id else None

def reset_all():
    """Reinicia la aplicación. Sin cambios."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_global_state()
    st.rerun()

# --- 4) FUNCIONES DE CACHÉ (MODIFICADO) ---
# Se eliminan las funciones de caché relacionadas con el antiguo inquiry_engine.
# El nuevo motor gestiona su propia lógica y no necesita ser cacheado de esta manera.

@st.cache_data
def cached_run_dea_analysis(_df, dmu_col, input_cols, output_cols, model_key, period_col):
    """Sin cambios."""
    return execute_analysis(_df.copy(), dmu_col, input_cols, output_cols, model_key, period_column=period_col)

# ... (resto de funciones de caché sin cambios) ...

# --- 5) FUNCIÓN PRINCIPAL DE LA APLICACIÓN (MODIFICADO) ---

def main():
    """Función principal que orquesta la aplicación multi-escenario."""
    initialize_global_state()

    st.sidebar.image("https://i.imgur.com/8y0N5c5.png", width=200)
    st.sidebar.title("DEA Deliberative Modeler")
    st.sidebar.markdown("Herramienta para análisis de eficiencia y deliberación metodológica asistida por IA.")
    if st.sidebar.button("🔴 Empezar Nueva Sesión", help="Borra todos los datos y escenarios."):
        reset_all()
    st.sidebar.divider()
    
    # --- Llamada a los componentes de la UI ---
    render_scenario_navigator()

    st.sidebar.divider()
    with st.sidebar.expander("Ver Log de Eventos (Debug)"):
         st.json(get_event_log())

    active_scenario = get_active_scenario()

    # Si no hay escenario (inicio de sesión), mostramos la pantalla de carga
    if not active_scenario:
        st.header("Paso 1: Carga tus Datos para Iniciar la Sesión", divider="blue")
        st.info("Para comenzar, sube un conjunto de datos en formato CSV.")
        # ... (Aquí va la lógica de render_upload_step) ...
        return # Termina la ejecución aquí hasta que se carguen los datos

    # --- Lógica de Pestañas ---
    analysis_tab, comparison_tab, challenges_tab = st.tabs([
        "Análisis del Escenario Activo", "Comparar Escenarios", "Retos del DEA"
    ])

    with analysis_tab:
        app_status = active_scenario.get('app_status', 'initial')
        
        # --- Flujo de la aplicación basado en estados ---
        if app_status == 'data_loaded':
            render_preliminary_analysis_step(active_scenario)
        elif app_status == 'file_loaded':
            render_proposal_step(active_scenario)
        elif app_status == 'proposal_selected':
            render_validation_step(active_scenario)
        elif app_status in ["validated", "results_ready"]:
            # Primero, renderizamos el dashboard de análisis DEA
            render_main_dashboard(active_scenario, cached_run_dea_analysis, validate_data)
            
            # Si ya hay resultados, mostramos el taller de deliberación
            if active_scenario.get('dea_results'):
                render_dynamic_inquiry_workshop(
                    active_scenario,
                    st.session_state.inquiry_engine # Pasamos la instancia del motor
                )
                # ... (Aquí irían las llamadas a render_optimization_workshop y render_download_section)

    with comparison_tab:
        render_comparison_view(st.session_state.scenarios, get_active_scenario)

    with challenges_tab:
        render_dea_challenges_tab()


if __name__ == "__main__":
    main()
