# /src/main.py
# --- VERSIÓN FINAL CON FLUJO DE ESTADOS CORREGIDO ---

import sys
import os
import pandas as pd
import streamlit as st
import io

# --- 1) CONFIGURACIÓN INICIAL Y PYTHONPATH ---
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
st.set_page_config(layout="wide", page_title="DEA Deliberative Modeler")

# --- 2) IMPORTACIONES DE MÓDULOS ---
# Lógica de negocio y análisis
from analysis_dispatcher import execute_analysis
from epistemic_metrics import compute_eee
from data_validator import validate as validate_data
from openai_helpers import explain_inquiry_tree

# Módulo central para la gestión del estado de la sesión
from session_manager import *

# Módulo que contiene todos los componentes de la interfaz de usuario
from ui_components import (
    render_upload_step,
    render_scenario_navigator,
    render_preliminary_analysis_step,
    render_proposal_step,
    render_validation_step,
    render_main_dashboard,
    render_deliberation_workshop,
    render_comparison_view,
    render_dea_challenges_tab
)

# --- 3) FUNCIONES DE CACHÉ ---
@st.cache_data
def cached_run_dea_analysis(_df, dmu_col, inputs, outputs, model, period):
    """Ejecuta el análisis DEA y cachea el resultado para evitar re-cálculos."""
    return execute_analysis(_df.copy(), dmu_col, inputs, outputs, model, period_column=period)

@st.cache_data
def cached_explain_tree(_tree_node):
    """Genera una explicación para el árbol de auditoría y la cachea."""
    # Esta función debería ser adaptada si la entrada de explain_inquiry_tree cambia
    return explain_inquiry_tree(_tree_node)

# --- 4) FUNCIÓN PRINCIPAL DE LA APLICACIÓN ---
def main():
    """Función principal que orquesta la aplicación multi-escenario."""
    initialize_global_state()

    st.sidebar.image("https://i.imgur.com/8y0N5c5.png", width=200)
    st.sidebar.title("DEA Deliberative Modeler")
    if st.sidebar.button("🔴 Empezar Nueva Sesión"):
        reset_all()
        st.rerun()
    st.sidebar.divider()

    active_scenario = get_active_scenario()
    render_scenario_navigator(active_scenario)

    # Manejar las solicitudes de la UI para crear/clonar escenarios
    if st.session_state.pop('_new_scenario_requested', False):
        create_new_scenario(name=f"Nuevo Modelo {len(st.session_state.scenarios) + 1}")
        st.rerun()
    if clone_id := st.session_state.pop('_clone_scenario_requested', None):
        create_new_scenario(source_scenario_id=clone_id)
        st.rerun()

    # Si no hay ningún escenario activo (al inicio), muestra la pantalla de carga de archivos.
    if not active_scenario:
        uploaded_file = render_upload_step()
        if uploaded_file:
            try:
                # Intenta leer como UTF-8, que es el estándar más común
                df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')))
            except Exception:
                # Si falla, intenta con latin-1, común en algunos sistemas europeos
                df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('latin-1')), sep=';')
            
            st.session_state.global_df = df
            create_new_scenario()
            st.rerun()
        return

    # Pestañas principales de la aplicación
    analysis_tab, comparison_tab, challenges_tab = st.tabs([
        "Análisis del Escenario Activo", "Comparar Escenarios", "Retos del DEA"
    ])

    with analysis_tab:
        # Recupera el estado actual del escenario para decidir qué mostrar
        app_status = active_scenario.get('app_status', 'data_loaded')
        
        # --- MÁQUINA DE ESTADOS PARA NAVEGAR ENTRE PASOS ---
        # Este es el núcleo que garantiza la fluidez de la aplicación.
        if app_status == 'data_loaded':
            render_preliminary_analysis_step(active_scenario)
        elif app_status == 'proposal_selection':
            render_proposal_step(active_scenario)
        elif app_status == 'validation':
            render_validation_step(active_scenario, validate_data)
        elif app_status == 'analysis_setup':
            render_main_dashboard(active_scenario, cached_run_dea_analysis)
        elif app_status == 'deliberation':
            if active_scenario.get('dea_results'):
                render_deliberation_workshop(active_scenario, compute_eee)
            else:
                st.warning("Primero debes ejecutar un análisis en el Paso 3 para poder deliberar.")
                # Vuelve a mostrar el dashboard de análisis para que el usuario pueda ejecutarlo
                render_main_dashboard(active_scenario, cached_run_dea_analysis)

    with comparison_tab:
        render_comparison_view(st.session_state.scenarios, compute_eee)

    with challenges_tab:
        render_dea_challenges_tab()

if __name__ == "__main__":
    main()
