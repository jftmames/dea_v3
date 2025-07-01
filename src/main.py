# /src/main.py
# --- VERSIÓN COMPLETA, FINAL Y ORQUESTADORA ---

import sys
import os
import pandas as pd
import streamlit as st
import io

# --- 1) CONFIGURACIÓN INICIAL ---
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
st.set_page_config(layout="wide", page_title="DEA Deliberative Modeler")

# --- 2) IMPORTACIONES DE MÓDULOS ---
# Lógica de negocio
from analysis_dispatcher import execute_analysis
from epistemic_metrics import compute_eee
from data_validator import validate as validate_data
from openai_helpers import explain_inquiry_tree

# Módulo de gestión de estado
from session_manager import *

# Módulo de componentes de UI
from ui_components import *

# --- 3) FUNCIONES DE CACHÉ ---
@st.cache_data
def cached_run_dea_analysis(_df, dmu_col, inputs, outputs, model, period):
    return execute_analysis(_df.copy(), dmu_col, inputs, outputs, model, period_column=period)

@st.cache_data
def cached_explain_tree(_tree_node):
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

    # Si no hay ningún escenario activo (al inicio o tras borrar todos),
    # se muestra la pantalla de carga.
    if not active_scenario:
        uploaded_file = render_upload_step()
        if uploaded_file:
            try:
                df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')))
            except Exception:
                df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('latin-1')), sep=';')
            
            st.session_state.global_df = df
            create_new_scenario()
            st.rerun()
        return

    # Lógica de Pestañas
    analysis_tab, comparison_tab, challenges_tab = st.tabs([
        "Análisis del Escenario Activo", "Comparar Escenarios", "Retos del DEA"
    ])

    with analysis_tab:
        app_status = active_scenario.get('app_status', 'data_loaded')
        
        # Flujo de la aplicación basado en estados
        if app_status == 'data_loaded':
            render_preliminary_analysis_step(active_scenario)
        elif app_status == 'proposal_selection':
            # La llamada a esta función ahora debería funcionar correctamente.
            render_proposal_step(active_scenario)
        elif app_status == 'validation':
            # Asumiendo que render_validation_step existe en ui_components.py
            render_validation_step(active_scenario, validate_data)
        elif app_status == 'analysis':
            # Asumiendo que render_main_dashboard existe en ui_components.py
            render_main_dashboard(active_scenario, cached_run_dea_analysis, validate_data)
        elif app_status == 'deliberation':
            # Asumiendo que render_deliberation_workshop existe en ui_components.py
            render_deliberation_workshop(active_scenario, compute_eee)
    
    with comparison_tab:
        # Asumiendo que render_comparison_view existe en ui_components.py
        render_comparison_view(st.session_state.scenarios, compute_eee)

    with challenges_tab:
        render_dea_challenges_tab()

if __name__ == "__main__":
    main()
