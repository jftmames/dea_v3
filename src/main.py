# /src/main.py
# --- VERSIÓN FINAL CON FLUJO DE ESTADOS CORREGIDO ---

import sys
import os
import pandas as pd
import streamlit as st
import io

# --- 1) CONFIGURACIÓN INICIAL ---
# (sin cambios)
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
st.set_page_config(layout="wide", page_title="DEA Deliberative Modeler")

# --- 2) IMPORTACIONES DE MÓDULOS ---
from analysis_dispatcher import execute_analysis
from epistemic_metrics import compute_eee
from data_validator import validate as validate_data
from openai_helpers import explain_inquiry_tree
from session_manager import *
from ui_components import *

# --- 3) FUNCIONES DE CACHÉ ---
# (sin cambios)
@st.cache_data
def cached_run_dea_analysis(df, dmu, i, o, m, p):
    return execute_analysis(df, dmu, i, o, m, period_column=p)

# --- 4) FUNCIÓN PRINCIPAL DE LA APLICACIÓN ---
def main():
    initialize_global_state()

    st.sidebar.image("https://i.imgur.com/8y0N5c5.png", width=200)
    st.sidebar.title("DEA Deliberative Modeler")
    if st.sidebar.button("🔴 Empezar Nueva Sesión"):
        reset_all()
        st.rerun()
    st.sidebar.divider()

    active_scenario = get_active_scenario()
    render_scenario_navigator(active_scenario)

    if st.session_state.pop('_new_scenario_requested', False):
        create_new_scenario(name=f"Nuevo Modelo {len(st.session_state.scenarios) + 1}")
        st.rerun()
    if clone_id := st.session_state.pop('_clone_scenario_requested', None):
        create_new_scenario(source_scenario_id=clone_id)
        st.rerun()

    if not active_scenario:
        uploaded_file = render_upload_step()
        if uploaded_file:
            df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')))
            st.session_state.global_df = df
            create_new_scenario()
            st.rerun()
        return

    analysis_tab, comparison_tab, challenges_tab = st.tabs([
        "Análisis del Escenario Activo", "Comparar Escenarios", "Retos del DEA"
    ])

    with analysis_tab:
        app_status = active_scenario.get('app_status', 'data_loaded')
        
        # --- LÓGICA DE FLUJO DE ESTADOS MEJORADA ---
        if app_status == 'data_loaded':
            render_preliminary_analysis_step(active_scenario)
        elif app_status == 'proposal_selection':
            render_proposal_step(active_scenario)
        elif app_status == 'validation':
            render_validation_step(active_scenario, validate_data)
        elif app_status == 'analysis_setup':
            render_main_dashboard(active_scenario, cached_run_dea_analysis)
        elif app_status == 'deliberation':
            render_deliberation_workshop(active_scenario, compute_eee)

    with comparison_tab:
        render_comparison_view(st.session_state.scenarios, compute_eee)

    with challenges_tab:
        render_dea_challenges_tab()

if __name__ == "__main__":
    main()
