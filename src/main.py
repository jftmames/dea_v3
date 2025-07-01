# /src/main.py
# --- VERSIÓN COMPLETA, CORREGIDA Y FINAL ---

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
from report_generator import generate_html_report, generate_excel_report
from dea_models.auto_tuner import generate_candidates, evaluate_candidates
from openai_helpers import explain_inquiry_tree

# ¡NUEVO! Se importa todo desde el session_manager
from session_manager import *

# Componentes de la UI
from ui_components import *

# --- 4) FUNCIONES DE CACHÉ ---
@st.cache_data
def cached_run_dea_analysis(_df, dmu_col, inputs, outputs, model, period):
    return execute_analysis(_df.copy(), dmu_col, inputs, outputs, model, period_column=period)

@st.cache_data
def cached_explain_tree(_tree_node):
    return explain_inquiry_tree(_tree_node)
    
# ... (Otras funciones de caché que puedas tener)

# --- 5) FUNCIÓN PRINCIPAL DE LA APLICACIÓN ---
def main():
    initialize_global_state()

    st.sidebar.image("https://i.imgur.com/8y0N5c5.png", width=200)
    st.sidebar.title("DEA Deliberative Modeler")
    if st.sidebar.button("🔴 Empezar Nueva Sesión"):
        reset_all()
        st.rerun()
    st.sidebar.divider()

    render_scenario_navigator()

    # Manejar las solicitudes de la UI para crear/clonar escenarios
    if st.session_state.pop('_new_scenario_requested', False):
        create_new_scenario(name=f"Nuevo Modelo {len(st.session_state.scenarios) + 1}")
        st.rerun()
    if clone_id := st.session_state.pop('_clone_scenario_requested', None):
        create_new_scenario(source_scenario_id=clone_id)
        st.rerun()

    active_scenario = get_active_scenario()

    if not active_scenario:
        st.header("Paso 1: Carga tus Datos", divider="blue")
        uploaded_file = st.file_uploader("Sube un fichero CSV", type=["csv"], label_visibility="collapsed")
        if uploaded_file:
            try:
                df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')))
            except:
                df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('latin-1')), sep=';')
            
            st.session_state.global_df = df
            create_new_scenario()
            st.rerun()
        return

    # Lógica de pestañas
    analysis_tab, comparison_tab, challenges_tab = st.tabs([
        "Análisis del Escenario Activo", "Comparar Escenarios", "Retos del DEA"
    ])

    with analysis_tab:
        app_status = active_scenario.get('app_status', 'data_loaded')
        
        if app_status == 'data_loaded':
            render_preliminary_analysis_step(active_scenario)
        elif app_status == 'file_loaded':
            render_proposal_step(active_scenario)
        elif app_status == 'proposal_selected':
            render_validation_step(active_scenario)
        elif app_status in ["validated", "results_ready"]:
            # Las funciones de renderizado ahora no necesitan tantos argumentos,
            # ya que pueden acceder al estado a través del session_manager.
            render_main_dashboard(active_scenario, cached_run_dea_analysis, validate_data)
            
            if active_scenario.get('dea_results'):
                render_dynamic_inquiry_workshop(active_scenario, compute_eee)
                # render_optimization_workshop(...)
                # render_download_section(...)

    with comparison_tab:
        render_comparison_view(st.session_state.scenarios, compute_eee)

    with challenges_tab:
        render_dea_challenges_tab()

if __name__ == "__main__":
    main()
