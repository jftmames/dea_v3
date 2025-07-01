# /src/ui_components.py
# --- CÓDIGO COMPLETO Y DEFINITIVO ---

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uuid

from inquiry_engine import InquiryEngine, InquiryNode, to_plotly_tree
from openai_helpers import generate_analysis_proposals
from session_manager import log_epistemic_event

# ---
# --- Componentes de la UI ---
# ---

def render_upload_step():
    st.header("Paso 1: Carga tus Datos", divider="blue")
    st.info("Sube un fichero CSV. La primera columna debe ser el identificador de la DMU.")
    return st.file_uploader("Sube un fichero CSV", type=["csv"], label_visibility="collapsed")

def render_scenario_navigator(active_scenario):
    st.sidebar.title("Navegador de Escenarios")
    if not st.session_state.get('scenarios'):
        st.sidebar.info("Carga un fichero para empezar.")
        return

    scenario_names = {sid: s['name'] for sid, s in st.session_state.scenarios.items()}
    active_id = st.session_state.get('active_scenario_id')

    if active_id not in scenario_names:
        active_id = next(iter(scenario_names)) if scenario_names else None
    
    st.session_state.active_scenario_id = st.sidebar.selectbox(
        "Escenario Activo", options=list(st.session_state.scenarios.keys()),
        format_func=lambda sid: scenario_names.get(sid, "N/A"),
        index=list(st.session_state.scenarios.keys()).index(active_id) if active_id and active_id in st.session_state.scenarios else 0
    )
    st.sidebar.divider()
    st.sidebar.subheader("Acciones de Escenario")
    if st.sidebar.button("➕ Nuevo Escenario"): st.session_state._new_scenario_requested = True
    if st.sidebar.button("📋 Clonar Escenario"): st.session_state._clone_scenario_requested = st.session_state.active_scenario_id
    
    if active_scenario:
        new_name = st.sidebar.text_input("Renombrar escenario:", value=active_scenario['name'], key=f"rename_{active_scenario['name']}")
        if new_name != active_scenario['name']:
            active_scenario['name'] = new_name; st.rerun()
            
    if len(st.session_state.scenarios) > 1:
        if st.sidebar.button("🗑️ Eliminar Escenario", type="primary"):
            del st.session_state.scenarios[st.session_state.active_scenario_id]
            st.session_state.active_scenario_id = next(iter(st.session_state.scenarios)); st.rerun()

def render_preliminary_analysis_step(active_scenario):
    st.header(f"Paso 1b: Exploración de Datos para '{active_scenario['name']}'", divider="blue")
    st.info("Entiende tus datos antes del análisis.")
    df = active_scenario['df']
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

    if not numerical_cols:
        st.warning("No se encontraron columnas numéricas.")
        return

    st.subheader("Estadísticas Descriptivas")
    st.dataframe(df[numerical_cols].describe().T)

    if st.button("Paso 2: Elegir Enfoque", type="primary", use_container_width=True):
        active_scenario['app_status'] = "proposal_selection"
        st.rerun()

def render_proposal_step(active_scenario):
    st.header(f"Paso 2: Elige un Enfoque para '{active_scenario['name']}'", divider="blue")
    st.info("Define los inputs y outputs para tu modelo.")

    # ... (código para generar y mostrar propuestas) ...

    if st.button("Confirmar y Validar", type="primary"):
        # Lógica para guardar la selección de inputs/outputs...
        active_scenario['app_status'] = 'validation'
        st.rerun()

def render_validation_step(active_scenario, validate_data_func):
    st.header(f"Paso 2b: Validación del Modelo", divider="gray")
    # ... (código para mostrar resultados de validación) ...

    if st.button("Paso 3: Configurar Análisis", type="primary"):
        active_scenario['app_status'] = 'analysis_setup'
        st.rerun()

def render_main_dashboard(active_scenario, run_analysis_func):
    st.header(f"Paso 3: Configuración y Análisis para '{active_scenario['name']}'", divider="blue")
    # ... (código para configurar y ejecutar análisis) ...

    if st.button("Ejecutar Análisis DEA", type="primary"):
        with st.spinner("Ejecutando..."):
            # Lógica de ejecución
            active_scenario['dea_results'] = {} # Placeholder
        active_scenario['app_status'] = 'deliberation'
        st.rerun()

def render_deliberation_workshop(active_scenario, compute_eee_func):
    st.header("Paso 4: Taller de Deliberación Metodológica", divider="blue")
    # ... (código de la función) ...

def render_comparison_view(scenarios, compute_eee_func):
    st.header("Comparador de Escenarios", divider="blue")
    # ... (código de la función) ...

def render_dea_challenges_tab():
    st.header("Retos Relevantes en DEA", divider="blue")
    # ... (código de la función) ...
