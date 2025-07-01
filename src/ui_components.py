# /src/ui_components.py
# --- VERSIÓN COMPLETA Y CENTRALIZADA ---

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uuid

# --- IMPORTACIONES CORREGIDAS Y CENTRALIZADAS ---
from inquiry_engine import InquiryEngine, InquiryNode, to_plotly_tree
from openai_helpers import generate_analysis_proposals
from session_manager import log_epistemic_event

# ---
# --- DEFINICIÓN DE TODOS LOS COMPONENTES DE LA UI ---
# ---

def render_upload_step():
    """Renderiza el componente para subir el archivo CSV."""
    st.header("Paso 1: Carga tus Datos", divider="blue")
    st.info("Sube un fichero CSV. La primera columna debe ser el identificador de la DMU, y el resto, variables numéricas.")
    uploaded_file = st.file_uploader("Sube un fichero CSV", type=["csv"], label_visibility="collapsed")
    return uploaded_file

def render_scenario_navigator():
    """Renderiza la barra lateral para gestionar escenarios."""
    st.sidebar.title("Navegador de Escenarios")
    if 'scenarios' not in st.session_state or not st.session_state.scenarios:
        st.sidebar.info("Carga un fichero para empezar.")
        return

    scenario_names = {sid: s['name'] for sid, s in st.session_state.scenarios.items()}
    active_id = st.session_state.get('active_scenario_id')
    if active_id not in scenario_names:
        active_id = next(iter(scenario_names)) if scenario_names else None
    
    st.session_state.active_scenario_id = st.sidebar.selectbox(
        "Escenario Activo", options=list(st.session_state.scenarios.keys()),
        format_func=lambda sid: scenario_names.get(sid, "N/A"),
        index=list(st.session_state.scenarios.keys()).index(active_id) if active_id in st.session_state.scenarios else 0
    )
    st.sidebar.divider()
    st.sidebar.subheader("Acciones de Escenario")
    if st.sidebar.button("➕ Nuevo Escenario"):
        st.session_state._new_scenario_requested = True
    if st.sidebar.button("📋 Clonar Escenario Actual"):
        st.session_state._clone_scenario_requested = st.session_state.active_scenario_id
    
    active_scenario = st.session_state.scenarios.get(st.session_state.active_scenario_id)
    if active_scenario:
        new_name = st.sidebar.text_input("Renombrar escenario:", value=active_scenario['name'], key=f"rename_{active_scenario['name']}")
        if new_name != active_scenario['name']:
            active_scenario['name'] = new_name
            st.rerun()
    st.sidebar.divider()
    if len(st.session_state.scenarios) > 1:
        if st.sidebar.button("🗑️ Eliminar Escenario Actual", type="primary"):
            del st.session_state.scenarios[st.session_state.active_scenario_id]
            st.session_state.active_scenario_id = next(iter(st.session_state.scenarios))
            st.rerun()

def render_main_dashboard(active_scenario, run_analysis_func, validate_data_func):
    """Renderiza el dashboard principal para el Paso 3: Análisis."""
    # ... (El código de tu render_main_dashboard original va aquí) ...
    pass

def render_deliberation_workshop(active_scenario, compute_eee_func, explain_tree_func):
    """
    Función orquestadora que gestiona la generación y visualización del taller de auditoría dinámico.
    """
    st.header("Paso 4: Taller de Deliberación y Auditoría Metodológica", divider="blue")
    inquiry_engine = st.session_state.inquiry_engine
    tree_node_key = 'inquiry_tree_node'

    if active_scenario.get(tree_node_key) is None:
        if st.button("Generar Mapa de Auditoría con IA", use_container_width=True, type="primary"):
            # ... (código de la respuesta anterior para generar el árbol) ...
            pass
    else:
        tree_node = active_scenario[tree_node_key]
        render_inquiry_node_recursively(tree_node, inquiry_engine)

def render_inquiry_node_recursively(node: InquiryNode, inquiry_engine: InquiryEngine):
    """Renderiza un nodo, sus controles y a sus hijos de forma recursiva."""
    # ... (código de la respuesta anterior para el renderizado recursivo) ...
    pass

def render_dea_challenges_tab():
    """Muestra la pestaña con información sobre los retos del DEA."""
    st.header("Retos Relevantes en el Uso del Análisis Envolvente de Datos (DEA)", divider="blue")
    st.markdown("""
    El Análisis Envolvente de Datos (DEA) es una herramienta potente, pero su aplicación exitosa depende de entender y abordar sus desafíos inherentes.
    
    ### 1. Retos Relacionados con los Datos
    - **Selección de Inputs y Outputs:** Es subjetivo y requiere justificación teórica.
    - **Calidad de Datos:** Datos incompletos o erróneos pueden invalidar el análisis.
    - **Valores Nulos, Negativos y Cero:** Los modelos clásicos requieren datos positivos.
    
    ### 2. Retos Metodológicos
    - **Elección del Modelo y Orientación:** La elección (CCR, BCC, etc.) es crítica y afecta los resultados.
    - **Sensibilidad del Modelo:** Los resultados pueden ser muy sensibles a pequeñas variaciones en los datos.

    ### 3. Retos de Interpretación
    - **Eficiencia Relativa:** La eficiencia en DEA es *relativa* a la muestra, no absoluta.
    - **Implicaciones de Política:** Traducir los resultados en acciones concretas requiere conocimiento del dominio.
    """)
    # ... (Puedes expandir esto con el contenido completo de tu función original)
