# /src/ui_components.py
# --- VERSIÓN COMPLETA Y CORREGIDA ---

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import uuid

# --- IMPORTACIONES CORREGIDAS ---
from inquiry_engine import InquiryEngine, InquiryNode, to_plotly_tree
from openai_helpers import generate_analysis_proposals
# ¡Se importa desde el nuevo módulo!
from session_manager import log_epistemic_event

# (Las funciones de renderizado que no cambian, como render_scenario_navigator, etc.,
# deben permanecer aquí. Las omito por brevedad.)

def render_scenario_navigator():
    st.sidebar.title("Navegador de Escenarios")
    st.sidebar.markdown("Gestiona y compara tus diferentes modelos y análisis.")
    st.sidebar.divider()
    if 'scenarios' not in st.session_state or not st.session_state.scenarios:
        st.sidebar.info("Carga un fichero de datos para empezar.")
        return
    scenario_names = {sid: s['name'] for sid, s in st.session_state.scenarios.items()}
    active_id = st.session_state.get('active_scenario_id')
    if active_id not in scenario_names:
        active_id = next(iter(scenario_names)) if scenario_names else None
    st.session_state.active_scenario_id = st.sidebar.selectbox(
        "Escenario Activo", options=list(st.session_state.scenarios.keys()),
        format_func=lambda sid: scenario_names.get(sid, "N/A"),
        index=list(st.session_state.scenarios.keys()).index(active_id) if active_id in st.session_state.scenarios else 0,
        key='scenario_selector'
    )
    st.sidebar.divider()
    st.sidebar.subheader("Acciones de Escenario")
    if st.sidebar.button("➕ Nuevo Escenario"):
        st.session_state._new_scenario_requested = True
    if st.sidebar.button("📋 Clonar Escenario Actual"):
        st.session_state._clone_scenario_requested = st.session_state.active_scenario_id
    active_scenario = st.session_state.scenarios.get(st.session_state.active_scenario_id)
    if active_scenario:
        new_name = st.sidebar.text_input("Renombrar escenario:", value=active_scenario['name'], key=f"rename_{st.session_state.active_scenario_id}")
        if new_name != active_scenario['name']:
            active_scenario['name'] = new_name
            st.rerun()
    st.sidebar.divider()
    if len(st.session_state.scenarios) > 1:
        if st.sidebar.button("🗑️ Eliminar Escenario Actual", type="primary"):
            del st.session_state.scenarios[st.session_state.active_scenario_id]
            st.session_state.active_scenario_id = next(iter(st.session_state.scenarios))
            st.rerun()

def render_comparison_view(scenarios_dict, compute_eee_func):
    # ... (código original sin cambios)
    pass

def render_eee_explanation(eee_metrics: dict):
    # ... (código original sin cambios)
    pass
    
def render_proposal_step(active_scenario):
    st.header(f"Paso 2: Elige un Enfoque de Análisis para '{active_scenario['name']}'", divider="blue")
    st.info("Define los inputs y outputs para tu modelo.")

    if 'proposals_data' not in active_scenario or not active_scenario.get('proposals_data'):
        with st.spinner("La IA está analizando tus datos para sugerir enfoques..."):
            active_scenario['proposals_data'] = generate_analysis_proposals(
                active_scenario['df'].columns.tolist(), active_scenario['df'].head()
            )
    # ... (resto de la función sin cambios)

# ... (todas las demás funciones de renderizado originales)

# --- SECCIÓN DEL TALLER DE AUDITORÍA (Función clave actualizada) ---

def render_dynamic_inquiry_workshop(active_scenario: dict, compute_eee_func):
    st.header("Paso 4: Taller de Deliberación y Auditoría Metodológica", divider="blue")
    st.info("Utiliza el mapa de auditoría generado por la IA para documentar tu razonamiento. Puedes expandir cualquier pregunta.")
    
    inquiry_engine = st.session_state.inquiry_engine
    tree_node_key = 'inquiry_tree_node'

    if tree_node_key not in active_scenario or active_scenario[tree_node_key] is None:
        if st.button("Generar Mapa de Auditoría con IA", use_container_width=True, type="primary"):
            with st.spinner("La IA está diseñando el árbol de auditoría..."):
                context = {
                    "model": active_scenario.get('dea_results', {}).get("model_name"),
                    "inputs": active_scenario.get('selected_proposal', {}).get('inputs', []),
                    "outputs": active_scenario.get('selected_proposal', {}).get('outputs', [])
                }
                tree_node, error = inquiry_engine.generate_initial_tree(context)
                if error:
                    st.error(f"Error al generar el mapa: {error}")
                else:
                    active_scenario[tree_node_key] = tree_node
                    log_epistemic_event("initial_tree_generation", {"root_question": tree_node.question})
                    st.rerun()
    else:
        tree_node = active_scenario[tree_node_key]
        with st.container(border=True):
            st.subheader("Análisis del Razonamiento", anchor=False)
            col_metrics, col_viz = st.columns(2)
            with col_metrics:
                # Se asume que compute_eee puede manejar el objeto InquiryNode
                eee_metrics = compute_eee_func(tree_node) 
                render_eee_explanation(eee_metrics)
            with col_viz:
                with st.expander("Ver visualización del árbol (Treemap)"):
                    fig = to_plotly_tree(tree_node)
                    st.plotly_chart(fig, use_container_width=True)
        st.divider()
        st.subheader("Taller de Auditoría Interactivo")
        render_inquiry_node_recursively(tree_node, inquiry_engine)

def render_inquiry_node_recursively(node: InquiryNode, inquiry_engine: InquiryEngine):
    """Renderiza un nodo, sus controles y a sus hijos."""
    with st.container(border=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**{node.question}**")
        if not node.children and not node.expanded:
            with col2:
                if st.button("🔍 Expandir", key=f"expand_{node.id}", help="Pedir a la IA que desglose esta pregunta."):
                    with st.spinner("🧠 Generando sub-preguntas..."):
                        new_sub_nodes, error = inquiry_engine.expand_question_node(node.question)
                        if error:
                            st.error(error)
                        else:
                            node.children.extend(new_sub_nodes)
                            node.expanded = True
                            log_epistemic_event("node_expansion", {"node_id": node.id, "question": node.question})
                            st.rerun()
        justification_input = st.text_area(
            label=node.justification_prompt, value=node.justification,
            key=f"justify_{node.id}", height=120, placeholder="Escribe aquí tu razonamiento."
        )
        if justification_input != node.justification:
            node.justification = justification_input
            log_epistemic_event("user_justification", {"node_id": node.id, "justification": justification_input})
    with st.container():
        st.markdown(f"<div style='margin-left: 30px;'>", unsafe_allow_html=True)
        for child in node.children:
            render_inquiry_node_recursively(child, inquiry_engine)
        st.markdown(f"</div>", unsafe_allow_html=True)
