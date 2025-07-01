# /src/ui_components.py
# --- VERSIÓN COMPLETA Y REFACTORIZADA ---

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

# --- NUEVAS IMPORTACIONES ---
# Se importan las clases y funciones necesarias del motor de indagación refactorizado.
from inquiry_engine import InquiryEngine, InquiryNode, to_plotly_tree
# Se importan las funciones del tracker que estarán definidas en main.py
from main import log_epistemic_event

# ---
# --- FUNCIONES DE RENDERIZADO (Sección sin cambios) ---
# ---
# Las siguientes funciones se mantienen tal como estaban en tu versión original.
# Las he colapsado aquí por brevedad, pero deben estar completas en tu archivo.

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
        st.rerun()
    if st.sidebar.button("📋 Clonar Escenario Actual"):
        st.session_state._clone_scenario_requested = st.session_state.active_scenario_id
        st.rerun()
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

def render_comparison_view(scenarios_dict, get_active_scenario_func, compute_eee_func):
    st.header("Comparador de Escenarios Metodológicos", divider="blue")
    st.info("Selecciona dos escenarios para comparar sus resultados, configuraciones y calidad del razonamiento (EEE).")
    if len(scenarios_dict) < 2:
        st.warning("Necesitas al menos dos escenarios para comparar.")
        return
    col1, col2 = st.columns(2)
    scenario_names = {sid: s['name'] for sid, s in scenarios_dict.items()}
    with col1:
        id_a = st.selectbox("Escenario A:", list(scenarios_dict.keys()), format_func=lambda sid: scenario_names[sid], key='compare_a')
    with col2:
        options_b = [sid for sid in scenarios_dict.keys() if sid != id_a] or [id_a]
        id_b = st.selectbox("Escenario B:", options_b, format_func=lambda sid: scenario_names[sid], key='compare_b')
    st.divider()
    scenario_a = scenarios_dict.get(id_a)
    scenario_b = scenarios_dict.get(id_b)
    if not scenario_a or not scenario_b:
        st.error("No se pudieron cargar los escenarios.")
        return
    res_col1, res_col2 = st.columns(2)
    for sc, col in [(scenario_a, res_col1), (scenario_b, res_col2)]:
        with col:
            st.subheader(f"Resultados de: {sc['name']}")
            with st.container(border=True):
                if sc.get('dea_results'):
                    st.markdown("**Configuración:**")
                    st.json(sc.get('dea_config', {}), expanded=False)
                    st.markdown(f"**Inputs:** {sc['selected_proposal'].get('inputs')}")
                    st.markdown(f"**Outputs:** {sc['selected_proposal'].get('outputs')}")
                    st.markdown("**Resultados (Top 5):**")
                    st.dataframe(sc['dea_results']['main_df'].head())
                    # Cambio clave: ahora se usa 'inquiry_tree_node'
                    if sc.get('inquiry_tree_node'):
                        # La función compute_eee ahora recibirá el objeto InquiryNode
                        eee_metrics = compute_eee_func(sc['inquiry_tree_node'])
                        st.metric("Calidad del Juicio (EEE)", f"{eee_metrics['score']:.2%}")
                else:
                    st.info("Escenario no calculado.")

def render_eee_explanation(eee_metrics: dict):
    st.info(f"**Calidad del Juicio Metodológico (EEE): {eee_metrics['score']:.2%}**")
    with st.expander("Ver desglose y consejos para mejorar"):
        # ... (código original sin cambios)
        pass

# ... (El resto de funciones como render_optimization_workshop, render_download_section, etc., se mantienen sin cambios)


# ---
# --- SECCIÓN DEL TALLER DE AUDITORÍA (COMPLETAMENTE REFACTORIZADA) ---
# ---

def render_dynamic_inquiry_workshop(active_scenario: dict, inquiry_engine: InquiryEngine, compute_eee_func):
    """
    Función orquestadora que gestiona la generación y visualización del taller de auditoría dinámico.
    Reemplaza a las antiguas `render_deliberation_workshop` y `render_interactive_inquiry_tree`.
    """
    st.header("Paso 4: Taller de Deliberación y Auditoría Metodológica", divider="blue")
    st.info("""
    Esta etapa es crucial para abordar los **retos metodológicos y de interpretación** del DEA.
    Utiliza el mapa de auditoría generado por la IA para documentar tu razonamiento.
    **Puedes expandir cualquier pregunta** para profundizar en un tema específico.
    """)

    # Clave de estado actualizada para el árbol de nodos
    tree_node_key = 'inquiry_tree_node'

    # --- Botón para generar el árbol de auditoría inicial ---
    if tree_node_key not in active_scenario or active_scenario[tree_node_key] is None:
        if st.button("Generar Mapa de Auditoría con IA", use_container_width=True, type="primary"):
            with st.spinner("La IA está diseñando el árbol de auditoría inicial..."):
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
        # Si el árbol ya existe, lo obtenemos y lo renderizamos
        tree_node = active_scenario[tree_node_key]

        with st.container(border=True):
            # --- Visualización y métricas EEE ---
            st.subheader("Análisis del Razonamiento", anchor=False)
            col_metrics, col_viz = st.columns(2)
            with col_metrics:
                # La función de cálculo de EEE ahora debe ser compatible con InquiryNode
                eee_metrics = compute_eee_func(tree_node)
                render_eee_explanation(eee_metrics)

            with col_viz:
                with st.expander("Ver visualización del árbol (Treemap)"):
                    fig = to_plotly_tree(tree_node)
                    st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Taller de Auditoría Interactivo")

        # --- Llamada a la nueva función de renderizado recursivo ---
        render_inquiry_node_recursively(tree_node, inquiry_engine)

def render_inquiry_node_recursively(node: InquiryNode, inquiry_engine: InquiryEngine):
    """
    NUEVA FUNCIÓN RECURSIVA: Muestra un nodo, sus controles y a sus hijos.
    """
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
                            log_epistemic_event(
                                "node_expansion",
                                {"node_id": node.id, "question": node.question}
                            )
                            st.rerun()

        justification_input = st.text_area(
            label=node.justification_prompt,
            value=node.justification,
            key=f"justify_{node.id}",
            height=120,
            placeholder="Escribe aquí tu razonamiento. Sé lo más detallado posible."
        )

        if justification_input != node.justification:
            node.justification = justification_input
            log_epistemic_event(
                "user_justification",
                {"node_id": node.id, "justification": justification_input}
            )

    # Llamada recursiva para los hijos, con indentación visual
    with st.container():
        st.markdown(f"<div style='margin-left: 30px;'>", unsafe_allow_html=True)
        for child in node.children:
            render_inquiry_node_recursively(child, inquiry_engine)
        st.markdown(f"</div>", unsafe_allow_html=True)
