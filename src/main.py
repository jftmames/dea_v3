import sys
import os
import pandas as pd
import streamlit as st
import io
import json
import uuid 
import openai 
import plotly.express as px # Importar plotly.express para gráficos exploratorios

# --- 0) AJUSTE DEL PYTHONPATH Y CONFIGURACIÓN INICIAL ---
# Asegura que los módulos locales se puedan importar correctamente.
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Configuración de la página de Streamlit. "wide" aprovecha mejor el espacio.
st.set_page_config(layout="wide", page_title="DEA Deliberative Modeler")

# --- 1) IMPORTACIONES DE MÓDULOS ---
# Importa todas las funciones necesarias de los otros archivos del proyecto.
from analysis_dispatcher import execute_analysis
from inquiry_engine import generate_inquiry, to_plotly_tree
from epistemic_metrics import compute_eee
from data_validator import validate as validate_data
from report_generator import generate_html_report, generate_excel_report
from dea_models.visualizations import plot_hypothesis_distribution, plot_correlation
from dea_models.auto_tuner import generate_candidates, evaluate_candidates 
from openai_helpers import explain_inquiry_tree

# --- 2) GESTIÓN DE ESTADO MULTI-ESCENARIO ---
# Este es el núcleo de la refactorización. Se pasa de un estado único
# a una gestión de múltiples escenarios.

def create_new_scenario(name: str = "Modelo Base", source_scenario_id: str = None):
    """Crea un nuevo escenario, ya sea en blanco o clonando uno existente."""
    new_id = str(uuid.uuid4()) 
    
    if source_scenario_id and source_scenario_id in st.session_state.scenarios:
        st.session_state.scenarios[new_id] = st.session_state.scenarios[source_scenario_id].copy()
        st.session_state.scenarios[new_id]['name'] = f"Copia de {st.session_state.scenarios[source_scenario_id]['name']}"
        if st.session_state.scenarios[new_id]['dea_results']:
            st.session_state.scenarios[new_id]['dea_results'] = st.session_state.scenarios[source_scenario_id]['dea_results'].copy()
        if st.session_state.scenarios[new_id]['inquiry_tree']:
            st.session_state.scenarios[new_id]['inquiry_tree'] = st.session_state.scenarios[source_scenario_id]['inquiry_tree'].copy()
        st.session_state.scenarios[new_id]['user_justifications'] = {} 
        st.session_state.scenarios[new_id]['app_status'] = "data_loaded" if st.session_state.get("global_df") is not None else "initial"
        st.session_state.scenarios[new_id]['dea_results'] = None 
        st.session_state.scenarios[new_id]['inquiry_tree'] = None 
    else:
        st.session_state.scenarios[new_id] = {
            "name": name,
            "df": st.session_state.get("global_df", None), 
            "app_status": "initial", 
            "proposals_data": None,
            "selected_proposal": None,
            "dea_config": {},
            "dea_results": None,
            "inquiry_tree": None,
            "tree_explanation": None,
            "chart_to_show": None,
            "user_justifications": {}, 
            "data_overview": {} 
        }
    st.session_state.active_scenario_id = new_id

def get_active_scenario():
    """Devuelve el diccionario del escenario actualmente activo."""
    active_id = st.session_state.get('active_scenario_id')
    if active_id and active_id in st.session_state.scenarios:
        return st.session_state.scenarios[active_id]
    return None

def initialize_global_state():
    """Inicializa el estado global de la app."""
    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = {}
        st.session_state.active_scenario_id = None
        st.session_state.global_df = None

def reset_all():
    """Reinicia la aplicación a su estado inicial, eliminando todos los datos y escenarios."""
    cached_get_analysis_proposals.clear()
    cached_run_dea_analysis.clear()
    cached_run_inquiry_engine.clear()
    cached_explain_tree.clear()
    cached_generate_candidates.clear()
    cached_evaluate_candidates.clear()

    st.session_state.clear() 

    pass 


# --- 3) FUNCIONES DE CACHÉ Y LÓGICA DE IA ---

@st.cache_data
def cached_get_analysis_proposals(_df):
    return generate_analysis_proposals(_df.columns.tolist(), _df.head())

@st.cache_data
def cached_run_dea_analysis(_df, dmu_col, input_cols, output_cols, model_key, period_col):
    return execute_analysis(_df.copy(), dmu_col, input_cols, output_cols, model_key, period_column=period_col)

@st.cache_data
def cached_run_inquiry_engine(root_question, _context):
    return generate_inquiry(root_question, context=_context)

@st.cache_data
def cached_explain_tree(_tree):
    return explain_inquiry_tree(_tree)

@st.cache_data
def cached_generate_candidates(_df, dmu_col, input_cols, output_cols, inquiry_tree, eee_score):
    return generate_candidates(_df, dmu_col, input_cols, output_cols, inquiry_tree, eee_score)

@st.cache_data
def cached_evaluate_candidates(_df, dmu_col, candidates, model):
    return evaluate_candidates(_df, dmu_col, candidates, model)


def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("La clave de API de OpenAI no ha sido configurada.")
        st.info("Añade tu clave 'OPENAI_API_KEY' en los 'Secrets' de la app y refresca la página.")
        return None 
    try:
        return openai.OpenAI(api_key=api_key) 
    except Exception as e:
        st.error(f"Error al inicializar el cliente de OpenAI: {e}")
        return None

def chat_completion(prompt: str, use_json_mode: bool = False):
    client = get_openai_client()
    if client is None: 
        return {"error": "API Key de OpenAI no configurada o error de inicialización.", "raw_content": "No se pudo conectar a OpenAI."}

    params = {"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}], "temperature": 0.5}
    if use_json_mode:
        params["response_format"] = {"type": "json_object"}
    
    try:
        return client.chat.completions.create(**params)
    except Exception as e:
        return {"error": f"Error al llamar a la API de OpenAI: {str(e)}", "raw_content": "Error en la llamada a la API."}


def generate_analysis_proposals(df_columns: list[str], df_head: pd.DataFrame):
    prompt = (
        "Eres un consultor experto en Data Envelopment Analysis (DEA). Has recibido un conjunto de datos con las siguientes columnas: "
        f"{df_columns}. A continuación se muestran las primeras filas:\n\n{df_head.to_string()}\n\n"
        "Tu tarea es proponer entre 2 y 4 modelos de análisis DEA distintos y bien fundamentados que se podrían aplicar a estos datos. "
        "Para cada propuesta, proporciona un título, un breve razonamiento sobre su utilidad y las listas de inputs y outputs sugeridas.\n\n"
        "Devuelve únicamente un objeto JSON válido con una sola clave raíz 'proposals'. El valor de 'proposals' debe ser una lista de objetos, donde cada objeto representa una propuesta y contiene las claves 'title', 'reasoning', 'inputs' y 'outputs'."
    )
    content = "No se recibió contenido."
    try:
        resp = chat_completion(prompt, use_json_mode=True)
        if isinstance(resp, dict) and resp.get("error"): 
            return {"error": f"Error al procesar la respuesta de la IA: {resp['error']}", "raw_content": resp['raw_content']}
        
        content = resp.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        return {"error": f"Error al procesar la respuesta de la IA: {str(e)}", "raw_content": content}

# --- FUNCIONES DE RENDERIZADO DE LA UI (Definidas antes de main()) ---

def render_scenario_navigator():
    st.sidebar.title("Navegador de Escenarios")
    st.sidebar.markdown("Gestiona y compara tus diferentes modelos y análisis. Cada escenario guarda su propia configuración y resultados.")
    st.sidebar.divider()

    if not st.session_state.scenarios:
        st.sidebar.info("Carga un fichero de datos para empezar a crear y gestionar escenarios de análisis DEA.")
        return

    scenario_names = {sid: s['name'] for sid, s in st.session_state.scenarios.items()}
    active_id = st.session_state.get('active_scenario_id')
    
    if active_id not in scenario_names:
        active_id = next(iter(scenario_names)) if scenario_names else None
    
    st.session_state.active_scenario_id = st.sidebar.selectbox(
        "Escenario Activo",
        options=list(st.session_state.scenarios.keys()),
        format_func=lambda sid: scenario_names.get(sid, "Escenario no válido"),
        index=list(st.session_state.scenarios.keys()).index(active_id) if active_id in st.session_state.scenarios else 0,
        key='scenario_selector'
    )

    st.sidebar.divider()
    st.sidebar.subheader("Acciones de Escenario")
    st.sidebar.markdown("Utiliza estas opciones para organizar y probar diferentes enfoques sin perder tu trabajo.")
    if st.sidebar.button("➕ Nuevo Escenario", help="Crea un nuevo modelo desde cero con los mismos datos cargados. Ideal para empezar un análisis completamente nuevo."):
        create_new_scenario(name=f"Nuevo Modelo {len(st.session_state.scenarios) + 1}")
        st.rerun()

    if st.sidebar.button("📋 Clonar Escenario Actual", help="Crea una copia exacta del escenario actualmente activo. Útil para probar pequeñas variaciones en la configuración o las variables sin afectar el original."):
        create_new_scenario(source_scenario_id=st.session_state.active_scenario_id)
        st.rerun()
    
    active_scenario = get_active_scenario()
    if active_scenario:
        new_name = st.sidebar.text_input("Renombrar escenario:", value=active_scenario['name'], key=f"rename_{st.session_state.active_scenario_id}")
        if new_name != active_scenario['name']:
            active_scenario['name'] = new_name
            st.rerun()

    st.sidebar.divider()
    if len(st.session_state.scenarios) > 1:
        if st.sidebar.button("🗑️ Eliminar Escenario Actual", type="primary", help="Borra el escenario activo de forma permanente. Ten precaución, esta acción no se puede deshacer."):
            del st.session_state.scenarios[st.session_state.active_scenario_id]
            st.session_state.active_scenario_id = next(iter(st.session_state.scenarios))
            st.rerun()

def render_comparison_view():
    st.header("Comparador de Escenarios Metodológicos", divider="blue")
    st.info("Selecciona dos escenarios diferentes para comparar sus resultados de eficiencia, configuraciones de inputs/outputs y la calidad de su razonamiento metodológico (EEE).")

    if len(st.session_state.scenarios) < 2:
        st.warning("Necesitas al menos dos escenarios creados y analizados para poder realizar una comparación. Utiliza la barra lateral para crear nuevos escenarios.")
        return

    col1, col2 = st.columns(2)
    scenario_names = {sid: s['name'] for sid, s in st.session_state.scenarios.items()}

    with col1:
        id_a = st.selectbox(
            "Comparar Escenario A:", 
            list(st.session_state.scenarios.keys()), 
            format_func=lambda sid: scenario_names[sid], 
            key='compare_a',
            help="Selecciona el primer escenario para la comparación."
        )
    with col2:
        options_b = [sid for sid in st.session_state.scenarios.keys() if sid != id_a] or [id_a] 
        id_b = st.selectbox(
            "Con Escenario B:", 
            options_b, 
            format_func=lambda sid: scenario_names[sid], 
            key='compare_b',
            help="Selecciona el segundo escenario para la comparación. Asegúrate de que sea diferente al Escenario A."
        )

    st.divider()
    scenario_a = st.session_state.scenarios.get(id_a)
    scenario_b = st.session_state.scenarios.get(id_b)

    if not scenario_a or not scenario_b: 
        st.error("No se pudieron cargar los escenarios seleccionados. Por favor, verifica tus selecciones.")
        return

    res_col1, res_col2 = st.columns(2)
    for sc, col in [(scenario_a, res_col1), (scenario_b, res_col2)]:
        with col:
            st.subheader(f"Resultados de: {sc['name']}")
            with st.container(border=True):
                if sc.get('dea_results'):
                    st.markdown("**Configuración del Modelo:**")
                    st.json(sc.get('dea_config', {}), expanded=False) 
                    if sc.get('selected_proposal'):
                        st.markdown(f"**Inputs Seleccionados:** {sc['selected_proposal'].get('inputs')}")
                        st.markdown(f"**Outputs Seleccionados:** {sc['selected_proposal'].get('outputs')}")
                    st.markdown("**Primeras 5 Filas de Resultados:**")
                    st.dataframe(sc['dea_results']['main_df']) 
                    if sc.get('inquiry_tree'):
                        eee_metrics = compute_eee(sc['inquiry_tree'], depth_limit=3, breadth_limit=5)
                        st.metric("Calidad del Juicio (EEE)", f"{eee_metrics['score']:.2%}", help="El Índice de Equilibrio Erotético (EEE) mide la profundidad, pluralidad y robustez de tu mapa de razonamiento. Una puntuación más alta indica un análisis metodológico más sólido y deliberado.")
                else:
                    st.info("Este escenario aún no ha sido calculado. Ejecuta el análisis en la pestaña 'Análisis del Escenario Activo' para ver sus resultados aquí.")

def render_eee_explanation(eee_metrics: dict):
    st.info(f"**Calidad del Juicio Metodológico (EEE): {eee_metrics['score']:.2%}**")
    st.markdown("El Índice de Equilibrio Erotético (EEE) te ayuda a evaluar la calidad de tu proceso de razonamiento metodológico. Una puntuación más alta indica un análisis más profundo y robusto.")
    with st.expander("Ver desglose y consejos para mejorar tu puntuación EEE"):
        def interpret_score(name, score):
           if score >= 0.8: return f"**{name}:** Tu puntuación es **excelente** ({score:.0%}). Bien hecho."
           if score >= 0.5: return f"**{name}:** Tu puntuación es **buena** ({score:.0%}). Puedes mejorarla profundizando más."
           return f"**{name}:** Tu puntuación es **baja** ({score:.0%}). Considera expandir tu razonamiento."
        st.markdown(f"""
        - {interpret_score("Profundidad (D1)", eee_metrics['D1'])}: Mide si tu análisis metodológico es lo suficientemente profundo. Para mejorar, asegúrate de que el mapa de auditoría tenga varios niveles de sub-preguntas.
        - {interpret_score("Pluralidad (D2)", eee_metrics['D2'])}: Valora si has considerado múltiples facetas o riesgos metodológicos desde el inicio. Para mejorar, genera mapas de auditoría con varias ramas principales.
        - {interpret_score("Robustez (D5)", eee_metrics['D5'])}: Evalúa la solidez y el detalle general del árbol de preguntas y tus justificaciones. Para mejorar, proporciona justificaciones exhaustivas y considera las diferentes perspectivas planteadas por la IA.
        """)

def render_interactive_inquiry_tree(active_scenario):
    """Muestra el árbol de preguntas y captura las justificaciones del usuario."""
    st.subheader("Taller de Auditoría Metodológica", anchor=False)
    st.info("Este es el corazón de tu informe deliberativo. Responde a cada pregunta de auditoría generada por la IA, documentando tu razonamiento, citando literatura o explicando las decisiones clave que tomaste. Tu objetivo es justificar la robustez y validez de tu análisis DEA.")
    
    tree = active_scenario.get("inquiry_tree")
    if not tree:
        st.warning("Aún no se ha generado un mapa metodológico. Haz clic en el botón 'Generar Mapa Metodológico' en la sección superior para crearlo y empezar a documentar tu análisis.")
        return

    # Helper function to recursively render nodes
    def _render_node_recursively(node_dict, level=0, path_prefix=""):
        for question, children in node_dict.items():
            # Create a unique path for the current question
            # This path ensures the Streamlit key is unique across all text_areas
            # Convert question to a safe string for key
            safe_question = "".join(c for c in question if c.isalnum() or c in [' ', '_', '-']).replace(' ', '_')[:50]
            current_path = f"{path_prefix}__{safe_question}_{level}" if path_prefix else f"{safe_question}_{level}"
            
            st.markdown(f"<div style='margin-left: {level*20}px; border-left: 2px solid #ccc; padding-left: 10px; margin-top: 10px;'>"
                        f"<b>Pregunta de Auditoría:</b> {question}"
                        f"</div>", unsafe_allow_html=True)
            
            justification_key = f"just_{st.session_state.active_scenario_id}_{current_path}"
            # Use the original question text for dictionary lookup, not the full path
            current_justification = active_scenario['user_justifications'].get(question, "")
            
            user_input = st.text_area(
                "Tu justificación metodológica:",
                value=current_justification,
                key=justification_key,
                label_visibility="collapsed",
                placeholder="Escribe aquí tu razonamiento, citando literatura, explicando decisiones o refutando la pregunta de la IA. ¡Sé lo más detallado posible!"
            )
            
            # Store justification using the original question as key
            active_scenario['user_justifications'][question] = user_input
            
            if isinstance(children, dict) and children:
                _render_node_recursively(children, level + 1, current_path)

    # Initial call to start the recursive rendering
    _render_node_recursively(tree)

def render_deliberation_workshop(active_scenario):
    if not active_scenario.get('dea_results'): 
        st.info("Ejecuta un análisis DEA en el Paso 3 para desbloquear el Taller de Deliberación Metodológica.")
        return
    
    st.header("Paso 4: Deliberación y Justificación Metodológica", divider="blue")
    st.info("Esta etapa es crucial para abordar los **retos metodológicos y de interpretación** del DEA. Utiliza el mapa de auditoría generado por la IA para documentar tu razonamiento y justificar las decisiones clave del análisis. El objetivo es construir una narrativa robusta que acompañe tus resultados cuantitativos.")

    with st.container(border=True):
        st.subheader("Mapa de Auditoría (IA)", anchor=False)
        st.markdown("La IA generará un árbol de preguntas de auditoría basado en la configuración de tu modelo. Este mapa te ayudará a reflexionar sobre la validez, la robustez y las mejores prácticas de tu análisis DEA.")
        root_question_methodology = (
            f"Para un modelo DEA con enfoque '{active_scenario['selected_proposal'].get('title', 'N/A')}', "
            f"inputs {active_scenario['selected_proposal']['inputs']} y "
            f"outputs {active_scenario['selected_proposal']['outputs']},"
            "¿cuáles son los principales desafíos metodológicos y las mejores prácticas para asegurar la robustez del análisis?"
        )
        if st.button("Generar Mapa Metodológico", use_container_width=True, key=f"gen_map_{st.session_state.active_scenario_id}", help="Genera un árbol de preguntas para auditar la validez de tu análisis DEA."):
            with st.spinner("La IA está generando un árbol de auditoría... Esto puede tardar un momento."):
                context = {
                    "model": active_scenario['dea_results'].get("model_name"),
                    "inputs": active_scenario['selected_proposal']['inputs'],
                    "outputs": active_scenario['selected_proposal']['outputs'],
                    "num_dmus": len(active_scenario['df'])
                }
                tree, error = cached_run_inquiry_engine(root_question_methodology, context)
                if error: st.error(f"Error al generar el mapa: {error}")
                active_scenario['inquiry_tree'] = tree
                active_scenario['user_justifications'] = {} # Limpia justificaciones al generar nuevo árbol
        
        if active_scenario.get("inquiry_tree"):
            with st.expander("Ver visualización del árbol y explicación de la IA", expanded=False):
                st.markdown("Explora la estructura jerárquica de las preguntas de auditoría.")
                st.plotly_chart(to_plotly_tree(active_scenario['inquiry_tree']), use_container_width=True)
                if st.button("Generar Explicación del Mapa por IA", key=f"explain_tree_{st.session_state.active_scenario_id}", help="Obtén una explicación en lenguaje natural del propósito y la estructura del mapa de auditoría generado."):
                    with st.spinner("La IA está redactando la explicación..."):
                        explanation_data = cached_explain_tree(active_scenario['inquiry_tree'])
                        if explanation_data.get("error"):
                            st.error(f"Error al generar explicación: {explanation_data['error']}")
                        else:
                            active_scenario['tree_explanation'] = explanation_data['text']
                if active_scenario.get('tree_explanation'):
                    st.markdown("---")
                    st.subheader("Explicación del Mapa (Generada por IA)")
                    st.markdown(active_scenario['tree_explanation'])
            
            eee_metrics = compute_eee(active_scenario['inquiry_tree'], depth_limit=3, breadth_limit=5)
            render_eee_explanation(eee_metrics)

    st.divider()
    
    # Nuevo componente interactivo para las justificaciones
    render_interactive_inquiry_tree(active_scenario)

def render_optimization_workshop(active_scenario):
    if not active_scenario.get('dea_results'): 
        st.info("Ejecuta un análisis DEA en el Paso 3 para desbloquear el Taller de Optimización Asistida por IA.")
        return
    
    st.header("Paso 5: Optimización Asistida por IA", divider="blue")
    st.info("Esta sección te ayuda a mitigar la **sensibilidad del modelo** y explorar **diferentes aproximaciones** a tus datos. La IA puede sugerir variaciones en la selección de inputs/outputs o el tipo de modelo para buscar configuraciones potencialmente más robustas o informativas.")

    if active_scenario.get('selected_proposal') and active_scenario.get('dea_results'):
        current_inputs = active_scenario['selected_proposal']['inputs']
        current_outputs = active_scenario['selected_proposal']['outputs']
        current_model = active_scenario['dea_config'].get('model', 'CCR_BCC') 
        
        eee_score = 0.5 
        if active_scenario.get('inquiry_tree'):
            eee_metrics = compute_eee(active_scenario['inquiry_tree'], depth_limit=3, breadth_limit=5)
            eee_score = eee_metrics['score']
            
        st.subheader("Generar Candidatos de Configuración")
        st.markdown("Permite que la IA proponga configuraciones alternativas de inputs y outputs. Esto es útil para explorar la robustez de tus resultados ante diferentes especificaciones de modelo.")
        if st.button("Sugerir Configuraciones Alternativas", key=f"gen_candidates_{st.session_state.active_scenario_id}", help="La IA generará un conjunto de combinaciones alternativas de inputs y outputs basadas en el conjunto de datos cargado."):
            with st.spinner("La IA está generando candidatos de configuración..."):
                candidates = cached_generate_candidates(
                    active_scenario['df'],
                    active_scenario['df'].columns[0],
                    current_inputs,
                    current_outputs,
                    active_scenario['inquiry_tree'], 
                    eee_score
                )
                active_scenario['optimization_candidates'] = candidates
                active_scenario['optimization_evaluations'] = None
        
        if active_scenario.get('optimization_candidates'):
            st.subheader("Evaluar Candidatos")
            st.markdown("Una vez generados los candidatos, evalúa rápidamente su impacto en la eficiencia promedio y una métrica simulada de calidad de juicio (EEE).")
            if st.button("Evaluar Candidatos Sugeridos", key=f"eval_candidates_{st.session_state.active_scenario_id}", help="Ejecuta un análisis preliminar de eficiencia para cada configuración sugerida por la IA."):
                with st.spinner("Evaluando candidatos... Esto puede llevar un tiempo para un gran número de candidatos."):
                    evaluations = cached_evaluate_candidates(
                        active_scenario['df'],
                        active_scenario['df'].columns[0],
                        active_scenario['optimization_candidates'],
                        current_model
                    )
                    active_scenario['optimization_evaluations'] = evaluations
            
            if active_scenario.get('optimization_evaluations') is not None:
                st.subheader("Resultados de la Evaluación de Candidatos")
                st.markdown("Revisa cómo las diferentes combinaciones de inputs y outputs afectan la eficiencia promedio y compara con tu escenario actual (`delta_eff`).")
                st.dataframe(active_scenario['optimization_evaluations'])

                selected_candidate_index = st.selectbox(
                    "Selecciona un candidato para aplicar:",
                    options=range(len(active_scenario['optimization_evaluations'])),
                    format_func=lambda idx: f"Candidato {idx+1}: I={active_scenario['optimization_evaluations'].loc[idx, 'inputs']}, O={active_scenario['optimization_evaluations'].loc[idx, 'outputs']}",
                    key=f"select_opt_cand_{st.session_state.active_scenario_id}",
                    help="Elige la configuración que deseas explorar en un nuevo escenario de análisis."
                )

                if st.button("Aplicar Configuración Seleccionada (Crea Nuevo Escenario)", key=f"apply_opt_cand_{st.session_state.active_scenario_id}", help="Crea un nuevo escenario en el Navegador de Escenarios (barra lateral) con esta configuración, listo para que realices un análisis completo."):
                    chosen_candidate = active_scenario['optimization_evaluations'].loc[selected_candidate_index]
                    
                    new_scenario_name = f"Optimizado - {active_scenario['name']} (Candidato {selected_candidate_index+1})"
                    new_id = str(uuid.uuid4()) 

                    st.session_state.scenarios[new_id] = {
                        "name": new_scenario_name,
                        "df": active_scenario['df'],
                        "app_status": "proposal_selected", 
                        "proposals_data": None, 
                        "selected_proposal": { 
                            "title": f"Configuración Optimizada de {active_scenario['name']}",
                            "reasoning": "Configuración sugerida por la IA durante la fase de optimización.",
                            "inputs": chosen_candidate['inputs'],
                            "outputs": chosen_candidate['outputs']
                        },
                        "dea_config": active_scenario['dea_config'].copy(), 
                        "dea_results": None,
                        "inquiry_tree": None,
                        "tree_explanation": None,
                        "user_justifications": {},
                        "data_overview": active_scenario['data_overview'].copy() 
                    }
                    st.session_state.active_scenario_id = new_id
                    st.rerun()
            else:
                st.info("Haz clic en 'Evaluar Candidatos Sugeridos' para ver los resultados de cada configuración alternativa propuesta por la IA.")
        else:
            st.info("Haz clic en 'Sugerir Configuraciones Alternativas' para que la IA proponga nuevas combinaciones de variables que podrías explorar.")
    else:
        st.warning("Necesitas haber ejecutado un análisis DEA base (Paso 3) para usar esta sección de optimización.")


def render_download_section(active_scenario):
    results = active_scenario.get('dea_results')
    if not results: return

    st.subheader("Exportar Análisis del Escenario", divider="gray")
    st.markdown(f"Descarga los resultados cuantitativos y el informe deliberativo (que incluye tus justificaciones y el mapa de auditoría de la IA) para el escenario **'{active_scenario['name']}'**.")
    col1, col2 = st.columns(2)
    with col1:
        html_report = generate_html_report(
            analysis_results=results,
            inquiry_tree=active_scenario.get("inquiry_tree"),
            user_justifications=active_scenario.get("user_justifications", {}),
            data_overview_info=active_scenario.get("data_overview", {})
        )
        st.download_button(label="Descargar Informe en HTML", data=html_report, file_name=f"report_{active_scenario['name'].replace(' ', '_')}.html", mime="text/html", use_container_width=True, help="Descarga un informe HTML completo con los resultados, el mapa de auditoría y tus justificaciones.")
    with col2:
        excel_report = generate_excel_report(
            analysis_results=results,
            inquiry_tree=active_scenario.get("inquiry_tree"),
            user_justifications=active_scenario.get("user_justifications", {}),
            data_overview_info=active_scenario.get("data_overview", {})
        )
        st.download_button(label="Descargar Informe en Excel", data=excel_report, file_name=f"report_{active_scenario['name'].replace(' ', '_')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, help="Descarga un archivo Excel con los resultados numéricos del DEA, el resumen de datos y una tabla del mapa de auditoría.")


def render_main_dashboard(active_scenario):
    st.header(f"Paso 3: Configuración y Análisis para '{active_scenario['name']}'", divider="blue")
    st.markdown("En este paso, configurarás y ejecutarás el modelo DEA. Asegúrate de que los inputs y outputs seleccionados sean los correctos y de que la elección del modelo se alinee con los objetivos de tu análisis.")
    
    st.markdown("---")
    st.subheader("Configuración Actual de Inputs y Outputs:", anchor=False)
    st.markdown("Verifica y, si es necesario, ajusta los inputs (recursos consumidos) y outputs (productos generados) para tu modelo DEA. Estos fueron definidos en el Paso 2 o seleccionados de una propuesta de la IA.")
    col_inputs, col_outputs = st.columns(2)
    with col_inputs:
        all_cols = active_scenario['df'].columns.tolist()
        available_cols = [col for col in all_cols if col != active_scenario['df'].columns[0]] 
        
        current_inputs = active_scenario['selected_proposal'].get('inputs', [])
        current_outputs = active_scenario['selected_proposal'].get('outputs', [])

        st.markdown("**Inputs Seleccionados:**")
        selected_inputs = st.multiselect(
            "Selecciona o edita los inputs:",
            options=available_cols,
            default=current_inputs,
            key=f"manual_inputs_{st.session_state.active_scenario_id}",
            help="Elige las columnas que representan los insumos utilizados por tus DMUs."
        )
        active_scenario['selected_proposal']['inputs'] = selected_inputs
    
    with col_outputs:
        st.markdown("**Outputs Seleccionados:**")
        selected_outputs = st.multiselect(
            "Selecciona o edita los outputs:",
            options=available_cols,
            default=current_outputs,
            key=f"manual_outputs_{st.session_state.active_scenario_id}",
            help="Elige las columnas que representan los productos o resultados generados por tus DMUs."
        )
        active_scenario['selected_proposal']['outputs'] = selected_outputs

    st.markdown("---")

    model_options = {"Radial (CCR/BCC)": "CCR_BCC", "No Radial (SBM)": "SBM", "Productividad (Malmquist)": "MALMQUIST"}
    
    current_model_key = active_scenario['dea_config'].get('model', 'CCR_BCC')
    current_model_name = [name for name, key in model_options.items() if key == current_model_key][0]
    
    model_name = st.selectbox(
        "1. Selecciona el tipo de modelo DEA:", 
        list(model_options.keys()), 
        index=list(model_options.values()).index(current_model_key), 
        key=f"model_select_{st.session_state.active_scenario_id}",
        help="Elige el modelo DEA a ejecutar. Cada modelo tiene diferentes supuestos y objetivos (ej. eficiencia técnica vs. eficiencia de holgura, análisis de productividad)."
    )
    model_key = model_options[model_name]
    
    active_scenario['dea_config']['model'] = model_key

    st.info(f"**Reto Metodológico: Elección del Modelo DEA.** La selección del modelo ({model_name}) y su orientación (ej. inputs vs. outputs, rendimientos a escala) es crucial. Afecta la forma de la frontera de eficiencia y las puntuaciones resultantes. Asegúrate de que esta elección sea coherente con la realidad operativa del proceso que estás analizando. Por ejemplo, el modelo radial (CCR/BCC) asume reducciones proporcionales de inputs, mientras que el no radial (SBM) aborda las holguras directamente.")


    period_col = None
    if model_key == 'MALMQUIST':
        st.markdown("Para el Índice de Productividad de Malmquist, debes especificar una columna que represente los períodos de tiempo de tus datos panel.")
        period_col_options = [None] + [col for col in active_scenario['df'].columns.tolist() if col != active_scenario['df'].columns[0]]
        current_period_col = active_scenario['dea_config'].get('period_col', None)
        period_col_index = period_col_options.index(current_period_col) if current_period_col in period_col_options else 0
        period_col = st.selectbox(
            "2. Selecciona la columna de período:", 
            period_col_options, 
            index=period_col_index, 
            key=f"period_col_{st.session_state.active_scenario_id}",
            help="Selecciona la columna que identifica los diferentes períodos de tiempo para el cálculo del Índice de Malmquist."
        )
        if not period_col: 
            st.warning("El modelo Malmquist requiere una columna de período válida. Por favor, selecciona una columna de tiempo o cambia el tipo de modelo.")
            st.stop() 
        active_scenario['dea_config']['period_col'] = period_col
    
    if st.button(f"Ejecutar Análisis DEA para '{active_scenario['name']}'", type="primary", use_container_width=True, help="Inicia el cálculo del modelo DEA con la configuración actual. Los resultados y las visualizaciones aparecerán a continuación."):
        with st.spinner(f"Ejecutando {model_name} para '{active_scenario['name']}'... Esto puede tomar unos segundos o minutos dependiendo del tamaño de tus datos."):
            df = active_scenario['df']
            proposal = active_scenario['selected_proposal']
            
            validation_results = validate_data(df, proposal.get('inputs', []), proposal.get('outputs', []))
            active_scenario['data_overview']['llm_validation_results'] = validation_results
            
            if validation_results['formal_issues']:
                for issue in validation_results['formal_issues']:
                    st.error(f"Error de validación formal: {issue}")
                st.warning("Por favor, corrige los errores de validación antes de ejecutar el análisis. Puedes volver al Paso 2 para ajustar las variables.")
                return 
            
            try:
                results = cached_run_dea_analysis(df, df.columns[0], proposal.get('inputs', []), proposal.get('outputs', []), model_key, period_col)
                active_scenario['dea_results'] = results
                active_scenario['app_status'] = "results_ready"
            except Exception as e:
                st.error(f"Error durante el análisis: {e}. Asegúrate de que tus datos sean adecuados y que las variables estén bien seleccionadas. Consulta la pestaña 'Retos del DEA' para más información.")
                active_scenario['dea_results'] = None
        st.rerun() 

    if active_scenario.get("dea_results"):
        results = active_scenario["dea_results"]
        st.header(f"Resultados para: {results['model_name']}", divider="blue")
        st.markdown("Aquí puedes explorar los resultados de eficiencia de tus DMUs y las visualizaciones clave. Estos resultados son la base para tu deliberación metodológica y tu informe final.")
        st.dataframe(results['main_df']) 
        if results.get("charts"):
            st.subheader("Visualizaciones de Resultados")
            st.markdown("Los gráficos te ayudarán a entender la distribución de la eficiencia y otras relaciones importantes en tus datos.")
            for chart_title, fig in results["charts"].items():
                st.plotly_chart(fig, use_container_width=True, help=f"Gráfico: {chart_title}")
        
        render_deliberation_workshop(active_scenario)
        render_optimization_workshop(active_scenario)
        render_download_section(active_scenario)

def render_preliminary_analysis_step(active_scenario):
    st.header(f"Paso 1b: Exploración Preliminar de Datos para '{active_scenario['name']}'", divider="blue")
    st.info("Este paso es crucial para **entender tus datos** antes de realizar el análisis DEA. Te ayudará a identificar posibles problemas (como outliers o multicolinealidad) y a tomar decisiones informadas sobre la selección de inputs y outputs. La visualización es clave para el **pensamiento crítico** aquí.")

    df = active_scenario['df']
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

    if not numerical_cols:
        st.warning("No se encontraron columnas numéricas para realizar el análisis exploratorio. Asegúrate de que tu archivo CSV contenga datos numéricos.")
        if st.button("Proceder al Paso 2: Elegir Enfoque", key=f"proceed_to_step2_no_numeric_{st.session_state.active_scenario_id}"):
            active_scenario['app_status'] = "file_loaded"
            st.rerun()
        return

    # Store charts in data_overview for reporting
    # No es necesario re-calcular si ya están en el caché del escenario
    if 'preliminary_analysis_charts' not in active_scenario['data_overview']:
        active_scenario['data_overview']['preliminary_analysis_charts'] = {}
    
    # Recalcular la matriz de correlación si no existe o si se ha actualizado el DF global
    if 'correlation_matrix' not in active_scenario['data_overview'] or \
       active_scenario['data_overview'].get('last_df_hash') != hash(df.to_numpy().tobytes()): # Simple hash check
        active_scenario['data_overview']['correlation_matrix'] = df[numerical_cols].corr().to_dict()
        active_scenario['data_overview']['last_df_hash'] = hash(df.to_numpy().tobytes())


    st.subheader("1. Estadísticas Descriptivas:", anchor=False)
    st.markdown("Un resumen rápido de las características centrales, dispersión y forma de tus datos numéricos.")
    st.dataframe(df[numerical_cols].describe().T, help="Estadísticas descriptivas para todas las columnas numéricas: conteo, media, desviación estándar, valores mínimos y máximos, y cuartiles. Esto te da una primera idea de la distribución de tus variables.")

    st.subheader("2. Distribución de Variables (Histogramas):", anchor=False)
    st.markdown("Visualiza la distribución de cada variable numérica. Esto te ayuda a identificar asimetrías, rangos de valores y la presencia de posibles outliers.")
    
    if 'histograms' not in active_scenario['data_overview']['preliminary_analysis_charts']:
        active_scenario['data_overview']['preliminary_analysis_charts']['histograms'] = {}

    for col in numerical_cols:
        fig = px.histogram(df, x=col, title=f"Distribución de {col}", 
                           labels={col: col}, 
                           template="plotly_white")
        st.plotly_chart(fig, use_container_width=True, key=f"hist_{col}_{st.session_state.active_scenario_id}", help=f"Histograma de la columna '{col}'. Observa la forma de la distribución, si es simétrica, sesgada, o si hay valores atípicos.")
        # Guardar la figura como JSON para el reporte
        active_scenario['data_overview']['preliminary_analysis_charts']['histograms'][col] = fig.to_json()


    st.subheader("3. Matriz de Correlación (Mapa de Calor):", anchor=False)
    st.markdown("Examina las relaciones lineales entre tus variables numéricas. Una alta correlación entre inputs o entre outputs puede indicar **multicolinealidad**, un reto potencial en DEA.")
    
    # Recuperar la matriz de correlación del estado si ya existe
    corr_matrix_dict = active_scenario['data_overview'].get('correlation_matrix', {})
    if corr_matrix_dict:
        corr_matrix = pd.DataFrame(corr_matrix_dict) # Reconstruir DataFrame desde el dict guardado
        fig_corr = px.imshow(corr_matrix, 
                            text_auto=True, 
                            aspect="auto",
                            color_continuous_scale=px.colors.sequential.RdBu,
                            range_color=[-1,1],
                            title="Matriz de Correlación entre Variables Numéricas",
                            labels=dict(color="Correlación"))
        st.plotly_chart(fig_corr, use_container_width=True, key=f"corr_heatmap_{st.session_state.active_scenario_id}", help="Mapa de calor de la matriz de correlación. Valores cercanos a 1 o -1 indican fuerte correlación. Valores cercanos a 0 indican poca o ninguna correlación lineal. La alta correlación entre inputs u outputs puede indicar multicolinealidad, lo que puede afectar los pesos de las variables en DEA.")
    else:
        st.info("No se pudo generar la matriz de correlación. Asegúrate de tener al menos dos columnas numéricas.")
    
    st.markdown("---")
    st.subheader("Conclusiones de la Exploración Preliminar:")
    st.info("""
    Después de revisar estas visualizaciones:
    * **Identifica posibles Outliers:** ¿Hay puntos de datos que parecen muy diferentes del resto en los histogramas? Esto puede afectar la frontera de eficiencia en DEA.
    * **Evalúa la Multicolinealidad:** En el mapa de calor de correlación, ¿hay pares de inputs o de outputs con correlaciones muy altas (cercanas a 1 o -1)? Si es así, podría ser recomendable elegir solo una de esas variables en el Paso 2 para evitar redundancias y problemas de interpretación en los pesos del DEA.
    * **Distribución de Datos:** ¿Las distribuciones de tus variables son muy asimétricas? Esto podría influir en la robustez del modelo.

    Utiliza esta información para tomar decisiones más informadas al seleccionar tus inputs y outputs en el siguiente paso.
    """)

    if st.button("Proceder al Paso 2: Elegir Enfoque de Análisis", type="primary", use_container_width=True, help="Haz clic aquí para continuar y aplicar las ideas de esta exploración inicial a la selección de tus variables DEA."):
        active_scenario['app_status'] = "file_loaded" # Mueve al siguiente estado para la selección de variables
        st.rerun()

def render_validation_step(active_scenario):
    st.header(f"Paso 2b: Validación del Modelo para '{active_scenario['name']}'", divider="gray")
    st.info("Antes de ejecutar el análisis DEA, es fundamental validar la calidad de tus datos y la coherencia de tu selección de inputs y outputs. Esta sección te mostrará los resultados de una doble validación: formal y asistida por IA.")
    
    proposal = active_scenario.get('selected_proposal')
    
    if not proposal or not proposal.get('inputs') or not proposal.get('outputs'):
        st.error("La propuesta de análisis de este escenario está incompleta. Por favor, vuelve al Paso 2 para definir inputs y outputs.")
        return
    
    st.markdown(f"**Propuesta Seleccionada:** *{proposal.get('title', 'Configuración Manual')}*")
    st.markdown(f"**Inputs:** {proposal.get('inputs', [])}")
    st.markdown(f"**Outputs:** {proposal.get('outputs', [])}")
    st.markdown(f"**Razonamiento:** {proposal.get('reasoning', 'Configuración definida por el usuario o sugerida por la IA.')}")

    with st.spinner("La IA está validando la coherencia de los datos y el modelo... Esto puede tomar unos segundos."):
        validation_results = validate_data(active_scenario['df'], proposal['inputs'], proposal['outputs'])
        active_scenario['data_overview']['llm_validation_results'] = validation_results
    
    if validation_results['formal_issues']:
        st.error("**Reto de Datos: Datos Problemáticos.** Se encontraron problemas de validación formal en los datos o columnas seleccionadas. El DEA requiere que los inputs y outputs sean estrictamente positivos. La presencia de valores nulos, negativos o cero, o columnas no numéricas, puede causar errores o resultados inválidos en el modelo.")
        for issue in validation_results['formal_issues']:
            st.warning(f"- {issue}")
        st.info("Por favor, regresa al Paso 2 para ajustar las columnas o el dataset. Es crucial que los datos cumplan con los requisitos del DEA.")
    else:
        st.success("La validación formal inicial de datos y columnas ha sido exitosa. ¡Buen trabajo!")
    
    if validation_results['llm']['issues']:
        st.info("**Reto de Datos: Idoneidad y Homogeneidad.** La IA ha detectado posibles problemas conceptuales o sugerencias sobre la idoneidad de las variables o la homogeneidad de las DMUs. Considera estas observaciones para asegurar que tus DMUs son verdaderamente comparables y que las variables capturan el proceso de producción de forma adecuada. Resuelve estos puntos antes de proceder para asegurar la validez de tu análisis.")
        for issue in validation_results['llm']['issues']:
            st.warning(f"- {issue}")
        if validation_results['llm']['suggested_fixes']:
            st.markdown("**Sugerencias de la IA para mejorar:**")
            for fix in validation_results['llm']['suggested_fixes']:
                st.info(f"- {fix}")

    if not validation_results['formal_issues']: 
        if st.button("Proceder al Análisis", key=f"validate_{st.session_state.active_scenario_id}", type="primary", use_container_width=True, help="Si los resultados de validación son satisfactorios, haz clic aquí para pasar al siguiente paso y ejecutar el análisis DEA."):
            active_scenario['app_status'] = "validated"
            st.rerun()
    else:
        st.warning("Por favor, resuelve los problemas de validación formal antes de proceder al análisis. El DEA no funcionará correctamente con datos inválidos.")


def render_dea_challenges_tab():
    st.header("Retos Relevantes en el Uso del Análisis Envolvente de Datos (DEA)", divider="blue")
    st.markdown("""
    El Análisis Envolvente de Datos (DEA) es una herramienta potente, pero su aplicación exitosa depende de entender y abordar sus desafíos inherentes.
    """)

    st.subheader("1. Retos Relacionados con los Datos")
    st.markdown("""
    * **Selección de Insumos (Inputs) y Productos (Outputs):** Elegir las variables adecuadas es subjetivo y requiere justificación teórica. Una mala elección puede sesgar los resultados.
        * **La aplicación ayuda:** En el **Paso 2**, la IA sugiere inputs/outputs y permite la edición manual para asegurar la relevancia.
    * **Disponibilidad y Calidad de Datos:** Datos incompletos o erróneos pueden invalidar el análisis.
    * **Número de Variables vs. DMUs:** Demasiadas variables para pocas DMUs pueden inflar artificialmente la eficiencia.
    * **Valores Nulos, Negativos y Cero:** Los modelos DEA clásicos requieren datos positivos. Estos valores deben tratarse adecuadamente.
        * **La aplicación ayuda:** En el **Paso 1**, se ofrece un informe rápido de los datos cargados y una guía de preparación. En el **Paso 2b**, se realizan validaciones formales para detectar y advertir sobre estos problemas antes del análisis.
    * **Outliers (Valores Atípicos):** El DEA es sensible a los outliers, que pueden distorsionar la frontera de eficiencia.
    * **Homogeneidad de las DMUs:** Las unidades analizadas deben ser comparables entre sí. Comparar entidades muy dispares lleva a conclusiones erróneas.
        * **La aplicación ayuda:** En el **Paso 2**, se enfatiza la importancia de la homogeneidad, y la IA puede ofrecer sugerencias al respecto.
    """)

    st.subheader("2. Retos Metodológicos y de Especificación del Modelo")
    st.markdown("""
    * **Elección del Modelo DEA (CCR, BCC, SBM, etc.) y su Orientación:** La decisión sobre los rendimientos a escala (CRS vs. VRS) y la orientación (minimizar inputs vs. maximizar outputs) es crítica y afecta la forma de la frontera y las puntuaciones.
        * **La aplicación ayuda:** En el **Paso 3**, se guía al usuario en la selección del modelo y se ofrece información contextual sobre las implicaciones de cada elección.
    * **Falta de Pruebas de Significación Estadística:** El DEA no ofrece pruebas de significancia tradicionales, lo que dificulta generalizar los resultados.
    * **Sensibilidad del Modelo:** Los resultados pueden ser muy sensibles a pequeñas variaciones en los datos o en la inclusión/exclusión de DMUs.
        * **La aplicación ayuda:** En el **Paso 5**, la IA asiste en la exploración de configuraciones alternativas para evaluar la robustez del modelo.
    """)

    st.subheader("3. Retos de Interpretación y Aplicabilidad")
    st.markdown("""
    * **Interpretación de Puntuaciones de Eficiencia:** La eficiencia en DEA es *relativa* a la muestra, no absoluta.
    * **Identificación de Benchmarks:** Replicar las "mejores prácticas" de los benchmarks puede ser difícil en la realidad.
    * **Implicaciones de Política:** Traducir los resultados en acciones concretas requiere un profundo conocimiento del dominio.
    * **Dimensionalidad de la Proyección:** Entender las proyecciones para DMUs ineficientes puede ser complejo.
    * **La aplicación ayuda:** El **Paso 4 (Taller de Auditoría)** y los informes generados están diseñados para ayudar al investigador a deliberar, justificar y documentar la interpretación de los resultados, transformando el análisis cuantitativo en conocimiento accionable.
    """)

# --- 6) FLUJO PRINCIPAL DE LA APLICACIÓN ---
def main():
    """Función principal que orquesta la aplicación multi-escenario."""
    initialize_global_state() 

    st.sidebar.image("https://i.imgur.com/8y0N5c5.png", width=200)
    st.sidebar.title("DEA Deliberative Modeler")
    st.sidebar.markdown("Una herramienta para el análisis de eficiencia y la deliberación metodológica asistida por IA. Sigue los pasos para un estudio DEA robusto.")
    if st.sidebar.button("🔴 Empezar Nueva Sesión", help="Borra todos los datos y escenarios actuales para empezar un análisis desde cero. ¡Cuidado, esta acción no se puede deshacer!"):
        reset_all()
        st.rerun() 
    st.sidebar.divider()
    
    render_scenario_navigator()

    st.sidebar.markdown("---")
    st.sidebar.info("Una herramienta para el análisis de eficiencia y la deliberación metodológica asistida por IA.")

    active_scenario = get_active_scenario() 

    # Se usa una lógica de estados más explícita para controlar qué función de renderizado se llama
    # dentro del tab principal, dependiendo del progreso del usuario.
    analysis_tab, comparison_tab, challenges_tab = st.tabs([
        "Análisis del Escenario Activo", 
        "Comparar Escenarios", 
        "Retos del DEA"
    ])

    with analysis_tab:
        app_status = active_scenario.get('app_status', 'initial') if active_scenario else 'initial'

        if app_status == "initial":
            render_upload_step()
        elif app_status == "data_loaded":
            render_preliminary_analysis_step(active_scenario)
        elif app_status == "file_loaded": # Este estado se usa después de la exploración, para elegir enfoque
            render_proposal_step(active_scenario)
        elif app_status == "proposal_selected":
            render_validation_step(active_scenario)
        elif app_status in ["validated", "results_ready"]:
            render_main_dashboard(active_scenario)
    
    with comparison_tab:
        render_comparison_view()
    
    with challenges_tab:
        render_dea_challenges_tab()


if __name__ == "__main__":
    main()

