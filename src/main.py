import sys
import os
import pandas as pd
import streamlit as st
import io
import json
import uuid
from openai import OpenAI

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
from openai_helpers import explain_inquiry_tree

# --- 2) GESTIÓN DE ESTADO MULTI-ESCENARIO ---
# Este es el núcleo de la refactorización. Se pasa de un estado único
# a una gestión de múltiples escenarios de análisis.

def create_new_scenario(name: str = "Modelo Base", source_scenario_id: str = None):
    """Crea un nuevo escenario, ya sea en blanco o clonando uno existente."""
    new_id = str(uuid.uuid4())
    
    # Si se proporciona un escenario fuente, clónalo.
    if source_scenario_id and source_scenario_id in st.session_state.scenarios:
        # Crea una copia profunda del diccionario del escenario fuente.
        st.session_state.scenarios[new_id] = st.session_state.scenarios[source_scenario_id].copy()
        st.session_state.scenarios[new_id]['name'] = f"Copia de {st.session_state.scenarios[source_scenario_id]['name']}"
    else:
        # Si no, crea un escenario virgen con valores por defecto.
        st.session_state.scenarios[new_id] = {
            "name": name,
            "df": st.session_state.get("global_df", None), # Usa el dataframe global si existe
            "app_status": "file_loaded" if st.session_state.get("global_df") is not None else "initial",
            "proposals_data": None,
            "selected_proposal": None,
            "dea_config": {},
            "dea_results": None,
            "inquiry_tree": None,
            "tree_explanation": None,
            "chart_to_show": None,
            "user_justifications": {} # Para guardar las justificaciones del usuario
        }
    # Activa el escenario recién creado.
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
    initialize_global_state()


# --- 3) FUNCIONES DE CACHÉ Y LÓGICA DE IA (Sin cambios significativos) ---

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

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("La clave de API de OpenAI no ha sido configurada.")
        st.info("Añade tu clave 'OPENAI_API_KEY' en los 'Secrets' de la app y refresca la página.")
        st.stop()
    return OpenAI(api_key=api_key)

def chat_completion(prompt: str, use_json_mode: bool = False):
    client = get_openai_client()
    params = {"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}], "temperature": 0.5}
    if use_json_mode:
        params["response_format"] = {"type": "json_object"}
    return client.chat.completions.create(**params)

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
        content = resp.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        return {"error": f"Error al procesar la respuesta de la IA: {str(e)}", "raw_content": content}

# --- 4) COMPONENTES DE LA UI ---

def render_scenario_navigator():
    st.sidebar.title("Scenario Navigator")
    st.sidebar.markdown("Gestiona y compara tus modelos metodológicos.")
    st.sidebar.divider()

    if not st.session_state.scenarios:
        st.sidebar.info("Carga un fichero de datos para empezar a crear escenarios.")
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
    if st.sidebar.button("➕ Nuevo Escenario", help="Crea un nuevo modelo desde cero con los mismos datos."):
        create_new_scenario(name=f"Nuevo Modelo {len(st.session_state.scenarios) + 1}")
        st.rerun()

    if st.sidebar.button("📋 Clonar Escenario Actual", help="Crea una copia de este escenario para probar variaciones."):
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
        if st.sidebar.button("🗑️ Eliminar Escenario Actual", type="primary"):
            del st.session_state.scenarios[st.session_state.active_scenario_id]
            st.session_state.active_scenario_id = next(iter(st.session_state.scenarios))
            st.rerun()

def render_comparison_view():
    st.header("Comparador de Escenarios Metodológicos")
    st.info("Selecciona dos escenarios para comparar sus resultados, configuraciones y calidad de razonamiento.")

    if len(st.session_state.scenarios) < 2:
        st.warning("Necesitas al menos dos escenarios para poder comparar.")
        return

    col1, col2 = st.columns(2)
    scenario_names = {sid: s['name'] for sid, s in st.session_state.scenarios.items()}

    with col1:
        id_a = st.selectbox("Comparar Escenario A:", list(st.session_state.scenarios.keys()), format_func=lambda sid: scenario_names[sid], key='compare_a')
    with col2:
        options_b = [sid for sid in st.session_state.scenarios.keys() if sid != id_a] or [id_a]
        id_b = st.selectbox("Con Escenario B:", options_b, format_func=lambda sid: scenario_names[sid], key='compare_b')

    st.divider()
    scenario_a = st.session_state.scenarios.get(id_a)
    scenario_b = st.session_state.scenarios.get(id_b)

    if not scenario_a or not scenario_b: return

    res_col1, res_col2 = st.columns(2)
    for sc, col in [(scenario_a, res_col1), (scenario_b, res_col2)]:
        with col:
            st.subheader(f"Resultados de: {sc['name']}")
            with st.container(border=True):
                if sc.get('dea_results'):
                    st.markdown("**Configuración:**"); st.json(sc.get('dea_config', {}))
                    st.markdown("**Resultados (Top 5):**"); st.dataframe(sc['dea_results']['main_df'].head())
                    if sc.get('inquiry_tree'):
                        eee_metrics = compute_eee(sc['inquiry_tree'], depth_limit=3, breadth_limit=5)
                        st.metric("Calidad del Juicio (EEE)", f"{eee_metrics['score']:.2%}")
                else:
                    st.info("Escenario no calculado.")

def render_eee_explanation(eee_metrics: dict):
    st.info(f"**Calidad del Juicio Metodológico (EEE): {eee_metrics['score']:.2%}**")
    def interpret_score(name, score):
       if score >= 0.8: return f"**{name}:** Tu puntuación es **excelente** ({score:.0%})."
       if score >= 0.5: return f"**{name}:** Tu puntuación es **buena** ({score:.0%})."
       return f"**{name}:** Tu puntuación es **baja** ({score:.0%})."
    with st.expander("Ver desglose y consejos"):
        st.markdown(f"""
        - {interpret_score("Profundidad (D1)", eee_metrics['D1'])}: Mide si el análisis metodológico es profundo.
        - {interpret_score("Pluralidad (D2)", eee_metrics['D2'])}: Valora si se han considerado múltiples riesgos o facetas metodológicas.
        - {interpret_score("Robustez (D5)", eee_metrics['D5'])}: Evalúa la solidez y el detalle del árbol de preguntas.
        """)

# --- 5) COMPONENTES DE LA UI MEJORADOS ---

def render_interactive_inquiry_tree(active_scenario):
    """Muestra el árbol de preguntas y captura las justificaciones del usuario."""
    st.subheader("Taller de Auditoría Metodológica", anchor=False)
    st.info("Utiliza este espacio para responder a las preguntas de auditoría de la IA y documentar tu razonamiento. Este será el núcleo de tu informe metodológico.")
    
    tree = active_scenario.get("inquiry_tree")
    if not tree:
        st.warning("Aún no se ha generado un mapa metodológico. Haz clic en el botón de arriba para crearlo.")
        return

    def render_node_with_justification(node_dict, level=0):
        """Función recursiva para renderizar cada nodo del árbol con su campo de texto."""
        for question, children in node_dict.items():
            st.markdown(f"<div style='margin-left: {level*20}px; border-left: 2px solid #ccc; padding-left: 10px; margin-top: 10px;'>"
                        f"<b>Pregunta de Auditoría:</b> {question}"
                        f"</div>", unsafe_allow_html=True)
            
            justification_key = f"just_{st.session_state.active_scenario_id}_{question}"
            current_justification = active_scenario['user_justifications'].get(question, "")
            
            user_input = st.text_area(
                "Tu justificación metodológica:",
                value=current_justification,
                key=justification_key,
                label_visibility="collapsed",
                placeholder="Escribe aquí tu razonamiento, citando literatura si es necesario..."
            )
            
            active_scenario['user_justifications'][question] = user_input
            
            if isinstance(children, dict) and children:
                render_node_with_justification(children, level + 1)

    render_node_with_justification(tree)

def render_deliberation_workshop(active_scenario):
    if not active_scenario.get('dea_results'): return
    
    st.header("Paso 4: Deliberación y Justificación Metodológica", divider="blue")
    
    with st.container(border=True):
        st.subheader("Mapa de Auditoría (IA)", anchor=False)
        root_question_methodology = (
            f"Para un modelo DEA con enfoque '{active_scenario['selected_proposal']['title']}', "
            f"inputs {active_scenario['selected_proposal']['inputs']} y "
            f"outputs {active_scenario['selected_proposal']['outputs']}, "
            "¿cuáles son los principales desafíos metodológicos y las mejores prácticas para asegurar la robustez del análisis?"
        )
        if st.button("Generar Mapa Metodológico", use_container_width=True, key=f"gen_map_{st.session_state.active_scenario_id}"):
            with st.spinner("La IA está generando un árbol de auditoría..."):
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
            with st.expander("Ver visualización del árbol y explicación de la IA"):
                st.plotly_chart(to_plotly_tree(active_scenario['inquiry_tree']), use_container_width=True)
                # ... Lógica para explicación de la IA ...
            eee_metrics = compute_eee(active_scenario['inquiry_tree'], depth_limit=3, breadth_limit=5)
            render_eee_explanation(eee_metrics)

    st.divider()
    
    # Nuevo componente interactivo para las justificaciones
    render_interactive_inquiry_tree(active_scenario)

def render_download_section(active_scenario):
    # (Esta función se usará en el siguiente paso para generar el informe con las justificaciones)
    # Por ahora, se mantiene sin cambios.
    results = active_scenario.get('dea_results')
    if not results: return

    st.subheader("Exportar Análisis del Escenario", divider="gray")
    st.markdown(f"Descarga los resultados y el informe deliberativo para el escenario **'{active_scenario['name']}'**.")
    col1, col2 = st.columns(2)
    with col1:
        # Aquí se pasará user_justifications en el futuro
        html_report = generate_html_report(analysis_results=results, inquiry_tree=active_scenario.get("inquiry_tree"))
        st.download_button(label="Descargar Informe en HTML", data=html_report, file_name=f"report_{active_scenario['name'].replace(' ', '_')}.html", mime="text/html", use_container_width=True, key=f"html_dl_{st.session_state.active_scenario_id}")
    with col2:
        excel_report = generate_excel_report(analysis_results=results, inquiry_tree=active_scenario.get("inquiry_tree"))
        st.download_button(label="Descargar Informe en Excel", data=excel_report, file_name=f"report_{active_scenario['name'].replace(' ', '_')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, key=f"excel_dl_{st.session_state.active_scenario_id}")


def render_main_dashboard(active_scenario):
    # (El resto de funciones de renderizado se mantienen, operando sobre `active_scenario`)
    st.header(f"Paso 3: Configuración y Análisis para '{active_scenario['name']}'", divider="blue")
    st.markdown(f"**Enfoque seleccionado:** *{active_scenario['selected_proposal'].get('title', 'N/A')}*")
    
    model_options = {"Radial (CCR/BCC)": "CCR_BCC", "No Radial (SBM)": "SBM", "Productividad (Malmquist)": "MALMQUIST"}
    
    # Recuperar la selección actual del estado o usar un valor por defecto
    current_model_key = active_scenario['dea_config'].get('model', 'CCR_BCC')
    current_model_name = [name for name, key in model_options.items() if key == current_model_key][0]
    
    model_name = st.selectbox("1. Selecciona el tipo de modelo DEA:", list(model_options.keys()), index=list(model_options.values()).index(current_model_key), key=f"model_select_{st.session_state.active_scenario_id}")
    model_key = model_options[model_name]
    
    active_scenario['dea_config']['model'] = model_key

    period_col = None
    if model_key == 'MALMQUIST':
        period_col_options = [None] + active_scenario['df'].columns.tolist()
        period_col = st.selectbox("2. Selecciona la columna de período:", period_col_options, index=1, key=f"period_col_{st.session_state.active_scenario_id}")
        if not period_col: st.warning("El modelo Malmquist requiere una columna de período."); st.stop()
        active_scenario['dea_config']['period_col'] = period_col
    
    if st.button(f"Ejecutar Análisis", type="primary", use_container_width=True, key=f"run_{st.session_state.active_scenario_id}"):
        with st.spinner(f"Ejecutando {model_name} para '{active_scenario['name']}'..."):
            df = active_scenario['df']
            proposal = active_scenario['selected_proposal']
            try:
                results = cached_run_dea_analysis(df, df.columns[0], proposal.get('inputs', []), proposal.get('outputs', []), model_key, period_col)
                active_scenario['dea_results'] = results
                active_scenario['app_status'] = "results_ready"
            except Exception as e:
                st.error(f"Error durante el análisis: {e}")
                active_scenario['dea_results'] = None
        st.rerun()

    if active_scenario.get("dea_results"):
        results = active_scenario["dea_results"]
        st.header(f"Resultados para: {results['model_name']}", divider="blue")
        st.dataframe(results['main_df'])
        if results.get("charts"):
            for chart_title, fig in results["charts"].items():
                st.plotly_chart(fig, use_container_width=True)
        
        # El taller de deliberación y la descarga se muestran aquí
        render_deliberation_workshop(active_scenario)
        render_download_section(active_scenario)

def render_validation_step(active_scenario):
    #... (sin cambios importantes)
    st.header(f"Paso 2b: Validación del Modelo para '{active_scenario['name']}'", divider="gray")
    proposal = active_scenario.get('selected_proposal')
    if not proposal or not proposal.get('inputs') or not proposal.get('outputs'):
        st.error("La propuesta de análisis de este escenario está incompleta. Por favor, vuelve al paso anterior o elige otro escenario.")
        return

    with st.spinner("La IA está validando la coherencia de los datos y el modelo..."):
        validation_results = validate_data(active_scenario['df'], proposal['inputs'], proposal['outputs'])
    
    if st.button("Proceder al Análisis", key=f"validate_{st.session_state.active_scenario_id}"):
        active_scenario['app_status'] = "validated"
        st.rerun()

def render_proposal_step(active_scenario):
    #... (sin cambios importantes)
    st.header(f"Paso 2: Elige un Enfoque de Análisis para '{active_scenario['name']}'", divider="blue")
    
    if not active_scenario.get('proposals_data'):
        with st.spinner("La IA está analizando tus datos para sugerir enfoques..."):
            active_scenario['proposals_data'] = cached_get_analysis_proposals(active_scenario['df'])
    
    proposals_data = active_scenario['proposals_data']
    proposals = proposals_data.get("proposals", [])
    st.info("La IA ha preparado varios enfoques. Elige uno para este escenario.")
    for i, proposal in enumerate(proposals):
        with st.expander(f"**{proposal.get('title', 'N/A')}**"):
            st.markdown(f"_{proposal.get('reasoning', '')}_")
            if st.button(f"Seleccionar", key=f"select_{i}_{st.session_state.active_scenario_id}"):
                if proposal.get('inputs') and proposal.get('outputs'):
                    active_scenario['selected_proposal'] = proposal
                    active_scenario['app_status'] = "proposal_selected"
                    st.rerun()
                else:
                    st.error("Propuesta incompleta.")

def render_upload_step():
    st.header("Paso 1: Carga tus Datos para Iniciar la Sesión", divider="blue")
    st.info("Sube un fichero CSV. Este fichero será la base para todos tus escenarios de análisis.")
    uploaded_file = st.file_uploader("Sube un fichero CSV", type=["csv"], label_visibility="collapsed")
    
    if uploaded_file:
        try:
            df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')))
        except Exception:
            df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('latin-1')), sep=';')
        
        st.session_state.global_df = df
        create_new_scenario(name="Modelo Base")
        st.rerun()

# --- 6) FLUJO PRINCIPAL DE LA APLICACIÓN (REFACTORIZADO) ---
def main():
    """Función principal que orquesta la aplicación multi-escenario."""
    if 'scenarios' not in st.session_state:
        initialize_global_state()

    st.sidebar.image("https://i.imgur.com/8y0N5c5.png", width=200)
    st.sidebar.title("DEA Deliberative Modeler")
    if st.sidebar.button("🔴 Empezar Nueva Sesión", help="Borra todos los datos y escenarios."):
        reset_all()
        st.rerun()
    st.sidebar.divider()
    
    render_scenario_navigator()

    st.sidebar.markdown("---")
    st.sidebar.info("Una herramienta para el análisis de eficiencia y la deliberación metodológica asistida por IA.")

    active_scenario = get_active_scenario()

    if not active_scenario:
        render_upload_step()
    else:
        analysis_tab, comparison_tab = st.tabs(["Análisis del Escenario Activo", "Comparar Escenarios"])

        with analysis_tab:
            app_status = active_scenario.get('app_status', 'initial')
            if app_status == "file_loaded":
                render_proposal_step(active_scenario)
            elif app_status == "proposal_selected":
                render_validation_step(active_scenario)
            elif app_status in ["validated", "results_ready"]:
                render_main_dashboard(active_scenario)
        
        with comparison_tab:
            render_comparison_view()

if __name__ == "__main__":
    main()
