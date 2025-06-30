import sys
import os
import pandas as pd
import streamlit as st
import io
import json
import uuid
import openai
import plotly.express as px

# --- 0) AJUSTE DEL PYTHONPATH Y CONFIGURACIÓN INICIAL ---
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Configuración de la página de Streamlit. "wide" aprovecha mejor el espacio.
st.set_page_config(layout="wide", page_title="DEA Deliberative Modeler")

# --- 1) IMPORTACIONES DE MÓDULOS DEL PROYECTO ---
from analysis_dispatcher import execute_analysis
from inquiry_engine import generate_inquiry, to_plotly_tree
from epistemic_metrics import compute_eee
from data_validator import validate as validate_data
from report_generator import generate_html_report, generate_excel_report
from dea_models.visualizations import plot_hypothesis_distribution, plot_correlation
from dea_models.auto_tuner import generate_candidates, evaluate_candidates
from openai_helpers import explain_inquiry_tree

# --- 2) GESTIÓN DE ESTADO MULTI-ESCENARIO ---

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
    st.cache_data.clear()
    st.cache_resource.clear()

    for key in list(st.session_state.keys()):
        del st.session_state[key]

    initialize_global_state()


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


# --- CLASE ENCAPSULADORA DE FUNCIONES DE RENDERIZADO DE LA UI ---
class AppRenderer:
    def __init__(self):
        pass

    def render_scenario_navigator(self):
        st.sidebar.title("Navegador de Escenarios")
        st.sidebar.markdown("Gestiona y compara tus diferentes modelos y análisis.")
        st.sidebar.divider()

        if not st.session_state.scenarios:
            st.sidebar.info("Carga datos para empezar a crear escenarios de análisis.")
            return

        scenario_names = {sid: s['name'] for sid, s in st.session_state.scenarios.items()}
        active_id = st.session_state.get('active_scenario_id')

        if active_id not in scenario_names:
            active_id = next(iter(scenario_names)) if scenario_names else None

        st.session_state.active_scenario_id = st.sidebar.selectbox(
            "Escenario Activo",
            options=list(st.session_state.scenarios.keys()),
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

        active_scenario_from_state = get_active_scenario()
        if active_scenario_from_state:
            new_name = st.sidebar.text_input("Renombrar escenario:", value=active_scenario_from_state['name'], key=f"rename_{st.session_state.active_scenario_id}")
            if new_name != active_scenario_from_state['name']:
                active_scenario_from_state['name'] = new_name
                st.rerun()

        st.sidebar.divider()
        if len(st.session_state.scenarios) > 1:
            if st.sidebar.button("🗑️ Eliminar Escenario Actual", type="primary"):
                del st.session_state.scenarios[st.session_state.active_scenario_id]
                if st.session_state.scenarios:
                    st.session_state.active_scenario_id = next(iter(st.session_state.scenarios.keys()))
                else:
                    reset_all()
                st.rerun()

    # (El resto de tus funciones de renderizado como render_comparison_view, render_eee_explanation, etc. van aquí sin cambios)
    # ...
    # PEGA AQUÍ EL RESTO DE TUS FUNCIONES DE LA CLASE AppRenderer SIN MODIFICAR
    # ...
    
    # MODIFICACIÓN DE LA FUNCIÓN render_upload_step
    def render_upload_step(self):
        st.header("Paso 1: Carga tus Datos para Iniciar la Sesión", divider="blue")
        st.info("Para comenzar, selecciona una fuente de datos. Puedes subir tu propio archivo CSV o utilizar uno de nuestros casos de estudio para empezar rápidamente.")

        source_option = st.radio(
            "Elige una fuente de datos:",
            ('Usar un caso de estudio', 'Subir un archivo CSV'),
            horizontal=True,
            label_visibility="collapsed"
        )
        
        df = None
        uploaded_file_name = None

        if source_option == 'Usar un caso de estudio':
            datasets_path = 'datasets/'
            try:
                available_datasets = [f for f in os.listdir(datasets_path) if f.endswith('.csv')]
                if not available_datasets:
                    st.warning("No se encontraron datasets en la carpeta `datasets`.")
                    return
            except FileNotFoundError:
                st.error("La carpeta `datasets` no existe. Por favor, créala y añade los archivos CSV.")
                return

            selected_dataset = st.selectbox(
                'Selecciona un caso de estudio:',
                available_datasets
            )

            if selected_dataset:
                try:
                    file_path = os.path.join(datasets_path, selected_dataset)
                    df = pd.read_csv(file_path)
                    uploaded_file_name = selected_dataset
                except Exception as e:
                    st.error(f"Error al cargar el dataset '{selected_dataset}': {e}")
                    return

        elif source_option == 'Subir un archivo CSV':
            uploaded_file = st.file_uploader(
                "Sube tu archivo CSV",
                type=["csv"],
                label_visibility="collapsed"
            )
            if uploaded_file:
                try:
                    df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')))
                except Exception:
                    df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('latin-1')), sep=';')
                uploaded_file_name = uploaded_file.name

        if df is not None:
            st.session_state.global_df = df
            create_new_scenario(name="Modelo Base")
            active_scenario = get_active_scenario()
            data_overview = {
                "shape": df.shape,
                "column_types": df.dtypes.astype(str).to_dict(),
                "numerical_summary": df.describe(include='number').to_dict(),
                "null_counts": df.isnull().sum().to_dict(),
                "non_numeric_issues": {}
            }
            # ... (resto de tu lógica de data_overview)
            active_scenario['data_overview'] = data_overview
            active_scenario['app_status'] = "data_loaded"
            st.success(f"Datos de '{uploaded_file_name}' cargados. Continúa con la exploración preliminar.")
            st.rerun()

        if st.session_state.get('global_df') is not None:
            active_scenario = get_active_scenario()
            if active_scenario and active_scenario.get('data_overview'):
                # (El código para mostrar el informe rápido de datos va aquí)
                pass # Tu lógica existente para mostrar data_overview


# --- FUNCIÓN PRINCIPAL DE LA APLICACIÓN ---
def main():
    """Función principal que orquesta la aplicación multi-scenario."""
    initialize_global_state()

    # Asegúrate de tener una carpeta "assets" con "logo.png"
    st.sidebar.image("assets/logo.png", width=200) 
    st.sidebar.title("DEA Deliberative Modeler")
    st.sidebar.markdown("Una herramienta para el análisis de eficiencia y la deliberación metodológica asistida por IA.")
    if st.sidebar.button("🔴 Empezar Nueva Sesión"):
        reset_all()
        st.rerun()
    st.sidebar.divider()

    renderer = AppRenderer()
    renderer.render_scenario_navigator()

    st.sidebar.markdown("---")
    st.sidebar.info("Sigue los pasos en la pestaña principal para un estudio DEA robusto.")

    active_scenario = get_active_scenario()

    analysis_tab, comparison_tab, challenges_tab = st.tabs([
        "Análisis del Escenario Activo",
        "Comparar Escenarios",
        "Retos del DEA"
    ])

    with analysis_tab:
        app_status = active_scenario.get('app_status', 'initial') if active_scenario else 'initial'

        if '_new_scenario_requested' in st.session_state and st.session_state._new_scenario_requested:
            create_new_scenario(name=f"Nuevo Modelo {len(st.session_state.scenarios) + 1}")
            del st.session_state._new_scenario_requested
            st.rerun()
        if '_clone_scenario_requested' in st.session_state and st.session_state._clone_scenario_requested:
            create_new_scenario(source_scenario_id=st.session_state._clone_scenario_requested)
            del st.session_state._clone_scenario_requested
            st.rerun()

        if app_status == "initial":
            renderer.render_upload_step()
        elif app_status == "data_loaded":
            renderer.render_preliminary_analysis_step(active_scenario)
        elif app_status == "file_loaded":
            renderer.render_proposal_step(active_scenario)
        elif app_status == "proposal_selected":
            renderer.render_validation_step(active_scenario)
        elif app_status in ["validated", "results_ready"]:
            renderer.render_main_dashboard(active_scenario)

    with comparison_tab:
        renderer.render_comparison_view()

    with challenges_tab:
        renderer.render_dea_challenges_tab()


if __name__ == "__main__":
    main()
