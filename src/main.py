import sys
import os
import pandas as pd
import streamlit as st
import io
import json
from openai import OpenAI

# --- 0) AJUSTE DEL PYTHONPATH Y CONFIGURACIÓN INICIAL ---
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
st.set_page_config(layout="wide", page_title="DEA Deliberativo con IA")

# --- 1) IMPORTACIONES DE MÓDULOS ---
from analysis_dispatcher import execute_analysis
from inquiry_engine import generate_inquiry, to_plotly_tree
from epistemic_metrics import compute_eee
from data_validator import validate as validate_data
from report_generator import generate_html_report, generate_excel_report
from dea_models.visualizations import plot_hypothesis_distribution, plot_correlation
from openai_helpers import explain_inquiry_tree

# --- 2) GESTIÓN DE ESTADO ---
def initialize_state():
    """Reinicia de forma segura el estado de la sesión a sus valores iniciales."""
    st.session_state.app_status = "initial"
    st.session_state.df = None
    st.session_state.proposals_data = None
    st.session_state.selected_proposal = None
    st.session_state.dea_results = None
    st.session_state.inquiry_tree = None
    st.session_state.tree_explanation = None
    st.session_state.chart_to_show = None

if 'app_status' not in st.session_state:
    initialize_state()

# --- 3) FUNCIONES DE IA Y CACHÉ (CON INICIALIZACIÓN SEGURA) ---

def get_openai_client():
    """Inicializa de forma segura el cliente de OpenAI, mostrando un error si la clave no existe."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("La clave de API de OpenAI no ha sido configurada.")
        st.info("Por favor, añade tu clave 'OPENAI_API_KEY' en la sección de 'Secrets' de la configuración de tu aplicación en Streamlit Community Cloud.")
        st.stop()
    return OpenAI(api_key=api_key)

def chat_completion(prompt: str, use_json_mode: bool = False):
    client = get_openai_client() # Inicialización segura
    params = {"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}], "temperature": 0.5}
    if use_json_mode:
        params["response_format"] = {"type": "json_object"}
    return client.chat.completions.create(**params)

def generate_analysis_proposals(df_columns: list[str], df_head: pd.DataFrame):
    prompt = (
        f"Eres un consultor experto en Data Envelopment Analysis (DEA)...\n" # Prompt abreviado por claridad
        "Devuelve únicamente un objeto JSON válido con una sola clave raíz 'proposals'..."
    )
    content = "No se recibió contenido."
    try:
        resp = chat_completion(prompt, use_json_mode=True)
        content = resp.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        return {"error": f"Error al procesar la respuesta de la IA: {str(e)}", "raw_content": content}

@st.cache_data
def cached_get_analysis_proposals(_df):
    return generate_analysis_proposals(_df.columns.tolist(), _df.head())

# ... (Resto de funciones de caché sin cambios) ...
@st.cache_data
def cached_run_dea_analysis(_df, dmu_col, input_cols, output_cols, model_key, period_col):
    return execute_analysis(_df.copy(), dmu_col, input_cols, output_cols, model_key, period_column=period_col)

@st.cache_data
def cached_run_inquiry_engine(root_question, _context):
    return generate_inquiry(root_question, context=_context)

@st.cache_data
def cached_explain_tree(_tree):
    return explain_inquiry_tree(_tree)

# --- 4) COMPONENTES MODULARES DE LA UI (SIN CAMBIOS EN SU LÓGICA INTERNA) ---
# (Pega aquí las funciones render_* completas de la respuesta anterior)

def render_eee_explanation(eee_metrics: dict):
    # ... (código completo de la función)
    pass

def render_deliberation_workshop(results):
    # ... (código completo de la función)
    pass

def render_download_section(results):
    # ... (código completo de la función)
    pass

def render_main_dashboard():
    # ... (código completo de la función)
    pass

def render_validation_step():
    # ... (código completo de la función)
    pass

def render_proposal_step():
    # ... (código completo de la función)
    pass

def render_upload_step():
    # ... (código completo de la función)
    pass

# --- 5) FLUJO PRINCIPAL DE LA APLICACIÓN ---
def main():
    st.sidebar.title("DEA Deliberativo")
    if st.sidebar.button("Empezar de Nuevo"):
        initialize_state()
        st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.info("Una herramienta para el análisis de eficiencia y la deliberación estratégica con asistencia de IA.")

    # Máquina de estados que controla qué se renderiza en la pantalla
    if st.session_state.app_status == "initial":
        render_upload_step()
    elif st.session_state.app_status == "file_loaded":
        render_proposal_step()
    elif st.session_state.app_status == "proposal_selected":
        render_validation_step()
    elif st.session_state.app_status in ["validated", "results_ready"]:
        render_main_dashboard()

if __name__ == "__main__":
    main()
