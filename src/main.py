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
    for key in list(st.session_state.keys()):
        if not key.startswith('_'):
            del st.session_state[key]
    st.session_state.app_status = "initial"

if 'app_status' not in st.session_state:
    initialize_state()

# --- 3) FUNCIONES DE IA Y CACHÉ ---

# Las funciones conflictivas se mueven aquí para romper el ciclo de importación
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat_completion(prompt: str, use_json_mode: bool = False):
    params = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
    }
    if use_json_mode:
        params["response_format"] = {"type": "json_object"}
    return client.chat.completions.create(**params)

def generate_analysis_proposals(df_columns: list[str], df_head: pd.DataFrame):
    prompt = (
        f"Eres un consultor experto en Data Envelopment Analysis (DEA). Has recibido un conjunto de datos con las siguientes columnas: {df_columns}. A continuación se muestran las primeras filas:\n\n{df_head.to_string()}\n\n"
        "Tu tarea es proponer entre 2 y 4 modelos de análisis DEA distintos y bien fundamentados. Para cada propuesta, proporciona un título, un breve razonamiento, y las listas de inputs y outputs sugeridas.\n\n"
        "Devuelve únicamente un objeto JSON válido con una sola clave raíz 'proposals', que sea una lista de objetos, donde cada objeto contiene 'title', 'reasoning', 'inputs' y 'outputs'."
    )
    try:
        resp = chat_completion(prompt, use_json_mode=True)
        content = resp.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        return {"error": str(e), "text": "No se pudieron generar las propuestas."}

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

# --- 4) COMPONENTES MODULARES DE LA UI (sin cambios, solo se pega el código) ---

def render_eee_explanation(eee_metrics: dict):
    st.info(f"**Calidad del Razonamiento (EEE): {eee_metrics['score']:.2%}**")
    def interpret_score(name, score):
        if score >= 0.8: return f"**{name}:** Tu puntuación es **excelente** ({score:.0%})."
        if score >= 0.5: return f"**{name}:** Tu puntuación es **buena** ({score:.0%})."
        return f"**{name}:** Tu puntuación es **baja** ({score:.0%}), indicando un área de mejora."
    with st.expander("Ver desglose y consejos"):
        st.markdown(f"""
        - {interpret_score("Profundidad (D1)", eee_metrics['D1'])}
          - *Consejo:* Si es baja, elige una causa y vuelve a generar un mapa sobre ella para profundizar.
        - {interpret_score("Pluralidad (D2)", eee_metrics['D2'])}
          - *Consejo:* Si es baja, inspírate con un nuevo mapa para considerar más hipótesis iniciales.
        - {interpret_score("Robustez (D5)", eee_metrics['D5'])}
          - *Consejo:* Si es baja, asegúrate de que tu mapa descomponga las ideas principales en sub-causas.
        """)

def render_deliberation_workshop(results):
    st.header("Paso 4: Razona y Explora las Causas con IA", divider="blue")
    # ... (El resto del código de esta función y las demás se mantiene igual)
    # Aquí iría el código completo de las funciones render_* que ya te he proporcionado antes.
    # Por brevedad, no lo repito, pero debes asegurarte de que están todas aquí.
    pass

def render_download_section(results):
    pass # Pega aquí el código de esta función

def render_main_dashboard():
    pass # Pega aquí el código de esta función

def render_validation_step():
    pass # Pega aquí el código de esta función

def render_proposal_step():
    st.header("Paso 2: Elige un Enfoque de Análisis", divider="blue")
    if 'proposals' not in st.session_state:
        with st.spinner("La IA está analizando tus datos..."):
            st.session_state.proposals = cached_get_analysis_proposals(st.session_state.df).get("proposals", [])
    if not st.session_state.get("proposals"):
        st.error("La IA no pudo generar propuestas."); st.stop()
    st.info("La IA ha preparado varios enfoques. Elige el que mejor se adapte a tu objetivo.")
    for i, proposal in enumerate(st.session_state.get("proposals", [])):
        with st.expander(f"**Propuesta {i+1}: {proposal['title']}**", expanded=i==0):
            st.markdown(f"**Razonamiento:** *{proposal['reasoning']}*")
            st.markdown(f"**Inputs sugeridos:** `{proposal['inputs']}`")
            st.markdown(f"**Outputs sugeridos:** `{proposal['outputs']}`")
            if st.button(f"Seleccionar: {proposal['title']}", key=f"select_{i}"):
                st.session_state.selected_proposal = proposal
                st.session_state.app_status = "proposal_selected"; st.rerun()

def render_upload_step():
    pass # Pega aquí el código de esta función

# --- 5) FLUJO PRINCIPAL DE LA APLICACIÓN ---
def main():
    st.sidebar.title("DEA Deliberativo")
    if st.sidebar.button("Empezar de Nuevo"):
        initialize_state(); st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.info("Análisis de eficiencia y deliberación estratégica con IA.")
    if st.session_state.app_status == "initial": render_upload_step()
    elif st.session_state.app_status == "file_loaded": render_proposal_step()
    elif st.session_state.app_status == "proposal_selected": render_validation_step()
    elif st.session_state.app_status in ["validated", "results_ready"]: render_main_dashboard()

if __name__ == "__main__":
    main()
