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
# Se importan los módulos principales de la aplicación.
from analysis_dispatcher import execute_analysis
from inquiry_engine import generate_inquiry, to_plotly_tree
from epistemic_metrics import compute_eee
from data_validator import validate as validate_data
from report_generator import generate_html_report, generate_excel_report
from dea_models.visualizations import plot_hypothesis_distribution, plot_correlation
from openai_helpers import explain_inquiry_tree

# --- 2) GESTIÓN DE ESTADO ---
def initialize_state():
    """
    Reinicia de forma segura el estado de la sesión a sus valores iniciales.
    Este método es más seguro que borrar claves indiscriminadamente.
    """
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

# --- 3) FUNCIONES DE IA Y CACHÉ ---
# Las funciones que causaban el conflicto de importación se definen aquí directamente.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat_completion(prompt: str, use_json_mode: bool = False):
    """Llamada genérica a OpenAI Chat Completion, ahora local en main.py."""
    params = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
    }
    if use_json_mode:
        params["response_format"] = {"type": "json_object"}
    return client.chat.completions.create(**params)

def generate_analysis_proposals(df_columns: list[str], df_head: pd.DataFrame):
    """
    Analiza las columnas y propone modelos DEA.
    Esta función ahora es local en main.py para evitar el ImportError.
    """
    prompt = (
        f"Eres un consultor experto en Data Envelopment Analysis (DEA). Has recibido un conjunto de datos con las siguientes columnas: {df_columns}. A continuación se muestran las primeras filas:\n\n{df_head.to_string()}\n\n"
        "Tu tarea es proponer entre 2 y 4 modelos de análisis DEA distintos y bien fundamentados. Para cada propuesta, proporciona un título, un breve razonamiento, y las listas de inputs y outputs sugeridas.\n\n"
        "Devuelve únicamente un objeto JSON válido con una sola clave raíz 'proposals', que sea una lista de objetos, donde cada objeto contiene 'title', 'reasoning', 'inputs' y 'outputs'."
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
    """Función de caché que ahora llama a la función local."""
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

# --- 4) COMPONENTES MODULARES DE LA UI ---

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
    col_map, col_workbench = st.columns([2, 1])
    with col_map:
        st.subheader("Mapa de Razonamiento (IA)", anchor=False)
        if st.button("Generar/Inspirar con nuevo Mapa de Razonamiento", use_container_width=True):
            with st.spinner("La IA está generando un mapa de ideas..."):
                main_df = results.get('main_df', pd.DataFrame())
                num_efficient = 0
                if not main_df.empty and len(main_df.columns) > 1:
                    num_efficient = int((main_df.iloc[:, 1] >= 0.999).sum())
                context = {
                    "model": results.get("model_name"),
                    "inputs": st.session_state.selected_proposal['inputs'],
                    "outputs": st.session_state.selected_proposal['outputs'],
                    "num_efficient_dmus": num_efficient
                }
                root_question = f"Bajo el enfoque '{st.session_state.selected_proposal['title']}', ¿cuáles son las posibles causas de la ineficiencia observada?"
                tree, error = cached_run_inquiry_engine(root_question, context)
                if error: st.error(f"Error al generar el mapa: {error}")
                st.session_state.inquiry_tree = tree
                st.session_state.tree_explanation = None

        if st.session_state.get("inquiry_tree"):
            if not st.session_state.get("tree_explanation"):
                with st.spinner("La IA está interpretando el mapa para ti..."):
                    explanation_result = cached_explain_tree(st.session_state.inquiry_tree)
                    st.session_state.tree_explanation = explanation_result
            if st.session_state.get("tree_explanation"):
                explanation = st.session_state.tree_explanation
                with st.container(border=True):
                    st.markdown(explanation.get("text", "No se pudo generar la explicación."))
            st.plotly_chart(to_plotly_tree(st.session_state.inquiry_tree), use_container_width=True)
            eee_metrics = compute_eee(st.session_state.inquiry_tree, depth_limit=3, breadth_limit=5)
            render_eee_explanation(eee_metrics)

    with col_workbench:
        st.subheader("Taller de Hipótesis (Usuario)", anchor=False)
        st.info("Usa este taller para explorar tus propias hipótesis.")

def render_download_section(results):
    st.subheader("Exportar Análisis Completo", divider="gray")
    col1, col2 = st.columns(2)
    with col1:
        html_report = generate_html_report(analysis_results=results, inquiry_tree=st.session_state.get("inquiry_tree"))
        st.download_button(
            label="Descargar Informe en HTML", data=html_report,
            file_name=f"reporte_dea_{results.get('model_name', 'विश्लेषण').replace(' ', '_').lower()}.html",
            mime="text/html", use_container_width=True
        )
    with col2:
        excel_report = generate_excel_report(analysis_results=results, inquiry_tree=st.session_state.get("inquiry_tree"))
        st.download_button(
            label="Descargar Informe en Excel", data=excel_report,
            file_name=f"reporte_dea_{results.get('model_name', 'विश्लेषण').replace(' ', '_').lower()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True
        )

def render_main_dashboard():
    st.header("Paso 3: Configuración y Ejecución del Análisis", divider="blue")
    st.markdown(f"**Enfoque seleccionado:** *{st.session_state.selected_proposal['title']}*")
    model_options = {"Radial (CCR/BCC)": "CCR_BCC", "No Radial (SBM)": "SBM", "Productividad (Malmquist)": "MALMQUIST"}
    model_name = st.selectbox("1. Selecciona el tipo de modelo DEA a aplicar:", list(model_options.keys()))
    model_key = model_options[model_name]
    period_col = None
    if model_key == 'MALMQUIST':
        period_col_options = [None] + st.session_state.df.columns.tolist()
        period_col = st.selectbox("2. Selecciona la columna que identifica el período:", period_col_options, index=1)
        if not period_col: st.warning("El modelo Malmquist requiere una columna de período."); st.stop()
    if st.button(f"Ejecutar Análisis con Modelo: {model_name}", type="primary", use_container_width=True):
        with st.spinner(f"Ejecutando {model_name}..."):
            df = st.session_state.df
            proposal = st.session_state.selected_proposal
            try:
                st.session_state.dea_results = cached_run_dea_analysis(df, df.columns[0], proposal['inputs'], proposal['outputs'], model_key, period_col)
                st.session_state.app_status = "results_ready"
            except Exception as e:
                st.error(f"Error durante el análisis: {e}"); st.session_state.dea_results = None
    if st.session_state.get("dea_results"):
        results = st.session_state.dea_results
        st.header(f"Resultados para: {results['model_name']}", divider="blue")
        st.dataframe(results['main_df'])
        if results.get("charts"):
            for chart_title, fig in results["charts"].items():
                st.plotly_chart(fig, use_container_width=True)
        render_download_section(results)
        render_deliberation_workshop(results)

def render_validation_step():
    st.header("Paso 2b: Validación del Modelo", divider="gray")
    proposal = st.session_state.selected_proposal
    with st.spinner("La IA está validando la coherencia de los datos y el modelo..."):
        validation_results = validate_data(st.session_state.df, proposal['inputs'], proposal['outputs'])
    formal_issues = validation_results.get("formal_issues", [])
    llm_results = validation_results.get("llm", {})
    if formal_issues:
        st.error("**Se encontraron problemas críticos en los datos que impiden el análisis:**")
        for issue in formal_issues: st.markdown(f"- {issue}")
        st.warning("Por favor, corrige tu fichero de datos y vuelve a cargarlo."); st.stop()
    else:
        st.success("¡Validación formal superada! Tus datos tienen el formato correcto.")
    if llm_results.get("issues"):
        st.warning("Consejos de la IA sobre tu selección de variables:")
        for issue in llm_results["issues"]: st.markdown(f"- *{issue}*")
        if llm_results.get("suggested_fixes"):
            st.markdown("**Sugerencias de mejora:**")
            for fix in llm_results["suggested_fixes"]: st.markdown(f"- *{fix}*")
    if st.button("Proceder al Análisis"):
        st.session_state.app_status = "validated"; st.rerun()

def render_proposal_step():
    """Renderiza la selección de propuestas de análisis."""
    st.header("Paso 2: Elige un Enfoque de Análisis", divider="blue")
    
    if 'proposals_data' not in st.session_state:
        with st.spinner("La IA está analizando tus datos para sugerir enfoques..."):
            st.session_state.proposals_data = cached_get_analysis_proposals(st.session_state.df)
    
    proposals_data = st.session_state.proposals_data

    if "error" in proposals_data:
        st.error("La IA no pudo generar propuestas debido a un error.")
        with st.expander("Ver detalles del error técnico"):
            st.code(proposals_data["error"])
            st.markdown("**Contenido recibido de la IA (si lo hubo):**")
            st.code(proposals_data.get("raw_content", "N/A"))
        st.stop()

    proposals = proposals_data.get("proposals", [])
    
    if not proposals:
        st.error("La IA no devolvió ninguna propuesta válida. Revisa el formato de tus datos o intenta de nuevo.")
        with st.expander("Ver respuesta completa recibida de la IA"):
            st.json(proposals_data)
        st.stop()

    st.info("La IA ha preparado varios enfoques para analizar tus datos. Elige el que mejor se adapte a tu objetivo.")
    for i, proposal in enumerate(proposals):
        with st.expander(f"**Propuesta {i+1}: {proposal['title']}**", expanded=i==0):
            # --- Las siguientes 3 líneas son las corregidas ---
            st.markdown(f"**Razonamiento:** *{proposal['reasoning']}*")
            st.markdown(f"**Inputs sugeridos:** `{proposal['inputs']}`")
            st.markdown(f"**Outputs sugeridos:** `{proposal['outputs']}`")
            if st.button(f"Seleccionar: {proposal['title']}", key=f"select_{i}"):
                st.session_state.selected_proposal = proposal
                st.session_state.app_status = "proposal_selected"
                st.rerun()
