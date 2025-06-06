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

# --- 2) DEFINICIÓN DE TODAS LAS FUNCIONES ---

# -- Funciones de IA y Caché --
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

# -- Función de Gestión de Estado --
def initialize_state():
    """Reinicia de forma segura el estado de la sesión Y LIMPIA LA CACHÉ."""
    cached_get_analysis_proposals.clear()
    cached_run_dea_analysis.clear()
    st.session_state.app_status = "initial"
    st.session_state.df = None
    st.session_state.proposals_data = None
    st.session_state.selected_proposal = None
    st.session_state.dea_results = None
    st.session_state.inquiry_tree = None
    st.session_state.tree_explanation = None
    st.session_state.chart_to_show = None

# -- Funciones de Lógica de IA (movidas aquí para evitar errores) --
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
        "Eres un consultor experto en Data Envelopment Analysis (DEA)...\n" # Abreviado
        "Devuelve únicamente un objeto JSON válido con una sola clave raíz 'proposals'..."
    )
    content = "No se recibió contenido."
    try:
        resp = chat_completion(prompt, use_json_mode=True)
        content = resp.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        return {"error": f"Error al procesar la respuesta de la IA: {str(e)}", "raw_content": content}


# -- Componentes Modulares de la UI --
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
                    efficiency_col = main_df.columns[1]
                    num_efficient = int((main_df[efficiency_col] >= 0.999).sum())
                context = {"model": results.get("model_name"),"inputs": st.session_state.selected_proposal['inputs'],"outputs": st.session_state.selected_proposal['outputs'],"num_efficient_dmus": num_efficient}
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
                with st.container(border=True): st.markdown(explanation.get("text", "No se pudo generar la explicación."))
            st.plotly_chart(to_plotly_tree(st.session_state.inquiry_tree), use_container_width=True)
            eee_metrics = compute_eee(st.session_state.inquiry_tree, depth_limit=3, breadth_limit=5)
            render_eee_explanation(eee_metrics)
    with col_workbench:
        st.subheader("Taller de Hipótesis (Usuario)", anchor=False)
        st.info("Usa este taller para explorar tus propias hipótesis.")
        if results.get("model_name") not in ["Índice de Productividad de Malmquist"]:
            all_vars = st.session_state.selected_proposal['inputs'] + st.session_state.selected_proposal['outputs']
            chart_type = st.selectbox("1. Elige un tipo de análisis:", ["Análisis de Distribución", "Análisis de Correlación"], key="wb_chart_type")
            df_eff = results.get('main_df')
            if df_eff is not None and not df_eff.empty:
                efficiency_col_name = df_eff.columns[1]
                df_eff_generic = df_eff.rename(columns={efficiency_col_name: "efficiency"})
                if chart_type == "Análisis de Distribución":
                    var_dist = st.selectbox("2. Elige la variable a analizar:", all_vars, key="wb_var_dist")
                    if st.button("Generar Gráfico de Distribución"):
                        fig = plot_hypothesis_distribution(df_eff_generic, st.session_state.df, var_dist, st.session_state.df.columns[0])
                        st.session_state.chart_to_show = fig
                elif chart_type == "Análisis de Correlación":
                    var_x = st.selectbox("2. Elige la variable para el eje X:", all_vars, key="wb_var_x")
                    var_y = st.selectbox("3. Elige la variable para el eje Y:", all_vars, key="wb_var_y")
                    if st.button("Generar Gráfico de Correlación"):
                        fig = plot_correlation(df_eff_generic, st.session_state.df, var_x, var_y, st.session_state.df.columns[0])
                        st.session_state.chart_to_show = fig
            else: st.warning("No hay datos de resultados para generar gráficos.")
    if st.session_state.get("chart_to_show"):
        st.subheader("Resultado de tu Hipótesis", anchor=False)
        st.plotly_chart(st.session_state.chart_to_show, use_container_width=True)
        if st.button("Limpiar gráfico"): st.session_state.chart_to_show = None; st.rerun()

def render_download_section(results):
    st.subheader("Exportar Análisis Completo", divider="gray")
    col1, col2 = st.columns(2)
    with col1:
        html_report = generate_html_report(analysis_results=results, inquiry_tree=st.session_state.get("inquiry_tree"))
        st.download_button(label="Descargar Informe en HTML", data=html_report, file_name=f"reporte_dea_{results.get('model_name', 'विश्लेषण').replace(' ', '_').lower()}.html", mime="text/html", use_container_width=True)
    with col2:
        excel_report = generate_excel_report(analysis_results=results, inquiry_tree=st.session_state.get("inquiry_tree"))
        st.download_button(label="Descargar Informe en Excel", data=excel_report, file_name=f"reporte_dea_{results.get('model_name', 'विश्लेषण').replace(' ', '_').lower()}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

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
    else: st.success("¡Validación formal superada! Tus datos tienen el formato correcto.")
    if llm_results.get("issues"):
        st.warning("Consejos de la IA sobre tu selección de variables:")
        for issue in llm_results["issues"]: st.markdown(f"- *{issue}*")
        if llm_results.get("suggested_fixes"):
            st.markdown("**Sugerencias de mejora:**")
            for fix in llm_results["suggested_fixes"]: st.markdown(f"- *{fix}*")
    if st.button("Proceder al Análisis"): st.session_state.app_status = "validated"; st.rerun()

def render_proposal_step():
    st.header("Paso 2: Elige un Enfoque de Análisis", divider="blue")
    if not st.session_state.get('proposals_data'):
        with st.spinner("La IA está analizando tus datos para sugerir enfoques..."):
            st.session_state.proposals_data = cached_get_analysis_proposals(st.session_state.df)
    proposals_data = st.session_state.proposals_data
    if "error" in proposals_data:
        st.error("La IA no pudo generar propuestas debido a un error.")
        with st.expander("Ver detalles del error técnico"):
            st.code(proposals_data["error"]); st.markdown("**Contenido recibido de la IA (si lo hubo):**"); st.code(proposals_data.get("raw_content", "N/A"))
        st.stop()
    proposals = proposals_data.get("proposals", [])
    if not proposals:
        st.error("La IA no devolvió ninguna propuesta válida."); st.json(proposals_data); st.stop()
    st.info("La IA ha preparado varios enfoques para analizar tus datos. Elige el que mejor se adapte a tu objetivo.")
    for i, proposal in enumerate(proposals):
        with st.expander(f"**Propuesta {i+1}: {proposal['title']}**", expanded=i==0):
            st.markdown(f"**Razonamiento:** *{proposal['reasoning']}*"); st.markdown(f"**Inputs sugeridos:** `{proposal['inputs']}`"); st.markdown(f"**Outputs sugeridos:** `{proposal['outputs']}`")
            if st.button(f"Seleccionar: {proposal['title']}", key=f"select_{i}"):
                st.session_state.selected_proposal = proposal
                st.session_state.app_status = "proposal_selected"; st.rerun()

def render_upload_step():
    st.header("Paso 1: Carga tus Datos", divider="blue")
    uploaded_file = st.file_uploader("Sube un fichero CSV", type=["csv"], on_change=initialize_state)
    if uploaded_file:
        try: st.session_state.df = pd.read_csv(uploaded_file)
        except Exception: uploaded_file.seek(0); st.session_state.df = pd.read_csv(uploaded_file, sep=';')
        st.session_state.app_status = "file_loaded"; st.rerun()

# --- 5) FLUJO PRINCIPAL DE LA APLICACIÓN ---
def main():
    """Función principal que orquesta la aplicación."""
    # CORRECCIÓN: La inicialización del estado se mueve aquí para asegurar
    # que todas las funciones ya han sido definidas antes de ser llamadas.
    if 'app_status' not in st.session_state:
        initialize_state()

    st.sidebar.title("DEA Deliberativo")
    if st.sidebar.button("Empezar de Nuevo"):
        initialize_state()
        st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.info("Una herramienta para el análisis de eficiencia y la deliberación estratégica con asistencia de IA.")

    # Máquina de estados que controla qué se renderiza en la pantalla
    if st.session_state.app_status == "initial": render_upload_step()
    elif st.session_state.app_status == "file_loaded": render_proposal_step()
    elif st.session_state.app_status == "proposal_selected": render_validation_step()
    elif st.session_state.app_status in ["validated", "results_ready"]: render_main_dashboard()

if __name__ == "__main__":
    main()
