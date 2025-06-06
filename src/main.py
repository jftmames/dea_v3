import sys
import os
import pandas as pd
import streamlit as st
import io

# --- 0) AJUSTE DEL PYTHONPATH Y CONFIGURACIÓN INICIAL ---
# Asegura que los módulos del proyecto se puedan importar correctamente.
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

st.set_page_config(layout="wide", page_title="DEA Deliberativo con IA")

# --- 1) IMPORTACIONES DE MÓDULOS DEL PROYECTO ---
from analysis_dispatcher import execute_analysis
from inquiry_engine import generate_inquiry, to_plotly_tree
from epistemic_metrics import compute_eee
from openai_helpers import generate_analysis_proposals
from data_validator import validate as validate_data
from report_generator import generate_html_report, generate_excel_report
from dea_models.visualizations import plot_hypothesis_distribution, plot_correlation

# --- 2) GESTIÓN DE ESTADO ---
def initialize_state():
    """Limpia la sesión para un nuevo análisis."""
    for key in list(st.session_state.keys()):
        if not key.startswith('_'): # No borrar claves internas de Streamlit
            del st.session_state[key]
    st.session_state.app_status = "initial"

if 'app_status' not in st.session_state:
    initialize_state()

# --- 3) FUNCIONES DE CACHÉ ---
@st.cache_data
def cached_get_analysis_proposals(_df):
    """Cachea las propuestas de la IA para un DataFrame dado."""
    return generate_analysis_proposals(_df.columns.tolist(), _df.head())

@st.cache_data
def cached_run_dea_analysis(_df, dmu_col, input_cols, output_cols, model_key, period_col):
    """Cachea los resultados del análisis DEA para cualquier modelo."""
    return execute_analysis(_df.copy(), dmu_col, input_cols, output_cols, model_key, period_column=period_col)

@st.cache_data
def cached_run_inquiry_engine(root_question, _context):
    """Cachea el árbol de indagación generado por la IA."""
    return generate_inquiry(root_question, context=_context)

# --- 4) COMPONENTES MODULARES DE LA UI ---

def render_eee_explanation(eee_metrics: dict):
    """Muestra la explicación contextual y dinámica del score EEE."""
    st.info(f"**Calidad del Razonamiento (EEE): {eee_metrics['score']:.2%}**")
    
    def interpret_score(name, score):
        if score >= 0.8: return f"**{name}:** Tu puntuación es **excelente** ({score:.0%})."
        if score >= 0.5: return f"**{name}:** Tu puntuación es **buena** ({score:.0%})."
        return f"**{name}:** Tu puntuación es **baja** ({score:.0%}), indicando un área de mejora."

    with st.expander("Ver desglose y consejos para mejorar tu análisis"):
        st.markdown(f"""
        - {interpret_score("Profundidad (D1)", eee_metrics['D1'])}
          - *Consejo:* Si es baja, elige una causa y vuelve a generar un mapa sobre ella para profundizar.
        - {interpret_score("Pluralidad (D2)", eee_metrics['D2'])}
          - *Consejo:* Si es baja, inspírate con un nuevo mapa para considerar más hipótesis iniciales.
        - {interpret_score("Robustez (D5)", eee_metrics['D5'])}
          - *Consejo:* Si es baja, asegúrate de que tu mapa descomponga las ideas principales en al menos dos sub-causas.
        """)

def render_deliberation_workshop(results):
    """Muestra el taller de hipótesis y el mapa de razonamiento."""
    st.header("Paso 4: Razona y Explora las Causas con IA", divider="blue")
    col_map, col_workbench = st.columns([2, 1])

    with col_map:
        st.subheader("Mapa de Razonamiento (IA)", anchor=False)
        if st.button("Generar/Inspirar con nuevo Mapa de Razonamiento", use_container_width=True):
            with st.spinner("La IA está generando un mapa de ideas..."):
                # El contexto puede variar según el modelo, aquí un ejemplo genérico
                context = {
                    "model": results.get("model_name"),
                    "inputs": st.session_state.selected_proposal['inputs'],
                    "outputs": st.session_state.selected_proposal['outputs'],
                    "num_efficient_dmus": (results['main_df'].iloc[:, 1] >= 0.999).sum() if not results['main_df'].empty else 0
                }
                root_question = f"Bajo el enfoque '{st.session_state.selected_proposal['title']}', ¿cuáles son las posibles causas de la ineficiencia observada?"
                tree, error = cached_run_inquiry_engine(root_question, context)
                if error: st.error(f"Error al generar el mapa: {error}")
                st.session_state.inquiry_tree = tree

        if st.session_state.get("inquiry_tree"):
            st.plotly_chart(to_plotly_tree(st.session_state.inquiry_tree), use_container_width=True)
            eee_metrics = compute_eee(st.session_state.inquiry_tree, depth_limit=3, breadth_limit=5)
            render_eee_explanation(eee_metrics)

    with col_workbench:
        st.subheader("Taller de Hipótesis (Usuario)", anchor=False)
        st.info("Usa este taller para explorar tus propias hipótesis.")
        # El taller de hipótesis se asocia mejor con modelos de corte transversal como CCR/BCC o SBM
        if results.get("model_name") not in ["Índice de Productividad de Malmquist"]:
            all_vars = st.session_state.selected_proposal['inputs'] + st.session_state.selected_proposal['outputs']
            chart_type = st.selectbox("1. Elige un tipo de análisis:", ["Análisis de Distribución", "Análisis de Correlación"])

            if chart_type == "Análisis de Distribución":
                var_dist = st.selectbox("2. Elige la variable a analizar:", all_vars)
                if st.button("Generar Gráfico de Distribución"):
                    # Necesitamos un df de eficiencia para colorear, usamos la columna de eficiencia principal
                    df_eff = results['main_df'][[st.session_state.df.columns[0], df_eff.columns[1]]].rename(columns={df_eff.columns[1]: "efficiency"})
                    fig = plot_hypothesis_distribution(df_eff, st.session_state.df, var_dist, st.session_state.df.columns[0])
                    st.session_state.chart_to_show = fig
            # ... (lógica para correlación)

        if st.session_state.get("chart_to_show"):
            st.plotly_chart(st.session_state.chart_to_show, use_container_width=True)


def render_download_section(results):
    """Muestra los botones para descargar los informes."""
    st.subheader("Exportar Análisis Completo", divider="gray")
    col1, col2 = st.columns(2)

    with col1:
        html_report = generate_html_report(
            analysis_results=results,
            inquiry_tree=st.session_state.get("inquiry_tree")
        )
        st.download_button(
            label="Descargar Informe en HTML",
            data=html_report,
            file_name=f"reporte_dea_{results.get('model_name', 'विश्लेषण').replace(' ', '_').lower()}.html",
            mime="text/html",
            use_container_width=True
        )

    with col2:
        excel_report = generate_excel_report(
            analysis_results=results,
            inquiry_tree=st.session_state.get("inquiry_tree")
        )
        st.download_button(
            label="Descargar Informe en Excel",
            data=excel_report,
            file_name=f"reporte_dea_{results.get('model_name', 'विश्लेषण').replace(' ', '_').lower()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

def render_main_dashboard():
    """Renderiza el dashboard principal con selección de modelo y resultados."""
    st.header(f"Paso 3: Configuración y Ejecución del Análisis", divider="blue")
    st.markdown(f"**Enfoque seleccionado:** *{st.session_state.selected_proposal['title']}*")

    model_options = {
        "Radial (CCR/BCC)": "CCR_BCC",
        "No Radial (SBM)": "SBM",
        "Productividad (Malmquist)": "MALMQUIST"
    }
    model_name = st.selectbox("1. Selecciona el tipo de modelo DEA a aplicar:", list(model_options.keys()))
    model_key = model_options[model_name]

    period_col = None
    if model_key == 'MALMQUIST':
        period_col_options = [None] + st.session_state.df.columns.tolist()
        period_col = st.selectbox("2. Selecciona la columna que identifica el período:", period_col_options, index=1)
        if not period_col:
            st.warning("El modelo Malmquist requiere una columna de período.")
            st.stop()

    if st.button(f"Ejecutar Análisis con Modelo: {model_name}", type="primary", use_container_width=True):
        with st.spinner(f"Ejecutando {model_name}..."):
            df = st.session_state.df
            proposal = st.session_state.selected_proposal
            try:
                st.session_state.dea_results = cached_run_dea_analysis(
                    df, df.columns[0], proposal['inputs'], proposal['outputs'], model_key, period_col
                )
                st.session_state.app_status = "results_ready"
            except Exception as e:
                st.error(f"Error durante el análisis: {e}")
                st.session_state.dea_results = None

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
    """Renderiza el paso de validación formal y por IA."""
    st.header("Paso 2b: Validación del Modelo", divider="gray")
    proposal = st.session_state.selected_proposal
    
    with st.spinner("La IA está validando la coherencia de los datos y el modelo..."):
        validation_results = validate_data(st.session_state.df, proposal['inputs'], proposal['outputs'])

    formal_issues = validation_results.get("formal_issues", [])
    llm_results = validation_results.get("llm", {})

    if formal_issues:
        st.error("**Se encontraron problemas críticos en los datos que impiden el análisis:**")
        for issue in formal_issues: st.markdown(f"- {issue}")
        st.warning("Por favor, corrige tu fichero de datos y vuelve a cargarlo.")
        st.stop()
    else:
        st.success("¡Validación formal superada! Tus datos tienen el formato correcto.")

    if llm_results.get("issues"):
        st.warning("Consejos de la IA sobre tu selección de variables:")
        for issue in llm_results["issues"]: st.markdown(f"- *{issue}*")
        if ll.get("suggested_fixes"):
            st.markdown("**Sugerencias de mejora:**")
            for fix in llm_results["suggested_fixes"]: st.markdown(f"- *{fix}*")

    if st.button("Proceder al Análisis"):
        st.session_state.app_status = "validated"
        st.rerun()

def render_proposal_step():
    """Renderiza la selección de propuestas de análisis."""
    st.header("Paso 2: Elige un Enfoque de Análisis", divider="blue")
    if 'proposals' not in st.session_state:
        with st.spinner("La IA está analizando tus datos para sugerir enfoques..."):
            st.session_state.proposals = cached_get_analysis_proposals(st.session_state.df).get("proposals", [])
    
    if not st.session_state.get("proposals"):
        st.error("La IA no pudo generar propuestas. Revisa el formato de tus datos."); st.stop()

    st.info("La IA ha preparado varios enfoques para analizar tus datos. Elige el que mejor se adapte a tu objetivo.")
    for i, proposal in enumerate(st.session_state.get("proposals", [])):
        with st.expander(f"**Propuesta {i+1}: {proposal['title']}**", expanded=i==0):
            st.markdown(f"**Razonamiento:** *{proposal['reasoning']}*")
            st.markdown(f"**Inputs sugeridos:** `{proposal['inputs']}`")
            st.markdown(f"**Outputs sugeridos:** `{proposal['outputs']}`")
            if st.button(f"Seleccionar: {proposal['title']}", key=f"select_{i}"):
                st.session_state.selected_proposal = proposal
                st.session_state.app_status = "proposal_selected"
                st.rerun()

def render_upload_step():
    """Renderiza la carga inicial de datos."""
    st.header("Paso 1: Carga tus Datos", divider="blue")
    uploaded_file = st.file_uploader("Sube un fichero CSV", type=["csv"], on_change=initialize_state)
    
    if uploaded_file:
        try: st.session_state.df = pd.read_csv(uploaded_file)
        except Exception: uploaded_file.seek(0); st.session_state.df = pd.read_csv(uploaded_file, sep=';')
        st.session_state.app_status = "file_loaded"; st.rerun()

# --- 5) FLUJO PRINCIPAL DE LA APLICACIÓN ---
def main():
    st.sidebar.title("DEA Deliberativo")
    if st.sidebar.button("Empezar de Nuevo"):
        initialize_state(); st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.markdown("Herramienta de análisis de eficiencia con asistencia de IA.")

    if st.session_state.app_status == "initial": render_upload_step()
    elif st.session_state.app_status == "file_loaded": render_proposal_step()
    elif st.session_state.app_status == "proposal_selected": render_validation_step()
    elif st.session_state.app_status in ["validated", "results_ready"]: render_main_dashboard()

if __name__ == "__main__":
    main()
