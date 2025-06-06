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
from results import mostrar_resultados
from inquiry_engine import generate_inquiry, to_plotly_tree
from epistemic_metrics import compute_eee
from openai_helpers import generate_analysis_proposals
from data_validator import validate as validate_data
from dea_models.visualizations import plot_hypothesis_distribution, plot_benchmark_spider, plot_correlation

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
def cached_run_dea_analysis(_df, dmu_col, input_cols, output_cols):
    """Cachea los resultados del análisis DEA."""
    # Nota: Ampliar esta función para que acepte el tipo de modelo (CCR, SBM, etc.)
    return mostrar_resultados(_df.copy(), dmu_col, input_cols, output_cols)

@st.cache_data
def cached_run_inquiry_engine(root_question, _context):
    """Cachea el árbol de indagación generado por la IA."""
    return generate_inquiry(root_question, context=_context)

# --- 4) COMPONENTES DE LA UI ---

def render_eee_explanation(eee_metrics: dict):
    """Muestra la explicación contextual del score EEE."""
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

def render_hypothesis_workshop(proposal, results):
    """Muestra la sección para que el usuario valide sus propias hipótesis."""
    st.subheader("Taller de Hipótesis (Usuario)", anchor=False)
    st.info("Usa este taller para explorar tus propias hipótesis, inspirado por el mapa de la IA.")

    all_vars = proposal['inputs'] + proposal['outputs']
    chart_type = st.selectbox("1. Elige un tipo de análisis:", ["Análisis de Distribución", "Análisis de Correlación"])

    if chart_type == "Análisis de Distribución":
        var_dist = st.selectbox("2. Elige la variable a analizar:", all_vars)
        if st.button("Generar Gráfico de Distribución"):
            st.session_state.chart_to_show = {"type": "distribution", "var": var_dist}

    elif chart_type == "Análisis de Correlación":
        var_x = st.selectbox("2. Elige la variable para el eje X:", all_vars, key="wb_var_x")
        var_y = st.selectbox("3. Elige la variable para el eje Y:", all_vars, key="wb_var_y")
        if st.button("Generar Gráfico de Correlación"):
            st.session_state.chart_to_show = {"type": "correlation", "var_x": var_x, "var_y": var_y}

    if st.session_state.get("chart_to_show"):
        chart_info = st.session_state.chart_to_show
        st.subheader("Resultado de tu Hipótesis", anchor=False)
        dmu_col = st.session_state.df.columns[0]
        if chart_info["type"] == "distribution":
            fig = plot_hypothesis_distribution(results['df_ccr'], st.session_state.df, chart_info["var"], dmu_col)
        elif chart_info["type"] == "correlation":
            fig = plot_correlation(results['df_ccr'], st.session_state.df, chart_info["var_x"], chart_info["var_y"], dmu_col)
        st.plotly_chart(fig, use_container_width=True)
        if st.button("Limpiar gráfico"):
            st.session_state.chart_to_show = None
            st.rerun()

def render_main_dashboard():
    """Renderiza el dashboard principal con los resultados y el taller deliberativo."""
    st.header(f"Analizando: '{st.session_state.selected_proposal['title']}'", divider="blue")
    
    # --- EJECUCIÓN DEL ANÁLISIS ---
    if 'dea_results' not in st.session_state:
        with st.spinner("Realizando análisis DEA..."):
            df = st.session_state.df
            proposal = st.session_state.selected_proposal
            st.session_state.dea_results = cached_run_dea_analysis(
                df, df.columns[0], proposal['inputs'], proposal['outputs']
            )
        st.session_state.app_status = "results_ready"

    results = st.session_state.dea_results

    # --- TALLER DE RAZONAMIENTO Y DELIBERACIÓN ---
    st.header("Paso 4: Razona y Explora las Causas con IA", divider="blue")
    col_map, col_workbench = st.columns([2, 1])

    with col_map:
        st.subheader("Mapa de Razonamiento (IA)", anchor=False)
        if st.button("Generar/Inspirar con nuevo Mapa de Razonamiento", use_container_width=True):
            with st.spinner("La IA está generando un mapa de ideas..."):
                context = {
                    "inputs": st.session_state.selected_proposal['inputs'],
                    "outputs": st.session_state.selected_proposal['outputs'],
                    "avg_efficiency_ccr": results["df_ccr"]["tec_efficiency_ccr"].mean(),
                }
                root_question = f"Bajo el enfoque '{st.session_state.selected_proposal['title']}', ¿cuáles son las posibles causas de la ineficiencia?"
                tree, error = cached_run_inquiry_engine(root_question, context)
                if error: st.error(f"Error al generar el mapa: {error}")
                st.session_state.inquiry_tree = tree

        if st.session_state.get("inquiry_tree"):
            st.plotly_chart(to_plotly_tree(st.session_state.inquiry_tree), use_container_width=True)
            eee_metrics = compute_eee(st.session_state.inquiry_tree, depth_limit=3, breadth_limit=5)
            render_eee_explanation(eee_metrics)

    with col_workbench:
        render_hypothesis_workshop(st.session_state.selected_proposal, results)

    # --- PESTAÑAS DE RESULTADOS NUMÉRICOS ---
    st.header("Paso 5: Resultados Numéricos y Gráficos Detallados", divider="blue")
    # (El código para mostrar las pestañas de resultados CCR/BCC iría aquí, es similar al original)
    # ...

def render_proposal_step():
    """Renderiza el paso de selección de propuestas de la IA."""
    st.header("Paso 2: Elige un Enfoque de Análisis", divider="blue")
    if 'proposals' not in st.session_state:
        with st.spinner("La IA está analizando tus datos para sugerir enfoques..."):
            proposals_data = cached_get_analysis_proposals(st.session_state.df)
            st.session_state.proposals = proposals_data.get("proposals", [])
    
    if not st.session_state.get("proposals"):
        st.error("La IA no pudo generar propuestas. Revisa el formato de tus datos.")
        st.stop()

    st.info("La IA ha preparado varios enfoques para analizar tus datos. Elige el que mejor se adapte a tu objetivo.")
    for i, proposal in enumerate(st.session_state.get("proposals", [])):
        with st.expander(f"**Propuesta {i+1}: {proposal['title']}**", expanded=i==0):
            st.markdown(f"**Razonamiento:** *{proposal['reasoning']}*")
            st.markdown(f"**Inputs sugeridos:** `{proposal['inputs']}`")
            st.markdown(f"**Outputs sugeridos:** `{proposal['outputs']}`")
            if st.button(f"Seleccionar este análisis", key=f"select_{i}"):
                st.session_state.selected_proposal = proposal
                st.session_state.app_status = "proposal_selected"
                st.rerun()

def render_validation_step():
    """Renderiza el paso de validación de datos (formal y por IA)."""
    st.header("Paso 3: Validación del Modelo Seleccionado", divider="blue")
    proposal = st.session_state.selected_proposal
    
    with st.spinner("Validando la coherencia de los datos y el modelo..."):
        validation_results = validate_data(
            st.session_state.df, proposal['inputs'], proposal['outputs']
        )

    formal_issues = validation_results.get("formal_issues", [])
    llm_results = validation_results.get("llm", {})

    if formal_issues:
        for issue in formal_issues:
            st.error(f"**Error Crítico en los Datos:** {issue}")
        st.warning("El análisis no puede continuar. Por favor, corrige tu fichero de datos y vuelve a cargarlo.")
        st.stop()
    else:
        st.success("¡Validación formal superada! Tus datos tienen el formato correcto.")

    if llm_results.get("issues"):
        st.warning("Consejos de la IA sobre tu modelo:")
        for issue in llm_results["issues"]:
            st.markdown(f"- *{issue}*")
        if llm_results.get("suggested_fixes"):
            st.markdown("**Sugerencias de mejora:**")
            for fix in llm_results["suggested_fixes"]:
                st.markdown(f"- *{fix}*")

    if st.button("Continuar al Análisis"):
        st.session_state.app_status = "validated"
        st.rerun()

def render_upload_step():
    """Renderiza el paso inicial de carga de fichero."""
    st.header("Paso 1: Carga tus Datos", divider="blue")
    uploaded_file = st.file_uploader("Sube un fichero CSV", type=["csv"], on_change=initialize_state)
    
    if uploaded_file:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            st.session_state.df = pd.read_csv(uploaded_file, sep=';')
        st.session_state.app_status = "file_loaded"
        st.rerun()

# --- 5) FLUJO PRINCIPAL DE LA APLICACIÓN ---
def main():
    st.title("💡 DEA Deliberativo con IA")
    st.markdown("Una herramienta para analizar la eficiencia y razonar sobre sus causas con ayuda de Inteligencia Artificial.")
    
    if st.button("Empezar de Nuevo"):
        initialize_state()
        st.rerun()

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
