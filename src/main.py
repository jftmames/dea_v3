import sys
import os
import pandas as pd
import datetime
import streamlit as st

# -------------------------------------------------------
# 0) Ajuste del PYTHONPATH
# -------------------------------------------------------
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# -------------------------------------------------------
# 1) Importaciones
# -------------------------------------------------------
from data_validator import validate
from results import mostrar_resultados
from report_generator import generate_html_report, generate_excel_report
# Se eliminan las importaciones de session_manager
from inquiry_engine import generate_inquiry, to_plotly_tree
from epistemic_metrics import compute_eee
from dea_models.visualizations import plot_benchmark_spider, plot_efficiency_histogram, plot_3d_inputs_outputs

# -------------------------------------------------------
# 2) Configuración y BD
# -------------------------------------------------------
st.set_page_config(layout="wide")
# Se elimina la llamada a init_db()

# -------------------------------------------------------
# 3) Funciones de inicialización y carga
# -------------------------------------------------------
def initialize_state():
    """Inicializa o resetea el estado de la sesión para prevenir errores."""
    st.session_state.app_status = "initial"
    st.session_state.df = None
    st.session_state.dmu_col = None
    st.session_state.input_cols = []
    st.session_state.output_cols = []
    st.session_state.dea_results = None
    st.session_state.inquiry_tree = None
    st.session_state.df_tree = None
    st.session_state.eee_metrics = None
    st.session_state.df_eee = None
    st.session_state.selected_dmu = None

if 'app_status' not in st.session_state:
    initialize_state()

@st.cache_data
def run_dea_analysis(_df, dmu_col, input_cols, output_cols):
    """Encapsula los cálculos DEA para ser cacheados."""
    if not input_cols or not output_cols:
        return None
    return mostrar_resultados(_df.copy(), dmu_col, input_cols, output_cols)

@st.cache_data
def get_inquiry_and_eee(_root_q, _context, _df_hash):
    """Encapsula las llamadas al LLM y EEE para ser cacheados."""
    if not os.getenv("OPENAI_API_KEY"):
        return None, {"score": 0, "D1": 0, "D2": 0, "D3": 0, "D4": 0, "D5": 0}
    inquiry_tree = generate_inquiry(_root_q, context=_context)
    eee_metrics = compute_eee(inquiry_tree, depth_limit=5, breadth_limit=5)
    return inquiry_tree, eee_metrics

# -------------------------------------------------------
# 4) Sidebar
# -------------------------------------------------------
# La barra lateral ahora está vacía o puedes usarla para otros controles en el futuro.
st.sidebar.header("Configuración")
st.sidebar.info("La funcionalidad de guardar/cargar sesiones ha sido desactivada.")


# -------------------------------------------------------
# 5) Área principal
# -------------------------------------------------------
st.title("Simulador Econométrico-Deliberativo – DEA")
uploaded_file = st.file_uploader("Cargar nuevo archivo CSV", type=["csv"])
if uploaded_file is not None:
    if st.session_state.df is None:
        initialize_state()
        try:
            st.session_state.df = pd.read_csv(uploaded_file, sep=',')
        except Exception:
            try:
                uploaded_file.seek(0)
                st.session_state.df = pd.read_csv(uploaded_file, sep=';')
            except Exception as e:
                st.error(f"Error al leer el fichero CSV. Detalle: {e}")
                st.session_state.df = None
        if st.session_state.df is not None:
            st.rerun()

if 'df' in st.session_state and st.session_state.df is not None:
    df = st.session_state.df
    st.subheader("Configuración del Análisis")
    
    def apply_scenario(new_inputs, new_outputs):
        st.session_state.input_cols = new_inputs
        st.session_state.output_cols = new_outputs

    col1, col2 = st.columns(2)
    with col1:
        dmu_col_index = df.columns.tolist().index(st.session_state.get('dmu_col')) if st.session_state.get('dmu_col') in df.columns else 0
        st.selectbox("Columna de DMU", df.columns.tolist(), key='dmu_col', index=dmu_col_index)
    with col2:
        st.multiselect("Columnas de Inputs", [c for c in df.columns.tolist() if c != st.session_state.dmu_col], key='input_cols', default=st.session_state.get('input_cols', []))
        st.multiselect("Columnas de Outputs", [c for c in df.columns.tolist() if c not in [st.session_state.dmu_col] + st.session_state.input_cols], key='output_cols', default=st.session_state.get('output_cols', []))

    if st.button("🚀 Ejecutar Análisis DEA", use_container_width=True):
        if not st.session_state.input_cols or not st.session_state.output_cols:
            st.error("Por favor, selecciona al menos un input y un output.")
        else:
            with st.spinner("Realizando análisis completo..."):
                st.session_state.dea_results = run_dea_analysis(df, st.session_state.dmu_col, st.session_state.input_cols, st.session_state.output_cols)
                context = {"inputs": st.session_state.input_cols, "outputs": st.session_state.output_cols}
                df_hash = pd.util.hash_pandas_object(df).sum()
                st.session_state.inquiry_tree, st.session_state.eee_metrics = get_inquiry_and_eee("Diagnóstico de ineficiencia", context, df_hash)
                st.session_state.app_status = "results_ready"
            st.success("Análisis completado.")

# --- Mostrar resultados ---
if st.session_state.get('app_status') == "results_ready" and st.session_state.get('dea_results'):
    results = st.session_state.dea_results
    st.header("Resultados del Análisis DEA", divider='rainbow')
    
    tab_ccr, tab_bcc = st.tabs(["**Análisis CCR**", "**Análisis BCC**"])
    
    with tab_ccr:
        st.subheader("📊 Tabla de Eficiencias (CCR)")
        st.dataframe(results["df_ccr"])
        st.subheader("Visualizaciones de Eficiencia (CCR)")
        col1, col2 = st.columns(2)
        with col1:
            if 'hist_ccr' in results:
                st.plotly_chart(results['hist_ccr'], use_container_width=True)
        with col2:
            if 'scatter3d_ccr' in results:
                st.plotly_chart(results['scatter3d_ccr'], use_container_width=True)
        st.subheader("🕷️ Benchmark Spider (CCR)")
        dmu_options_ccr = results["df_ccr"][st.session_state.dmu_col].astype(str).tolist()
        selected_dmu_ccr = st.selectbox("Seleccionar DMU para comparar (CCR):", options=dmu_options_ccr, key="dmu_ccr")
        if selected_dmu_ccr:
            spider_fig_ccr = plot_benchmark_spider(results["merged_ccr"], selected_dmu_ccr, st.session_state.input_cols, st.session_state.output_cols)
            st.plotly_chart(spider_fig_ccr, use_container_width=True)

    with tab_bcc:
        st.subheader("📊 Tabla de Eficiencias (BCC)")
        st.dataframe(results["df_bcc"])
        st.subheader("Visualización de Eficiencia (BCC)")
        if 'hist_bcc' in results:
            st.plotly_chart(results['hist_bcc'], use_container_width=True)

    if st.session_state.get('inquiry_tree'):
        st.header("Análisis Deliberativo Asistido por IA", divider='rainbow')
        
        st.subheader("🔬 Escenarios Interactivos del Complejo de Indagación")
        st.info("La IA ha generado las siguientes hipótesis. Cada una propone una acción para probar un escenario alternativo. Pulsa un botón para pre-seleccionar un escenario y luego haz clic en 'Ejecutar Análisis DEA' para ver los resultados.")
        
        main_hypotheses = list(st.session_state.inquiry_tree.get(list(st.session_state.inquiry_tree.keys())[0], {}).keys())
        
        for i, hypothesis in enumerate(main_hypotheses):
            with st.container(border=True):
                st.markdown(f"##### Hipótesis de la IA: *«{hypothesis}»*")
                
                new_inputs = st.session_state.input_cols.copy()
                new_outputs = st.session_state.output_cols.copy()
                can_apply = False
                action_text = "No se puede aplicar con la selección actual (se requiere más de 1 input/output)."

                if "input" in hypothesis.lower() and len(new_inputs) > 1:
                    removed_var = new_inputs.pop(0)
                    can_apply = True
                    action_text = f"Probar el análisis **eliminando el input:** `{removed_var}`"
                elif "output" in hypothesis.lower() and len(new_outputs) > 1:
                    removed_var = new_outputs.pop(0)
                    can_apply = True
                    action_text = f"Probar el análisis **eliminando el output:** `{removed_var}`"
                
                st.write(f"**Acción Sugerida:** {action_text}")
                
                st.button(
                    "Aplicar este escenario",
                    key=f"hyp_{i}",
                    on_click=apply_scenario,
                    args=(new_inputs, new_outputs),
                    disabled=not can_apply,
                    use_container_width=True
                )
        
        st.subheader("🧠 Métrica de Calidad del Diagnóstico (EEE)")
        eee = st.session_state.get('eee_metrics')
        if eee:
            st.metric(label="Puntuación EEE Total", value=f"{eee.get('score', 0):.4f}")
            with st.expander("Ver desglose y significado de la Métrica EEE"):
                st.markdown("""El **Índice de Equilibrio Erotético (EEE)** mide la calidad del árbol de diagnóstico. Una puntuación alta indica un análisis más completo.""")
                st.markdown("**D1: Profundidad del Análisis**")
                st.progress(eee.get('D1', 0), text=f"Puntuación: {eee.get('D1', 0):.2f}")
                st.markdown("**D2: Pluralidad Semántica (Variedad de hipótesis)**")
                st.progress(eee.get('D2', 0), text=f"Puntuación: {eee.get('D2', 0):.2f}")
                st.markdown("**D3: Trazabilidad del Razonamiento**")
                st.progress(eee.get('D3', 0), text=f"Puntuación: {eee.get('D3', 0):.2f}")

    # --- SECCIÓN DE REPORTES (Se mantiene) ---
    st.header("Generar Reportes", divider='rainbow')
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Descargar HTML",
            generate_html_report(results.get("df_ccr"), st.session_state.get('df_tree'), st.session_state.get('df_eee')),
            "reporte_dea.html", "text/html", use_container_width=True)
    with col2:
        st.download_button(
            "Descargar Excel",
            generate_excel_report(results.get("df_ccr"), st.session_state.get('df_tree'), st.session_state.get('df_eee')),
            "reporte_dea.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
