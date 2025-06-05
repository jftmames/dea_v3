import sys
import os

# -------------------------------------------------------
# 0) Ajuste del PYTHONPATH
# -------------------------------------------------------
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import streamlit as st
import pandas as pd
import datetime

# -------------------------------------------------------
# 1) Importaciones
# -------------------------------------------------------
from data_validator import validate
from results import mostrar_resultados
from report_generator import generate_html_report, generate_excel_report
from session_manager import init_db, save_session, load_sessions
from inquiry_engine import generate_inquiry, to_plotly_tree
from epistemic_metrics import compute_eee
from dea_models.visualizations import plot_benchmark_spider

# -------------------------------------------------------
# 2) Configuración y BD
# -------------------------------------------------------
st.set_page_config(layout="wide")
init_db()
default_user_id = "user_1"

# -------------------------------------------------------
# 3) Funciones cacheadas para optimizar rendimiento
# -------------------------------------------------------
@st.cache_data
def run_dea_analysis(_df, dmu_col, input_cols, output_cols):
    return mostrar_resultados(_df.copy(), dmu_col, input_cols, output_cols)

@st.cache_data
def get_inquiry_and_eee(root_q, context, _df_hash):
    inquiry_tree = generate_inquiry(root_q, context=context)
    eee_score = compute_eee(inquiry_tree, depth_limit=5, breadth_limit=5)
    return inquiry_tree, eee_score

# -------------------------------------------------------
# 4) Inicializar session_state
# -------------------------------------------------------
for key, default in [("df", None), ("dmu_col", None), ("input_cols", []), ("output_cols", []), ("dea_results", None), ("inquiry_tree", None), ("df_tree", None), ("eee_score", 0.0), ("df_eee", None), ("selected_dmu", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# -------------------------------------------------------
# 5) Sidebar: cargar sesiones previas
# -------------------------------------------------------
st.sidebar.header("Simulador DEA – Sesiones Guardadas")
sessions = load_sessions(user_id=default_user_id)
if sessions:
    ids = [s["session_id"] for s in sessions]
    selected_session_id = st.sidebar.selectbox("Seleccionar sesión para recargar", ids, index=None, placeholder="Elige una sesión...")
    if selected_session_id:
        sess = next((s for s in sessions if s["session_id"] == selected_session_id), {})
        st.sidebar.markdown(f"**ID:** {sess.get('session_id')}")
        st.sidebar.markdown(f"**Fecha:** {sess.get('timestamp')}")
        st.sidebar.markdown(f"**EEE Score:** {sess.get('eee_score')}")
        st.sidebar.markdown(f"**Notas:** {sess.get('notes')}")

        if st.sidebar.button("Cargar esta sesión"):
            st.session_state.df = pd.DataFrame(sess['df_data']) if 'df_data' in sess and sess['df_data'] is not None else None
            st.session_state.dmu_col = sess.get('dmu_col')
            st.session_state.input_cols = sess.get('input_cols', [])
            st.session_state.output_cols = sess.get('output_cols', [])
            st.session_state.selected_dmu = None # Resetear para evitar errores
            st.session_state.dea_results = None # Forzar recálculo
            st.success(f"Datos de la sesión '{selected_session_id}' cargados. Vuelve a ejecutar el análisis.")
            st.experimental_rerun()
else:
    st.sidebar.write("No hay sesiones guardadas.")

# -------------------------------------------------------
# 6) Área principal: Título y carga de archivo
# -------------------------------------------------------
st.title("Simulador Econométrico-Deliberativo – DEA")
uploaded_file = st.file_uploader("Cargar archivo CSV (con DMUs)", type=["csv"])
if uploaded_file is not None:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.session_state.dea_results = None # Limpiar resultados al cargar nuevo archivo

# -------------------------------------------------------
# 7) Selección de columnas y ejecución de análisis
# -------------------------------------------------------
if st.session_state.df is not None:
    df = st.session_state.df
    st.subheader("Configuración del Análisis")
    
    col1, col2 = st.columns(2)
    with col1:
        all_columns = df.columns.tolist()
        st.session_state.dmu_col = st.selectbox("Columna de DMU (identificador único)", all_columns, index=all_columns.index(st.session_state.dmu_col) if st.session_state.dmu_col in all_columns else 0)
    
    with col2:
        st.session_state.input_cols = st.multiselect("Columnas de Inputs", [c for c in all_columns if c != st.session_state.dmu_col], default=[c for c in st.session_state.input_cols if c in all_columns])
        st.session_state.output_cols = st.multiselect("Columnas de Outputs", [c for c in all_columns if c not in [st.session_state.dmu_col] + st.session_state.input_cols], default=[c for c in st.session_state.output_cols if c in all_columns])

    if st.button("🚀 Ejecutar Análisis DEA"):
        if not st.session_state.input_cols or not st.session_state.output_cols:
            st.error("Por favor, selecciona al menos un input y un output.")
        else:
            with st.spinner("Calculando eficiencias y generando árbol de indagación…"):
                st.session_state.dea_results = run_dea_analysis(df, st.session_state.dmu_col, st.session_state.input_cols, st.session_state.output_cols)
                context = {"inputs": st.session_state.input_cols, "outputs": st.session_state.output_cols}
                df_hash = pd.util.hash_pandas_object(df).sum()
                st.session_state.inquiry_tree, st.session_state.eee_score = get_inquiry_and_eee("Diagnóstico de ineficiencia", context, df_hash)
            st.success("Análisis completado.")

# -------------------------------------------------------
# 8) Mostrar resultados si existen
# -------------------------------------------------------
if st.session_state.dea_results:
    st.subheader("📊 Resultados del Análisis de Eficiencia")
    st.dataframe(st.session_state.dea_results["df_ccr"])

    st.subheader("🕷️ Benchmark Spider CCR")
    # --- CORRECCIÓN 1: GESTIÓN DE ESTADO DEL SELECTBOX ---
    dmu_options = st.session_state.dea_results["df_ccr"][st.session_state.dmu_col].astype(str).tolist()
    # Usamos el parámetro `key` para vincular el widget al session_state
    st.selectbox(
        "Seleccionar DMU para comparar contra peers eficientes:",
        options=dmu_options,
        index=dmu_options.index(st.session_state.selected_dmu) if st.session_state.selected_dmu in dmu_options else 0,
        key="selected_dmu" # Esta es la clave de la corrección
    )
    if st.session_state.selected_dmu:
        spider_fig = plot_benchmark_spider(
            st.session_state.dea_results["merged_ccr"],
            st.session_state.selected_dmu,
            st.session_state.input_cols,
            st.session_state.output_cols
        )
        st.plotly_chart(spider_fig, use_container_width=True)

    st.subheader("🌳 Complejo de Indagación (Árbol)")
    # --- CORRECCIÓN 2: TÍTULO DEL ÁRBOL ---
    if st.session_state.inquiry_tree:
        tree_map_fig = to_plotly_tree(st.session_state.inquiry_tree, title="Árbol de Diagnóstico: Causas y Estrategias de Mejora")
        st.plotly_chart(tree_map_fig, use_container_width=True)

    st.subheader("🧠 Métricas Epistémicas (EEE)")
    # --- CORRECCIÓN 3: EXPLICACIÓN DE EEE ---
    st.info(
        """
        El **Índice de Equilibrio Erotético (EEE)** mide la calidad y robustez del árbol de diagnóstico (0 a 1).
        Un valor más alto indica que el árbol es más profundo, explora más alternativas y es más completo para la toma de decisiones.
        """
    )
    st.metric(label="Puntuación EEE", value=f"{st.session_state.eee_score:.4f}")

    # --- CORRECCIONES 4 Y 5: GUARDADO Y REPORTES ---
    st.subheader("💾 Guardar y Exportar")
    notes = st.text_area("Notas sobre la sesión (opcional)")
    
    if st.button("Guardar Sesión"):
        with st.spinner("Guardando..."):
            save_session(
                user_id=default_user_id,
                inquiry_tree=st.session_state.inquiry_tree,
                eee_score=st.session_state.eee_score,
                notes=notes,
                dmu_col=st.session_state.dmu_col,
                input_cols=st.session_state.input_cols,
                output_cols=st.session_state.output_cols,
                df_data=st.session_state.df.to_dict('records')
            )
        st.success("¡Sesión guardada correctamente!")
        # --- CORRECCIÓN 4: FORZAR REFRESCO ---
        st.cache_data.clear() # Limpiar caché para recargar sesiones
        st.experimental_rerun()

    col1, col2 = st.columns(2)
    with col1:
        html_report = generate_html_report(st.session_state.dea_results["df_ccr"], pd.DataFrame(), pd.DataFrame([{"Métrica": "EEE Score", "Valor": st.session_state.eee_score}]))
        st.download_button("Descargar Reporte HTML", html_report, f"reporte_dea_{datetime.datetime.now().strftime('%Y%m%d')}.html", "text/html", use_container_width=True)
    with col2:
        excel_report = generate_excel_report(st.session_state.dea_results["df_ccr"], pd.DataFrame(), pd.DataFrame([{"Métrica": "EEE Score", "Valor": st.session_state.eee_score}]))
        st.download_button("Descargar Reporte Excel", excel_report, f"reporte_dea_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
