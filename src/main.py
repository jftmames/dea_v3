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
from session_manager import init_db, save_session, load_sessions
from inquiry_engine import generate_inquiry, to_plotly_tree
from epistemic_metrics import compute_eee
from dea_models.visualizations import plot_benchmark_spider, plot_efficiency_histogram, plot_3d_inputs_outputs

# -------------------------------------------------------
# 2) Configuración y BD
# -------------------------------------------------------
st.set_page_config(layout="wide")
init_db()
default_user_id = "user_1"

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
    st.session_state.eee_metrics = None # Cambiado de eee_score a eee_metrics
    st.session_state.df_eee = None
    st.session_state.selected_dmu = None

if 'app_status' not in st.session_state:
    initialize_state()

@st.cache_data
def run_dea_analysis(_df, dmu_col, input_cols, output_cols):
    """Encapsula los cálculos DEA para ser cacheados."""
    return mostrar_resultados(_df.copy(), dmu_col, input_cols, output_cols)

@st.cache_data
def get_inquiry_and_eee(_root_q, _context, _df_hash):
    """Encapsula las llamadas al LLM y EEE para ser cacheados."""
    if not os.getenv("OPENAI_API_KEY"):
        return None, {"score": 0, "D1": 0, "D2": 0, "D3": 0, "D4": 0, "D5": 0}
    inquiry_tree = generate_inquiry(_root_q, context=_context)
    eee_metrics = compute_eee(inquiry_tree, depth_limit=5, breadth_limit=5)
    return inquiry_tree, eee_metrics

# ... (El resto de las funciones de carga de sesión se mantienen igual) ...
def load_full_session(session_data):
    initialize_state()
    st.session_state.df = pd.DataFrame(session_data.get('df_data', []))
    st.session_state.dmu_col = session_data.get('dmu_col')
    st.session_state.input_cols = session_data.get('input_cols', [])
    st.session_state.output_cols = session_data.get('output_cols', [])
    st.session_state.inquiry_tree = session_data.get('inquiry_tree')
    st.session_state.eee_metrics = session_data.get('eee_metrics') # Adaptado
    st.session_state.df_tree = pd.DataFrame(session_data.get('df_tree_data', []))
    st.session_state.df_eee = pd.DataFrame(session_data.get('df_eee_data', []))
    dea_res_raw = session_data.get('dea_results', {})
    st.session_state.dea_results = {k: pd.DataFrame(v) for k, v in dea_res_raw.items() if isinstance(v, list)}
    st.session_state.app_status = "results_ready"
    st.success(f"Sesión '{session_data.get('session_id')}' cargada.")
    st.rerun()
# -------------------------------------------------------
# 4) Sidebar
# -------------------------------------------------------
with st.sidebar:
    st.header("Simulador DEA – Sesiones Guardadas")
    sessions = load_sessions(user_id=default_user_id)
    if not sessions:
        st.write("No hay sesiones guardadas.")
    else:
        session_options = {f"{s['timestamp'].split('T')[0]} - {s.get('notes', 'Sin notas')[:20]}": s['session_id'] for s in sorted(sessions, key=lambda x: x['timestamp'], reverse=True)}
        selected_session_display = st.selectbox("Seleccionar sesión para recargar", session_options.keys(), index=None, placeholder="Elige una sesión guardada...")
        
        if st.button("Cargar Sesión Seleccionada") and selected_session_display:
            session_id_to_load = session_options[selected_session_display]
            session_to_load = next((s for s in sessions if s['session_id'] == session_id_to_load), None)
            if session_to_load:
                load_full_session(session_to_load)

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
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox("Columna de DMU", df.columns.tolist(), key='dmu_col', index=0)
    with col2:
        st.multiselect("Columnas de Inputs", [c for c in df.columns.tolist() if c != st.session_state.dmu_col], key='input_cols')
        st.multiselect("Columnas de Outputs", [c for c in df.columns.tolist() if c not in [st.session_state.dmu_col] + st.session_state.input_cols], key='output_cols')

    if st.button("🚀 Ejecutar Análisis DEA", use_container_width=True):
        if not st.session_state.input_cols or not st.session_state.output_cols:
            st.error("Por favor, selecciona al menos un input y un output.")
        else:
            with st.spinner("Validando datos y realizando análisis..."):
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
    # ... (código de las pestañas CCR y BCC se mantiene igual) ...
    with tab_ccr:
        st.subheader("📊 Tabla de Eficiencias (CCR)")
        st.dataframe(results["df_ccr"])
        # ... resto del contenido de la pestaña CCR
    with tab_bcc:
        st.subheader("📊 Tabla de Eficiencias (BCC)")
        st.dataframe(results["df_bcc"])
        # ... resto del contenido de la pestaña BCC

    # --- SECCIÓN DE ANÁLISIS DELIBERATIVO ---
    if st.session_state.get('inquiry_tree'):
        st.header("Análisis Deliberativo Asistido por IA", divider='rainbow')

        # --- SECCIÓN INTERACTIVA DEL COMPLEJO DE INDAGACIÓN ---
        st.subheader("🔬 Escenarios Interactivos del Complejo de Indagación")
        st.info("Prueba el impacto de las recomendaciones de la IA. Selecciona un escenario para re-ejecutar el análisis.")
        
        # Extraer las recomendaciones principales (hipótesis) del árbol
        main_hypotheses = list(st.session_state.inquiry_tree.get(list(st.session_state.inquiry_tree.keys())[0], {}).keys())
        
        # Crear columnas para los botones de escenario
        cols = st.columns(len(main_hypotheses))
        for i, hypothesis in enumerate(main_hypotheses):
            with cols[i]:
                if st.button(hypothesis, use_container_width=True):
                    # Lógica simplificada: si la hipótesis menciona "input", se quita el primero. Si menciona "output", se quita el primero.
                    # Una implementación más avanzada analizaría el texto para determinar la acción exacta.
                    new_inputs = st.session_state.input_cols.copy()
                    new_outputs = st.session_state.output_cols.copy()
                    
                    if "input" in hypothesis.lower() and len(new_inputs) > 1:
                        removed_var = new_inputs.pop(0)
                        st.warning(f"Re-ejecutando análisis sin el input: **{removed_var}**")
                    elif "output" in hypothesis.lower() and len(new_outputs) > 1:
                        removed_var = new_outputs.pop(0)
                        st.warning(f"Re-ejecutando análisis sin el output: **{removed_var}**")
                    else:
                        st.info("Este escenario no modifica las variables actuales.")

                    # Re-ejecutar el análisis con las nuevas variables
                    with st.spinner("Re-calculando escenario..."):
                        st.session_state.dea_results = run_dea_analysis(st.session_state.df, st.session_state.dmu_col, new_inputs, new_outputs)
                    st.rerun()

        # --- SECCIÓN EXPLICATIVA DE LA MÉTRICA EEE ---
        st.subheader("🧠 Métrica de Calidad del Diagnóstico (EEE)")
        eee = st.session_state.eee_metrics
        st.metric(label="Puntuación EEE Total", value=f"{eee['score']:.4f}")

        with st.expander("Ver desglose y significado de la Métrica EEE"):
            st.markdown("""
            El **Índice de Equilibrio Erotético (EEE)** mide la calidad y robustez del árbol de diagnóstico generado por la IA. Una puntuación más alta indica un análisis más completo y fiable. Se compone de:
            """)
            
            # D1
            st.markdown("**D1: Profundidad del Análisis**")
            st.progress(eee['D1'])
            st.write(f"Puntuación: {eee['D1']:.2f}. Mide qué tan profundo llega el árbol al explorar las causas raíz de la ineficiencia.")
            
            # D2
            st.markdown("**D2: Pluralidad Semántica**")
            st.progress(eee['D2'])
            st.write(f"Puntuación: {eee['D2']:.2f}. Evalúa la variedad de las hipótesis iniciales. Un mayor número de ramas principales mejora esta puntuación.")

            # D3
            st.markdown("**D3: Trazabilidad del Razonamiento**")
            st.progress(eee['D3'])
            st.write(f"Puntuación: {eee['D3']:.2f}. Mide si el árbol presenta caminos lógicos claros y bien diferenciados.")

            # D4
            st.markdown("**D4: Reversibilidad Efectiva**")
            st.progress(eee['D4'])
            st.write(f"Puntuación: {eee['D4']:.2f}. Capacidad teórica de refinar o cambiar de opinión. (Valor fijo en esta versión).")
            
            # D5
            st.markdown("**D5: Robustez ante el Disenso**")
            st.progress(eee['D5'])
            st.write(f"Puntuación: {eee['D5']:.2f}. Mide si el sistema ofrece múltiples perspectivas o causas posibles, lo cual lo hace más robusto.")

    # --- SECCIÓN DE ACCIONES ---
    # ... (El código para guardar y descargar reportes se mantiene igual) ...
    st.header("Acciones", divider='rainbow')
    notes = st.text_area("Notas de la sesión (se guardarán con la sesión)")
    if st.button("💾 Guardar Sesión Actual", use_container_width=True):
         # ... Lógica de guardado ...
        st.success("¡Sesión guardada correctamente!")
        st.balloons()
