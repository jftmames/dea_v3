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

# Configuración de la página de Streamlit
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
    new_id = str(uuid.uuid4())
    if source_scenario_id and source_scenario_id in st.session_state.scenarios:
        st.session_state.scenarios[new_id] = st.session_state.scenarios[source_scenario_id].copy()
        st.session_state.scenarios[new_id]['name'] = f"Copia de {st.session_state.scenarios[source_scenario_id]['name']}"
        st.session_state.scenarios[new_id]['dea_results'] = None
        st.session_state.scenarios[new_id]['inquiry_tree'] = None
        st.session_state.scenarios[new_id]['user_justifications'] = {}
        st.session_state.scenarios[new_id]['app_status'] = "data_loaded"
    else:
        st.session_state.scenarios[new_id] = {
            "name": name, "df": st.session_state.get("global_df"), "app_status": "initial",
            "proposals_data": None, "selected_proposal": None, "dea_config": {},
            "dea_results": None, "inquiry_tree": None, "tree_explanation": None,
            "chart_to_show": None, "user_justifications": {}, "data_overview": {}
        }
    st.session_state.active_scenario_id = new_id

def get_active_scenario():
    active_id = st.session_state.get('active_scenario_id')
    return st.session_state.scenarios.get(active_id)

def initialize_global_state():
    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = {}
        st.session_state.active_scenario_id = None
        st.session_state.global_df = None

def reset_all():
    st.cache_data.clear()
    st.cache_resource.clear()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_global_state()

# --- 3) FUNCIONES DE CACHÉ Y LÓGICA DE IA ---
# (Sin cambios en esta sección)
@st.cache_data
def cached_run_dea_analysis(_df, dmu_col, input_cols, output_cols, model_key, period_col):
    return execute_analysis(_df.copy(), dmu_col, input_cols, output_cols, model_key, period_column=period_col)
@st.cache_data
def cached_run_inquiry_engine(root_question, _context):
    return generate_inquiry(root_question, context=_context)
# ... (resto de funciones cacheadas sin cambios) ...

# --- CLASE ENCAPSULADORA DE LA UI (CON CORRECCIONES) ---
class AppRenderer:
    def __init__(self):
        pass

    def render_upload_step(self):
        st.header("Paso 1: Carga tus Datos", divider="blue")
        st.info("Para comenzar, selecciona una fuente de datos. Puedes subir tu propio archivo CSV o utilizar uno de nuestros casos de estudio para empezar rápidamente.")

        source_option = st.radio(
            "Elige una fuente de datos:",
            ('Usar un caso de estudio', 'Subir un archivo CSV'),
            key='data_source_radio'
        )

        df_to_load = None
        file_name = None

        if source_option == 'Usar un caso de estudio':
            datasets_path = os.path.join(script_dir, 'datasets')
            try:
                available_datasets = [f for f in os.listdir(datasets_path) if f.endswith('.csv')]
                if not available_datasets:
                    st.warning("No se encontraron datasets en la carpeta `datasets`.")
                    return
                
                selected_dataset = st.selectbox('Selecciona un caso de estudio:', available_datasets)
                if selected_dataset:
                    file_path = os.path.join(datasets_path, selected_dataset)
                    df_to_load = pd.read_csv(file_path)
                    file_name = selected_dataset

            except FileNotFoundError:
                st.error(f"Error: La carpeta `datasets` no se encuentra. Asegúrate de que esté en la misma ubicación que `main.py` (dentro de `src`).")
                return

        elif source_option == 'Subir un archivo CSV':
            uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"], label_visibility="collapsed")
            if uploaded_file:
                try:
                    df_to_load = pd.read_csv(uploaded_file)
                    file_name = uploaded_file.name
                except Exception as e:
                    st.error(f"No se pudo leer el archivo. Error: {e}")
                    return
        
        # --- BOTÓN DE CARGA EXPLÍCITO ---
        if st.button("Cargar y Analizar Datos", type="primary", use_container_width=True):
            if df_to_load is not None:
                st.session_state.global_df = df_to_load
                create_new_scenario(name="Modelo Base")
                active_scenario = get_active_scenario()
                active_scenario['app_status'] = "data_loaded"
                st.success(f"Datos de '{file_name}' cargados. El análisis preliminar está listo.")
                st.rerun()
            else:
                st.error("Por favor, selecciona un archivo válido antes de cargar los datos.")

    def render_scenario_navigator(self):
        st.sidebar.title("Navegador de Escenarios")
        st.sidebar.markdown("Gestiona y compara tus modelos.")
        st.sidebar.divider()

        if not st.session_state.scenarios:
            st.sidebar.info("Carga datos para empezar.")
            return

        scenario_names = {sid: s['name'] for sid, s in st.session_state.scenarios.items()}
        active_id = st.session_state.get('active_scenario_id')
        if active_id not in scenario_names:
            active_id = next(iter(scenario_names), None)

        st.session_state.active_scenario_id = st.sidebar.selectbox(
            "Escenario Activo", options=list(st.session_state.scenarios.keys()),
            format_func=lambda sid: scenario_names.get(sid, "N/A"),
            index=list(st.session_state.scenarios.keys()).index(active_id) if active_id in st.session_state.scenarios else 0,
            key='scenario_selector'
        )
        # ... (resto de la función sin cambios) ...


    # ... (TODAS LAS OTRAS FUNCIONES DE RENDERIZADO VAN AQUÍ SIN CAMBIOS) ...
    # PEGA AQUÍ EL RESTO DE TUS FUNCIONES DE LA CLASE AppRenderer
    def render_preliminary_analysis_step(self, active_scenario):
        st.header(f"Paso 1b: Exploración Preliminar de Datos para '{active_scenario['name']}'", divider="blue")
        st.info("Este paso es crucial para entender tus datos antes de realizar el análisis DEA. Te ayudará a identificar posibles problemas (como outliers o multicolinealidad) y a tomar decisiones informadas sobre la selección de inputs y outputs.")
        df = active_scenario['df']
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

        if not numerical_cols:
            st.warning("No se encontraron columnas numéricas para el análisis exploratorio.")
            if st.button("Proceder al Paso 2", key=f"proceed_no_numeric_{st.session_state.active_scenario_id}"):
                active_scenario['app_status'] = "file_loaded"
                st.rerun()
            return
        
        st.subheader("1. Estadísticas Descriptivas:", anchor=False)
        st.dataframe(df[numerical_cols].describe().T)

        st.subheader("2. Distribución de Variables (Histogramas):", anchor=False)
        for col in numerical_cols:
            fig = px.histogram(df, x=col, title=f"Distribución de {col}", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True, key=f"hist_{col}_{st.session_state.active_scenario_id}")

        st.subheader("3. Matriz de Correlación (Mapa de Calor):", anchor=False)
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu', range_color=[-1,1], title="Matriz de Correlación")
            st.plotly_chart(fig_corr, use_container_width=True, key=f"corr_heatmap_{st.session_state.active_scenario_id}")
        else:
            st.info("Se necesitan al menos dos columnas numéricas para generar una matriz de correlación.")

        if st.button("Proceder al Paso 2: Elegir Enfoque de Análisis", type="primary", use_container_width=True):
            active_scenario['app_status'] = "file_loaded"
            st.rerun()


# --- FUNCIÓN PRINCIPAL DE LA APLICACIÓN ---
def main():
    initialize_global_state()

    # --- CARGA DEL LOGO (MÉTODO CORREGIDO) ---
    logo_path = os.path.join(script_dir, 'assets', 'logo.png')
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=200)
    else:
        st.sidebar.warning("Logo no encontrado. Verifica la ruta: src/assets/logo.png")

    st.sidebar.title("DEA Deliberative Modeler")
    if st.sidebar.button("🔴 Empezar Nueva Sesión"):
        reset_all()
        st.rerun()
    st.sidebar.divider()

    renderer = AppRenderer()
    renderer.render_scenario_navigator()
    st.sidebar.divider()

    active_scenario = get_active_scenario()
    
    analysis_tab, comparison_tab, challenges_tab = st.tabs([
        "Análisis del Escenario Activo", "Comparar Escenarios", "Retos del DEA"
    ])

    with analysis_tab:
        app_status = 'initial'
        if active_scenario:
            app_status = active_scenario.get('app_status', 'initial')

        if app_status == "initial":
            renderer.render_upload_step()
        elif app_status == "data_loaded":
            renderer.render_preliminary_analysis_step(active_scenario)
        # ... (resto del flujo de la app sin cambios) ...

if __name__ == "__main__":
    main()
