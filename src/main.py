import sys
import os
import pandas as pd
import streamlit as st
import io
import json
import uuid 
import openai 
import plotly.express as px # Importar plotly.express para gráficos exploratorios

# --- 0) AJUSTE DEL PYTHONPATH Y CONFIGURACIÓN INICIAL ---
# Asegura que los módulos locales se puedan importar correctamente.
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Configuración de la página de Streamlit. "wide" aprovecha mejor el espacio.
st.set_page_config(layout="wide", page_title="DEA Deliberative Modeler")

# --- 1) IMPORTACIONES DE MÓDULOS ---
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
    """Crea un nuevo escenario, ya sea en blanco o clonando uno existente."""
    new_id = str(uuid.uuid4()) 
    
    if source_scenario_id and source_scenario_id in st.session_state.scenarios:
        st.session_state.scenarios[new_id] = st.session_state.scenarios[source_scenario_id].copy()
        st.session_state.scenarios[new_id]['name'] = f"Copia de {st.session_state.scenarios[source_scenario_id]['name']}"
        if st.session_state.scenarios[new_id]['dea_results']:
            st.session_state.scenarios[new_id]['dea_results'] = st.session_state.scenarios[source_scenario_id]['dea_results'].copy()
        if st.session_state.scenarios[new_id]['inquiry_tree']:
            st.session_state.scenarios[new_id]['inquiry_tree'] = st.session_state.scenarios[source_scenario_id]['inquiry_tree'].copy()
        st.session_state.scenarios[new_id]['user_justifications'] = {} 
        st.session_state.scenarios[new_id]['app_status'] = "data_loaded" if st.session_state.get("global_df") is not None else "initial"
        st.session_state.scenarios[new_id]['dea_results'] = None 
        st.session_state.scenarios[new_id]['inquiry_tree'] = None 
    else:
        st.session_state.scenarios[new_id] = {
            "name": name,
            "df": st.session_state.get("global_df", None), 
            "app_status": "initial", 
            "proposals_data": None,
            "selected_proposal": None,
            "dea_config": {},
            "dea_results": None,
            "inquiry_tree": None,
            "tree_explanation": None,
            "chart_to_show": None,
            "user_justifications": {}, 
            "data_overview": {} 
        }
    st.session_state.active_scenario_id = new_id

def get_active_scenario():
    """Devuelve el diccionario del escenario actualmente activo."""
    active_id = st.session_state.get('active_scenario_id')
    if active_id and active_id in st.session_state.scenarios:
        return st.session_state.scenarios[active_id]
    return None

def initialize_global_state():
    """Inicializa el estado global de la app."""
    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = {}
        st.session_state.active_scenario_id = None
        st.session_state.global_df = None

def reset_all():
    """Reinicia la aplicación a su estado inicial, eliminando todos los datos y escenarios."""
    cached_get_analysis_proposals.clear()
    cached_run_dea_analysis.clear()
    cached_run_inquiry_engine.clear()
    cached_explain_tree.clear()
    cached_generate_candidates.clear()
    cached_evaluate_candidates.clear()

    st.session_state.clear() 

    pass 


# --- 3) FUNCIONES DE CACHÉ Y LÓGICA DE IA ---

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

@st.cache_data
def cached_generate_candidates(_df, dmu_col, input_cols, output_cols, inquiry_tree, eee_score):
    return generate_candidates(_df, dmu_col, input_cols, output_cols, inquiry_tree, eee_score)

@st.cache_data
def cached_evaluate_candidates(_df, dmu_col, candidates, model):
    return evaluate_candidates(_df, dmu_col, candidates, model)


def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("La clave de API de OpenAI no ha sido configurada.")
        st.info("Añade tu clave 'OPENAI_API_KEY' en los 'Secrets' de la app y refresca la página.")
        return None 
    try:
        return openai.OpenAI(api_key=api_key) 
    except Exception as e:
        st.error(f"Error al inicializar el cliente de OpenAI: {e}")
        return None

def chat_completion(prompt: str, use_json_mode: bool = False):
    client = get_openai_client()
    if client is None: 
        return {"error": "API Key de OpenAI no configurada o error de inicialización.", "raw_content": "No se pudo conectar a OpenAI."}

    params = {"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}], "temperature": 0.5}
    if use_json_mode:
        params["response_format"] = {"type": "json_object"}
    
    try:
        return client.chat.completions.create(**params)
    except Exception as e:
        return {"error": f"Error al llamar a la API de OpenAI: {str(e)}", "raw_content": "Error en la llamada a la API."}


def generate_analysis_proposals(df_columns: list[str], df_head: pd.DataFrame):
    prompt = (
        "Eres un consultor experto en Data Envelopment Analysis (DEA). Has recibido un conjunto de datos con las siguientes columnas: "
        f"{df_columns}. A continuación se muestran las primeras filas:\n\n{df_head.to_string()}\n\n"
        "Tu tarea es proponer entre 2 y 4 modelos de análisis DEA distintos y bien fundamentados que se podrían aplicar a estos datos. "
        "Para cada propuesta, proporciona un título, un breve razonamiento sobre su utilidad y las listas de inputs y outputs sugeridas.\n\n"
        "Devuelve únicamente un objeto JSON válido con una sola clave raíz 'proposals'. El valor de 'proposals' debe ser una lista de objetos, donde cada objeto representa una propuesta y contiene las claves 'title', 'reasoning', 'inputs' y 'outputs'."
    )
    content = "No se recibió contenido."
    try:
        resp = chat_completion(prompt, use_json_mode=True)
        if isinstance(resp, dict) and resp.get("error"): 
            return {"error": f"Error al procesar la respuesta de la IA: {resp['error']}", "raw_content": resp['raw_content']}
        
        content = resp.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        return {"error": f"Error al procesar la respuesta de la IA: {str(e)}", "raw_content": content}

# --- FUNCIONES DE RENDERIZADO DE LA UI (Todas definidas antes de main()) ---

def render_upload_step():
    st.header("Paso 1: Carga tus Datos para Iniciar la Sesión", divider="blue")
    st.info("Para comenzar, sube tu conjunto de datos en formato CSV. Este fichero será la base para todos tus análisis DEA en esta sesión. Asegúrate de que la primera columna contenga los identificadores únicos de tus Unidades de Toma de Decisiones (DMUs).")
    uploaded_file = st.file_uploader("Sube un fichero CSV", type=["csv"], label_visibility="collapsed", help="Selecciona un archivo CSV desde tu ordenador. Un buen archivo CSV para DEA debe tener la primera columna como identificadores de DMU y las demás columnas como valores numéricos de inputs y outputs.")
    
    if uploaded_file:
        try:
            df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')))
        except Exception:
            df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('latin-1')), sep=';')
        
        st.session_state.global_df = df
        
        create_new_scenario(name="Modelo Base") 
        
        active_scenario = get_active_scenario() 

        data_overview = {
            "shape": df.shape,
            "column_types": df.dtypes.astype(str).to_dict(),
            "numerical_summary": df.describe(include='number').to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "non_numeric_issues": {}
        }
        
        zero_neg_issues = {}
        for col in df.select_dtypes(include='number').columns:
            if (df[col] <= 0).any():
                zero_neg_issues[col] = (df[col] <= 0).sum()
        data_overview["zero_negative_counts"] = zero_neg_issues

        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]) and not df[col].isnull().all():
                if pd.to_numeric(df[col], errors='coerce').isnull().any() and df[col].notnull().any():
                    data_overview["non_numeric_issues"][col] = True

        active_scenario['data_overview'] = data_overview 
        active_scenario['app_status'] = "data_loaded" 
        
        st.rerun() 
    
    if st.session_state.get('global_df') is not None:
        active_scenario = get_active_scenario() 
        if active_scenario and active_scenario.get('data_overview'):
            data_overview = active_scenario['data_overview']
            
            with st.expander("📊 Informe Rápido de los Datos Cargados", expanded=True):
                st.subheader("Dimensiones del DataFrame:", anchor=False)
                st.write(f"Filas: {data_overview['shape'][0]}, Columnas: {data_overview['shape'][1]}")

                st.subheader("Tipos de Datos por Columna:", anchor=False)
                df_types = pd.DataFrame(data_overview['column_types'].items(), columns=['Columna', 'Tipo de Dato'])
                st.dataframe(df_types, hide_index=True, help="Muestra el tipo de dato inferido por Streamlit para cada columna. Asegúrate de que tus variables de interés sean numéricas.")

                st.subheader("Resumen Estadístico (Columnas Numéricas):", anchor=False)
                df_numerical_summary = pd.DataFrame(data_overview['numerical_summary'])
                st.dataframe(df_numerical_summary, help="Estadísticas descriptivas básicas para las columnas numéricas. Revisa los valores mínimos y máximos.")

                st.subheader("Problemas Potenciales de Datos Detectados:", anchor=False)
                issues_found = False

                if any(data_overview['null_counts'].values()):
                    st.warning("⛔ Valores Nulos Detectados:")
                    df_nulls = pd.Series(data_overview['null_counts'])[pd.Series(data_overview['null_counts']) > 0].rename("Cantidad de Nulos")
                    st.dataframe(df_nulls.reset_index().rename(columns={'index': 'Columna'}), hide_index=True, help="Columnas que contienen valores nulos (vacíos). El DEA no puede procesar nulos.")
                    issues_found = True

                if data_overview['non_numeric_issues']:
                    st.error("❌ Columnas con Valores No Numéricos (Potenciales Errores):")
                    for col in data_overview['non_numeric_issues']:
                        st.write(f"- La columna '{col}' parece contener valores que no son números. Esto impedirá el análisis DEA.")
                    issues_found = True
                
                if data_overview['zero_negative_counts']:
                    st.warning("⚠️ Columnas Numéricas con Ceros o Valores Negativos:")
                    df_zero_neg = pd.Series(data_overview['zero_negative_counts'])[pd.Series(data_overview['zero_negative_counts']) > 0].rename("Cantidad (Cero/Negativo)")
                    st.dataframe(df_zero_neg.reset_index().rename(columns={'index': 'Columna'}), hide_index=True, help="El DEA tradicionalmente requiere valores positivos para los inputs y outputs. La presencia de ceros o negativos puede requerir transformaciones o el uso de modelos específicos.")
                    st.info("El DEA tradicionalmente requiere valores positivos para los inputs y outputs. Estos datos necesitarán atención en los pasos de validación y modelo.")
                    issues_found = True
                
                if not issues_found:
                    st.success("✅ No se detectaron problemas obvios (nulos, no numéricos, ceros/negativos) en este informe rápido.")
                else:
                    st.markdown("---")
                    st.warning("Se han detectado problemas potenciales en tus datos. Es **altamente recomendable** que realices una limpieza y preparación de tus datos antes de continuar para asegurar la validez de tu análisis DEA.")


            st.markdown("---")
            st.subheader("Guía para la Limpieza y Preparación de Datos")
            st.info("""
            Los **Retos de Datos** son uno de los principales desafíos en DEA. Para asegurar la validez de tu análisis, considera los siguientes puntos:
            * **Manejo de Nulos:** **Elimina** las filas con valores nulos o **rellénalos** con métodos apropiados (ej. media, mediana) *antes* de subir tu CSV.
            * **Valores Positivos:** Asegúrate de que todos los inputs y outputs sean estrictamente positivos ($>0$). Si tienes ceros o valores negativos, considera transformaciones (ej. añadir una constante muy pequeña) o el uso de modelos DEA que soporten estos valores.
            * **Outliers:** El DEA es sensible a los valores atípicos. **Investiga** si son errores de medición o valores reales, y decide si deben ser eliminados o ajustados.
            * **Homogeneidad:** Asegúrate de que las DMUs que comparas son realmente comparables. Factores contextuales o de tamaño pueden requerir **segmentación** de la muestra o el uso de variables contextuales.
            * **Tipo de Dato:** Confirma que todas las columnas que usarás como inputs/outputs sean **numéricas**.

            **Importante:** Esta aplicación no realiza la limpieza de datos por ti. Te recomendamos encarecidamente preparar y limpiar tus datos en una herramienta externa (ej. Excel, Python con Pandas) antes de subirlos para un análisis DEA óptimo.
            """)


def render_preliminary_analysis_step(active_scenario):
    st.header(f"Paso 1b: Exploración Preliminar de Datos para '{active_scenario['name']}'", divider="blue")
    st.info("Este paso es crucial para **entender tus datos** antes de realizar el análisis DEA. Te ayudará a identificar posibles problemas (como outliers o multicolinealidad) y a tomar decisiones informadas sobre la selección de inputs y outputs. La visualización es clave para el **pensamiento crítico** aquí.")

    df = active_scenario['df']
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

    if not numerical_cols:
        st.warning("No se encontraron columnas numéricas para realizar el análisis exploratorio. Asegúrate de que tu archivo CSV contenga datos numéricos.")
        if st.button("Proceder al Paso 2: Elegir Enfoque", key=f"proceed_to_step2_no_numeric_{st.session_state.active_scenario_id}"):
            active_scenario['app_status'] = "file_loaded"
            st.rerun()
        return

    if 'preliminary_analysis_charts' not in active_scenario['data_overview']:
        active_scenario['data_overview']['preliminary_analysis_charts'] = {}
    
    if 'correlation_matrix' not in active_scenario['data_overview'] or \
       active_scenario['data_overview'].get('last_df_hash') != hash(df.to_numpy().tobytes()): 
        active_scenario['data_overview']['correlation_matrix'] = df[numerical_cols].corr().to_dict()
        active_scenario['data_overview']['last_df_hash'] = hash(df.to_numpy().tobytes())


    st.subheader("1. Estadísticas Descriptivas:", anchor=False)
    st.markdown("Un resumen rápido de las características centrales, dispersión y forma de tus datos numéricos.")
    st.dataframe(df[numerical_cols].describe().T, help="Estadísticas descriptivas para todas las columnas numéricas: conteo, media, desviación estándar, valores mínimos y máximos, y cuartiles. Esto te da una primera idea de la distribución de tus variables.")

    st.subheader("2. Distribución de Variables (Histogramas):", anchor=False)
    st.markdown("Visualiza la distribución de cada variable numérica. Esto te ayuda a identificar asimetrías, rangos de valores y la presencia de posibles outliers.")
    
    if 'histograms' not in active_scenario['data_overview']['preliminary_analysis_charts']:
        active_scenario['data_overview']['preliminary_analysis_charts']['histograms'] = {}

    for col in numerical_cols:
        fig = px.histogram(df, x=col, title=f"Distribución de {col}", 
                           labels={col: col}, 
                           template="plotly_white")
        st.plotly_chart(fig, use_container_width=True, key=f"hist_{col}_{st.session_state.active_scenario_id}", help=f"Histograma de la columna '{col}'. Observa la forma de la distribución, si es simétrica, sesgada, o si hay valores atípicos.")
        active_scenario['data_overview']['preliminary_analysis_charts']['histograms'][col] = fig.to_json()


    st.subheader("3. Matriz de Correlación (Mapa de Calor):", anchor=False)
    st.markdown("Examina las relaciones lineales entre tus variables numéricas. Una alta correlación entre inputs o entre outputs puede indicar **multicolinealidad**, un reto potencial en DEA.")
    
    corr_matrix_dict = active_scenario['data_overview'].get('correlation_matrix', {})
    if corr_matrix_dict:
        corr_matrix = pd.DataFrame(corr_matrix_dict) 
        fig_corr = px.imshow(corr_matrix, 
                            text_auto=True, 
                            aspect="auto",
                            color_continuous_scale=px.colors.sequential.RdBu,
                            range_color=[-1,1],
                            title="Matriz de Correlación entre Variables Numéricas",
                            labels=dict(color="Correlación"))
        st.plotly_chart(fig_corr, use_container_width=True, key=f"corr_heatmap_{st.session_state.active_scenario_id}", help="Mapa de calor de la matriz de correlación. Valores cercanos a 1 o -1 indican fuerte correlación. Valores cercanos a 0 indican poca o ninguna correlación lineal. La alta correlación entre inputs u outputs puede indicar multicolinealidad, lo que puede afectar los pesos de las variables en DEA.")
    else:
        st.info("No se pudo generar la matriz de correlación. Asegúrate de tener al menos dos columnas numéricas.")
    
    st.markdown("---")
    st.subheader("Conclusiones de la Exploración Preliminar:")
    st.info("""
    Después de revisar estas visualizaciones:
    * **Identifica posibles Outliers:** ¿Hay puntos de datos que parecen muy diferentes del resto en los histogramas? Esto puede afectar la frontera de eficiencia en DEA.
    * **Evalúa la Multicolinealidad:** En el mapa de calor de correlación, ¿hay pares de inputs o de outputs con correlaciones muy altas (cercanas a 1 o -1)? Si es así, podría ser recomendable elegir solo una de esas variables en el Paso 2 para evitar redundancias y problemas de interpretación en los pesos del DEA.
    * **Distribución de Datos:** ¿Las distribuciones de tus variables son muy asimétricas? Esto podría influir en la robustez del modelo.

    Utiliza esta información para tomar decisiones más informadas al seleccionar tus inputs y outputs en el siguiente paso.
    """)

    if st.button("Proceder al Paso 2: Elegir Enfoque de Análisis", type="primary", use_container_width=True, help="Haz clic aquí para continuar y aplicar las ideas de esta exploración inicial a la selección de tus variables DEA."):
        active_scenario['app_status'] = "file_loaded" 
        st.rerun()

def render_proposal_step(active_scenario):
    st.header(f"Paso 2: Elige un Enfoque de Análisis para '{active_scenario['name']}'", divider="blue")
    st.info("En este paso, seleccionarás o definirás los **inputs** (recursos utilizados) y **outputs** (resultados producidos) que tu modelo DEA analizará. Esta es una decisión crítica que impacta directamente la validez de tus resultados.")
    st.info("**Reto de Datos: Selección de Insumos (Inputs) y Productos (Outputs).** La elección de las variables en DEA es crucial y puede ser subjetiva. Una selección inadecuada puede sesgar los resultados. Asegúrate de que tus inputs y outputs estén teóricamente justificados y sean relevantes para el proceso de eficiencia que deseas medir. Considera también la **homogeneidad de las DMUs**; solo deben compararse unidades que operen en entornos y con objetivos similares.")

    if not active_scenario.get('proposals_data'):
        with st.spinner("La IA está analizando tus datos para sugerir enfoques. Esto puede tardar un momento, ¡gracias por tu paciencia!"):
            active_scenario['proposals_data'] = cached_get_analysis_proposals(active_scenario['df'])
    
    proposals_data = active_scenario['proposals_data']
    proposals = proposals_data.get("proposals", [])
    
    if proposals_data.get("error"):
        st.error(f"Error al generar propuestas de la IA: {proposals_data['error']}. Contenido crudo: {proposals_data.get('raw_content', 'N/A')}")
        st.warning("No se pudieron generar propuestas automáticas de la IA. Por favor, procede con la configuración manual de inputs y outputs.")
        selected_option = "Configuración Manual"
    else:
        st.info("La IA ha preparado varias propuestas de enfoques de análisis DEA basadas en tus datos. Puedes seleccionar una de ellas o configurar tus propias variables manualmente.")
        options_list = ["Configuración Manual"] + [prop.get('title', f"Propuesta {i+1}") for i, prop in enumerate(proposals)]
        selected_option = st.selectbox(
            "Selecciona una opción:",
            options=options_list,
            key=f"proposal_selection_{st.session_state.active_scenario_id}",
            help="Elige una propuesta de la IA o selecciona 'Configuración Manual' para definir tus propias variables."
        )

    st.markdown("---")

    col_df_info, col_manual_config = st.columns([1, 2])

    with col_df_info:
        st.subheader("Datos Cargados:", anchor=False)
        st.markdown("Aquí puedes ver las primeras filas de tu conjunto de datos, lo que te ayudará a entender la estructura y las columnas disponibles.")
        st.dataframe(active_scenario['df'].head())
        st.markdown(f"**Columnas disponibles:** {', '.join(active_scenario['df'].columns.tolist())}")
        st.markdown(f"**Número de DMUs (Filas):** {len(active_scenario['df'])}")

    with col_manual_config:
        st.subheader("Detalles de la Propuesta Seleccionada:", anchor=False)
        selected_inputs = []
        selected_outputs = []
        proposal_title = ""
        proposal_reasoning = ""
        
        # Excluir la primera columna (asumida como DMU ID) de las opciones de selección de inputs/outputs
        all_cols_for_selection = [col for col in active_scenario['df'].columns.tolist() if col != active_scenario['df'].columns[0]]

        if selected_option == "Configuración Manual":
            proposal_title = "Configuración Manual del Modelo"
            proposal_reasoning = "El usuario ha definido las variables de forma personalizada."
            st.markdown("Define tus propios inputs y outputs para el análisis DEA. Recuerda que deben ser variables numéricas y positivas.")
            
            selected_inputs = st.multiselect(
                "Selecciona las columnas de **Inputs** (Insumos/Recursos):",
                options=all_cols_for_selection,
                default=[],
                key=f"manual_inputs_initial_{st.session_state.active_scenario_id}",
                help="Elige una o más columnas que representen los recursos que tus DMUs consumen."
            )
            selected_outputs = st.multiselect(
                "Selecciona las columnas de **Outputs** (Productos/Resultados):",
                options=all_cols_for_selection,
                default=[],
                key=f"manual_outputs_initial_{st.session_state.active_scenario_id}",
                help="Elige una o más columnas que representen los resultados que tus DMUs producen."
            )

        else:
            selected_ai_proposal = next((p for p in proposals if p.get('title') == selected_option), None)
            if selected_ai_proposal:
                proposal_title = selected_ai_proposal.get('title', '')
                proposal_reasoning = selected_ai_proposal.get('reasoning', '')
                selected_inputs = selected_ai_proposal.get('inputs', [])
                selected_outputs = selected_ai_proposal.get('outputs', [])

                st.markdown(f"**Título de la Propuesta:** {proposal_title}")
                st.markdown(f"**Razonamiento de la IA:** _{proposal_reasoning}_")
                st.markdown("La IA ha sugerido estas variables. Puedes ajustarlas si lo consideras necesario para refinar tu modelo.")
                
                selected_inputs = st.multiselect(
                    "Inputs sugeridos (puedes ajustar):",
                    options=all_cols_for_selection,
                    default=selected_inputs,
                    key=f"ai_inputs_adjustable_{st.session_state.active_scenario_id}",
                    help="Lista de inputs sugeridos por la IA. Puedes añadir o quitar variables."
                )
                selected_outputs = st.multiselect(
                    "Outputs sugeridos (puedes ajustar):",
                    options=all_cols_for_selection,
                    default=selected_outputs,
                    key=f"ai_outputs_adjustable_{st.session_state.active_scenario_id}",
                    help="Lista de outputs sugeridos por la IA. Puedes añadir o quitar variables."
                )
            else:
                st.warning("Propuesta no encontrada. Por favor, selecciona otra opción o ve a 'Configuración Manual'.")

        st.markdown("---")
        if st.button("Confirmar y Validar Configuración", type="primary", use_container_width=True, help="Guarda tu selección de inputs y outputs y pasa al paso de validación para asegurar que los datos cumplen los requisitos del DEA."):
            if not selected_inputs or not selected_outputs:
                st.error("Debes seleccionar al menos un input y un output para poder continuar.")
            else:
                active_scenario['selected_proposal'] = {
                    "title": proposal_title,
                    "reasoning": proposal_reasoning,
                    "inputs": selected_inputs,
                    "outputs": selected_outputs
                }
                active_scenario['app_status'] = "proposal_selected"
                st.rerun()

def render_upload_step():
    st.header("Paso 1: Carga tus Datos para Iniciar la Sesión", divider="blue")
    st.info("Para comenzar, sube tu conjunto de datos en formato CSV. Este fichero será la base para todos tus análisis DEA en esta sesión. Asegúrate de que la primera columna contenga los identificadores únicos de tus Unidades de Toma de Decisiones (DMUs).")
    uploaded_file = st.file_uploader("Sube un fichero CSV", type=["csv"], label_visibility="collapsed", help="Selecciona un archivo CSV desde tu ordenador. Un buen archivo CSV para DEA debe tener la primera columna como identificadores de DMU y las demás columnas como valores numéricos de inputs y outputs.")
    
    if uploaded_file:
        try:
            df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')))
        except Exception:
            df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('latin-1')), sep=';')
        
        st.session_state.global_df = df
        
        create_new_scenario(name="Modelo Base") 
        
        active_scenario = get_active_scenario() 

        data_overview = {
            "shape": df.shape,
            "column_types": df.dtypes.astype(str).to_dict(),
            "numerical_summary": df.describe(include='number').to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "non_numeric_issues": {}
        }
        
        zero_neg_issues = {}
        for col in df.select_dtypes(include='number').columns:
            if (df[col] <= 0).any():
                zero_neg_issues[col] = (df[col] <= 0).sum()
        data_overview["zero_negative_counts"] = zero_neg_issues

        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]) and not df[col].isnull().all():
                if pd.to_numeric(df[col], errors='coerce').isnull().any() and df[col].notnull().any():
                    data_overview["non_numeric_issues"][col] = True

        active_scenario['data_overview'] = data_overview 
        active_scenario['app_status'] = "data_loaded" 
        
        st.rerun() 
    
    if st.session_state.get('global_df') is not None:
        active_scenario = get_active_scenario() 
        if active_scenario and active_scenario.get('data_overview'):
            data_overview = active_scenario['data_overview']
            
            with st.expander("📊 Informe Rápido de los Datos Cargados", expanded=True):
                st.subheader("Dimensiones del DataFrame:", anchor=False)
                st.write(f"Filas: {data_overview['shape'][0]}, Columnas: {data_overview['shape'][1]}")

                st.subheader("Tipos de Datos por Columna:", anchor=False)
                df_types = pd.DataFrame(data_overview['column_types'].items(), columns=['Columna', 'Tipo de Dato'])
                st.dataframe(df_types, hide_index=True, help="Muestra el tipo de dato inferido por Streamlit para cada columna. Asegúrate de que tus variables de interés sean numéricas.")

                st.subheader("Resumen Estadístico (Columnas Numéricas):", anchor=False)
                df_numerical_summary = pd.DataFrame(data_overview['numerical_summary'])
                st.dataframe(df_numerical_summary, help="Estadísticas descriptivas básicas para las columnas numéricas. Revisa los valores mínimos y máximos.")

                st.subheader("Problemas Potenciales de Datos Detectados:", anchor=False)
                issues_found = False

                if any(data_overview['null_counts'].values()):
                    st.warning("⛔ Valores Nulos Detectados:")
                    df_nulls = pd.Series(data_overview['null_counts'])[pd.Series(data_overview['null_counts']) > 0].rename("Cantidad de Nulos")
                    st.dataframe(df_nulls.reset_index().rename(columns={'index': 'Columna'}), hide_index=True, help="Columnas que contienen valores nulos (vacíos). El DEA no puede procesar nulos.")
                    issues_found = True

                if data_overview['non_numeric_issues']:
                    st.error("❌ Columnas con Valores No Numéricos (Potenciales Errores):")
                    for col in data_overview['non_numeric_issues']:
                        st.write(f"- La columna '{col}' parece contener valores que no son números. Esto impedirá el análisis DEA.")
                    issues_found = True
                
                if data_overview['zero_negative_counts']:
                    st.warning("⚠️ Columnas Numéricas con Ceros o Valores Negativos:")
                    df_zero_neg = pd.Series(data_overview['zero_negative_counts'])[pd.Series(data_overview['zero_negative_counts']) > 0].rename("Cantidad (Cero/Negativo)")
                    st.dataframe(df_zero_neg.reset_index().rename(columns={'index': 'Columna'}), hide_index=True, help="El DEA tradicionalmente requiere valores positivos para los inputs y outputs. La presencia de ceros o negativos puede requerir transformaciones o el uso de modelos específicos.")
                    st.info("El DEA tradicionalmente requiere valores positivos para los inputs y outputs. Estos datos necesitarán atención en los pasos de validación y modelo.")
                    issues_found = True
                
                if not issues_found:
                    st.success("✅ No se detectaron problemas obvios (nulos, no numéricos, ceros/negativos) en este informe rápido.")
                else:
                    st.markdown("---")
                    st.warning("Se han detectado problemas potenciales en tus datos. Es **altamente recomendable** que realices una limpieza y preparación de tus datos antes de continuar para asegurar la validez de tu análisis DEA.")


            st.markdown("---")
            st.subheader("Guía para la Limpieza y Preparación de Datos")
            st.info("""
            Los **Retos de Datos** son uno de los principales desafíos en DEA. Para asegurar la validez de tu análisis, considera los siguientes puntos:
            * **Manejo de Nulos:** **Elimina** las filas con valores nulos o **rellénalos** con métodos apropiados (ej. media, mediana) *antes* de subir tu CSV.
            * **Valores Positivos:** Asegúrate de que todos los inputs y outputs sean estrictamente positivos ($>0$). Si tienes ceros o valores negativos, considera transformaciones (ej. añadir una constante muy pequeña) o el uso de modelos DEA que soporten estos valores.
            * **Outliers:** El DEA es sensible a los valores atípicos. **Investiga** si son errores de medición o valores reales, y decide si deben ser eliminados o ajustados.
            * **Homogeneidad:** Asegúrate de que las DMUs que comparas son realmente comparables. Factores contextuales o de tamaño pueden requerir **segmentación** de la muestra o el uso de variables contextuales.
            * **Tipo de Dato:** Confirma que todas las columnas que usarás como inputs/outputs sean **numéricas**.

            **Importante:** Esta aplicación no realiza la limpieza de datos por ti. Te recomendamos encarecidamente preparar y limpiar tus datos en una herramienta externa (ej. Excel, Python con Pandas) antes de subirlos para un análisis DEA óptimo.
            """)


def render_preliminary_analysis_step(active_scenario):
    st.header(f"Paso 1b: Exploración Preliminar de Datos para '{active_scenario['name']}'", divider="blue")
    st.info("Este paso es crucial para **entender tus datos** antes de realizar el análisis DEA. Te ayudará a identificar posibles problemas (como outliers o multicolinealidad) y a tomar decisiones informadas sobre la selección de inputs y outputs. La visualización es clave para el **pensamiento crítico** aquí.")

    df = active_scenario['df']
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

    if not numerical_cols:
        st.warning("No se encontraron columnas numéricas para realizar el análisis exploratorio. Asegúrate de que tu archivo CSV contenga datos numéricos.")
        if st.button("Proceder al Paso 2: Elegir Enfoque", key=f"proceed_to_step2_no_numeric_{st.session_state.active_scenario_id}"):
            active_scenario['app_status'] = "file_loaded"
            st.rerun()
        return

    if 'preliminary_analysis_charts' not in active_scenario['data_overview']:
        active_scenario['data_overview']['preliminary_analysis_charts'] = {}
    
    if 'correlation_matrix' not in active_scenario['data_overview'] or \
       active_scenario['data_overview'].get('last_df_hash') != hash(df.to_numpy().tobytes()): 
        active_scenario['data_overview']['correlation_matrix'] = df[numerical_cols].corr().to_dict()
        active_scenario['data_overview']['last_df_hash'] = hash(df.to_numpy().tobytes())


    st.subheader("1. Estadísticas Descriptivas:", anchor=False)
    st.markdown("Un resumen rápido de las características centrales, dispersión y forma de tus datos numéricos.")
    st.dataframe(df[numerical_cols].describe().T, help="Estadísticas descriptivas para todas las columnas numéricas: conteo, media, desviación estándar, valores mínimos y máximos, y cuartiles. Esto te da una primera idea de la distribución de tus variables.")

    st.subheader("2. Distribución de Variables (Histogramas):", anchor=False)
    st.markdown("Visualiza la distribución de cada variable numérica. Esto te ayuda a identificar asimetrías, rangos de valores y la presencia de posibles outliers.")
    
    if 'histograms' not in active_scenario['data_overview']['preliminary_analysis_charts']:
        active_scenario['data_overview']['preliminary_analysis_charts']['histograms'] = {}

    for col in numerical_cols:
        fig = px.histogram(df, x=col, title=f"Distribución de {col}", 
                           labels={col: col}, 
                           template="plotly_white")
        st.plotly_chart(fig, use_container_width=True, key=f"hist_{col}_{st.session_state.active_scenario_id}", help=f"Histograma de la columna '{col}'. Observa la forma de la distribución, si es simétrica, sesgada, o si hay valores atípicos.")
        active_scenario['data_overview']['preliminary_analysis_charts']['histograms'][col] = fig.to_json()


    st.subheader("3. Matriz de Correlación (Mapa de Calor):", anchor=False)
    st.markdown("Examina las relaciones lineales entre tus variables numéricas. Una alta correlación entre inputs o entre outputs puede indicar **multicolinealidad**, un reto potencial en DEA.")
    
    corr_matrix_dict = active_scenario['data_overview'].get('correlation_matrix', {})
    if corr_matrix_dict:
        corr_matrix = pd.DataFrame(corr_matrix_dict) 
        fig_corr = px.imshow(corr_matrix, 
                            text_auto=True, 
                            aspect="auto",
                            color_continuous_scale=px.colors.sequential.RdBu,
                            range_color=[-1,1],
                            title="Matriz de Correlación entre Variables Numéricas",
                            labels=dict(color="Correlación"))
        st.plotly_chart(fig_corr, use_container_width=True, key=f"corr_heatmap_{st.session_state.active_scenario_id}", help="Mapa de calor de la matriz de correlación. Valores cercanos a 1 o -1 indican fuerte correlación. Valores cercanos a 0 indican poca o ninguna correlación lineal. La alta correlación entre inputs u outputs puede indicar multicolinealidad, lo que puede afectar los pesos de las variables en DEA.")
    else:
        st.info("No se pudo generar la matriz de correlación. Asegúrate de tener al menos dos columnas numéricas.")
    
    st.markdown("---")
    st.subheader("Conclusiones de la Exploración Preliminar:")
    st.info("""
    Después de revisar estas visualizaciones:
    * **Identifica posibles Outliers:** ¿Hay puntos de datos que parecen muy diferentes del resto en los histogramas? Esto puede afectar la frontera de eficiencia en DEA.
    * **Evalúa la Multicolinealidad:** En el mapa de calor de correlación, ¿hay pares de inputs o de outputs con correlaciones muy altas (cercanas a 1 o -1)? Si es así, podría ser recomendable elegir solo una de esas variables en el Paso 2 para evitar redundancias y problemas de interpretación en los pesos del DEA.
    * **Distribución de Datos:** ¿Las distribuciones de tus variables son muy asimétricas? Esto podría influir en la robustez del modelo.

    Utiliza esta información para tomar decisiones más informadas al seleccionar tus inputs y outputs en el siguiente paso.
    """)

    if st.button("Proceder al Paso 2: Elegir Enfoque de Análisis", type="primary", use_container_width=True, help="Haz clic aquí para continuar y aplicar las ideas de esta exploración inicial a la selección de tus variables DEA."):
        active_scenario['app_status'] = "file_loaded" 
        st.rerun()

def render_proposal_step(active_scenario):
    st.header(f"Paso 2: Elige un Enfoque de Análisis para '{active_scenario['name']}'", divider="blue")
    st.info("En este paso, seleccionarás o definirás los **inputs** (recursos utilizados) y **outputs** (resultados producidos) que tu modelo DEA analizará. Esta es una decisión crítica que impacta directamente la validez de tus resultados.")
    st.info("**Reto de Datos: Selección de Insumos (Inputs) y Productos (Outputs).** La elección de las variables en DEA es crucial y puede ser subjetiva. Una selección inadecuada puede sesgar los resultados. Asegúrate de que tus inputs y outputs estén teóricamente justificados y sean relevantes para el proceso de eficiencia que deseas medir. Considera también la **homogeneidad de las DMUs**; solo deben compararse unidades que operen en entornos y con objetivos similares.")

    if not active_scenario.get('proposals_data'):
        with st.spinner("La IA está analizando tus datos para sugerir enfoques. Esto puede tardar un momento, ¡gracias por tu paciencia!"):
            active_scenario['proposals_data'] = cached_get_analysis_proposals(active_scenario['df'])
    
    proposals_data = active_scenario['proposals_data']
    proposals = proposals_data.get("proposals", [])
    
    if proposals_data.get("error"):
        st.error(f"Error al generar propuestas de la IA: {proposals_data['error']}. Contenido crudo: {proposals_data.get('raw_content', 'N/A')}")
        st.warning("No se pudieron generar propuestas automáticas de la IA. Por favor, procede con la configuración manual de inputs y outputs.")
        selected_option = "Configuración Manual"
    else:
        st.info("La IA ha preparado varias propuestas de enfoques de análisis DEA basadas en tus datos. Puedes seleccionar una de ellas o configurar tus propias variables manualmente.")
        options_list = ["Configuración Manual"] + [prop.get('title', f"Propuesta {i+1}") for i, prop in enumerate(proposals)]
        selected_option = st.selectbox(
            "Selecciona una opción:",
            options=options_list,
            key=f"proposal_selection_{st.session_state.active_scenario_id}",
            help="Elige una propuesta de la IA o selecciona 'Configuración Manual' para definir tus propias variables."
        )

    st.markdown("---")

    col_df_info, col_manual_config = st.columns([1, 2])

    with col_df_info:
        st.subheader("Datos Cargados:", anchor=False)
        st.markdown("Aquí puedes ver las primeras filas de tu conjunto de datos, lo que te ayudará a entender la estructura y las columnas disponibles.")
        st.dataframe(active_scenario['df'].head())
        st.markdown(f"**Columnas disponibles:** {', '.join(active_scenario['df'].columns.tolist())}")
        st.markdown(f"**Número de DMUs (Filas):** {len(active_scenario['df'])}")

    with col_manual_config:
        st.subheader("Detalles de la Propuesta Seleccionada:", anchor=False)
        selected_inputs = []
        selected_outputs = []
        proposal_title = ""
        proposal_reasoning = ""
        
        # Excluir la primera columna (asumida como DMU ID) de las opciones de selección de inputs/outputs
        all_cols_for_selection = [col for col in active_scenario['df'].columns.tolist() if col != active_scenario['df'].columns[0]]

        if selected_option == "Configuración Manual":
            proposal_title = "Configuración Manual del Modelo"
            proposal_reasoning = "El usuario ha definido las variables de forma personalizada."
            st.markdown("Define tus propios inputs y outputs para el análisis DEA. Recuerda que deben ser variables numéricas y positivas.")
            
            selected_inputs = st.multiselect(
                "Selecciona las columnas de **Inputs** (Insumos/Recursos):",
                options=all_cols_for_selection,
                default=[],
                key=f"manual_inputs_initial_{st.session_state.active_scenario_id}",
                help="Elige una o más columnas que representen los recursos que tus DMUs consumen."
            )
            selected_outputs = st.multiselect(
                "Selecciona las columnas de **Outputs** (Productos/Resultados):",
                options=all_cols_for_selection,
                default=[],
                key=f"manual_outputs_initial_{st.session_state.active_scenario_id}",
                help="Elige una o más columnas que representen los resultados que tus DMUs producen."
            )

        else:
            selected_ai_proposal = next((p for p in proposals if p.get('title') == selected_option), None)
            if selected_ai_proposal:
                proposal_title = selected_ai_proposal.get('title', '')
                proposal_reasoning = selected_ai_proposal.get('reasoning', '')
                selected_inputs = selected_ai_proposal.get('inputs', [])
                selected_outputs = selected_ai_proposal.get('outputs', [])

                st.markdown(f"**Título de la Propuesta:** {proposal_title}")
                st.markdown(f"**Razonamiento de la IA:** _{proposal_reasoning}_")
                st.markdown("La IA ha sugerido estas variables. Puedes ajustarlas si lo consideras necesario para refinar tu modelo.")
                
                selected_inputs = st.multiselect(
                    "Inputs sugeridos (puedes ajustar):",
                    options=all_cols_for_selection,
                    default=selected_inputs,
                    key=f"ai_inputs_adjustable_{st.session_state.active_scenario_id}",
                    help="Lista de inputs sugeridos por la IA. Puedes añadir o quitar variables."
                )
                selected_outputs = st.multiselect(
                    "Outputs sugeridos (puedes ajustar):",
                    options=all_cols_for_selection,
                    default=selected_outputs,
                    key=f"ai_outputs_adjustable_{st.session_state.active_scenario_id}",
                    help="Lista de outputs sugeridos por la IA. Puedes añadir o quitar variables."
                )
            else:
                st.warning("Propuesta no encontrada. Por favor, selecciona otra opción o ve a 'Configuración Manual'.")

        st.markdown("---")
        if st.button("Confirmar y Validar Configuración", type="primary", use_container_width=True, help="Guarda tu selección de inputs y outputs y pasa al paso de validación para asegurar que los datos cumplen los requisitos del DEA."):
            if not selected_inputs or not selected_outputs:
                st.error("Debes seleccionar al menos un input y un output para poder continuar.")
            else:
                active_scenario['selected_proposal'] = {
                    "title": proposal_title,
                    "reasoning": proposal_reasoning,
                    "inputs": selected_inputs,
                    "outputs": selected_outputs
                }
                active_scenario['app_status'] = "proposal_selected"
                st.rerun()

def render_validation_step(active_scenario):
    st.header(f"Paso 2b: Validación del Modelo para '{active_scenario['name']}'", divider="gray")
    st.info("Antes de ejecutar el análisis DEA, es fundamental validar la calidad de tus datos y la coherencia de tu selección de inputs y outputs. Esta sección te mostrará los resultados de una doble validación: formal y asistida por IA.")
    
    proposal = active_scenario.get('selected_proposal')
    
    if not proposal or not proposal.get('inputs') or not proposal.get('outputs'):
        st.error("La propuesta de análisis de este escenario está incompleta. Por favor, vuelve al Paso 2 para definir inputs y outputs.")
        return
    
    st.markdown(f"**Propuesta Seleccionada:** *{proposal.get('title', 'Configuración Manual')}*")
    st.markdown(f"**Inputs:** {proposal.get('inputs', [])}")
    st.markdown(f"**Outputs:** {proposal.get('outputs', [])}")
    st.markdown(f"**Razonamiento:** {proposal.get('reasoning', 'Configuración definida por el usuario o sugerida por la IA.')}")

    with st.spinner("La IA está validando la coherencia de los datos y el modelo... Esto puede tomar unos segundos."):
        validation_results = validate_data(active_scenario['df'], proposal['inputs'], proposal['outputs'])
        active_scenario['data_overview']['llm_validation_results'] = validation_results
    
    if validation_results['formal_issues']:
        st.error("**Reto de Datos: Datos Problemáticos.** Se encontraron problemas de validación formal en los datos o columnas seleccionadas. El DEA requiere que los inputs y outputs sean estrictamente positivos. La presencia de valores nulos, negativos o cero, o columnas no numéricas, puede causar errores o resultados inválidos en el modelo.")
        for issue in validation_results['formal_issues']:
            st.warning(f"- {issue}")
        st.info("Por favor, regresa al Paso 2 para ajustar las columnas o el dataset. Es crucial que los datos cumplan con los requisitos del DEA.")
    else:
        st.success("La validación formal inicial de datos y columnas ha sido exitosa. ¡Buen trabajo!")
    
    if validation_results['llm']['issues']:
        st.info("**Reto de Datos: Idoneidad y Homogeneidad.** La IA ha detectado posibles problemas conceptuales o sugerencias sobre la idoneidad de las variables o la homogeneidad de las DMUs. Considera estas observaciones para asegurar que tus DMUs son verdaderamente comparables y que las variables capturan el proceso de producción de forma adecuada. Resuelve estos puntos antes de proceder para asegurar la validez de tu análisis.")
        for issue in validation_results['llm']['issues']:
            st.warning(f"- {issue}")
        if validation_results['llm']['suggested_fixes']:
            st.markdown("**Sugerencias de la IA para mejorar:**")
            for fix in validation_results['llm']['suggested_fixes']:
                st.info(f"- {fix}")

    if not validation_results['formal_issues']: 
        if st.button("Proceder al Análisis", key=f"validate_{st.session_state.active_scenario_id}", type="primary", use_container_width=True, help="Si los resultados de validación son satisfactorios, haz clic aquí para pasar al siguiente paso y ejecutar el análisis DEA."):
            active_scenario['app_status'] = "validated"
            st.rerun()
    else:
        st.warning("Por favor, resuelve los problemas de validación formal antes de proceder al análisis. El DEA no funcionará correctamente con datos inválidos.")


def render_dea_challenges_tab():
    st.header("Retos Relevantes en el Uso del Análisis Envolvente de Datos (DEA)", divider="blue")
    st.markdown("""
    El Análisis Envolvente de Datos (DEA) es una herramienta potente, pero su aplicación exitosa depende de entender y abordar sus desafíos inherentes.
    """)

    st.subheader("1. Retos Relacionados con los Datos")
    st.markdown("""
    * **Selección de Insumos (Inputs) y Productos (Outputs):** Elegir las variables adecuadas es subjetivo y requiere justificación teórica. Una mala elección puede sesgar los resultados.
        * **La aplicación ayuda:** En el **Paso 2**, la IA sugiere inputs/outputs y permite la edición manual para asegurar la relevancia.
    * **Disponibilidad y Calidad de Datos:** Datos incompletos o erróneos pueden invalidar el análisis.
    * **Número de Variables vs. DMUs:** Demasiadas variables para pocas DMUs pueden inflar artificialmente la eficiencia.
    * **Valores Nulos, Negativos y Cero:** Los modelos DEA clásicos requieren datos positivos. Estos valores deben tratarse adecuadamente.
        * **La aplicación ayuda:** En el **Paso 1**, se ofrece un informe rápido de los datos cargados y una guía de preparación. En el **Paso 2b**, se realizan validaciones formales para detectar y advertir sobre estos problemas antes del análisis.
    * **Outliers (Valores Atípicos):** El DEA es sensible a los outliers, que pueden distorsionar la frontera de eficiencia.
    * **Homogeneidad de las DMUs:** Las unidades analizadas deben ser comparables entre sí. Comparar entidades muy dispares lleva a conclusiones erróneas.
        * **La aplicación ayuda:** En el **Paso 2**, se enfatiza la importancia de la homogeneidad, y la IA puede ofrecer sugerencias al respecto.
    """)

    st.subheader("2. Retos Metodológicos y de Especificación del Modelo")
    st.markdown("""
    * **Elección del Modelo DEA (CCR, BCC, SBM, etc.) y su Orientación:** La decisión sobre los rendimientos a escala (CRS vs. VRS) y la orientación (minimizar inputs vs. maximizar outputs) es crítica y afecta la forma de la frontera y las puntuaciones.
        * **La aplicación ayuda:** En el **Paso 3**, se guía al usuario en la selección del modelo y se ofrece información contextual sobre las implicaciones de cada elección.
    * **Falta de Pruebas de Significación Estadística:** El DEA no ofrece pruebas de significancia tradicionales, lo que dificulta generalizar los resultados.
    * **Sensibilidad del Modelo:** Los resultados pueden ser muy sensibles a pequeñas variaciones en los datos o en la inclusión/exclusión de DMUs.
        * **La aplicación ayuda:** En el **Paso 5**, la IA asiste en la exploración de configuraciones alternativas para evaluar la robustez del modelo.
    """)

    st.subheader("3. Retos de Interpretación y Aplicabilidad")
    st.markdown("""
    * **Interpretación de Puntuaciones de Eficiencia:** La eficiencia en DEA es *relativa* a la muestra, no absoluta.
    * **Identificación de Benchmarks:** Replicar las "mejores prácticas" de los benchmarks puede ser difícil en la realidad.
    * **Implicaciones de Política:** Traducir los resultados en acciones concretas requiere un profundo conocimiento del dominio.
    * **Dimensionalidad de la Proyección:** Entender las proyecciones para DMUs ineficientes puede ser complejo.
    * **La aplicación ayuda:** El **Paso 4 (Taller de Auditoría)** y los informes generados están diseñados para ayudar al investigador a deliberar, justificar y documentar la interpretación de los resultados, transformando el análisis cuantitativo en conocimiento accionable.
    """)

# --- 6) FLUJO PRINCIPAL DE LA APLICACIÓN ---
def main():
    """Función principal que orquesta la aplicación multi-escenario."""
    initialize_global_state() 

    st.sidebar.image("https://i.imgur.com/8y0N5c5.png", width=200)
    st.sidebar.title("DEA Deliberative Modeler")
    st.sidebar.markdown("Una herramienta para el análisis de eficiencia y la deliberación metodológica asistida por IA. Sigue los pasos para un estudio DEA robusto.")
    if st.sidebar.button("🔴 Empezar Nueva Sesión", help="Borra todos los datos y escenarios actuales para empezar un análisis desde cero. ¡Cuidado, esta acción no se puede deshacer!"):
        reset_all()
        st.rerun() 
    st.sidebar.divider()
    
    render_scenario_navigator()

    st.sidebar.markdown("---")
    st.sidebar.info("Una herramienta para el análisis de eficiencia y la deliberación metodológica asistida por IA.")

    active_scenario = get_active_scenario() 

    # Los tabs se definen una sola vez en el ámbito principal.
    analysis_tab, comparison_tab, challenges_tab = st.tabs([
        "Análisis del Escenario Activo", 
        "Comparar Escenarios", 
        "Retos del DEA"
    ])

    with analysis_tab:
        app_status = active_scenario.get('app_status', 'initial') if active_scenario else 'initial'

        if app_status == "initial":
            render_upload_step()
        elif app_status == "data_loaded":
            render_preliminary_analysis_step(active_scenario)
        elif app_status == "file_loaded": 
            render_proposal_step(active_scenario)
        elif app_status == "proposal_selected":
            render_validation_step(active_scenario)
        elif app_status in ["validated", "results_ready"]:
            render_main_dashboard(active_scenario)
    
    with comparison_tab:
        render_comparison_view()
    
    with challenges_tab:
        render_dea_challenges_tab()


if __name__ == "__main__":
    main()

