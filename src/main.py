# src/main.py

import sys
import os

# Obtener la ruta del directorio donde se encuentra este script (main.py).
# Esta ruta es, efectivamente, la de la carpeta 'src'.
script_dir = os.path.dirname(__file__)

# Añadir esta ruta ('src') al sys.path si aún no está.
# Esto hará que todos los módulos hermanos (data_validator, results, etc.)
# y las subcarpetas (como dea_models) sean directamente importables por su nombre.
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import streamlit as st
import pandas as pd
import datetime

# Las importaciones ahora deberían funcionar directamente desde los nombres de los módulos
# ya que el directorio 'src' ha sido explícitamente añadido al sys.path.
from data_validator import validate
from results import mostrar_resultados, plot_benchmark_spider, plot_efficiency_histogram, plot_3d_inputs_outputs
from report_generator import generate_html_report, generate_excel_report
from session_manager import init_db, save_session, load_sessions
from inquiry_engine import generate_inquiry, to_plotly_tree
from epistemic_metrics import compute_eee


# -------------------------------------------------------
# 0) Configuración inicial de Streamlit
# -------------------------------------------------------
st.set_page_config(layout="wide")

# -------------------------------------------------------
# 1) Inicializar base de datos de sesiones
# -------------------------------------------------------
init_db()
default_user_id = "user_1"

# -------------------------------------------------------
# 2) Sidebar: cargar sesiones previas
# -------------------------------------------------------
st.sidebar.header("Simulador DEA – Sesiones Guardadas")

sessions = load_sessions(user_id=default_user_id)
selected_session_id = None
sess = {}

if sessions:
    ids = [s["session_id"] for s in sessions]
    selected_session_id = st.sidebar.selectbox("Seleccionar sesión para recargar", ids)
    if selected_session_id:
        sess = next(s for s in sessions if s["session_id"] == selected_session_id)
        st.sidebar.markdown(f"**Sesión:** {sess['session_id']}")
        st.sidebar.markdown(f"- Timestamp: {sess['timestamp']}")
        st.sidebar.markdown(f"- EEE Score: {sess['eee_score']}")
        st.sidebar.markdown(f"- Notas: {sess['notes']}")

        # Recargar el estado de la sesión si se seleccionó una
        if st.sidebar.button("Cargar esta sesión"):
            st.session_state.df = None # Limpiar datos actuales para cargar desde la sesión
            # Cargar DataFrame desde lista de diccionarios
            if 'df_data' in sess and isinstance(sess['df_data'], list):
                st.session_state.df = pd.DataFrame(sess['df_data']) 
            st.session_state.dmu_col = sess.get('dmu_col')
            st.session_state.input_cols = sess.get('input_cols', [])
            st.session_state.output_cols = sess.get('output_cols', [])
            
            # Cargar dea_results si está disponible y convertir diccionarios internos de vuelta a DataFrames
            if 'dea_results' in sess and isinstance(sess['dea_results'], dict):
                loaded_dea_results = {}
                for k, v in sess['dea_results'].items():
                    if isinstance(v, list): # Suponiendo que se guardó como to_dict('records')
                        loaded_dea_results[k] = pd.DataFrame(v)
                    else:
                        loaded_dea_results[k] = v # Mantener otros tipos tal cual
                st.session_state.dea_results = loaded_dea_results
            else:
                st.session_state.dea_results = None

            st.session_state.inquiry_tree = sess.get('inquiry_tree')

            if 'df_tree' in sess and isinstance(sess['df_tree'], list):
                st.session_state.df_tree = pd.DataFrame(sess['df_tree'])
            else:
                st.session_state.df_tree = pd.DataFrame()

            st.session_state.eee_score = sess.get('eee_score', 0.0)
            
            if 'df_eee' in sess and isinstance(sess['df_eee'], list):
                st.session_state.df_eee = pd.DataFrame(sess['df_eee'])
            else:
                st.session_state.df_eee = pd.DataFrame()

            st.session_state.selected_dmu = sess.get('selected_dmu')
            st.success(f"Sesión '{selected_session_id}' cargada.")
            st.experimental_rerun() # Volver a ejecutar para actualizar la página principal con los datos cargados

else:
    st.sidebar.write("No hay sesiones guardadas.")

st.sidebar.markdown("---")
st.sidebar.write("**Instrucciones:**")
st.sidebar.write("- Carga un CSV, selecciona columnas y ejecuta DEA.")
st.sidebar.write("- Luego podrás guardar y generar reportes.")
st.sidebar.markdown("---")

# -------------------------------------------------------
# 3) Asegurar claves en session_state antes de dibujar widgets
# -------------------------------------------------------
# Esto es crucial para evitar KeyError cuando Streamlit re-ejecuta el script
if "df" not in st.session_state:
    st.session_state.df = None
if "dmu_col" not in st.session_state:
    st.session_state.dmu_col = None
if "input_cols" not in st.session_state:
    st.session_state.input_cols = []
if "output_cols" not in st.session_state:
    st.session_state.output_cols = []
if "dea_results" not in st.session_state:
    st.session_state.dea_results = None
if "inquiry_tree" not in st.session_state: # Almacenar el diccionario del árbol de indagación en bruto
    st.session_state.inquiry_tree = None
if "df_tree" not in st.session_state: # Almacenar la representación del DataFrame del árbol para mostrar
    st.session_state.df_tree = None
if "eee_score" not in st.session_state: # Almacenar la puntuación EEE en bruto
    st.session_state.eee_score = 0.0
if "df_eee" not in st.session_state: # Almacenar la representación del DataFrame de EEE
    st.session_state.df_eee = None
if "selected_dmu" not in st.session_state:
    st.session_state.selected_dmu = None

# -------------------------------------------------------
# 4) Área principal: Título y carga de archivo CSV
# -------------------------------------------------------
st.title("Simulador Econométrico-Deliberativo – DEA")

uploaded_file = st.file_uploader("Cargar archivo CSV (con DMUs)", type=["csv"])

if uploaded_file is not None:
    st.session_state.df = pd.read_csv(uploaded_file)
    # Restablecer otras variables de estado de sesión cuando se carga un nuevo archivo
    st.session_state.dmu_col = None
    st.session_state.input_cols = []
    st.session_state.output_cols = []
    st.session_state.dea_results = None
    st.session_state.inquiry_tree = None
    st.session_state.df_tree = None
    st.session_state.eee_score = 0.0
    st.session_state.df_eee = None
    st.session_state.selected_dmu = None


# -------------------------------------------------------
# 5) Si ya hay DataFrame cargado, avanzar con selección
# -------------------------------------------------------
if st.session_state.df is not None:
    df = st.session_state.df.copy()
    st.subheader("Datos cargados")
    st.dataframe(df)

    # -------------------------------------------------------
    # 6) Selección de columnas: DMU, inputs y outputs
    # -------------------------------------------------------
    all_columns = df.columns.tolist()

    # 6.1) Selectbox para la columna DMU
    # Asegurar que se utiliza un índice predeterminado válido, o 0 si dmu_col no está en todas las columnas
    default_dmu_index = 0
    if st.session_state.dmu_col and st.session_state.dmu_col in all_columns:
        default_dmu_index = all_columns.index(st.session_state.dmu_col)
    
    st.session_state.dmu_col = st.selectbox(
        "Columna que identifica cada DMU",
        all_columns,
        index=default_dmu_index
    )

    # 6.2) Multiselect para inputs
    candidate_inputs = [c for c in all_columns if c != st.session_state.dmu_col]
    valid_input_defaults = [col for col in st.session_state.input_cols if col in candidate_inputs]
    st.session_state.input_cols = st.multiselect(
        "Seleccionar columnas de inputs",
        options=candidate_inputs,
        default=valid_input_defaults
    )

    # 6.3) Multiselect para outputs
    candidate_outputs = [c for c in all_columns if c not in st.session_state.input_cols + [st.session_state.dmu_col]]
    valid_output_defaults = [col for col in st.session_state.output_cols if col in candidate_outputs]
    st.session_state.output_cols = st.multiselect(
        "Seleccionar columnas de outputs",
        options=candidate_outputs,
        default=valid_output_defaults
    )

    # -------------------------------------------------------
    # 7) Botón para ejecutar DEA (CCR y BCC)
    # -------------------------------------------------------
    if st.button("Ejecutar DEA (CCR y BCC)"):
        if not st.session_state.input_cols or not st.session_state.output_cols:
            st.error("Por favor, selecciona al menos una columna de input y una de output.")
        else:
            errors = validate(df, st.session_state.input_cols, st.session_state.output_cols)
            llm_ready = errors.get("llm", {}).get("ready", True)

            if errors["formal_issues"] or not llm_ready:
                st.error("Se encontraron problemas en los datos o sugerencias del LLM:")
                if errors["formal_issues"]:
                    st.write("– Formal issues:")
                    for issue in errors["formal_issues"]:
                        st.write(f"  • {issue}")
                if "issues" in errors["llm"] and errors["llm"]["issues"]:
                    st.write("– LLM issues:")
                    for issue in errors["llm"]["issues"]:
                        st.write(f"  • {issue}")
            else:
                with st.spinner("Calculando eficiencias y generando árbol/EEE…"):
                    # Ejecutar modelos DEA
                    resultados = mostrar_resultados(
                        df.copy(),
                        st.session_state.dmu_col,
                        st.session_state.input_cols,
                        st.session_state.output_cols
                    )
                    st.session_state.dea_results = resultados

                    # Generar Árbol de Indagación
                    root_q = "Diagnóstico de ineficiencia y estrategias de mejora"
                    context_for_llm = {
                        "inputs": st.session_state.input_cols,
                        "outputs": st.session_state.output_cols,
                        "ccr_efficiencies_summary": st.session_state.dea_results["df_ccr"]["efficiency"].describe().to_dict(),
                        "bcc_efficiencies_summary": st.session_state.dea_results["df_bcc"]["efficiency"].describe().to_dict(),
                        "sample_data_head": df.head().to_dict('records')
                    }
                    st.session_state.inquiry_tree = generate_inquiry(root_q, context=context_for_llm)
                    
                    # Convertir árbol de indagación a DataFrame para mostrar
                    tree_data_list = []
                    def flatten_tree(node, parent_path=""):
                        for key, value in node.items():
                            current_path = f"{parent_path}/{key}" if parent_path else key
                            if isinstance(value, dict):
                                tree_data_list.append({"Nodo": key, "Padre": parent_path or "Raíz", "Tipo": "Pregunta/Categoría"})
                                flatten_tree(value, current_path)
                            else:
                                tree_data_list.append({"Nodo": key, "Padre": parent_path, "Tipo": "Sugerencia/Información", "Detalle": value})
                    
                    if st.session_state.inquiry_tree:
                        flatten_tree(st.session_state.inquiry_tree)
                    st.session_state.df_tree = pd.DataFrame(tree_data_list)
                    
                    # Calcular puntuación EEE
                    depth_limit = 5
                    breadth_limit = 5
                    st.session_state.eee_score = compute_eee(st.session_state.inquiry_tree, depth_limit, breadth_limit)
                    
                    st.session_state.df_eee = pd.DataFrame({
                        "Métrica": ["EEE Score Calculado"],
                        "Valor": [st.session_state.eee_score]
                    })

                st.success("Cálculos completados y árbol de indagación generado.")

    # -------------------------------------------------------
    # 8) Mostrar resultados si ya se calcularon
    # -------------------------------------------------------
    if st.session_state.dea_results is not None:
        resultados = st.session_state.dea_results
        df_ccr = resultados["df_ccr"]
        df_bcc = resultados["df_bcc"]

        st.subheader("Resultados CCR")
        st.dataframe(df_ccr)

        st.subheader("Resultados BCC")
        st.dataframe(df_bcc)

        st.subheader("Histograma de eficiencias CCR")
        st.plotly_chart(resultados["hist_ccr"], use_container_width=True)

        st.subheader("Histograma de eficiencias BCC")
        st.plotly_chart(resultados["hist_bcc"], use_container_width=True)

        st.subheader("Scatter 3D Inputs vs Output (CCR)")
        st.plotly_chart(resultados["scatter3d_ccr"], use_container_width=True)

        # -------------------------------------------------------
        # 9) Benchmark Spider CCR (uso dinámico de la columna DMU)
        # -------------------------------------------------------
        st.subheader("Benchmark Spider CCR")
        dmu_col = st.session_state.dmu_col
        dmu_options = df_ccr[dmu_col].astype(str).tolist()

        # Asegurar que selected_dmu sigue siendo válido si los datos cambiaron o se cargó un nuevo archivo
        if st.session_state.selected_dmu not in dmu_options:
            st.session_state.selected_dmu = dmu_options[0] if dmu_options else None

        # Manejar el caso en que dmu_options podría estar vacío
        selected_dmu_index = 0
        if st.session_state.selected_dmu and st.session_state.selected_dmu in dmu_options:
            selected_dmu_index = dmu_options.index(st.session_state.selected_dmu)
        elif not dmu_options: # Si dmu_options está vacío, establecer selected_dmu a None
            st.session_state.selected_dmu = None

        st.session_state.selected_dmu = st.selectbox(
            "Seleccionar DMU para comparar contra peers eficientes",
            dmu_options,
            index=selected_dmu_index
        )

        if st.session_state.selected_dmu:
            merged_ccr = df_ccr.merge(df, on=dmu_col, how="left")
            spider_fig = plot_benchmark_spider(
                merged_ccr,
                st.session_state.selected_dmu,
                st.session_state.input_cols,
                st.session_state.output_cols
            )
            st.plotly_chart(spider_fig, use_container_width=True)

        # -------------------------------------------------------
        # 10) Mostrar “Complejo de Indagación” (df_tree)
        # -------------------------------------------------------
        st.subheader("Complejo de Indagación (Árbol)")
        if st.session_state.df_tree is not None and not st.session_state.df_tree.empty:
            st.dataframe(st.session_state.df_tree)
            # Mostrar mapa de árbol de Plotly
            if st.session_state.inquiry_tree:
                tree_map_fig = to_plotly_tree(st.session_state.inquiry_tree)
                st.plotly_chart(tree_map_fig, use_container_width=True)
        else:
            st.write("No hay datos del árbol de indagación para mostrar.")

        # -------------------------------------------------------
        # 11) Mostrar métricas EEE
        # -------------------------------------------------------
        st.subheader("Métricas EEE")
        if st.session_state.df_eee is not None and not st.session_state.df_eee.empty:
            st.dataframe(st.session_state.df_eee)
        else:
            st.write("No hay métricas EEE para mostrar.")

        # -------------------------------------------------------
        # 12) Guardar sesión
        # -------------------------------------------------------
        st.subheader("Guardar esta sesión")
        
        # Inicializar notas con las existentes si se cargaron, o vacías
        current_notes = sess.get("notes", "") if selected_session_id else ""
        notes = st.text_area(
            "Notas sobre la sesión",
            value=current_notes
        )

        if st.button("Guardar sesión actual"):
            # Preparar df para guardar: convertir a diccionario si existe
            df_to_save = None
            if st.session_state.df is not None:
                df_to_save = st.session_state.df.to_dict('records') # Guardar como lista de diccionarios para serialización JSON
            
            # Preparar df_tree para guardar: convertir a diccionario si existe
            df_tree_to_save = None
            if st.session_state.df_tree is not None and not st.session_state.df_tree.empty:
                df_tree_to_save = st.session_state.df_tree.to_dict('records')

            # Preparar df_eee para guardar: convertir a diccionario si existe
            df_eee_to_save = None
            if st.session_state.df_eee is not None and not st.session_state.df_eee.empty:
                df_eee_to_save = st.session_state.df_eee.to_dict('records')

            serializable_dea_results = {}
            if st.session_state.dea_results:
                for key, value in st.session_state.dea_results.items():
                    # Solo guardar DataFrames como diccionarios, omitir figuras de Plotly
                    if isinstance(value, pd.DataFrame):
                        serializable_dea_results[key] = value.to_dict('records')
            
            # Guardar el estado completo de la sesión usando la función save_session mejorada
            save_session(
                user_id=default_user_id,
                inquiry_tree=st.session_state.inquiry_tree, # inquiry_tree es un dict, debería ser serializable
                eee_score=st.session_state.eee_score,
                notes=notes,
                dmu_col=st.session_state.dmu_col,
                input_cols=st.session_state.input_cols,
                output_cols=st.session_state.output_cols,
                df_data=df_to_save,
                dea_results=serializable_dea_results,
                df_tree_data=df_tree_to_save,
                df_eee_data=df_eee_to_save
            )
            st.success("Sesión guardada correctamente.")

        # -------------------------------------------------------
        # 13) Generar reportes
        # -------------------------------------------------------
        st.subheader("Generar reportes")

        # Reporte HTML
        html_str = generate_html_report(
            df_dea=df_ccr, # Solo se pasa CCR, considerar pasar ambos o uno combinado
            df_tree=st.session_state.df_tree,
            df_eee=st.session_state.df_eee
        )
        st.download_button(
            label="Descargar Reporte HTML",
            data=html_str,
            file_name=f"reporte_dea_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html"
        )

        # Reporte Excel
        excel_io = generate_excel_report(
            df_dea=df_ccr, # Solo se pasa CCR
            df_tree=st.session_state.df_tree,
            df_eee=st.session_state.df_eee
        )
        st.download_button(
            label="Descargar Reporte Excel",
            data=excel_io,
            file_name=f"reporte_dea_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
