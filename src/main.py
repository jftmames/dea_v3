# src/main.py

import streamlit as st
import pandas as pd
import datetime

from data_validator import validate
from results import mostrar_resultados, plot_benchmark_spider
from report_generator import generate_html_report, generate_excel_report
from session_manager import init_db, save_session, load_sessions

# -------------------------------------------------------
# 0) Configuración inicial de Streamlit
# -------------------------------------------------------
st.set_page_config(layout="wide")

# -------------------------------------------------------
# 1) Inicialización de la base de datos de sesiones
# -------------------------------------------------------
init_db()
default_user_id = "user_1"

# -------------------------------------------------------
# 2) Sidebar: cargar sesiones previas
# -------------------------------------------------------
st.sidebar.header("Simulador DEA - Sesiones Guardadas")

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

        # Recargar datos de la sesión seleccionada
        # Aquí asumimos que tienes una forma de cargar el DataFrame original de la sesión
        # Por ahora, simularemos la carga o podrías tener un campo 'df_path' en tu DB
        # Para propósitos de este código, no se recarga el df original automáticamente,
        # pero las selecciones de columnas sí se restauran si el DF actual lo permite.
        
        # Actualizar st.session_state con los datos de la sesión cargada
        # Cargar los DataFrames de CCR y BCC de la sesión
        if 'df_ccr' in sess and 'df_bcc' in sess:
            st.session_state.dea_results = {
                "df_ccr": sess['df_ccr'],
                "df_bcc": sess['df_bcc'],
                # Estas figuras no se guardan en la DB, se recrearían si es necesario
                # Por simplicidad, no las regeneramos aquí al cargar la sesión
                "hist_ccr": None,
                "hist_bcc": None,
                "scatter3d_ccr": None,
                "scatter3d_bcc": None,
            }
        
        if 'dmu_column' in sess:
            st.session_state.dmu_col = sess['dmu_column']
        if 'input_cols' in sess:
            st.session_state.input_cols = sess['input_cols']
        if 'output_cols' in sess:
            st.session_state.output_cols = sess['output_cols']
        
        # Estos pueden no estar en todas las sesiones, si la sesión se guardó antes de generarlos
        st.session_state.df_tree = sess.get("inquiry_tree", pd.DataFrame()) # Asumiendo que inquiry_tree se guarda aquí
        st.session_state.df_eee = pd.DataFrame([{"EEE Score": sess.get("eee_score", 0.0), "Notes": sess.get("notes", "")}])


else:
    st.sidebar.write("No hay sesiones guardadas.")

st.sidebar.markdown("---")
st.sidebar.write("**Instrucciones:**")
st.sidebar.write("- Carga un CSV, selecciona columnas y ejecuta DEA.")
st.sidebar.write("- Luego podrás guardar y generar reportes.")
st.sidebar.markdown("---")

# -------------------------------------------------------
# 3) Área principal: Título y carga de archivo CSV
# -------------------------------------------------------
st.title("Simulador Econométrico-Deliberativo – DEA")

# Inicializar valores en session_state
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
if "df_tree" not in st.session_state:
    st.session_state.df_tree = None
if "df_eee" not in st.session_state:
    st.session_state.df_eee = None
if "selected_dmu" not in st.session_state:
    st.session_state.selected_dmu = None


uploaded_file = st.file_uploader("Cargar archivo CSV (con DMUs)", type=["csv"])

if uploaded_file is not None:
    st.session_state.df = pd.read_csv(uploaded_file)
    # Resetear selecciones al cargar un nuevo archivo para evitar errores si las columnas cambian
    # st.session_state.dmu_col = None # Podría ser útil resetear, pero lo dejamos a la lógica de selectbox
    st.session_state.input_cols = []
    st.session_state.output_cols = []
    st.session_state.dea_results = None # Limpiar resultados anteriores
    st.session_state.df_tree = None
    st.session_state.df_eee = None


# Mostrar DataFrame si ya existe
if st.session_state.df is not None:
    df = st.session_state.df.copy()
    st.subheader("Datos cargados")
    st.dataframe(df)

    # -------------------------------------------------------
    # 4) Selección de columnas: DMU, inputs y outputs
    # -------------------------------------------------------
    all_columns = df.columns.tolist()

    # Selección de DMU
    st.session_state.dmu_col = st.selectbox(
        "Columna que identifica cada DMU",
        all_columns,
        index=all_columns.index(st.session_state.dmu_col) if st.session_state.dmu_col in all_columns else (0 if all_columns else None) # Añadir manejo de lista vacía
    )

    # ---------------------------------------------------
    # Inputs
    # ---------------------------------------------------
    candidate_inputs = [c for c in all_columns if c != st.session_state.dmu_col]

    # Filtramos session_state.input_cols para que sólo quede lo que sigue existiendo en candidate_inputs
    valid_input_defaults = [
        col for col in st.session_state.input_cols if col in candidate_inputs
    ]
    st.session_state.input_cols = st.multiselect(
        "Seleccionar columnas de inputs",
        options=candidate_inputs,
        default=valid_input_defaults
    )

    # ---------------------------------------------------
    # Outputs
    # ---------------------------------------------------
    candidate_outputs = [
        c for c in all_columns
        if c not in st.session_state.input_cols + [st.session_state.dmu_col]
    ]
    valid_output_defaults = [
        col for col in st.session_state.output_cols if col in candidate_outputs
    ]
    st.session_state.output_cols = st.multiselect(
        "Seleccionar columnas de outputs",
        options=candidate_outputs,
        default=valid_output_defaults
    )

    # -------------------------------------------------------
    # 5) Botón para ejecutar DEA (CCR y BCC)
    # -------------------------------------------------------
    if st.button("Ejecutar DEA (CCR y BCC)"):
        # Asegurarse de que al menos un input y un output estén seleccionados
        if not st.session_state.input_cols or not st.session_state.output_cols:
            st.error("Por favor, selecciona al menos una columna de input y una de output.")
        else:
            errors = validate(df, st.session_state.input_cols, st.session_state.output_cols)
            llm_ready = errors.get("llm", {}).get("ready", True) # Asumir True si no hay respuesta LLM

            if errors["formal_issues"] or not llm_ready:
                st.error("Se encontraron problemas en los datos o sugerencias del LLM:")
                if errors["formal_issues"]:
                    st.write("– Problemas formales:")
                    for issue in errors["formal_issues"]:
                        st.write(f"  • {issue}")
                if "issues" in errors["llm"] and errors["llm"]["issues"]:
                    st.write("– Problemas del LLM:")
                    for issue in errors["llm"]["issues"]:
                        st.write(f"  • {issue}")
                if "raw" in errors["llm"]: # Mostrar el contenido crudo si no es JSON válido
                    st.warning("Respuesta cruda del LLM (no JSON válido):")
                    st.code(errors["llm"]["raw"])
            else:
                with st.spinner("Calculando eficiencias y generando árbol/EEE…"):
                    # pasar una copia de df para evitar Side effects
                    resultados = mostrar_resultados(
                        df.copy(), # Se pasa una copia de df
                        st.session_state.dmu_col,
                        st.session_state.input_cols,
                        st.session_state.output_cols
                    )
                st.session_state.dea_results = resultados
                # df_tree y df_eee deberían ser parte de 'resultados' si se generaran ahí.
                # Si no, se mantienen como None o se definen aquí si es que se calculan
                # en otro lugar o se cargan de 'sess'.
                # Por ahora, asumo que se pasan resultados.get("df_tree")
                # pero el código de `mostrar_resultados` no devuelve `df_tree` ni `df_eee`.
                # Necesitarías actualizar `mostrar_resultados` para que devuelva esos DFs.
                # Para este ejemplo, si no están, se mantienen como None o se usan DataFrames vacíos.
                st.session_state.df_tree = resultados.get("df_tree", pd.DataFrame()) # Asumo que mostrar_resultados devolverá esto
                st.session_state.df_eee = resultados.get("df_eee", pd.DataFrame()) # Asumo que mostrar_resultados devolverá esto
                st.success("Cálculos DEA completados.")


    # -------------------------------------------------------
    # 6) Mostrar resultados si ya se calcularon
    # -------------------------------------------------------
    if st.session_state.dea_results is not None:
        resultados = st.session_state.dea_results
        df_ccr = resultados["df_ccr"]
        df_bcc = resultados["df_bcc"]

        st.subheader("Resultados CCR")
        st.dataframe(df_ccr)

        st.subheader("Resultados BCC")
        st.dataframe(df_bcc)

        # Regenerar las figuras si se cargó la sesión y son None
        if resultados["hist_ccr"] is None:
            resultados["hist_ccr"] = plot_efficiency_histogram(df_ccr)
        if resultados["hist_bcc"] is None:
            resultados["hist_bcc"] = plot_efficiency_histogram(df_bcc)
        if resultados["scatter3d_ccr"] is None:
            resultados["scatter3d_ccr"] = plot_3d_inputs_outputs(df, st.session_state.input_cols, st.session_state.output_cols, df_ccr, st.session_state.dmu_col)
        if resultados["scatter3d_bcc"] is None:
            resultados["scatter3d_bcc"] = plot_3d_inputs_outputs(df, st.session_state.input_cols, st.session_state.output_cols, df_bcc, st.session_state.dmu_col)


        st.subheader("Histograma de eficiencias CCR")
        st.plotly_chart(resultados["hist_ccr"], use_container_width=True)

        st.subheader("Histograma de eficiencias BCC")
        st.plotly_chart(resultados["hist_bcc"], use_container_width=True)

        # Solo mostrar scatter 3D si hay al menos 2 inputs y 1 output para un gráfico significativo
        if len(st.session_state.input_cols) >= 2 and len(st.session_state.output_cols) >= 1:
            st.subheader("Scatter 3D Inputs vs Output (CCR)")
            st.plotly_chart(resultados["scatter3d_ccr"], use_container_width=True)
            st.subheader("Scatter 3D Inputs vs Output (BCC)")
            st.plotly_chart(resultados["scatter3d_bcc"], use_container_width=True)
        elif len(st.session_state.input_cols) == 1 and len(st.session_state.output_cols) >= 1:
            st.info("Para visualizar un Scatter 3D, se necesitan al menos dos inputs o dos outputs. Mostrando Scatter 2D si es posible.")
            # Podrías agregar un gráfico 2D aquí como alternativa si lo deseas
        else:
            st.info("No hay suficientes inputs/outputs para generar un Scatter 3D.")


        # -------------------------------------------------------
        # 7) Benchmark Spider CCR
        # -------------------------------------------------------
        st.subheader("Benchmark Spider CCR")
        dmu_options = df_ccr[st.session_state.dmu_col].astype(str).tolist() # Usar st.session_state.dmu_col
        
        # Ajustar el índice por defecto si la DMU seleccionada ya no existe en las opciones actuales
        default_index_dmu = 0
        if st.session_state.selected_dmu in dmu_options:
            default_index_dmu = dmu_options.index(st.session_state.selected_dmu)

        st.session_state.selected_dmu = st.selectbox(
            "Seleccionar DMU para comparar contra peers eficientes",
            dmu_options,
            index=default_index_dmu
        )

        if st.session_state.selected_dmu:
            # Asegurarse de que 'merged_ccr' esté disponible o recrearlo
            # La función mostrar_resultados ya devuelve merged_ccr, pero si se carga la sesión, puede que no esté
            # Por simplicidad, recreamos merged_ccr para plot_benchmark_spider si es necesario.
            # Idealmente, 'mostrar_resultados' debería devolver 'merged_ccr' y 'merged_bcc'
            # y los usaríamos directamente desde `resultados`.
            merged_ccr_for_spider = resultados.get("merged_ccr")
            if merged_ccr_for_spider is None:
                merged_ccr_for_spider = df_ccr.merge(df, on=st.session_state.dmu_col, how="left")

            spider_fig = plot_benchmark_spider(
                merged_ccr_for_spider, # Usar el merged_ccr adecuado
                st.session_state.selected_dmu,
                st.session_state.input_cols,
                st.session_state.output_cols
            )
            st.plotly_chart(spider_fig, use_container_width=True)


        # -------------------------------------------------------
        # 8) Mostrar el "Complejo de indagación" (df_tree)
        # -------------------------------------------------------
        st.subheader("Complejo de Indagación (Árbol)")
        if st.session_state.df_tree is not None and not st.session_state.df_tree.empty:
            st.dataframe(st.session_state.df_tree)
        else:
            st.write("No hay datos del árbol de indagación para mostrar.")


        # -------------------------------------------------------
        # 9) Mostrar métricas EEE
        # -------------------------------------------------------
        st.subheader("Métricas EEE")
        if st.session_state.df_eee is not None and not st.session_state.df_eee.empty:
            st.dataframe(st.session_state.df_eee)
        else:
            st.write("No hay métricas EEE para mostrar.")


        # -------------------------------------------------------
        # 10) Guardar sesión
        # -------------------------------------------------------
        st.subheader("Guardar esta sesión")
        
        # Obtener los valores actuales de inquiry_tree y eee_score,
        # o los de la sesión cargada si no se han generado nuevos.
        current_inquiry_tree = st.session_state.df_tree.to_dict(orient='records') if st.session_state.df_tree is not None else {}
        current_eee_score = st.session_state.df_eee["EEE Score"].iloc[0] if st.session_state.df_eee is not None and not st.session_state.df_eee.empty else 0.0
        
        notes = st.text_area(
            "Notas sobre la sesión",
            value=sess.get("notes", "") if selected_session_id else "" # Valor por defecto desde la sesión cargada
        )

        if st.button("Guardar sesión actual"):
            # Generar un session_id único para la nueva sesión
            session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Asegúrate de pasar todos los argumentos requeridos por save_session
            save_session(
                session_id=session_id,
                user_id=default_user_id,
                inquiry_tree=current_inquiry_tree,
                eee_score=current_eee_score,
                notes=notes,
                dmu_column=st.session_state.dmu_col,
                input_cols=st.session_state.input_cols,
                output_cols=st.session_state.output_cols,
                df_ccr=df_ccr, # Usar el df_ccr actual
                df_bcc=df_bcc  # Usar el df_bcc actual
            )
            st.success(f"Sesión '{session_id}' guardada correctamente.")


        # -------------------------------------------------------
        # 11) Generar reportes
        # -------------------------------------------------------
        st.subheader("Generar reportes")

        # 11.1) Reporte HTML
        html_str = generate_html_report(
            df_dea=df_ccr, # Usar df_ccr para la tabla DEA
            df_tree=st.session_state.df_tree,
            df_eee=st.session_state.df_eee
        )
        st.download_button(
            label="Descargar Reporte HTML",
            data=html_str,
            file_name=f"reporte_dea_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html"
        )

        # 11.2) Reporte Excel
        excel_io = generate_excel_report(
            df_dea=df_ccr, # Usar df_ccr para la tabla DEA
            df_tree=st.session_state.df_tree,
            df_eee=st.session_state.df_eee
        )
        st.download_button(
            label="Descargar Reporte Excel",
            data=excel_io,
            file_name=f"reporte_dea_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
