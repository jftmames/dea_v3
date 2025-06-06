# --- 1) IMPORTACIONES DE MÓDULOS DEL PROYECTO (ACTUALIZADO) ---
# ... (importaciones existentes)
from analysis_dispatcher import execute_analysis # <-- Reemplaza a 'mostrar_resultados'

# --- 2) GESTIÓN DE ESTADO (SIN CAMBIOS) ---
# ...

# --- 3) FUNCIONES DE CACHÉ (ACTUALIZADO) ---
@st.cache_data
def cached_run_dea_analysis(_df, dmu_col, input_cols, output_cols, model_key, period_col):
    """Cachea los resultados del análisis DEA para cualquier modelo."""
    return execute_analysis(_df.copy(), dmu_col, input_cols, output_cols, model_key, period_column=period_col)

# ... (otras funciones de caché)

# --- 4) COMPONENTES DE LA UI (ACTUALIZADO) ---

# ... (render_eee_explanation, render_hypothesis_workshop sin cambios)

def render_main_dashboard():
    """Renderiza el dashboard principal, ahora dinámico según el modelo."""
    st.header(f"Analizando: '{st.session_state.selected_proposal['title']}'", divider="blue")
    
    # --- WIDGET PARA SELECCIÓN DE MODELO ---
    model_options = {
        "Radial (CCR/BCC)": "CCR_BCC",
        "No Radial (SBM)": "SBM",
        "Productividad (Malmquist)": "MALMQUIST"
    }
    model_name = st.selectbox("Selecciona el tipo de modelo DEA a aplicar:", list(model_options.keys()))
    model_key = model_options[model_name]

    # Widget condicional para la columna de período
    period_col = None
    if model_key == 'MALMQUIST':
        # Asumimos que la columna de período es la segunda columna si no se especifica
        period_col_options = [None] + st.session_state.df.columns.tolist()
        period_col = st.selectbox("Selecciona la columna que identifica el período:", period_col_options, index=2)
        if not period_col:
            st.warning("El modelo Malmquist requiere una columna de período.")
            st.stop()

    # --- EJECUCIÓN DEL ANÁLISIS ---
    if st.button(f"Ejecutar Análisis con Modelo {model_name}", use_container_width=True):
        with st.spinner(f"Ejecutando análisis con {model_name}..."):
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

    # --- VISUALIZACIÓN DINÁMICA DE RESULTADOS ---
    if st.session_state.get("dea_results"):
        results = st.session_state.dea_results
        st.subheader(f"Resultados para: {results['model_name']}", divider="gray")
        
        st.dataframe(results['main_df'])
        
        # Mostrar los gráficos generados por el despachador
        if results.get("charts"):
            for chart_title, fig in results["charts"].items():
                st.plotly_chart(fig, use_container_width=True)

        # El taller deliberativo puede seguir funcionando con los resultados radiales si existen
        if "CCR_BCC" in results["model_name"]:
             # --- TALLER DE RAZONAMIENTO Y DELIBERACIÓN ---
             # ... (el código del taller y EEE iría aquí, sin cambios)
             pass

# ... (resto de funciones de renderizado, como render_proposal_step, etc., actualizadas para
#      llamar al dashboard principal de la nueva manera)

# --- 5) FLUJO PRINCIPAL DE LA APLICACIÓN (ACTUALIZADO) ---
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

    # El dashboard principal se muestra después de la validación
    elif st.session_state.app_status in ["validated", "results_ready"]:
        render_main_dashboard()


if __name__ == "__main__":
    main()
