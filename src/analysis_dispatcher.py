import pandas as pd
import plotly.express as px

# Importar todos los modelos necesarios de la biblioteca
from dea_models.radial import run_ccr, run_bcc
from dea_models.nonradial import run_sbm
from dea_models.mpi import compute_malmquist_phi
from dea_models.visualizations import plot_efficiency_histogram

def execute_analysis(
    df: pd.DataFrame,
    dmu_column: str,
    inputs: list[str],
    outputs: list[str],
    model_key: str,
    period_column: str = None # Nuevo parámetro para modelos de panel
) -> dict:
    """
    Ejecuta el análisis DEA seleccionado y devuelve un diccionario estandarizado con los resultados.
    
    Args:
        model_key (str): Clave que identifica el modelo a ejecutar (ej: 'CCR_BCC', 'SBM', 'MALMQUIST').
        period_column (str): Nombre de la columna de período, necesario para Malmquist.

    Returns:
        dict: Un diccionario con los resultados, incluyendo dataframes y figuras de plotly.
    """
    results = {"model_name": model_key, "main_df": pd.DataFrame(), "charts": {}}

    # --- Lógica de Despacho ---

    if model_key == 'CCR_BCC':
        results["model_name"] = "Radial (CCR y BCC)"
        df_ccr = run_ccr(df, dmu_column, inputs, outputs)
        # Para el histograma de CCR
        df_ccr_hist = df_ccr.rename(columns={"tec_efficiency_ccr": "efficiency"})
        
        # El modelo BCC necesita los resultados de CCR para calcular la eficiencia de escala
        df_bcc = run_bcc(df, dmu_column, inputs, outputs, df_ccr_results=df_ccr_hist)
        
        # Unimos ambos resultados para una visualización consolidada
        main_df = df_ccr.merge(
            df_bcc[[dmu_column, 'efficiency', 'scale_efficiency', 'rts_label']],
            on=dmu_column
        )
        main_df = main_df.rename(columns={"efficiency": "pure_efficiency_bcc"})
        
        results["main_df"] = main_df
        results["charts"]["hist_ccr"] = plot_efficiency_histogram(df_ccr_hist)
        results["charts"]["hist_bcc"] = plot_efficiency_histogram(df_bcc)
        return results

    elif model_key == 'SBM':
        results["model_name"] = "No Radial (SBM)"
        # El modelo SBM no orientado es una buena opción por defecto
        df_sbm = run_sbm(df, dmu_column, inputs, outputs, orientation="non-oriented")
        df_sbm_hist = df_sbm.rename(columns={"efficiency_sbm": "efficiency"})

        results["main_df"] = df_sbm
        results["charts"]["hist_sbm"] = plot_efficiency_histogram(df_sbm_hist)
        return results

    elif model_key == 'MALMQUIST':
        results["model_name"] = "Índice de Productividad de Malmquist"
        if not period_column or period_column not in df.columns:
            raise ValueError("Para el modelo Malmquist, se debe proporcionar una columna de período válida.")
            
        df_mpi = compute_malmquist_phi(df, dmu_column, period_column, inputs, outputs)
        results["main_df"] = df_mpi
        # Podríamos añadir un gráfico de la evolución promedio del MPI
        if not df_mpi.empty:
            avg_mpi = df_mpi.groupby('period_t1')[['MPI', 'efficiency_change', 'technical_change']].mean().reset_index()
            fig = px.line(avg_mpi, x='period_t1', y=['MPI', 'efficiency_change', 'technical_change'],
                          title="Evolución Promedio del Índice de Malmquist y sus Componentes")
            results["charts"]["mpi_evolution"] = fig
        return results
    
    # Se podrían añadir más modelos (Window, Network, etc.) aquí
    
    else:
        raise NotImplementedError(f"El modelo '{model_key}' no está implementado en el despachador.")
