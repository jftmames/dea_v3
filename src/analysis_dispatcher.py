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
    period_column: str = None
) -> dict:
    """
    Ejecuta el análisis DEA seleccionado y devuelve un diccionario estandarizado con los resultados.
    """
    results = {"model_name": model_key, "main_df": pd.DataFrame(), "charts": {}}

    if model_key == 'CCR_BCC':
        results["model_name"] = "Radial (CCR y BCC)"
        df_ccr = run_ccr(df, dmu_column, inputs, outputs)

        # Verificación de robustez: se comprueba que el cálculo de CCR generó la columna esperada.
        if 'tec_efficiency_ccr' not in df_ccr.columns:
            raise ValueError(
                "El cálculo del modelo CCR no produjo la columna de resultados esperada ('tec_efficiency_ccr'). "
                "Esto puede deberse a un problema con la estructura de los datos de entrada que impide al solver encontrar una solución."
            )
        
        # Se crea una copia con la columna renombrada solo para el gráfico de histograma.
        df_ccr_hist = df_ccr.rename(columns={"tec_efficiency_ccr": "efficiency"})
        
        # CORRECCIÓN: Se pasa el DataFrame original 'df_ccr' a run_bcc.
        df_bcc = run_bcc(df, dmu_column, inputs, outputs, df_ccr_results=df_ccr)
        
        main_df = pd.DataFrame()
        # Se verifica que ambos DataFrames no estén vacíos antes de unirlos.
        if not df_ccr.empty and not df_bcc.empty:
            # Se renombra la columna de eficiencia de BCC para evitar conflictos en el merge.
            df_bcc_renamed = df_bcc.rename(columns={"efficiency": "pure_efficiency_bcc"})
            main_df = df_ccr.merge(
                df_bcc_renamed[[dmu_column, 'pure_efficiency_bcc', 'scale_efficiency', 'rts_label']],
                on=dmu_column
            )
        
        results["main_df"] = main_df
        results["charts"]["hist_ccr"] = plot_efficiency_histogram(df_ccr_hist)
        
        if not df_bcc.empty:
            results["charts"]["hist_bcc"] = plot_efficiency_histogram(df_bcc)
            
        return results

    elif model_key == 'SBM':
        results["model_name"] = "No Radial (SBM)"
        df_sbm = run_sbm(df, dmu_column, inputs, outputs, orientation="non-oriented")
        if 'efficiency_sbm' not in df_sbm.columns:
            raise ValueError("El cálculo del modelo SBM no produjo la columna de resultados esperada ('efficiency_sbm').")
            
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
        
        if not df_mpi.empty:
            avg_mpi = df_mpi.groupby('period_t1')[['MPI', 'efficiency_change', 'technical_change']].mean().reset_index()
            fig = px.line(avg_mpi, x='period_t1', y=['MPI', 'efficiency_change', 'technical_change'],
                          title="Evolución Promedio del Índice de Malmquist y sus Componentes")
            results["charts"]["mpi_evolution"] = fig
        return results
    
    else:
        raise NotImplementedError(f"El modelo '{model_key}' no está implementado en el despachador.")
