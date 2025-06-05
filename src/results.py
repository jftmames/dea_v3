import pandas as pd
import plotly.express as px

from dea_models.radial import run_ccr, run_bcc
from dea_models.visualizations import (
    plot_benchmark_spider,
    plot_efficiency_histogram,
    plot_3d_inputs_outputs
)

def mostrar_resultados(
    df: pd.DataFrame,
    dmu_column: str,
    inputs: list[str],
    outputs: list[str],
    model_type: str,
    orientation: str
) -> dict:
    """
    Ejecuta el modelo DEA seleccionado (CCR o BCC) con la orientación especificada.
    Devuelve un diccionario con todos los resultados y figuras para ese modelo.
    """
    resultados = {"model_type": model_type, "orientation": orientation}

    # El modelo BCC necesita los resultados de CCR para calcular la eficiencia de escala.
    df_ccr_results = run_ccr(
        df=df,
        dmu_column=dmu_column,
        input_cols=inputs,
        output_cols=outputs,
        orientation=orientation
    )
    
    if model_type == 'CCR':
        df_main_results = df_ccr_results
    elif model_type == 'BCC':
        df_main_results = run_bcc(
            df=df,
            dmu_column=dmu_column,
            input_cols=inputs,
            output_cols=outputs,
            df_ccr_results=df_ccr_results,
            orientation=orientation
        )
    else:
        raise ValueError("El tipo de modelo debe ser 'CCR' o 'BCC'")

    # Renombrar columna de eficiencia para un nombre estándar en los gráficos
    if "tec_efficiency_ccr" in df_main_results.columns:
        df_plot_ready = df_main_results.rename(columns={"tec_efficiency_ccr": "efficiency"})
    else:
        df_plot_ready = df_main_results

    # Almacenar resultados principales
    resultados["df_results"] = df_main_results
    resultados["merged_df"] = df_main_results.merge(df, on=dmu_column, how="left")

    # Generar visualizaciones
    resultados["histogram"] = plot_efficiency_histogram(df_plot_ready)
    resultados["scatter_3d"] = plot_3d_inputs_outputs(df, inputs, outputs, df_plot_ready, dmu_column)
    
    return resultados
