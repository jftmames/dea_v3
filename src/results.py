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
    model_type: str, # 'CCR' o 'BCC'
    orientation: str # 'input' o 'output'
) -> dict:
    """
    Ejecuta el modelo DEA seleccionado (CCR o BCC) con la orientación especificada.
    Devuelve un diccionario con todos los resultados y figuras para ese modelo.
    """
    resultados = {"model_type": model_type, "orientation": orientation}

    # Para calcular la eficiencia de escala en el modelo BCC, siempre necesitamos
    # ejecutar primero el CCR. El resultado se gestiona internamente.
    df_ccr_results = run_ccr(
        df=df,
        dmu_column=dmu_column,
        input_cols=inputs,
        output_cols=outputs,
        orientation=orientation,
        super_eff=False
    )
    # Guardamos una copia con el nombre original para el spider plot si es necesario
    df_ccr_for_spider = df_ccr_results.copy()
    
    if model_type == 'CCR':
        df_results = df_ccr_results.rename(columns={"tec_efficiency_ccr": "efficiency"})
        merged_df = df_ccr_for_spider.merge(df, on=dmu_column, how="left")
    elif model_type == 'BCC':
        df_results = run_bcc(
            df=df,
            dmu_column=dmu_column,
            input_cols=inputs,
            output_cols=outputs,
            df_ccr_results=df_ccr_results,
            orientation=orientation,
            super_eff=False
        )
        merged_df = df_results.merge(df, on=dmu_column, how="left")
    else:
        raise ValueError("El tipo de modelo debe ser 'CCR' o 'BCC'")

    # Almacenar resultados principales
    resultados["df_results"] = df_results
    resultados["merged_df"] = merged_df

    # Generar visualizaciones basadas en el resultado
    resultados["histogram"] = plot_efficiency_histogram(df_results)
    resultados["scatter_3d"] = plot_3d_inputs_outputs(df, inputs, outputs, df_results, dmu_column)
    
    return resultados
