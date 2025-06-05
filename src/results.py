# jftmames/-dea-deliberativo-mvp/-dea-deliberativo-mvp-b44b8238c978ae0314af30717b9399634d28f8f9/src/results.py
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
    outputs: list[str]
) -> dict:
    """
    Versión completa que ejecuta ambos modelos (CCR y BCC),
    genera los DataFrames resultantes y produce todas las figuras de Plotly.
    Devuelve un diccionario con todos los resultados y figuras.
    """
    resultados = {}

    # 1) Ejecutar CCR
    df_ccr = run_ccr(
        df=df,
        dmu_column=dmu_column,
        input_cols=inputs,
        output_cols=outputs,
        orientation="input"
    )
    print(f"--- DEBUG: `df_ccr` es de tipo {type(df_ccr)} ---") # Mensaje de depuración

    # 2) Ejecutar BCC
    df_bcc = run_bcc(
        df=df,
        dmu_column=dmu_column,
        input_cols=inputs,
        output_cols=outputs,
        df_ccr_results=df_ccr
    )
    print(f"--- DEBUG: `df_bcc` es de tipo {type(df_bcc)} ---") # Mensaje de depuración
    
    # 3) Renombrar columnas de eficiencia para consistencia en los histogramas
    df_ccr_for_plots = df_ccr.rename(columns={"tec_efficiency_ccr": "efficiency"}) if df_ccr is not None else pd.DataFrame()
    
    resultados["df_ccr"] = df_ccr
    resultados["df_bcc"] = df_bcc

    # 4) Unir resultados con df original para visualizaciones (spider plot)
    # --- Añadimos una guarda para evitar el error si los dataframes son None ---
    if df_ccr is not None:
        merged_ccr = df_ccr.merge(df, on=dmu_column, how="left")
        resultados["merged_ccr"] = merged_ccr

    if df_bcc is not None:
        merged_bcc = df_bcc.merge(df, on=dmu_column, how="left")
        resultados["merged_bcc"] = merged_bcc

    # 5) Crear figuras de histograma
    if df_ccr is not None:
        hist_ccr = plot_efficiency_histogram(df_ccr_for_plots)
        resultados["hist_ccr"] = hist_ccr
    if df_bcc is not None:
        hist_bcc = plot_efficiency_histogram(df_bcc) 
        resultados["hist_bcc"] = hist_bcc

    # 6) Crear figuras de scatter 3D
    if df_ccr is not None and df_bcc is not None:
        scatter3d_ccr = plot_3d_inputs_outputs(df, inputs, outputs, df_ccr_for_plots, dmu_column)
        scatter3d_bcc = plot_3d_inputs_outputs(df, inputs, outputs, df_bcc, dmu_column)
        resultados["scatter3d_ccr"] = scatter3d_ccr
        resultados["scatter3d_bcc"] = scatter3d_bcc

    return resultados
