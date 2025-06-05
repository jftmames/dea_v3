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
        orientation="input",
        super_eff=False
    )

    # 2) Ejecutar BCC pasando los resultados de CCR para optimizar
    df_bcc = run_bcc(
        df=df,
        dmu_column=dmu_column,
        input_cols=inputs,
        output_cols=outputs,
        df_ccr_results=df_ccr,
        orientation="input",
        super_eff=False
    )
    
    # 3) Renombrar columnas de eficiencia para consistencia en los histogramas
    df_ccr_for_plots = df_ccr.rename(columns={"tec_efficiency_ccr": "efficiency"})
    df_bcc_for_plots = df_bcc.rename(columns={"efficiency": "efficiency"}) # Ya se llama 'efficiency' pero lo aseguramos

    resultados["df_ccr"] = df_ccr
    resultados["df_bcc"] = df_bcc

    # 4) Unir resultados con df original para visualizaciones (spider plot)
    merged_ccr = df_ccr.merge(df, on=dmu_column, how="left")
    merged_bcc = df_bcc.merge(df, on=dmu_column, how="left")
    resultados["merged_ccr"] = merged_ccr
    resultados["merged_bcc"] = merged_bcc

    # 5) Crear figuras de histograma
    hist_ccr = plot_efficiency_histogram(df_ccr_for_plots)
    hist_bcc = plot_efficiency_histogram(df_bcc_for_plots)
    resultados["hist_ccr"] = hist_ccr
    resultados["hist_bcc"] = hist_bcc

    # 6) Crear figuras de scatter 3D
    scatter3d_ccr = plot_3d_inputs_outputs(df, inputs, outputs, df_ccr_for_plots, dmu_column)
    scatter3d_bcc = plot_3d_inputs_outputs(df, inputs, outputs, df_bcc_for_plots, dmu_column)
    resultados["scatter3d_ccr"] = scatter3d_ccr
    resultados["scatter3d_bcc"] = scatter3d_bcc

    return resultados
