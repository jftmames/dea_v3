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
    Versión optimizada que ejecuta ambos modelos (CCR y BCC) eficientemente,
    genera los DataFrames resultantes y produce las figuras de Plotly.
    Devuelve un diccionario con todos los resultados y figuras.
    """
    resultados = {}

    # 1) Ejecutar CCR (una sola vez)
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
        df_ccr_results=df_ccr,  # <-- Se pasan los resultados de CCR
        orientation="input",
        super_eff=False
    )
    
    # 3) RENOMBRAR columnas de eficiencia para consistencia
    # Guardamos una copia con el nombre original para el spider plot si es necesario
    df_ccr_for_spider = df_ccr.copy()
    df_ccr = df_ccr.rename(columns={"tec_efficiency_ccr": "efficiency"})

    resultados["df_ccr"] = df_ccr
    resultados["df_bcc"] = df_bcc

    # 4) Unir resultados de eficiencia con df original para visualizaciones
    # merged_ccr necesita la columna de eficiencia original de ccr para el spider
    merged_ccr = df_ccr_for_spider.merge(df, on=dmu_column, how="left")
    merged_bcc = df_bcc.merge(df, on=dmu_column, how="left")
    resultados["merged_ccr"] = merged_ccr
    resultados["merged_bcc"] = merged_bcc

    # 5) Crear figuras de histograma (usan la columna 'efficiency' renombrada)
    hist_ccr = plot_efficiency_histogram(df_ccr)
    hist_bcc = plot_efficiency_histogram(df_bcc)
    resultados["hist_ccr"] = hist_ccr
    resultados["hist_bcc"] = hist_bcc

    # 6) Crear figuras de scatter 3D
    scatter3d_ccr = plot_3d_inputs_outputs(df, inputs, outputs, df_ccr, dmu_column)
    scatter3d_bcc = plot_3d_inputs_outputs(df, inputs, outputs, df_bcc, dmu_column)
    resultados["scatter3d_ccr"] = scatter3d_ccr
    resultados["scatter3d_bcc"] = scatter3d_bcc

    return resultados
