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
    Ejecuta ambos modelos (CCR y BCC), genera los DataFrames resultantes
    y produce figuras de Plotly para los histogramas y scatter 3D.
    Devuelve un diccionario con:
      - df_ccr: DataFrame con resultados CCR
      - df_bcc: DataFrame con resultados BCC
      - merged_ccr, merged_bcc: unión con df original para visualizaciones
      - hist_ccr: figura Plotly del histograma CCR
      - hist_bcc: figura Plotly del histograma BCC
      - scatter3d_ccr: figura Plotly del scatter 3D (CCR)
    """

    resultados = {} #

    # 1) Ejecutar CCR
    df_ccr = run_ccr(
        df=df,
        dmu_column=dmu_column,
        input_cols=inputs,
        output_cols=outputs,
        orientation="input",
        super_eff=False
    )
    # RENOMBRAR la columna tec_efficiency_ccr → efficiency
    df_ccr = df_ccr.rename(columns={"tec_efficiency_ccr": "efficiency"})
    resultados["df_ccr"] = df_ccr # Guardar en el diccionario

    # 2) Ejecutar BCC
    df_bcc = run_bcc(
        df=df,
        dmu_column=dmu_column,
        input_cols=inputs,
        output_cols=outputs,
        orientation="input",
        super_eff=False
    )
    # RENOMBRAR la columna tec_efficiency_bcc → efficiency
    # Ensure this column actually exists before renaming.
    if "efficiency" not in df_bcc.columns and "tec_efficiency_bcc" in df_bcc.columns:
        df_bcc = df_bcc.rename(columns={"tec_efficiency_bcc": "efficiency"})
    elif "efficiency" not in df_bcc.columns: # If neither exists, something is wrong
        pass
    resultados["df_bcc"] = df_bcc # Guardar en el diccionario


    # 3) Unir resultados de eficiencia con df original (ahora 'efficiency' es el nombre común)
    merged_ccr = df_ccr.merge(df, on=dmu_column, how="left")
    merged_bcc = df_bcc.merge(df, on=dmu_column, how="left")
    resultados["merged_ccr"] = merged_ccr
    resultados["merged_bcc"] = merged_bcc


    # 4) Crear figuras de histograma (ahora ya existe 'efficiency' en ambos dfs)
    hist_ccr = plot_efficiency_histogram(df_ccr)
    hist_bcc = plot_efficiency_histogram(df_bcc)
    resultados["hist_ccr"] = hist_ccr
    resultados["hist_bcc"] = hist_bcc

    # 5) Crear figuras de scatter 3D
    scatter3d_ccr = plot_3d_inputs_outputs(df, inputs, outputs, df_ccr, dmu_column)
    scatter3d_bcc = plot_3d_inputs_outputs(df, inputs, outputs, df_bcc, dmu_column)
    resultados["scatter3d_ccr"] = scatter3d_ccr
    resultados["scatter3d_bcc"] = scatter3d_bcc

    return resultados
