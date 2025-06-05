# src/results.py
import pandas as pd
import plotly.express as px

from .dea_models.radial import run_ccr, run_bcc # Changed to relative import
from .dea_models.visualizations import plot_benchmark_spider, plot_efficiency_histogram, plot_3d_inputs_outputs # Changed to relative import


def mostrar_resultados(
    df: pd.DataFrame,
    dmu_column: str,
    inputs: list[str],
    outputs: list[str]
) -> dict:
    """
    Ejecuta los modelos CCR y BCC sobre el DataFrame proporcionado, devuelve:
      - df_ccr, df_bcc: resultados de eficiencias
      - merged_ccr, merged_bcc: unión con df original para visualizaciones
      - hist_ccr, hist_bcc: figura de histograma de eficiencias
      - scatter3d_ccr, scatter3d_bcc: figura 3D inputs/outputs coloreado por eficiencia
      - plot_benchmark_spider se invoca desde main.py, no aquí.
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
        # This case should ideally be caught by run_bcc returning a well-formed DF
        # For robustness, we could add a placeholder or error.
        # But given the contract of run_bcc, 'efficiency' should be present.
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


# The functions plot_efficiency_histogram and plot_3d_inputs_outputs are
# defined in dea_models/visualizations.py and imported here.
# So, they don't need to be redefined in this file.
