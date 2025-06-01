# src/results.py

import pandas as pd
import plotly.express as px

from dea_models.radial import run_ccr, run_bcc
from dea_models import plot_benchmark_spider  # importamos la función del paquete


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
    # 1) Ejecutar CCR
    df_ccr = run_ccr(
        df=df,
        dmu_column=dmu_column,
        input_cols=inputs,
        output_cols=outputs,
        orientation="input",
        super_eff=False
    )

    # 2) Ejecutar BCC
    df_bcc = run_bcc(
        df=df,
        dmu_column=dmu_column,
        input_cols=inputs,
        output_cols=outputs,
        orientation="input",
        super_eff=False
    )

    # 3) Unir resultados de eficiencia con df original
    merged_ccr = df_ccr.merge(df, on=dmu_column, how="left")
    merged_bcc = df_bcc.merge(df, on=dmu_column, how="left")

    # 4) Crear figuras
    hist_ccr = plot_efficiency_histogram(df_ccr)
    hist_bcc = plot_efficiency_histogram(df_bcc)

    scatter3d_ccr = plot_3d_inputs_outputs(df, inputs, outputs, df_ccr, dmu_column)
    scatter3d_bcc = plot_3d_inputs_outputs(df, inputs, outputs, df_bcc, dmu_column)

    return {
        "df_ccr": df_ccr,
        "df_bcc": df_bcc,
        "merged_ccr": merged_ccr,
        "merged_bcc": merged_bcc,
        "hist_ccr": hist_ccr,
        "hist_bcc": hist_bcc,
        "scatter3d_ccr": scatter3d_ccr,
        "scatter3d_bcc": scatter3d_bcc,
    }


def plot_efficiency_histogram(dea_df: pd.DataFrame, bins: int = 20):
    """
    Devuelve un histograma de eficiencias DEA usando Plotly Express.
    Asume que dea_df tiene la columna 'efficiency'.
    """
    fig = px.histogram(
        dea_df,
        x="efficiency",
        nbins=bins,
        title="Distribución de eficiencias",
        labels={"efficiency": "Eficiencia DEA"},
    )
    fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
    return fig


def plot_3d_inputs_outputs(
    orig_df: pd.DataFrame,
    inputs: list[str],
    outputs: list[str],
    dea_df: pd.DataFrame,
    dmu_column: str
):
    """
    Scatter 3D: ejes = primeros 2 inputs, tercer eje = primer output,
    coloreado según eficiencia. dea_df debe tener dmu_column y 'efficiency'.
    """
    merged = dea_df.merge(orig_df, on=dmu_column, how="left")

    x_col = inputs[0]
    y_col = inputs[1] if len(inputs) >= 2 else inputs[0]
    z_col = outputs[0]

    if x_col not in merged.columns or y_col not in merged.columns or z_col not in merged.columns:
        raise ValueError(f"Columnas {x_col}, {y_col} o {z_col} no existen en los datos combinados.")

    fig = px.scatter_3d(
        merged,
        x=x_col,
        y=y_col,
        z=z_col,
        color="efficiency",
        hover_name=dmu_column,
        title="Scatter 3D de Inputs vs Output (coloreado por eficiencia)",
        labels={x_col: x_col, y_col: y_col, z_col: z_col, "efficiency": "Eficiencia"},
    )
    fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
    return fig

# Ya no definimos plot_benchmark_spider aquí; se importa desde dea_models.
