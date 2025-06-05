import pandas as pd
import plotly.express as px

from dea_models.radial import run_ccr, run_bcc
from dea_models.visualizations import plot_benchmark_spider, plot_efficiency_histogram, plot_3d_inputs_outputs

def mostrar_resultados(df: pd.DataFrame, dmu_col: str, input_cols: list[str], output_cols: list[str]) -> dict:
    """
    Ejecuta ambos modelos (CCR y BCC), genera los DataFrames resultantes
    y produce figuras de Plotly para los histogramas y scatter 3D.
    Devuelve un diccionario con:
      - df_ccr: DataFrame con resultados CCR
      - df_bcc: DataFrame con resultados BCC
      - hist_ccr: figura Plotly del histograma CCR (no se usa directamente en main.py)
      - hist_bcc: figura Plotly del histograma BCC (no se usa directamente en main.py)
      - scatter3d_ccr: figura Plotly del scatter 3D (CCR)
    """

    # 1) Ejecución CCR y BCC
    df_ccr = run_ccr(df, dmu_col, input_cols, output_cols, orientation="input", super_eff=False)
    df_bcc = run_bcc(df, dmu_col, input_cols, output_cols, orientation="input", super_eff=False)

    # 2) Generar “histogramas” básicos con Plotly Express (pueden no usarse en main.py)
    hist_ccr = px.histogram(df_ccr, x="tec_efficiency_ccr", title="Distribución de eficiencias CCR")
    hist_bcc = px.histogram(df_bcc, x="efficiency", title="Distribución de eficiencias BCC")

    # 3) Generar scatter 3D para CCR (primeras dos columnas de inputs y primer output)
    if len(input_cols) >= 2 and len(output_cols) >= 1:
        scatter3d_ccr = plot_3d_inputs_outputs(df_ccr, input_cols, output_cols, color_col="tec_efficiency_ccr")
    else:
        # Si no hay suficientes columnas, generamos un gráfico vacío
        scatter3d_ccr = px.scatter_3d(pd.DataFrame({input_cols[0]: [], input_cols[1]: [], output_cols[0]: []}),
                                      x=input_cols[0], y=input_cols[1], z=output_cols[0],
                                      title="No hay suficientes columnas para Scatter 3D")
    return {
        "df_ccr": df_ccr,
        "df_bcc": df_bcc,
        "hist_ccr": hist_ccr,
        "hist_bcc": hist_bcc,
        "scatter3d_ccr": scatter3d_ccr
    }
