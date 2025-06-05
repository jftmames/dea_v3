import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def plot_efficiency_histogram(df_eff: pd.DataFrame, eff_col: str, title: str = "Efficiency Histogram"):
    """
    df_eff: DataFrame que contiene una columna con eficiencias (por ejemplo, "efficiency" o "tec_efficiency_ccr")
    eff_col: nombre de la columna de eficiencias dentro de df_eff
    title: título para la gráfica
    Devuelve una figura de Plotly Express (histograma).
    """
    if eff_col not in df_eff.columns:
        # Devolvemos un histograma vacío
        return px.histogram(pd.DataFrame({eff_col: []}), x=eff_col, title=title)

    fig = px.histogram(df_eff, x=eff_col, nbins=20, title=title)
    fig.update_layout(xaxis_title="Eficiencia", yaxis_title="Frecuencia")
    return fig

def plot_benchmark_spider(df_merged: pd.DataFrame, selected_dmu: str, input_cols: list[str], output_cols: list[str]):
    """
    df_merged: DataFrame que resulta de hacer merge(df_ccr, df_original, on=dmu_col)
               Debe contener al menos: dmu_col, columnas de inputs, columnas de outputs, y efﬁciencia (“tec_efficiency_ccr”).
    selected_dmu: cadena con el ID de la DMU a graficar
    input_cols: lista de nombres de columnas de inputs
    output_cols: lista de nombres de columnas de outputs

    Lo que hace:
      - Identifica los peers eficientes (donde tec_efficiency_ccr == 1.0)
      - Calcula el promedio de inputs y outputs de esos peers
      - Construye un radar chart comparando “selected_dmu” vs “peers promedio”
    Devuelve un objeto de Plotly Graph Objects.
    """

    # 1) Filtrar solo la DMU seleccionada
    if selected_dmu not in df_merged[df_merged.columns[0]].astype(str).tolist():
        # Si la DMU no existe, devolvemos un gráfico vacío
        df_empty = pd.DataFrame({ "variable": [], "valor_dmu": [], "valor_peers": [] })
        return px.line_polar(df_empty, r="valor_dmu", theta="variable", line_close=True, title="Sin datos")

    # 2) Identificar peers eficientes
    # Suponemos que la columna de eficiencia se llama “tec_efficiency_ccr”
    peers = df_merged[df_merged["tec_efficiency_ccr"] == 1.0]
    if peers.empty:
        # Si no hay peers eficientes, devolvemos también vacío
        df_empty = pd.DataFrame({ "variable": [], "valor_dmu": [], "valor_peers": [] })
        return px.line_polar(df_empty, r="valor_dmu", theta="variable", line_close=True, title="Sin peers eficientes")

    # 3) Extraer valores de la DMU seleccionada
    dmu_row = df_merged[df_merged[df_merged.columns[0]].astype(str) == selected_dmu].iloc[0]

    # 4) Calcular promedio de peers eficientes para cada columna de input/output
    peers_avg = peers[input_cols + output_cols].mean()

    # 5) Preparar DataFrame para el radar
    variables = input_cols + output_cols
    valores_dmu = [float(dmu_row[col]) for col in variables]
    valores_peers = [float(peers_avg[col]) for col in variables]

    df_radar = pd.DataFrame({
        "variable": variables * 2,
        "valor": valores_dmu + valores_peers,
        "grupo": [f"DMU {selected_dmu}"] * len(variables) + ["Peers promedio"] * len(variables)
    })

    fig = px.line_polar(
        df_radar,
        r="valor",
        theta="variable",
        color="grupo",
        line_close=True,
        title=f"Benchmark Spider: {selected_dmu} vs Peers eficaces"
    )
    fig.update_traces(fill="toself")
    return fig

def plot_3d_inputs_outputs(df_eff: pd.DataFrame, input_cols: list[str], output_cols: list[str], color_col: str = None):
    """
    df_eff: DataFrame que contiene resultados (por ejemplo df_ccr) con al menos:
            - columnas de inputs (input_cols)
            - columnas de outputs (output_cols)
            - una columna de color (color_col), opcional (por ejemplo, “tec_efficiency_ccr”)
    input_cols: lista de al menos 2 nombres de columnas de inputs
    output_cols: lista de al menos 1 nombre de columna de outputs
    color_col: (opcional) nombre de columna que usaremos para el color
    Devuelve un scatter 3D de Plotly Express.
    """

    # Validaciones mínimas
    if len(input_cols) < 2 or len(output_cols) < 1:
        # Gráfico vacío
        return px.scatter_3d(
            pd.DataFrame({input_cols[0]: [], input_cols[1]: [], output_cols[0]: []}),
            x=input_cols[0], y=input_cols[1], z=output_cols[0],
            title="No hay suficientes columnas para Scatter 3D"
        )

    # Tomamos las primeras dos columnas de inputs y la primera de outputs
    x_col = input_cols[0]
    y_col = input_cols[1]
    z_col = output_cols[0]

    if color_col and color_col in df_eff.columns:
        fig = px.scatter_3d(
            df_eff,
            x=x_col,
            y=y_col,
            z=z_col,
            color=color_col,
            hover_data=[x_col, y_col, z_col, color_col],
            title="Scatter 3D Inputs vs Output"
        )
    else:
        fig = px.scatter_3d(
            df_eff,
            x=x_col,
            y=y_col,
            z=z_col,
            hover_data=[x_col, y_col, z_col],
            title="Scatter 3D Inputs vs Output"
        )
    return fig
