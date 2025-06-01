# results.py

import plotly.express as px
import pandas as pd

def plot_efficiency_histogram(dea_df: pd.DataFrame, bins: int = 20):
    fig = px.histogram(dea_df, x="efficiency", nbins=bins, title="Distribución de Eficiencias")
    fig.update_layout(xaxis_title="Eficiencia", yaxis_title="Cantidad de DMU")
    return fig

def plot_3d_inputs_outputs(
    df_original: pd.DataFrame,
    inputs: list[str],
    outputs: list[str],
    dea_df: pd.DataFrame,
):
    """
    Genera un scatter 3D:
      - Ejes X e Y: los dos primeros inputs
      - Eje Z: primer output
      - Color: eficiencia
    Requiere al menos 2 inputs y 1 output.
    """
    if len(inputs) < 2 or len(outputs) < 1:
        raise ValueError("Se necesitan al menos 2 Inputs y 1 Output para el Scatter 3D.")
    # Tomamos los dos primeros inputs y el primer output
    x_col, y_col = inputs[0], inputs[1]
    z_col = outputs[0]

    # Merge dea_df con df_original para tener eficiencia junto a inputs/outputs
    df_plot = dea_df.merge(
        df_original[[x_col, y_col, z_col] + (["DMU"] if "DMU" in df_original.columns else [])],
        on="DMU",
        how="left"
    )

    fig = px.scatter_3d(
        df_plot,
        x=x_col,
        y=y_col,
        z=z_col,
        color="efficiency",
        hover_name="DMU",
        title=f"Scatter 3D: {x_col} vs {y_col} vs {z_col}"
    )
    return fig

def plot_benchmark_spider(
    merged_df: pd.DataFrame,
    selected_dmu: str,
    inputs: list[str],
    outputs: list[str],
):
    """
    Crea un gráfico de radar/spider comparando la DMU seleccionada contra peers eficientes.
    merged_df debe contener: 'DMU', 'efficiency', todos los inputs y outputs.
    """
    # Encontrar la fila de la DMU seleccionada
    row = merged_df[merged_df["DMU"] == selected_dmu]
    if row.empty:
        raise ValueError(f"DMU '{selected_dmu}' no encontrada en merged_df.")

    # Extraemos valores de inputs y outputs para la DMU seleccionada
    values_selected = row[inputs + outputs].iloc[0].values.tolist()

    # Tomamos solo peers eficientes (efficiency == 1)
    peers = merged_df[merged_df["efficiency"] == 1]
    if peers.empty:
        raise ValueError("No hay peers eficientes para comparar.")

    # Calculamos el promedio de inputs/outputs de los peers
    peers_avg = peers[inputs + outputs].mean().tolist()

    categories = inputs + outputs

    # Para radar, necesitamos repetir el primer valor al final
    values_sel = values_selected + [values_selected[0]]
    values_peer = peers_avg + [peers_avg[0]]
    categories_cycle = categories + [categories[0]]

    fig = px.line_polar(
        r=values_sel,
        theta=categories_cycle,
        line_close=True,
        name=selected_dmu,
        title=f"Benchmark Spider: {selected_dmu} vs Peers eficientes"
    )
    fig.add_scatterpolar(
        r=values_peer,
        theta=categories_cycle,
        line_close=True,
        name="Peers Promedio"
    )
    return fig
