# src/dea_models/visualizations.py

import pandas as pd
import plotly.express as px

def plot_benchmark_spider(
    merged_for_spider: pd.DataFrame,
    selected_dmu: str,
    inputs: list[str],
    outputs: list[str],
):
    """
    Radar/spider plot comparando la DMU seleccionada vs. la media de DMUs eficientes.
    merged_for_spider: DataFrame con columnas 'DMU', inputs, outputs y 'efficiency'.
    selected_dmu: identificador de la DMU a plotear.
    inputs: lista de columnas de insumo
    outputs: lista de columnas de producto
    """
    # Filtrar DMUs eficientes (efficiency == 1)
    efficient_peers = merged_for_spider.query("efficiency == 1")
    if efficient_peers.empty:
        raise ValueError("No hay DMU eficientes para benchmark.")

    # Variables a comparar
    vars_all = inputs + outputs
    peer_means = efficient_peers[vars_all].mean()

    # Valores de la DMU seleccionada
    sel_row = merged_for_spider.loc[merged_for_spider["DMU"] == selected_dmu]
    if sel_row.empty:
        raise ValueError(f"No existe la DMU '{selected_dmu}' en merged_for_spider.")
    sel_values = sel_row.iloc[0][vars_all]

    # Cerrar ciclo para radar
    categories = vars_all + [vars_all[0]]
    peer_vals = peer_means.tolist() + [peer_means.tolist()[0]]
    sel_vals = sel_values.tolist() + [sel_values.tolist()[0]]

    # Construir DataFrame para Plotly
    df_spider = pd.DataFrame({
        "variable": categories * 2,
        "valor": peer_vals + sel_vals,
        "tipo": ["Peer (medio)"] * len(categories) + [f"DMU {selected_dmu}"] * len(categories),
    })

    fig = px.line_polar(
        df_spider,
        r="valor",
        theta="variable",
        color="tipo",
        line_close=True,
        title=f"Benchmark Spider: DMU {selected_dmu} vs. Promedio de Peers Eficientes",
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig
