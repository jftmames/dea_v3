# src/dea_models/visualizations.py
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt 

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

def plot_slack_waterfall(
    slacks_in: dict[str, float],
    slacks_out: dict[str, float],
    dmu_name: str
):
    """
    Dibuja un gráfico de cascada:
      - Inputs con slack positivo (exceso) como barras negativas.
      - Outputs con slack positivo (escasez) como barras positivas.
    slacks_in: dict {input_var: slack_value}
    slacks_out: dict {output_var: slack_value}
    dmu_name: nombre de la DMU para título.
    """
    categories = list(slacks_in.keys()) + list(slacks_out.keys())
    values = [-slacks_in[k] for k in slacks_in] + [slacks_out[k] for k in slacks_out]

    non_zero_slacks = [(cat, val) for cat, val in zip(categories, values) if val != 0]
    
    if not non_zero_slacks:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"No slacks to display for DMU {dmu_name}", 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes)
        ax.set_title(f"Waterfall de slacks para DMU {dmu_name}")
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    categories_plot = [item[0] for item in non_zero_slacks]
    values_plot = [item[1] for item in non_zero_slacks]
    
    fig, ax = plt.subplots(figsize=(max(8, len(categories_plot) * 0.8), 6)) 
    
    start = 0
    for cat, val in zip(categories_plot, values_plot):
        color = "C0" if val >= 0 else "C1" 
        ax.bar(cat, val, bottom=start, color=color, label=f'{cat}: {val:.2f}')
        start += val 

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Waterfall de slacks para DMU {dmu_name}")
    ax.set_ylabel("Magnitud del Slack (Outputs: positivo = shortfall, Inputs: negativo = excess)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='C1', label='Exceso de Input (Reducción Sugerida)')
    blue_patch = mpatches.Patch(color='C0', label='Escasez de Output (Aumento Sugerido)')
    ax.legend(handles=[red_patch, blue_patch])

    return fig
