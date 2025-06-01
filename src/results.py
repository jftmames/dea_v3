import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_efficiency_histogram(dea_df: pd.DataFrame, bins: int = 20):
    """
    Devuelve un histograma de eficiencias DEA usando Plotly Express.
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
):
    """
    Scatter 3D: ejes = primeros 2 inputs, tercer eje = primer output,
    coloreado según eficiencia.
    """
    # Primero unimos dea_df (que tiene 'DMU' y 'efficiency') con orig_df
    merged = dea_df.merge(orig_df, on="DMU", how="left")

    # Seleccionamos las 2 primeras columnas de inputs y la primera de outputs
    x_col = inputs[0]
    y_col = inputs[1] if len(inputs) >= 2 else inputs[0]
    z_col = outputs[0]

    # Si faltan columnas, levantar error
    if x_col not in merged.columns or y_col not in merged.columns or z_col not in merged.columns:
        raise ValueError(f"Columnas {x_col}, {y_col} o {z_col} no existen en los datos combinados.")

    fig = px.scatter_3d(
        merged,
        x=x_col,
        y=y_col,
        z=z_col,
        color="efficiency",
        hover_name="DMU",
        title="Scatter 3D de Inputs vs Output (coloreado por eficiencia)",
        labels={x_col: x_col, y_col: y_col, z_col: z_col, "efficiency": "Eficiencia"},
    )
    fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
    return fig


def plot_benchmark_spider(
    merged_for_spider: pd.DataFrame,
    selected_dmu: str,
    inputs: list[str],
    outputs: list[str],
):
    """
    Radar/spider plot comparando la DMU seleccionada vs. la media (o máximos) de DMUs eficientes.
    merged_for_spider: DataFrame que ya contiene 'DMU', columnas de inputs y outputs y 'efficiency'.
    selected_dmu: identificador de la DMU a plotear.
    """
    # Filtramos solo las DMU eficientes (efficiency == 1)
    efficient_peers = merged_for_spider.query("efficiency == 1")
    if efficient_peers.empty:
        raise ValueError("No hay DMU eficientes para benchmark.")

    # Agrupamos por variables (inputs + outputs) en los peers eficientes: tomamos valor medio
    vars_all = inputs + outputs
    peer_means = efficient_peers[vars_all].mean()

    # Obtenemos los valores de la DMU seleccionada
    sel_row = merged_for_spider.loc[merged_for_spider["DMU"] == selected_dmu]
    if sel_row.empty:
        raise ValueError(f"No existe la DMU '{selected_dmu}' en merged_for_spider.")
    sel_values = sel_row.iloc[0][vars_all]

    # Para radar, necesitamos cerrar el ciclo: la última variable debe repetirse
    categories = vars_all + [vars_all[0]]
    peer_vals = peer_means.tolist() + [peer_means.tolist()[0]]
    sel_vals = sel_values.tolist() + [sel_values.tolist()[0]]

    # Construimos el DataFrame para px.line_polar
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
