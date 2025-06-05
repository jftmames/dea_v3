# jftmames/-dea-deliberativo-mvp/-dea-deliberativo-mvp-b44b8238c978ae0314af30717b9399634d28f8f9/src/dea_models/visualizations.py
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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


def plot_benchmark_spider(df_merged: pd.DataFrame, selected_dmu: str, input_cols: list[str], output_cols: list[str]):
    """
    Devuelve un radar chart comparando la DMU seleccionada vs sus peers eficientes.
    """
    dmu_col = df_merged.columns[0]
    if selected_dmu not in df_merged[dmu_col].astype(str).tolist():
        return px.line_polar(title="Sin datos de DMU")

    if "tec_efficiency_ccr" not in df_merged.columns:
        return px.line_polar(title="Columna de eficiencia no encontrada")

    peers = df_merged[df_merged["tec_efficiency_ccr"] == 1.0]
    if peers.empty:
        return px.line_polar(title="Sin peers eficientes para comparar")

    dmu_row = df_merged[df_merged[dmu_col].astype(str) == selected_dmu].iloc[0]
    peers_avg = peers[input_cols + output_cols].mean()

    variables = input_cols + output_cols
    df_radar = pd.DataFrame({
        "variable": variables * 2,
        "valor": list(dmu_row[variables]) + list(peers_avg[variables]),
        "grupo": [f"DMU {selected_dmu}"] * len(variables) + ["Peers promedio"] * len(variables)
    })

    fig = px.line_polar(
        df_radar, r="valor", theta="variable", color="grupo", line_close=True,
        title=f"Benchmark Spider: {selected_dmu} vs Peers eficientes"
    )
    fig.update_traces(fill="toself")
    return fig


def plot_3d_inputs_outputs(
    orig_df: pd.DataFrame,
    inputs: list[str],
    outputs: list[str],
    dea_df: pd.DataFrame,
    dmu_column: str
):
    """
    Scatter 3D: ejes = primeros 2 inputs, tercer eje = primer output, coloreado según eficiencia.
    """
    merged = dea_df.merge(orig_df, on=dmu_column, how="left")
    x_col = inputs[0]
    y_col = inputs[1] if len(inputs) >= 2 else inputs[0]
    z_col = outputs[0]

    fig = px.scatter_3d(
        merged, x=x_col, y=y_col, z=z_col, color="efficiency",
        hover_name=dmu_column, title="Scatter 3D de Inputs vs Output (coloreado por eficiencia)",
        labels={x_col: x_col, y_col: y_col, z_col: z_col, "efficiency": "Eficiencia"},
    )
    fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
    return fig


def plot_slack_waterfall(slacks_in: dict, slacks_out: dict, dmu_name: str):
    """ Dibuja un gráfico de cascada con los slacks. """
    data = []
    for k, v in slacks_in.items():
        if v > 1e-6: data.append({'variable': k, 'valor': -v, 'tipo': 'Exceso de Input'})
    for k, v in slacks_out.items():
        if v > 1e-6: data.append({'variable': k, 'valor': v, 'tipo': 'Déficit de Output'})

    if not data:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"DMU {dmu_name} es eficiente (sin slacks).", ha='center', va='center')
        return fig
    
    df_plot = pd.DataFrame(data)
    fig = px.bar(df_plot, x='variable', y='valor', color='tipo', title=f"Slacks para DMU {dmu_name}",
                 color_discrete_map={'Exceso de Input': 'red', 'Déficit de Output': 'green'})
    return fig

# --- NUEVA FUNCIÓN ---
def plot_hypothesis_distribution(
    df_results: pd.DataFrame,
    df_original: pd.DataFrame,
    variable: str,
    dmu_col: str,
    efficiency_col: str = 'tec_efficiency_ccr'
) -> go.Figure:
    """
    Genera un box plot para comparar la distribución de una variable entre
    unidades eficientes e ineficientes.
    """
    if variable not in df_original.columns:
        return go.Figure().update_layout(title_text=f"Error: La variable '{variable}' no existe.")

    # Unir resultados con datos originales
    df_merged = df_results.merge(df_original[[dmu_col, variable]], on=dmu_col, how="left")

    # Crear columna de estado (Eficiente / Ineficiente)
    df_merged['Estatus'] = df_merged[efficiency_col].apply(lambda x: 'Eficiente' if x >= 0.999 else 'Ineficiente')

    # Crear el gráfico
    fig = px.box(
        df_merged,
        x='Estatus',
        y=variable,
        color='Estatus',
        title=f"Comparación de '{variable}' entre Unidades Eficientes e Ineficientes",
        points="all", # Muestra todos los puntos
        labels={"Estatus": "Estatus de Eficiencia", variable: f"Valor de {variable}"},
        color_discrete_map={'Eficiente': 'green', 'Ineficiente': 'red'}
    )
    fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    return fig
