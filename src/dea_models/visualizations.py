import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt # Added import

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
    df_merged: DataFrame que resulta de hacer merge(df_ccr, df_original, on=dmu_col)
               Debe contener al menos: dmu_col, columnas de inputs, columnas de outputs, y eficiencia (“tec_efficiency_ccr”).
    selected_dmu: cadena con el ID de la DMU a graficar
    input_cols: lista de nombres de columnas de inputs
    output_cols: lista de nombres de columnas de outputs

    Devuelve un radar chart de Plotly Graph Objects comparando la DMU seleccionada vs sus peers eficientes.
    """
    # 1) Verificar que la DMU exista
    dmu_col = df_merged.columns[0]
    if selected_dmu not in df_merged[dmu_col].astype(str).tolist():
        df_empty = pd.DataFrame({"variable": [], "valor": [], "grupo": []})
        return px.line_polar(df_empty, r="valor", theta="variable", line_close=True, title="Sin datos")

    # 2) Filtrar peers eficientes (tec_efficiency_ccr == 1.0)
    if "tec_efficiency_ccr" not in df_merged.columns:
        df_empty = pd.DataFrame({"variable": [], "valor": [], "grupo": []})
        return px.line_polar(df_empty, r="valor", theta="variable", line_close=True, title="Sin datos")

    peers = df_merged[df_merged["tec_efficiency_ccr"] == 1.0]
    if peers.empty:
        df_empty = pd.DataFrame({"variable": [], "valor": [], "grupo": []})
        return px.line_polar(df_empty, r="valor", theta="variable", line_close=True, title="Sin peers eficientes")

    # 3) Valores de la DMU seleccionada
    dmu_row = df_merged[df_merged[dmu_col].astype(str) == selected_dmu].iloc[0]

    # 4) Promedio de peers eficientes para cada columna de inputs y outputs
    peers_avg = peers[input_cols + output_cols].mean()

    # 5) Preparar DataFrame para radar chart
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
    # Values for inputs are negative (excess), outputs are positive (shortfall/potential increase)
    values = [-slacks_in[k] for k in slacks_in] + [slacks_out[k] for k in slacks_out]

    # Filter out zero-slack variables to avoid clutter if many variables have no slack
    non_zero_slacks = [(cat, val) for cat, val in zip(categories, values) if val != 0]
    
    if not non_zero_slacks:
        # Handle case where there are no slacks to plot
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
    
    fig, ax = plt.subplots(figsize=(max(8, len(categories_plot) * 0.8), 6)) # Adjust width based on num categories
    
    # Sticking to the user's provided loop structure for `cumulative` for now,
    # assuming it's a specific style of waterfall they want.
    # This will result in bars starting where the previous one ended if they have the same sign,
    # or crossing the axis. This is non-standard for typical "waterfall charts" which show sequential contributions.
    # A more standard representation for slacks might be grouped bar charts or just simple bars from zero.
    
    # Given the user's code:
    # `ax.bar(cat, val, bottom=cumulative, color=color)`
    # This will make each bar start where the previous one ended.
    # This is a valid way to construct a waterfall if `values` are net changes.
    # Here, `values` are absolute slack magnitudes (inputs made negative).
    
    # Let's make a slight adjustment for clarity in what this waterfall represents:
    # It will show the "journey" through slacks.

    start = 0
    for cat, val in zip(categories_plot, values_plot):
        color = "C0" if val >= 0 else "C1" # Blue for positive (output slack), Orange for negative (input slack)
        ax.bar(cat, val, bottom=start, color=color, label=f'{cat}: {val:.2f}')
        start += val # This is the typical waterfall update step

    # Draw lines connecting the bars for a waterfall effect
    # Store the tops of the bars
    bar_tops = []
    current_total = 0
    for val in values_plot:
        bar_tops.append(current_total + val)
        current_total += val
    
    # Draw connecting lines (optional, but common in waterfall)
    # For this to work well, categories_plot should be ordered meaningfully.
    # For now, we'll omit explicit connecting lines as the `bottom` argument does the main job.

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Waterfall de slacks para DMU {dmu_name}")
    ax.set_ylabel("Magnitud del Slack (Outputs: positivo = shortfall, Inputs: negativo = excess)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Add a legend if there are few enough items, or to explain colors
    # handles, labels = ax.get_legend_handles_labels()
    # if handles:
    #    ax.legend(handles, labels)
    # Or a more general legend:
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='C1', label='Exceso de Input (Reducción Sugerida)')
    blue_patch = mpatches.Patch(color='C0', label='Escasez de Output (Aumento Sugerido)')
    ax.legend(handles=[red_patch, blue_patch])

    return fig
