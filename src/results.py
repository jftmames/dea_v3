import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def plot_efficiency_histogram(dea_df: pd.DataFrame, bins: int = 20) -> go.Figure:
    """
    Genera un histograma interactivo de las eficiencias.
    dea_df debe tener una columna "efficiency" con valores numéricos.
    """
    fig = px.histogram(
        dea_df,
        x="efficiency",
        nbins=bins,
        title="Distribución de Eficiencias DEA",
        labels={"efficiency": "Eficiencia"}
    )
    fig.update_layout(
        xaxis_title="Eficiencia",
        yaxis_title="Conteo de DMU",
        bargap=0.05
    )
    return fig


def plot_3d_inputs_outputs(
    df: pd.DataFrame,
    inputs: list[str],
    outputs: list[str],
    dea_df: pd.DataFrame,
    size_factor: float = 5.0
) -> go.Figure:
    """
    Genera un scatter 3D de los dos primeros inputs vs. el primer output,
    coloreado por eficiencia.
    - df: DataFrame original con columnas de inputs y outputs.
    - inputs: lista de al menos 2 columnas numéricas de insumos.
    - outputs: lista de al menos 1 columna numérica de producto.
    - dea_df: DataFrame de resultados DEA (contiene "DMU" y "efficiency").
    - size_factor: factor de escala para el tamaño de marca en el 3D.
    """
    if len(inputs) < 2 or len(outputs) < 1:
        raise ValueError("Se requieren ≥2 inputs y ≥1 output para el scatter 3D.")

    # Fusionar df y dea_df para que cada DMU tenga su eficiencia
    if "DMU" in df.columns:
        merged = df.merge(dea_df[["DMU", "efficiency"]], on="DMU", how="left")
    else:
        df_copy = df.copy()
        df_copy["DMU"] = df_copy.index.astype(str)
        merged = df_copy.merge(dea_df[["DMU", "efficiency"]], on="DMU", how="left")

    x_col = inputs[0]
    y_col = inputs[1]
    z_col = outputs[0]

    fig = px.scatter_3d(
        merged,
        x=x_col,
        y=y_col,
        z=z_col,
        color="efficiency",
        size="efficiency",
        size_max=size_factor,
        title=f"Scatter 3D: {x_col} vs {y_col} vs {z_col}",
        labels={
            x_col: x_col,
            y_col: y_col,
            z_col: z_col,
            "efficiency": "Eficiencia"
        }
    )
    fig.update_layout(legend_title_text="Eficiencia")
    return fig


def plot_benchmark_spider(
    merged_df: pd.DataFrame,
    target_dmu: str,
    inputs: list[str],
    outputs: list[str]
) -> go.Figure:
    """
    Genera un gráfico tipo radar (spider) comparando la DMU seleccionada
    frente al promedio de las DMU eficientes (efficiency == 1) para cada input/output.
    - merged_df: DataFrame que contiene columnas inputs+outputs+DMU+efficiency.
    - target_dmu: ID de la DMU que se quiere comparar.
    - inputs: lista de columnas de insumos.
    - outputs: lista de columnas de productos.
    """
    # Filtrar DMU eficientes
    eff_df = merged_df.query("efficiency == 1")
    if eff_df.empty:
        raise ValueError("No hay DMU eficientes (efficiency == 1) para benchmark.")

    # Promedio de cada columna (inputs + outputs) para DMU eficientes
    cols = inputs + outputs
    avg_values = eff_df[cols].mean(axis=0)

    # Valores de la DMU objetivo
    if target_dmu not in merged_df["DMU"].values:
        raise KeyError(f"La DMU '{target_dmu}' no existe en el DataFrame.")
    target_row = merged_df.loc[merged_df["DMU"] == target_dmu, cols].iloc[0]

    # Preparar categorías y cerrar el polígono
    categories = cols.copy()
    categories.append(cols[0])

    target_vals = [float(target_row[c]) for c in cols]
    target_vals.append(target_vals[0])

    avg_vals = [float(avg_values[c]) for c in cols]
    avg_vals.append(avg_vals[0])

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=avg_vals,
        theta=categories,
        fill="toself",
        name="Promedio Eficientes"
    ))
    fig.add_trace(go.Scatterpolar(
        r=target_vals,
        theta=categories,
        fill="toself",
        name=f"DMU: {target_dmu}"
    ))

    max_range = max(max(target_vals), max(avg_vals)) * 1.1
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_range]
            )
        ),
        showlegend=True,
        title=f"Benchmark Spider: '{target_dmu}' vs Promedio de Eficientes"
    )
    return fig
