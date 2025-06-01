# ── src/results.py ──

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_efficiency_histogram(dea_df: pd.DataFrame, bins: int = 20):
    """
    Dibuja un histograma de la columna 'efficiency' del DataFrame de resultados DEA.
    """
    # Asegurarse de que 'efficiency' exista y sea numérico
    if "efficiency" not in dea_df.columns:
        raise ValueError("La columna 'efficiency' no existe en el DataFrame.")
    df = dea_df.copy()
    df["efficiency"] = pd.to_numeric(df["efficiency"], errors="coerce")

    fig = px.histogram(
        df,
        x="efficiency",
        nbins=bins,
        title="Distribución de Eficiencias DEA",
        labels={"efficiency": "Eficiencia"},
    )
    fig.update_layout(
        xaxis_title="Eficiencia",
        yaxis_title="Conteo de DMU",
        bargap=0.1,
    )
    return fig


def plot_3d_inputs_outputs(
    original_df: pd.DataFrame,
    inputs: list[str],
    outputs: list[str],
    dea_df: pd.DataFrame,
):
    """
    Genera un scatter 3D coloreado por eficiencia.
    - Usa los dos primeros inputs en los ejes X e Y.
    - Usa el primer output en el eje Z.
    - Colorea por la eficiencia calculada.
    """
    # Validaciones mínimas
    if len(inputs) < 2:
        raise ValueError("Se necesitan al menos 2 columnas de Inputs para el Scatter 3D.")
    if len(outputs) < 1:
        raise ValueError("Se necesita al menos 1 columna de Output para el Scatter 3D.")
    if "DMU" not in dea_df.columns:
        raise ValueError("El DataFrame de DEA debe contener la columna 'DMU'.")
    if "efficiency" not in dea_df.columns:
        raise ValueError("El DataFrame de DEA debe contener la columna 'efficiency'.")

    # Tomamos los dos primeros inputs y el primer output
    x_col = inputs[0]
    y_col = inputs[1]
    z_col = outputs[0]

    # Hacemos merge entre el df original y dea_df para tener los valores de inputs/outputs junto a 'efficiency'
    # Antes, nos aseguramos de que ambas tablas tengan la columna 'DMU'
    df_orig = original_df.copy()
    df_orig["DMU"] = df_orig["DMU"].astype(str) if "DMU" in df_orig.columns else df_orig.index.astype(str)
    dea_copy = dea_df.copy()
    dea_copy["DMU"] = dea_copy["DMU"].astype(str)

    merged = pd.merge(
        dea_copy[["DMU", "efficiency"]],
        df_orig[[x_col, y_col, z_col, "DMU"]],
        on="DMU",
        how="inner"
    )

    # Forzamos numérico en las columnas de interés
    merged[x_col] = pd.to_numeric(merged[x_col], errors="coerce")
    merged[y_col] = pd.to_numeric(merged[y_col], errors="coerce")
    merged[z_col] = pd.to_numeric(merged[z_col], errors="coerce")
    merged["efficiency"] = pd.to_numeric(merged["efficiency"], errors="coerce")

    # Eliminar filas con NaN en cualquiera de esas columnas
    merged = merged.dropna(subset=[x_col, y_col, z_col, "efficiency"])

    if merged.empty:
        raise ValueError("No hay datos válidos para dibujar el Scatter 3D.")

    # Crear el scatter 3D
    fig = px.scatter_3d(
        merged,
        x=x_col,
        y=y_col,
        z=z_col,
        color="efficiency",
        hover_name="DMU",
        title=f"Scatter 3D: {x_col} vs {y_col} vs {z_col} (coloreado por eficiencia)",
        color_continuous_scale="Viridis",
        labels={
            x_col: x_col,
            y_col: y_col,
            z_col: z_col,
            "efficiency": "Eficiencia"
        }
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col,
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )
    return fig


def plot_benchmark_spider(
    merged_df: pd.DataFrame,
    selected_dmu: str,
    inputs: list[str],
    outputs: list[str],
):
    """
    Genera un gráfico tipo spider (radar) comparando la DMU seleccionada
    contra el máximo valor entre todas las DMU para cada variable de input/output.
    - merged_df debe contener al menos las columnas: 'DMU', cada input y cada output.
    """
    # Validaciones mínimas
    if "DMU" not in merged_df.columns:
        raise ValueError("El DataFrame debe tener la columna 'DMU'.")
    if selected_dmu not in merged_df["DMU"].astype(str).values:
        raise ValueError(f"La DMU '{selected_dmu}' no existe en el DataFrame.")
    if len(inputs) + len(outputs) < 1:
        raise ValueError("Se necesita al menos un Input u Output para Spider.")

    # Filtrar solo los valores numéricos de inputs+outputs
    # y normalizarlos en [0, 1] usando como referencia el máximo global
    df_num = merged_df.copy()
    df_num["DMU"] = df_num["DMU"].astype(str)

    # Calcular valores máximos para cada variable (inputs+outputs)
    max_vals = {}
    for col in inputs + outputs:
        df_num[col] = pd.to_numeric(df_num[col], errors="coerce")
        max_vals[col] = df_num[col].max(skipna=True)
        if pd.isna(max_vals[col]) or max_vals[col] == 0:
            max_vals[col] = 1  # evitar división por cero

    # Extraer valores de la DMU seleccionada
    row = df_num.loc[df_num["DMU"] == selected_dmu]
    if row.empty:
        raise ValueError(f"No se encontró la fila para '{selected_dmu}' en el DataFrame.")

    # Preparar lista de categorías y valores normalizados
    categories = []
    values = []
    for col in inputs + outputs:
        categories.append(col)
        val = row[col].iloc[0]
        try:
            norm_val = float(val) / float(max_vals[col])
        except Exception:
            norm_val = 0
        values.append(norm_val)

    # Radar espera valores “cerrados”, es decir, repetir el primer valor al final
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill="toself",
                name=f"DMU: {selected_dmu}",
                marker=dict(symbol="circle")
            )
        ]
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
        ),
        title=f"Benchmark Spider: {selected_dmu}",
        showlegend=False,
    )
    return fig
