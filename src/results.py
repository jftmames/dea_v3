# ---------- src/results.py ----------
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_efficiency_histogram(dea_df: pd.DataFrame, bins: int = 20) -> go.Figure:
    """
    Histograma de eficiencias (columna 'efficiency' en dea_df).
    """
    if "efficiency" not in dea_df.columns:
        raise ValueError("La columna 'efficiency' no existe en dea_df.")
    fig = px.histogram(
        dea_df,
        x="efficiency",
        nbins=bins,
        title="Distribución de Eficiencias DEA",
        labels={"efficiency": "Eficiencia"},
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


def plot_3d_inputs_outputs(
    df_original: pd.DataFrame,
    inputs: list[str],
    outputs: list[str],
    dea_df: pd.DataFrame,
) -> go.Figure:
    """
    Scatter 3D de los dos primeros inputs vs el primer output, coloreado por eficiencia.
    - Si no existe columna 'DMU' en dea_df, usa el índice.
    - Asegura que inputs ≥ 2 y outputs ≥ 1 antes de llamar a esta función.
    """
    # Comprobar tamaños
    if len(inputs) < 2 or len(outputs) < 1:
        raise ValueError("Se requieren al menos 2 inputs y 1 output para este gráfico.")

    # Unir dea_df con datos originales para obtener variables inputs/outputs
    if "DMU" in dea_df.columns:
        merge_on = "DMU"
        dea_for_merge = dea_df.copy()
    else:
        # si no existe la columna DMU, creamos una en dea_df basada en el índice
        dea_for_merge = dea_df.copy()
        dea_for_merge["DMU_index"] = dea_for_merge.index.astype(str)
        merge_on = "DMU_index"

        df_original = df_original.copy()
        # crear columna paralela en df_original
        if "DMU" in df_original.columns:
            df_original["DMU_index"] = df_original["DMU"].astype(str)
        else:
            df_original["DMU_index"] = df_original.index.astype(str)

    merged = pd.merge(
        dea_for_merge,
        df_original[[*inputs, *outputs, merge_on]],
        left_on=merge_on,
        right_on=merge_on,
        how="left",
    )

    # Comprobar que las columnas de inputs/outputs estén en merged
    for col in inputs + outputs:
        if col not in merged.columns:
            raise ValueError(f"La columna '{col}' no existe en los datos fusionados.")

    # Elegimos los dos primeros inputs y el primer output
    in_x, in_y = inputs[0], inputs[1]
    out_z = outputs[0]
    efficiency_col = "efficiency"
    if efficiency_col not in merged.columns:
        raise ValueError("La columna 'efficiency' no existe en dea_df.")

    # Si existe columna 'DMU', la usamos para hover, si no, el índice
    hover_data = merged["DMU"] if "DMU" in merged.columns else merged[merge_on]

    fig = px.scatter_3d(
        merged,
        x=in_x,
        y=in_y,
        z=out_z,
        color=efficiency_col,
        hover_name=hover_data,
        title=f"Scatter 3D: {in_x} vs {in_y} vs {out_z}",
        labels={in_x: in_x, in_y: in_y, out_z: out_z, efficiency_col: "Eficiencia"},
        color_continuous_scale="Viridis",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig


def plot_benchmark_spider(
    merged_for_spider: pd.DataFrame,
    selected_dmu: str,
    inputs: list[str],
    outputs: list[str],
) -> go.Figure:
    """
    Gráfico tipo radar (spider) comparando la DMU seleccionada contra el promedio
    de las DMU eficientes (efficiency == 1) en cada variable de inputs+outputs.
    Debe recibir un DataFrame que ya contenga 'DMU' y columnas inputs+outputs.
    """
    # Verificar que exista 'DMU'
    if "DMU" not in merged_for_spider.columns:
        raise ValueError("La columna 'DMU' no existe en merged_for_spider.")

    # Filtrar por eficiencia == 1 para calcular benchmark
    reeks = merged_for_spider.query("efficiency == 1")
    if reeks.empty:
        raise ValueError("No hay DMU eficientes (efficiency == 1) para benchmark.")

    # Variables a comparar
    vars_to_compare = inputs + outputs

    # Promedio de DMU eficientes
    benchmark = reeks[vars_to_compare].mean().to_dict()

    # Valores de la DMU seleccionada
    row = merged_for_spider.loc[merged_for_spider["DMU"] == selected_dmu]
    if row.empty:
        raise ValueError(f"No se encontró la DMU '{selected_dmu}' en merged_for_spider.")
    selected_values = {var: float(row[var].values[0]) for var in vars_to_compare}

    # Construir figuras radar con go.Scatterpolar
    categories = vars_to_compare.copy()
    # para cerrar el polígono, repetimos el primer elemento al final
    categories.append(categories[0])

    bench_values = [benchmark[var] for var in vars_to_compare]
    bench_values.append(bench_values[0])

    sel_values = [selected_values[var] for var in vars_to_compare]
    sel_values.append(sel_values[0])

    fig = go.Figure()

    # Trazar benchmark
    fig.add_trace(
        go.Scatterpolar(
            r=bench_values,
            theta=categories,
            fill="toself",
            opacity=0.6,
            name="Benchmark (promedio eficiente)",
        )
    )
    # Trazar DMU seleccionada
    fig.add_trace(
        go.Scatterpolar(
            r=sel_values,
            theta=categories,
            fill="toself",
            opacity=0.6,
            name=f"DMU {selected_dmu}",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(max(bench_values), max(sel_values))])
        ),
        showlegend=True,
        title=f"Spider / Radar: DMU {selected_dmu} vs Promedio Eficientes",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig
