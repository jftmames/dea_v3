# src/results.py

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dea_models.radial import run_ccr, run_bcc


def mostrar_resultados(df: pd.DataFrame, dmu_column: str, inputs: list[str], outputs: list[str]):
    """
    Ejecuta los modelos CCR y BCC sobre el DataFrame proporcionado, mostrando resultados
    y devolviendo un dict con DataFrames y figuras para su posterior uso.
    """
    # 1) Ejecutar CCR
    df_ccr = run_ccr(
        df=df,
        dmu_column=dmu_column,
        input_cols=inputs,
        output_cols=outputs,
        orientation="input",
        super_eff=False
    )

    # 2) Ejecutar BCC
    df_bcc = run_bcc(
        df=df,
        dmu_column=dmu_column,
        input_cols=inputs,
        output_cols=outputs,
        orientation="input",
        super_eff=False
    )

    # 3) Unir resultados de eficiencia con df original
    merged_ccr = df_ccr.merge(df, on="DMU", how="left")
    merged_bcc = df_bcc.merge(df, on="DMU", how="left")

    # 4) Crear figuras
    hist_ccr = plot_efficiency_histogram(df_ccr)
    hist_bcc = plot_efficiency_histogram(df_bcc)

    scatter3d_ccr = plot_3d_inputs_outputs(df, inputs, outputs, df_ccr)
    scatter3d_bcc = plot_3d_inputs_outputs(df, inputs, outputs, df_bcc)

    # 5) Empaquetar todo en un dict
    resultados = {
        "df_ccr": df_ccr,
        "df_bcc": df_bcc,
        "merged_ccr": merged_ccr,
        "merged_bcc": merged_bcc,
        "hist_ccr": hist_ccr,
        "hist_bcc": hist_bcc,
        "scatter3d_ccr": scatter3d_ccr,
        "scatter3d_bcc": scatter3d_bcc,
    }
    return resultados


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


def plot_3d_inputs_outputs(
    orig_df: pd.DataFrame,
    inputs: list[str],
    outputs: list[str],
    dea_df: pd.DataFrame,
):
    """
    Scatter 3D: ejes = primeros 2 inputs, tercer eje = primer output,
    coloreado según eficiencia. 'dea_df' debe tener columnas 'DMU' y 'efficiency'.
    """
    # Uno dea_df con orig_df para obtener valores de inputs/outputs
    merged = dea_df.merge(orig_df, on="DMU", how="left")

    # Seleccionamos primeras 2 columnas de inputs y la primera de outputs
    x_col = inputs[0]
    y_col = inputs[1] if len(inputs) >= 2 else inputs[0]
    z_col = outputs[0]

    # Validar existencia de columnas
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
    Radar/spider plot comparando la DMU seleccionada vs. la media de las DMU eficientes.
    merged_for_spider: DataFrame con columnas 'DMU', all inputs, all outputs y 'efficiency'.
    selected_dmu: identificador de la DMU a plotear.
    """
    # Filtrar DMUs eficientes (efficiency == 1)
    efficient_peers = merged_for_spider.query("efficiency == 1")
    if efficient_peers.empty:
        raise ValueError("No hay DMU eficientes para benchmark.")

    # Variables a comparar (inputs + outputs)
    vars_all = inputs + outputs

    # Promedio de peers eficientes
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

    # DataFrame para Plotly
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
