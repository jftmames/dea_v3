# /mount/src/-dea-deliberativo-mvp/src/dea_models/visualizations.py

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st # Importa streamlit, necesario si usas st.error

# --- Funciones de Visualización ---

def plot_efficiency_histogram(df_results: pd.DataFrame, efficiency_col: str = 'tec_efficiency_ccr') -> go.Figure:
    """
    Genera un histograma de la distribución de la eficiencia técnica.
    """
    fig = px.histogram(
        df_results,
        x=efficiency_col,
        nbins=20,
        title='Distribución de la Eficiencia Técnica (CCR)',
        labels={efficiency_col: 'Eficiencia Técnica'},
        template='plotly_white'
    )
    fig.update_layout(bargap=0.1)
    return fig

def plot_benchmark_spider(
    df_results: pd.DataFrame,
    df_original: pd.DataFrame,
    dmu_col: str,
    inputs: list,
    outputs: list,
    efficiency_col: str = 'tec_efficiency_ccr',
    title: str = 'Spider Plot de Rendimiento Promedio (Eficientes vs Ineficientes)'
) -> go.Figure:
    """
    Genera un gráfico de araña (spider plot) para visualizar los inputs/outputs
    normalizados de DMUs eficientes e ineficientes.

    Parámetros:
    df_results (pd.DataFrame): DataFrame con los resultados del análisis DEA,
                               incluyendo la columna de eficiencia.
    df_original (pd.DataFrame): DataFrame original con los datos de inputs/outputs.
    dmu_col (str): Nombre de la columna que identifica las DMUs.
    inputs (list): Lista de nombres de columnas de inputs.
    outputs (list): Lista de nombres de columnas de outputs.
    efficiency_col (str): Nombre de la columna de eficiencia (default: 'tec_efficiency_ccr').
    title (str): Título del gráfico.
    """
    all_vars = inputs + outputs

    # Asegurarse de que las columnas necesarias existen en df_original
    required_original_cols = [dmu_col] + all_vars
    if not all(col in df_original.columns for col in required_original_cols):
        st.error(f"Error: Faltan columnas de inputs/outputs en el DataFrame original. "
                 f"Asegúrate de que {all_vars} estén en {dmu_col}.")
        return go.Figure().update_layout(title_text="Error en datos de entrada para Spider Plot.")

    # Combinar resultados de eficiencia con datos originales
    df_merged = df_results.merge(df_original[[dmu_col] + all_vars], on=dmu_col, how="left")

    # Separar eficientes e ineficientes
    df_efficient = df_merged[df_merged[efficiency_col] >= 0.999].copy()
    df_inefficient = df_merged[df_merged[efficiency_col] < 0.999].copy()

    # Normalizar los datos para el spider plot
    # Normalizamos cada variable entre 0 y 1 usando el rango global (min y max)
    # de cada variable en el DataFrame original completo para una comparación justa.
    df_normalized = df_original[all_vars].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Calcular los promedios normalizados para el spider plot
    avg_efficient_normalized = df_normalized.loc[df_merged[efficiency_col] >= 0.999].mean().tolist()
    avg_inefficient_normalized = df_normalized.loc[df_merged[efficiency_col] < 0.999].mean().tolist()

    categories = all_vars

    # Para cerrar el círculo en el spider plot, se repite el primer valor al final
    avg_efficient_normalized = avg_efficient_normalized + [avg_efficient_normalized[0]] if avg_efficient_normalized else [0] * (len(categories) + 1)
    avg_inefficient_normalized = avg_inefficient_normalized + [avg_inefficient_normalized[0]] if avg_inefficient_normalized else [0] * (len(categories) + 1)
    categories_closed = categories + [categories[0]] if categories else []

    fig = go.Figure()

    if avg_efficient_normalized:
        fig.add_trace(go.Scatterpolar(
            r=avg_efficient_normalized,
            theta=categories_closed,
            fill='toself',
            name='DMUs Eficientes (Promedio Normalizado)',
            marker=dict(color='green'),
            opacity=0.7
        ))

    if avg_inefficient_normalized:
        fig.add_trace(go.Scatterpolar(
            r=avg_inefficient_normalized,
            theta=categories_closed,
            fill='toself',
            name='DMUs Ineficientes (Promedio Normalizado)',
            marker=dict(color='red'),
            opacity=0.7
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1] # Rango de 0 a 1 para datos normalizados
            )),
        showlegend=True,
        title_text=title
    )

    return fig

def plot_3d_inputs_outputs(
    df_results: pd.DataFrame,
    df_original: pd.DataFrame,
    dmu_col: str,
    input_cols: list,
    output_cols: list,
    efficiency_col: str = 'tec_efficiency_ccr'
) -> go.Figure:
    """
    Genera un scatter plot 3D de inputs y outputs, coloreado por eficiencia.
    Requiere al menos 2 inputs y 1 output, o 1 input y 2 outputs, o 3 variables en total.
    """
    all_plotting_vars = []
    if len(input_cols) >= 1:
        all_plotting_vars.append(input_cols[0])
    if len(output_cols) >= 1:
        all_plotting_vars.append(output_cols[0])
    if len(input_cols) >= 2:
        all_plotting_vars.append(input_cols[1])
    if len(output_cols) >= 2 and len(all_plotting_vars) < 3: # Asegura que no agregamos más de 3 si ya tenemos suficientes
        all_plotting_vars.append(output_cols[1])
    
    # Tomar las primeras 3 variables disponibles para el gráfico 3D
    x_col = all_plotting_vars[0] if len(all_plotting_vars) > 0 else None
    y_col = all_plotting_vars[1] if len(all_plotting_vars) > 1 else None
    z_col = all_plotting_vars[2] if len(all_plotting_vars) > 2 else None

    if not x_col or not y_col or not z_col:
        st.warning("No hay suficientes inputs/outputs para un gráfico 3D significativo (se necesitan al menos 3 variables).")
        return go.Figure().update_layout(title_text="Gráfico 3D no disponible: Insuficientes variables.")

    df_merged = df_results.merge(df_original[[dmu_col, x_col, y_col, z_col]], on=dmu_col, how="left")
    df_merged['Estatus'] = df_merged[efficiency_col].apply(lambda x: 'Eficiente' if x >= 0.999 else 'Ineficiente')

    fig = px.scatter_3d(
        df_merged,
        x=x_col,
        y=y_col,
        z=z_col,
        color='Estatus',
        title=f"Rendimiento en 3D: {x_col} vs {y_col} vs {z_col}",
        color_discrete_map={'Eficiente': 'green', 'Ineficiente': 'red'},
        hover_name=dmu_col
    )
    fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    return fig

def plot_slack_waterfall(
    df_results: pd.DataFrame,
    dmu_col: str,
    slack_cols: list,
    dmu_selected: str
) -> go.Figure:
    """
    Genera un gráfico de cascada (waterfall) para mostrar las holguras de inputs/outputs
    para una DMU seleccionada.
    """
    if dmu_selected not in df_results[dmu_col].values:
        return go.Figure().update_layout(title_text=f"Error: La DMU '{dmu_selected}' no se encontró.")

    df_dmu = df_results[df_results[dmu_col] == dmu_selected].iloc[0]

    data = []
    measures = []
    x_labels = []

    for col in slack_cols:
        if col in df_dmu:
            value = df_dmu[col]
            if value != 0:
                data.append(value)
                measures.append("relative") # Todas son relativas al inicio, no una base inicial
                x_labels.append(col)

    if not data:
        return go.Figure().update_layout(title_text=f"No hay holguras para mostrar para {dmu_selected}.")

    fig = go.Figure(go.Waterfall(
        name="Holguras",
        orientation="v",
        measure=measures,
        x=x_labels,
        textposition="outside",
        text=[f"{val:.2f}" for val in data],
        y=data,
        connector={"line":{"color":"rgb(63, 63, 63)"}},
    ))

    fig.update_layout(
        title = f"Holguras para DMU: {dmu_selected}",
        showlegend = True
    )
    return fig


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

    df_merged = df_results.merge(df_original[[dmu_col, variable]], on=dmu_col, how="left")
    df_merged['Estatus'] = df_merged[efficiency_col].apply(lambda x: 'Eficiente' if x >= 0.999 else 'Ineficiente')

    fig = px.box(
        df_merged, x='Estatus', y=variable, color='Estatus',
        title=f"Comparación de '{variable}' entre Unidades Eficientes e Ineficientes",
        points="all", labels={"Estatus": "Estatus de Eficiencia", variable: f"Valor de {variable}"},
        color_discrete_map={'Eficiente': 'green', 'Ineficiente': 'red'}
    )
    fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    return fig

# --- NUEVA FUNCIÓN PARA EL TALLER DE HIPÓTESIS ---
def plot_correlation(
    df_results: pd.DataFrame,
    df_original: pd.DataFrame,
    var_x: str,
    var_y: str,
    dmu_col: str,
    efficiency_col: str = 'tec_efficiency_ccr'
) -> go.Figure:
    """
    Genera un scatter plot para analizar la correlación entre dos variables,
    coloreando por estatus de eficiencia.
    """
    if var_x not in df_original.columns or var_y not in df_original.columns:
        return go.Figure().update_layout(title_text=f"Error: Una de las variables no existe.")
        
    df_merged = df_results.merge(df_original[[dmu_col, var_x, var_y]], on=dmu_col, how="left")
    df_merged['Estatus'] = df_merged[efficiency_col].apply(lambda x: 'Eficiente' if x >= 0.999 else 'Ineficiente')

    fig = px.scatter(
        df_merged, x=var_x, y=var_y, color='Estatus',
        title=f"Correlación entre '{var_x}' y '{var_y}'",
        color_discrete_map={'Eficiente': 'green', 'Ineficiente': 'red'},
        hover_name=dmu_col,
        trendline="ols", # Añade una línea de tendencia
        trendline_scope="overall"
    )
    fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    return fig
