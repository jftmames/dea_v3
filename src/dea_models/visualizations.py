# jftmames/-dea-deliberativo-mvp/-dea-deliberativo-mvp-b44b8238c978ae0314af30717b9399634d28f8f9/src/dea_models/visualizations.py
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ... (funciones existentes: plot_efficiency_histogram, plot_benchmark_spider, plot_3d_inputs_outputs, plot_slack_waterfall, plot_hypothesis_distribution) ...
# (Asegúrate de que todas las funciones anteriores estén aquí)


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
