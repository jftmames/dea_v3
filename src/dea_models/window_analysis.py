# jftmames/-dea-deliberativo-mvp/-dea-deliberativo-mvp-b44b8238c978ae0314af30717b9399634d28f8f9/src/dea_models/window_analysis.py
import pandas as pd
import numpy as np

from .radial import run_ccr, run_bcc
from .utils import validate_dataframe

def run_window_dea(
    df_panel: pd.DataFrame,
    dmu_column: str,
    period_column: str,
    input_cols: list[str],
    output_cols: list[str],
    window_size: int = 3,
    rts: str = "CRS" # Se pasa a run_ccr o run_bcc
) -> pd.DataFrame:
    """
    Corre DEA sobre ventanas temporales deslizantes.
    """
    if dmu_column not in df_panel.columns or period_column not in df_panel.columns:
        raise ValueError("Las columnas de DMU y período deben existir.")
    validate_dataframe(df_panel, input_cols, output_cols, allow_zero=False, allow_negative=False)

    periods = sorted(df_panel[period_column].unique())
    if len(periods) < window_size:
        raise ValueError("No hay suficientes períodos para la ventana solicitada.")

    all_results = []

    for i in range(len(periods) - window_size + 1):
        current_window_periods = periods[i : i + window_size]
        start_p, end_p = current_window_periods[0], current_window_periods[-1]
        
        # --- CORRECCIÓN LÓGICA ---
        # El DataFrame de referencia debe ser toda la ventana.
        # Las DMUs a evaluar son las del último período de la ventana.
        df_window = df_panel[df_panel[period_column].isin(current_window_periods)].copy()
        
        # Para evitar conflictos con nombres de DMU repetidos en distintos períodos,
        # creamos un ID temporal único para el cálculo.
        df_window['temp_dmu_id'] = df_window[dmu_column].astype(str) + " | P:" + df_window[period_column].astype(str)
        
        model_func = run_ccr if rts == "CRS" else run_bcc
        
        # Para BCC, necesitamos resultados de CCR para la eficiencia de escala
        df_ccr_results = run_ccr(
            df=df_window,
            dmu_column='temp_dmu_id',
            input_cols=input_cols,
            output_cols=output_cols
        )
        # Renombramos columna para que run_bcc la encuentre
        df_ccr_results = df_ccr_results.rename(columns={'tec_efficiency_ccr': 'efficiency'})


        if rts == "CRS":
            df_eff = df_ccr_results.rename(columns={'efficiency':'tec_efficiency_ccr'})
        else:
             df_eff = run_bcc(
                df=df_window,
                dmu_column='temp_dmu_id',
                input_cols=input_cols,
                output_cols=output_cols,
                df_ccr_results=df_ccr_results
            )

        # Filtramos para quedarnos solo con los resultados del último período de la ventana
        results_end_period = df_eff[df_eff['temp_dmu_id'].str.contains(f"P:{end_p}")]

        for _, row in results_end_period.iterrows():
            original_dmu = row['temp_dmu_id'].split(" | P:")[0]
            all_results.append({
                "DMU": original_dmu,
                "start_period": start_p,
                "end_period": end_p,
                "efficiency_window": row['tec_efficiency_ccr'] if rts == "CRS" else row['efficiency']
            })

    return pd.DataFrame(all_results)
