# src/dea_models/window_analysis.py

import pandas as pd
import numpy as np

from .radial import _run_dea_core_panel
from .utils import validate_positive_dataframe

def run_window_dea(
    df_panel: pd.DataFrame,
    dmu_column: str,
    period_column: str,
    input_cols: list[str],
    output_cols: list[str],
    window_size: int = 3,
    step: int = 1,
    rts: str = "CRS"
) -> pd.DataFrame:
    """
    Corre DEA en ventanas temporales:
      - df_panel: DataFrame con columnas [dmu_column, period_column, inputs..., outputs...]
      - window_size: número de períodos en cada ventana
      - step: desplazamiento de ventana
      - rts: "CRS" o "VRS"
    Devuelve DataFrame con columnas:
      DMU, start_period, end_period, efficiency
    """
    # 1) Validar positividad en inputs/outputs
    cols = input_cols + output_cols
    validate_positive_dataframe(df_panel, cols)

    # 2) Lista de períodos ordenada
    periods = sorted(df_panel[period_column].unique())
    if len(periods) < window_size:
        raise ValueError("No hay suficientes períodos para la ventana indicada.")

    resultados = []
    dmus = df_panel[dmu_column].astype(str).unique().tolist()

    # 3) Iterar ventanas
    for start_idx in range(0, len(periods) - window_size + 1, step):
        start_p = periods[start_idx]
        end_p = periods[start_idx + window_size - 1]
        df_win = df_panel[(df_panel[period_column] >= start_p) & (df_panel[period_column] <= end_p)]

        # Construir X y Y agregados (concatenar columnas o promediar según criterio)
        # Para simplicidad, asumimos que se toma un snapshot final (end_p)
        df_snapshot = df_panel[df_panel[period_column] == end_p]
        X = df_snapshot[input_cols].to_numpy().T
        Y = df_snapshot[output_cols].to_numpy().T

        for i, dmu in enumerate(dmus):
            if dmu not in df_snapshot[dmu_column].astype(str).tolist():
                continue
            idx = df_snapshot.index[df_snapshot[dmu_column] == dmu][0]
            eff = _run_dea_core_panel(X, Y, idx, rts)
            resultados.append({
                "DMU": dmu,
                "start_period": start_p,
                "end_period": end_p,
                "efficiency": eff
            })

    return pd.DataFrame(resultados)
