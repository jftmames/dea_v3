# src/dea_models/window_analysis.py
import pandas as pd
import numpy as np

from .radial import run_ccr   
from .utils import validate_dataframe

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
    Corre DEA sobre ventanas temporales consecutivas.
    - df_panel: DataFrame con columnas [dmu_column, period_column, inputs…, outputs…].
    - window_size: número de períodos por ventana.
    - step: desplazamiento entre ventanas.
    - rts: "CRS" o "VRS" (se pasa luego a run_ccr).
    Retorna DataFrame con columnas:
      DMU, start_period, end_period, efficiency_window.
    """
    # 1) Validar que exista la columna DMU y la de período
    if dmu_column not in df_panel.columns:
        raise ValueError(f"La columna DMU '{dmu_column}' no existe en el DataFrame.")
    if period_column not in df_panel.columns:
        raise ValueError(f"La columna de periodo '{period_column}' no existe en el DataFrame.")

    # 2) Validar positividad en inputs/outputs
    validate_dataframe(df_panel, input_cols, output_cols, allow_zero=False, allow_negative=False)

    # 3) Lista de períodos ordenada
    periods = sorted(df_panel[period_column].unique())
    if len(periods) < window_size:
        raise ValueError("No hay suficientes períodos para la ventana solicitada.")

    registros = []

    # 4) Iterar ventanas (start_idx: 0 → len(periods)-window_size, paso=step)
    for start_idx in range(0, len(periods) - window_size + 1, step):
        start_p = periods[start_idx]
        end_p = periods[start_idx + window_size - 1]

        df_window = df_panel[df_panel[period_column].isin(periods[start_idx : start_idx + window_size])]

        df_snapshot = df_window[df_window[period_column] == end_p].copy()
        if df_snapshot.empty:
            continue

        try:
            df_eff = run_ccr( # Run CCR for this snapshot
                df=df_snapshot,
                dmu_column=dmu_column,
                input_cols=input_cols,
                output_cols=output_cols,
                orientation="input", # assuming input-oriented for simplicity or based on CCR default
                super_eff=False
            )
        except Exception as e:
            raise RuntimeError(f"Error al correr run_ccr para ventana {start_p}–{end_p}: {e}")

        for _, row in df_eff.iterrows():
            registros.append({
                dmu_column: row[dmu_column],
                "start_period": start_p,
                "end_period": end_p,
                "efficiency_window": row["efficiency"]
            })

    return pd.DataFrame(registros)
