# src/dea_models/window_analysis.py

import pandas as pd
import numpy as np

from .radial import run_ccr   # <- usamos run_ccr en lugar de _run_dea_core_panel
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
    #    (window-analysis asume datos >0; si quieres permitir ceros, cambiar flags)
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

        # Filtrar todos los registros cuyos period_column estén en esta ventana
        df_window = df_panel[df_panel[period_column].isin(periods[start_idx : start_idx + window_size])]

        # Para calcular eficiencia de cada DMU en la ventana, llamamos a run_ccr
        # pero run_ccr espera un DataFrame con una sola "snapshot" (interfaces), 
        # así que para cada DMU tomamos el conjunto de datos de la ventana y calculamos:
        #
        #   df_snapshot = df_window[df_window[period_column] == end_p]
        #   y luego correr run_ccr sobre ese snapshot (inputs+outputs),
        #   asumiendo que dentro de la ventana nos interesa el estado final.
        #
        # Si quisieras promediar eficiencias en toda la ventana, tendrías que iterar
        # por cada período de la ventana. Aquí tomamos el snapshot en end_p.

        df_snapshot = df_window[df_window[period_column] == end_p].copy()
        if df_snapshot.empty:
            continue

        # Construir el DataFrame que pide run_ccr:
        # - debe incluir la columna DMU y las columnas input_cols+output_cols
        # - run_ccr (df, dmu_column, input_cols, output_cols, model="CCR", orientation="input", super_eff=False)
        try:
            df_eff = run_ccr(
                df=df_snapshot,
                dmu_column=dmu_column,
                input_cols=input_cols,
                output_cols=output_cols,
                orientation="input",
                super_eff=False
            )
        except Exception as e:
            raise RuntimeError(f"Error al correr run_ccr para ventana {start_p}–{end_p}: {e}")

        # df_eff tiene columns ['DMU', 'efficiency', 'model', 'orientation', 'super_eff']
        # Guardamos, para cada DMU, la eficiencia correspondiente a esa ventana:
        for _, row in df_eff.iterrows():
            registros.append({
                dmu_column: row[dmu_column],
                "start_period": start_p,
                "end_period": end_p,
                "efficiency_window": row["efficiency"]
            })

    return pd.DataFrame(registros)
