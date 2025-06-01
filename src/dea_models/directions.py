# src/dea_models/directions.py

import numpy as np
import pandas as pd

def get_direction_vector(df: pd.DataFrame, input_cols: list[str], output_cols: list[str], method: str = "max_ratios"):
    """
    Genera un vector direccional (g_x, g_y).
    method:
      - "max_ratios": g_x[i] = max(df[input_cols[i]]), g_y[r] = max(df[output_cols[r]])
      - "unit": todos 1's
    Retorna dict {'g_x': array(m,), 'g_y': array(s,)}
    """
    m = len(input_cols)
    s = len(output_cols)

    if method == "max_ratios":
        g_x = df[input_cols].max().to_numpy()
        g_y = df[output_cols].max().to_numpy()
    elif method == "unit":
        g_x = np.ones(m)
        g_y = np.ones(s)
    else:
        raise ValueError(f"Método '{method}' no soportado para dirección.")

    return {"g_x": g_x, "g_y": g_y}

def get_custom_direction_vector(gx_list: list[float], gy_list: list[float]):
    """
    Retorna {'g_x': array(gx_list), 'g_y': array(gy_list)}.
    """
    if len(gx_list) == 0 or len(gy_list) == 0:
        raise ValueError("Listas de dirección no pueden estar vacías.")
    return {"g_x": np.array(gx_list), "g_y": np.array(gy_list)}
