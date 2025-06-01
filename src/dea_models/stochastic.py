# src/dea_models/stochastic.py

import numpy as np
import pandas as pd

from .radial import _run_dea_core_panel
from .utils import validate_positive_dataframe

def run_stochastic_dea(
    df: pd.DataFrame,
    dmu_column: str,
    input_cols: list[str],
    output_cols: list[str],
    orientation: str = "input",
    rts: str = "CRS",
    n_bootstrap: int = 1000
) -> pd.DataFrame:
    """
    Corre DEA con bootstrapping: por cada muestra re-muestreada,
    calcula eficiencia y retorna intervalos de confianza.
    Retorna DataFrame con:
      DMU, eff_mean, ci_lower, ci_upper, original_efficiency
    """
    if dmu_column not in df.columns:
        raise ValueError(f"La columna DMU '{dmu_column}' no existe en el DataFrame.")

    # 1) Validar positividad
    cols = input_cols + output_cols
    validate_positive_dataframe(df, cols)

    dmus = df[dmu_column].astype(str).tolist()
    n = len(dmus)

    # 2) Eficiencia original (snapshot único)
    X = df[input_cols].to_numpy().T
    Y = df[output_cols].to_numpy().T

    original = []
    for i in range(n):
        eff_i = _run_dea_core_panel(X, Y, i, rts)
        original.append(eff_i)

    # 3) Bootstrapping
    bootstrap_vals = {dmu: [] for dmu in dmus}
    for _ in range(n_bootstrap):
        # Re-muestrear con reemplazo
        df_b = df.sample(n=n, replace=True).reset_index(drop=True)
        Xb = df_b[input_cols].to_numpy().T
        Yb = df_b[output_cols].to_numpy().T

        for i, dmu in enumerate(dmus):
            # Encontrar índice de la DMU en la muestra bootstrapped
            indices = df_b.index[df_b[dmu_column] == dmu].tolist()
            if not indices:
                continue
            idx_b = indices[0]
            eff_b = _run_dea_core_panel(Xb, Yb, idx_b, rts)
            bootstrap_vals[dmu].append(eff_b)

    # 4) Construir DataFrame de resultados
    rows = []
    for i, dmu in enumerate(dmus):
        arr = np.array(bootstrap_vals[dmu])
        if arr.size == 0:
            continue
        mean_b = float(np.mean(arr))
        lower = float(np.percentile(arr, 2.5))
        upper = float(np.percentile(arr, 97.5))
        rows.append({
            "DMU": dmu,
            "eff_mean": mean_b,
            "ci_lower": lower,
            "ci_upper": upper,
            "original_efficiency": original[i]
        })

    return pd.DataFrame(rows)
