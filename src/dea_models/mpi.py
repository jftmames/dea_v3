# src/dea_models/mpi.py

import numpy as np
import pandas as pd

from .radial import _dea_core  # usamos el núcleo DEA radial
from .utils import validate_positive_dataframe

def compute_malmquist_phi(
    df_panel: pd.DataFrame,
    dmu_column: str,
    period_column: str,
    input_cols: list[str],
    output_cols: list[str],
    rts: str = "CRS"
) -> pd.DataFrame:
    """
    Calcula el índice de Malmquist para cada par de períodos consecutivos.
    df_panel: DataFrame con todas las DMUs y datos por período.
    period_column: nombre de la columna de período (ej. 'year').
    input_cols, output_cols: listas de columnas.
    Retorna DataFrame con columnas:
      DMU, period_t, period_t+1, MPI, Efficiency_t, Efficiency_t_plus1, 技change (catch-up), frontier_shift
    """
    # 1) Validar positividad en todos los datos
    cols = input_cols + output_cols + [period_column, dmu_column]
    # Extraemos subset numérico
    df_num = df_panel.copy()
    # Verificar que inputs/outputs sean numéricos y >0
    validate_positive_dataframe(df_num, input_cols + output_cols)

    # 2) Obtener lista ordenada de períodos
    periods = sorted(df_panel[period_column].unique())
    if len(periods) < 2:
        raise ValueError("Se requieren al menos dos períodos para Malmquist.")

    resultados = []
    dmus = df_panel[dmu_column].astype(str).unique().tolist()

    # 3) Para cada DMU, calcular índices en cada par (t, t+1)
    for i, dmu in enumerate(dmus):
        df_dmu = df_panel[df_panel[dmu_column] == dmu]
        for idx in range(len(periods) - 1):
            t = periods[idx]
            t1 = periods[idx + 1]

            df_t = df_panel[df_panel[period_column] == t]
            df_t1 = df_panel[df_panel[period_column] == t1]

            # Validar que la DMU existe en ambos períodos
            if dmu not in df_t[dmu_column].astype(str).tolist() or dmu not in df_t1[dmu_column].astype(str).tolist():
                continue

            # Construir X_t, Y_t, X_t1, Y_t1
            X_t = df_t[input_cols].to_numpy().T
            Y_t = df_t[output_cols].to_numpy().T
            X_t1 = df_t1[input_cols].to_numpy().T
            Y_t1 = df_t1[output_cols].to_numpy().T

            # Indices de DMU en cada panel
            idx_t = df_t.index[df_t[dmu_column] == dmu][0]
            idx_t1 = df_t1.index[df_t1[dmu_column] == dmu][0]

            # 4) Eficiencia de DMU(t) con frontera t y t+1
            eff_t_t = _run_dea_core_panel(X_t, Y_t, idx_t, rts)      # efficiency en periodo t usando frontera t
            eff_t1_t = _run_dea_core_panel(X_t1, Y_t1, idx_t1, rts)  # efficiency en periodo t+1 usando frontera t
            eff_t_t1 = _run_dea_core_panel(X_t, Y_t, idx_t, rts)     # efficiency en periodo t usando frontera t+1
            eff_t1_t1 = _run_dea_core_panel(X_t1, Y_t1, idx_t1, rts) # efficiency en periodo t+1 usando frontera t+1

            # 5) Calcular índice Malmquist
            # MPI = sqrt((eff_t1_t / eff_t_t) * (eff_t1_t1 / eff_t_t1))
            try:
                mpi_val = np.sqrt((eff_t1_t / eff_t_t) * (eff_t1_t1 / eff_t_t1))
            except:
                mpi_val = np.nan

            # catch-up: eff_t1_t / eff_t_t
            catch_up = eff_t1_t / eff_t_t if eff_t_t > 0 else np.nan
            # frontier shift: eff_t1_t1 / eff_t_t1
            front_shift = eff_t1_t1 / eff_t_t1 if eff_t_t1 > 0 else np.nan

            resultados.append({
                "DMU": dmu,
                "period_t": t,
                "period_t1": t1,
                "MPI": mpi_val,
                "Efficiency_t": eff_t_t,
                "Efficiency_t1": eff_t1_t1,
                "catch_up": catch_up,
                "frontier_shift": front_shift
            })

    return pd.DataFrame(resultados)


def _run_dea_core_panel(
    X: np.ndarray,
    Y: np.ndarray,
    idx: int,
    rts: str
) -> float:
    """
    Función auxiliar que corre el núcleo DEA (CCR o BCC) en panel:
    - X: m×n, Y: s×n
    - idx: índice de la DMU en ese panel
    - rts: "CRS" o "VRS"
    Retorna eficiencia float.
    """
    m, n = X.shape
    lambdas = cp.Variable((n, 1), nonneg=True)
    theta = cp.Variable()

    # Restricciones
    cons = []
    cons.append(Y @ lambdas >= Y[:, [idx]])
    cons.append(X @ lambdas <= theta * X[:, [idx]])
    if rts == "VRS":
        cons.append(cp.sum(lambdas) == 1)

    obj = cp.Minimize(theta)
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.ECOS, abstol=1e-6, reltol=1e-6, feastol=1e-8, verbose=False)

    return float(theta.value) if theta.value is not None else np.nan
