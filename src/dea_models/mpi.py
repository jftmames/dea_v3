# src/dea_models/mpi.py
import numpy as np
import pandas as pd
import cvxpy as cp

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
      DMU, period_t, period_t1, MPI, Efficiency_t, Efficiency_t1, catch_up, frontier_shift
    """
    # Verificar que existan las columnas DMU y período
    if dmu_column not in df_panel.columns:
        raise ValueError(f"La columna DMU '{dmu_column}' no existe en el DataFrame.")
    if period_column not in df_panel.columns:
        raise ValueError(f"La columna de periodo '{period_column}' no existe en el DataFrame.")

    # 1) Validar positividad en todos los datos de inputs/outputs
    validate_positive_dataframe(df_panel, input_cols, output_cols)

    # 2) Obtener lista ordenada de períodos
    periods = sorted(df_panel[period_column].unique())
    if len(periods) < 2:
        raise ValueError("Se requieren al menos dos períodos para Malmquist.")

    resultados = []
    dmus_unique = df_panel[dmu_column].astype(str).unique().tolist() # Unique DMUs across all periods

    # 3) Para cada DMU, calcular índices en cada par (t, t+1)
    for dmu_id in dmus_unique:
        # Filtramos por DMU, no por período aquí, para luego iterar sobre períodos
        # df_dmu = df_panel[df_panel[dmu_column] == dmu] # This line is not needed here

        for idx in range(len(periods) - 1):
            t = periods[idx]
            t1 = periods[idx + 1]

            df_t = df_panel[df_panel[period_column] == t]
            df_t1 = df_panel[df_panel[period_column] == t1]

            # Validar que la DMU existe en ambos períodos
            dmu_in_t = df_t[df_t[dmu_column].astype(str) == dmu_id]
            dmu_in_t1 = df_t1[df_t1[dmu_column].astype(str) == dmu_id]

            if dmu_in_t.empty or dmu_in_t1.empty:
                continue # Skip if DMU does not exist in both consecutive periods

            # Construir X_t, Y_t, X_t1, Y_t1 para todas las DMUs en el respectivo período
            X_t_all = df_t[input_cols].to_numpy().T
            Y_t_all = df_t[output_cols].to_numpy().T
            X_t1_all = df_t1[input_cols].to_numpy().T
            Y_t1_all = df_t1[output_cols].to_numpy().T

            # Índices de la DMU actual (dmu_id) en cada panel (df_t y df_t1)
            # Asegurarse de que el índice es correcto para la DMU dentro de df_t y df_t1
            idx_in_t = df_t[dmu_column].astype(str).tolist().index(dmu_id)
            idx_in_t1 = df_t1[dmu_column].astype(str).tolist().index(dmu_id)

            # 4) Calcular las 4 eficiencias necesarias para Malmquist:
            # - E_t_t: Eficiencia de DMU(t) con frontera(t)
            # - E_t1_t1: Eficiencia de DMU(t+1) con frontera(t+1)
            # - E_t_t1: Eficiencia de DMU(t) con frontera(t+1)
            # - E_t1_t: Eficiencia de DMU(t+1) con frontera(t)

            # E_t_t
            eff_t_t = _run_dea_core_panel(X_t_all, Y_t_all, idx_in_t, rts)
            # E_t1_t1
            eff_t1_t1 = _run_dea_core_panel(X_t1_all, Y_t1_all, idx_in_t1, rts)

            # Para E_t_t1 (DMU en t, frontera en t+1)
            # La DMU en t es la fila `idx_in_t` del dataframe df_t.
            # La frontera es el conjunto de datos de df_t1.
            # Aquí, la matriz de inputs/outputs para la DMU a evaluar debe ser X_t[:,[idx_in_t]] y Y_t[:,[idx_in_t]]
            # y la matriz de referencia debe ser X_t1_all, Y_t1_all
            eff_t_t1 = _run_dea_core_cross_period(X_t_all[:,[idx_in_t]], Y_t_all[:,[idx_in_t]], X_t1_all, Y_t1_all, rts)

            # Para E_t1_t (DMU en t+1, frontera en t)
            # La DMU en t+1 es la fila `idx_in_t1` del dataframe df_t1.
            # La frontera es el conjunto de datos de df_t.
            eff_t1_t = _run_dea_core_cross_period(X_t1_all[:,[idx_in_t1]], Y_t1_all[:,[idx_in_t1]], X_t_all, Y_t_all, rts)


            # 5) Calcular índice Malmquist
            mpi_val = np.nan
            catch_up = np.nan
            front_shift = np.nan

            if eff_t_t > 0 and eff_t_t1 > 0: # Ensure denominators are positive
                # Catch-up (o recovery) = E_t1_t / E_t_t
                catch_up = eff_t1_t / eff_t_t

                # Frontier shift (o technological change) = sqrt((E_t_t / E_t_t1) * (E_t1_t / E_t1_t1))
                # The provided formula in original code for MPI is: sqrt((eff_t1_t / eff_t_t) * (eff_t1_t1 / eff_t_t1))
                # This corresponds to: sqrt(Catch-up * Efficiency_change) where Efficiency_change is E_t1_t1/E_t_t1
                # Or, often it's: MPI = Technical_Efficiency_Change * Technological_Change
                # Technical_Efficiency_Change = eff_t1_t1 / eff_t_t (from t to t+1 against own frontier)
                # Technological_Change = sqrt((eff_t_t / eff_t_t1) * (eff_t1_t / eff_t1_t1)) -- this is not correct for the ratio
                # The standard decomposition is: MPI = (eff_t1_t1 / eff_t_t) * sqrt((eff_t_t / eff_t_t1) * (eff_t1_t / eff_t1_t1))
                # Let's stick to the MPI formula provided in the original code, which is usually for overall.
                # MPI = sqrt((eff_t1_t / eff_t_t) * (eff_t1_t1 / eff_t_t1))
                # Frontier Shift = sqrt((E_t_t / E_t_t1) * (E_t1_t / E_t1_t1))
                
                # Re-calculating Frontier Shift based on common Malmquist decomposition:
                # Technological Change Index (TCI) = sqrt( (Eff_t_t / Eff_t_t1) * (Eff_t1_t / Eff_t1_t1) )
                if eff_t_t1 > 0 and eff_t1_t > 0 and eff_t1_t1 > 0:
                    front_shift = np.sqrt((eff_t_t / eff_t_t1) * (eff_t1_t / eff_t1_t1))
                else:
                    front_shift = np.nan # Avoid division by zero or negative efficiencies
                
                if not np.isnan(catch_up) and not np.isnan(front_shift):
                    mpi_val = catch_up * front_shift # MPI = Catch-up * Frontier-shift (this is the decomposition)
                else:
                    mpi_val = np.nan # If any component is nan, MPI is nan

            resultados.append({
                "DMU": dmu_id,
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
    Función auxiliar que corre el núcleo DEA (CCR o BCC) para una DMU específica
    dentro de un panel, contra su propia frontera temporal.
    - X: m×n (inputs de todas las DMUs en el período actual), Y: s×n (outputs de todas las DMUs en el período actual)
    - idx: índice de la DMU a evaluar en ese panel (column index in X and Y)
    - rts: "CRS" o "VRS"
    Retorna eficiencia float.
    """
    m, n = X.shape
    lambdas = cp.Variable((n, 1), nonneg=True)
    theta = cp.Variable()

    # Target DMU's inputs and outputs
    x_i = X[:, [idx]]
    y_i = Y[:, [idx]]

    cons = []
    cons.append(Y @ lambdas >= y_i)
    cons.append(X @ lambdas <= theta * x_i)
    if rts == "VRS":
        cons.append(cp.sum(lambdas) == 1)

    obj = cp.Minimize(theta)
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.ECOS, abstol=1e-6, reltol=1e-6, feastol=1e-8, verbose=False)
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return float(theta.value) if theta.value is not None else np.nan
        else:
            return np.nan # Solver did not find an optimal solution
    except Exception:
        return np.nan # Solver failed


def _run_dea_core_cross_period(
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    X_ref: np.ndarray,
    Y_ref: np.ndarray,
    rts: str
) -> float:
    """
    Función auxiliar para calcular eficiencia de una DMU de un período
    contra la frontera de otro período.
    - x_eval: m×1 (inputs de la DMU a evaluar)
    - y_eval: s×1 (outputs de la DMU a evaluar)
    - X_ref: m×n (inputs de DMUs de referencia), Y_ref: s×n (outputs de DMUs de referencia)
    - rts: "CRS" o "VRS"
    Retorna eficiencia float.
    """
    m, n_ref = X_ref.shape
    lambdas = cp.Variable((n_ref, 1), nonneg=True)
    theta = cp.Variable()

    cons = []
    cons.append(Y_ref @ lambdas >= y_eval)
    cons.append(X_ref @ lambdas <= theta * x_eval)
    if rts == "VRS":
        cons.append(cp.sum(lambdas) == 1)

    obj = cp.Minimize(theta)
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.ECOS, abstol=1e-6, reltol=1e-6, feastol=1e-8, verbose=False)
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return float(theta.value) if theta.value is not None else np.nan
        else:
            return np.nan # Solver did not find an optimal solution
    except Exception:
        return np.nan # Solver failed
