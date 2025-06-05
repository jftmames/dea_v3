# jftmames/-dea-deliberativo-mvp/-dea-deliberativo-mvp-b44b8238c978ae0314af30717b9399634d28f8f9/src/dea_models/mpi.py
import numpy as np
import pandas as pd
import cvxpy as cp

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
    """
    if dmu_column not in df_panel.columns:
        raise ValueError(f"La columna DMU '{dmu_column}' no existe.")
    if period_column not in df_panel.columns:
        raise ValueError(f"La columna de periodo '{period_column}' no existe.")

    validate_positive_dataframe(df_panel, input_cols + output_cols)
    periods = sorted(df_panel[period_column].unique())
    if len(periods) < 2:
        raise ValueError("Se requieren al menos dos períodos para Malmquist.")

    resultados = []
    dmus_unique = df_panel[dmu_column].astype(str).unique().tolist()

    for dmu_id in dmus_unique:
        for idx in range(len(periods) - 1):
            t, t1 = periods[idx], periods[idx + 1]
            df_t = df_panel[df_panel[period_column] == t]
            df_t1 = df_panel[df_panel[period_column] == t1]

            dmu_in_t = df_t[df_t[dmu_column].astype(str) == dmu_id]
            dmu_in_t1 = df_t1[df_t1[dmu_column].astype(str) == dmu_id]
            if dmu_in_t.empty or dmu_in_t1.empty: continue

            X_t_all, Y_t_all = df_t[input_cols].to_numpy().T, df_t[output_cols].to_numpy().T
            X_t1_all, Y_t1_all = df_t1[input_cols].to_numpy().T, df_t1[output_cols].to_numpy().T

            idx_in_t = df_t[dmu_column].astype(str).tolist().index(dmu_id)
            idx_in_t1 = df_t1[dmu_column].astype(str).tolist().index(dmu_id)

            # E_t_t, E_t1_t1, E_t_t1, E_t1_t
            eff_t_t = _run_dea_core_panel(X_t_all, Y_t_all, idx_in_t, rts)
            eff_t1_t1 = _run_dea_core_panel(X_t1_all, Y_t1_all, idx_in_t1, rts)
            eff_t_t1 = _run_dea_core_cross_period(X_t_all[:,[idx_in_t]], Y_t_all[:,[idx_in_t]], X_t1_all, Y_t1_all, rts)
            eff_t1_t = _run_dea_core_cross_period(X_t1_all[:,[idx_in_t1]], Y_t1_all[:,[idx_in_t1]], X_t_all, Y_t_all, rts)

            # --- CORRECCIÓN LÓGICA ---
            # La fórmula original del catch_up era incorrecta.
            # Catch-up (Efficiency Change) = E_t1_t1 / E_t_t
            catch_up = eff_t1_t1 / eff_t_t if eff_t_t > 1e-9 else np.nan
            
            # Frontier Shift (Technical Change)
            front_shift = np.sqrt((eff_t_t / eff_t_t1) * (eff_t1_t / eff_t1_t1)) if eff_t_t1 > 1e-9 and eff_t1_t1 > 1e-9 else np.nan
            
            mpi_val = catch_up * front_shift if not np.isnan(catch_up) and not np.isnan(front_shift) else np.nan
            
            resultados.append({
                "DMU": dmu_id, "period_t": t, "period_t1": t1,
                "MPI": mpi_val, "efficiency_change": catch_up, "technical_change": front_shift,
                "E_t_t": eff_t_t, "E_t1_t1": eff_t1_t1
            })
    return pd.DataFrame(resultados)

def _run_dea_core_panel(X, Y, idx, rts):
    m, n = X.shape
    theta, lambdas = cp.Variable(), cp.Variable((n, 1), nonneg=True)
    cons = [Y @ lambdas >= Y[:, [idx]], X @ lambdas <= theta * X[:, [idx]]]
    if rts == "VRS": cons.append(cp.sum(lambdas) == 1)
    prob = cp.Problem(cp.Minimize(theta), cons)
    try:
        prob.solve(solver=cp.ECOS, abstol=1e-8)
        return float(theta.value) if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] else np.nan
    except Exception: return np.nan

def _run_dea_core_cross_period(x_eval, y_eval, X_ref, Y_ref, rts):
    m, n_ref = X_ref.shape
    theta, lambdas = cp.Variable(), cp.Variable((n_ref, 1), nonneg=True)
    cons = [Y_ref @ lambdas >= y_eval, X_ref @ lambdas <= theta * x_eval]
    if rts == "VRS": cons.append(cp.sum(lambdas) == 1)
    prob = cp.Problem(cp.Minimize(theta), cons)
    try:
        prob.solve(solver=cp.ECOS, abstol=1e-8)
        return float(theta.value) if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] else np.nan
    except Exception: return np.nan
