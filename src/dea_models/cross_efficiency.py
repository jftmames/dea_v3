# jftmames/-dea-deliberativo-mvp/-dea-deliberativo-mvp-b44b8238c978ae0314af30717b9399634d28f8f9/src/dea_models/cross_efficiency.py
import numpy as np
import pandas as pd
import cvxpy as cp

from .utils import validate_positive_dataframe

def _solve_ccr_dual(X: np.ndarray, Y: np.ndarray, dmu_index: int):
    """
    Resuelve el problema dual del CCR para una DMU para obtener los pesos u, v.
    """
    m, n = X.shape
    s = Y.shape[0]
    x_k = X[:, [dmu_index]]
    y_k = Y[:, [dmu_index]]

    u = cp.Variable((s, 1), nonneg=True)  # Pesos de outputs
    v = cp.Variable((m, 1), nonneg=True)  # Pesos de inputs

    # Restricciones del dual
    # La eficiencia de ninguna DMU puede ser > 1 con estos pesos
    cons = [
        u.T @ Y - v.T @ X <= 0,
        v.T @ x_k == 1 # Normalización
    ]
    
    # Objetivo: maximizar la eficiencia de la DMU k
    obj = cp.Maximize(u.T @ y_k)
    
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.ECOS, abstol=1e-8)
    
    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and u.value is not None and v.value is not None:
        return u.value, v.value
    return None, None

def compute_cross_efficiency(
    df: pd.DataFrame,
    dmu_column: str,
    input_cols: list[str],
    output_cols: list[str],
    rts: str = "CRS", # Cross-efficiency se define clásicamente para CCR (CRS)
) -> pd.DataFrame:
    """
    Calcula la matriz de eficiencias cruzadas.
    """
    if dmu_column not in df.columns:
        raise ValueError(f"La columna DMU '{dmu_column}' no existe en el DataFrame.")
    validate_positive_dataframe(df, input_cols + output_cols)

    dmus = df[dmu_column].astype(str).tolist()
    n = len(dmus)
    X = df[input_cols].to_numpy().T
    Y = df[output_cols].to_numpy().T
    
    # 1. Obtener los pesos óptimos (u_j, v_j) para cada DMU j
    all_weights = {}
    for j in range(n):
        u_j, v_j = _solve_ccr_dual(X, Y, j)
        if u_j is not None and v_j is not None:
            all_weights[j] = (u_j, v_j)

    # 2. Calcular la matriz de eficiencia cruzada E_ij
    cross_eff_matrix = np.full((n, n), np.nan)
    for j, (u_j, v_j) in all_weights.items(): # DMU 'j' es la que evalúa
        for i in range(n): # DMU 'i' es la evaluada
            x_i = X[:, [i]]
            y_i = Y[:, [i]]
            
            numerator = u_j.T @ y_i
            denominator = v_j.T @ x_i
            
            # Evitar división por cero
            if denominator > 1e-9:
                cross_eff_matrix[i, j] = (numerator / denominator).item()

    # 3. Crear el DataFrame final
    df_cross = pd.DataFrame(cross_eff_matrix, index=dmus, columns=dmus)
    
    # 4. Calcular el score promedio para cada DMU
    df_cross['Average Score'] = df_cross.mean(axis=1)
    df_cross = df_cross.sort_values('Average Score', ascending=False)
    
    return df_cross
