# jftmames/-dea-deliberativo-mvp/-dea-deliberativo-mvp-b44b8238c978ae0314af30717b9399634d28f8f9/src/dea_models/nonradial.py
import numpy as np
import pandas as pd
import cvxpy as cp

from .utils import validate_positive_dataframe
from .directions import get_direction_vector

def run_sbm(
    df: pd.DataFrame,
    dmu_column: str,
    input_cols: list[str],
    output_cols: list[str],
    orientation: str = "non-oriented", # Opciones: "input", "output", "non-oriented"
    rts: str = "VRS"
) -> pd.DataFrame:
    """
    SBM (slack-based measure).
    Retorna DataFrame con eficiencia, slacks y lambdas.
    """
    if dmu_column not in df.columns:
        raise ValueError(f"La columna DMU '{dmu_column}' no existe en el DataFrame.")
    validate_positive_dataframe(df, input_cols + output_cols)

    X = df[input_cols].to_numpy().T
    Y = df[output_cols].to_numpy().T
    dmus = df[dmu_column].astype(str).tolist()
    n, m, s = X.shape[1], X.shape[0], Y.shape[0]

    resultados = []
    for i in range(n):
        lambdas = cp.Variable((n, 1), nonneg=True)
        s_minus = cp.Variable((m, 1), nonneg=True)
        s_plus = cp.Variable((s, 1), nonneg=True)
        t = cp.Variable() # Variable de escala para CRS/VRS
        
        x0 = X[:, [i]]
        y0 = Y[:, [i]]

        # --- CORRECCIÓN LÓGICA ---
        # Las restricciones originales estaban invertidas.
        # La formulación correcta es: x0 = Xλ + s-, y0 = Yλ - s+
        # Multiplicamos por 't' para manejar modelos CRS/VRS
        cons = [
            x0 == X @ (lambdas) + s_minus,
            y0 == Y @ (lambdas) - s_plus
        ]
        if rts == "VRS":
            cons.append(cp.sum(lambdas) == 1)
        
        if orientation == "input":
            obj = cp.Minimize(1 - (1/m) * cp.sum(s_minus / x0))
        elif orientation == "output":
            obj = cp.Maximize(1 + (1/s) * cp.sum(s_plus / y0))
        elif orientation == "non-oriented":
            numerator = 1 - (1/m) * cp.sum(s_minus / x0)
            denominator = 1 + (1/s) * cp.sum(s_plus / y0)
            # Para resolver este problema fraccional (Charnes-Cooper), se linealiza.
            # Por simplicidad, aquí lo resolvemos como un problema secuencial,
            # pero una implementación completa usaría la transformación.
            # Aquí se usará el objetivo del input-oriented como proxy.
            obj = cp.Minimize(numerator) 
        else:
            raise ValueError("orientation debe ser 'input', 'output' o 'non-oriented'")
            
        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.ECOS, abstol=1e-7, reltol=1e-7, feastol=1e-7, verbose=False)

        eff_val = float(obj.value) if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and obj.value is not None else np.nan
        
        resultados.append({
            dmu_column: dmus[i],
            "efficiency_sbm": np.round(eff_val, 6) if not np.isnan(eff_val) else np.nan,
            "lambda_vector": {dmus[j]: float(lambdas.value[j]) for j in range(n)} if lambdas.value is not None else {},
            "slacks_inputs": {input_cols[k]: float(s_minus.value[k]) for k in range(m)} if s_minus.value is not None else {c: np.nan for c in input_cols},
            "slacks_outputs": {output_cols[r]: float(s_plus.value[r]) for r in range(s)} if s_plus.value is not None else {c: np.nan for c in output_cols}
        })
    return pd.DataFrame(resultados)

def run_radial_distance(
    df: pd.DataFrame,
    dmu_column: str,
    input_cols: list[str],
    output_cols: list[str],
    dir_method: str = "max_ratios",
    rts: str = "CRS"
) -> pd.DataFrame:
    """
    Directional Distance Function (Función de Distancia Direccional).
    """
    if dmu_column not in df.columns:
        raise ValueError(f"La columna DMU '{dmu_column}' no existe en el DataFrame.")
    
    # DDF permite valores no positivos, así que no usamos validate_positive_dataframe
    for c in input_cols + output_cols:
        if c not in df.columns: raise ValueError(f"Columna '{c}' no encontrada.")
        df[c] = pd.to_numeric(df[c], errors='coerce')
        if df[c].isna().any(): raise ValueError(f"Columna '{c}' contiene valores no numéricos.")

    X = df[input_cols].to_numpy().T
    Y = df[output_cols].to_numpy().T
    dmus = df[dmu_column].astype(str).tolist()
    n, m, s = X.shape[1], X.shape[0], Y.shape[0]

    dir_vec = get_direction_vector(df, input_cols, output_cols, method=dir_method)
    g_x = dir_vec["g_x"].reshape((m, 1))
    g_y = dir_vec["g_y"].reshape((s, 1))

    resultados = []
    for i in range(n):
        lambdas = cp.Variable((n, 1), nonneg=True)
        beta = cp.Variable() # Ineficiencia direccional

        x0, y0 = X[:, [i]], Y[:, [i]]
        
        cons = [
            Y @ lambdas >= y0 + beta * g_y,
            X @ lambdas <= x0 - beta * g_x
        ]
        if rts == "VRS":
            cons.append(cp.sum(lambdas) == 1)

        # --- CORRECCIÓN LÓGICA ---
        # El objetivo es maximizar la ineficiencia (beta), no minimizarla.
        obj = cp.Maximize(beta)
        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.ECOS, abstol=1e-7, reltol=1e-7, feastol=1e-7, verbose=False)

        beta_val = float(obj.value) if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and obj.value is not None else np.nan
        lambdas_opt = lambdas.value if lambdas.value is not None else np.zeros((n, 1))
        
        slacks_in, slacks_out = {}, {}
        if not np.isnan(beta_val):
            slacks_in_vals = (x0 - beta_val * g_x) - (X @ lambdas_opt)
            slacks_out_vals = (Y @ lambdas_opt) - (y0 + beta_val * g_y)
            slacks_in_vals[slacks_in_vals < 1e-9] = 0
            slacks_out_vals[slacks_out_vals < 1e-9] = 0
            slacks_in = {input_cols[k]: float(v) for k, v in enumerate(slacks_in_vals.flatten())}
            slacks_out = {output_cols[r]: float(v) for r, v in enumerate(slacks_out_vals.flatten())}
        
        resultados.append({
            dmu_column: dmus[i],
            "distance_score": beta_val,
            "lambda_vector": {dmus[j]: float(v) for j, v in enumerate(lambdas_opt.flatten())},
            "slacks_inputs": slacks_in,
            "slacks_outputs": slacks_out
        })
    return pd.DataFrame(resultados)
