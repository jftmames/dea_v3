# src/dea_models/nonradial.py
import numpy as np
import pandas as pd
import cvxpy as cp

from src.dea_models.utils import validate_positive_dataframe # Corregido a importación absoluta
from src.dea_models.directions import get_direction_vector, get_custom_direction_vector # Corregido a importación absoluta

def run_sbm(
    df: pd.DataFrame,
    dmu_column: str,
    input_cols: list[str],
    output_cols: list[str],
    orientation: str = "input",
    rts: str = "VRS"
) -> pd.DataFrame:
    """
    SBM (slack-based measure) input-oriented o output-oriented.
    Parámetros:
      - df: DataFrame con todos los datos.
      - dmu_column: nombre de la columna DMU.
      - input_cols, output_cols: listas de columnas numéricas (permiten >0).
      - orientation: "input" o "output".
      - rts: "CRS" o "VRS".
    Retorna DataFrame con columnas:
      DMU, efficiency_sbm, slacks_inputs (dict), slacks_outputs (dict), lambda_vector (dict).
    """
    if dmu_column not in df.columns:
        raise ValueError(f"La columna DMU '{dmu_column}' no existe en el DataFrame.")

    # 1) Validación explicitando que no hay ceros/negativos (solo positivos)
    cols = input_cols + output_cols
    validate_positive_dataframe(df, cols)

    # 2) Construir matrices (igual que en radial, pero en formato m×n y s×n)
    X = df[input_cols].to_numpy().T    # shape m×n
    Y = df[output_cols].to_numpy().T   # shape s×n
    dmus = df[dmu_column].astype(str).tolist()
    n = X.shape[1]
    m = X.shape[0]
    s = Y.shape[0]

    resultados = []
    for i in range(n):
        # Variables lambda (n variables)
        lambdas = cp.Variable((n, 1), nonneg=True)
        # Variables slack de inputs y outputs
        s_minus = cp.Variable((m, 1), nonneg=True)
        s_plus = cp.Variable((s, 1), nonneg=True)

        # Escape rápido si orientación no es válida
        if orientation not in ("input", "output"):
            raise ValueError("orientation debe ser 'input' u 'output'")

        # Construir restricciones y objetivo según SBM
        x0 = X[:, [i]]    # (m×1)
        y0 = Y[:, [i]]    # (s×1)

        cons = []
        # Y λ + s_plus = y0
        cons.append(Y @ lambdas + s_plus == y0)
        # X λ - s_minus = x0
        cons.append(X @ lambdas - s_minus == x0)
        # Σ λ = 1 (solo si VRS)
        if rts == "VRS":
            cons.append(cp.sum(lambdas) == 1)
        elif rts != "CRS":
            raise ValueError("rts debe ser 'CRS' o 'VRS'")

        if orientation == "input":
            # Función objetivo: min (1 - (1/m) Σ (s_i^-)/x0_i)
            # Que es equivalente a: min (1/m) Σ (x0_i - s_i^-)/x0_i
            # Para SBM, la eficiencia es 1 - (1/m) * sum(s_minus / x0)
            # El objetivo de cvxpy es minimizar (sum(s_minus / x0))
            # Luego la eficiencia es 1 - (1/m) * optimal_obj_value
            obj = cp.Minimize(cp.sum(s_minus / x0) / m) # Minimiza el promedio de slacks fraccionales
        else:  # output-oriented
            # Función objetivo: max (1 + (1/s) Σ (s_r^+)/y0_r)
            # O equivalentemente: max (1/s) Σ (y0_r + s_r^+)/y0_r
            # El objetivo de cvxpy es maximizar (sum(s_plus / y0))
            # Luego la eficiencia es 1 + (1/s) * optimal_obj_value
            obj = cp.Maximize(cp.sum(s_plus / y0) / s) # Maximiza el promedio de slacks fraccionales

        prob = cp.Problem(obj, cons)
        prob.solve(
            solver=cp.ECOS,
            abstol=1e-6,
            reltol=1e-6,
            feastol=1e-8,
            verbose=False
        )

        # Leer eficiencia, slacks y lambdas
        eff_val = np.nan
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            if orientation == "input":
                # Eficiencia SBM = 1 - (1/m) * sum(s_minus / x0)
                # El valor objetivo de CVXPY es (sum(s_minus / x0)) / m
                eff_val = 1 - float(prob.value) if prob.value is not None else np.nan
            else:  # output-oriented
                # Eficiencia SBM = 1 + (1/s) * sum(s_plus / y0)
                # El valor objetivo de CVXPY es (sum(s_plus / y0)) / s
                eff_val = 1 + float(prob.value) if prob.value is not None else np.nan
        
        lambdas_vals = {dmus[j]: float(lambdas.value[j]) for j in range(n)} if lambdas.value is not None else {}
        slacks_in = {input_cols[k]: float(s_minus.value[k]) for k in range(m)} if s_minus.value is not None else {col:np.nan for col in input_cols}
        slacks_out = {output_cols[r]: float(s_plus.value[r]) for r in range(s)} if s_plus.value is not None else {col:np.nan for col in output_cols}

        resultados.append({
            dmu_column: dmus[i],
            "efficiency_sbm": eff_val,
            "lambda_vector": lambdas_vals,
            "slacks_inputs": slacks_in,
            "slacks_outputs": slacks_out
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
    Radial Distance Function (Directional Distance).
    dir_method: 'max_ratios' o 'unit' o 'custom'.
    Retorna DataFrame con columnas:
      DMU, distance_score, lambda_vector, slacks_inputs, slacks_outputs.
    """
    if dmu_column not in df.columns:
        raise ValueError(f"La columna DMU '{dmu_column}' no existe en el DataFrame.")

    # 1) Validación (permitimos zeros/negativos en RD)
    cols = input_cols + output_cols
    # Validar solo conversión a float; permitimos ≤0
    df_num = df.copy()
    for c in cols:
        df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
        if df_num[c].isna().any():
            raise ValueError(f"Columna '{c}' tiene valores no numéricos.")

    # Matrices X, Y
    X = df_num[input_cols].to_numpy().T  # m×n
    Y = df_num[output_cols].to_numpy().T # s×n
    dmus = df[dmu_column].astype(str).tolist()
    n = X.shape[1]
    m = X.shape[0]
    s = Y.shape[0]

    # 2) Determinar dirección (g_x, g_y)
    dir_vec = get_direction_vector(df_num, input_cols, output_cols, method=dir_method)
    g_x = dir_vec["g_x"].reshape((m, 1))
    g_y = dir_vec["g_y"].reshape((s, 1))

    resultados = []
    for i in range(n):
        lambdas = cp.Variable((n, 1), nonneg=True)
        # Variable t (distancia direccional)
        t = cp.Variable()

        x0 = X[:, [i]]
        y0 = Y[:, [i]]

        cons = []
        # Y λ >= y0 + t * g_y
        cons.append(Y @ lambdas >= y0 + t * g_y)
        # X λ <= x0 - t * g_x
        cons.append(X @ lambdas <= x0 - t * g_x)
        if rts == "VRS":
            cons.append(cp.sum(lambdas) == 1)
        elif rts != "CRS":
            raise ValueError("rts debe ser 'CRS' o 'VRS'")

        obj = cp.Minimize(t)
        prob = cp.Problem(obj, cons)
        prob.solve(
            solver=cp.ECOS,
            abstol=1e-6,
            reltol=1e-6,
            feastol=1e-8,
            verbose=False
        )

        t_val = float(t.value) if t.value is not None else np.nan
        lambdas_vals = {dmus[j]: float(lambdas.value[j]) for j in range(n)} if lambdas.value is not None else {}

        # Slack de insumos: x0 - t*g_x - Xλ
        slacks_in = {}
        if lambdas.value is not None:
            Xl = X @ lambdas.value
            for k in range(m):
                slacks_in[input_cols[k]] = float(x0[k] - t_val * g_x[k] - Xl[k])
                if slacks_in[input_cols[k]] < 1e-9: slacks_in[input_cols[k]] = 0.0 # Umbral para cero
        else:
            slacks_in = {col: np.nan for col in input_cols}

        # Slack de outputs: Yλ - y0 - t*g_y
        slacks_out = {}
        if lambdas.value is not None:
            Yl = Y @ lambdas.value
            for r in range(s):
                slacks_out[output_cols[r]] = float(Yl[r] - y0[r] - t_val * g_y[r])
                if slacks_out[output_cols[r]] < 1e-9: slacks_out[output_cols[r]] = 0.0 # Umbral para cero
        else:
            slacks_out = {col: np.nan for col in output_cols}

        resultados.append({
            dmu_column: dmus[i],
            "distance_score": t_val,
            "lambda_vector": lambdas_vals,
            "slacks_inputs": slacks_in,
            "slacks_outputs": slacks_out
        })

    return pd.DataFrame(resultados)
