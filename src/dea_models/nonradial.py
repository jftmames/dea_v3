import numpy as np
import pandas as pd
import cvxpy as cp

# Importaciones relativas (dentro de dea_models/)
from .utils import validate_positive_dataframe
from .directions import get_direction_vector, get_custom_direction_vector

def run_sbm(
    df: pd.DataFrame,
    dmu_column: str,
    input_cols: list[str],
    output_cols: list[str],
    orientation: str = "input",
    rts: str = "VRS"
) -> pd.DataFrame:
    """
    SBM (Slack-Based Measure) input-oriented u output-oriented.
    Parámetros:
      - df: DataFrame con todos los datos.
      - dmu_column: nombre de la columna DMU.
      - input_cols, output_cols: listas de columnas numéricas (>0).
      - orientation: "input" u "output".
      - rts: "CRS" o "VRS".
    Retorna DataFrame con columnas:
      DMU, efficiency_sbm, slacks_inputs (dict), slacks_outputs (dict), lambda_vector (dict).
    """
    if dmu_column not in df.columns:
        raise ValueError(f"La columna DMU '{dmu_column}' no existe en el DataFrame.")

    # 1) Validación de que no hay ceros/negativos (solo positivos)
    cols = input_cols + output_cols
    validate_positive_dataframe(df, cols)

    # 2) Construir matrices X (m×n) e Y (s×n)
    X = df[input_cols].to_numpy().T    # shape: (m, n)
    Y = df[output_cols].to_numpy().T   # shape: (s, n)
    dmus = df[dmu_column].astype(str).tolist()
    n = X.shape[1]
    m = X.shape[0]
    s = Y.shape[0]

    resultados = []
    for i in range(n):
        lambdas = cp.Variable((n, 1), nonneg=True)       # vector λ (n × 1)
        s_minus = cp.Variable((m, 1), nonneg=True)       # slacks de inputs (m × 1)
        s_plus = cp.Variable((s, 1), nonneg=True)        # slacks de outputs (s × 1)

        if orientation not in ("input", "output"):
            raise ValueError("orientation debe ser 'input' u 'output'")

        x0 = X[:, [i]]   # insumos de la DMU i  (m × 1)
        y0 = Y[:, [i]]   # productos de la DMU i  (s × 1)

        cons = []
        # Y·λ + s_plus = y0
        cons.append(Y @ lambdas + s_plus == y0)
        # X·λ - s_minus = x0
        cons.append(X @ lambdas - s_minus == x0)
        # Σ λ = 1 si VRS
        if rts == "VRS":
            cons.append(cp.sum(lambdas) == 1)
        elif rts != "CRS":
            raise ValueError("rts debe ser 'CRS' o 'VRS'")

        if orientation == "input":
            # Objetivo (minimizar sum(s_minus / x0) / m)
            obj = cp.Minimize(cp.sum(s_minus / x0) / m)
        else:  # output-oriented
            # Objetivo (maximizar sum(s_plus / y0) / s)
            obj = cp.Maximize(cp.sum(s_plus / y0) / s)

        prob = cp.Problem(obj, cons)
        prob.solve(
            solver=cp.ECOS,
            abstol=1e-6,
            reltol=1e-6,
            feastol=1e-8,
            verbose=False
        )

        eff_val = np.nan
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            if orientation == "input":
                eff_val = 1 - float(prob.value) if prob.value is not None else np.nan
            else:
                eff_val = 1 + float(prob.value) if prob.value is not None else np.nan

        lambdas_vals = {}
        if lambdas.value is not None:
            for j in range(n):
                lambdas_vals[dmus[j]] = float(lambdas.value[j, 0])

        slacks_in = {}
        if s_minus.value is not None:
            for k in range(m):
                slacks_in[input_cols[k]] = float(s_minus.value[k, 0])
        else:
            for col in input_cols:
                slacks_in[col] = np.nan

        slacks_out = {}
        if s_plus.value is not None:
            for r in range(s):
                slacks_out[output_cols[r]] = float(s_plus.value[r, 0])
        else:
            for col in output_cols:
                slacks_out[col] = np.nan

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
    Parámetros:
      - df: DataFrame original.
      - dmu_column: nombre de la columna DMU.
      - input_cols, output_cols: listas de columnas numéricas (permitimos ≤0).
      - dir_method: 'max_ratios', 'unit' o 'custom'.
      - rts: 'CRS' o 'VRS'.
    Retorna DataFrame con:
      DMU, distance_score, lambda_vector, slacks_inputs, slacks_outputs.
    """
    if dmu_column not in df.columns:
        raise ValueError(f"La columna DMU '{dmu_column}' no existe en el DataFrame.")

    # Validación de que input_cols/output_cols sean numéricos (≤0 permitido aquí)
    cols = input_cols + output_cols
    df_num = df.copy()
    for c in cols:
        df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
        if df_num[c].isna().any():
            raise ValueError(f"Columna '{c}' tiene valores no numéricos.")

    X = df_num[input_cols].to_numpy().T   # m × n
    Y = df_num[output_cols].to_numpy().T  # s × n
    dmus = df[dmu_column].astype(str).tolist()
    n = X.shape[1]
    m = X.shape[0]
    s = Y.shape[0]

    # 2) Determinar vector dirección (g_x, g_y)
    dir_vec = get_direction_vector(df_num, input_cols, output_cols, method=dir_method)
    g_x = dir_vec["g_x"].reshape((m, 1))  # (m × 1)
    g_y = dir_vec["g_y"].reshape((s, 1))  # (s × 1)

    resultados = []
    for i in range(n):
        lambdas = cp.Variable((n, 1), nonneg=True)
        t = cp.Variable()

        x0 = X[:, [i]]
        y0 = Y[:, [i]]

        cons = []
        # Y·λ ≥ y0 + t·g_y
        cons.append(Y @ lambdas >= y0 + t * g_y)
        # X·λ ≤ x0 - t·g_x
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
        lambdas_vals = {}
        if lambdas.value is not None:
            for j in range(n):
                lambdas_vals[dmus[j]] = float(lambdas.value[j, 0])

        # Calcular slacks de inputs: x0 - t·g_x - X·λ
        slacks_in = {}
        if lambdas.value is not None:
            Xl = X @ lambdas.value
            for k in range(m):
                raw = float(x0[k, 0] - t_val * g_x[k, 0] - Xl[k, 0])
                slacks_in[input_cols[k]] = 0.0 if raw < 1e-9 else raw
        else:
            for col in input_cols:
                slacks_in[col] = np.nan

        # Calcular slacks de outputs: Y·λ - y0 - t·g_y
        slacks_out = {}
        if lambdas.value is not None:
            Yl = Y @ lambdas.value
            for r in range(s):
                raw = float(Yl[r, 0] - y0[r, 0] - t_val * g_y[r, 0])
                slacks_out[output_cols[r]] = 0.0 if raw < 1e-9 else raw
        else:
            for col in output_cols:
                slacks_out[col] = np.nan

        resultados.append({
            dmu_column: dmus[i],
            "distance_score": t_val,
            "lambda_vector": lambdas_vals,
            "slacks_inputs": slacks_in,
            "slacks_outputs": slacks_out
        })

    return pd.DataFrame(resultados)
