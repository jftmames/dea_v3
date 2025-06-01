# src/dea_models/radial.py

import numpy as np
import cvxpy as cp
import pandas as pd

from .utils import validate_positive_dataframe

# ------------------------------------------------------------------
# 1. Núcleo DEA (CCR / BCC) — input/output orientation
# ------------------------------------------------------------------
def _dea_core(
    X: np.ndarray,
    Y: np.ndarray,
    rts: str = "CRS",
    orientation: str = "input",
    super_eff: bool = False,
) -> np.ndarray:
    """
    X: np.array shape (m, n)  (inputs)
    Y: np.array shape (s, n)  (outputs)
    rts: "CRS" (CCR) o "VRS" (BCC)
    orientation: "input" o "output"
    super_eff: si True, excluye la DMU actual para super-eficiencia
    Devuelve vector de eficiencias (length n).
    """
    m, n = X.shape
    eff = np.zeros(n)

    for i in range(n):
        if super_eff:
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            X_mat = X[:, mask]
            Y_mat = Y[:, mask]
            num_vars = n - 1
        else:
            X_mat = X
            Y_mat = Y
            num_vars = n

        lambdas = cp.Variable((num_vars, 1), nonneg=True)

        if orientation == "input":
            theta = cp.Variable()
            cons = [
                Y_mat @ lambdas >= Y[:, [i]],
                X_mat @ lambdas <= theta * X[:, [i]],
            ]
            if rts == "VRS":
                cons.append(cp.sum(lambdas) == 1)
            obj = cp.Minimize(theta)
        else:  # output-oriented
            phi = cp.Variable()
            cons = [
                Y_mat @ lambdas >= phi * Y[:, [i]],
                X_mat @ lambdas <= X[:, [i]],
            ]
            if rts == "VRS":
                cons.append(cp.sum(lambdas) == 1)
            obj = cp.Maximize(phi)

        prob = cp.Problem(obj, cons)
        prob.solve(
            solver=cp.ECOS,
            abstol=1e-6,
            reltol=1e-6,
            feastol=1e-8,
            verbose=False
        )

        if orientation == "input":
            eff[i] = float(theta.value) if theta.value is not None else np.nan
        else:
            eff[i] = float(phi.value) if phi.value is not None else np.nan

    return eff


# ------------------------------------------------------------------
# 2. Función interna que antes se llamaba run_dea
# ------------------------------------------------------------------
def _run_dea_internal(
    df: pd.DataFrame,
    inputs: list[str],
    outputs: list[str],
    model: str = "CCR",
    orientation: str = "input",
    super_eff: bool = False,
) -> pd.DataFrame:
    """
    Ejecuta DEA (CCR/BCC, input/output, opcional super-eficiencia).
    Parámetros:
      df: DataFrame original (incluye identificador index o columna "DMU")
      inputs: lista de nombres de columnas de inputs (numéricos y > 0)
      outputs: lista de nombres de columnas de outputs (numéricos y > 0)
      model: "CCR" o "BCC"
      orientation: "input" o "output"
      super_eff: True para super-eficiencia (excluye DMU actual)
    Retorna DataFrame con columnas:
      DMU, efficiency, model, orientation, super_eff
    """
    # validaciones iniciales
    assert orientation in ("input", "output"), "orientation debe ser 'input' u 'output'"
    assert model.upper() in ("CCR", "BCC"), "model debe ser 'CCR' o 'BCC'"

    # 1) Extraer sólo columnas requeridas y convertir a float
    cols = inputs + outputs
    df_num = validate_positive_dataframe(df, cols)

    # 2) Construir matrices X y Y (shape: m×n, s×n)
    X = df_num[inputs].to_numpy().T
    Y = df_num[outputs].to_numpy().T

    # 3) Definir returns-to-scale
    rts = "CRS" if model.upper() == "CCR" else "VRS"

    # 4) Llamar al núcleo
    eff = _dea_core(X, Y, rts=rts, orientation=orientation, super_eff=super_eff)

    # 5) Preparar el DataFrame de salida
    if "DMU" in df.columns:
        dmu_ids = df["DMU"].astype(str)
    else:
        dmu_ids = df.index.astype(str)

    return pd.DataFrame({
        "DMU": dmu_ids,
        "efficiency": np.round(eff, 6),
        "model": model.upper(),
        "orientation": orientation,
        "super_eff": bool(super_eff),
    })


# ------------------------------------------------------------------
# 3. Función pública: run_ccr
# ------------------------------------------------------------------
def run_ccr(
    df: pd.DataFrame,
    dmu_column: str,
    input_cols: list[str],
    output_cols: list[str],
    orientation: str = "input",
    super_eff: bool = False
) -> pd.DataFrame:
    """
    Ejecuta CCR radial y retorna DataFrame con columnas:
      DMU, tec_efficiency_ccr, lambda_vector, slacks_inputs, slacks_outputs, rts_label
    """
    if dmu_column not in df.columns:
        raise ValueError(f"La columna DMU '{dmu_column}' no existe.")

    # Validar que todos los inputs y outputs sean positivos
    cols = input_cols + output_cols
    validate_positive_dataframe(df, cols)

    # Matrices de datos
    X = df[input_cols].to_numpy().T  # shape: (m, n)
    Y = df[output_cols].to_numpy().T  # shape: (s, n)
    dmus = df[dmu_column].astype(str).tolist()
    n = X.shape[1]  # número de DMUs
    m = X.shape[0]  # número de inputs
    s = Y.shape[0]  # número de outputs

    resultados = []
    for i in range(n):
        # Variables de decisión
        lambdas = cp.Variable((n, 1), nonneg=True)
        theta = cp.Variable()
        slacks_in = cp.Variable((m, 1), nonneg=True)
        slacks_out = cp.Variable((s, 1), nonneg=True)

        # Observaciones de la DMU i
        x0 = X[:, [i]]
        y0 = Y[:, [i]]

        cons = []
        # Restricciones principales
        cons.append(Y @ lambdas >= y0)
        cons.append(X @ lambdas + slacks_in == theta * x0)
        cons.append(Y @ lambdas - slacks_out == y0)

        if super_eff:
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            # Ajustar X_mat, Y_mat y lambdas para excluir la DMU i
            lambdas = cp.Variable((n - 1, 1), nonneg=True)
            X_mat = X[:, mask]
            Y_mat = Y[:, mask]
            cons = [
                Y_mat @ lambdas >= y0,
                X_mat @ lambdas + slacks_in == theta * x0,
                Y_mat @ lambdas - slacks_out == y0,
            ]

        # No hay restricción Σλ = 1 en CCR (CRS)
        obj = cp.Minimize(theta)
        prob = cp.Problem(obj, cons)
        prob.solve(
            solver=cp.ECOS,
            abstol=1e-6,
            reltol=1e-6,
            feastol=1e-8,
            verbose=False
        )

        # Extracción de valores
        theta_val = float(theta.value) if theta.value is not None else np.nan

        # Vector lambda
        if super_eff:
            lambdas_vals = {}
            idx = 0
            for j in range(n):
                if j == i:
                    continue
                lambdas_vals[dmus[j]] = float(lambdas.value[idx])
                idx += 1
        else:
            lambdas_vals = {dmus[j]: float(lambdas.value[j]) for j in range(n)}

        slacks_input = {input_cols[k]: float(slacks_in.value[k]) for k in range(m)}
        slacks_output = {output_cols[r]: float(slacks_out.value[r]) for r in range(s)}

        resultados.append({
            dmu_column: dmus[i],
            "tec_efficiency_ccr": theta_val,
            "lambda_vector": lambdas_vals,
            "slacks_inputs": slacks_input,
            "slacks_outputs": slacks_output,
            "rts_label": "CRS"   # CCR siempre CRS
        })

    return pd.DataFrame(resultados)


# ------------------------------------------------------------------
# 4. Función pública: run_bcc (actualizada)
# ------------------------------------------------------------------
def run_bcc(
    df: pd.DataFrame,
    dmu_column: str,
    input_cols: list[str],
    output_cols: list[str],
    orientation: str = "input",
    super_eff: bool = False
) -> pd.DataFrame:
    """
    Ejecuta BCC radial y retorna DataFrame con columnas:
      DMU, tec_efficiency_bcc, lambda_vector, slacks_inputs, slacks_outputs, scale_efficiency, rts_label
    """
    if dmu_column not in df.columns:
        raise ValueError(f"La columna DMU '{dmu_column}' no existe.")

    cols = input_cols + output_cols
    validate_positive_dataframe(df, cols)

    X = df[input_cols].to_numpy().T  # shape: (m, n)
    Y = df[output_cols].to_numpy().T  # shape: (s, n)
    dmus = df[dmu_column].astype(str).tolist()
    n = X.shape[1]  # número de DMUs
    m = X.shape[0]  # número de inputs
    s = Y.shape[0]  # número de outputs

    resultados = []
    for i in range(n):
        # Variables de decisión
        lambdas = cp.Variable((n, 1), nonneg=True)
        theta = cp.Variable()
        slacks_in = cp.Variable((m, 1), nonneg=True)
        slacks_out = cp.Variable((s, 1), nonneg=True)

        # Observaciones de la DMU i
        x0 = X[:, [i]]
        y0 = Y[:, [i]]

        cons = []
        # Restricciones principales (VRS incluye Σλ = 1)
        cons.append(Y @ lambdas >= y0)
        cons.append(X @ lambdas <= theta * x0)
        cons.append(cp.sum(lambdas) == 1)

        obj = cp.Minimize(theta)
        prob = cp.Problem(obj, cons)
        prob.solve(
            solver=cp.ECOS,
            abstol=1e-6,
            reltol=1e-6,
            feastol=1e-8,
            verbose=False
        )

        theta_val = float(theta.value) if theta.value is not None else np.nan
        lambdas_vals = {dmus[j]: float(lambdas.value[j]) for j in range(n)}
        slacks_input = {input_cols[k]: float(slacks_in.value[k]) for k in range(m)}
        slacks_output = {output_cols[r]: float(slacks_out.value[r]) for r in range(s)}

        # Para calcular scale_efficiency: corremos CCR
        df_ccr = run_ccr(
            df=df,
            dmu_column=dmu_column,
            input_cols=input_cols,
            output_cols=output_cols,
            orientation=orientation,
            super_eff=False
        )
        ccr_eff = df_ccr.loc[df_ccr[dmu_column] == dmus[i], "tec_efficiency_ccr"].iloc[0]
        scale_eff = ccr_eff / theta_val if (theta_val and theta_val > 0) else np.nan

        # RTS_label según duales:
        # si el dual de Σλ=1 en BCC es ±0 => CRS;
        # si dual < 0 => IRS; si dual > 0 => DRS.
        dual_sum = prob.constraints[-1].dual_value  # restricción Σλ = 1
        if abs(dual_sum) < 1e-6:
            rts_label = "CRS"
        elif dual_sum < 0:
            rts_label = "IRS"
        else:
            rts_label = "DRS"

        resultados.append({
            dmu_column: dmus[i],
            "tec_efficiency_bcc": theta_val,
            "lambda_vector": lambdas_vals,
            "slacks_inputs": slacks_input,
            "slacks_outputs": slacks_output,
            "scale_efficiency": scale_eff,
            "rts_label": rts_label
        })

    return pd.DataFrame(resultados)
