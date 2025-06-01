# dea_models/radial.py

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
# 3. Funciones públicas: run_ccr y run_bcc
# ------------------------------------------------------------------
def run_ccr(
    df: pd.DataFrame,
    dmu_column: str,
    input_cols: list[str],
    output_cols: list[str],
    orientation: str = "input",
    super_eff: bool = False,
) -> pd.DataFrame:
    """
    Ejecuta DEA CCR radial (input/output, opcional super-eficiencia).
    Parámetros:
      df: DataFrame original
      dmu_column: nombre de la columna que identifica cada DMU
      input_cols: lista de columnas de inputs
      output_cols: lista de columnas de outputs
      orientation: "input" o "output"
      super_eff: True para super-eficiencia
    Retorna:
      DataFrame con columnas [DMU, efficiency, model, orientation, super_eff]
    """
    if dmu_column not in df.columns:
        raise ValueError(f"La columna DMU '{dmu_column}' no existe en el DataFrame.")

    return _run_dea_internal(
        df=df,
        inputs=input_cols,
        outputs=output_cols,
        model="CCR",
        orientation=orientation,
        super_eff=super_eff
    )


def run_bcc(
    df: pd.DataFrame,
    dmu_column: str,
    input_cols: list[str],
    output_cols: list[str],
    orientation: str = "input",
    super_eff: bool = False,
) -> pd.DataFrame:
    """
    Ejecuta DEA BCC radial (input/output, opcional super-eficiencia).
    Parámetros similares a run_ccr, pero model="BCC".
    """
    if dmu_column not in df.columns:
        raise ValueError(f"La columna DMU '{dmu_column}' no existe en el DataFrame.")

    return _run_dea_internal(
        df=df,
        inputs=input_cols,
        outputs=output_cols,
        model="BCC",
        orientation=orientation,
        super_eff=super_eff
    )
