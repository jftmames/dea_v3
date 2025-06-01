# src/dea_models/nonradial.py

import numpy as np
import pandas as pd
import cvxpy as cp

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
    # 1) Validación explicitando que no hay ceros/negativos (solo positivos)
    cols = input_cols + output_cols
    validate_positive_dataframe(df, cols)

    # 2) Construir matrices (igual que en radial, pero en formato m×n y s×n)
    #    Para SBM, conviene X y Y como arrays (inputs) de forma (m, n), (s, n)
    X = df[input_cols].to_numpy().T   # shape m×n
    Y = df[output_cols].to_numpy().T  # shape s×n
    dmus = df[dmu_column].astype(str).tolist()
    n = X.shape[1]
    m = X.shape[0]
    s = Y.shape[0]

    resultados = []
    for i in range(n):
        # PREPARAR subconjunto para “super-eff” (si se quisiera)
        # pero por definición de SBM, típicamente no usamos super-eff aquí.
        # Más adelante podría añadirse lógica similar a run_ccr para super_eff.

        # Variables lambdas (n variables)
        lambdas = cp.Variable((n, 1), nonneg=True)
        # Variables slack de inputs y outputs
        s_minus = cp.Variable((m, 1), nonneg=True)
        s_plus = cp.Variable((s, 1), nonneg=True)

        # Escape rápido si orientación no es válida
        if orientation not in ("input", "output"):
            raise ValueError("orientation debe ser 'input' u 'output'")

        # Construir restricciones y objetivo según SBM (Allen et al., 1997)
        if orientation == "input":
            # Fraccional: min (1/m Σ_i (x_{i0} - s_i^-) / x_{i0}) sujeto a:
            #    Y λ + s_plus = y0
            #    X λ - s_minus = x0
            #    Σ λ = 1  (si rts == "VRS")
            x0 = X[:, [i]]    # (m×1)
            y0 = Y[:, [i]]    # (s×1)

            cons = []
            #  Y λ + s_plus = y0
            cons.append(Y @ lambdas + s_plus == y0)
            #  X λ - s_minus = x0
            cons.append(X @ lambdas - s_minus == x0)
            #  Σ λ = 1   (solo si VRS)
            if rts == "VRS":
                cons.append(cp.sum(lambdas) == 1)

            # Objetivo fr
