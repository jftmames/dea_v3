# src/dea_models/cross_efficiency.py

import numpy as np
import pandas as pd

from .radial import _run_dea_internal  # reutilizamos el núcleo CCR/BCC
from .utils import validate_positive_dataframe

def compute_cross_efficiency(
    df: pd.DataFrame,
    dmu_column: str,
    input_cols: list[str],
    output_cols: list[str],
    rts: str = "CRS",
    method: str = "average"
) -> pd.DataFrame:
    """
    Calcula la matriz de cross-efficiencies:
      1. Resolver CCR o BCC para cada DMU => obtener pesos óptimos (u_j, v_j) implícitos.
      2. Para cada par (i, j), calcular eficiencia de i usando pesos de j.
    method:
      - "average": eficiencia_i = promedio sobre j
      - "aggressive": eficiencia_i = min_j θ_{i|j}
      - "benevolent": eficiencia_i = max_j θ_{i|j}
    Retorna DataFrame n×n con columnas y filas = DMUs, y una fila adicional con ranking.
    Retorna DataFrame n×n con columnas y filas = DMUs, y una columna adicional con ranking.
    """
    # Validar que la columna DMU exista
    if dmu_column not in df.columns:
        raise ValueError(f"La columna DMU '{dmu_column}' no existe en el DataFrame.")

    # 1) Validar positividades
    cols = input_cols + output_cols
    validate_positive_dataframe(df, cols)

    dmus = df[dmu_column].astype(str).tolist()
    n = len(dmus)

    # 2) Resolver modelos individuales para obtener lambdas
    lambdas_list = []  # lista de dicts: un dict por j con pesos óptimos
    df_ccr_full = _run_dea_internal(
        df=df,
        inputs=input_cols,
        outputs=output_cols,
        model="CCR",
        orientation="input",
        super_eff=False
    )
    # df_ccr_full contiene una fila por DMU, con 'lambda_vector' dict
    for j in range(n):
        lambdas_list.append(df_ccr_full.loc[j, "lambda_vector"])
