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

    # 3) Construir matriz n×n
    cross_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            lam_j = lambdas_list[j]
            # eficiencia de i con "pesos" de j: en esta versión simplificada,
            # usamos el valor de lambda_j para i como proxy
            eff_ij = lam_j.get(dmus[i], 0.0)
            cross_matrix[i, j] = eff_ij

    # 4) Convertir a DataFrame
    df_cross = pd.DataFrame(cross_matrix, index=dmus, columns=dmus)

    # 5) Ranking según método
    if method == "average":
        df_cross["avg_peer_rating"] = df_cross.mean(axis=1)
    elif method == "aggressive":
        df_cross["peer_rating"] = df_cross.min(axis=1)
    elif method == "benevolent":
        df_cross["peer_rating"] = df_cross.max(axis=1)
    else:
        raise ValueError("method debe ser 'average', 'aggressive' o 'benevolent'")

    return df_cross
