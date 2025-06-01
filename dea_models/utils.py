# dea_models/utils.py

import pandas as pd

def validate_positive_dataframe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Convierte las columnas 'cols' de df a float.
    Si alguno de los valores no es convertible o ≤ 0, lanza ValueError.
    Retorna un DataFrame con solo esas columnas (conversión a float).
    """
    df_copy = df.copy()
    bad_cols = []

    for c in cols:
        # convertir a numérico (NaN en casos inválidos)
        df_copy[c] = pd.to_numeric(df_copy[c], errors="coerce")
        # verificar NaN o ≤ 0
        if df_copy[c].isna().any() or (df_copy[c] <= 0).any():
            bad_cols.append(c)

    if bad_cols:
        raise ValueError(
            f"Columnas con datos inválidos (no numéricos o ≤ 0): {bad_cols}"
        )

    return df_copy[cols]
