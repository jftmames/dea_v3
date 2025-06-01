# src/dea_models/utils.py

import pandas as pd

def validate_dataframe(
    df: pd.DataFrame,
    input_cols: list[str],
    output_cols: list[str],
    allow_zero: bool = False,
    allow_negative: bool = False
):
    """
    Valida que el DataFrame tenga:
      - Columnas input_cols y output_cols existentes y numéricas.
      - Si allow_zero=False, no permite ceros en inputs/outputs.
      - Si allow_negative=False, no permite negativos.
    Lanza ValueError si falla, retorna True si ok.
    """
    cols = input_cols + output_cols
    faltantes = set(cols) - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas: {faltantes}")

    # Verificar numérico y condiciones de cero/negativos
    for col in cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Columna '{col}' no es numérica.")

        if not allow_zero and (df[col] == 0).any():
            cnt = int((df[col] == 0).sum())
            raise ValueError(f"Columna '{col}' tiene {cnt} ceros; no permitidos.")

        if not allow_negative and (df[col] < 0).any():
            cnt = int((df[col] < 0).sum())
            raise ValueError(f"Columna '{col}' tiene {cnt} valores negativos; no permitidos.")

    return True


def check_positive_data(df: pd.DataFrame, columns: list[str]):
    """
    Similar a validate_dataframe(..., allow_zero=False, allow_negative=False).
    """
    return validate_dataframe(df, columns, columns, allow_zero=False, allow_negative=False)


def check_zero_negative_data(df: pd.DataFrame, columns: list[str]):
    """
    Similar a validate_dataframe(..., allow_zero=True, allow_negative=False).
    """
    return validate_dataframe(df, columns, columns, allow_zero=True, allow_negative=False)

