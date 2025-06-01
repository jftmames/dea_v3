# dea_models/utils.py

import pandas as pd

from .constants import DEFAULT_TOLERANCE

def check_positive_data(df: pd.DataFrame, columns: list):
    """
    Verifica que en las columnas especificadas no haya valores <= 0.
    Lanza ValueError si encuentra ceros o negativos.
    """
    for col in columns:
        if (df[col] <= 0).any():
            negativos = df.loc[df[col] <= 0, col].count()
            raise ValueError(
                f"En '{col}' hay {negativos} valores menores o iguales a cero. "
                "Para usar modelos radiales CCR/BCC, todos los valores deben ser positivos."
            )

def check_zero_negative_data(df: pd.DataFrame, columns: list):
    """
    Verifica que en las columnas especificadas no haya valores negativos.
    Lanza ValueError si encuentra negativos, pero permite ceros.
    """
    for col in columns:
        if (df[col] < 0).any():
            negativos = df.loc[df[col] < 0, col].count()
            raise ValueError(
                f"En '{col}' hay {negativos} valores negativos. "
                "El modelo seleccionado no permite datos negativos."
            )

def validate_dataframe(
    df: pd.DataFrame,
    input_cols: list,
    output_cols: list,
    allow_zero: bool = False,
    allow_negative: bool = False
):
    """
    Valida que el DataFrame tenga las columnas de inputs y outputs,
    que sean numéricas, y chequea condiciones de ceros/negativos según flags.

    Parámetros:
    - df: DataFrame con todos los datos.
    - input_cols: lista de nombres de columnas consideradas inputs.
    - output_cols: lista de nombres de columnas consideradas outputs.
    - allow_zero: si False, NO se permiten ceros en inputs/outputs.
    - allow_negative: si False, NO se permiten valores negativos.

    Lanza ValueError si la validación falla.
    """
    # 1. Verificar existencia de columnas
    faltantes = set(input_cols + output_cols) - set(df.columns)
    if faltantes:
        raise ValueError(f"Las columnas {faltantes} no existen en el DataFrame.")

    # 2. Verificar que sean numéricas
    no_numericas = [
        col for col in input_cols + output_cols
        if not pd.api.types.is_numeric_dtype(df[col])
    ]
    if no_numericas:
        raise ValueError(f"Las columnas {no_numericas} no son numéricas.")

    # 3. Si no se permiten ceros, castear ceros como inválidos
    if not allow_zero:
        for col in input_cols + output_cols:
            if (df[col] == 0).any():
                conteo = df.loc[df[col] == 0, col].count()
                raise ValueError(
                    f"'{col}' contiene {conteo} ceros. "
                    "Para usar este modelo, no se admiten valores cero."
                )

    # 4. Si no se permiten negativos, castear negativos como inválidos
    if not allow_negative:
        for col in input_cols + output_cols:
            if (df[col] < 0).any():
                conteo = df.loc[df[col] < 0, col].count()
                raise ValueError(
                    f"'{col}' contiene {conteo} valores negativos. "
                    "Para usar este modelo, no se admiten valores negativos."
                )

    # Si todo pasa, devolvemos True (o simplemente salimos sin error)
    return True
