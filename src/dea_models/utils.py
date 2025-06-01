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
    Valida que todas las columnas de ``input_cols`` + ``output_cols`` existan y sean
    numéricas.

    - Si ``allow_zero`` es ``False``, se rechazan valores cero.
    - Si ``allow_negative`` es ``False``, se rechazan valores negativos.

    Parameters
    ----------
    df : pd.DataFrame
        El DataFrame a validar.
    input_cols : list[str]
        Lista de nombres de columnas de inputs.
    output_cols : list[str]
        Lista de nombres de columnas de outputs.
    allow_zero : bool, optional (default=False)
        Permitir valores cero.
    allow_negative : bool, optional (default=False)
        Permitir valores negativos.

    Returns
    -------
    bool
        ``True`` si el DataFrame pasa todas las validaciones.

    Raises
    ------
    ValueError
        Si alguna de las verificaciones falla.
    """
    cols = input_cols + output_cols
    faltantes = set(cols) - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas: {faltantes}")

    for col in cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Columna '{col}' no es numérica.")

        if not allow_zero and (df[col] == 0).any():
            cnt = int((df[col] == 0).sum())
            raise ValueError(
                f"Columna '{col}' tiene {cnt} ceros; no permitidos."
            )

        if not allow_negative and (df[col] < 0).any():
            cnt = int((df[col] < 0).sum())
            raise ValueError(
                f"Columna '{col}' tiene {cnt} valores negativos; no permitidos."
            )

    return True


def check_positive_data(df: pd.DataFrame, columns: list[str]):
    """Alias para validar que *no* haya ceros ni negativos en ``columns``.

    Equivalente a::

        validate_dataframe(df, columns, [], allow_zero=False, allow_negative=False)
    """
    return validate_dataframe(df, columns, [], allow_zero=False, allow_negative=False)


def check_zero_negative_data(df: pd.DataFrame, columns: list[str]):
    """Alias para validar que se permitan ceros pero *no* negativos en ``columns``.

    Equivalente a::

        validate_dataframe(df, columns, [], allow_zero=True, allow_negative=False)
    """
    return validate_dataframe(df, columns, [], allow_zero=True, allow_negative=False)


# ---------------------------------------------------------------------------
# Alias adicional solicitado
# ---------------------------------------------------------------------------

def validate_positive_dataframe(df: pd.DataFrame, columns: list[str]):
    """Alias para validar que todas las columnas en ``columns`` existan, sean numéricas
    y no contengan ceros ni valores negativos."""
    return validate_dataframe(df, columns, [], allow_zero=False, allow_negative=False)


# ---------------------------------------------------------------------------
# Otras utilidades
# ---------------------------------------------------------------------------

def format_lambda_table(lambda_dicts: list[dict[str, float]]) -> pd.DataFrame:
    """Convierte una lista de diccionarios con pesos *lambda* a una tabla.

    Cada elemento de ``lambda_dicts`` es un ``dict`` que mapea *peer* → *lambda*.
    Se asume que la lista está en el mismo orden que las DMUs.

    Devuelve
    -------
    pd.DataFrame
        Tabla con columnas ``[DMU, peer1, peer2, …]``.
    """
    if not lambda_dicts:
        return pd.DataFrame()

    # Determinar el conjunto de peers/DMUs a partir del primer elemento
    dmus = list(lambda_dicts[0].keys())

    # Construir filas asegurando orden consistente de columnas
    rows = [
        {peer: lam.get(peer, 0.0) for peer in dmus}
        for lam in lambda_dicts
    ]

    df = pd.DataFrame(rows, index=dmus)

    # Poner los nombres de las DMUs como columna explícita
    df.reset_index(inplace=True)
    df.rename(columns={"index": "DMU"}, inplace=True)

    return df
