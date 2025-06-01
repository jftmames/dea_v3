import numpy as np
import cvxpy as cp
import pandas as pd


# ------------------------------------------------------------------
def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Intenta convertir cada columna a float.
    Devuelve df_numérico o lanza ValueError con lista de columnas problemáticas.
    """
    bad_cols = []
    converted = {}
    for c in cols:
        converted[c] = pd.to_numeric(df[c], errors="coerce")
        if converted[c].isna().any():
            bad_cols.append(c)

    if bad_cols:
        raise ValueError(
            f"Estas columnas tienen valores no numéricos o vacíos: {bad_cols}. "
            "Límpialas o elimínalas de la selección."
        )
    return pd.DataFrame(converted)


# ------------------------------------------------------------------
def _dea_core(X, Y, returns_to_scale="CRS"):
    """CCR/BCC input-oriented"""
    m, n = X.shape
    e = np.ones((n, 1))
    eff = np.zeros(n)

    for i in range(n):
        x_i = X[:, [i]]
        y_i = Y[:, [i]]
        lambdas = cp.Variable((n, 1), nonneg=True)
        theta = cp.Variable()

        cons = [Y @ lambdas >= y_i, X @ lambdas <= theta * x_i]
        if returns_to_scale == "VRS":
            cons.append(e.T @ lambdas == 1)

        cp.Problem(cp.Minimize(theta), cons).solve(solver=cp.SCS, verbose=False)
        eff[i] = theta.value if theta.value else np.nan
    return eff


# ------------------------------------------------------------------
def run_dea(
    df: pd.DataFrame,
    inputs: list[str],
    outputs: list[str],
    model: str = "CCR",
    orientation: str = "input",
) -> pd.DataFrame:
    assert orientation == "input", "Solo orientación input implementada"

    df_num = _safe_numeric(df, inputs + outputs)  # <- validación fuerte
    X = df_num[inputs].to_numpy().T
    Y = df_num[outputs].to_numpy().T

    rts = "CRS" if model.upper() == "CCR" else "VRS"
    eff = _dea_core(X, Y, returns_to_scale=rts)

    return pd.DataFrame(
        {
            "DMU": df.index.astype(str),
            "efficiency": np.round(eff, 4),
            "model": model.upper(),
            "orientation": orientation,
        }
    )
