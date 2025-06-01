import numpy as np
import cvxpy as cp
import pandas as pd


# ------------------------------------------------------------------
def run_dea_with_progress(df, inputs, outputs, progress):
    res_df = run_dea(...)  # llama la función actual
    progress.progress(1.0, text="Completado")
    return res_df

def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convierte a float y, si alguna celda es NaN, lanza ValueError (no suprime columnas)."""
    bad_cols = []
    df_copy = df.copy()

    for c in cols:
        df_copy[c] = pd.to_numeric(df_copy[c], errors="coerce")
        if df_copy[c].isna().any():
            bad_cols.append(c)

    if bad_cols:
        raise ValueError(
            f"Estas columnas contienen valores no numéricos o vacíos: {bad_cols}"
        )
    return df_copy[cols]  # garantizado: todas las columnas presentes y numéricas



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

        cp.Problem(cp.Minimize(theta), cons).solve(
            solver=cp.ECOS, max_iters=1_000, abstol=1e-6, reltol=1e-6, feastol=1e-8
        )

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
