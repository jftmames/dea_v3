import numpy as np
import cvxpy as cp
import pandas as pd

# ------------------------------------------------------------------
# CORE DEA (CCR / BCC, orientación input)
# ------------------------------------------------------------------
def _dea_core(X, Y, returns_to_scale: str = "CRS"):
    """
    X: np.array shape (m, n)  — inputs
    Y: np.array shape (s, n)  — outputs
    returns_to_scale: "CRS" (CCR) | "VRS" (BCC)
    """
    m, n = X.shape
    s, _ = Y.shape
    e = np.ones((n, 1))
    efficiencies = np.zeros(n)

    for i in range(n):
        x_i = X[:, [i]]
        y_i = Y[:, [i]]

        lambdas = cp.Variable((n, 1), nonneg=True)
        theta = cp.Variable()

        constraints = [
            Y @ lambdas >= y_i,
            X @ lambdas <= theta * x_i,
        ]
        if returns_to_scale == "VRS":          # BCC
            constraints.append(e.T @ lambdas == 1)

        prob = cp.Problem(cp.Minimize(theta), constraints)
        prob.solve(solver=cp.SCS, verbose=False)
        efficiencies[i] = theta.value if theta.value else np.nan

    return efficiencies


# ------------------------------------------------------------------
# API pública
# ------------------------------------------------------------------
def run_dea(
    df: pd.DataFrame,
    inputs: list[str],
    outputs: list[str],
    model: str = "CCR",
    orientation: str = "input",
) -> pd.DataFrame:
    """
    Ejecuta un modelo DEA (CCR o BCC), orientación input.
    Devuelve DataFrame con DMU, eficiencia y metadatos.
    """
    assert orientation == "input", "Solo orientación input implementada"

    # -------- aseguramos que todo sea numérico --------
    try:
        df_numeric = df[inputs + outputs].astype(float)
    except ValueError as e:
        raise ValueError(f"Conversión a float falló: {e}")

    X = df_numeric[inputs].to_numpy().T   # shape (m, n)
    Y = df_numeric[outputs].to_numpy().T  # shape (s, n)

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
