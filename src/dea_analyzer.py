import numpy as np
import cvxpy as cp
import pandas as pd

# ------------- CORE DEA -------------
def _dea_core(X, Y, returns_to_scale="CRS"):
    """
    X: np.array shape (m, n)  inputs
    Y: np.array shape (s, n)  outputs
    returns_to_scale: "CRS" (CCR) | "VRS" (BCC)
    """
    m, n = X.shape
    s, _ = Y.shape
    e = np.ones((n, 1))
    efficiencies = np.zeros(n)

    # Loop DMU by DMU (simple, readable; can be vectorised later)
    for i in range(n):
        x_i = X[:, [i]]
        y_i = Y[:, [i]]

        lambdas = cp.Variable((n, 1), nonneg=True)
        theta = cp.Variable()

        constraints = [
            Y @ lambdas >= y_i,
            X @ lambdas <= theta * x_i
        ]
        if returns_to_scale == "VRS":
            constraints.append(e.T @ lambdas == 1)

        prob = cp.Problem(cp.Minimize(theta), constraints)
        prob.solve(solver=cp.SCS, verbose=False)
        efficiencies[i] = theta.value if theta.value else np.nan

    return efficiencies

# ------------- PUBLIC API -------------
def run_dea(
    df: pd.DataFrame,
    inputs: list[str],
    outputs: list[str],
    model: str = "CCR",
    orientation: str = "input",
) -> pd.DataFrame:
    """
    Returns a dataframe with DMU, efficiency and model metadata.
    Only input-oriented for now.
    """
    assert orientation == "input", "Solo orientación input implementada"
    X = df[inputs].to_numpy().T  # shape (m, n)
    Y = df[outputs].to_numpy().T  # shape (s, n)

    rts = "CRS" if model.upper() == "CCR" else "VRS"
    eff = _dea_core(X, Y, returns_to_scale=rts)

    return pd.DataFrame(
        {
            "DMU": df.index.astype(str),
            "efficiency": eff.round(4),
            "model": model.upper(),
            "orientation": orientation,
        }
    )
