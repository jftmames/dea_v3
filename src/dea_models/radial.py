# jftmames/-dea-deliberativo-mvp/-dea-deliberativo-mvp-b44b8238c978ae0314af30717b9399634d28f8f9/src/dea_models/radial.py
import numpy as np
import cvxpy as cp
import pandas as pd

from .utils import validate_positive_dataframe, validate_dataframe

# ------------------------------------------------------------------
# 1. Núcleo DEA (CCR / BCC) — input/output orientation
# (Esta función es un buen concepto pero no es utilizada por las funciones públicas)
# ------------------------------------------------------------------
def _dea_core(
    X: np.ndarray,
    Y: np.ndarray,
    rts: str = "CRS",
    orientation: str = "input",
    super_eff: bool = False,
) -> np.ndarray:
    """
    X: np.array shape (m, n)  (inputs)
    Y: np.array shape (s, n)  (outputs)
    rts: "CRS" (CCR) o "VRS" (BCC)
    orientation: "input" o "output"
    super_eff: si True, excluye la DMU actual para super-eficiencia
    Devuelve vector de eficiencias (length n).
    """
    m, n_total_dmus = X.shape 
    s = Y.shape[0]
    eff = np.zeros(n_total_dmus)

    for i in range(n_total_dmus):
        x_i = X[:, [i]]
        y_i = Y[:, [i]]

        if super_eff:
            mask = np.ones(n_total_dmus, dtype=bool)
            mask[i] = False
            X_ref, Y_ref = X[:, mask], Y[:, mask]
            num_ref_dmus = n_total_dmus - 1
            if num_ref_dmus == 0:
                eff[i] = 1.0
                continue
        else:
            X_ref, Y_ref = X, Y
            num_ref_dmus = n_total_dmus

        lambdas = cp.Variable((num_ref_dmus, 1), nonneg=True)

        if orientation == "input":
            theta = cp.Variable()
            cons = [Y_ref @ lambdas >= y_i, X_ref @ lambdas <= theta * x_i]
            if rts == "VRS":
                cons.append(cp.sum(lambdas) == 1)
            obj = cp.Minimize(theta)
        else:
            phi_var = cp.Variable()
            cons = [Y_ref @ lambdas >= phi_var * y_i, X_ref @ lambdas <= x_i]
            if rts == "VRS":
                cons.append(cp.sum(lambdas) == 1)
            obj = cp.Maximize(phi_var)

        prob = cp.Problem(obj, cons)
        try:
            prob.solve(solver=cp.ECOS, abstol=1e-7, reltol=1e-7, feastol=1e-7, verbose=False)
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                eff[i] = float(obj.value) if obj.value is not None else np.nan
            else:
                eff[i] = np.nan
        except (cp.error.SolverError, Exception):
            eff[i] = np.nan
    return eff

# ------------------------------------------------------------------
# 2. Función pública: run_ccr
# ------------------------------------------------------------------
def run_ccr(
    df: pd.DataFrame,
    dmu_column: str,
    input_cols: list[str],
    output_cols: list[str],
    orientation: str = "input", 
    super_eff: bool = False
) -> pd.DataFrame:
    """
    Ejecuta CCR radial.
    """
    if dmu_column not in df.columns:
        raise ValueError(f"La columna DMU '{dmu_column}' no existe en el DataFrame.")
    validate_positive_dataframe(df.copy(), input_cols + output_cols)

    X = df[input_cols].to_numpy().T
    Y = df[output_cols].to_numpy().T
    dmus = df[dmu_column].astype(str).tolist()
    n, m, s = X.shape[1], X.shape[0], Y.shape[0]
    
    resultados = []
    for i in range(n):
        x0, y0 = X[:, [i]], Y[:, [i]]

        ref_indices, X_ref, Y_ref, dmus_ref = (
            (list(range(n)), X, Y, dmus) 
            if not super_eff or n == 1 
            else ([j for j, v in enumerate(np.arange(n) != i) if v], X[:, np.arange(n) != i], Y[:, np.arange(n) != i], [d for j, d in enumerate(dmus) if j != i])
        )
        
        lambdas_var = cp.Variable((len(ref_indices), 1), nonneg=True)

        if orientation == "input":
            theta = cp.Variable()
            obj = cp.Minimize(theta)
            cons = [X_ref @ lambdas_var <= theta * x0, Y_ref @ lambdas_var >= y0]
        else: # output
            phi = cp.Variable()
            obj = cp.Maximize(phi)
            cons = [X_ref @ lambdas_var <= x0, Y_ref @ lambdas_var >= phi * y0]
            
        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.ECOS, abstol=1e-7, reltol=1e-7, feastol=1e-7, verbose=False)

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] or obj.value is None:
             resultados.append({
                dmu_column: dmus[i], "tec_efficiency_ccr": np.nan, "lambda_vector": {},
                "slacks_inputs": {col: np.nan for col in input_cols},
                "slacks_outputs": {col: np.nan for col in output_cols}, "rts_label": "CRS"
            })
             continue

        eff_val = float(obj.value)
        lambdas_opt = lambdas_var.value if lambdas_var.value is not None else np.zeros((len(ref_indices), 1))

        # Cálculo de slacks post-eficiencia
        if orientation == "input":
            slacks_in_vals = (eff_val * x0) - (X_ref @ lambdas_opt)
            slacks_out_vals = (Y_ref @ lambdas_opt) - y0
        else: # output
            slacks_in_vals = x0 - (X_ref @ lambdas_opt)
            slacks_out_vals = (Y_ref @ lambdas_opt) - (eff_val * y0)
        
        slacks_in_vals[slacks_in_vals < 1e-9] = 0
        slacks_out_vals[slacks_out_vals < 1e-9] = 0

        resultados.append({
            dmu_column: dmus[i],
            "tec_efficiency_ccr": np.round(1/eff_val if orientation=='output' else eff_val, 6),
            "lambda_vector": {dmus_ref[j]: float(v) for j, v in enumerate(lambdas_opt.flatten())},
            "slacks_inputs": {input_cols[k]: float(v) for k, v in enumerate(slacks_in_vals.flatten())},
            "slacks_outputs": {output_cols[r]: float(v) for r, v in enumerate(slacks_out_vals.flatten())},
            "rts_label": "CRS"
        })
    return pd.DataFrame(resultados)


# ------------------------------------------------------------------
# 3. Función pública: run_bcc
# ------------------------------------------------------------------
def run_bcc(
    df: pd.DataFrame,
    dmu_column: str,
    input_cols: list[str],
    output_cols: list[str],
    df_ccr_results: pd.DataFrame,
    orientation: str = "input",
    super_eff: bool = False,
) -> pd.DataFrame:
    """
    Ejecuta modelo BCC (VRS) de forma robusta.
    """
    if dmu_column not in df.columns:
        raise ValueError(f"La columna DMU '{dmu_column}' no existe en el DataFrame.")
    validate_positive_dataframe(df.copy(), input_cols + output_cols)

    X = df[input_cols].to_numpy().T
    Y = df[output_cols].to_numpy().T
    dmus = df[dmu_column].astype(str).tolist()
    n, m, s = X.shape[1], X.shape[0], Y.shape[0]
    
    registros = []
    for i in range(n):
        x0, y0 = X[:, [i]], Y[:, [i]]

        ref_indices, X_ref, Y_ref, dmus_ref = (
            (list(range(n)), X, Y, dmus) 
            if not super_eff or n == 1
            else ([j for j, v in enumerate(np.arange(n) != i) if v], X[:, np.arange(n) != i], Y[:, np.arange(n) != i], [d for j, d in enumerate(dmus) if j != i])
        )
        
        lambdas_var = cp.Variable((len(ref_indices), 1), nonneg=True)
        convexity_constraint = cp.sum(lambdas_var) == 1
        
        if orientation == "input":
            theta = cp.Variable()
            obj = cp.Minimize(theta)
            cons = [X_ref @ lambdas_var <= theta * x0, Y_ref @ lambdas_var >= y0, convexity_constraint]
        else: # output
            phi = cp.Variable()
            obj = cp.Maximize(phi)
            cons = [X_ref @ lambdas_var <= x0, Y_ref @ lambdas_var >= phi * y0, convexity_constraint]

        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.ECOS, abstol=1e-7, reltol=1e-7, feastol=1e-7, verbose=False)

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] or obj.value is None:
            registros.append({
                dmu_column: dmus[i], "efficiency": np.nan, "model": "BCC", "orientation": orientation, 
                "super_eff": bool(super_eff), "lambda_vector": {}, "slacks_inputs": {col: np.nan for col in input_cols},
                "slacks_outputs": {col: np.nan for col in output_cols}, "scale_efficiency": np.nan, "rts_label": "Error"
            })
            continue

        eff_val = float(obj.value)
        bcc_eff = 1/eff_val if orientation == 'output' else eff_val
        lambdas_opt = lambdas_var.value if lambdas_var.value is not None else np.zeros((len(ref_indices), 1))
        
        ccr_eff_series = df_ccr_results.loc[df_ccr_results[dmu_column] == dmus[i], "tec_efficiency_ccr"]
        ccr_eff = ccr_eff_series.iloc[0] if not ccr_eff_series.empty else np.nan
        scale_eff = (ccr_eff / bcc_eff) if not np.isnan(bcc_eff) and not np.isnan(ccr_eff) and bcc_eff != 0 else np.nan

        # Cálculo de slacks
        if orientation == "input":
            slacks_in_vals = (eff_val * x0) - (X_ref @ lambdas_opt)
            slacks_out_vals = (Y_ref @ lambdas_opt) - y0
        else:
            slacks_in_vals = x0 - (X_ref @ lambdas_opt)
            slacks_out_vals = (Y_ref @ lambdas_opt) - (eff_val * y0)
        
        slacks_in_vals[slacks_in_vals < 1e-9] = 0
        slacks_out_vals[slacks_out_vals < 1e-9] = 0

        # Identificación de retornos a escala
        rts_label = "VRS"
        dual_val = convexity_constraint.dual_value
        if dual_val is not None:
            if abs(dual_val) < 1e-6: rts_label = "CRS"
            elif dual_val < 0: rts_label = "IRS"
            else: rts_label = "DRS"
        
        registros.append({
            dmu_column: dmus[i],
            "efficiency": np.round(bcc_eff, 6),
            "model": "BCC", "orientation": orientation, "super_eff": bool(super_eff),
            "lambda_vector": {dmus_ref[j]: float(v) for j, v in enumerate(lambdas_opt.flatten())},
            "slacks_inputs": {input_cols[k]: float(v) for k, v in enumerate(slacks_in_vals.flatten())},
            "slacks_outputs": {output_cols[r]: float(v) for r, v in enumerate(slacks_out_vals.flatten())},
            "scale_efficiency": np.round(scale_eff, 6) if not np.isnan(scale_eff) else np.nan,
            "rts_label": rts_label
        })
    return pd.DataFrame(registros)
