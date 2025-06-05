import numpy as np
import cvxpy as cp
import pandas as pd

from .utils import validate_positive_dataframe, validate_dataframe

# ------------------------------------------------------------------
# 1. Núcleo DEA (CCR / BCC) — input/output orientation
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
    m, n_total_dmus = X.shape # n_total_dmus is the original number of DMUs
    s = Y.shape[0]
    eff = np.zeros(n_total_dmus)

    for i in range(n_total_dmus):
        # Target DMU's inputs and outputs
        x_i = X[:, [i]]
        y_i = Y[:, [i]]

        if super_eff:
            # Create mask to exclude DMU i
            mask = np.ones(n_total_dmus, dtype=bool)
            mask[i] = False
            X_ref = X[:, mask]
            Y_ref = Y[:, mask]
            num_ref_dmus = n_total_dmus - 1
            if num_ref_dmus == 0: # Cannot run super-efficiency with 1 DMU
                eff[i] = 1.0 if orientation == "input" else 1.0 # Or np.nan, depending on desired behavior
                continue
        else:
            X_ref = X
            Y_ref = Y
            num_ref_dmus = n_total_dmus

        lambdas = cp.Variable((num_ref_dmus, 1), nonneg=True)

        if orientation == "input":
            theta = cp.Variable()
            cons = [
                Y_ref @ lambdas >= y_i,
                X_ref @ lambdas <= theta * x_i,
            ]
            if rts == "VRS":
                cons.append(cp.sum(lambdas) == 1)
            obj = cp.Minimize(theta)
        else:  # output-oriented
            phi_var = cp.Variable() # Using phi_var to avoid conflict if phi is used later
            cons = [
                Y_ref @ lambdas >= phi_var * y_i,
                X_ref @ lambdas <= x_i,
            ]
            if rts == "VRS":
                cons.append(cp.sum(lambdas) == 1)
            obj = cp.Maximize(phi_var)

        prob = cp.Problem(obj, cons)
        try:
            prob.solve(
                solver=cp.ECOS,
                abstol=1e-7, # Slightly adjusted for common practice
                reltol=1e-7,
                feastol=1e-7,
                verbose=False
            )

            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                if orientation == "input":
                    eff[i] = float(theta.value) if theta.value is not None else np.nan
                else:
                    eff[i] = float(phi_var.value) if phi_var.value is not None else np.nan
                    if eff[i] != np.nan and eff[i] != 0: 
                           pass 
            else:
                eff[i] = np.nan # Problem not solved optimally
        except cp.error.SolverError:
            eff[i] = np.nan # Solver failed
        except Exception:
            eff[i] = np.nan


    return eff


# ------------------------------------------------------------------
# 2. Función interna que antes se llamaba run_dea
# ------------------------------------------------------------------
def _run_dea_internal(
    df: pd.DataFrame,
    inputs: list[str],
    outputs: list[str],
    model: str = "CCR",
    orientation: str = "input",
    super_eff: bool = False,
    dmu_col_name: str = "DMU" 
) -> pd.DataFrame:
    """
    Ejecuta DEA (CCR/BCC, input/output, opcional super-eficiencia).
    Parámetros:
      df: DataFrame original (incluye identificador index o columna "DMU")
      inputs: lista de nombres de columnas de inputs (numéricos y > 0)
      outputs: lista de nombres de columnas de outputs (numéricos y > 0)
      model: "CCR" o "BCC"
      orientation: "input" o "output"
      super_eff: True para super-eficiencia (excluye DMU actual)
      dmu_col_name: Name of the DMU identifier column.
    Retorna DataFrame con columnas:
      DMU, efficiency, model, orientation, super_eff
    """
    # validaciones iniciales
    assert orientation in ("input", "output"), "orientation debe ser 'input' u 'output'"
    assert model.upper() in ("CCR", "BCC"), "model debe ser 'CCR' o 'BCC'"

    # 1) Extraer sólo columnas requeridas y convertir a float
    cols_to_validate = inputs + outputs
    df_num = validate_positive_dataframe(df.copy(), cols_to_validate) # Use a copy

    # 2) Construir matrices X y Y (shape: m×n, s×n)
    X_data = df_num[inputs].to_numpy().T
    Y_data = df_num[outputs].to_numpy().T

    # 3) Definir returns-to-scale
    rts_model = "CRS" if model.upper() == "CCR" else "VRS"

    # 4) Llamar al núcleo
    eff_scores = _dea_core(X_data, Y_data, rts=rts_model, orientation=orientation, super_eff=super_eff)

    # 5) Preparar el DataFrame de salida
    if dmu_col_name in df.columns:
        dmu_ids = df[dmu_col_name].astype(str)
    else:
        dmu_ids = df.index.astype(str)
        if dmu_ids.name is None: 
            dmu_ids.name = "DMU_Index"

    return pd.DataFrame({
        dmu_col_name: dmu_ids,
        "efficiency": np.round(eff_scores, 6), 
        "model": model.upper(),
        "orientation": orientation,
        "super_eff": bool(super_eff),
    })


# ------------------------------------------------------------------
# 3. Función pública: run_ccr
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
    Ejecuta CCR radial y retorna DataFrame con columnas:
      DMU, tec_efficiency_ccr, lambda_vector, slacks_inputs, slacks_outputs, rts_label
    """
    if dmu_column not in df.columns:
        raise ValueError(f"La columna DMU '{dmu_column}' no existe en el DataFrame.")

    # Validar que todos los inputs y outputs sean positivos
    cols_to_validate = input_cols + output_cols
    validate_positive_dataframe(df.copy(), cols_to_validate)

    # Matrices de datos
    X = df[input_cols].to_numpy().T  # shape: (m, n)
    Y = df[output_cols].to_numpy().T  # shape: (s, n)
    dmus = df[dmu_column].astype(str).tolist()
    n = X.shape[1]  # número de DMUs
    m = X.shape[0]  # número de inputs
    s = Y.shape[0]  # número de outputs

    resultados = []
    for i in range(n):
        x0 = X[:, [i]]
        y0 = Y[:, [i]]

        if super_eff:
            if n == 1: 
                resultados.append({
                    dmu_column: dmus[i],
                    "tec_efficiency_ccr": 1.0, 
                    "lambda_vector": {dmus[i]: 1.0},
                    "slacks_inputs": {col: 0.0 for col in input_cols},
                    "slacks_outputs": {col: 0.0 for col in output_cols},
                    "rts_label": "CRS"
                })
                continue
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            X_ref = X[:, mask]
            Y_ref = Y[:, mask]
            lambdas_var = cp.Variable((n - 1, 1), nonneg=True)
            ref_dmus_indices = [idx for idx, val in enumerate(mask) if val]
        else:
            X_ref = X
            Y_ref = Y
            lambdas_var = cp.Variable((n, 1), nonneg=True)
            ref_dmus_indices = list(range(n))

        if orientation == "input":
            theta_rad = cp.Variable()
            cons_stage1 = [X_ref @ lambdas_var <= theta_rad * x0,
                           Y_ref @ lambdas_var >= y0]
            obj_stage1 = cp.Minimize(theta_rad)
        else: # Output-oriented CCR
            phi_rad = cp.Variable()
            cons_stage1 = [X_ref @ lambdas_var <= x0,
                           Y_ref @ lambdas_var >= phi_rad * y0]
            obj_stage1 = cp.Maximize(phi_rad)

        prob_stage1 = cp.Problem(obj_stage1, cons_stage1)
        prob_stage1.solve(solver=cp.ECOS, abstol=1e-7, reltol=1e-7, feastol=1e-7, verbose=False)

        if prob_stage1.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            resultados.append({
                dmu_column: dmus[i], "tec_efficiency_ccr": np.nan, "lambda_vector": {},
                "slacks_inputs": {col: np.nan for col in input_cols},
                "slacks_outputs": {col: np.nan for col in output_cols}, "rts_label": "CRS"
            })
            continue
            
        eff_val_rad = theta_rad.value if orientation == "input" else phi_rad.value
        if eff_val_rad is None: eff_val_rad = np.nan

        tec_eff = eff_val_rad
        lambdas_opt = lambdas_var.value
        if lambdas_opt is None: 
            lambdas_opt = np.zeros((lambdas_var.shape[0], 1))

        if orientation == "input":
            slacks_input_values = (tec_eff * x0) - (X_ref @ lambdas_opt)
            slacks_output_values = (Y_ref @ lambdas_opt) - y0
        else: # output
            slacks_input_values = x0 - (X_ref @ lambdas_opt)
            slacks_output_values = (eff_val_rad * y0) - (Y_ref @ lambdas_opt)
            
        slacks_input_values[slacks_input_values < 0] = 0
        slacks_output_values[slacks_output_values < 0] = 0

        lambdas_vals = {}
        if lambdas_opt is not None:
            for idx, ref_idx in enumerate(ref_dmus_indices):
                lambdas_vals[dmus[ref_idx]] = float(lambdas_opt[idx, 0])
            
        slacks_input_dict = {input_cols[k]: float(slacks_input_values[k,0]) for k in range(m)}
        slacks_output_dict = {output_cols[r]: float(slacks_output_values[r,0]) for r in range(s)}

        resultados.append({
            dmu_column: dmus[i],
            "tec_efficiency_ccr": np.round(tec_eff, 6) if tec_eff is not None else np.nan,
            "lambda_vector": lambdas_vals,
            "slacks_inputs": slacks_input_dict,
            "slacks_outputs": slacks_output_dict,
            "rts_label": "CRS"  # CCR siempre CRS
        })

    return pd.DataFrame(resultados)


# ------------------------------------------------------------------
# 4. Función pública: run_bcc (VERSIÓN OPTIMIZADA Y ROBUSTA)
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
    Ejecuta modelo BCC (VRS). Esta versión es robusta y maneja errores
    del optimizador para DMUs individuales sin detener la ejecución.
    """
    if dmu_column not in df.columns:
        raise ValueError(f"La columna DMU '{dmu_column}' no existe en el DataFrame.")

    from .utils import validate_dataframe
    validate_dataframe(df, input_cols, output_cols, allow_zero=False, allow_negative=False)

    df_num = df[[dmu_column] + input_cols + output_cols].copy()
    X = df_num[input_cols].to_numpy().T
    Y = df_num[output_cols].to_numpy().T
    dmus = df_num[dmu_column].astype(str).tolist()
    m, n = X.shape
    s, _ = Y.shape
    
    registros = [None] * n

    for i in range(n):
        try:
            if super_eff:
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                X_ref, Y_ref = X[:, mask], Y[:, mask]
                dmus_ref = [dmus[j] for j in range(n) if j != i]
                num_vars = n - 1
            else:
                X_ref, Y_ref = X, Y
                dmus_ref = dmus
                num_vars = n

            if num_vars == 0:
                raise ValueError("No reference DMUs to compare against.")

            lambdas_var = cp.Variable((num_vars, 1), nonneg=True)
            slacks_in = cp.Variable((m, 1), nonneg=True)
            slacks_out = cp.Variable((s, 1), nonneg=True)
            x_i, y_i = X[:, [i]], Y[:, [i]]

            convexity_constraint = cp.sum(lambdas_var) == 1
            constraints = [convexity_constraint]

            if orientation == "input":
                theta = cp.Variable()
                constraints.extend([
                    Y_ref @ lambdas_var + slacks_out >= y_i,
                    X_ref @ lambdas_var <= theta * x_i - slacks_in,
                ])
                objective = cp.Minimize(theta)
            else: # output
                phi = cp.Variable()
                constraints.extend([
                    Y_ref @ lambdas_var >= phi * y_i + slacks_out, 
                    X_ref @ lambdas_var + slacks_in <= x_i,
                ])
                objective = cp.Maximize(phi)

            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.ECOS, abstol=1e-6, reltol=1e-6, feastol=1e-8, verbose=False)

            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                raise ValueError(f"Solver status was {prob.status}, not optimal.")
            
            value = theta.value if orientation == 'input' else phi.value
            current_eff_val = float(value) if value is not None else np.nan

            slack_in_arr = slacks_in.value if slacks_in.value is not None else np.zeros((m, 1))
            slack_out_arr = slacks_out.value if slacks_out.value is not None else np.zeros((s, 1))
            slacks_input = {input_cols[k]: float(v) for k, v in enumerate(slack_in_arr)}
            slacks_output = {output_cols[r]: float(v) for r, v in enumerate(slack_out_arr)}

            ccr_eff_series = df_ccr_results.loc[df_ccr_results[dmu_column] == dmus[i], "tec_efficiency_ccr"]
            ccr_eff = ccr_eff_series.iloc[0] if not ccr_eff_series.empty else np.nan
            scale_eff = (ccr_eff / current_eff_val) if not np.isnan(current_eff_val) and not np.isnan(ccr_eff) and current_eff_val != 0 else np.nan
            
            lambdas_vals = {ref_dmu_id: float(v) for ref_dmu_id, v in zip(dmus_ref, lambdas_var.value)} if lambdas_var.value is not None else {}
            
            rts_label = "VRS"
            dual_val = convexity_constraint.dual_value
            if dual_val is not None:
                dual_val = float(dual_val)
                if abs(dual_val) < 1e-6: rts_label = "CRS"
                elif dual_val < 0: rts_label = "IRS" if orientation == "input" else "DRS"
                else: rts_label = "DRS" if orientation == "input" else "IRS"

            registros[i] = {
                dmu_column: dmus[i],
                "efficiency": np.round(current_eff_val, 6) if not np.isnan(current_eff_val) else np.nan,
                "model": "BCC", "orientation": orientation, "super_eff": bool(super_eff),
                "lambda_vector": lambdas_vals, "slacks_inputs": slacks_input, "slacks_outputs": slacks_output,
                "scale_efficiency": np.round(scale_eff, 6) if not np.isnan(scale_eff) else np.nan,
                "rts_label": rts_label
            }

        except Exception as e:
            # Si algo falla para esta DMU, crea un registro de error y continúa.
            registros[i] = {
                dmu_column: dmus[i], "efficiency": np.nan, "model": "BCC",
                "orientation": orientation, "super_eff": bool(super_eff),
                "lambda_vector": {}, "slacks_inputs": {col: np.nan for col in input_cols},
                "slacks_outputs": {col: np.nan for col in output_cols},
                "scale_efficiency": np.nan, "rts_label": f"Error: {type(e).__name__}"
            }
            continue

    return pd.DataFrame(registros)
