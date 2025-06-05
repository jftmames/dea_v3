# src/dea_models/radial.py
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
# 4. Función pública: run_bcc (actualizada)
# ------------------------------------------------------------------
def run_bcc(
    df: pd.DataFrame,
    dmu_column: str,
    input_cols: list[str],
    output_cols: list[str],
    orientation: str = "input",
    super_eff: bool = False,
) -> pd.DataFrame:
    """
    Ejecuta modelo BCC (VRS) con orientación 'input' u 'output'.
    Devuelve DataFrame con columnas:
      DMU, efficiency, model, orientation, super_eff, slacks_inputs, slacks_outputs
    """
    # 1) Validar que exista la columna DMU
    if dmu_column not in df.columns:
        raise ValueError(f"La columna DMU '{dmu_column}' no existe en el DataFrame.")

    # 2) Validar que los inputs/outputs sean numéricos y > 0
    from .utils import validate_dataframe
    validate_dataframe(df, input_cols, output_cols, allow_zero=False, allow_negative=False)

    # 3) Preparar X (inputs) y Y (outputs)
    df_num = df[[dmu_column] + input_cols + output_cols].copy()
    X = df_num[input_cols].to_numpy().T    # tamaño m×n
    Y = df_num[output_cols].to_numpy().T  # tamaño s×n
    dmus = df_num[dmu_column].astype(str).tolist()
    m, n = X.shape    # m = #inputs, n = #DMUs
    s, _ = Y.shape    # s = #outputs

    # 4) Vector para almacenar eficiencias; lista para registros
    eff = np.zeros(n)
    registros: list[dict] = [None] * n

    # 5) Iterar sobre cada DMU i
    for i in range(n):
        # 5.1) Construir referencial X_ref, Y_ref excluyendo DMU i si super_eff=True
        if super_eff:
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            X_ref = X[:, mask]    # inputs de todas menos la i‐ésima
            Y_ref = Y[:, mask]    # outputs de todas menos la i‐ésima
            dmus_ref = [dmus[j] for j in range(n) if j != i]
            num_vars = n - 1
        else:
            X_ref = X
            Y_ref = Y
            dmus_ref = dmus
            num_vars = n

        # Manejar el caso de una única DMU con super_eff=True, o si no quedan DMUs de referencia
        if num_vars == 0:
            registros[i] = {
                dmu_column: dmus[i],
                "efficiency": 1.0 if orientation == "input" else 1.0, 
                "model": "BCC",
                "orientation": orientation,
                "super_eff": bool(super_eff),
                "lambda_vector": {dmus[i]: 1.0}, 
                "slacks_inputs": {col: 0.0 for col in input_cols},
                "slacks_outputs": {col: 0.0 for col in output_cols},
                "scale_efficiency": 1.0, 
                "rts_label": "VRS" 
            }
            continue 

        # 5.2) Definir variables de decisión comunes
        lambdas_var = cp.Variable((num_vars, 1), nonneg=True)
        slacks_in = cp.Variable((m, 1), nonneg=True)
        slacks_out = cp.Variable((s, 1), nonneg=True)

        theta = None
        phi = None

        x_i = X[:, [i]]    
        y_i = Y[:, [i]]    

        # 5.3) Construir restricciones y objetivo según orientación
        constraints: list = []
        if orientation == "input":
            theta = cp.Variable()
            constraints += [
                Y_ref @ lambdas_var + slacks_out >= y_i,
                X_ref @ lambdas_var <= theta * x_i - slacks_in, 
                cp.sum(lambdas_var) == 1
            ]
            objective = cp.Minimize(theta)

        else:
            phi = cp.Variable()
            constraints += [
                Y_ref @ lambdas_var >= phi * y_i + slacks_out, 
                X_ref @ lambdas_var + slacks_in <= x_i,
                cp.sum(lambdas_var) == 1
            ]
            objective = cp.Maximize(phi)

        # 5.4) Resolver el problema
        prob = cp.Problem(objective, constraints)
        prob.solve(
            solver=cp.ECOS,
            abstol=1e-6,
            reltol=1e-6,
            feastol=1e-8,
            verbose=False
        )

        # 6) Extraer eficiencia (θ o φ), comprobando si .value es None
        current_eff_val = np.nan
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            if orientation == "input":
                theta_val = theta.value
                current_eff_val = float(theta_val) if theta_val is not None else np.nan
            else:
                phi_val = phi.value
                current_eff_val = float(phi_val) if phi_val is not None else np.nan

        eff[i] = current_eff_val 

        # 7) Extraer slacks; si .value es None, usar ceros
        if slacks_in.value is None:
            slack_in_arr = np.zeros((m, 1))
        else:
            slack_in_arr = slacks_in.value
            slack_in_arr[slack_in_arr < 1e-9] = 0 

        if slacks_out.value is None:
            slack_out_arr = np.zeros((s, 1))
        else:
            slack_out_arr = slacks_out.value
            slack_out_arr[slack_out_arr < 1e-9] = 0 

        slacks_input = { input_cols[k]: float(slack_in_arr[k, 0]) for k in range(m) }
        slacks_output = { output_cols[r]: float(slack_out_arr[r, 0]) for r in range(s) }
        
        # 8) Calcular "scale efficiency": Se necesita la eficiencia CCR (CRS) para la misma DMU.
        df_ccr_for_scale = run_ccr(
            df=df,
            dmu_column=dmu_column,
            input_cols=input_cols,
            output_cols=output_cols,
            orientation=orientation,
            super_eff=False 
        )
        
        ccr_eff_series = df_ccr_for_scale.loc[df_ccr_for_scale[dmu_column] == dmus[i], "tec_efficiency_ccr"]
        
        ccr_eff = np.nan
        if not ccr_eff_series.empty:
            ccr_eff = ccr_eff_series.iloc[0]

        scale_eff = np.nan
        if current_eff_val is not None and ccr_eff is not None and current_eff_val != 0 and not np.isnan(current_eff_val) and not np.isnan(ccr_eff):
            if orientation == "input":
                scale_eff = ccr_eff / current_eff_val
            else: 
                scale_eff = ccr_eff / current_eff_val
        
        lambdas_vals = {}
        if lambdas_var.value is not None:
            for idx, ref_dmu_id in enumerate(dmus_ref): 
                lambdas_vals[ref_dmu_id] = float(lambdas_var.value[idx,0])
        else:
            for ref_dmu_id in dmus_ref: 
                lambdas_vals[ref_dmu_id] = np.nan

        # 9) Determinación de RTS (RTS_label)
        rts_label = "VRS" 
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            if len(prob.constraints) > 0 and prob.constraints[-1].is_dual_ проє_known():
                dual_sum_lambda_constraint = prob.constraints[-1].dual_value
                if dual_sum_lambda_constraint is not None:
                    if abs(dual_sum_lambda_constraint) < 1e-6: 
                        rts_label = "CRS" 
                    elif dual_sum_lambda_constraint < 0:
                        rts_label = "IRS" if orientation == "input" else "DRS"
                    else: 
                        rts_label = "DRS" if orientation == "input" else "IRS"
                else:
                    rts_label = "Dual N/A"
            else:
                rts_label = "Constraints N/A" 

        # 10) Guardar registro
        registros[i] = {
            dmu_column: dmus[i],
            "efficiency": np.round(current_eff_val, 6) if current_eff_val is not None else np.nan,
            "model": "BCC",
            "orientation": orientation,
            "super_eff": bool(super_eff),
            "lambda_vector": lambdas_vals, 
            "slacks_inputs": slacks_input,
            "slacks_outputs": slacks_output,
            "scale_efficiency": np.round(scale_eff, 6) if scale_eff is not None else np.nan,
            "rts_label": rts_label
        }

    # 11) Construir DataFrame de salida
    df_out = pd.DataFrame(registros)
    return df_out
