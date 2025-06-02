import numpy as np
import cvxpy as cp
import pandas as pd

from .utils import validate_positive_dataframe

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
                    if eff[i] != np.nan and eff[i] != 0: # For output, often efficiency is 1/phi_var if standard formulation is min 1/phi
                         # Based on common output formulation max phi, eff is phi.value directly.
                         # If it were min theta (where theta = 1/phi), then eff would be 1/theta.value
                         pass # Sticking to phi_var.value as the efficiency score based on Maximize(phi_var)
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
    dmu_col_name: str = "DMU" # Added to ensure DMU column is correctly identified
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
        if dmu_ids.name is None: # If index has no name, give it a default for consistency
            dmu_ids.name = "DMU_Index"
        # dmu_col_name = dmu_ids.name # Update dmu_col_name if using index

    return pd.DataFrame({
        dmu_col_name: dmu_ids,
        "efficiency": np.round(eff_scores, 6), # eff_scores should be eff_values
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
    orientation: str = "input", # Added orientation, though CCR is often input-oriented for theta
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
    # validate_positive_dataframe might modify df, so pass a copy if df is used later
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
        # Observaciones de la DMU i
        x0 = X[:, [i]]
        y0 = Y[:, [i]]

        # Determine reference set and lambda variables based on super_eff
        if super_eff:
            if n == 1: # Cannot run super-efficiency with only one DMU
                resultados.append({
                    dmu_column: dmus[i],
                    "tec_efficiency_ccr": 1.0, # Or np.nan
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

        # Decision variables for the two-stage CCR model (radial efficiency + slacks)
        # Stage 1: Calculate radial efficiency (theta_rad for input, phi_rad for output)
        # For CCR input-oriented model (standard):
        if orientation == "input":
            theta_rad = cp.Variable()
            cons_stage1 = [X_ref @ lambdas_var <= theta_rad * x0,
                           Y_ref @ lambdas_var >= y0]
            # No sum(lambdas)==1 for CRS
            obj_stage1 = cp.Minimize(theta_rad)
        else: # Output-oriented CCR
            phi_rad = cp.Variable()
            cons_stage1 = [X_ref @ lambdas_var <= x0,
                           Y_ref @ lambdas_var >= phi_rad * y0]
            obj_stage1 = cp.Maximize(phi_rad)

        prob_stage1 = cp.Problem(obj_stage1, cons_stage1)
        prob_stage1.solve(solver=cp.ECOS, abstol=1e-7, reltol=1e-7, feastol=1e-7, verbose=False)

        if prob_stage1.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            # Handle solver failure for stage 1
            resultados.append({
                dmu_column: dmus[i], "tec_efficiency_ccr": np.nan, "lambda_vector": {},
                "slacks_inputs": {col: np.nan for col in input_cols},
                "slacks_outputs": {col: np.nan for col in output_cols}, "rts_label": "CRS"
            })
            continue
        
        eff_val_rad = theta_rad.value if orientation == "input" else phi_rad.value
        if eff_val_rad is None: eff_val_rad = np.nan

        # Stage 2: Maximize sum of slacks, given the radial efficiency
        slacks_in_var = cp.Variable((m, 1), nonneg=True)
        slacks_out_var = cp.Variable((s, 1), nonneg=True)

        # Need to re-declare lambdas for stage 2 if they are to be re-optimized
        # or use the lambdas from stage 1. For classic two-stage, lambdas are fixed.
        # However, standard approach is to solve a new LP for slacks.
        # For simplicity here, we use the envelope from *all* DMUs (or ref_set for super_eff)
        # to find slacks for the specific x0, y0, using the calculated eff_val_rad.

        if orientation == "input":
            # Minimize -(sum(slacks_in) + sum(slacks_out)) which is Maximize sum of slacks
            # Constraints use eff_val_rad * x0 for inputs
            cons_stage2 = [X_ref @ lambdas_var + slacks_in_var == eff_val_rad * x0,
                           Y_ref @ lambdas_var - slacks_out_var == y0]
        else: # output-orientation
            # For output orientation, eff_val_rad = phi
            # Target outputs are eff_val_rad * y0. Inputs are x0.
             cons_stage2 = [X_ref @ lambdas_var + slacks_in_var == x0,
                           Y_ref @ lambdas_var - slacks_out_var == eff_val_rad * y0]
        
        # Objective for stage 2: Maximize sum of slacks
        # Using a common approach: sum of normalized slacks, or simply sum of slacks.
        # For simplicity, we can just solve the system of equations if lambdas are fixed.
        # If lambdas are variable in stage 2, then Maximize cp.sum(slacks_in_var) + cp.sum(slacks_out_var)
        
        # Here, the provided code structure defines slacks_in, slacks_out as variables in a single problem.
        # Let's stick to the original structure of run_ccr which seems to be a single stage model
        # defining theta and slacks together for input orientation.
        
        # Reverting to a single-stage model for CCR that includes slacks, as per original structure
        # This is an "enhanced" CCR model, not strictly two-stage as often described,
        # but solves for theta and slacks simultaneously.
        # This typically involves maximizing slacks after minimizing theta, or a weighted sum.
        # The original code solves for theta, and then slacks are derived.
        # For input orientation:
        # Minimize theta - eps * (sum(s_in) + sum(s_out))
        # s.t. X_lambda - s_in = theta * x0
        #      Y_lambda + s_out = y0
        #
        # The original run_ccr used:
        # cons.append(X @ lambdas + slacks_in == theta * x0)
        # cons.append(Y @ lambdas - slacks_out == y0)
        # obj = cp.Minimize(theta)
        # This is not quite right as slacks are defined to be non-negative.
        # X_lambda <= theta * x0  => X_lambda + s_in = theta * x0
        # Y_lambda >= y0          => Y_lambda - s_out = y0

        # Let's use the standard radial model and then calculate slacks based on optimal lambdas and theta/phi
        # This is more aligned with _dea_core's approach.
        # So, tec_efficiency_ccr is eff_val_rad.
        # Then calculate slacks.

        tec_eff = eff_val_rad
        lambdas_opt = lambdas_var.value
        if lambdas_opt is None: # If solver failed to find lambdas
            lambdas_opt = np.zeros((lambdas_var.shape[0], 1))

        if orientation == "input":
            slacks_input_values = (tec_eff * x0) - (X_ref @ lambdas_opt)
            slacks_output_values = (Y_ref @ lambdas_opt) - y0
        else: # output
            slacks_input_values = x0 - (X_ref @ lambdas_opt)
            slacks_output_values = (eff_val_rad * y0) - (Y_ref @ lambdas_opt)
        
        # Ensure slacks are non-negative (due to numerical precision, they might be slightly negative)
        slacks_input_values[slacks_input_values < 0] = 0
        slacks_output_values[slacks_output_values < 0] = 0

        # Populate lambdas_vals dictionary
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
    super_eff: bool = False
) -> pd.DataFrame:
    """
    Ejecuta BCC radial y retorna DataFrame con columnas:
      DMU, tec_efficiency_bcc, lambda_vector, slacks_inputs, slacks_outputs, scale_efficiency, rts_label
    """
    if dmu_column not in df.columns:
        raise ValueError(f"La columna DMU '{dmu_column}' no existe.")

    cols_to_validate = input_cols + output_cols
    validate_positive_dataframe(df.copy(), cols_to_validate)

    X_orig = df[input_cols].to_numpy().T  # shape: (m, n)
    Y_orig = df[output_cols].to_numpy().T  # shape: (s, n)
    dmus = df[dmu_column].astype(str).tolist()
    n_total_dmus = X_orig.shape[1]  # número de DMUs
    m_inputs = X_orig.shape[0]  # número de inputs
    s_outputs = Y_orig.shape[0]  # número de outputs

    resultados = []
    for i in range(n_total_dmus):
        # Observaciones de la DMU i
        x_i = X_orig[:, [i]]
        y_i = Y_orig[:, [i]]

        # Handle super_eff: set up reference matrices and lambda variables
        if super_eff:
            if n_total_dmus == 1: # Cannot run super-efficiency with only one DMU
                # Append a default/placeholder result for this DMU
                resultados.append({
                    dmu_column: dmus[i], "tec_efficiency_bcc": 1.0, # Or np.nan
                    "lambda_vector": {dmus[i]: 1.0},
                    "slacks_inputs": {col: 0.0 for col in input_cols},
                    "slacks_outputs": {col: 0.0 for col in output_cols},
                    "scale_efficiency": 1.0, "rts_label": "CRS" # Placeholder
                })
                continue
            mask = np.ones(n_total_dmus, dtype=bool)
            mask[i] = False
            X_ref_mat = X_orig[:, mask]
            Y_ref_mat = Y_orig[:, mask]
            lambdas_var = cp.Variable((n_total_dmus - 1, 1), nonneg=True)
            ref_dmus_indices = [idx for idx, val in enumerate(mask) if val]
        else:
            X_ref_mat = X_orig
            Y_ref_mat = Y_orig
            lambdas_var = cp.Variable((n_total_dmus, 1), nonneg=True)
            ref_dmus_indices = list(range(n_total_dmus))

        # --- fragmento de dea_models/radial.py dentro de run_bcc  ---
        # … (ya definiste variables lambda, theta/phi, etc. - lambdas_var is defined above)

        # DEFINIMOS SLACKS AQUÍ, para input y output, siempre aunque no se usen
        # slacks_in[k] >= 0 para cada insumo k
        slacks_in = cp.Variable((m_inputs, 1), nonneg=True)
        # slacks_out[r] >= 0 para cada producto r
        slacks_out = cp.Variable((s_outputs, 1), nonneg=True)
        
        cons = []

        # Ahora, en el modelo BCC input-oriented, las restricciones originales se vuelven:
        if orientation == "input":
            phi = cp.Variable() # As per snippet for input-orientation
            # input‐oriented BCC (from snippet):
            cons = [
                Y_ref_mat @ lambdas_var + slacks_out >= phi * y_i,    # (1) productos con slack de exceso
                X_ref_mat @ lambdas_var - slacks_in <= x_i,          # (2) insumos con slack de exceso
            ]
            # Para BCC: suma(lambdas) == 1
            cons.append(cp.sum(lambdas_var) == 1)
            # Y definimos el objeto objetivo MAXIMIZAR phi      
            obj = cp.Maximize(phi)

        else: # output-oriented
            theta = cp.Variable() # As per snippet for output-orientation
            # output-oriented BCC (from snippet):
            cons = [
                Y_ref_mat @ lambdas_var <= y_i,                      # (1) productos
                X_ref_mat @ lambdas_var + slacks_in >= theta * x_i,  # (2) insumos con slack
            ]
            cons.append(cp.sum(lambdas_var) == 1)
            obj = cp.Minimize(theta)

        # IMPORTANTE: si model es BCC, el rts está forzado a "VRS"
        # (ya te aseguraste antes de pasar rts="VRS" cuando llamas a run_bcc) - This is handled by sum(lambdas)==1

        # Construcción y resolución del problema
        prob = cp.Problem(obj, cons)
        prob.solve(
            solver=cp.ECOS,
            abstol=1e-7, # Using slightly more robust tolerance
            reltol=1e-7,
            feastol=1e-7, # Adjusted feastol
            verbose=False
        )

        # --- EXTRAEMOS VALORES, revisando si .value es None ---
        current_eff_val = np.nan
        # 1) Eficiencia (theta o phi)
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            if orientation == "input":
                phi_val = phi.value
                current_eff_val = float(phi_val) if phi_val is not None else np.nan
            else: # output-oriented
                theta_val = theta.value
                current_eff_val = float(theta_val) if theta_val is not None else np.nan
        
        # 2) SLACKS: si slacks_in.value es None, los rellenamos con ceros
        if slacks_in.value is None:
            slack_in_arr = np.zeros((m_inputs, 1))
        else:
            slack_in_arr = slacks_in.value

        if slacks_out.value is None:
            slack_out_arr = np.zeros((s_outputs, 1))
        else:
            slack_out_arr = slacks_out.value

        # Convertimos a diccionario {nombre_insumo: slack_value}
        slacks_input_dict = {}
        for k_idx in range(len(input_cols)):
            # slack_in_arr tiene forma (m,1), m = número de inputs
            # A veces CVXPY regresa arrays columna de numpy, así que accedemos con [k,0]
            raw_in = slack_in_arr[k_idx, 0] if slack_in_arr.ndim > 1 else slack_in_arr[k_idx]
            slacks_input_dict[input_cols[k_idx]] = float(raw_in)

        slacks_output_dict = {}
        for r_idx in range(len(output_cols)):
            raw_out = slack_out_arr[r_idx, 0] if slack_out_arr.ndim > 1 else slack_out_arr[r_idx]
            slacks_output_dict[output_cols[r_idx]] = float(raw_out)
        
        # Lambda values
        lambdas_vals = {}
        if lambdas_var.value is not None:
            for idx, ref_idx in enumerate(ref_dmus_indices):
                lambdas_vals[dmus[ref_idx]] = float(lambdas_var.value[idx,0])
        else: # Handle case where lambdas might not be found (e.g. solver issue)
            for ref_idx in ref_dmus_indices:
                 lambdas_vals[dmus[ref_idx]] = np.nan


        # Para calcular scale_efficiency: corremos CCR
        # Note: run_ccr itself handles super_eff=False correctly
        df_ccr_for_scale = run_ccr(
            df=df, # Pass the original full DataFrame
            dmu_column=dmu_column,
            input_cols=input_cols,
            output_cols=output_cols,
            orientation=orientation, # Use the same orientation for CCR to compare
            super_eff=False # Scale efficiency is typically against the full frontier
        )
        
        ccr_eff_series = df_ccr_for_scale.loc[df_ccr_for_scale[dmu_column] == dmus[i], "tec_efficiency_ccr"]
        
        ccr_eff = np.nan
        if not ccr_eff_series.empty:
            ccr_eff = ccr_eff_series.iloc[0]

        scale_eff = np.nan
        # Standard scale efficiency: SE = TE_CRS / TE_VRS
        # If input-oriented, tec_efficiency_bcc (current_eff_val) is theta_VRS. ccr_eff is theta_CRS.
        # So SE = theta_CRS / theta_VRS.
        # If output-oriented (snippet model), current_eff_val is theta_VRS (input factor for output model).
        # And ccr_eff would be theta_CRS (input factor for output model CCR).
        # The snippet model is unconventional. Assuming current_eff_val is comparable to CCR efficiency.
        # If orientation = "input", current_eff_val is phi_VRS (output expansion). ccr_eff is phi_CRS (output expansion)
        # SE = Overall / Pure = CRS_eff / VRS_eff.
        # If using standard theta (input) / phi (output) interpretation for efficiency scores:
        if orientation == "input": # snippet model current_eff_val = phi_BCC, ccr_eff should be phi_CCR
            # So, if phi is output expansion, SE = phi_CCR / phi_BCC
            if current_eff_val is not None and current_eff_val != 0 and not np.isnan(current_eff_val) and not np.isnan(ccr_eff):
                 scale_eff = ccr_eff / current_eff_val
            else:
                 scale_eff = np.nan
        else: # orientation == "output", snippet model current_eff_val = theta_BCC (input factor), ccr_eff should be theta_CCR (input factor)
              # Here, efficiency is 1/theta. So TE_VRS = 1/current_eff_val, TE_CRS = 1/ccr_eff
              # SE = TE_CRS / TE_VRS = (1/ccr_eff) / (1/current_eff_val) = current_eff_val / ccr_eff
            if ccr_eff is not None and ccr_eff != 0 and not np.isnan(current_eff_val) and not np.isnan(ccr_eff):
                scale_eff = current_eff_val / ccr_eff
            else:
                scale_eff = np.nan


        # RTS_label según duales:
        rts_label = "N/A" # Default if not determinable
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and len(prob.constraints) > 0 : # Ensure problem was solved and constraints exist
            # The sum(lambdas)==1 constraint is the last one added for BCC.
            dual_sum_lambda_constraint = prob.constraints[-1].dual_value
            if dual_sum_lambda_constraint is not None:
                if abs(dual_sum_lambda_constraint) < 1e-6: # Threshold for zero
                    rts_label = "CRS"
                elif dual_sum_lambda_constraint < 0:
                    rts_label = "IRS" # For input-oriented models, negative dual implies IRS
                else: # dual_sum_lambda_constraint > 0
                    rts_label = "DRS" # For input-oriented models, positive dual implies DRS
                # Note: Interpretation of dual for RTS can depend on orientation and model formulation.
                # Standard for input-oriented BCC: u_0 < 0 IRS, u_0 > 0 DRS, u_0 = 0 CRS
                # (where u_0 is the dual of sum(lambda)=1 constraint)
            else:
                rts_label = "Dual N/A"

        resultados.append({
            dmu_column: dmus[i],
            "tec_efficiency_bcc": np.round(current_eff_val, 6) if current_eff_val is not None else np.nan,
            "lambda_vector": lambdas_vals,
            "slacks_inputs": slacks_input_dict,
            "slacks_outputs": slacks_output_dict,
            "scale_efficiency": np.round(scale_eff, 6) if scale_eff is not None else np.nan,
            "rts_label": rts_label
        })

    return pd.DataFrame(resultados)
