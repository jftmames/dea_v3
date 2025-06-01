# src/dea_models/network.py

import numpy as np
import pandas as pd
import cvxpy as cp

from .utils import validate_positive_dataframe

def run_network_dea(
    df: pd.DataFrame,
    dmu_column: str,
    stage1_inputs: list[str],
    stage1_outputs: list[str],
    stage2_inputs: list[str],
    stage2_outputs: list[str],
    linkage_matrix: np.ndarray,
    rts_stage1: str = "CRS",
    rts_stage2: str = "CRS"
) -> pd.DataFrame:
    """
    DEA en dos etapas. 
    - stage1_inputs/outputs: columnas de etapa 1.
    - stage2_inputs/outputs: columnas de etapa 2.
    - linkage_matrix: array que conecta outputs etapa1 → inputs etapa2 
                      (shape: len(stage2_inputs) × len(stage1_outputs)).
    Retorna DataFrame con columnas:
      DMU, efficiency_stage1, efficiency_stage2, efficiency_overall, lambda_stage1 (dict), lambda_stage2 (dict)
    """
    if dmu_column not in df.columns:
        raise ValueError(f"La columna DMU '{dmu_column}' no existe en el DataFrame.")

    # 1) Validar positividad en todas las columnas de ambas etapas
    cols = stage1_inputs + stage1_outputs + stage2_inputs + stage2_outputs
    validate_positive_dataframe(df, cols)

    # 2) Construir X1, Y1, X2, Y2
    X1 = df[stage1_inputs].to_numpy().T    # m1 × n
    Y1 = df[stage1_outputs].to_numpy().T   # s1 × n
    X2 = df[stage2_inputs].to_numpy().T    # m2 × n
    Y2 = df[stage2_outputs].to_numpy().T   # s2 × n

    dmus = df[dmu_column].astype(str).tolist()
    n = X1.shape[1]
    m1 = X1.shape[0]
    s1 = Y1.shape[0]
    m2 = X2.shape[0]
    s2 = Y2.shape[0]

    resultados = []
    for i in range(n):
        # Variables λ^1 (para etapa 1) y λ^2 (para etapa 2)
        lambda1 = cp.Variable((n, 1), nonneg=True)
        lambda2 = cp.Variable((n, 1), nonneg=True)

        # Variables theta1, theta2
        theta1 = cp.Variable()
        theta2 = cp.Variable()

        # Datos DMU i
        x1_0 = X1[:, [i]]
        y1_0 = Y1[:, [i]]
        x2_0 = X2[:, [i]]
        y2_0 = Y2[:, [i]]

        cons = []
        # Etapa 1: Y1 λ1 >= y1_0, X1 λ1 <= θ1 * x1_0
        cons.append(Y1 @ lambda1 >= y1_0)
        cons.append(X1 @ lambda1 <= theta1 * x1_0)
        if rts_stage1 == "VRS":
            cons.append(cp.sum(lambda1) == 1)

        # Etapa 2: Y2 λ2 >= y2_0, X2 λ2 <= θ2 * x2_0
        cons.append(Y2 @ lambda2 >= y2_0)
        cons.append(X2 @ lambda2 <= theta2 * x2_0)
        if rts_stage2 == "VRS":
            cons.append(cp.sum(lambda2) == 1)

        # Interconexión: linkage_matrix @ (Y1 λ1) == X2 λ2
        cons.append(linkage_matrix @ (Y1 @ lambda1) == X2 @ lambda2)

        # Objetivo: minimizar (θ1 + θ2)
        obj = cp.Minimize(theta1 + theta2)

        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.ECOS, abstol=1e-6, reltol=1e-6, feastol=1e-8, verbose=False)

        theta1_val = float(theta1.value) if theta1.value is not None else np.nan
        theta2_val = float(theta2.value) if theta2.value is not None else np.nan
        overall_eff = (
            np.nan 
            if np.isnan(theta1_val) or np.isnan(theta2_val) 
            else float((theta1_val + theta2_val) / 2)
        )

        lambda1_vals = {dmus[j]: float(lambda1.value[j]) for j in range(n)}
        lambda2_vals = {dmus[j]: float(lambda2.value[j]) for j in range(n)}

        resultados.append({
            dmu_column: dmus[i],
            "efficiency_stage1": theta1_val,
            "efficiency_stage2": theta2_val,
            "efficiency_overall": overall_eff,
            "lambda_stage1": lambda1_vals,
            "lambda_stage2": lambda2_vals
        })

    return pd.DataFrame(resultados)


def run_multi_stage_network(
    df: pd.DataFrame,
    dmu_column: str,
    stages: list[tuple[list[str], list[str]]],
    linkages: list[np.ndarray],
    rts_list: list[str]
) -> pd.DataFrame:
    """
    DEA en N etapas.
    - stages: lista de tuplas [(input_cols_k, output_cols_k), ...]
    - linkages: lista de matrices Z_k que conectan outputs k → inputs k+1
    - rts_list: lista de "CRS" o "VRS" por cada etapa
    Retorna DataFrame con columna 'DMU' y eficiencia por etapa y overall.
    """
    # Validar que len(stages) = len(rts_list) y len(linkages)=len(stages)-1
    num_etapas = len(stages)
    if len(rts_list) != num_etapas or len(linkages) != num_etapas - 1:
        raise ValueError("Dimensiones de stages/rts_list/linkages no coinciden.")

    # Validar positividades en todas las columnas de todas las etapas
    cols_todas = []
    for inp, out in stages:
        cols_todas.extend(inp)
        cols_todas.extend(out)
    validate_positive_dataframe(df, cols_todas)

    # Construir X_k, Y_k para cada etapa
    X_list = []
    Y_list = []
    for inp, out in stages:
        X_list.append(df[inp].to_numpy().T)
        Y_list.append(df[out].to_numpy().T)
    dmus = df[dmu_column].astype(str).tolist()
    n = X_list[0].shape[1]

    resultados = []
    for i in range(n):
        # Variables y lazos para cada etapa
        lambdas_vars = []
        thetas = []
        cons = []
        for k in range(num_etapas):
            Xk = X_list[k]
            Yk = Y_list[k]
            mk, nk = Xk.shape[0], Xk.shape[1]

            lamb_k = cp.Variable((nk, 1), nonneg=True)
            theta_k = cp.Variable()
            xk_0 = Xk[:, [i]]
            yk_0 = Yk[:, [i]]

            # Restricciones en etapa k
            cons.append(Yk @ lamb_k >= yk_0)
            cons.append(Xk @ lamb_k <= theta_k * xk_0)
            if rts_list[k] == "VRS":
                cons.append(cp.sum(lamb_k) == 1)

            lambdas_vars.append(lamb_k)
            thetas.append(theta_k)

        # Interconexiones: Z_k (linkages[k]) conecta Y_k λ_k → X_{k+1} λ_{k+1}
        for k in range(num_etapas - 1):
            Yk = Y_list[k]
            lamb_k = lambdas_vars[k]
            Xk1 = X_list[k + 1]
            lamb_k1 = lambdas_vars[k + 1]
            Zk = linkages[k]  # debe ser shape: (#inputs_{k+1} × #outputs_k)
            cons.append(Zk @ (Yk @ lamb_k) == Xk1 @ lamb_k1)

        # Objetivo: minimizar suma de thetas
        obj = cp.Minimize(sum(thetas))
        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.ECOS, abstol=1e-6, reltol=1e-6, feastol=1e-8, verbose=False)

        # Leer valores
        theta_vals = [
            float(t.value) if t.value is not None else np.nan 
            for t in thetas
        ]
        overall = (
            np.nan 
            if any(np.isnan(v) for v in theta_vals) 
            else float(sum(theta_vals) / num_etapas)
        )
        lambda_vals_dicts = []
        for k in range(num_etapas):
            lamk = lambdas_vars[k]
            lambda_vals_dicts.append({dmus[j]: float(lamk.value[j]) for j in range(n)})

        fila = {dmu_column: dmus[i]}
        for k in range(num_etapas):
            fila[f"eff_stage_{k+1}"] = theta_vals[k]
            fila[f"lambda_stage_{k+1}"] = lambda_vals_dicts[k]
        fila["eff_overall"] = overall
        resultados.append(fila)

    return pd.DataFrame(resultados)
