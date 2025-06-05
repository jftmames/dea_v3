# src/dea_models/stochastic.py
import numpy as np
import pandas as pd

from .radial import run_ccr
from .utils import validate_dataframe

def run_stochastic_dea(
    df: pd.DataFrame,
    dmu_column: str,
    input_cols: list[str],
    output_cols: list[str],
    orientation: str = "input",
    rts: str = "CRS",  
    n_bootstrap: int = 1000
) -> pd.DataFrame:
    """
    Corre DEA con bootstrapping. Por cada muestra remuestreada (con reemplazo),
    calcula eficiencia CCR para cada DMU y construye un intervalo de confianza.
    Retorna DataFrame con columnas:
      DMU, eff_mean, ci_lower, ci_upper, original_efficiency
    """

    # 1) Validar que exista la columna DMU
    if dmu_column not in df.columns:
        raise ValueError(f"La columna DMU '{dmu_column}' no existe en el DataFrame.")
    # 2) Validar que las columnas de inputs/outputs existan y sean >0
    validate_dataframe(df, input_cols, output_cols, allow_zero=False, allow_negative=False)

    # Lista de IDs de DMUs
    dmus = df[dmu_column].astype(str).tolist()

    # 3) Eficiencia original (snapshot completo)
    try:
        df_orig_eff = run_ccr(
            df=df,
            dmu_column=dmu_column,
            input_cols=input_cols,
            output_cols=output_cols,
            orientation=orientation,
            super_eff=False  
        )
    except Exception as e:
        raise RuntimeError(f"Error al calcular eficiencia original con run_ccr: {e}")

    original_eff = {
        row[dmu_column]: float(row["efficiency"])
        for _, row in df_orig_eff.iterrows()
    }

    # 4) Obtener valores bootstrap usando la nueva función
    bootstrap_vals = bootstrap_efficiencies(
        df=df,
        dmu_column=dmu_column,
        input_cols=input_cols,
        output_cols=output_cols,
        n_bootstrap=n_bootstrap,
        orientation=orientation,
        rts=rts 
    )

    # 5) Construir DataFrame final con medias e intervalos de confianza
    filas = []
    for dmu in dmus:
        arr = np.array(bootstrap_vals.get(dmu, [])) 
        if arr.size == 0:
            filas.append({
                "DMU": dmu,
                "eff_mean": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "original_efficiency": original_eff.get(dmu, np.nan)
            })
        else:
            mean_b = float(np.mean(arr))
            lower = float(np.percentile(arr, 2.5))
            upper = float(np.percentile(arr, 97.5))
            filas.append({
                "DMU": dmu,
                "eff_mean": mean_b,
                "ci_lower": lower,
                "ci_upper": upper,
                "original_efficiency": original_eff.get(dmu, np.nan)
            })

    return pd.DataFrame(filas)

def bootstrap_efficiencies(
    df: pd.DataFrame,
    dmu_column: str,
    input_cols: list[str],
    output_cols: list[str],
    n_bootstrap: int = 1000,
    orientation: str = "input",
    rts: str = "CRS" 
) -> dict[str, list[float]]:
    """
    Retorna un diccionario {DMU: [lista de eficiencias en cada bootstrap]}.
    """
    if dmu_column not in df.columns:
        raise ValueError(f"La columna DMU '{dmu_column}' no existe en el DataFrame.")
    validate_dataframe(df, input_cols, output_cols, allow_zero=False, allow_negative=False)

    dmus = df[dmu_column].astype(str).tolist()
    n = len(df) 

    bootstrap_vals = {dmu: [] for dmu in dmus}

    for _ in range(n_bootstrap): 
        df_b = df.sample(n=n, replace=True).reset_index(drop=True)

        try:
            df_b_eff = run_ccr( # Assuming CCR is used for stochastic DEA based on the original code
                df=df_b,
                dmu_column=dmu_column,
                input_cols=input_cols,
                output_cols=output_cols,
                orientation=orientation,
                super_eff=False 
            )
        except Exception:
            continue

        for _, row in df_b_eff.iterrows():
            dmu_id = row[dmu_column]
            if dmu_id in bootstrap_vals: 
                eff_b = float(row["efficiency"])
                bootstrap_vals[dmu_id].append(eff_b)

    return bootstrap_vals
