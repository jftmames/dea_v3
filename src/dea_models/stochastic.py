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
    rts: str = "CRS",  # Although rts is passed, it's not used by run_ccr directly in this version
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
    # n = len(dmus) # n is now used within bootstrap_efficiencies

    # 3) Eficiencia original (snapshot completo)
    #    Para ello, corremos run_ccr sobre todo el df de entrada
    try:
        df_orig_eff = run_ccr(
            df=df,
            dmu_column=dmu_column,
            input_cols=input_cols,
            output_cols=output_cols,
            orientation=orientation,
            super_eff=False  # Assuming CRS for original efficiency as rts is not directly used here
        )
    except Exception as e:
        raise RuntimeError(f"Error al calcular eficiencia original con run_ccr: {e}")

    # Extraemos la eficiencia para cada DMU en orden
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
        rts=rts # Pass rts to bootstrap_efficiencies, which then passes to run_ccr if needed
    )

    # 5) Construir DataFrame final con medias e intervalos de confianza
    #    (This part was step 6 in the original code)
    filas = []
    for dmu in dmus:
        arr = np.array(bootstrap_vals.get(dmu, [])) # Use .get for safety, though bootstrap_vals should have all dmus
        if arr.size == 0:
            # Si no tenemos valores bootstrap (p. ej. todos los bootstraps fallaron para esta DMU o no apareció)
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

# al final de src/dea_models/stochastic.py

def bootstrap_efficiencies(
    df: pd.DataFrame,
    dmu_column: str,
    input_cols: list[str],
    output_cols: list[str],
    n_bootstrap: int = 1000,
    orientation: str = "input",
    rts: str = "CRS" # Added rts here to be consistent and potentially used by run_ccr
) -> dict[str, list[float]]:
    """
    Retorna un diccionario {DMU: [lista de eficiencias en cada bootstrap]}.
    """
    if dmu_column not in df.columns:
        raise ValueError(f"La columna DMU '{dmu_column}' no existe en el DataFrame.")
    validate_dataframe(df, input_cols, output_cols, allow_zero=False, allow_negative=False)

    dmus = df[dmu_column].astype(str).tolist()
    n = len(df) # Number of rows in the original dataframe for sampling

    # Inicializar diccionario vacío de listas
    bootstrap_vals = {dmu: [] for dmu in dmus}

    for _ in range(n_bootstrap): # Use _ if b is not used
        # Re-muestrear con reemplazo todo el DataFrame
        # Ensuring that the sample size is the same as the original df
        df_b = df.sample(n=n, replace=True).reset_index(drop=True)

        # Para cada DMU en la muestra remuestreada, calcular eficiencia CCR
        try:
            df_b_eff = run_ccr(
                df=df_b,
                dmu_column=dmu_column,
                input_cols=input_cols,
                output_cols=output_cols,
                orientation=orientation,
                super_eff=False # Assuming CCR, as rts is not directly used by run_ccr in the provided snippets
                                # If run_ccr can handle VRS/CRS, then rts should be passed here.
                                # For now, sticking to CCR as implied by "calcula eficiencia CCR"
            )
        except Exception:
            # Si falla un bootstrap particular, lo ignoramos y continuamos
            continue

        # df_b_eff contiene la eficiencia CCR de cada DMU presente en df_b
        # We need to ensure we are adding efficiencies to the *original* DMUs
        # even if some don't appear in a particular bootstrap sample.
        # The current logic correctly appends to bootstrap_vals[dmu_id]
        # which covers DMUs present in the bootstrap sample.
        # DMUs not in df_b for a given iteration simply won't get an efficiency score for that iteration.
        for _, row in df_b_eff.iterrows():
            dmu_id = row[dmu_column]
            # It's possible a dmu_id from df_b might not be in the original dmus list if df_b has different dmu ids after sampling.
            # However, df.sample will sample rows, so dmu_column values will be from the original df.
            # And bootstrap_vals is initialized with original dmus.
            if dmu_id in bootstrap_vals: # Ensure dmu_id is one of the original DMUs
                eff_b = float(row["efficiency"])
                bootstrap_vals[dmu_id].append(eff_b)

    return bootstrap_vals
