import pandas as pd
from dea_models.radial import run_ccr, run_bcc
from dea_models.visualizations import plot_efficiency_histogram

def mostrar_resultados(
    df: pd.DataFrame,
    dmu_column: str,
    inputs: list[str],
    outputs: list[str]
) -> dict:
    """
    Versión base que ejecuta CCR y BCC y devuelve los dataframes de resultados.
    """
    resultados = {}

    # Ejecutar CCR
    df_ccr = run_ccr(
        df=df,
        dmu_column=dmu_column,
        input_cols=inputs,
        output_cols=outputs,
        orientation="input"
    )
    resultados["df_ccr"] = df_ccr

    # Ejecutar BCC
    df_bcc = run_bcc(
        df=df,
        dmu_column=dmu_column,
        input_cols=inputs,
        output_cols=outputs,
        df_ccr_results=df_ccr
    )
    resultados["df_bcc"] = df_bcc

    return resultados
