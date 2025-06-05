# jftmames/-dea-deliberativo-mvp/-dea-deliberativo-mvp-b44b8238c978ae0314af30717b9399634d28f8f9/src/dea_models/auto_tuner.py
import pandas as pd
import numpy as np
import uuid

from .radial import _run_dea_internal
from .utils import validate_positive_dataframe

def generate_candidates(
    df: pd.DataFrame,
    dmu_column: str,
    input_cols: list[str],
    output_cols: list[str],
    inquiry_tree: dict,
    eee_score: float,
    n_candidates: int = 5
) -> list[dict]:
    """
    Genera n_candidates propuestas de {inputs: new_inputs, outputs: new_outputs}
    basado en subpreguntas e eee_score.
    """
    all_vars = input_cols + output_cols
    candidates = []
    
    # Añadimos la configuración actual como candidato base
    candidates.append({
        "candidate_id": str(uuid.uuid4()),
        "inputs": input_cols,
        "outputs": output_cols
    })

    for var in all_vars:
        new_inputs = input_cols.copy()
        new_outputs = output_cols.copy()

        if var in new_inputs:
            new_inputs.remove(var)
        elif var in new_outputs:
            new_outputs.remove(var)

        if not new_inputs or not new_outputs:
            continue

        candidates.append({
            "candidate_id": str(uuid.uuid4()),
            "inputs": new_inputs,
            "outputs": new_outputs
        })
        if len(candidates) > n_candidates: # > para contar el base
            break
            
    return candidates


def evaluate_candidates(
    df: pd.DataFrame,
    dmu_column: str,
    candidates: list[dict],
    model: str = "CCR"
) -> pd.DataFrame:
    """
    Para cada candidato (inputs/outputs), calcula eficiencia promedio y EEE simulado.
    """
    rows = []
    
    for cand in candidates:
        inp = cand["inputs"]
        outp = cand["outputs"]
        
        try:
            validate_positive_dataframe(df, inp + outp)
            df_eff = _run_dea_internal(
                df=df,
                dmu_col_name=dmu_column,
                inputs=inp,
                outputs=outp,
                model=model,
                orientation="input",
                super_eff=False
            )
            avg_eff = float(df_eff["efficiency"].mean()) if not df_eff.empty and "efficiency" in df_eff.columns else np.nan
        except ValueError as e:
            print(f"Skipping candidate {inp}/{outp} due to validation error: {e}")
            avg_eff = np.nan

        eee_sim = avg_eff * 100 if not np.isnan(avg_eff) else np.nan

        rows.append({
            "candidate_id": cand["candidate_id"],
            "inputs": inp,
            "outputs": outp,
            "avg_efficiency": avg_eff,
            "eee_score_sim": eee_sim,
            "delta_eff": None,
            "delta_eee": None
        })

    df_cand = pd.DataFrame(rows)
    if not df_cand.empty and len(df_cand) > 1:
        base_eff = df_cand.loc[0, "avg_efficiency"]
        base_eee = df_cand.loc[0, "eee_score_sim"]
        if not pd.isna(base_eff):
            df_cand["delta_eff"] = df_cand["avg_efficiency"] - base_eff
        if not pd.isna(base_eee):
            df_cand["delta_eee"] = df_cand["eee_score_sim"] - base_eee

    return df_cand
