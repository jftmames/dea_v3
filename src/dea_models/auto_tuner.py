# src/dea_models/auto_tuner.py

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
    Aquí se puede invocar a un LLM para sugerir combinaciones.
    Por simplicidad, generamos candidatos removiendo de a 1 input o output.
    """
    all_vars = input_cols + output_cols
    candidates = []
    for var in all_vars:
        if var in input_cols:
            new_inputs = [v for v in input_cols if v != var]
            new_outputs = output_cols.copy()
        else:
            new_inputs = input_cols.copy()
            new_outputs = [v for v in output_cols if v != var]
        if not new_inputs or not new_outputs:
            continue

        candidates.append({
            "candidate_id": str(uuid.uuid4()),
            "inputs": new_inputs,
            "outputs": new_outputs
        })
        if len(candidates) >= n_candidates:
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
    Retorna DataFrame:
      candidate_id, inputs, outputs, avg_efficiency, eee_score_sim, delta_eff, delta_eee
    """
    rows = []
    # Eficiencia original con input_cols/output_cols de referencia
    # (Se asume que los candidatos incluyen la configuración base también)

    # Aquí solo iteramos y computamos avg efficiency
    for cand in candidates:
        inp = cand["inputs"]
        outp = cand["outputs"]
        # Validar
        validate_positive_dataframe(df, inp + outp)

        # Correr DEA para todo el dataset
        df_eff = _run_dea_internal(
            df=df,
            inputs=inp,
            outputs=outp,
            model=model,
            orientation="input",
            super_eff=False
        )
        avg_eff = float(df_eff["efficiency"].mean())

        # Calcular EEE simulado (placeholder: misma fórmula original, se reemplaza con lógica real)
        eee_sim = avg_eff * 100  # ejemplo simplificado

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
    # Calcular cambios relativos respecto a baseline (primera fila)
    if not df_cand.empty:
        base_eff = df_cand.loc[0, "avg_efficiency"]
        base_eee = df_cand.loc[0, "eee_score_sim"]
        df_cand["delta_eff"] = df_cand["avg_efficiency"] - base_eff
        df_cand["delta_eee"] = df_cand["eee_score_sim"] - base_eee

    return df_cand
