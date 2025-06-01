# src/data_validator.py

import os
import json
import pandas as pd
from openai import OpenAI

# Importamos la función de validación de dea_models/utils.py
from dea_models.utils import validate_positive_dataframe

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------- reglas formales básicas ----------
def _formal_checks(
    df: pd.DataFrame,
    inputs: list[str],
    outputs: list[str]
) -> list[str]:
    """
    Verifica reglas básicas:
      1. Que las columnas inputs+outputs existan y sean numéricas > 0.
      2. Que no existan valores nulos en esas columnas.
      3. Devuelve una lista de mensajes de error si hay problemas.
    """
    issues = []

    # 1) Validar positividad y conversión a float usando validate_positive_dataframe
    try:
        cols = inputs + outputs
        validate_positive_dataframe(df, cols)
    except ValueError as e:
        issues.append(str(e))

    # 2) Valores nulos en todo el DataFrame
    if df.isnull().values.any():
        issues.append("Se encontraron valores nulos en el DataFrame.")

    # 3) Columnas no numéricas (fuera de inputs/outputs)
    for col in df.columns:
        if col not in inputs + outputs and not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"Columna '{col}' no es numérica.")

    return issues


# ---------- consulta al LLM (RAG) ----------
def _llm_suggest(
    df_head: str,
    inputs: list[str],
    outputs: list[str]
) -> dict:
    """
    Envía al modelo un prompt para evaluar si las columnas de inputs/outputs
    son adecuadas y sugiere correcciones. Devuelve JSON con keys:
      - ready: bool
      - issues: lista de strings sobre problemas detectados
      - suggested_fixes: lista de sugerencias concretas
    """
    prompt = (
        "Eres un experto en DEA. Evalúa si las columnas INPUTS y OUTPUTS "
        "son adecuadas y sugiere mejoras.\n\n"
        f"HEAD del DataFrame (JSON):\n{df_head}\n\n"
        f"INPUTS: {inputs}\nOUTPUTS: {outputs}\n\n"
        "Responde **SOLO** en JSON con las claves 'ready', 'issues', 'suggested_fixes'."
    )
    chat = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content = chat.choices[0].message.content
    try:
        return json.loads(content)
    except Exception:
        return {"raw": content}


# ---------- API pública ----------
def validate(
    df: pd.DataFrame,
    inputs: list[str],
    outputs: list[str]
) -> dict:
    """
    Ejecuta las validaciones formales y consulta al LLM.
    Retorna dict con:
      - 'formal_issues': lista de strings (errores formales)
      - 'llm': dict con la respuesta JSON del LLM (o {'raw': texto} si no fue JSON)
    """
    formal_issues = _formal_checks(df, inputs, outputs)
    llm_json = _llm_suggest(df.head().to_json(), inputs, outputs)
    return {"formal_issues": formal_issues, "llm": llm_json}


# ---------- Ejemplo de uso rápido (opcional) ----------
if __name__ == "__main__":
    df_ejemplo = pd.DataFrame({
        "DMU": ["A", "B", "C"],
        "input1": [1.0, 2.0, 3.0],
        "input2": [1.0, 1.0, 2.0],
        "output1": [1.0, 2.0, 1.0]
    })
    inputs = ["input1", "input2"]
    outputs = ["output1"]

    resultado_validacion = validate(df_ejemplo, inputs, outputs)
    print("Validación formal y RAG:", resultado_validacion)
