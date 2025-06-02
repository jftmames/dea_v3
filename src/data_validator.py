# src/data_validator.py

import os
import json
import pandas as pd
from openai import OpenAI

# Importamos la función de validación de dea_models/utils.py
from dea_models.utils import validate_positive_dataframe, validate_dataframe


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
        # Se usa validate_dataframe ahora que está disponible y es más general
        validate_dataframe(df, inputs, outputs, allow_zero=False, allow_negative=False)
    except ValueError as e:
        issues.append(str(e))

    # 2) Valores nulos en todo el DataFrame (solo en las columnas relevantes, como lo hace validate_dataframe)
    # validate_dataframe ya debería manejar esto para inputs/outputs.
    # Si queremos validar todo el DataFrame, podemos mantener esta línea.
    # Si es solo para inputs/outputs, validate_dataframe ya lo cubre.
    # Asumo que quieres una comprobación general adicional para todo el DF.
    if df.isnull().values.any():
        issues.append("Se encontraron valores nulos en el DataFrame.")

    # 3) Columnas no numéricas (fuera de inputs/outputs)
    # Esta parte no es estrictamente necesaria si validate_dataframe ya se encarga de inputs/outputs.
    # Si hay otras columnas en el DF que no son inputs/outputs y no son numéricas, esto las detectará.
    for col in df.columns:
        if col not in inputs + outputs and not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"Columna '{col}' no es numérica y no es un input/output.")


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
      - raw: (opcional) contenido crudo si la respuesta no es JSON válido
    """
    prompt = (
        "Eres un experto en DEA. Evalúa si las columnas INPUTS y OUTPUTS "
        "son adecuadas y sugiere mejoras.\n\n"
        f"HEAD:\n{df_head}\n\n"
        f"INPUTS: {inputs}\nOUTPUTS: {outputs}\n\n"
        "Responde en JSON con 'ready', 'issues', 'suggested_fixes'."
    )
    try:
        chat = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = chat.choices[0].message.content.strip()

        # Si viene vacío o no JSON, json.loads lanzará JSONDecodeError
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "ready": False,
                "issues": ["Respuesta del LLM no es JSON válido."],
                "raw": content
            }
    except Exception as e:
        # Si la llamada al cliente OpenAI falla (timeout, clave, etc.), devolvemos un mensaje claro
        return {
            "ready": False,
            "issues": [f"Error al consultar el LLM: {e}"],
            "suggested_fixes": []
        }


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
      - 'llm': dict con la respuesta JSON del LLM (o error si falla)
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
