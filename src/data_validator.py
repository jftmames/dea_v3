# src/data_validator.py
import os
import json
import pandas as pd
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- reglas formales básicas ----------
def _formal_checks(df: pd.DataFrame, inputs: list[str], outputs: list[str]) -> list[str]:
    """
    Comprueba únicamente las columnas de 'inputs' y 'outputs':
      - que existan en el DataFrame,
      - que no tengan valores nulos,
      - que sean numéricas,
      - que todos los valores sean > 0.
    Devuelve lista de mensajes de error (vacía si todo OK).
    """
    issues = []

    # 1) Verificar existencia de columnas seleccionadas
    for col in inputs + outputs:
        if col not in df.columns:
            issues.append(f"La columna '{col}' no existe en el DataFrame.")
            # No seguimos validando esa columna si no existe
    if issues:
        return issues

    # 2) Para cada columna de inputs/outputs, chequear nulos, tipo y positividad
    for col in inputs + outputs:
        # 2.1) Nulos
        if df[col].isnull().any():
            issues.append(f"Columna '{col}' contiene valores nulos.")
        # 2.2) Tipo numérico
        if not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"Columna '{col}' no es numérica.")
        else:
            # 2.3) Todos > 0
            if (df[col] <= 0).any():
                issues.append(f"Columna '{col}' contiene valores ≤ 0; DEA requiere positivos.")

    return issues

# ---------- consulta al LLM ----------
def _llm_suggest(df_head: str, inputs: list[str], outputs: list[str]) -> dict:
    """
    Envía un prompt al LLM para que valide la idoneidad de inputs/outputs. 
    Si la respuesta no es JSON válido, devuelve {'ready': True, 'raw': <texto>}.
    """
    prompt = (
        "Eres un experto en DEA. Evalúa si las columnas INPUTS y OUTPUTS "
        "son adecuadas y sugiere mejoras.\n\n"
        f"HEAD:\n{df_head}\n\n"
        f"INPUTS: {inputs}\nOUTPUTS: {outputs}\n\n"
        "Responde únicamente en formato JSON, con llaves: "
        "'ready' (bool), 'issues' (lista), 'suggested_fixes' (lista)."
    )
    try:
        chat = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = chat.choices[0].message.content
        # Intentar parsear como JSON
        return json.loads(content)
    except Exception:
        # Si no pudo parsear o falló la llamada, asumimos que el LLM no bloquea
        return {"ready": True, "issues": [], "raw": content if 'content' in locals() else ""}

# ---------- API pública ----------
def validate(df: pd.DataFrame, inputs: list[str], outputs: list[str]) -> dict:
    """
    Ejecuta validaciones formales y consulta al LLM:
      - _formal_checks: revisa solo inputs y outputs.
      - _llm_suggest: envía el head a GPT y trata fallos de parseo.
    Retorna:
      {
        "formal_issues": [str, ...],
        "llm": { "ready": bool, "issues": [str,...], "suggested_fixes": [str,...], "raw": str? }
      }
    """
    # 1) Validación formal de inputs/outputs
    formal_issues = _formal_checks(df, inputs, outputs)

    # 2) Preparar HEAD para LLM (solo primeras filas, en JSON)
    try:
        df_head = df.head().to_json()
    except Exception:
        df_head = "{}"  # fallback si no puede serializar

    # 3) Llamada al LLM
    llm_json = _llm_suggest(df_head, inputs, outputs)

    # 4) Si el JSON no contiene 'ready', forzamos ready=True
    if "ready" not in llm_json:
        llm_json["ready"] = True
        if "issues" not in llm_json:
            llm_json["issues"] = []

    return {"formal_issues": formal_issues, "llm": llm_json}
