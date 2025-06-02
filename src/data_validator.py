# src/data_validator.py

import os
import json
import pandas as pd
from openai import OpenAI

# Importamos dea_models.utils.validate_positive_dataframe si ya existe,
# o el validador que hayas definido.
from dea_models.utils import validate_positive_dataframe

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- reglas formales básicas ----------
def _formal_checks(df: pd.DataFrame, inputs: list[str], outputs: list[str]):
    """
    Comprueba solo las columnas de 'inputs' y 'outputs' para:
      - que no tengan valores nulos,
      - que sean numéricas,
      - que los valores numéricos sean > 0.
    Devuelve lista de mensajes de error (vacía si todo OK).
    """
    issues = []

    # 1) Asegurarnos de que las columnas existan
    for col in inputs + outputs:
        if col not in df.columns:
            issues.append(f"La columna '{col}' no existe en el DataFrame.")
            # Si una columna ni siquiera existe, no seguimos validando su contenido
    # Si ya hay errores de columnas faltantes, devolvemos
    if issues:
        return issues

    # 2) Tomar solo las columnas de inputs y outputs para validarlas
    df_num = df[inputs + outputs]

    # 2.1) Valores nulos
    if df_num.isnull().values.any():
        issues.append("Se encontraron valores nulos en inputs/outputs.")

    # 2.2) Tipo numérico y > 0
    for col in inputs + outputs:
        # Verificar que el dtype sea numérico
        if not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"Columna '{col}' no es numérica.")
        else:
            # Verificar que todos los valores sean > 0
            if (df[col] <= 0).any():
                issues.append(f"Columna '{col}' contiene valores ≤ 0; DEA requiere estrictamente positivos.")

    return issues

# ---------- consulta al LLM ----------
def _llm_suggest(df_head: str, inputs: list[str], outputs: list[str]):
    prompt = (
        "Eres un experto en DEA. Evalúa si las columnas INPUTS y OUTPUTS "
        "son adecuadas y sugiere mejoras.\n\n"
        f"HEAD:\n{df_head}\n\n"
        f"INPUTS: {inputs}\nOUTPUTS: {outputs}\n\n"
        "Responde en JSON con 'ready', 'issues', 'suggested_fixes'."
    )
    chat = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    # Puede venir como string JSON; intentamos parsear
    try:
        return json.loads(chat.choices[0].message.content)
    except Exception:
        return {"raw": chat.choices[0].message.content}

# ---------- API pública ----------
def validate(df: pd.DataFrame, inputs: list[str], outputs: list[str]):
    """
    Ejecuta validaciones formales y consulta al LLM.
    1) __formal_checks: revisa solo las columnas de inputs y outputs.
    2) __llm_suggest: envía el head a GPT para que confirme la adecuación.
    Retorna dict con:
      - 'formal_issues': lista de mensajes (vacía si no hay errores).
      - 'llm': resultado (parsed JSON o {'raw': str}).
    """
    formal_issues = _formal_checks(df, inputs, outputs)
    # Serializar solo las primeras filas para enviar al LLM
    df_head = df.head().to_json()
    llm_json = _llm_suggest(df_head, inputs, outputs)
    return {"formal_issues": formal_issues, "llm": llm_json}
