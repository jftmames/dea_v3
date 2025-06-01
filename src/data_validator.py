import os
import json
import pandas as pd
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- reglas formales básicas ----------
def _formal_checks(df: pd.DataFrame):
    issues = []

    # valores nulos
    if df.isnull().values.any():
        issues.append("Se encontraron valores nulos.")

    # positivos
    if (df.select_dtypes(include="number") <= 0).any().any():
        issues.append("Hay números ≤ 0; DEA requiere positivos.")

    # columnas no numéricas
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"Columna '{col}' no es numérica.")

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
    # puede venir como string JSON; intentamos parsear
    try:
        return json.loads(chat.choices[0].message.content)
    except Exception:
        return {"raw": chat.choices[0].message.content}


# ---------- API pública ----------
def validate(df: pd.DataFrame, inputs: list[str], outputs: list[str]):
    formal_issues = _formal_checks(df)
    llm_json = _llm_suggest(df.head().to_json(), inputs, outputs)
    return {"formal_issues": formal_issues, "llm": llm_json}
