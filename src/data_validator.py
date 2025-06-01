import os
import pandas as pd
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- reglas formales básicas ---
def _formal_checks(df: pd.DataFrame):
    issues = []
    if df.isnull().values.any():
        issues.append("Hay valores nulos.")
    if (df.select_dtypes(include="number") <= 0).any().any():
        issues.append("Existen números ≤ 0, no válidos para DEA.")
    return issues

# --- llamada LLM para sugerencias ---
def _llm_suggest(df_head: str, inputs: list[str], outputs: list[str]):
    prompt = (
        "Eres un experto en DEA. Evalúa si las columnas INPUTS y OUTPUTS "
        "son apropiadas y sugiere mejoras.\n\n"
        f"HEAD:\n{df_head}\n\n"
        f"INPUTS: {inputs}\nOUTPUTS: {outputs}\n\n"
        "Responde en JSON con 'ready', 'issues', 'suggested_fixes'."
    )
    chat_completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return chat_completion.choices[0].message.content

# --- API pública ---
def validate(df: pd.DataFrame, inputs: list[str], outputs: list[str]):
    formal_issues = _formal_checks(df)
    llm_json = _llm_suggest(df.head().to_json(), inputs, outputs)
    return {"formal_issues": formal_issues, "llm": llm_json}
