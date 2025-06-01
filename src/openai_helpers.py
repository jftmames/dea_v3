import os
from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def chat_completion(prompt: str, tools: list | None = None, temperature: float = 0.5) -> dict:
    """
    Llamada genérica a OpenAI Chat Completion.
    Retorna el dict completo de respuesta.
    """
    params = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if tools:
        params["tools"] = tools
        params["tool_choice"] = "auto"
    return client.chat.completions.create(**params)


def explain_orientation(inputs: list[str], outputs: list[str], orientation: str) -> dict:
    """
    Sugiere si la orientación ('input' o 'output') es adecuada, dadas las columnas seleccionadas.
    """
    prompt = (
        f"Tengo {len(inputs)} variables de insumo: {inputs} y {len(outputs)} variables de producto: {outputs}.\n"
        f"¿Es apropiado un modelo DEA orientado a '{orientation}'? "
        "Explica brevemente en qué casos es mejor input-oriented u output-oriented."
    )
    try:
        resp = chat_completion(prompt, temperature=0.3)
        text = resp.choices[0].message.content
        return {"text": text}
    except Exception as e:
        return {"error": str(e)}


def recommend_alternatives(df_columns: list[str], inputs: list[str], outputs: list[str]) -> dict:
    """
    Recomienda columnas alternativas para inputs/outputs si el validador detecta problemas.
    """
    prompt = (
        f"Columnas disponibles: {df_columns}\n"
        f"Has seleccionado inputs = {inputs} y outputs = {outputs} para un modelo DEA.\n"
        "Sugiere alternativas de columnas que puedan mejorar el modelo "
        "en caso de que haya colinealidad o variables redundantes. "
        "Devuelve un JSON con llaves 'recommend_inputs' y 'recommend_outputs', "
        "cada una con una lista de nombres posibles."
    )
    try:
        resp = chat_completion(prompt, temperature=0.3)
        text = resp.choices[0].message.content
        # Intentamos cargar JSON
        import json as _json
        data = _json.loads(text)
        return data
    except Exception:
        # Si falla parseo, devolvemos el texto sin procesar
        return {"text": resp.choices[0].message.content}
