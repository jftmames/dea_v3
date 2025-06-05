# jftmames/-dea-deliberativo-mvp/-dea-deliberativo-mvp-b44b8238c978ae0314af30717b9399634d28f8f9/src/openai_helpers.py
import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat_completion(prompt: str, tools: list | None = None, use_json_mode: bool = False, temperature: float = 0.5) -> dict:
    """
    Llamada genérica a OpenAI Chat Completion, con soporte para modo JSON.
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
    if use_json_mode:
        params["response_format"] = {"type": "json_object"}
        
    return client.chat.completions.create(**params)


def explain_orientation(inputs: list[str], outputs: list[str], orientation: str) -> dict:
    """
    Sugiere si la orientación ('input' o 'output') es adecuada.
    """
    prompt = (
        f"Para un modelo DEA con {len(inputs)} insumos ({inputs}) y {len(outputs)} productos ({outputs}), "
        f"¿es apropiada una orientación a '{orientation}'? Explica brevemente cuándo es mejor cada orientación."
    )
    try:
        resp = chat_completion(prompt, temperature=0.3)
        text = resp.choices[0].message.content
        return {"text": text}
    except Exception as e:
        return {"error": str(e)}


def recommend_alternatives(df_columns: list[str], inputs: list[str], outputs: list[str]) -> dict:
    """
    Recomienda columnas alternativas para inputs/outputs usando el modo JSON.
    """
    prompt = (
        f"Columnas disponibles: {df_columns}\n"
        f"Has seleccionado inputs = {inputs} y outputs = {outputs} para un modelo DEA.\n"
        "Sugiere alternativas de columnas que puedan mejorar el modelo "
        "en caso de que haya colinealidad o variables redundantes. "
        "Devuelve únicamente un objeto JSON con dos claves: 'recommended_inputs' y 'recommended_outputs', "
        "cada una con una lista de nombres de columnas sugeridas."
    )
    try:
        resp = chat_completion(prompt, use_json_mode=True, temperature=0.3)
        content = resp.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        return {"error": str(e), "text": "No se pudieron generar recomendaciones."}
