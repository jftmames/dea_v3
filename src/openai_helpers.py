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


def generate_analysis_proposals(df_columns: list[str], df_head: pd.DataFrame) -> dict:
    """
    Analiza las columnas de un DataFrame y propone varios modelos de análisis DEA.
    """
    prompt = (
        "Eres un consultor experto en Data Envelopment Analysis (DEA). Has recibido un conjunto de datos con las siguientes columnas: "
        f"{df_columns}. A continuación se muestran las primeras filas:\n\n{df_head.to_string()}\n\n"
        "Tu tarea es proponer entre 2 y 4 modelos de análisis DEA distintos y bien fundamentados que se podrían aplicar a estos datos. "
        "Para cada propuesta, proporciona un título, un breve razonamiento sobre su utilidad y las listas de inputs y outputs sugeridas.\n\n"
        "Devuelve únicamente un objeto JSON válido con una sola clave raíz 'proposals'. El valor de 'proposals' debe ser una lista de objetos, donde cada objeto representa una propuesta y contiene las claves 'title', 'reasoning', 'inputs' y 'outputs'."
    )
    try:
        resp = chat_completion(prompt, use_json_mode=True, temperature=0.5)
        content = resp.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        return {"error": str(e), "text": "No se pudieron generar las propuestas de análisis."}


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
