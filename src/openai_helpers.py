# src/openai_helpers.py

import os
import json
from typing import List, Optional, Dict, Any
from openai import OpenAI

# ---------------------------
# Cliente de OpenAI singleton
# ---------------------------
_client: Optional[OpenAI] = None

def _get_client() -> OpenAI:
    """
    Inicializa (si es necesario) y retorna el cliente de OpenAI usando la variable de entorno OPENAI_API_KEY.
    """
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


# -------------------------------------
# Función genérica para chat completions
# -------------------------------------
def chat_completion(
    prompt: str,
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.0
) -> Dict[str, Any]:
    """
    Envía un prompt a la API de chat completions de OpenAI.
    - Si se provee 'tools', intenta llamar a function-calling.
    - Si ocurre un error (por ejemplo BadRequest), reintenta sin tools.
    Retorna el dict completo de la respuesta.
    """
    client = _get_client()
    try:
        if tools:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                tools=tools,
                tool_choice="auto",
                temperature=temperature,
            )
        else:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
        return resp
    except Exception as e:
        # Si falla con tools, reintentar sin ellos
        if tools:
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                return resp
            except Exception:
                raise
        else:
            raise


# --------------------------------------------------
# Explica si la orientación elegida es apropiada
# --------------------------------------------------
def explain_orientation(
    inputs: List[str],
    outputs: List[str],
    orientation: str
) -> str:
    """
    Genera una explicación breve sobre si la orientación ('input' u 'output')
    es adecuada para las columnas suministradas.
    Retorna simplemente el texto de la IA.
    """
    prompt = (
        "Eres un experto en DEA. "
        f"Tienes estos insumos (inputs): {', '.join(inputs)}. "
        f"Tienes estos productos (outputs): {', '.join(outputs)}. "
        f"La orientación elegida es '{orientation}'. "
        "Explica brevemente si esa orientación es adecuada para analizar esas variables "
        "y sugiere en qué caso cambiaría."
    )
    resp = chat_completion(prompt, temperature=0.3)
    return resp.choices[0].message.content.strip()


# -------------------------------------------------------
# Recomienda sets alternativos de inputs y outputs
# -------------------------------------------------------
def recommend_alternatives(
    df_columns: List[str],
    inputs: List[str],
    outputs: List[str]
) -> Dict[str, Any]:
    """
    A partir de la lista de columnas numéricas disponibles en el dataset ('df_columns'),
    los inputs actuales y outputs actuales, sugiere hasta 3 combinaciones alternativas
    de inputs y 3 combinaciones alternativas de outputs. Devuelve un dict con dos llaves:
      - 'recommend_inputs': lista de listas de posibles inputs
      - 'recommend_outputs': lista de listas de posibles outputs
    Si la IA no devuelve JSON, retorna {'text': <texto completo>}
    """
    prompt = (
        "Eres un analista de DEA. Un usuario tiene estas columnas numéricas en su dataset: "
        f"{', '.join(df_columns)}. Actualmente usa como inputs: {', '.join(inputs)}; "
        f"y como outputs: {', '.join(outputs)}. "
        "Sugiere hasta 3 conjuntos alternativos de inputs y 3 conjuntos alternativos de outputs "
        "que podrían mejorar la validez del modelo, indicando brevemente por qué cada uno."
    )
    resp = chat_completion(prompt, temperature=0.5)
    text = resp.choices[0].message.content.strip()

    # Intentamos extraer JSON si la IA lo formatea así
    try:
        parsed = json.loads(text)
        return {
            "recommend_inputs": parsed.get("recommend_inputs", []),
            "recommend_outputs": parsed.get("recommend_outputs", [])
        }
    except json.JSONDecodeError:
        # Si no es JSON, devolvemos el texto entero bajo la llave 'text'
        return {"text": text}


# --------------------
# Ejemplo de prueba
# --------------------
if __name__ == "__main__":
    # Estas líneas no se ejecutarán al importarse, solo al hacer `python openai_helpers.py`
    all_cols = ["Coste", "Horas", "Ventas", "Clientes", "DMU"]
    ins = ["Coste", "Horas"]
    outs = ["Ventas"]
    print("=== Explicación de orientación ===")
    print(explain_orientation(ins, outs, "input"))
    print("\n=== Recomendaciones alternativas ===")
    print(recommend_alternatives(all_cols, ins, outs))
