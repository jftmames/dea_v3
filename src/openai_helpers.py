import os
import json
import pandas as pd
from openai import OpenAI

# Este cliente puede ser local si la función que lo usa es local.
# O se puede pasar como argumento. Para mantenerlo simple, lo definiremos en main.py también.

def explain_inquiry_tree(tree: dict) -> dict:
    """Usa un LLM para generar una explicación en lenguaje natural de un árbol de indagación."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    try:
        tree_str = json.dumps(tree, indent=2, ensure_ascii=False)
    except TypeError:
        tree_str = str(tree)

    prompt = (
        "Eres un consultor de gestión y experto en análisis de datos. Has generado el siguiente árbol de hipótesis (en formato JSON) para ayudar a un usuario a entender las causas de la ineficiencia en su organización. Tu tarea es explicar este mapa de razonamiento de una forma clara y accionable.\n\n"
        f"ÁRBOL DE HIPÓTESIS:\n```json\n{tree_str}\n```\n\n"
        "Por favor, redacta una explicación que cubra los siguientes puntos:\n"
        "1. **Propósito del Mapa:** Explica brevemente qué es este mapa y cómo debe usarlo el usuario.\n"
        "2. **Análisis de la Pregunta Raíz:** Identifica la pregunta principal que el mapa intenta responder.\n"
        "3. **Desglose de las Hipótesis Principales:** Describe las ramas principales.\n"
        "4. **Relación y Consecuencias:** Explica la lógica jerárquica.\n\n"
        "Usa un lenguaje claro, directo y orientado a la acción. Utiliza formato Markdown con negritas para resaltar los puntos clave."
    )
    try:
        # La función chat_completion ahora es local a esta función o se importa de otro sitio.
        # Por simplicidad, recreamos la llamada aquí.
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        text = resp.choices[0].message.content
        return {"text": text}
    except Exception as e:
        return {"error": str(e), "text": "No se pudo generar la explicación del mapa de razonamiento."}
