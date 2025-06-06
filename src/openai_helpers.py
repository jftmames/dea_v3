import os
import json
import pandas as pd
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ... (las funciones existentes como chat_completion y generate_analysis_proposals se mantienen igual) ...

def explain_inquiry_tree(tree: dict) -> dict:
    """
    Usa un LLM para generar una explicación en lenguaje natural de un árbol de indagación.
    """
    # Pre-procesar el árbol para que sea más legible en el prompt
    try:
        tree_str = json.dumps(tree, indent=2, ensure_ascii=False)
    except TypeError:
        # Fallback por si algún tipo de dato sigue sin ser serializable
        tree_str = str(tree)


    prompt = (
        "Eres un consultor de gestión y experto en análisis de datos. Has generado el siguiente árbol de hipótesis (en formato JSON) para ayudar a un usuario a entender las causas de la ineficiencia en su organización. Tu tarea es explicar este mapa de razonamiento de una forma clara y accionable.\n\n"
        f"ÁRBOL DE HIPÓTESIS:\n```json\n{tree_str}\n```\n\n"
        "Por favor, redacta una explicación que cubra los siguientes puntos:\n"
        "1. **Propósito del Mapa:** Explica brevemente qué es este mapa y cómo debe usarlo el usuario (como una guía para investigar, no como una conclusión definitiva).\n"
        "2. **Análisis de la Pregunta Raíz:** Identifica la pregunta principal que el mapa intenta responder y por qué es importante.\n"
        "3. **Desglose de las Hipótesis Principales:** Describe las 2-4 ramas principales que nacen de la raíz. Explica qué significa cada una en términos de negocio.\n"
        "4. **Relación y Consecuencias:** Explica la lógica jerárquica: cómo los 'nodos hijos' son sub-causas o formas de investigar los 'nodos padres'. Explica las consecuencias de encontrar evidencia en una de las ramas (ej. 'Si la hipótesis sobre el exceso de personal es correcta, la consecuencia es que se podrían reasignar recursos o reducir plantilla').\n\n"
        "Usa un lenguaje claro, directo y orientado a la acción. Utiliza formato Markdown con negritas para resaltar los puntos clave."
    )
    try:
        # Usamos una temperatura ligeramente más alta para una explicación más rica
        resp = chat_completion(prompt, temperature=0.4)
        text = resp.choices[0].message.content
        return {"text": text}
    except Exception as e:
        return {"error": str(e), "text": "No se pudo generar la explicación del mapa de razonamiento."}
