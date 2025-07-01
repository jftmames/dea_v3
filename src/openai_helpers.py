# /src/openai_helpers.py
# --- VERSIÓN CORREGIDA Y CENTRALIZADA ---

import os
import json
import openai
import pandas as pd
from inquiry_engine import InquiryNode  # Asumimos que InquiryNode está en este nivel

# --- Cliente de OpenAI ---
def get_openai_client():
    """Inicializa y devuelve el cliente de OpenAI, manejando errores de API Key."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # En una app real, esto podría mostrar un error en la UI
        print("Error: La clave de API de OpenAI no ha sido configurada.")
        return None
    try:
        return openai.OpenAI(api_key=api_key)
    except Exception as e:
        print(f"Error al inicializar el cliente de OpenAI: {e}")
        return None

CLIENT = get_openai_client()

# --- Funciones de Llamada a la API ---

def chat_completion(prompt: str, use_json_mode: bool = False):
    """Función genérica para realizar una llamada de chat completion a la API de OpenAI."""
    if not CLIENT:
        return {"error": "Cliente de OpenAI no inicializado.", "raw_content": ""}

    params = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4
    }
    if use_json_mode:
        params["response_format"] = {"type": "json_object"}

    try:
        response = CLIENT.chat.completions.create(**params)
        return response.choices[0].message.content
    except Exception as e:
        return {"error": f"Error en la llamada a la API de OpenAI: {str(e)}"}

def generate_analysis_proposals(df_columns: list[str], df_head: pd.DataFrame):
    """
    MOVIMOS ESTA FUNCIÓN AQUÍ: Usa la IA para generar propuestas de análisis DEA.
    """
    prompt = (
        "Eres un consultor experto en Data Envelopment Analysis (DEA). Has recibido un conjunto de datos con las siguientes columnas: "
        f"{df_columns}. A continuación se muestran las primeras filas:\n\n{df_head.to_string()}\n\n"
        "Tu tarea es proponer entre 2 y 4 modelos de análisis DEA distintos y bien fundamentados. "
        "Para cada propuesta, proporciona un título, un breve razonamiento sobre su utilidad y las listas de inputs y outputs sugeridas.\n\n"
        "Devuelve únicamente un objeto JSON válido con una sola clave raíz 'proposals'. El valor de 'proposals' debe ser una lista de objetos, donde cada objeto representa una propuesta y contiene las claves 'title', 'reasoning', 'inputs' y 'outputs'."
    )
    raw_response = chat_completion(prompt, use_json_mode=True)

    if isinstance(raw_response, dict) and 'error' in raw_response:
        return {"error": raw_response['error'], "raw_content": ""}
    
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        return {"error": "La IA no devolvió un JSON válido.", "raw_content": raw_response}

def explain_inquiry_tree(tree_node: InquiryNode):
    """
    Usa la IA para generar una explicación en lenguaje natural de un árbol de auditoría.
    Adaptado para recibir un objeto InquiryNode.
    """
    # Función para convertir el nodo a un formato de texto simple para el prompt
    def node_to_text(node: InquiryNode, level=0):
        indent = "  " * level
        text = f"{indent}- {node.question}\n"
        for child in node.children:
            text += node_to_text(child, level + 1)
        return text

    tree_text = node_to_text(tree_node)
    
    prompt = (
        "Eres un experto metodólogo en econometría. El siguiente es un árbol de auditoría metodológica para un análisis DEA. "
        "Tu tarea es explicar en 2 o 3 párrafos cuál es el propósito de este árbol, qué áreas clave está evaluando y por qué esta estructura de preguntas es útil para asegurar un análisis robusto.\n\n"
        f"--- ÁRBOL DE AUDITORÍA ---\n{tree_text}\n\n"
        "--- EXPLICACIÓN ---\n"
    )
    
    explanation = chat_completion(prompt, use_json_mode=False)
    
    if isinstance(explanation, dict) and 'error' in explanation:
        return {"error": explanation['error'], "text": ""}
        
    return {"text": explanation}
