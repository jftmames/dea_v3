# src/inquiry_engine.py

import os
import json
import re
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from openai import OpenAI
import plotly.graph_objects as go

# Importación del módulo data_validator para reutilizar su función de LLM
try:
    from data_validator import _llm_suggest
except ImportError:
    # Fallback por si el fichero no está disponible o da error circular
    def _llm_suggest(*args, **kwargs):
        return {"ready": True, "issues": ["Error: data_validator no encontrado."], "raw": ""}

# Inicializar cliente OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---- Tool schema para la generación del árbol ----
FUNCTION_SPEC = {
    "name": "return_tree",
    "description": "Árbol jerárquico de subpreguntas para el análisis de eficiencia.",
    "parameters": {
        "type": "object",
        "properties": {
            "tree": {
                "type": "object",
                "description": "Nodo raíz con subnodos anidados que representan el árbol de preguntas.",
                "additionalProperties": {"type": "object"},
            }
        },
        "required": ["tree"],
    },
}

# ---- Visualización del árbol ----
def to_plotly_tree(tree: Dict[str, Any], title: str = "Visualización del Árbol") -> go.Figure:
    """Convierte un diccionario anidado en un objeto Treemap de Plotly."""
    labels, parents = [], []
    if not tree or not isinstance(tree, dict):
        return go.Figure()

    def walk(node: Dict[str, Any], parent: str):
        for pregunta, hijos in node.items():
            labels.append(pregunta)
            parents.append(parent)
            if isinstance(hijos, dict):
                walk(hijos, pregunta)

    walk(tree, "")
    
    fig = go.Figure(go.Treemap(labels=labels, parents=parents, root_color="lightgrey"))
    fig.update_layout(title_text=title, title_x=0.5, margin=dict(t=50, l=25, r=25, b=25))
    return fig

# ---- Generador de preguntas con manejo de errores explícito ----
def generate_inquiry(
    root_question: str,
    context: Optional[Dict[str, Any]] = None,
    depth: int = 3,
    breadth: int = 5,
    temperature: float = 0.3,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Devuelve una tupla (arbol_de_preguntas, mensaje_de_error).
    Si la operación es exitosa, mensaje_de_error es None.
    Si falla, arbol_de_preguntas es None y se devuelve un mensaje de error detallado.
    """
    ctx = f"Contexto:\n{json.dumps(context, indent=2)}\n\n" if context else ""
    user_prompt = (
        ctx
        + f"{root_question}\n\n"
        "Genera un *árbol JSON* con al menos 2 causas principales de ineficiencia. "
        f"Máximo {depth} niveles de profundidad y {breadth} subpreguntas por nodo. "
        "Basa las hipótesis en los nombres de las variables proporcionadas (inputs/outputs). "
        "Ejemplos de hipótesis: exceso de inputs, déficit de outputs, problemas de escala, factores del entorno."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": user_prompt}],
            tools=[{"type": "function", "function": FUNCTION_SPEC}],
            tool_choice="auto",
            temperature=temperature,
        )

        if resp.choices[0].message.tool_calls:
            args = resp.choices[0].message.tool_calls[0].function.arguments
            try:
                data = json.loads(args)
                tree = data.get("tree", data)
                if _tree_is_valid(tree):
                    return tree, None  # Éxito
            except json.JSONDecodeError as e:
                return None, f"La respuesta de la IA no tenía un formato JSON válido. Detalle: {e}"
        
        return _fallback_tree(root_question), "Respuesta de la IA inválida, usando árbol de respaldo."

    except Exception as e:
        # Captura cualquier error de la API (clave inválida, sin fondos, red, etc.)
        return None, f"Fallo en la conexión con la API de OpenAI. Por favor, revisa tu API Key, los fondos de tu cuenta y el estado del servicio de OpenAI. Detalle del error: {e}"

def _tree_is_valid(tree: Dict[str, Any]) -> bool:
    """Verifica que el árbol no sea un placeholder y tenga al menos un nivel de anidación."""
    if not isinstance(tree, dict) or not tree:
        return False
    first_level_values = tree.values()
    return any(isinstance(value, dict) for value in first_level_values)

def _fallback_tree(root_q: str) -> Dict[str, Any]:
    """Árbol de respaldo si la llamada a la IA falla."""
    return {
        root_q: {
            "¿Exceso de inputs?": {"Analizar variable por variable": "Revisar datos"},
            "¿Déficit de outputs?": {"Comparar con la media": "Revisar datos"},
        }
    }
