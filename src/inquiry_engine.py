# src/inquiry_engine.py

import os
import json
import re
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from openai import OpenAI
import plotly.graph_objects as go

# Corregido: Importación directa del módulo data_validator
from data_validator import _llm_suggest

# Inicializar cliente OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---- tool schema para generación de árbol de subpreguntas ----
FUNCTION_SPEC = {
    "name": "return_tree",
    "description": "Árbol jerárquico de subpreguntas DEA.",
    "parameters": {
        "type": "object",
        "properties": {
            "tree": {
                "type": "object",
                "description": "Nodo raíz con subnodos arbitrarios.",
                "additionalProperties": {"type": "object"},
            }
        },
        "required": ["tree"],
    },
}

# ---- Utilidad Plotly para visualizar el árbol (CON TÍTULO) ----
def to_plotly_tree(tree: Dict[str, Any], title: str = "Visualización del Árbol") -> go.Figure:
    """
    Convierte un diccionario anidado (tree) en un objeto Treemap de Plotly.
    Ahora acepta un parámetro 'title' para el gráfico.
    """
    labels, parents = [], []
    root_node_name = list(tree.keys())[0] if tree else "Raíz"

    def walk(node: Dict[str, Any], parent: str):
        for pregunta, hijos in node.items():
            labels.append(pregunta)
            parents.append(parent)
            if isinstance(hijos, dict):
                walk(hijos, pregunta)

    # Iniciar el recorrido desde el nodo raíz
    walk(tree, "")
    
    fig = go.Figure(go.Treemap(
        labels=labels, 
        parents=parents, 
        root_color="lightgrey"
    ))
    fig.update_layout(
        title_text=title,
        title_x=0.5,
        margin=dict(t=50, l=25, r=25, b=25)
    )
    return fig

# ---- Generador de subpreguntas con manejo de errores explícito ----
def generate_inquiry(
    root_question: str,
    context: Optional[Dict[str, Any]] = None,
    depth: int = 3,
    breadth: int = 5,
    temperature: float = 0.3,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Devuelve una tupla (árbol_de_preguntas, mensaje_de_error).
    Si la operación es exitosa, mensaje_de_error es None.
    Si falla, árbol_de_preguntas es None.
    """
    ctx = f"Contexto:\n{json.dumps(context, indent=2)}\n\n" if context else ""
    user_prompt = (
        ctx
        + f"{root_question}\n\n"
        "Genera un *árbol JSON* con al menos 2 causas principales de ineficiencia. "
        f"Máx {depth} niveles y {breadth} subpreguntas por nodo. "
        "Si los datos son incompletos, infiere hipótesis típicas: "
        "exceso de inputs, déficit de outputs, benchmarking, entorno regulatorio."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": user_prompt}],
            tools=[{"type": "function", "function": FUNCTION_SPEC}],
            tool_choice="auto",
            temperature=temperature,
        )

        # a) Intentar function-calling
        if resp.choices[0].message.tool_calls:
            args = resp.choices[0].message.tool_calls[0].function.arguments
            try:
                data = json.loads(args)
                tree = data.get("tree", data)
                if _tree_is_valid(tree):
                    return tree, None # Éxito
            except json.JSONDecodeError as e:
                return None, f"Error de OpenAI: La respuesta no es un JSON válido. ({e})"

        # b) Intentar parsear texto como JSON
        raw_content = resp.choices[0].message.content or ""
        if raw_content:
            try:
                tree = json.loads(raw_content)
                if _tree_is_valid(tree):
                    return tree, None # Éxito
            except Exception as e:
                return None, f"Error al procesar la respuesta de OpenAI. ({e})"

        # Si llegamos aquí, la respuesta de la IA fue vacía o inválida
        return _fallback_tree(root_question), "La respuesta de la IA fue vacía o inválida, usando árbol de respaldo."

    except Exception as e:
        # Si la API de OpenAI falla (ej. clave inválida, sin crédito, problema de red)
        return None, f"Fallo en la conexión con la API de OpenAI: {e}"

# ---- Validar estructura del árbol ----
def _tree_is_valid(tree: Dict[str, Any]) -> bool:
    """
    Verifica que el árbol no sea placeholder y tenga al menos un nivel de anidación.
    """
    if not isinstance(tree, dict) or not tree:
        return False
    
    first_level_values = tree.values()
    return any(isinstance(value, dict) for value in first_level_values)


# ---- Árbol placeholder por defecto ----
def _fallback_tree(root_q: str) -> Dict[str, Any]:
    return {
        root_q: {
            "¿Exceso de inputs?": {
                "¿Qué input consume más que los peers?": "Revisar uso de recursos vs. eficientes.",
                "¿Se puede reducir un 10% sin afectar output?": "Considerar ajustes en escala.",
            },
            "¿Déficit de outputs?": {
                "¿Output clave por debajo de la media eficiente?": "Identificar productos con bajo rendimiento.",
                "¿Implementar benchmarking con DMU eficiente?": "Analizar mejores prácticas de peers.",
            },
        }
    }
