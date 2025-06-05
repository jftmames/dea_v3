# src/inquiry_engine.py
import os
import json
from typing import Any, Dict, Optional, Tuple
from openai import OpenAI
import plotly.graph_objects as go

# Inicializar cliente OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

FUNCTION_SPEC = {
    "name": "return_tree",
    "description": "Árbol jerárquico de subpreguntas para análisis DEA.",
    "parameters": {
        "type": "object",
        "properties": {"tree": {"type": "object"}},
        "required": ["tree"],
    },
}

def to_plotly_tree(tree: Dict[str, Any], title: str = "Árbol de Indagación") -> go.Figure:
    """Convierte un diccionario anidado en un Treemap de Plotly."""
    labels, parents = [], []
    if not tree or not isinstance(tree, dict): return go.Figure()
    
    def walk(node: Dict[str, Any], parent: str):
        for pregunta, hijos in node.items():
            labels.append(pregunta)
            parents.append(parent)
            if isinstance(hijos, dict): walk(hijos, pregunta)
            
    walk(tree, "")
    fig = go.Figure(go.Treemap(labels=labels, parents=parents, root_color="lightgrey"))
    fig.update_layout(title_text=title, title_x=0.5)
    return fig

def generate_inquiry(
    root_question: str,
    context: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Devuelve una tupla (arbol_de_preguntas, mensaje_de_error).
    Si la operación es exitosa, mensaje_de_error es None.
    Si falla, se devuelve un mensaje de error detallado.
    """
    ctx = f"Contexto: {json.dumps(context)}\n\n" if context else ""
    prompt = (
        ctx + f"{root_question}\n\nGenera un árbol JSON con 2-3 hipótesis sobre las causas de la ineficiencia, "
        "basadas en las variables del contexto. Cada hipótesis debe ser una clave del JSON."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            tools=[{"type": "function", "function": FUNCTION_SPEC}],
            tool_choice="auto",
            temperature=0.3,
        )
        if resp.choices[0].message.tool_calls:
            args = resp.choices[0].message.tool_calls[0].function.arguments
            tree = json.loads(args).get("tree", {})
            if tree: 
                return tree, None # Éxito
        
        # Si la IA no devuelve una llamada a la función o el árbol está vacío
        return _fallback_tree(root_question), "La respuesta de la IA fue inválida, usando árbol de respaldo."

    except Exception as e:
        # Captura cualquier error de la API (clave, fondos, red, etc.) y lo devuelve
        return None, f"Fallo en la conexión con la API de OpenAI. Revisa tu clave, fondos y el estado del servicio. Detalle: {e}"

def _fallback_tree(root_q: str) -> Dict[str, Any]:
    """Árbol de respaldo si la llamada a la IA falla."""
    return {
        root_q: {
            "¿Posible exceso de inputs?": {"Analizar variable por variable": "Revisar datos"},
            "¿Posible déficit de outputs?": {"Comparar con la media": "Revisar datos"},
        }
    }
