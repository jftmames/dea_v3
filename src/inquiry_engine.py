import os
import json
from typing import Any, Dict, Optional, Tuple
from openai import OpenAI
import plotly.graph_objects as go

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
    """Devuelve una tupla (árbol, mensaje_de_error)."""
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
            if tree: return tree, None
        return _fallback_tree(root_question), "Respuesta de la IA inválida, usando árbol de respaldo."
    except Exception as e:
        return None, f"Fallo en la conexión con la API de OpenAI. Detalle: {e}"

def _fallback_tree(root_q: str) -> Dict[str, Any]:
    """Árbol de respaldo."""
    return {
        root_q: {
            "¿Posible exceso de inputs?": {},
            "¿Posible déficit de outputs?": {},
        }
    }
