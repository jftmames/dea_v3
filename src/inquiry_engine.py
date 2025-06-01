# ---------- src/inquiry_engine.py ----------
import os
import json
from typing import Any, Dict, Optional

from openai import OpenAI
import plotly.graph_objects as go

# ------------------------------------------------------------------
# 0. Cliente OpenAI
# ------------------------------------------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------------------------------------------------
# 1. Esquema mínimo para function-calling (siempre válido)
# ------------------------------------------------------------------
FUNCTION_SPEC = {
    "name": "return_tree",
    "description": "Devuelve un árbol jerárquico de subpreguntas DEA.",
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

# ------------------------------------------------------------------
# 2. Utilidad: convertir dict en Treemap Plotly
# ------------------------------------------------------------------
def to_plotly_tree(tree: Dict[str, Any]):
    labels, parents = [], []

    def walk(node, parent=""):
        for q, kids in node.items():
            labels.append(q)
            parents.append(parent)
            if isinstance(kids, dict):
                walk(kids, q)

    walk(tree)
    return go.Figure(go.Treemap(labels=labels, parents=parents, branchvalues="total"))

# ------------------------------------------------------------------
# 3. Generador principal (contexto opcional + triple robustez)
# ------------------------------------------------------------------
def generate_inquiry(
    root_question: str,
    context: Optional[Dict[str, Any]] = None,
    depth: int = 2,
    breadth: int = 4,
) -> Dict[str, Any]:
    """
    Devuelve un árbol de subpreguntas:
      1) intenta function-calling (JSON garantizado)
      2) si el modelo no llama a la función, parsea JSON del texto
      3) si todo falla o queda vacío, devuelve placeholder
    """
    # ---- prompt con contexto opcional ----
    ctx_str = f"Contexto JSON:\n{json.dumps(context, indent=2)}\n\n" if context else ""
    user_prompt = (
        ctx_str
        + f"Pregunta raíz: {root_question}\n"
        f"Crea un árbol con ≤{depth} niveles y ≤{breadth} subpreguntas por nodo."
    )

    # ---- 1) Function-calling ----
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": user_prompt}],
        tools=[{"type": "function", "function": FUNCTION_SPEC}],
        tool_choice="auto",
        temperature=0,
    )

    # a) Si llamó a la función → parsear arguments
    if resp.choices[0].message.tool_calls:
        args_json = resp.choices[0].message.tool_calls[0].function.arguments
        try:
            data = json.loads(args_json)
            tree = data.get("tree", data)
            if tree:
                return tree
        except json.JSONDecodeError:
            pass  # continuará al fallback de texto

    # b) Fallback: intentar JSON en contenido textual
    raw = resp.choices[0].message.content.strip()
    try:
        tree = json.loads(raw)
        if tree:
            return tree
    except Exception:
        pass

    # ---- 3) Placeholder si todo falla ----
    return _placeholder_tree(root_question)

# ------------------------------------------------------------------
# 4. Placeholder para no romper la app
# ------------------------------------------------------------------
def _placeholder_tree(root_q: str) -> Dict[str, Any]:
    return {root_q: {"ℹ️": "No se pudo generar subpreguntas"}}
