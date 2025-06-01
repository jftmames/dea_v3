# ---------- src/inquiry_engine.py ----------
import os
import json
from openai import OpenAI
import plotly.graph_objects as go

# --- cliente OpenAI ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- tool schema (válido y mínimo) ---
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

# --- util Plotly ---
def to_plotly_tree(tree: dict):
    labels, parents = [], []

    def walk(node, parent=""):
        for q, kids in node.items():
            labels.append(q)
            parents.append(parent)
            if isinstance(kids, dict):
                walk(kids, q)

    walk(tree)
    return go.Figure(
        go.Treemap(labels=labels, parents=parents, branchvalues="total")
    )

# --- generador principal ---
def generate_inquiry(root_question: str, depth: int = 2, breadth: int = 4) -> dict:
    """Genera árbol con 3 capas de robustez: function-call → json → placeholder."""
    prompt = (
        f"Pregunta raíz: {root_question}\n"
        f"Crea un árbol con ≤{depth} niveles y ≤{breadth} subpreguntas por nodo."
    )

    # ---------- 1) intento function-calling ----------
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        tools=[{"type": "function", "function": FUNCTION_SPEC}],
        tool_choice="auto",
        temperature=0,
    )

    if resp.choices[0].message.tool_calls:
        args = resp.choices[0].message.tool_calls[0].function.arguments
        try:
            data = json.loads(args)
            return data.get("tree", data) or _placeholder_tree(root_question)
        except json.JSONDecodeError:
            pass  # caeremos al fallback de texto

    # ---------- 2) intento parsear texto tradicional ----------
    raw = resp.choices[0].message.content.strip()
    try:
        return json.loads(raw) or _placeholder_tree(root_question)
    except Exception:
        return _placeholder_tree(root_question)


def _placeholder_tree(root_q: str):
    """Devuelve un árbol mínimo para evitar que la app quede vacía."""
    return {root_q: {"ℹ️": "No se pudo generar subpreguntas"}}


