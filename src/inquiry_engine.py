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
    """Genera subpreguntas usando function-calling y devuelve dict JSON."""
    prompt = (
        f"Pregunta raíz: {root_question}\n"
        f"Genera un árbol con máximo {depth} niveles y {breadth} subpreguntas por nodo."
    )

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        tools=[{"type": "function", "function": FUNCTION_SPEC}],
        tool_choice="auto",
        temperature=0,
    )

    args = resp.choices[0].message.tool_calls[0].function.arguments
    tree_wrapper = json.loads(args)  # {'tree': {...}}
    return tree_wrapper["tree"]
