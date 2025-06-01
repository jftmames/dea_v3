# ---------- src/inquiry_engine.py ----------
import os, json
from openai import OpenAI
import plotly.graph_objects as go

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---- tool schema (minimal y siempre válido) ----
FUNCTION_SPEC = {
    "name": "return_tree",
    "description": "Devuelve un árbol jerárquico de subpreguntas DEA.",
    "parameters": {        # sin validación estricta = no falla
        "type": "object",
        "properties": {},          # acepta cualquier clave
        "additionalProperties": True,
    },
}

# ---- util para treemap ----
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


# ---- generador principal ----
def generate_inquiry(root_question: str, depth: int = 2, breadth: int = 4) -> dict:
    """Genera subpreguntas con function-calling y devuelve dict JSON."""
    user_prompt = (
        f"Pregunta raíz: {root_question}\n"
        f"Genera un árbol con máximo {depth} niveles y {breadth} subpreguntas por nodo."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": user_prompt}],
        tools=[{"type": "function", "function": FUNCTION_SPEC}],
        tool_choice="auto",
        temperature=0,
    )

    args_json = response.choices[0].message.tool_calls[0].function.arguments
    return json.loads(args_json)   # siempre JSON válido
