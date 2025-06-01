import os
import json
from openai import OpenAI
import plotly.graph_objects as go

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- LLM para generar árbol ----------
def generate_inquiry(root_question: str, depth: int = 2, breadth: int = 4):
    """
    Devuelve dict con nodos y subnodos.
    """
    prompt = (
        "Eres un analista DEA que formula complejos de indagación.\n"
        f"Pregunta raíz: {root_question}\n"
        f"Genera un árbol JSON de máx. {depth} niveles, {breadth} subpreguntas por nodo.\n"
        "Devuelve SOLO JSON."
    )
    chat = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return json.loads(chat.choices[0].message.content)

# ---------- util → Plotly Treemap ----------
def to_plotly_tree(tree: dict):
    labels, parents = [], []

    def _walk(node, parent_label=""):
        for q, children in node.items():
            labels.append(q)
            parents.append(parent_label)
            if isinstance(children, dict):
                _walk(children, q)

    _walk(tree)
    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        root_color="lightblue",
        branchvalues="total"
    ))
    fig.update_layout(margin=dict(t=40, l=0, r=0, b=0))
    return fig
