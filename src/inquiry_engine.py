import re, json, ast
from openai import OpenAI
import plotly.graph_objects as go
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _clean_json(s: str) -> str:
    """Extrae la primera llave/objeto JSON del texto"""
    # quita triple backticks
    s = re.sub(r"```.*?```", "", s, flags=re.S)
    # recorta hasta la primera y última llave
    first, last = s.find("{"), s.rfind("}")
    return s[first:last + 1] if first != -1 else s

def generate_inquiry(root_question: str, depth: int = 2, breadth: int = 4):
    prompt = (
        "Eres analista DEA; genera un árbol de subpreguntas en JSON puro "
        f"(máx {depth} niveles, {breadth} subpreguntas por nodo).\n"
        f"Pregunta raíz: {root_question}\n"
        "No devuelvas texto extra ni markdown."
    )
    chat = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    raw = chat.choices[0].message.content.strip()
    cleaned = _clean_json(raw)

    # 1) intento JSON estándar
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 2) intento eval literal
    try:
        return ast.literal_eval(cleaned)
    except Exception:
        pass

    # 3) fallback: devuelve diccionario plano con el texto para no romper la app
    return {root_question: {"ERROR": "No se pudo parsear respuesta LLM", "raw": raw}}


def to_plotly_tree(tree: dict):
    labels, parents = [], []
    def walk(node, parent=""):
        for q, kids in node.items():
            labels.append(q)
            parents.append(parent)
            if isinstance(kids, dict):
                walk(kids, q)
    walk(tree)
    return go.Figure(go.Treemap(labels=labels, parents=parents, branchvalues="total"))
