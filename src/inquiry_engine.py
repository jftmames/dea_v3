# ---------- src/inquiry_engine.py ----------
import os
import json
from typing import Any, Dict, Optional

from openai import OpenAI
import plotly.graph_objects as go

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---- tool schema, minimal y siempre válido ----
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

# ---- util Plotly ----
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

# ---- generador con contexto y triple-fallback ----
def generate_inquiry(
    root_question: str,
    context: Optional[Dict[str, Any]] = None,
    depth: int = 3,
    breadth: int = 5,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """
    Devuelve un árbol siempre no vacío:
      1) function-calling            → JSON garantía
      2) parse JSON del texto        → si no se invoca la función
      3) placeholder enriquecido     → si todo falla / vacío
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

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": user_prompt}],
        tools=[{"type": "function", "function": FUNCTION_SPEC}],
        tool_choice="auto",
        temperature=temperature,
    )

    # a) intentamos function-calling
    if resp.choices[0].message.tool_calls:
        args = resp.choices[0].message.tool_calls[0].function.arguments
        try:
            data = json.loads(args)
            tree = data.get("tree", data)
            if _tree_is_valid(tree):
                return tree
        except json.JSONDecodeError:
            pass

    # b) intentamos parsear texto (JSON plano)
    raw_content = resp.choices[0].message.content
    raw = (raw_content or "").strip()           # <-- protege cuando es None
    try:
        tree = json.loads(raw)
        if _tree_is_valid(tree):
            return tree
    except Exception:
        pass

    # c) placeholder enriquecido
    return _fallback_tree(root_question)

# ---- helpers ----
def _tree_is_valid(tree: Dict[str, Any]) -> bool:
    if not tree:
        return False
    first_key = next(iter(tree))
    # Árbol válido si no es placeholder y hay ≥2 subnodos o subniveles
    return not ("ℹ️" in tree.get(first_key, {})) and (
        len(tree[first_key]) >= 2
        or any(isinstance(v, dict) for v in tree[first_key].values())
    )

def _fallback_tree(root_q: str) -> Dict[str, Any]:
    return {
        root_q: {
            "¿Exceso de inputs?": {
                "¿Qué input consume más que los peers?": {},
                "¿Se puede reducir un 10 % sin afectar output?": {},
            },
            "¿Déficit de outputs?": {
                "¿Output clave por debajo de la media eficiente?": {},
                "¿Implementar benchmarking con DMU eficiente?": {},
            },
        }
    }
