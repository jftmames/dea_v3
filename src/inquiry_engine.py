import os, json
from openai import OpenAI
import plotly.graph_objects as go

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- función util para Plotly ----------
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


# ---------- Function-Calling ----------
FUNCTION_SPEC = {
    "name": "return_tree",
    "description": "Devuelve un árbol de subpreguntas anidadas (clave = pregunta, valor = subárbol).",
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



def generate_inquiry(root_question: str, depth: int = 2, breadth: int = 4) -> dict:
    # prompt solo describe la tarea; la salida vendrá vía tools
    user_prompt = (
        f"Pregunta raíz: {root_question}\n"
        f"Genera un árbol con máximo {depth} niveles y {breadth} subpreguntas por nodo."
    )

resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": user_prompt}],
    tools=[{"type": "function", "function": FUNCTION_SPEC}],
    tool_choice="auto",      # deja que el modelo invoque la función
    temperature=0,
)

args_json = resp.choices[0].message.tool_calls[0].function.arguments
tree_wrapper = json.loads(args_json)          # {'tree': {...}}
return tree_wrapper["tree"]

