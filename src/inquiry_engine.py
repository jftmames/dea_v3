# src/inquiry_engine.py

import os
import json
import re
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from openai import OpenAI
import plotly.graph_objects as go

# Corregido: Importación directa del módulo data_validator
from data_validator import _llm_suggest

# Inicializar cliente OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---- tool schema para generación de árbol de subpreguntas ----
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

# ---- Utilidad Plotly para visualizar el árbol (CON TÍTULO) ----
def to_plotly_tree(tree: Dict[str, Any], title: str = "Visualización del Árbol") -> go.Figure:
    """
    Convierte un diccionario anidado (tree) en un objeto Treemap de Plotly.
    Ahora acepta un parámetro 'title' para el gráfico.
    """
    labels, parents = [], []
    root_node_name = list(tree.keys())[0] if tree else "Raíz"

    def walk(node: Dict[str, Any], parent: str):
        for pregunta, hijos in node.items():
            labels.append(pregunta)
            parents.append(parent)
            if isinstance(hijos, dict):
                walk(hijos, pregunta)

    # Iniciar el recorrido desde el nodo raíz
    walk(tree, "")
    
    fig = go.Figure(go.Treemap(
        labels=labels, 
        parents=parents, 
        root_color="lightgrey"
    ))
    fig.update_layout(
        title_text=title,
        title_x=0.5,
        margin=dict(t=50, l=25, r=25, b=25)
    )
    return fig

# ---- Generador de subpreguntas con fallback triple ----
def generate_inquiry(
    root_question: str,
    context: Optional[Dict[str, Any]] = None,
    depth: int = 3,
    breadth: int = 5,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """
    Devuelve un árbol de subpreguntas basado en DEA para root_question.
    1) Intenta function-calling para devolver JSON válido.
    2) Si falla, parsea el contenido textual como JSON.
    3) Si aún falla o es vacío, devuelve un árbol placeholder.
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

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": user_prompt}],
            tools=[{"type": "function", "function": FUNCTION_SPEC}],
            tool_choice="auto",
            temperature=temperature,
        )

        # a) Intentar function-calling
        if resp.choices[0].message.tool_calls:
            args = resp.choices[0].message.tool_calls[0].function.arguments
            try:
                data = json.loads(args)
                tree = data.get("tree", data)
                if _tree_is_valid(tree):
                    return tree
            except json.JSONDecodeError:
                pass

        # b) Intentar parsear texto como JSON
        raw_content = resp.choices[0].message.content or ""
        raw = raw_content.strip()
        if raw:
            try:
                tree = json.loads(raw)
                if _tree_is_valid(tree):
                    return tree
            except Exception:
                pass
    except Exception:
        # Si la API de OpenAI falla, ir directamente al fallback
        pass

    # c) Fallback enriquecido
    return _fallback_tree(root_question)

# ---- Validar estructura del árbol ----
def _tree_is_valid(tree: Dict[str, Any]) -> bool:
    """
    Verifica que el árbol no sea placeholder y tenga al menos un nivel de anidación.
    """
    if not isinstance(tree, dict) or not tree:
        return False
    
    first_level_values = tree.values()
    # Es válido si al menos uno de sus hijos es también un diccionario (tiene sub-ramas)
    return any(isinstance(value, dict) for value in first_level_values)


# ---- Árbol placeholder por defecto ----
def _fallback_tree(root_q: str) -> Dict[str, Any]:
    return {
        root_q: {
            "¿Exceso de inputs?": {
                "¿Qué input consume más que los peers?": "Revisar uso de recursos vs. eficientes.",
                "¿Se puede reducir un 10% sin afectar output?": "Considerar ajustes en escala.",
            },
            "¿Déficit de outputs?": {
                "¿Output clave por debajo de la media eficiente?": "Identificar productos con bajo rendimiento.",
                "¿Implementar benchmarking con DMU eficiente?": "Analizar mejores prácticas de peers.",
            },
        }
    }

# ---- Nueva función: sugerir rango típico para un insumo ----
def suggest_input_range(
    df: pd.DataFrame,
    input_name: str
) -> Optional[Tuple[float, float, str]]:
    """
    Usa RAG para sugerir un rango [min–max] típico para un insumo.
    """
    df_head_json = df.head().to_json(orient="records")
    res = _llm_suggest(df_head_json, [input_name], [])

    if not res or "suggested_fixes" not in res:
        return None

    suggestion_text = res["suggested_fixes"][0] if res["suggested_fixes"] else ""

    match = re.search(r"\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]", suggestion_text)
    if match:
        try:
            min_sug, max_sug = float(match.group(1)), float(match.group(2))
            return min_sug, max_sug, suggestion_text
        except ValueError:
            pass
    
    return None
