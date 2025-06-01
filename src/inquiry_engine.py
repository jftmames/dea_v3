# src/inquiry_engine.py

import os
import json
import re
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from openai import OpenAI
import plotly.graph_objects as go

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

# ---- Utilidad Plotly para visualizar el árbol ----
def to_plotly_tree(tree: Dict[str, Any]) -> go.Figure:
    """
    Convierte un diccionario anidado (tree) en un objeto Treemap de Plotly.
    """
    labels, parents = [], []

    def walk(node: Dict[str, Any], parent: str = ""):
        for pregunta, hijos in node.items():
            labels.append(pregunta)
            parents.append(parent)
            if isinstance(hijos, dict):
                walk(hijos, pregunta)

    walk(tree)
    return go.Figure(go.Treemap(labels=labels, parents=parents, branchvalues="total"))

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
    try:
        tree = json.loads(raw)
        if _tree_is_valid(tree):
            return tree
    except Exception:
        pass

    # c) Fallback enriquecido
    return _fallback_tree(root_question)

# ---- Validar estructura del árbol ----
def _tree_is_valid(tree: Dict[str, Any]) -> bool:
    """
    Verifica que el árbol no sea placeholder y tenga al menos 2 subnodos.
    """
    if not tree:
        return False
    first_key = next(iter(tree))
    # Válido si no contiene el ícono de placeholder y hay ≥2 subnodos o más niveles
    is_placeholder = "ℹ️" in tree.get(first_key, {})
    tiene_dos_o_mas = len(tree[first_key]) >= 2
    tiene_nivel_extra = any(isinstance(v, dict) for v in tree[first_key].values())
    return not is_placeholder and (tiene_dos_o_mas or tiene_nivel_extra)

# ---- Árbol placeholder por defecto ----
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

# ---- Nueva función: sugerir rango típico para un insumo ----
def suggest_input_range(
    df: pd.DataFrame,
    input_name: str
) -> Optional[Tuple[float, float, str]]:
    """
    Usa RAG (Reinforcement with Retrieval) para sugerir un rango [min–max]
    típico para un insumo dado (input_name) en el DataFrame df.
    Retorna (min_sug, max_sug, citation) o None si no hay sugerencia.
    """
    # Convertir las primeras filas a JSON para contexto al LLM
    df_head_json = df.head().to_json(orient="records")

    # Invocar al LLM a través de data_validator._llm_suggest
    res = _llm_suggest(df_head_json, [input_name], [])

    if not res or "suggested_fixes" not in res:
        return None

    fixes = res["suggested_fixes"]
    if not fixes:
        return None

    # Tomar la primera sugerencia textual
    suggestion_text = fixes[0]

    # Intentar extraer un rango numérico en formato [min, max]
    match = re.search(r"\[\s*([0-9.+-eE]+)\s*,\s*([0-9.+-eE]+)\s*\]", suggestion_text)
    if match:
        try:
            min_sug = float(match.group(1))
            max_sug = float(match.group(2))
            return min_sug, max_sug, suggestion_text
        except ValueError:
            # Si no se parsea a float correctamente, seguir al fallback
            pass

    # Si no se encontró rango con corchetes, intentar extraer dos números cualesquiera
    nums = re.findall(r"([0-9]+(?:\.[0-9]+)?)", suggestion_text)
    if len(nums) >= 2:
        try:
            # Tomar los dos primeros como min y max
            min_sug = float(nums[0])
            max_sug = float(nums[1])
            return min_sug, max_sug, suggestion_text
        except ValueError:
            pass

    # Si no se pudo extraer rango, devolver toda la sugerencia como cita
    return None
