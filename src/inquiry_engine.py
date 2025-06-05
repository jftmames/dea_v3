# jftmames/-dea-deliberativo-mvp/-dea-deliberativo-mvp-b44b8238c978ae0314af30717b9399634d28f8f9/src/inquiry_engine.py
import os
import json
from typing import Any, Dict, Optional, Tuple
from openai import OpenAI
import plotly.graph_objects as go

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

FUNCTION_SPEC = {
    "name": "return_tree",
    "description": "Devuelve un árbol jerárquico de preguntas sobre las causas de la ineficiencia en un análisis DEA.",
    "parameters": {
        "type": "object",
        "properties": {"tree": {"type": "object", "description": "Objeto JSON anidado representando el árbol de preguntas."}},
        "required": ["tree"],
    },
}

def to_plotly_tree(tree: Dict[str, Any], title: str = "Árbol de Indagación") -> go.Figure:
    """Convierte un diccionario anidado en un Treemap de Plotly."""
    labels, parents = [], []
    if not tree or not isinstance(tree, dict): return go.Figure()
    
    root_label = list(tree.keys())[0]
    labels.append(root_label)
    parents.append("")

    def walk(node: Dict[str, Any], parent: str):
        for pregunta, hijos in node.items():
            if parent != "" and pregunta not in labels:
                 labels.append(pregunta)
                 parents.append(parent)

            if isinstance(hijos, dict):
                walk(hijos, pregunta)
            
    walk(tree, root_label)
    
    fig = go.Figure(go.Treemap(labels=labels, parents=parents, root_color="lightgrey"))
    fig.update_layout(title_text=title, title_x=0.5, margin=dict(t=50, l=25, r=25, b=25))
    return fig

def generate_inquiry(
    root_question: str,
    context: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Genera un árbol de preguntas usando el LLM.
    """
    ctx_str = json.dumps(context, indent=2) if context else "{}"
    
    # --- PROMPT MEJORADO: Instrucciones más estrictas ---
    prompt = (
        "Eres un experto mundial en Análisis Envolvente de Datos (DEA). "
        "Tu tarea es ayudar a un usuario a entender las causas de la ineficiencia en sus datos.\n\n"
        f"--- CONTEXTO DEL ANÁLISIS ---\n{ctx_str}\n\n"
        f"--- PREGUNTA CENTRAL ---\n{root_question}\n\n"
        "--- INSTRUCCIONES ESTRICTAS ---\n"
        "1. Debes descomponer la pregunta central en un árbol de hipótesis jerárquico.\n"
        "2. Tu única y exclusiva salida DEBE SER una llamada a la función `return_tree` con el objeto JSON que representa el árbol.\n"
        "3. NO escribas ningún texto, explicación o saludo. Tu única acción es llamar a la función con el JSON.\n"
        "4. El árbol debe tener 2-3 niveles. Las claves del primer nivel deben ser las hipótesis principales (ej. 'Uso excesivo de recursos'). Los niveles inferiores deben detallar estas hipótesis."
    )
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            tools=[{"type": "function", "function": FUNCTION_SPEC}],
            tool_choice={"type": "function", "function": {"name": "return_tree"}},
            temperature=0.4,
        )
        if resp.choices[0].message.tool_calls:
            args = resp.choices[0].message.tool_calls[0].function.arguments
            tree = json.loads(args).get("tree", {})
            if tree: 
                return {root_question: tree}, None # Éxito, se añade la raíz
        
        # Si la IA no devuelve una llamada a la función o el árbol está vacío
        return _fallback_tree(root_question), "La IA no devolvió un árbol válido. Usando árbol de respaldo."

    except Exception as e:
        return None, f"Fallo en la conexión con la API de OpenAI. Revisa tu clave. Detalle: {e}"

def _fallback_tree(root_q: str) -> Dict[str, Any]:
    """Árbol de respaldo si la llamada a la IA falla."""
    return {
        root_q: {
            "¿Posible exceso de inputs?": {"Analizar variable por variable": {}},
            "¿Posible déficit de outputs?": {"Comparar con la media de los eficientes": {}},
            "¿Combinación ineficiente de factores?": {"Buscar outliers": {}},
        }
    }
