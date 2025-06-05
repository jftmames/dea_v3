# jftmames/-dea-deliberativo-mvp/-dea-deliberativo-mvp-b44b8238c978ae0314af30717b9399634d28f8f9/src/inquiry_engine.py
import os
import json
import time
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
    max_retries: int = 2
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Genera un árbol de preguntas usando el LLM, con un bucle de reintentos para mayor fiabilidad.
    """
    ctx_str = json.dumps(context, indent=2) if context else "{}"
    
    base_prompt = (
        "Eres un experto mundial en Análisis Envolvente de Datos (DEA). "
        "Tu tarea es generar un árbol de hipótesis en formato JSON sobre las causas de la ineficiencia, basándote en un contexto.\n\n"
        f"--- CONTEXTO ---\n{ctx_str}\n\n"
        f"--- PREGUNTA RAÍZ ---\n{root_question}\n\n"
        "--- INSTRUCCIONES ESTRICTAS ---\n"
        "1. Descompón la pregunta en un árbol jerárquico de hipótesis con 2-3 niveles.\n"
        "2. Tu única salida DEBE SER una llamada a la función `return_tree` con el objeto JSON del árbol.\n"
        "3. NO escribas ningún texto o explicación. Solo llama a la función `return_tree`."
    )

    for attempt in range(max_retries + 1):
        prompt = base_prompt
        if attempt > 0:
            prompt += "\n\n--- AVISO ---\nTu intento anterior no produjo el formato correcto. Es crucial que tu única respuesta sea la llamada a la función `return_tree` con el JSON. No generes texto."
            print(f"--- DEBUG: Reintentando la llamada al motor de indagación (Intento {attempt + 1}) ---")
        
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                tools=[{"type": "function", "function": FUNCTION_SPEC}],
                tool_choice={"type": "function", "function": {"name": "return_tree"}},
                temperature=0.4 + (attempt * 0.1), # Aumenta la temperatura en cada reintento
            )
            
            if resp.choices[0].message.tool_calls:
                args = resp.choices[0].message.tool_calls[0].function.arguments
                tree = json.loads(args).get("tree", {})
                if tree: 
                    print("--- DEBUG: Árbol de indagación generado por la IA con éxito. ---")
                    return {root_question: tree}, None # Éxito

        except Exception as e:
            print(f"--- DEBUG: Error en la API de OpenAI en el intento {attempt + 1}: {e} ---")
            if attempt >= max_retries:
                return None, f"Fallo la conexión con la API de OpenAI tras {max_retries+1} intentos. Detalle: {e}"
            time.sleep(1) # Esperar 1 segundo antes de reintentar

    # Si todos los reintentos fallan, entonces usamos el árbol de respaldo
    print("--- DEBUG: La IA no devolvió un árbol válido tras todos los reintentos. Usando árbol de respaldo. ---")
    return _fallback_tree(root_question), "La IA no devolvió un árbol válido después de varios intentos. Usando árbol de respaldo."


def _fallback_tree(root_q: str) -> Dict[str, Any]:
    """Árbol de respaldo si la llamada a la IA falla."""
    return {
        root_q: {
            "¿Posible exceso de inputs?": {"Analizar variable por variable": {}},
            "¿Posible déficit de outputs?": {"Comparar con la media de los eficientes": {}},
            "¿Combinación ineficiente de factores?": {"Buscar outliers": {}},
        }
    }
