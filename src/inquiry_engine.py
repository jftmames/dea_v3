# jftmames/-dea-deliberativo-mvp/-dea-deliberativo-mvp-b44b8238c978ae0314af30717b9399634d28f8f9/src/inquiry_engine.py
import os
import json
import time
from typing import Any, Dict, Optional, Tuple
from openai import OpenAI
import plotly.graph_objects as go

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Ya no necesitamos FUNCTION_SPEC al usar el Modo JSON

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
    max_retries: int = 1 # Reducimos a 1 reintento, ya que el modo JSON es muy fiable
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Genera un árbol de preguntas usando el LLM en Modo JSON para máxima fiabilidad.
    """
    ctx_str = json.dumps(context, indent=2) if context else "{}"
    
    prompt = (
        "Eres un experto mundial en Análisis Envolvente de Datos (DEA). "
        "Tu tarea es generar un árbol de hipótesis en formato JSON sobre las causas de la ineficiencia, basándote en un contexto.\n\n"
        f"--- CONTEXTO ---\n{ctx_str}\n\n"
        f"--- PREGUNTA RAÍZ ---\n{root_question}\n\n"
        "--- INSTRUCCIONES ESTRICTAS ---\n"
        "1. Tu única y exclusiva salida DEBE SER un objeto JSON válido.\n"
        "2. El JSON debe tener una única clave raíz llamada 'tree', cuyo valor es el árbol de hipótesis (un objeto JSON anidado).\n"
        "3. NO escribas ningún texto, explicación, saludo o markdown. Tu respuesta debe ser únicamente el JSON.\n"
        "4. El árbol debe tener 2-3 niveles. Las claves del primer nivel deben ser las hipótesis principales (ej. 'Uso excesivo de recursos')."
    )

    for attempt in range(max_retries + 1):
        if attempt > 0:
            print(f"--- DEBUG: La IA no devolvió un JSON con la clave 'tree'. Reintentando (Intento {attempt + 1}) ---")
            time.sleep(1)
        
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}, # <- LA CLAVE DE LA SOLUCIÓN
                temperature=0.2 + (attempt * 0.2), 
            )
            
            # La API garantiza que 'content' es un string JSON válido
            response_json = json.loads(resp.choices[0].message.content)
            tree = response_json.get("tree", {})
            
            if tree: 
                print("--- DEBUG: Árbol de indagación generado por la IA con éxito usando Modo JSON. ---")
                return {root_question: tree}, None # Éxito

        except Exception as e:
            print(f"--- DEBUG: Error en la API de OpenAI en el intento {attempt + 1}: {e} ---")
            if attempt >= max_retries:
                return None, f"Fallo la conexión con la API de OpenAI tras {max_retries+1} intentos. Detalle: {e}"
            time.sleep(1)

    # Si todos los reintentos fallan, entonces usamos el árbol de respaldo
    print("--- DEBUG: La IA no devolvió un JSON con la clave 'tree' tras todos los reintentos. Usando árbol de respaldo. ---")
    return _fallback_tree(root_question), "La IA no generó un árbol con la estructura esperada. Usando árbol de respaldo."


def _fallback_tree(root_q: str) -> Dict[str, Any]:
    """Árbol de respaldo si la llamada a la IA falla."""
    return {
        root_q: {
            "¿Posible exceso de inputs?": {"Analizar variable por variable": {}},
            "¿Posible déficit de outputs?": {"Comparar con la media de los eficientes": {}},
            "¿Combinación ineficiente de factores?": {"Buscar outliers": {}},
        }
    }
