# jftmames/-dea-deliberativo-mvp/-dea-deliberativo-mvp-b44b8238c978ae0314af30717b9399634d28f8f9/src/inquiry_engine.py
import os
import json
import time
from typing import Any, Dict, Optional, Tuple
from openai import OpenAI
import plotly.graph_objects as go

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
            if isinstance(hijos, dict): walk(hijos, pregunta)
            
    walk(tree, root_label)
    
    fig = go.Figure(go.Treemap(labels=labels, parents=parents, root_color="lightgrey"))
    fig.update_layout(title_text=title, title_x=0.5, margin=dict(t=50, l=25, r=25, b=25))
    return fig

def generate_inquiry(
    root_question: str,
    context: Optional[Dict[str, Any]] = None,
    max_retries: int = 1
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Genera un árbol de preguntas usando el LLM en Modo JSON para máxima fiabilidad.
    """
    ctx_str = json.dumps(context, indent=2) if context else "{}"
    
    # --- PROMPT MEJORADO CON EJEMPLO EXPLÍCITO ---
    prompt = (
        "Eres un experto mundial en Análisis Envolvente de Datos (DEA). Tu tarea es generar un árbol de hipótesis en formato JSON sobre las causas de la ineficiencia.\n\n"
        f"--- CONTEXTO ---\n{ctx_str}\n\n"
        f"--- PREGUNTA RAÍZ ---\n{root_question}\n\n"
        "--- INSTRUCCIONES ESTRICTAS ---\n"
        "1. Tu única salida DEBE SER un objeto JSON válido.\n"
        "2. El JSON debe tener una única clave raíz 'tree', cuyo valor es el árbol de hipótesis.\n"
        "3. NO escribas ningún texto, solo el JSON.\n"
        "4. El árbol debe tener 2-3 niveles.\n"
        "5. Las hojas del árbol (nodos finales) deben ser preguntas accionables. USA ESTRICTAMENTE EL SIGUIENTE FORMATO para ellas: 'Analizar input: [nombre_exacto_del_input]' o 'Analizar output: [nombre_exacto_del_output]'. Por ejemplo, si una columna se llama 'gastos_personal', la hoja DEBE SER 'Analizar input: [gastos_personal]'."
    )

    for attempt in range(max_retries + 1):
        if attempt > 0:
            print(f"--- DEBUG: Reintentando la llamada al motor de indagación (Intento {attempt + 1}) ---")
            time.sleep(1)
        
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2 + (attempt * 0.2), 
            )
            
            response_json = json.loads(resp.choices[0].message.content)
            tree = response_json.get("tree", {})
            
            if tree: 
                print("--- DEBUG: Árbol generado por IA con éxito. ---")
                return {root_question: tree}, None

        except Exception as e:
            print(f"--- DEBUG: Error en API de OpenAI: {e} ---")
            if attempt >= max_retries: return None, f"Fallo la conexión con la API. Detalle: {e}"

    print("--- DEBUG: La IA no devolvió un JSON con la clave 'tree'. Usando árbol de respaldo. ---")
    return _fallback_tree(root_question), "La IA no generó un árbol con la estructura esperada."


def _fallback_tree(root_q: str) -> Dict[str, Any]:
    """Árbol de respaldo si la llamada a la IA falla."""
    return {
        root_q: {
            "Exceso de Inputs": {"Analizar input: [nombre_input]": {}},
            "Déficit de Outputs": {"Analizar output: [nombre_output]": {}},
        }
    }
