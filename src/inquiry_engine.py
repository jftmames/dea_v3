# jftmames/-dea-deliberativo-mvp/src/inquiry_engine.py
# --- VERSIÓN CORREGIDA ---

import os
import json
import time
from typing import Any, Dict, Optional, Tuple
from openai import OpenAI
import plotly.graph_objects as go

# Inicialización del cliente de OpenAI. La clave se lee de las variables de entorno.
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    # Manejo de error si la clave no está disponible.
    client = None
    print(f"Advertencia: No se pudo inicializar el cliente de OpenAI. {e}")


def to_plotly_tree(tree: Dict[str, Any], title: str = "Árbol de Auditoría Metodológica") -> go.Figure:
    """
    Convierte un diccionario anidado en un Treemap interactivo de Plotly.
    """
    labels, parents, values = [], [], []
    if not tree or not isinstance(tree, dict):
        # Devuelve una figura vacía si el árbol no es válido.
        return go.Figure(layout={"title": "No hay datos para mostrar en el árbol."})

    # El nodo raíz del árbol.
    root_label = list(tree.keys())[0]
    
    # Función recursiva para recorrer el árbol y construir las listas de Plotly.
    def walk(node: Dict[str, Any], parent: str):
        for pregunta, hijos in node.items():
            if pregunta not in labels:
                labels.append(pregunta)
                parents.append(parent)
                # Asignar un valor base para el tamaño del nodo
                values.append(10) 
            
            if isinstance(hijos, dict):
                walk(hijos, pregunta)
            
    # La raíz no tiene padre en la visualización de Plotly
    walk({root_label: tree[root_label]}, "")
    
    # Creación de la figura del Treemap.
    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values, # Añadimos valores para un mejor hoverinfo
        root_color="lightgrey",
        textinfo="label",
        # --- CORRECCIÓN ---
        # Se ha cambiado hoverinfo a una combinación válida.
        # 'label' muestra el nombre del nodo.
        # 'percent parent' muestra qué porcentaje representa del nodo padre.
        hoverinfo="label+percent parent" 
    ))
    fig.update_layout(
        title_text=title,
        title_x=0.5,
        margin=dict(t=50, l=25, r=25, b=25)
    )
    return fig

def generate_inquiry(
    root_question: str,
    context: Optional[Dict[str, Any]] = None,
    max_retries: int = 1
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Genera un árbol de preguntas metodológicas usando el LLM en Modo JSON.
    """
    if not client:
        # Si el cliente no se pudo inicializar, usa el árbol de respaldo.
        return _fallback_tree(root_question), "El cliente de OpenAI no está configurado."

    ctx_str = json.dumps(context, indent=2, ensure_ascii=False) if context else "{}"
    
    prompt = (
        "Eres un catedrático de econometría y experto mundial en Análisis Envolvente de Datos (DEA), "
        "revisando una propuesta de investigación para una revista de primer nivel (Q1).\n"
        "Tu tarea es generar un **árbol de auditoría metodológica** en formato JSON para evaluar la robustez de la especificación del modelo propuesto.\n\n"
        f"--- CONTEXTO DEL MODELO PROPUESTO ---\n{ctx_str}\n\n"
        f"--- PREGUNTA RAÍZ PARA LA AUDITORÍA ---\n{root_question}\n\n"
        "--- INSTRUCCIONES ESTRICTAS PARA EL ÁRBOL DE AUDITORÍA ---\n"
        "1. Tu única salida DEBE SER un objeto JSON válido y nada más.\n"
        "2. El JSON debe tener una única clave raíz 'tree', cuyo valor es el árbol de preguntas.\n"
        "3. El árbol debe tener entre 2 y 3 niveles de profundidad, descomponiendo los problemas metodológicos clave.\n"
        "4. Las preguntas deben ser críticas pero constructivas, enfocadas en los siguientes pilares de un buen análisis DEA:\n"
        "   a. **Justificación Teórica:** ¿Por qué se eligieron estos inputs/outputs y no otros? ¿Se basa en la literatura existente?\n"
        "   b. **Especificación del Modelo:** ¿La elección de rendimientos a escala (CRS vs. VRS) es adecuada para la muestra? ¿La orientación (input/output) es coherente con el problema?\n"
        "   c. **Calidad de los Datos:** ¿Cómo se manejarán los outliers, los ceros o los datos negativos? ¿Existe riesgo de multicolinealidad entre las variables?\n"
        "   d. **Robustez del Análisis:** ¿Qué pruebas de sensibilidad se realizarán para validar los resultados?\n"
    )

    for attempt in range(max_retries + 1):
        if attempt > 0:
            time.sleep(1)
        
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3 + (attempt * 0.2),
            )
            
            response_json = json.loads(resp.choices[0].message.content)
            tree = response_json.get("tree", {})
            
            if tree:
                return {root_question: tree}, None

        except Exception as e:
            if attempt >= max_retries:
                return None, f"Fallo la conexión con la API tras {max_retries + 1} intentos. Detalle: {e}"

    return _fallback_tree(root_question), "La IA no generó un árbol con la estructura esperada y se ha utilizado un mapa de respaldo."


def _fallback_tree(root_q: str) -> Dict[str, Any]:
    """
    Árbol de respaldo si la llamada a la IA falla.
    """
    return {
        root_q: {
            "Validación de la Selección de Variables": {
                "¿Cuál es la justificación teórica para cada input y output?": {},
                "¿Se ha comprobado la correlación entre las variables de entrada?": {}
            },
            "Adecuación de la Especificación del Modelo": {
                "¿Por qué se eligieron los rendimientos a escala (CRS/VRS)?": {},
                "¿La orientación del modelo (input/output) es la correcta?": {}
            },
            "Análisis de Robustez": {
                "¿Cómo se planea tratar los datos atípicos (outliers)?": {},
                "¿Se realizará algún análisis de sensibilidad sobre los resultados?": {}
            }
        }
    }
