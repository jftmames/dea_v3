# /src/inquiry_engine.py
# --- VERSIÓN REFACTORIZADA Y DINÁMICA ---

import os
import json
import uuid
import time
from typing import Any, Dict, Optional, Tuple, List
from openai import OpenAI
import plotly.graph_objects as go

# --- 1. CONFIGURACIÓN DEL CLIENTE (Sin cambios) ---
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    client = None
    print(f"Advertencia: No se pudo inicializar el cliente de OpenAI. {e}")


# --- 2. NUEVA CLASE ESTRUCTURAL: InquiryNode ---
# Reemplaza la estructura de diccionarios por un objeto más robusto y explícito.
class InquiryNode:
    """
    Representa un nodo en el árbol de indagación epistémica.
    Contiene la pregunta, un prompt para la justificación, y estado para la UI.
    """
    def __init__(self, question: str, justification_prompt: str = "", children: List['InquiryNode'] = None, expanded: bool = False):
        self.id = str(uuid.uuid4())
        self.question = question
        self.justification_prompt = justification_prompt
        self.children = children if children is not None else []
        self.expanded = expanded
        self.justification = "" # Campo para la respuesta del usuario

# --- 3. CLASE PRINCIPAL: InquiryEngine ---
# Encapsula toda la lógica de generación y expansión del árbol.
class InquiryEngine:
    """
    Motor para generar y gestionar el árbol de indagación metodológica.
    """
    def __init__(self, llm_client: Optional[OpenAI] = client):
        self.client = llm_client

    def generate_initial_tree(self, context: Optional[Dict[str, Any]] = None) -> Tuple[Optional[InquiryNode], Optional[str]]:
        """
        Genera el árbol inicial de preguntas usando el LLM, con prompts de justificación.
        """
        if not self.client:
            return _fallback_tree_node(), "El cliente de OpenAI no está configurado. Se usó un árbol de respaldo."

        ctx_str = json.dumps(context, indent=2) if context else "{}"
        prompt = (
            "Eres un catedrático de econometría y experto en DEA revisando una propuesta para una revista Q1.\n"
            "Tu tarea es generar un **árbol de auditoría metodológica** en formato JSON.\n\n"
            f"--- CONTEXTO DEL MODELO PROPUESTO ---\n{ctx_str}\n\n"
            "--- INSTRUCCIONES ESTRICTAS ---\n"
            "1. Tu única salida DEBE SER un objeto JSON válido.\n"
            "2. El JSON debe tener una clave raíz 'question' (la pregunta principal) y una clave 'children' (una lista de nodos hijos).\n"
            "3. Cada nodo (incluida la raíz y los hijos) DEBE tener dos claves: 'question' (la pregunta) y 'justification_prompt' (una pregunta específica para guiar la justificación del usuario).\n"
            "4. El árbol debe tener 2-3 niveles de profundidad, descomponiendo problemas metodológicos clave de DEA."
            """
            Ejemplo de formato de un nodo:
            {
                "question": "¿Qué tipo de rendimientos a escala has asumido?",
                "justification_prompt": "Fundamenta tu elección (CRS vs VRS). ¿Es razonable suponer que un aumento en los inputs conlleva un aumento proporcional en los outputs para todas las DMUs?",
                "children": [] 
            }
            """
        )

        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            response_json = json.loads(resp.choices[0].message.content)
            
            # Función recursiva para construir el árbol de nodos a partir del JSON
            def build_tree_from_json(data: Dict[str, Any]) -> InquiryNode:
                node = InquiryNode(
                    question=data.get("question", "Pregunta no encontrada"),
                    justification_prompt=data.get("justification_prompt", "Añade tu justificación.")
                )
                if "children" in data and data["children"]:
                    node.children = [build_tree_from_json(child) for child in data["children"]]
                return node

            root_node = build_tree_from_json(response_json)
            return root_node, None

        except Exception as e:
            return _fallback_tree_node(), f"Fallo la conexión con la API. Detalle: {e}"

    def expand_question_node(self, question_text: str) -> Tuple[Optional[List[InquiryNode]], Optional[str]]:
        """
        Usa el LLM para generar y devolver una lista de sub-nodos para una pregunta dada.
        """
        if not self.client:
            return [], "El cliente de OpenAI no está configurado."
        
        prompt = (
            "Eres un experto en metodología DEA. Dada la siguiente pregunta metodológica, desglósala en 2-3 sub-preguntas más específicas y detalladas.\n"
            f"Pregunta principal: \"{question_text}\"\n"
            "--- INSTRUCCIONES ESTRICTAS ---\n"
            "1. Tu única salida DEBE SER un objeto JSON válido y nada más.\n"
            "2. El JSON debe contener una única clave 'sub_nodes'.\n"
            "3. El valor de 'sub_nodes' debe ser una LISTA de objetos, donde cada objeto tiene las claves 'question' y 'justification_prompt'.\n"
            """
            Ejemplo de formato de salida:
            {
                "sub_nodes": [
                    { "question": "Sub-pregunta 1", "justification_prompt": "Prompt para la sub-pregunta 1" },
                    { "question": "Sub-pregunta 2", "justification_prompt": "Prompt para la sub-pregunta 2" }
                ]
            }
            """
        )
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.4,
            )
            response_json = json.loads(resp.choices[0].message.content)
            sub_nodes_data = response_json.get("sub_nodes", [])
            
            new_nodes = [
                InquiryNode(
                    question=data.get("question", "Sub-pregunta inválida"),
                    justification_prompt=data.get("justification_prompt", "Añade tu justificación.")
                ) for data in sub_nodes_data
            ]
            return new_nodes, None
        except Exception as e:
            return [], f"Fallo la expansión de la pregunta. Detalle: {e}"

# --- 4. ÁRBOL DE RESPALDO (ACTUALIZADO A InquiryNode) ---
def _fallback_tree_node() -> InquiryNode:
    """Árbol de respaldo si la IA falla, ahora construido con InquiryNode."""
    return InquiryNode(
        question="Auditoría Metodológica (Respaldo)",
        justification_prompt="Describe el propósito general de tu análisis DEA.",
        children=[
            InquiryNode(
                question="Validación de Variables",
                justification_prompt="¿Cuál es el criterio principal para la selección de variables?",
                children=[
                    InquiryNode(question="¿Cuál es la justificación teórica para cada input y output?", justification_prompt="Cita la literatura o la lógica económica detrás de tu elección."),
                    InquiryNode(question="¿Se ha comprobado la correlación entre variables?", justification_prompt="Describe si has realizado un test de correlación y cómo interpretarías los resultados.")
                ]
            ),
            InquiryNode(
                question="Especificación del Modelo",
                justification_prompt="¿Por qué la especificación elegida es la más adecuada?",
                children=[
                    InquiryNode(question="¿Por qué se eligieron los rendimientos a escala (CRS/VRS)?", justification_prompt="Argumenta tu elección en función de las características de las DMUs."),
                ]
            )
        ]
    )

# --- 5. VISUALIZACIÓN (CON CAPA DE COMPATIBILIDAD) ---
def nodetree_to_dict(node: InquiryNode) -> Dict[str, Any]:
    """Función de compatibilidad: Convierte un árbol de InquiryNode a un diccionario."""
    return {node.question: {child.question: nodetree_to_dict(child) for child in node.children}}

def to_plotly_tree(tree_node: InquiryNode, title: str = "Árbol de Auditoría Metodológica") -> go.Figure:
    """
    Convierte un árbol de InquiryNode en un Treemap interactivo de Plotly.
    Utiliza la capa de compatibilidad para no alterar la lógica de Plotly.
    """
    if not tree_node:
        return go.Figure(layout={"title": "No hay datos para mostrar en el árbol."})

    # Convertir el árbol de nodos al formato de diccionario que la función original espera
    tree_dict = nodetree_to_dict(tree_node)
    
    labels, parents, values = [], [], []
    root_label = list(tree_dict.keys())[0]
    
    def walk(node_dict: Dict[str, Any], parent: str):
        for pregunta, hijos in node_dict.items():
            if pregunta not in labels:
                labels.append(pregunta)
                parents.append(parent)
                values.append(10)
            if isinstance(hijos, dict) and hijos:
                walk(hijos, pregunta)

    walk({root_label: tree_dict[root_label]}, "")
    
    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        root_color="lightgrey",
        textinfo="label",
        hoverinfo="label+percent parent"
    ))
    fig.update_layout(title_text=title, title_x=0.5, margin=dict(t=50, l=25, r=25, b=25))
    return fig
