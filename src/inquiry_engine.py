# jftmames/-dea-deliberativo-mvp/-dea-deliberativo-mvp-b44b8238c978ae0314af30717b9399634d28f8f9/src/inquiry_engine.py
import os
import json
import time
from typing import Any, Dict, Optional, Tuple, List
from openai import OpenAI

# La API de OpenAI se inicializa en los módulos que la usan directamente.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def suggest_actionable_variables(
    context: Optional[Dict[str, Any]] = None,
    max_retries: int = 1
) -> Tuple[Optional[List[Dict]], Optional[str]]:
    """
    Pide a la IA que sugiera las variables más relevantes para investigar la ineficiencia.
    Devuelve una lista de diccionarios, cada uno con 'variable' y 'reasoning'.
    """
    ctx_str = json.dumps(context, indent=2) if context else "{}"
    
    prompt = (
        "Eres un consultor experto en Data Envelopment Analysis (DEA). Has analizado un conjunto de datos y has encontrado ineficiencias. Basándote en el siguiente contexto:\n\n"
        f"--- CONTEXTO ---\n{ctx_str}\n\n"
        "Tu tarea es identificar las 3 variables (inputs o outputs) más probablemente relacionadas con las causas de la ineficiencia.\n\n"
        "--- INSTRUCCIONES ESTRICTAS ---\n"
        "1. Tu única salida DEBE SER un objeto JSON válido.\n"
        "2. El JSON debe contener una única clave raíz: 'suggestions'.\n"
        "3. El valor de 'suggestions' debe ser una lista de objetos.\n"
        "4. Cada objeto en la lista debe tener exactamente dos claves: 'variable' (el nombre exacto de la columna del contexto) y 'reasoning' (una breve explicación de por qué esta variable es un buen punto de partida para el análisis)."
    )

    for attempt in range(max_retries + 1):
        if attempt > 0:
            print(f"--- DEBUG: Reintentando la sugerencia de variables (Intento {attempt + 1}) ---")
            time.sleep(1)
        
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2, 
            )
            
            response_json = json.loads(resp.choices[0].message.content)
            suggestions = response_json.get("suggestions", [])
            
            # Validar que la respuesta tiene el formato esperado
            if suggestions and isinstance(suggestions, list) and all('variable' in s and 'reasoning' in s for s in suggestions):
                print("--- DEBUG: Sugerencias de variables generadas por la IA con éxito. ---")
                return suggestions, None # Éxito

        except Exception as e:
            print(f"--- DEBUG: Error en API de OpenAI: {e} ---")
            if attempt >= max_retries: 
                return None, f"Fallo la conexión con la API. Detalle: {e}"

    return None, "La IA no devolvió sugerencias con el formato esperado."
