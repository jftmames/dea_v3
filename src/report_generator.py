# En report_generator.py

import pandas as pd
import datetime
from io import BytesIO
import json
import plotly.graph_objects as go

def generate_html_report(
    analysis_results: dict,
    inquiry_tree: dict | None = None,
    user_justifications: dict | None = None,
    data_overview_info: dict | None = None,
    checklist_responses: dict | None = None # <<< NUEVO PARÁMETRO
) -> str:
    # ... (tu código HTML inicial y las secciones 1 a 4 sin cambios)
    html = "<html><head>...</head><body>" # Simplificado para el ejemplo
    html += "<h1>Reporte de Análisis DEA Deliberativo</h1>"
    # ...

    # --- INICIO DEL NUEVO BLOQUE ---
    # Sección 5: Checklist Metodológico
    if checklist_responses:
        html += "<h2>5. Checklist de Verificación Metodológica</h2>"
        html += "<div class='section-box'>"
        html += "<p>A continuación se muestran las respuestas del investigador al checklist de buenas prácticas previo a la ejecución del modelo.</p>"
        html += "<ul>"
        
        checklist_map = {
            "homogeneity": "¿He verificado que las unidades (DMUs) son suficientemente homogéneas y comparables entre sí?",
            "rule_of_thumb": "¿He comprobado la regla empírica? (Nº de DMUs ≥ 3 * (Inputs + Outputs))",
            "isotonicity": "¿He considerado la isotocidad? (A más inputs, no debería haber menos outputs)."
        }

        for key, checked in checklist_responses.items():
            question_text = checklist_map.get(key, key.replace("_", " ").capitalize())
            status_icon = "✅ Sí" if checked else "❌ No / Sin responder"
            html += f"<li><b>{question_text}</b><br>Respuesta: {status_icon}</li>"
            
        html += "</ul></div>"
    # --- FIN DEL NUEVO BLOQUE ---

    if inquiry_tree:
        html += "<h2>6. Taller de Auditoría Metodológica</h2>"
        # ... (resto de tu código para el árbol de indagación)

    html += "</body></html>"
    return html


def generate_excel_report(
    analysis_results: dict,
    inquiry_tree: dict | None = None,
    user_justifications: dict | None = None,
    data_overview_info: dict | None = None,
    checklist_responses: dict | None = None # <<< NUEVO PARÁMETRO
) -> BytesIO:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # ... (Tu código para escribir las hojas de resultados, resumen, etc., sin cambios)

        # --- INICIO DEL NUEVO BLOQUE ---
        if checklist_responses:
            checklist_map = {
                "homogeneity": "¿He verificado que las unidades (DMUs) son suficientemente homogéneas y comparables entre sí?",
                "rule_of_thumb": "¿He comprobado la regla empírica? (Nº de DMUs ≥ 3 * (Inputs + Outputs))",
                "isotonicity": "¿He considerado la isotocidad?"
            }
            
            checklist_data = {
                "Pregunta": [checklist_map.get(k, k) for k in checklist_responses.keys()],
                "Respuesta": ["Sí" if v else "No" for v in checklist_responses.values()]
            }
            df_checklist = pd.DataFrame(checklist_data)
            
            if not df_checklist.empty:
                df_checklist.to_excel(writer, sheet_name="Checklist_Metodologico", index=False)
                worksheet = writer.sheets["Checklist_Metodologico"]
                worksheet.set_column('A:A', 80)
                worksheet.set_column('B:B', 15)
        # --- FIN DEL NUEVO BLOQUE ---

    output.seek(0)
    return output
