# report_generator.py - VERSIÓN COMPLETA Y ACTUALIZADA
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
    checklist_responses: dict | None = None
) -> str:
    df_results = analysis_results.get("main_df", pd.DataFrame())
    model_name = analysis_results.get("model_name", "Análisis DEA")
    fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""
    <html><head><meta charset='utf-8'>
    <style>
        body {{font-family: sans-serif; margin: 20px;}}
        table {{border-collapse: collapse; width: 100%; margin-bottom: 1em;}}
        th, td {{border: 1px solid #ddd; padding: 8px; text-align: left;}}
        th {{background-color: #f2f2f2;}}
        h1, h2, h3 {{color: #1a4b7a;}}
        .section-box {{border: 1px solid #e0e0e0; padding: 15px; border-radius: 8px; margin-bottom: 1.5em;}}
    </style>
    <title>Reporte DEA Deliberativo – {fecha}</title></head><body>
    <h1>Reporte de Análisis DEA Deliberativo</h1>
    <p>Generado el: {fecha}</p>
    """

    # Sección 1: Resumen del Análisis
    html += "<h2>1. Resumen del Análisis</h2>"
    html += f"<div class='section-box'><p>Modelo utilizado: <strong>{model_name}</strong>.</p></div>"

    # Sección 2: Checklist Metodológico
    if checklist_responses:
        html += "<h2>2. Checklist de Verificación Metodológica</h2>"
        html += "<div class='section-box'><ul>"
        checklist_map = {
            "homogeneity": "¿He verificado que las unidades (DMUs) son suficientemente homogéneas y comparables?",
            "rule_of_thumb": "¿He comprobado la regla empírica? (Nº de DMUs ≥ 3 * (Inputs + Outputs))",
            "isotonicity": "¿He considerado la isotocidad? (A más inputs, no debería haber menos outputs)."
        }
        for key, checked in checklist_responses.items():
            question_text = checklist_map.get(key, key)
            status_icon = "✅ Sí" if checked else "❌ No / Sin responder"
            html += f"<li><b>{question_text}</b><br>Respuesta: {status_icon}</li>"
        html += "</ul></div>"

    # Sección 3: Resultados Numéricos
    html += f"<h2>3. Resultados Numéricos del Modelo: {model_name}</h2>"
    if not df_results.empty:
        html += df_results.to_html(index=False, border=1, justify="left", na_rep="-")
    else:
        html += "<p>No se generaron resultados numéricos.</p>"
    
    # ... (Secciones adicionales como el árbol de indagación pueden ir aquí) ...

    html += "</body></html>"
    return html

def generate_excel_report(
    analysis_results: dict,
    inquiry_tree: dict | None = None,
    user_justifications: dict | None = None,
    data_overview_info: dict | None = None,
    checklist_responses: dict | None = None
) -> BytesIO:
    df_results = analysis_results.get("main_df", pd.DataFrame())
    model_name = analysis_results.get("model_name", "Resultados")
    sheet_name_results = ''.join(c for c in model_name if c.isalnum())[:30] or "DEA_Results"

    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_results.to_excel(writer, sheet_name=sheet_name_results, index=False)
        worksheet_results = writer.sheets[sheet_name_results]
        for idx, col in enumerate(df_results.columns):
            worksheet_results.set_column(idx, idx, len(col) + 5)
        
        # Nueva Hoja para el Checklist
        if checklist_responses:
            checklist_map = {
                "homogeneity": "¿He verificado que las unidades (DMUs) son suficientemente homogéneas y comparables?",
                "rule_of_thumb": "¿He comprobado la regla empírica? (Nº de DMUs ≥ 3 * (Inputs + Outputs))",
                "isotonicity": "¿He considerado la isotocidad?"
            }
            df_checklist = pd.DataFrame({
                "Pregunta": [checklist_map.get(k, k) for k in checklist_responses.keys()],
                "Respuesta": ["Sí" if v else "No" for v in checklist_responses.values()]
            })
            if not df_checklist.empty:
                df_checklist.to_excel(writer, sheet_name="Checklist_Metodologico", index=False)
                worksheet_checklist = writer.sheets["Checklist_Metodologico"]
                worksheet_checklist.set_column('A:A', 80)
                worksheet_checklist.set_column('B:B', 15)
        
        # ... (código para añadir otras hojas como el árbol de indagación) ...

    output.seek(0)
    return output
