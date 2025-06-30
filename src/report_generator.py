# report_generator.py - VERSIÓN COMPLETA Y FUNCIONAL
import pandas as pd
import datetime
from io import BytesIO

def generate_html_report(scenario_data: dict) -> str:
    analysis_results = scenario_data.get('dea_results', {})
    checklist_responses = scenario_data.get('checklist_responses', {})
    
    df_results = analysis_results.get("main_df", pd.DataFrame())
    model_name = analysis_results.get("model_name", "Análisis DEA")
    fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""
    <html><head><meta charset='utf-8'>
    <style>body{{font-family:sans-serif;margin:20px}}table{{border-collapse:collapse;width:100%;margin-bottom:1em}}th,td{{border:1px solid #ddd;padding:8px;text-align:left}}th{{background-color:#f2f2f2}}h1,h2{{color:#1a4b7a}}.section-box{{border:1px solid #e0e0e0;padding:15px;border-radius:8px;margin-bottom:1.5em}}</style>
    <title>Reporte DEA Deliberativo – {fecha}</title></head><body>
    <h1>Reporte de Análisis DEA Deliberativo</h1><p>Generado el: {fecha}</p>
    """

    html += "<h2>1. Resumen del Análisis</h2><div class='section-box'><p>Modelo utilizado: <strong>{model_name}</strong>.</p></div>"

    if checklist_responses:
        html += "<h2>2. Checklist de Verificación Metodológica</h2><div class='section-box'><ul>"
        checklist_map = {
            "homogeneity": "¿DMUs homogéneas?",
            "rule_of_thumb": "¿Nº DMUs > 3 * (Inputs+Outputs)?",
            "isotonicity": "¿Relación de isotocinidad verificada?"
        }
        for key, checked in checklist_responses.items():
            question_text = checklist_map.get(key, key)
            status_icon = "✅ Sí" if checked else "❌ No / Sin responder"
            html += f"<li><b>{question_text}</b>: {status_icon}</li>"
        html += "</ul></div>"

    html += f"<h2>3. Resultados Numéricos</h2>"
    if not df_results.empty:
        html += df_results.to_html(index=False, border=1)
    else:
        html += "<p>No se generaron resultados.</p>"

    html += "</body></html>"
    return html

def generate_excel_report(scenario_data: dict) -> BytesIO:
    analysis_results = scenario_data.get('dea_results', {})
    checklist_responses = scenario_data.get('checklist_responses', {})
    df_results = analysis_results.get("main_df", pd.DataFrame())
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_results.to_excel(writer, sheet_name="Resultados_DEA", index=False)
        if checklist_responses:
            df_checklist = pd.DataFrame(list(checklist_responses.items()), columns=["Verificación", "Respuesta"])
            df_checklist["Respuesta"] = df_checklist["Respuesta"].apply(lambda x: "Sí" if x else "No")
            df_checklist.to_excel(writer, sheet_name="Checklist", index=False)
    output.seek(0)
    return output
