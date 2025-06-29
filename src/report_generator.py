import pandas as pd
import datetime
from io import BytesIO
import json

def generate_html_report(
    analysis_results: dict,
    inquiry_tree: dict | None = None,
    user_justifications: dict | None = None
) -> str:
    """
    Genera un string con contenido HTML que se adapta a los resultados del análisis.
    
    Args:
        analysis_results (dict): El diccionario de resultados de analysis_dispatcher.
        inquiry_tree (dict): El árbol de indagación generado por la IA.
        user_justifications (dict): Diccionario con las justificaciones del usuario.
    """
    # Convertir None en DataFrame vacío
    df_results = analysis_results.get("main_df", pd.DataFrame())
    model_name = analysis_results.get("model_name", "Análisis DEA")
    
    df_tree_data = pd.DataFrame()
    if inquiry_tree:
        # Aplanar el árbol para una representación tabular simple
        rows = []
        # Asumiendo que la raíz está directamente en inquiry_tree
        root_question_key = list(inquiry_tree.keys())[0] if inquiry_tree else None

        def walk_tree_and_collect(node_dict, parent_question):
            if not isinstance(node_dict, dict):
                return
            for question, children in node_dict.items():
                rows.append({"Pregunta de Auditoría": question, "Pregunta Padre": parent_question})
                if isinstance(children, dict) and children:
                    walk_tree_and_collect(children, question)
        
        if root_question_key:
            walk_tree_and_collect({root_question_key: inquiry_tree[root_question_key]}, "N/A (Raíz)")
        
        df_tree_data = pd.DataFrame(rows)
        # Añadir justificaciones si existen
        if user_justifications:
            df_tree_data['Tu Justificación'] = df_tree_data['Pregunta de Auditoría'].map(user_justifications)
            df_tree_data['Tu Justificación'] = df_tree_data['Tu Justificación'].fillna("")


    fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = (
        "<html><head><meta charset='utf-8'>"
        "<style>"
        "body {font-family: sans-serif; line-height: 1.6; color: #333; margin: 20px;}"
        "table {border-collapse: collapse; width: 100%; margin-bottom: 2em;}"
        "th, td {border: 1px solid #ddd; padding: 10px; text-align: left; vertical-align: top;}"
        "th {background-color: #f2f2f2; font-weight: bold;}"
        "h1, h2, h3 {color: #1a4b7a; margin-top: 1.5em; margin-bottom: 0.8em;}"
        "p {margin-bottom: 1em;}"
        ".section-box {border: 1px solid #e0e0e0; padding: 15px; border-radius: 8px; margin-bottom: 1.5em; background-color: #fcfcfc;}"
        ".note {background-color: #e6f7ff; border-left: 5px solid #2196F3; padding: 10px; margin-bottom: 1em;}"
        "</style>"
        f"<title>Reporte DEA Deliberativo – {fecha}</title></head><body>"
    )
    html += f"<h1>Reporte de Análisis DEA Deliberativo</h1>"
    html += f"<p class='note'>Generado el: {fecha}</p>"

    # Sección de Resumen
    html += "<h2>1. Resumen del Análisis</h2>"
    html += f"<div class='section-box'><p>Este informe detalla los resultados de un análisis de Eficiencia Envolvente de Datos (DEA) realizado utilizando el modelo **{model_name}**.</p>"
    if analysis_results.get("dea_config"):
        html += f"<p><strong>Configuración del modelo:</strong> <code>{json.dumps(analysis_results['dea_config'], indent=2)}</code></p>"
    if analysis_results.get("selected_proposal"):
        proposal = analysis_results["selected_proposal"]
        html += f"<p><strong>Enfoque de Análisis:</strong> {proposal.get('title', 'N/A')}</p>"
        html += f"<p><strong>Inputs utilizados:</strong> {', '.join(proposal.get('inputs', []))}</p>"
        html += f"<p><strong>Outputs utilizados:</strong> {', '.join(proposal.get('outputs', []))}</p>"
        html += f"<p><strong>Razonamiento del enfoque:</strong> {proposal.get('reasoning', 'Sin razonamiento provisto.')}</p>"
    html += "</div>"


    # Sección de Resultados DEA (dinámica)
    html += f"<h2>2. Resultados Numéricos del Modelo: {model_name}</h2>"
    if not df_results.empty:
        html += df_results.to_html(index=False, border=1, justify="left", na_rep="-")
    else:
        html += "<p>No se generaron resultados numéricos para este modelo.</p>"

    # Sección Árbol de Indagación y Justificaciones
    if not df_tree_data.empty:
        html += "<h2>3. Taller de Auditoría Metodológica</h2>"
        html += "<div class='section-box'>"
        html += "<p>Esta sección resume el mapa de razonamiento generado por la IA para auditar la robustez metodológica del análisis DEA, junto con las justificaciones aportadas por el investigador.</p>"
        
        html += "<h3>3.1. Estructura del Mapa de Razonamiento y Justificaciones</h3>"
        html += df_tree_data.to_html(index=False, border=1, justify="left")
        html += "</div>"
        
        if analysis_results.get('tree_explanation'):
            html += "<div class='section-box'>"
            html += "<h3>3.2. Explicación del Mapa (Generada por IA)</h3>"
            html += f"<p>{analysis_results['tree_explanation']}</p>"
            html += "</div>"


    html += "</body></html>"
    return html


def generate_excel_report(
    analysis_results: dict,
    inquiry_tree: dict | None = None,
    user_justifications: dict | None = None
) -> BytesIO:
    """
    Genera un reporte Excel en memoria con pestañas dinámicas.
    
    Args:
        analysis_results (dict): El diccionario de resultados de analysis_dispatcher.
        inquiry_tree (dict): El árbol de indagación generado por la IA.
        user_justifications (dict): Diccionario con las justificaciones del usuario.
    """
    df_results = analysis_results.get("main_df", pd.DataFrame())
    model_name = analysis_results.get("model_name", "Resultados")
    # Limpiar el nombre del modelo para que sea un nombre de hoja válido
    sheet_name_results = ''.join(c for c in model_name if c.isalnum() or c == ' ')[:25].strip().replace(' ', '_')
    if not sheet_name_results: sheet_name_results = "DEA_Results"


    df_tree_data = pd.DataFrame()
    if inquiry_tree:
        rows = []
        root_question_key = list(inquiry_tree.keys())[0] if inquiry_tree else None

        def walk_tree_and_collect(node_dict, parent_question):
            if not isinstance(node_dict, dict):
                return
            for question, children in node_dict.items():
                rows.append({"Pregunta de Auditoría": question, "Pregunta Padre": parent_question})
                if isinstance(children, dict) and children:
                    walk_tree_and_collect(children, question)
        
        if root_question_key:
            walk_tree_and_collect({root_question_key: inquiry_tree[root_question_key]}, "N/A (Raíz)")
        
        df_tree_data = pd.DataFrame(rows)
        # Añadir justificaciones si existen
        if user_justifications:
            df_tree_data['Tu Justificación'] = df_tree_data['Pregunta de Auditoría'].map(user_justifications)
            df_tree_data['Tu Justificación'] = df_tree_data['Tu Justificación'].fillna("")


    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # 1) Hoja de Resultados (nombre dinámico)
        df_results.to_excel(writer, sheet_name=sheet_name_results, index=False)
        worksheet_results = writer.sheets[sheet_name_results]
        # Auto-ajustar columnas
        for idx, col in enumerate(df_results.columns):
            max_len = max(
                df_results[col].astype(str).map(len).max() if not df_results.empty else 0,
                len(str(col))
            ) + 2
            worksheet_results.set_column(idx, idx, max_len)

        # 2) Hoja del Árbol de Indagación
        if not df_tree_data.empty:
            df_tree_data.to_excel(writer, sheet_name="Mapa_Razonamiento", index=False)
            worksheet_tree = writer.sheets["Mapa_Razonamiento"]
            for idx, col in enumerate(df_tree_data.columns):
                max_len = max(
                    df_tree_data[col].astype(str).map(len).max() if not df_tree_data.empty else 0,
                    len(str(col))
                ) + 2
                worksheet_tree.set_column(idx, idx, max_len)

    output.seek(0)
    return output

