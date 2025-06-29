import pandas as pd
import datetime
from io import BytesIO
import json

def generate_html_report(
    analysis_results: dict,
    inquiry_tree: dict | None = None,
    user_justifications: dict | None = None,
    data_overview_info: dict | None = None # Nuevo parámetro
) -> str:
    """
    Genera un string con contenido HTML que se adapta a los resultados del análisis.
    
    Args:
        analysis_results (dict): El diccionario de resultados de analysis_dispatcher.
        inquiry_tree (dict): El árbol de indagación generado por la IA.
        user_justifications (dict): Diccionario con las justificaciones del usuario.
        data_overview_info (dict): Diccionario con el resumen inicial de los datos cargados y la validación.
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

    # Sección de Resumen del Análisis
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

    # Nueva sección: Estado de los Datos y Validación Inicial
    if data_overview_info:
        html += "<h2>2. Estado de los Datos y Validación Inicial</h2>"
        html += "<div class='section-box'>"
        html += "<p>Esta sección resume las características principales de los datos utilizados en el análisis y los problemas detectados durante la validación inicial.</p>"
        
        html += "<h3>2.1. Dimensiones del DataFrame</h3>"
        html += f"<p>Filas: {data_overview_info.get('shape', [0,0])[0]}, Columnas: {data_overview_info.get('shape', [0,0])[1]}</p>"

        html += "<h3>2.2. Tipos de Datos por Columna</h3>"
        html += "<table><tr><th>Columna</th><th>Tipo de Dato</th></tr>"
        for col, dtype in data_overview_info.get('column_types', {}).items():
            html += f"<tr><td>{col}</td><td>{dtype}</td></tr>"
        html += "</table>"

        html += "<h3>2.3. Resumen Estadístico (Columnas Numéricas)</h3>"
        # Convertir el diccionario de numerical_summary a DataFrame para un mejor HTML
        numerical_summary_df = pd.DataFrame(data_overview_info.get('numerical_summary', {}))
        html += numerical_summary_df.to_html(border=1, justify="left", na_rep="-")

        html += "<h3>2.4. Problemas Potenciales de Datos Detectados</h3>"
        issues_present = False
        
        if any(data_overview_info.get('null_counts', {}).values()):
            html += "<p><strong>Valores Nulos Detectados:</strong></p>"
            html += "<table><tr><th>Columna</th><th>Cantidad de Nulos</th></tr>"
            for col, count in data_overview_info['null_counts'].items():
                if count > 0:
                    html += f"<tr><td>{col}</td><td>{count}</td></tr>"
            html += "</table>"
            issues_present = True

        if data_overview_info.get('non_numeric_issues'):
            html += "<p><strong>Columnas con Valores No Numéricos (Potenciales Errores):</strong></p><ul>"
            for col in data_overview_info['non_numeric_issues']:
                html += f"<li>La columna '{col}' parece contener valores que no son números.</li>"
            html += "</ul>"
            issues_present = True
        
        if data_overview_info.get('zero_negative_counts'):
            html += "<p><strong>Columnas Numéricas con Ceros o Valores Negativos:</strong></p>"
            html += "<table><tr><th>Columna</th><th>Cantidad (Cero/Negativo)</th></tr>"
            for col, count in data_overview_info['zero_negative_counts'].items():
                if count > 0:
                    html += f"<tr><td>{col}</td><td>{count}</td></tr>"
            html += "</table>"
            issues_present = True
        
        # Add LLM validation issues if available
        llm_validation_results = data_overview_info.get('llm_validation_results', {}).get('llm', {})
        if llm_validation_results.get('issues'):
            html += "<p><strong>Sugerencias y Advertencias de la IA (Validación de Idoneidad/Homogeneidad):</strong></p><ul>"
            for issue in llm_validation_results['issues']:
                html += f"<li>{issue}</li>"
            html += "</ul>"
            issues_present = True
        if llm_validation_results.get('suggested_fixes'):
            html += "<p><strong>Sugerencias de la IA para Mejorar:</strong></p><ul>"
            for fix in llm_validation_results['suggested_fixes']:
                html += f"<li>{fix}</li>"
            html += "</ul>"
            issues_present = True

        if not issues_present:
            html += "<p>No se detectaron problemas obvios (nulos, no numéricos, ceros/negativos) en este informe rápido.</p>"
        
        html += "<p><strong>Nota sobre la limpieza:</strong> Esta aplicación no realiza la limpieza de datos por ti. Se recomienda encarecidamente preparar y limpiar tus datos en una herramienta externa antes de subirlos para un análisis DEA óptimo. Los problemas aquí indicados (nulos, no numéricos, ceros/negativos) son críticos para la fiabilidad del DEA.</p>"
        html += "</div>"


    # Sección de Resultados DEA (dinámica)
    html += f"<h2>3. Resultados Numéricos del Modelo: {model_name}</h2>"
    if not df_results.empty:
        html += df_results.to_html(index=False, border=1, justify="left", na_rep="-")
    else:
        html += "<p>No se generaron resultados numéricos para este modelo.</p>"

    # Sección Árbol de Indagación y Justificaciones
    if not df_tree_data.empty:
        html += "<h2>4. Taller de Auditoría Metodológica</h2>"
        html += "<div class='section-box'>"
        html += "<p>Esta sección resume el mapa de razonamiento generado por la IA para auditar la robustez metodológica del análisis DEA, junto con las justificaciones aportadas por el investigador.</p>"
        
        html += "<h3>4.1. Estructura del Mapa de Razonamiento y Justificaciones</h3>"
        html += df_tree_data.to_html(index=False, border=1, justify="left")
        html += "</div>"
        
        if analysis_results.get('tree_explanation'):
            html += "<div class='section-box'>"
            html += "<h3>4.2. Explicación del Mapa (Generada por IA)</h3>"
            html += f"<p>{analysis_results['tree_explanation']}</p>"
            html += "</div>"


    html += "</body></html>"
    return html


def generate_excel_report(
    analysis_results: dict,
    inquiry_tree: dict | None = None,
    user_justifications: dict | None = None,
    data_overview_info: dict | None = None # Nuevo parámetro
) -> BytesIO:
    """
    Genera un reporte Excel en memoria con pestañas dinámicas.
    
    Args:
        analysis_results (dict): El diccionario de resultados de analysis_dispatcher.
        inquiry_tree (dict): El árbol de indagación generado por la IA.
        user_justifications (dict): Diccionario con las justificaciones del usuario.
        data_overview_info (dict): Diccionario con el resumen inicial de los datos cargados y la validación.
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
        
        # 3) Nueva Hoja de Resumen de Datos
        if data_overview_info:
            summary_data = {
                "Metrica": [],
                "Valor": []
            }
            summary_data["Metrica"].append("Filas")
            summary_data["Valor"].append(data_overview_info.get('shape', [0,0])[0])
            summary_data["Metrica"].append("Columnas")
            summary_data["Valor"].append(data_overview_info.get('shape', [0,0])[1])

            # Tipos de datos
            df_types = pd.DataFrame(list(data_overview_info.get('column_types', {}).items()), columns=['Columna', 'Tipo de Dato'])
            if not df_types.empty:
                df_types.to_excel(writer, sheet_name="Resumen_Datos", startrow=0, startcol=3, index=False)
                worksheet_data = writer.sheets["Resumen_Datos"]
                for idx, col in enumerate(df_types.columns):
                    max_len = max(
                        df_types[col].astype(str).map(len).max() if not df_types.empty else 0,
                        len(str(col))
                    ) + 2
                    worksheet_data.set_column(idx + 3, idx + 3, max_len)
            
            # Resumen Numérico
            numerical_summary_df_excel = pd.DataFrame(data_overview_info.get('numerical_summary', {})).T
            if not numerical_summary_df_excel.empty:
                numerical_summary_df_excel.to_excel(writer, sheet_name="Resumen_Datos", startrow=len(df_types)+2, startcol=3)
                worksheet_data = writer.sheets["Resumen_Datos"]
                for idx, col in enumerate(numerical_summary_df_excel.columns):
                    max_len = max(
                        numerical_summary_df_excel[col].astype(str).map(len).max() if not numerical_summary_df_excel.empty else 0,
                        len(str(col))
                    ) + 2
                    worksheet_data.set_column(idx + 3 + len(df_types.columns), idx + 3 + len(df_types.columns), max_len) # Adjust start column


            # Problemas Potenciales
            issues_rows = []
            if any(data_overview_info.get('null_counts', {}).values()):
                for col, count in data_overview_info['null_counts'].items():
                    if count > 0:
                        issues_rows.append({"Tipo de Problema": "Valores Nulos", "Columna": col, "Cantidad": count})
            if data_overview_info.get('non_numeric_issues'):
                for col in data_overview_info['non_numeric_issues']:
                    issues_rows.append({"Tipo de Problema": "No Numérico", "Columna": col, "Cantidad": "N/A"})
            if data_overview_info.get('zero_negative_counts'):
                for col, count in data_overview_info['zero_negative_counts'].items():
                    if count > 0:
                        issues_rows.append({"Tipo de Problema": "Ceros/Negativos", "Columna": col, "Cantidad": count})
            
            llm_validation_results = data_overview_info.get('llm_validation_results', {}).get('llm', {})
            if llm_validation_results.get('issues'):
                for issue in llm_validation_results['issues']:
                    issues_rows.append({"Tipo de Problema": "Sugerencia IA (Conceptual)", "Columna": "N/A", "Cantidad": issue})
            if llm_validation_results.get('suggested_fixes'):
                for fix in llm_validation_results['suggested_fixes']:
                    issues_rows.append({"Tipo de Problema": "Sugerencia IA (Fix)", "Columna": "N/A", "Cantidad": fix})


            df_issues = pd.DataFrame(issues_rows)
            if not df_issues.empty:
                df_issues.to_excel(writer, sheet_name="Resumen_Datos", startrow=0, startcol=0, index=False)
                worksheet_data = writer.sheets["Resumen_Datos"]
                for idx, col in enumerate(df_issues.columns):
                    max_len = max(
                        df_issues[col].astype(str).map(len).max() if not df_issues.empty else 0,
                        len(str(col))
                    ) + 2
                    worksheet_data.set_column(idx, idx, max_len)

    output.seek(0)
    return output

