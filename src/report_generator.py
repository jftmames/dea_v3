import pandas as pd
import datetime
from io import BytesIO
import json
import plotly.graph_objects as go # Necesario para incrustar figuras de Plotly

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
        ".plotly-chart { margin-top: 1em; margin-bottom: 1em; border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px; background-color: #ffffff;}"
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

    # Sección: Estado de los Datos y Validación Inicial (Incluye Preliminar)
    if data_overview_info:
        html += "<h2>2. Estado de los Datos y Validación Inicial</h2>"
        html += "<div class='section-box'>"
        html += "<p>Esta sección resume las características principales de los datos utilizados en el análisis y los problemas detectados durante la validación inicial.</p>"
        
        html += "<h3>2.1. Dimensiones del DataFrame</h3>"
        html += f"<p>Filas: {data_overview_info.get('shape', [0,0])[0]}, Columnas: {data_overview_info.get('shape', [0,0])[1]}</p>"

        html += "<h3>2.2. Tipos de Datos por Columna</h3>"
        df_types = pd.DataFrame(data_overview_info.get('column_types', {}).items(), columns=['Columna', 'Tipo de Dato'])
        html += df_types.to_html(index=False, border=1, justify="left", na_rep="-")

        html += "<h3>2.3. Resumen Estadístico (Columnas Numéricas)</h3>"
        numerical_summary_df = pd.DataFrame(data_overview_info.get('numerical_summary', {})).T
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

        # Sección de Análisis Exploratorio de Datos Preliminar
        html += "<h2>3. Análisis Exploratorio de Datos Preliminar</h2>"
        html += "<div class='section-box'>"
        html += "<p>Esta sección presenta visualizaciones clave para entender la distribución de las variables y las relaciones entre ellas antes de la modelización DEA.</p>"

        # Histogramas
        if 'preliminary_analysis_charts' in data_overview_info and 'histograms' in data_overview_info['preliminary_analysis_charts']:
            html += "<h3>3.1. Distribución de Variables (Histogramas)</h3>"
            html += "<p>Los histogramas muestran la distribución de cada columna numérica, ayudando a identificar asimetrías y posibles valores atípicos (outliers).</p>"
            for col_name, chart_json in data_overview_info['preliminary_analysis_charts']['histograms'].items():
                # Reconstruir figura de Plotly desde JSON y exportar a HTML
                fig = go.Figure(json.loads(chart_json))
                html += f"<div class='plotly-chart'>{fig.to_html(full_html=False, include_plotlyjs='cdn')}</div>"

        # Matriz de Correlación
        if 'correlation_matrix' in data_overview_info and data_overview_info['correlation_matrix']:
            html += "<h3>3.2. Matriz de Correlación (Mapa de Calor)</h3>"
            html += "<p>Este mapa de calor visualiza las relaciones lineales entre todas las variables numéricas. Valores cercanos a 1 o -1 indican fuerte correlación, lo que podría sugerir multicolinealidad entre inputs u outputs, un reto potencial en DEA.</p>"
            # Convertir el diccionario de la matriz de correlación a DataFrame para Plotly
            corr_matrix_dict = data_overview_info['correlation_matrix']
            # Plotly expects a square matrix. Need to ensure column order if recreating from dict
            keys = list(corr_matrix_dict.keys())
            corr_df = pd.DataFrame(corr_matrix_dict).T
            
            fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_df.values,
                    x=corr_df.columns.tolist(),
                    y=corr_df.index.tolist(),
                    colorscale='RdBu', # Corresponde a RdBu de px.colors.sequential
                    zmin=-1, zmax=1,
                    colorbar=dict(title="Correlación")
                ))
            fig_corr.update_layout(title="Matriz de Correlación entre Variables Numéricas")
            html += f"<div class='plotly-chart'>{fig_corr.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
        html += "</div>" # Cierra section-box para preliminar

    # Sección de Resultados DEA (dinámica)
    html += f"<h2>4. Resultados Numéricos del Modelo: {model_name}</h2>"
    if not df_results.empty:
        html += df_results.to_html(index=False, border=1, justify="left", na_rep="-")
    else:
        html += "<p>No se generaron resultados numéricos para este modelo.</p>"

    # Sección Árbol de Indagación y Justificaciones
    if not df_tree_data.empty:
        html += "<h2>5. Taller de Auditoría Metodológica</h2>"
        html += "<div class='section-box'>"
        html += "<p>Esta sección resume el mapa de razonamiento generado por la IA para auditar la robustez metodológica del análisis DEA, junto con las justificaciones aportadas por el investigador.</p>"
        
        html += "<h3>5.1. Estructura del Mapa de Razonamiento y Justificaciones</h3>"
        html += df_tree_data.to_html(index=False, border=1, justify="left")
        html += "</div>"
        
        if analysis_results.get('tree_explanation'):
            html += "<div class='section-box'>"
            html += "<h3>5.2. Explicación del Mapa (Generada por IA)</h3>"
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
            
            # Tipos de datos
            df_types = pd.DataFrame(data_overview_info.get('column_types', {}).items(), columns=['Columna', 'Tipo de Dato'])
            if not df_types.empty:
                df_types.to_excel(writer, sheet_name="Resumen_Datos", startrow=0, startcol=0, index=False)
                worksheet_data = writer.sheets["Resumen_Datos"]
                for idx, col in enumerate(df_types.columns):
                    max_len = max(
                        df_types[col].astype(str).map(len).max() if not df_types.empty else 0,
                        len(str(col))
                    ) + 2
                    worksheet_data.set_column(idx, idx, max_len)
            
            # Resumen Numérico (Transpuesto para mejor lectura en Excel)
            numerical_summary_df_excel = pd.DataFrame(data_overview_info.get('numerical_summary', {})).T
            if not numerical_summary_df_excel.empty:
                numerical_summary_df_excel.to_excel(writer, sheet_name="Resumen_Datos", startrow=len(df_types)+2, startcol=0)
                worksheet_data = writer.sheets["Resumen_Datos"]
                # Ajustar el ancho de las columnas para el resumen numérico
                for idx, col in enumerate(numerical_summary_df_excel.columns):
                    max_len = max(
                        numerical_summary_df_excel[col].astype(str).map(len).max() if not numerical_summary_df_excel.empty else 0,
                        len(str(col))
                    ) + 2
                    worksheet_data.set_column(idx, idx, max_len)


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
                # Escribir debajo de las tablas anteriores o en una nueva columna
                start_row_issues = 0
                start_col_issues = 0 # Asumimos que no hay solapamiento con las tablas principales que van en (0,0)
                if not df_types.empty:
                    start_col_issues = df_types.shape[1] + 2 # Colocar después de la tabla de tipos
                
                df_issues.to_excel(writer, sheet_name="Resumen_Datos", startrow=start_row_issues, startcol=start_col_issues, index=False)
                worksheet_data = writer.sheets["Resumen_Datos"]
                for idx, col in enumerate(df_issues.columns):
                    max_len = max(
                        df_issues[col].astype(str).map(len).max() if not df_issues.empty else 0,
                        len(str(col))
                    ) + 2
                    worksheet_data.set_column(idx + start_col_issues, idx + start_col_issues, max_len)

            # Matriz de Correlación
            if 'correlation_matrix' in data_overview_info and data_overview_info['correlation_matrix']:
                corr_matrix_dict = data_overview_info['correlation_matrix']
                corr_df = pd.DataFrame(corr_matrix_dict).T # Reconstruir DataFrame de correlación
                
                if not corr_df.empty:
                    # Escribir en una nueva hoja para la matriz de correlación
                    corr_df.to_excel(writer, sheet_name="Matriz_Correlacion", index=True)
                    worksheet_corr = writer.sheets["Matriz_Correlacion"]
                    for idx, col in enumerate(corr_df.columns):
                        max_len = max(
                            corr_df[col].astype(str).map(len).max() if not corr_df.empty else 0,
                            len(str(col))
                        ) + 2
                        worksheet_corr.set_column(idx + 1, idx + 1, max_len) # +1 para la columna de índice
                    
                    # Ajustar ancho de la primera columna (índice)
                    max_idx_len = max(len(str(s)) for s in corr_df.index) + 2
                    worksheet_corr.set_column(0, 0, max_idx_len)


    output.seek(0)
    return output
