import pandas as pd
import datetime
from io import BytesIO

# Nota: Para guardar gráficos de Plotly en Excel, se necesitaría una librería adicional
# como 'xlsxwriter' y manejar los gráficos como imágenes. Por simplicidad en este paso,
# el informe Excel se centrará en los datos tabulares, que es lo más crítico.

def generate_html_report(
    analysis_results: dict,
    inquiry_tree: dict | None = None
) -> str:
    """
    Genera un string con contenido HTML que se adapta a los resultados del análisis.
    
    Args:
        analysis_results (dict): El diccionario de resultados de analysis_dispatcher.
        inquiry_tree (dict): El árbol de indagación generado por la IA.
    """
    # Convertir None en DataFrame vacío
    df_results = analysis_results.get("main_df", pd.DataFrame())
    model_name = analysis_results.get("model_name", "Análisis DEA")
    
    df_tree_data = pd.DataFrame()
    if inquiry_tree:
        # Aplanar el árbol para una representación tabular simple
        rows = []
        root_question = list(inquiry_tree.keys())[0]
        def walk(node, parent):
            for pregunta, hijos in node.items():
                rows.append({"parent": parent, "child": pregunta})
                if isinstance(hijos, dict): walk(hijos, pregunta)
        walk(inquiry_tree, "")
        df_tree_data = pd.DataFrame(rows)


    fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = (
        "<html><head><meta charset='utf-8'>"
        "<style>"
        "body {font-family: sans-serif;}"
        "table {border-collapse: collapse; width: 100%; margin-bottom: 2em;}"
        "th, td {border: 1px solid #ddd; padding: 8px; text-align: left;}"
        "th {background-color: #f2f2f2;}"
        "h1, h2 {color: #333;}"
        "</style>"
        f"<title>Reporte DEA Deliberativo – {fecha}</title></head><body>"
    )
    html += f"<h1>Reporte DEA Deliberativo – {fecha}</h1>"

    # Sección de Resultados DEA (dinámica)
    html += f"<h2>1. Resultados del Modelo: {model_name}</h2>"
    if not df_results.empty:
        html += df_results.to_html(index=False, border=1, justify="left", na_rep="-")
    else:
        html += "<p>No se generaron resultados numéricos para este modelo.</p>"

    # Sección Árbol de Indagación
    if not df_tree_data.empty:
        html += "<h2>2. Estructura del Mapa de Razonamiento</h2>"
        html += df_tree_data.to_html(index=False, border=1, justify="left")

    html += "</body></html>"
    return html


def generate_excel_report(
    analysis_results: dict,
    inquiry_tree: dict | None = None
) -> BytesIO:
    """
    Genera un reporte Excel en memoria con pestañas dinámicas.
    
    Args:
        analysis_results (dict): El diccionario de resultados de analysis_dispatcher.
        inquiry_tree (dict): El árbol de indagación generado por la IA.
    """
    df_results = analysis_results.get("main_df", pd.DataFrame())
    model_name = analysis_results.get("model_name", "Resultados")
    # Limpiar el nombre del modelo para que sea un nombre de hoja válido
    sheet_name = ''.join(c for c in model_name if c.isalnum())[:30]

    df_tree_data = pd.DataFrame()
    if inquiry_tree:
        rows = []
        def walk(node, parent):
            for pregunta, hijos in node.items():
                rows.append({"parent": parent, "child": pregunta})
                if isinstance(hijos, dict): walk(hijos, pregunta)
        walk(inquiry_tree, "")
        df_tree_data = pd.DataFrame(rows)


    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # 1) Hoja de Resultados (nombre dinámico)
        df_results.to_excel(writer, sheet_name=sheet_name, index=False)
        worksheet_results = writer.sheets[sheet_name]
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
