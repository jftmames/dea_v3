# ---------- src/report_generator.py ----------
import pandas as pd
import datetime

def generate_html_report(
    df_dea: pd.DataFrame,
    df_tree: pd.DataFrame,
    df_eee: pd.DataFrame,
) -> str:
    """
    Genera un string con contenido HTML que incluye:
      - Título con fecha
      - Tabla de resultados DEA
      - Tabla de árbol de indagación (padre/hijo)
      - Tabla de metadatos EEE
    Retorna HTML listo para escribir a disco o servir como descarga.
    """
    fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = f"<html><head><meta charset='utf-8'><title>Reporte DEA Deliberativo</title></head><body>"
    html += f"<h1>Reporte DEA Deliberativo – {fecha}</h1>"

    # Sección DEA
    html += "<h2>1. Resultados DEA</h2>"
    html += df_dea.to_html(index=False, border=1)

    # Sección Árbol
    html += "<h2>2. Estructura de Complejo de Indagación</h2>"
    html += df_tree.to_html(index=False, border=1)

    # Sección EEE
    html += "<h2>3. Métrico EEE y Metadatos</h2>"
    html += df_eee.to_html(index=False, border=1)

    html += "</body></html>"
    return html
