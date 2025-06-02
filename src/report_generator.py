import pandas as pd
import datetime
from io import BytesIO

def generate_html_report(
    df_dea: pd.DataFrame,
    df_tree: pd.DataFrame,
    df_eee: pd.DataFrame
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
    html = (
        "<html><head><meta charset='utf-8'>"
        "<style>"
        "table {border-collapse: collapse; width: 100%;}"
        "th, td {border: 1px solid #ddd; padding: 8px;}"
        "th {background-color: #f2f2f2;}"
        "</style>"
        f"<title>Reporte DEA Deliberativo – {fecha}</title></head><body>"
    )
    html += f"<h1>Reporte DEA Deliberativo – {fecha}</h1>"

    # Sección DEA
    html += "<h2>1. Resultados DEA</h2>"
    html += df_dea.to_html(index=False, border=1, justify="left")

    # Sección Árbol
    html += "<h2>2. Estructura de Complejo de Indagación</h2>"
    html += df_tree.to_html(index=False, border=1, justify="left")

    # Sección EEE
    html += "<h2>3. Métrico EEE y Metadatos</h2>"
    html += df_eee.to_html(index=False, border=1, justify="left")

    html += "</body></html>"
    return html


def generate_excel_report(
    df_dea: pd.DataFrame,
    df_tree: pd.DataFrame,
    df_eee: pd.DataFrame
) -> BytesIO:
    """
    Genera un reporte Excel en memoria con varias pestañas:
      - "DEA Results": resultados DEA
      - "Inquiry Tree": tabla de árbol de indagación
      - "EEE Metrics": tabla de EEE
    Retorna un objeto BytesIO con el contenido del archivo .xlsx.
    """
    output = BytesIO()
    # Usamos engine xlsxwriter por compatibilidad
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # 1) Hoja DEA Results
        df_dea.to_excel(writer, sheet_name="DEA Results", index=False)
        worksheet_dea = writer.sheets["DEA Results"]
        for idx, col in enumerate(df_dea.columns):
            max_len = (
                df_dea[col].astype(str).map(len).max()
                if not df_dea.empty else 0
            )
            max_len = max(max_len, len(col)) + 2
            worksheet_dea.set_column(idx, idx, max_len)

        # 2) Hoja Inquiry Tree
        df_tree.to_excel(writer, sheet_name="Inquiry Tree", index=False)
        worksheet_tree = writer.sheets["Inquiry Tree"]
        for idx, col in enumerate(df_tree.columns):
            max_len = (
                df_tree[col].astype(str).map(len).max()
                if not df_tree.empty else 0
            )
            max_len = max(max_len, len(col)) + 2
            worksheet_tree.set_column(idx, idx, max_len)

        # 3) Hoja EEE Metrics
        df_eee.to_excel(writer, sheet_name="EEE Metrics", index=False)
        worksheet_eee = writer.sheets["EEE Metrics"]
        for idx, col in enumerate(df_eee.columns):
            max_len = (
                df_eee[col].astype(str).map(len).max()
                if not df_eee.empty else 0
            )
            max_len = max(max_len, len(col)) + 2
            worksheet_eee.set_column(idx, idx, max_len)
        # Nota: no llamamos a writer.save(), el with se encarga

    output.seek(0)
    return output
