# src/report_generator.py

import pandas as pd
import datetime
from io import BytesIO
from pptx import Presentation
from pptx.util import Inches
import matplotlib.pyplot as plt

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
    html = (
        "<html><head><meta charset='utf-8'>"
        "<title>Reporte DEA Deliberativo</title></head><body>"
    )
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


def generate_excel_report(
    df_dea: pd.DataFrame,
    df_tree: pd.DataFrame,
    df_eee: pd.DataFrame
) -> BytesIO:
    """
    Genera un archivo Excel en memoria con pestañas:
      - 'DEA Results'
      - 'Inquiry Tree'
      - 'EEE Metrics'
    Retorna un BytesIO conteniendo el Excel.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # Escribir cada DataFrame en su hoja correspondiente
        df_dea.to_excel(writer, sheet_name="DEA Results", index=False)
        df_tree.to_excel(writer, sheet_name="Inquiry Tree", index=False)
        df_eee.to_excel(writer, sheet_name="EEE Metrics", index=False)

        # Formato: cabeceras en negrita, fondo gris, ancho de columna ajustado
        workbook = writer.book
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#D3D3D3"})
        for sheet_name in ["DEA Results", "Inquiry Tree", "EEE Metrics"]:
            worksheet = writer.sheets[sheet_name]
            # Aplicar formato a las cabeceras
            for col_num, col_name in enumerate(writer.sheets[sheet_name].table.columns
                                              if hasattr(writer.sheets[sheet_name], "table") 
                                              else df_dea.columns):
                worksheet.write(0, col_num, col_name, header_fmt)
            # Ajustar ancho de columnas (asume 15 caracteres por defecto)
            max_col = (
                len(writer.sheets[sheet_name].columns) 
                if hasattr(writer.sheets[sheet_name], "columns") 
                else df_dea.shape[1]
            )
            worksheet.set_column(0, max_col - 1, 15)

    output.seek(0)
    return output


def generate_pptx_report(
    df_dea: pd.DataFrame,
    df_tree: pd.DataFrame,
    df_eee: pd.DataFrame
) -> BytesIO:
    """
    Genera un archivo PPTX en memoria con slides para:
      1. Resultados DEA (tabla)
      2. Árbol de indagación (tabla)
      3. Métricas EEE (tabla)
    Retorna un BytesIO conteniendo el PPTX.
    """
    prs = Presentation()
    blank_slide_layout = prs.slide_layouts[6]  # layout en blanco

    def add_table_slide(df: pd.DataFrame, title: str):
        slide = prs.slides.add_slide(blank_slide_layout)
        slide.shapes.title.text = title

        rows, cols = df.shape
        left = Inches(0.5)
        top = Inches(1.5)
        width = Inches(9)
        height = Inches(5)
        table_shape = slide.shapes.add_table(rows + 1, cols, left, top, width, height)
        table = table_shape.table

        # Escribir cabeceras
        for j, col_name in enumerate(df.columns):
            table.cell(0, j).text = str(col_name)
        # Escribir datos
        for i in range(rows):
            for j, col_name in enumerate(df.columns):
                table.cell(i + 1, j).text = str(df.iloc[i, j])

    # Añadir slides para cada sección
    add_table_slide(df_dea, "Resultados DEA")
    add_table_slide(df_tree, "Árbol de Indagación")
    add_table_slide(df_eee, "Métricas EEE")

    output = BytesIO()
    prs.save(output)
    output.seek(0)
    return output
