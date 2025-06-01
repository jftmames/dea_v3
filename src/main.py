import streamlit as st
import pandas as pd

from data_validator import validate
from dea_analyzer import run_dea
from inquiry_engine import generate_inquiry, to_plotly_tree


# ---------- util: obtener la fila de la DMU ----------
def _get_row_by_dmu(df: pd.DataFrame, dmu: str) -> pd.DataFrame:
    """Devuelve la fila correspondiente a la DMU sin lanzar KeyError."""
    if "DMU" in df.columns:
        return df.loc[df["DMU"] == dmu]
    if dmu in df.index:
        return df.loc[[dmu]]
    mask = df.index.astype(str) == str(dmu)
    if mask.any():
        return df.loc[mask]
    return pd.DataFrame()  # no encontrada


# ---------- UI ----------
st.set_page_config(page_title="DEA Deliberativo MVP", layout="wide")
st.title("DEA Deliberativo – MVP")

# ------------------------------------------------------------------
# 1. Cargar CSV (área principal)
# ------------------------------------------------------------------
upload = st.file_uploader("Sube tu CSV", type="csv")

if upload:
    df = pd.read_csv(upload)
    st.subheader("Vista previa")
    st.dataframe(df.head(), use_container_width=True)

    # Validar que al menos haya 2 DMU si esperan usar BCC o super-eficiencia
    num_dmu = df.shape[0]

    # solo columnas numéricas para Inputs / Outputs
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.error("⚠️ El archivo no contiene columnas numéricas.")
        st.stop()

    # ------------------------------------------------------------------
    # Sidebar: controles de inputs/outputs y parámetros DEA
    # ------------------------------------------------------------------
    with st.sidebar:
        st.markdown("## Parámetros de entrada")

        inputs = st.multiselect(
            "Inputs",
            numeric_cols,
            default=numeric_cols[:-1],
            help="Columnas numéricas que actúan como insumos."
        )
        outputs = st.multiselect(
            "Outputs",
            numeric_cols,
            default=[numeric_cols[-1]],
            help="Columnas numéricas que actúan como productos."
        )

        st.markdown("## Configuración DEA")
        model = st.selectbox(
            "Modelo",
            ["CCR", "BCC"],
            index=0,
            help="CCR = retornos constantes (CRS); BCC = retornos variables (VRS)."
        )
        orientation = st.selectbox(
            "Orientación",
            ["input", "output"],
            index=0,
            help="input-oriented minimiza insumos; output-oriented maximiza productos."
        )
        super_eff = st.checkbox(
            "Super-eficiencia",
            value=False,
            help="Excluye la DMU actual del conjunto de comparación."
        )

        st.markdown("---")
        st.markdown("**Validaciones automáticas:**")
        if model == "BCC" and num_dmu < 2:
            st.error("Para BCC se necesitan al menos 2 DMU.")
        if super_eff and num_dmu < 2:
            st.error("Para super-eficiencia se necesitan al menos 2 DMU.")

        run_button = st.button(f"Ejecutar DEA ({model}-{orientation})")

    # ------------------------------------------------------------------
    # 2. Validación de datos (área principal)
    # ------------------------------------------------------------------
    if st.button("Validar datos"):
        result = validate(df, inputs, outputs)
        st.subheader("Resultado del validador")
        st.json(result)

    # ------------------------------------------------------------------
    # 3. Ejecutar DEA
    # ------------------------------------------------------------------
    if run_button:
        # Revalidar antes de ejecutar
        if model == "BCC" and num_dmu < 2:
            st.error("No se puede ejecutar BCC con menos de 2 DMU.")
            st.stop()
        if super_eff and num_dmu < 2:
            st.error("No se puede ejecutar super-eficiencia con menos de 2 DMU.")
            st.stop()

        with st.spinner(f"Optimizando DEA ({model}-{orientation})…"):
            try:
                res = run_dea(
                    df,
                    inputs,
                    outputs,
                    model=model,
                    orientation=orientation,
                    super_eff=super_eff,
                )
                if res["efficiency"].isna().all():
                    st.error("⚠️ El solver no devolvió soluciones válidas.")
                    st.stop()
            except (ValueError, KeyError) as e:
                st.error(f"❌ {e}")
                st.stop()

        # Guardamos en session_state el DataFrame y parámetros
        st.session_state["res_df"] = res
        st.session_state["dea_model"] = model
        st.session_state["dea_orientation"] = orientation
        st.session_state["dea_super_eff"] = super_eff

    # ------------------------------------------------------------------
    # 4. Mostrar Resultados DEA y habilitar exportaciones, árbol y EEE
    # ------------------------------------------------------------------
    if "res_df" in st.session_state:
        dea_df = st.session_state["res_df"]
        # Leemos los parámetros desde session_state
        model = st.session_state["dea_model"]
        orientation = st.session_state["dea_orientation"]
        super_eff = st.session_state["dea_super_eff"]

        # 4.1 Mostrar tabla de eficiencias
        st.subheader(f"Resultados DEA ({model}-{orientation})")
        st.dataframe(dea_df, use_container_width=True)

        # 4.2 Botón para exportar resultados DEA a CSV
        csv_dea = dea_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Descargar resultados DEA (CSV)",
            data=csv_dea,
            file_name="dea_results.csv",
            mime="text/csv",
        )

        # 4.3 Filtrar DMU ineficientes
        ineff_df = dea_df.query("efficiency < 1")
        if len(ineff_df) == 0:
            st.info("Todas las DMU son eficientes.")
        else:
            st.subheader("Generar Complejo de Indagación")
            dmu = st.selectbox("DMU ineficiente", ineff_df["DMU"])

            depth = st.slider(
                "Niveles del árbol",
                min_value=2,
                max_value=4,
                value=3,
                help="Cantidad de niveles jerárquicos en el árbol."
            )
            breadth = st.slider(
                "Subpreguntas / nodo",
                min_value=3,
                max_value=8,
                value=5,
                help="Máximo número de hijos por nodo."
            )

            if st.button("Crear árbol"):
                # localizamos la fila de la DMU
                row = _get_row_by_dmu(df, dmu)
                if row.empty:
                    st.error(f"No se encontró la DMU '{dmu}' en el DataFrame original.")
                    st.stop()

                # contexto rico para la IA
                context = {
                    "dmu": dmu,
                    "inputs": {c: float(row[c].values[0]) for c in inputs},
                    "outputs": {c: float(row[c].values[0]) for c in outputs},
                    "efficiency": float(
                        dea_df.set_index("DMU", drop=False).loc[dmu, "efficiency"]
                    ),
                    "peers": dea_df.query("efficiency == 1")["DMU"].tolist(),
                    "model": model,
                    "orientation": orientation,
                    "super_eff": super_eff,
                }

                with st.spinner("Generando árbol…"):
                    tree = generate_inquiry(
                        f"¿Por qué la {dmu} es ineficiente?",
                        context=context,
                        depth=depth,
                        breadth=breadth,
                        temperature=0.3,
                    )

                # 4.4 Mostrar árbol y JSON completo
                st.plotly_chart(to_plotly_tree(tree), use_container_width=True)
                with st.expander("JSON completo"):
                    st.json(tree)

                # 4.5 Cálculo y visualización del EEE
                from epistemic_metrics import compute_eee

                eee_score = compute_eee(tree, depth_limit=depth, breadth_limit=breadth)
                st.metric(label="Índice de Equilibrio Erotético (EEE)", value=eee_score)

                # 4.6 Exportaciones (CSV/JSON y reporte HTML)
                # A) Aplanar el árbol para CSV
                def _flatten_tree(tree: dict, parent: str = "") -> list[tuple[str, str]]:
                    rows = []
                    for q, kids in tree.items():
                        rows.append((q, parent))
                        if isinstance(kids, dict):
                            rows.extend(_flatten_tree(kids, q))
                    return rows

                flat = _flatten_tree(tree)
                df_tree = pd.DataFrame(flat, columns=["question", "parent"])
                csv_tree = df_tree.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Descargar árbol (CSV)",
                    data=csv_tree,
                    file_name="inquiry_tree.csv",
                    mime="text/csv",
                )

                # B) Descargar árbol en JSON
                import json
                json_tree = json.dumps(tree, ensure_ascii=False, indent=2).encode("utf-8")
                st.download_button(
                    label="📥 Descargar árbol (JSON)",
                    data=json_tree,
                    file_name="inquiry_tree.json",
                    mime="application/json",
                )

                # C) CSV con metadatos EEE
                eee_meta = {
                    "DMU": dmu,
                    "model": model,
                    "orientation": orientation,
                    "super_eff": super_eff,
                    "depth": depth,
                    "breadth": breadth,
                    "EEE_score": eee_score,
                }
                df_eee = pd.DataFrame.from_records([eee_meta])
                csv_eee = df_eee.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Descargar EEE (CSV)",
                    data=csv_eee,
                    file_name="eee_meta.csv",
                    mime="text/csv",
                )

                # D) Reporte HTML completo
                from report_generator import generate_html_report

                html_report = generate_html_report(
                    df_dea=dea_df, df_tree=df_tree, df_eee=df_eee
                )
                html_bytes = html_report.encode("utf-8")
                st.download_button(
                    label="📥 Descargar reporte completo (HTML)",
                    data=html_bytes,
                    file_name="reporte_dea_deliberativo.html",
                    mime="text/html",
                )
