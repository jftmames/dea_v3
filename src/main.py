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
# 1. Cargar CSV
# ------------------------------------------------------------------
upload = st.file_uploader("Sube tu CSV", type="csv")

if upload:
    df = pd.read_csv(upload)
    st.subheader("Vista previa")
    st.dataframe(df.head(), use_container_width=True)

    # solo columnas numéricas para Inputs / Outputs
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.error("⚠️ El archivo no contiene columnas numéricas.")
        st.stop()

    st.markdown("### Selecciona columnas de **inputs** y **outputs**")
    st.info("⚠️ Solo columnas numéricas funcionarán en DEA.")

    inputs = st.multiselect("Inputs", numeric_cols, default=numeric_cols[:-1])
    outputs = st.multiselect("Outputs", numeric_cols, default=[numeric_cols[-1]])

    # ------------------------------------------------------------------
    # 2. Validación de datos
    # ------------------------------------------------------------------
    if st.button("Validar datos"):
        result = validate(df, inputs, outputs)
        st.subheader("Resultado del validador")
        st.json(result)

    # ------------------------------------------------------------------
    # 3. Parámetros de DEA: modelo, orientación y super-eficiencia
    # ------------------------------------------------------------------
    st.markdown("### Configuración del modelo DEA")
    col1, col2, col3 = st.columns(3)

    with col1:
        model = st.selectbox(
            "Modelo",
            ["CCR", "BCC"],
            index=0,
            help="CCR = retornos constantes (CRS); BCC = retornos variables (VRS)"
        )
    with col2:
        orientation = st.selectbox(
            "Orientación",
            ["input", "output"],
            index=0,
            help="input‐oriented o output‐oriented"
        )
    with col3:
        super_eff = st.checkbox(
            "Super-eficiencia",
            value=False,
            help="Excluir DMU actual para super-eficiencia"
        )

    # ------------------------------------------------------------------
    # 4. Ejecutar DEA
    # ------------------------------------------------------------------
    if st.button(f"Ejecutar DEA ({model}-{orientation})"):
        with st.spinner("Optimizando…"):
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

        st.session_state["res_df"] = res
        st.subheader(f"Eficiencias DEA ({model}-{orientation})")
        st.dataframe(res, use_container_width=True)

# ------------------------------------------------------------------
# 5. Complejos de Indagación y exportaciones
# ------------------------------------------------------------------
if "res_df" in st.session_state:
    # Mostrar resultados DEA y botón de descarga
    dea_df = st.session_state["res_df"]
    st.subheader("Resultados DEA")
    st.dataframe(dea_df, use_container_width=True)

    # Botón para exportar DEA a CSV
    csv_dea = dea_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Descargar resultados DEA (CSV)",
        data=csv_dea,
        file_name="dea_results.csv",
        mime="text/csv",
    )

    # Filtrar DMU ineficientes
    ineff_df = dea_df.query("efficiency < 1")
    if len(ineff_df) == 0:
        st.info("Todas las DMU son eficientes.")
    else:
        st.subheader("Generar Complejo de Indagación")
        dmu = st.selectbox("DMU ineficiente", ineff_df["DMU"])

        depth = st.slider("Niveles", 2, 4, 3)
        breadth = st.slider("Subpreguntas / nodo", 3, 8, 5)

        if st.button("Crear árbol"):

            # localizar la fila de la DMU de forma segura
            row = _get_row_by_dmu(df, dmu)
            if row.empty:
                st.error(f"No se encontró la DMU '{dmu}' en el DataFrame original.")
                st.stop()

            # contexto rico para el modelo
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

            st.plotly_chart(to_plotly_tree(tree), use_container_width=True)
            with st.expander("JSON completo"):
                st.json(tree)

            # Cálculo y visualización del EEE
            from epistemic_metrics import compute_eee

            eee_score = compute_eee(tree, depth_limit=depth, breadth_limit=breadth)
            st.metric(label="Índice de Equilibrio Erotético (EEE)", value=eee_score)

            # --- 6.1 Exportar árbol y EEE ---
            # Aplanar el árbol para CSV
            def _flatten_tree(tree: dict, parent: str = "") -> list[tuple[str, str]]:
                rows = []
                for q, kids in tree.items():
                    rows.append((q, parent))
                    if isinstance(kids, dict):
                        rows.extend(_flatten_tree(kids, q))
                return rows

            flat = _flatten_tree(tree)  # [(pregunta, padre), …]
            df_tree = pd.DataFrame(flat, columns=["question", "parent"])

            csv_tree = df_tree.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Descargar árbol (CSV)",
                data=csv_tree,
                file_name="inquiry_tree.csv",
                mime="text/csv",
            )

            # CSV con el EEE y configuración
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

            # --- 6.2 Generar y descargar reporte completo en HTML ---
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
