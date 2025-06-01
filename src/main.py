import streamlit as st
import pandas as pd
import json

from data_validator import validate
from dea_analyzer import run_dea
from inquiry_engine import generate_inquiry, to_plotly_tree
from results import (
    plot_efficiency_histogram,
    plot_3d_inputs_outputs,
    plot_benchmark_spider,
)
from openai_helpers import explain_orientation, recommend_alternatives


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


# ---------- UI principal ----------
st.set_page_config(page_title="DEA Deliberativo MVP", layout="wide")
st.title("DEA Deliberativo – MVP")

# ------------------------------------------------------------------
# 1. Cargar CSV (área principal)
# ------------------------------------------------------------------
upload = st.file_uploader("Sube tu CSV", type="csv")

if upload:
    df = pd.read_csv(upload)

    # — Si no existe "DMU", pero sí "DMU_ID", renombramos/duplicamos para unificar —
    if "DMU" not in df.columns and "DMU_ID" in df.columns:
        df["DMU"] = df["DMU_ID"]

    st.subheader("Vista previa")
    st.dataframe(df.head(), use_container_width=True)

    num_dmu = df.shape[0]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.error("⚠️ El archivo no contiene columnas numéricas.")
        st.stop()

    # ------------------------------------------------------------------
    # Sidebar: controles de Inputs/Outputs y parámetros DEA
    # ------------------------------------------------------------------
    with st.sidebar:
        st.markdown("## Parámetros de entrada")

        inputs = st.multiselect(
            "Inputs",
            numeric_cols,
            default=numeric_cols[:-1],
            help="Columnas numéricas que se usarán como insumos (deben existir)."
        )
        outputs = st.multiselect(
            "Outputs",
            numeric_cols,
            default=[numeric_cols[-1]],
            help="Columnas numéricas que se usarán como productos (deben existir)."
        )

        st.markdown("---")
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
            help="Si se activa, la DMU actual se excluye al calcular su eficiencia."
        )

        st.markdown("---")
        st.markdown("**Validaciones automáticas:**")
        if model == "BCC" and num_dmu < 2:
            st.error("Para BCC se necesitan al menos 2 DMU.")
        if super_eff and num_dmu < 2:
            st.error("Para super-eficiencia se necesitan al menos 2 DMU.")

        run_button = st.button(f"Ejecutar DEA ({model}-{orientation})")

    # ------------------------------------------------------------------
    # 2. Validar datos (área principal)
    # ------------------------------------------------------------------
    if st.button("Validar datos"):
        result = validate(df, inputs, outputs)
        if result.get("issues"):
            st.error("❌ Problemas encontrados:")
            st.json(result)
        else:
            st.success("✅ Datos válidos")
            st.json(result)

    # ------------------------------------------------------------------
    # 2.1. Sugerir alternativas de Inputs/Outputs (nuevo)
    # ------------------------------------------------------------------
    if st.button("Sugerir alternativas de Inputs/Outputs"):
        if not inputs or not outputs:
            st.error("Selecciona primero al menos un Input y un Output.")
        else:
            rec = recommend_alternatives(
                df_columns=numeric_cols,
                inputs=inputs,
                outputs=outputs
            )
            # Guardamos la última recomendación en sesión
            st.session_state["last_reco"] = rec
            if rec.get("recommend_inputs") is not None:
                st.subheader("Recomendaciones de Inputs")
                st.write(rec["recommend_inputs"])
                st.subheader("Recomendaciones de Outputs")
                st.write(rec["recommend_outputs"])
            else:
                st.subheader("Recomendación (texto libre)")
                st.write(rec.get("text", "La IA no devolvió sugerencias en formato JSON."))

    # ------------------------------------------------------------------
    # 3. Ejecutar DEA
    # ------------------------------------------------------------------
    st.session_state["orig_df"] = df
    st.session_state["inputs"] = inputs
    st.session_state["outputs"] = outputs

    if run_button:
        # Revalidar antes de ejecutar
        if model == "BCC" and num_dmu < 2:
            st.error("No se puede ejecutar BCC con menos de 2 DMU.")
            st.stop()
        if super_eff and num_dmu < 2:
            st.error("No se puede ejecutar super-eficiencia con menos de 2 DMU.")
            st.stop()

        with st.spinner(f"Calculando eficiencias DEA para {num_dmu} DMU…"):
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

        # Guardamos en sesión el DataFrame y parámetros
        st.session_state["res_df"] = res
        st.session_state["dea_model"] = model
        st.session_state["dea_orientation"] = orientation
        st.session_state["dea_super_eff"] = super_eff

        st.success("✅ DEA calculado correctamente")

    # ------------------------------------------------------------------
    # 4. Mostrar Resultados DEA y habilitar exportaciones, árbol, EEE y gráficos
    # ------------------------------------------------------------------
    if "res_df" in st.session_state:
        dea_df = st.session_state["res_df"]
        model = st.session_state["dea_model"]
        orientation = st.session_state["dea_orientation"]
        super_eff = st.session_state["dea_super_eff"]

        # 4.1 Tabla de eficiencias
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

        # 4.3 Visualizaciones interactivas
        st.markdown("---")
        st.subheader("Visualizaciones interactivas")

        # A) Histograma de eficiencias
        with st.expander("📊 Histograma de Eficiencias"):
            hist_fig = plot_efficiency_histogram(dea_df, bins=20)
            st.plotly_chart(hist_fig, use_container_width=True)

        # B) Scatter 3D inputs vs outputs
        if len(inputs) >= 2 and len(outputs) >= 1:
            with st.expander("🔍 Scatter 3D Inputs vs Output (coloreado por eficiencia)"):
                try:
                    scatter3d_fig = plot_3d_inputs_outputs(df, inputs, outputs, dea_df)
                    st.plotly_chart(scatter3d_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error al generar Scatter 3D: {e}")
        else:
            st.info("Se requieren ≥2 Inputs y ≥1 Output para el Scatter 3D.")

        # C) Benchmark Spider
        with st.expander("🕸️ Benchmark Spider para DMU seleccionada"):
            if dea_df.query("efficiency == 1").empty:
                st.info("No hay DMU eficientes (efficiency == 1) para benchmark.")
            else:
                selected_dmu = st.selectbox("Elige DMU para spider", dea_df["DMU"])
                try:
                    # Unimos dea_df (que ya tiene columna 'DMU') con df original
                    merged_for_spider = dea_df.merge(
                        df, on="DMU", how="left"
                    )
                    spider_fig = plot_benchmark_spider(
                        merged_for_spider,
                        selected_dmu,
                        inputs,
                        outputs
                    )
                    st.plotly_chart(spider_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Imposible generar Spider: {e}")

        # 4.4 Filtrar DMU ineficientes
        ineff_df = dea_df.query("efficiency < 1")
        if len(ineff_df) == 0:
            st.info("Todas las DMU son eficientes.")
        else:
            st.subheader("Generar Complejo de Indagación")
            dmu = st.selectbox("DMU ineficiente", ineff_df["DMU"])

            # Definimos sliders con límites razonables
            depth = st.slider(
                "Niveles del árbol",
                min_value=2,
                max_value=4,
                value=3,
                help="Define cuántos niveles jerárquicos tendrá el árbol (2–4)."
            )
            breadth = st.slider(
                "Subpreguntas / nodo",
                min_value=2,
                max_value=6,
                value=3,
                help="Define cuántos hijos máximo puede tener cada nodo (2–6)."
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

                with st.spinner(f"Construyendo árbol (niveles={depth}, hijos={breadth})…"):
                    tree = generate_inquiry(
                        f"¿Por qué la {dmu} es ineficiente?",
                        context=context,
                        depth=depth,
                        breadth=breadth,
                        temperature=0.3,
                    )

                # Guardamos el árbol y los parámetros depth/breadth en sesión
                st.session_state["last_tree"] = tree
                st.session_state["last_depth"] = depth
                st.session_state["last_breadth"] = breadth
                st.session_state["last_dmu"] = dmu
                st.success("✅ Árbol generado correctamente")

        # 4.5 Mostrar árbol si existe
        if "last_tree" in st.session_state:
            tree = st.session_state["last_tree"]
            depth = st.session_state.get("last_depth", 2)
            breadth = st.session_state.get("last_breadth", 2)
            dmu = st.session_state.get("last_dmu", None)

            st.subheader("Árbol de Indagación (último generado)")
            st.plotly_chart(to_plotly_tree(tree), use_container_width=True)

            # 4.6 JSON editable
            st.markdown("**Editar árbol JSON (opcional)**")
            json_text = json.dumps(tree, ensure_ascii=False, indent=2)
            edited = st.text_area("Árbol JSON", value=json_text, height=200)
            if st.button("Actualizar árbol"):
                try:
                    new_tree = json.loads(edited)
                    if isinstance(new_tree, dict) and new_tree:
                        st.session_state["last_tree"] = new_tree
                        st.success("✅ Árbol actualizado correctamente.")
                    else:
                        st.error("❌ El JSON debe ser un objeto con al menos un nodo.")
                except Exception as e:
                    st.error(f"❌ JSON inválido: {e}")

            with st.expander("Ver JSON completo"):
                st.json(st.session_state["last_tree"])

            # 4.7 Cálculo y visualización del EEE
            from epistemic_metrics import compute_eee

            eee_score = compute_eee(tree, depth_limit=depth, breadth_limit=breadth)
            st.metric(label="Índice de Equilibrio Erotético (EEE)", value=eee_score)

            # 4.8 Exportaciones (CSV/JSON y reporte HTML)
            def _flatten_tree(tree: dict, parent: str = "") -> list[tuple[str, str]]:
                rows = []
                for q, kids in tree.items():
                    rows.append((q, parent))
                    if isinstance(kids, dict):
                        rows.extend(_flatten_tree(kids, q))
                return rows

            flat = _flatten_tree(tree)
            df_tree = pd.DataFrame(flat, columns=["question", "parent"])

            # A) Descargar árbol en CSV
            csv_tree = df_tree.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Descargar árbol (CSV)",
                data=csv_tree,
                file_name="inquiry_tree.csv",
                mime="text/csv",
            )

            # B) Descargar árbol en JSON
            json_tree_bytes = json.dumps(tree, ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button(
                label="📥 Descargar árbol (JSON)",
                data=json_tree_bytes,
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
