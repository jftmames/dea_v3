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


# ---------- Util: obtener la fila de la DMU ----------
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
st.markdown(
    "Este flujo único permite: **(1)** subir datos, ejecutar DEA, generar árbol de indagación; "
    "y **(2)**, una vez generado el árbol, reajustar el modelo DEA sin cambiar de pestaña."
)

# ------------------------------------------------------------------
# 1. Cargar CSV (área principal)
# ------------------------------------------------------------------
upload = st.file_uploader("Sube tu CSV", type="csv")
if upload:
    df = pd.read_csv(upload)

    # 1.a) Si no existe columna 'DMU', la creamos desde el índice
    if "DMU" not in df.columns:
        df.insert(0, "DMU", df.index.astype(str))

    # Guardamos el DataFrame original en sesión, para “Reajustar” más adelante
    st.session_state["orig_df"] = df.copy()

    st.subheader("Vista previa del CSV")
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
            help="Columnas numéricas que se usarán como insumos (mínimo 1)."
        )
        outputs = st.multiselect(
            "Outputs",
            numeric_cols,
            default=[numeric_cols[-1]],
            help="Columnas numéricas que se usarán como productos (mínimo 1)."
        )

        # Guardamos esas selecciones en sesión para reutilizar luego
        st.session_state["inputs"] = inputs
        st.session_state["outputs"] = outputs

        st.markdown("---")
        st.markdown("## Configuración DEA")

        model = st.selectbox(
            "Modelo",
            ["CCR", "BCC"],
            index=0,
            help="CCR = retornos constantes; BCC = retornos variables."
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

        # Guardamos en sesión el modelo/orientación/super_eff
        st.session_state["dea_model"] = model
        st.session_state["dea_orientation"] = orientation
        st.session_state["dea_super_eff"] = super_eff

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
    # 3. Sugerir alternativas de Inputs/Outputs (con OpenAI)
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
            if rec.get("recommend_inputs") is not None:
                st.subheader("Recomendaciones de Inputs")
                st.write(rec["recommend_inputs"])
                st.subheader("Recomendaciones de Outputs")
                st.write(rec["recommend_outputs"])
            else:
                st.subheader("Recomendación (texto libre)")
                st.write(rec.get("text", "La IA no devolvió sugerencias en formato JSON."))

    # ------------------------------------------------------------------
    # 4. Ejecutar DEA
    # ------------------------------------------------------------------
    if run_button:
        # Revalidar antes de ejecutar
        if model == "BCC" and num_dmu < 2:
            st.error("No se puede ejecutar BCC con menos de 2 DMU.")
            st.stop()
        if super_eff and num_dmu < 2:
            st.error("No se puede ejecutar super-eficiencia con menos de 2 DMU.")
            st.stop()
        if not inputs or not outputs:
            st.error("Debe seleccionar al menos un Input y un Output.")
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

        # Guardamos en sesión el DataFrame con eficiencias y parámetros
        st.session_state["res_df"] = res.copy()
        st.session_state["dea_model"] = model
        st.session_state["dea_orientation"] = orientation
        st.session_state["dea_super_eff"] = super_eff

        st.success("✅ DEA calculado correctamente")

    # ------------------------------------------------------------------
    # 5. Mostrar Resultados DEA y habilitar exportaciones, árbol y EEE
    # ------------------------------------------------------------------
    if "res_df" in st.session_state:
        dea_df = st.session_state["res_df"]
        model = st.session_state["dea_model"]
        orientation = st.session_state["dea_orientation"]
        super_eff = st.session_state["dea_super_eff"]

        st.markdown("---")
        st.subheader(f"Resultados DEA ({model}-{orientation})")
        st.dataframe(dea_df, use_container_width=True)

        # 5.1 Exportar DEA a CSV
        csv_dea = dea_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Descargar resultados DEA (CSV)",
            data=csv_dea,
            file_name="dea_results.csv",
            mime="text/csv",
        )

        # 5.2 Visualizaciones interactivas
st.markdown("---")
st.subheader("Visualizaciones interactivas")

# A) Histograma de eficiencias
with st.expander("📊 Histograma de Eficiencias"):
    hist_fig = plot_efficiency_histogram(dea_df, bins=20)
    st.plotly_chart(
        hist_fig,
        use_container_width=True,
        key="plot_histogram_efficiencies"
    )

# B) Scatter 3D inputs vs outputs (coloreado por eficiencia)
if len(st.session_state["inputs"]) >= 2 and len(st.session_state["outputs"]) >= 1:
    with st.expander("🔍 Scatter 3D Inputs vs Output (coloreado por eficiencia)"):
        try:
            scatter3d_fig = plot_3d_inputs_outputs(
                df,
                st.session_state["inputs"],
                st.session_state["outputs"],
                dea_df
            )
            st.plotly_chart(
                scatter3d_fig,
                use_container_width=True,
                key="plot_scatter3d_efficiency"
            )
        except Exception as e:
            st.error(f"Error al generar Scatter 3D: {e}")
else:
    st.info("Se requieren ≥2 Inputs y ≥1 Output para el Scatter 3D.")

# C) Benchmark Spider
with st.expander("🕸️ Benchmark Spider para DMU seleccionada"):
    if dea_df.query("efficiency == 1").empty:
        st.info("No hay DMU eficientes (efficiency == 1) para benchmark.")
    else:
        selected_dmu = st.selectbox(
            "Elige DMU para spider",
            dea_df["DMU"],
            key="select_spider_dmu"
        )
        try:
            merged_for_spider = dea_df.merge(
                df[st.session_state["inputs"] + st.session_state["outputs"] + ["DMU"]],
                on="DMU",
                how="left"
            )
            spider_fig = plot_benchmark_spider(
                merged_for_spider,
                selected_dmu,
                st.session_state["inputs"],
                st.session_state["outputs"]
            )
            st.plotly_chart(
                spider_fig,
                use_container_width=True,
                key="plot_benchmark_spider"
            )
        except Exception as e:
            st.error(f"Imposible generar Benchmark Spider: {e}")

        # 5.3 Filtrar DMU ineficientes y generar árbol
        ineff_df = dea_df.query("efficiency < 1")
        if len(ineff_df) == 0:
            st.info("Todas las DMU son eficientes.")
        else:
            st.markdown("---")
            st.subheader("Generar Complejo de Indagación para una DMU ineficiente")
            dmu = st.selectbox("DMU ineficiente", ineff_df["DMU"], key="dmu_ineff")

            depth = st.slider(
                "Niveles del árbol",
                min_value=2,
                max_value=4,
                value=3,
                help="Cuántos niveles jerárquicos tendrá el árbol (2–4).",
                key="slider_depth"
            )
            breadth = st.slider(
                "Subpreguntas / nodo",
                min_value=2,
                max_value=6,
                value=3,
                help="Cuántos hijos máximo puede tener cada nodo (2–6).",
                key="slider_breadth"
            )

            if st.button("Crear árbol", key="btn_create_tree"):
                row = _get_row_by_dmu(df, dmu)
                if row.empty:
                    st.error(f"No se encontró la DMU '{dmu}' en el DataFrame original.")
                    st.stop()

                context = {
                    "dmu": dmu,
                    "inputs": {c: float(row[c].values[0]) for c in st.session_state["inputs"]},
                    "outputs": {c: float(row[c].values[0]) for c in st.session_state["outputs"]},
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

                # Guardamos en sesión el árbol y sus parámetros para “Reajustar Modelo”
                st.session_state["last_tree"] = tree
                st.session_state["last_dmu"] = dmu
                st.session_state["last_depth"] = depth
                st.session_state["last_breadth"] = breadth

                st.success("✅ Árbol generado correctamente")

        # 5.4 Mostrar árbol y EEE (si existe)
        if "last_tree" in st.session_state:
            tree = st.session_state["last_tree"]
            dmu = st.session_state["last_dmu"]
            depth = st.session_state["last_depth"]
            breadth = st.session_state["last_breadth"]

            st.markdown("---")
            st.subheader("Árbol de Indagación (último generado)")
            st.plotly_chart(to_plotly_tree(tree), use_container_width=True, key="tree_chart")

            # JSON editable del árbol
            st.markdown("**Editar árbol JSON (opcional)**")
            json_text = json.dumps(tree, ensure_ascii=False, indent=2)
            edited = st.text_area("Árbol JSON", value=json_text, height=200, key="json_edit")
            if st.button("Actualizar árbol", key="btn_update_tree"):
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
                st.json(st.session_state["last_tree"], expanded=False)

            # Cálculo y visualización del EEE
            from epistemic_metrics import compute_eee

            eee_score = compute_eee(tree, depth_limit=depth, breadth_limit=breadth)
            st.metric(label="Índice de Equilibrio Erotético (EEE)", value=eee_score, key="eee_metric")

            # Exportaciones del árbol y del EEE
            def _flatten_tree(tree: dict, parent: str = "") -> list[tuple[str, str]]:
                rows = []
                for q, kids in tree.items():
                    rows.append((q, parent))
                    if isinstance(kids, dict):
                        rows.extend(_flatten_tree(kids, q))
                return rows

            flat = _flatten_tree(tree)
            df_tree = pd.DataFrame(flat, columns=["question", "parent"])

            # Descargar árbol en CSV
            csv_tree = df_tree.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Descargar árbol (CSV)",
                data=csv_tree,
                file_name="inquiry_tree.csv",
                mime="text/csv",
                key="download_tree_csv"
            )

            # Descargar árbol en JSON
            json_tree_bytes = json.dumps(tree, ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button(
                label="📥 Descargar árbol (JSON)",
                data=json_tree_bytes,
                file_name="inquiry_tree.json",
                mime="application/json",
                key="download_tree_json"
            )

            # Descargar EEE como CSV de metadatos
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
                key="download_eee_csv"
            )

            # Descargar reporte completo HTML
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
                key="download_full_report"
            )

        # ------------------------------------------------------------------
        # 6. REAJUSTAR MODELO DEA tras indagación (nueva sección)
        # ------------------------------------------------------------------
        if "last_tree" in st.session_state:
            st.markdown("---")
            st.subheader("Reajustar Modelo DEA según indagación")

            # 6.1 Mostrar contexto actual (inputs/outputs y árbol)
            st.markdown("**Inputs/Outputs actuales:**")
            st.write("• Inputs seleccionados:", st.session_state["inputs"])
            st.write("• Outputs seleccionados:", st.session_state["outputs"])
            st.write("• Modelo:", model, "|", "Orientación:", orientation, "|", "Super-eff:", super_eff)

            # 6.2 Opción de explicar orientación con IA
            if st.button("¿Debo cambiar orientación (input/output)?", key="btn_explain_orientation"):
                ex = explain_orientation(
                    inputs=st.session_state["inputs"],
                    outputs=st.session_state["outputs"],
                    orientation=st.session_state["dea_orientation"]
                )
                if ex.get("text"):
                    st.info(ex["text"])
                else:
                    st.error("La IA no devolvió sugerencia de orientación.")

            # 6.3 Formulario para reajustar Inputs/Outputs y parámetros
            new_inputs = st.multiselect(
                "Modificar Inputs (según subpreguntas)",
                options=st.session_state["orig_df"].select_dtypes(include="number").columns.tolist(),
                default=st.session_state["inputs"],
                help="Añada o quite insumos tras leer el árbol."
            )
            new_outputs = st.multiselect(
                "Modificar Outputs (según subpreguntas)",
                options=st.session_state["orig_df"].select_dtypes(include="number").columns.tolist(),
                default=st.session_state["outputs"],
                help="Añada o quite productos tras leer el árbol."
            )

            new_model = st.selectbox(
                "Modelo (CCR/BCC) reajustado",
                ["CCR", "BCC"],
                index=0 if st.session_state["dea_model"] == "CCR" else 1,
                key="reajust_model"
            )
            new_orientation = st.selectbox(
                "Orientación (input/output) reajustada",
                ["input", "output"],
                index=0 if st.session_state["dea_orientation"] == "input" else 1,
                key="reajust_orientation"
            )
            new_super_eff = st.checkbox(
                "Super-eficiencia (reajustada)",
                value=st.session_state["dea_super_eff"],
                help="Marque para excluir la DMU al calcular su eficiencia (super-eff).",
                key="reajust_super_eff"
            )

            if st.button("Reejecutar DEA con ajustes", key="btn_reajust_dea"):
                orig_df = st.session_state["orig_df"]
                num_dmu = orig_df.shape[0]

                # Validaciones
                if new_model == "BCC" and num_dmu < 2:
                    st.error("No se puede usar BCC con menos de 2 DMU.")
                    st.stop()
                if new_super_eff and num_dmu < 2:
                    st.error("No se puede usar super-eficiencia con menos de 2 DMU.")
                    st.stop()
                if not new_inputs or not new_outputs:
                    st.error("Debe seleccionar al menos un Input y un Output.")
                    st.stop()

                with st.spinner("Recalculando eficiencias DEA con parámetros ajustados…"):
                    try:
                        new_res = run_dea(
                            orig_df,
                            new_inputs,
                            new_outputs,
                            model=new_model,
                            orientation=new_orientation,
                            super_eff=new_super_eff,
                        )
                        if new_res["efficiency"].isna().all():
                            st.error("⚠️ El solver no devolvió soluciones válidas en la reejecución.")
                            st.stop()
                    except Exception as e:
                        st.error(f"❌ Error al recalcular DEA: {e}")
                        st.stop()

                st.success("✅ DEA reejecutado correctamente con parámetros ajustados.")
                st.subheader("Nuevas eficiencias DEA")
                st.dataframe(new_res, use_container_width=True)

                # Histograma de las nuevas eficiencias
                new_hist = plot_efficiency_histogram(new_res, bins=20)
                st.plotly_chart(new_hist, use_container_width=True, key="new_histogram")

                # Guardamos en sesión los nuevos resultados y parámetros
                st.session_state["res_df"] = new_res.copy()
                st.session_state["dea_model"] = new_model
                st.session_state["dea_orientation"] = new_orientation
                st.session_state["dea_super_eff"] = new_super_eff
                st.session_state["inputs"] = new_inputs
                st.session_state["outputs"] = new_outputs

else:
    st.info("Carga primero un CSV para comenzar.")
