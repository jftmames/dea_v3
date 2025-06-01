import streamlit as st
import pandas as pd

from data_validator import validate
from dea_analyzer import run_dea
from inquiry_engine import to_plotly_tree
from results import (
    plot_efficiency_histogram,
    plot_3d_inputs_outputs,
    plot_benchmark_spider,
)
from openai_helpers import explain_orientation

# ---------- Util: obtener fila de DMU (idéntico a main.py) ----------
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


st.set_page_config(page_title="Reajustar Modelo DEA", layout="wide")
st.title("02 – Reajustar Modelo DEA tras Indagación")

# 1) ¿Ya ejecutó el DEA en la página principal?
if "res_df" not in st.session_state:
    st.warning("Primero debe ejecutar el DEA en la primera pestaña (Main).")
    st.stop()

# 2) Obtenemos datos de sesión
orig_df = st.session_state.get("orig_df", None)  # opcional: si guardamos el DataFrame original
dea_df = st.session_state["res_df"]
tree = st.session_state.get("last_tree", None)
model_orig = st.session_state["dea_model"]
orientation_orig = st.session_state["dea_orientation"]
super_eff_orig = st.session_state["dea_super_eff"]
depth_orig = st.session_state.get("last_depth", None)
breadth_orig = st.session_state.get("last_breadth", None)
dmu_orig = st.session_state.get("last_dmu", None)

# Si no guardamos el DataFrame original en sesión, reasumimos que lo subió en main.py
# Una forma sencilla: 
if orig_df is None:
    orig_df = st.session_state.get("_uploaded_df", None)  # si main.py guardó df en sesión

# 3) Mostrar resultados originales brevemente
st.subheader("Eficiencias DEA (resultado original)")
st.dataframe(dea_df, use_container_width=True)

# Histograma rápido
hist_fig = plot_efficiency_histogram(dea_df, bins=20)
st.plotly_chart(hist_fig, use_container_width=True)

# 4) Mostrar árbol de la última DMU ineficiente (si existe)
if tree:
    st.subheader(f"Árbol de Indagación (DMU: {dmu_orig}, niveles={depth_orig}, hijos={breadth_orig})")
    st.plotly_chart(to_plotly_tree(tree), use_container_width=True)

    with st.expander("Ver subpreguntas (JSON completo)"):
        st.json(tree)

    # 4.1 Explicar orientación original con IA
    st.markdown("**¿La orientación DEA elegida es la más adecuada?**")
    explain_btn = st.button("Pedir sugerencia de orientación a la IA")
    if explain_btn:
        ex = explain_orientation(
            inputs=st.session_state.get("inputs", []),
            outputs=st.session_state.get("outputs", []),
            orientation=orientation_orig
        )
        if ex.get("text"):
            st.info(ex["text"])
        else:
            st.error("La IA no devolvió sugerencia.")

# 5) Formulario para “reajustar” el modelo DEA
st.markdown("---")
st.subheader("Reajustar parámetros del DEA según Indagación")

# Leemos los inputs/outputs originales de sesión (guardados en main.py)
inputs = st.session_state.get("inputs", [])
outputs = st.session_state.get("outputs", [])

# (a) Nuevos inputs/outputs:  
new_inputs = st.multiselect(
    "Inputs (posibles ajustes)",
    options=orig_df.select_dtypes(include="number").columns.tolist(),
    default=inputs,
    help="Incluya o quite insumos tras la indagación."
)
new_outputs = st.multiselect(
    "Outputs (posibles ajustes)",
    options=orig_df.select_dtypes(include="number").columns.tolist(),
    default=outputs,
    help="Incluya o quite productos tras la indagación."
)

# (b) Modelo/orientación/super-eff reajustados:
new_model = st.selectbox(
    "Modelo (CCR/BCC)",
    ["CCR", "BCC"],
    index=0 if model_orig == "CCR" else 1
)
new_orientation = st.selectbox(
    "Orientación (input/output)",
    ["input", "output"],
    index=0 if orientation_orig == "input" else 1
)
new_super_eff = st.checkbox(
    "Super-eficiencia",
    value=super_eff_orig,
    help="Si marca, cada vez excluye la DMU actual."
)

# (c) Botón para reejecutar
if st.button("Reejecutar DEA con parámetros ajustados"):
    num_dmu = orig_df.shape[0]

    # Validaciones
    if new_model == "BCC" and num_dmu < 2:
        st.error("No se puede usar BCC con menos de 2 DMU.")
        st.stop()
    if new_super_eff and num_dmu < 2:
        st.error("No se puede usar super-eficiencia con menos de 2 DMU.")
        st.stop()
    if not new_inputs or not new_outputs:
        st.error("Debe seleccionar al menos un input y un output.")
        st.stop()

    with st.spinner("Recalculando eficiencias DEA con parámetros nuevos…"):
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

    # 6) Mostrar nuevos resultados y sobrescribir sesión
    st.success("✅ DEA reejecutado correctamente con parámetros ajustados.")
    st.subheader("Nuevas eficiencias DEA")
    st.dataframe(new_res, use_container_width=True)

    # Histograma de las nuevas eficiencias
    new_hist = plot_efficiency_histogram(new_res, bins=20)
    st.plotly_chart(new_hist, use_container_width=True)

    # Guardamos en sesión (opcional, si queremos que Main.py lo use)
    st.session_state["res_df"] = new_res
    st.session_state["dea_model"] = new_model
    st.session_state["dea_orientation"] = new_orientation
    st.session_state["dea_super_eff"] = new_super_eff
    st.session_state["inputs"] = new_inputs
    st.session_state["outputs"] = new_outputs
