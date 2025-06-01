# pages/02_AdjustModel.py

import streamlit as st
import pandas as pd

from dea_analyzer import run_dea
from results import plot_efficiency_histogram  # si quieres graficar comparaciones

st.set_page_config(page_title="DEA Revision", layout="wide")
st.title("🔧 Revisión automática del Modelo DEA")

# 1) Verificar que existan las claves necesarias en session_state
if "res_df" not in st.session_state or "last_reco" not in st.session_state:
    st.error(
        "Primero debes generar un árbol y obtener recomendaciones en la página de Inicio."
        "\n\nVe a la página “Inicio” y pulsa “Crear árbol”."
    )
    st.stop()

# 2) Cargar en variables locales
dea_orig = st.session_state["res_df"]
reco = st.session_state["last_reco"]

# Los parámetros originales los podemos leer también:
model_orig = st.session_state.get("dea_model", "CCR")
orientation_orig = st.session_state.get("dea_orientation", "input")
super_eff_orig = st.session_state.get("dea_super_eff", False)
# A falta de inputs originales, los sacamos del DataFrame original:
# (asumimos que se guardaron en session_state, si no, conviene guardarlos en main.py)
inputs_orig = st.session_state.get("dea_inputs", [])
outputs_orig = st.session_state.get("dea_outputs", [])

st.markdown("## Recomendaciones detectadas tras el árbol de indagación")

# 3) Mostrar cada uno de los posibles ajustes
cols = ["switch_model", "switch_orientation", "new_inputs", "new_outputs", "recommendation"]
any_reco = False

for key in cols:
    if reco.get(key):
        any_reco = True
        if key == "switch_model":
            st.write(f"• Se sugiere cambiar modelo de **{model_orig}** a **{reco[key]}**.")
        elif key == "switch_orientation":
            st.write(f"• Se sugiere cambiar orientación de **{orientation_orig}** a **{reco[key]}**.")
        elif key == "new_inputs":
            st.write(f"• Se sugiere utilizar estos inputs en lugar de los originales:  `{reco[key]}`.")
        elif key == "new_outputs":
            st.write(f"• Se sugiere usar estos outputs en lugar de los originales:  `{reco[key]}`.")
        elif key == "recommendation":
            st.write(f"• {reco[key]}")

if not any_reco:
    st.info("No hay recomendaciones automáticas que aplicar.")
    st.stop()

st.markdown("---")

# 4) Botón para aplicar las recomendaciones
if st.button("✅ Aplicar ajustes recomendados"):
    # 4.1 Extraer los parámetros revisados, o caer al original si no hay cambio
    model_rev = reco.get("switch_model", model_orig)
    orientation_rev = reco.get("switch_orientation", orientation_orig)
    inputs_rev = reco.get("new_inputs", inputs_orig)
    outputs_rev = reco.get("new_outputs", outputs_orig)
    super_eff_rev = super_eff_orig  # no recomendamos cambiar esto en este ejemplo

    # 4.2 Validar que inputs_rev/outputs_rev no queden vacíos
    if len(inputs_rev) < 1 or len(outputs_rev) < 1:
        st.error("Los conjuntos de inputs o outputs quedarían vacíos: revisión inválida.")
        st.stop()

    # 4.3 Ejecutar DEA con los parámetros revisados
    with st.spinner("🔄 Recalculando DEA con ajustes…"):
        try:
            dea_rev = run_dea(
                st.session_state["original_df"],  # asume que guardaste el DF fuente
                inputs_rev,
                outputs_rev,
                model=model_rev,
                orientation=orientation_rev,
                super_eff=super_eff_rev,
            )
        except Exception as e:
            st.error(f"Error al recalcular DEA: {e}")
            st.stop()

    # 4.4 Guardar el DEA revisado en session_state para poder compararlo o descargarlo
    st.session_state["revision_df"] = dea_rev
    st.session_state["rev_model"] = model_rev
    st.session_state["rev_orientation"] = orientation_rev
    st.session_state["rev_super_eff"] = super_eff_rev
    st.session_state["rev_inputs"] = inputs_rev
    st.session_state["rev_outputs"] = outputs_rev

    st.success("✅ DEA revisado guardado en memoria. Desplázate hacia abajo para ver la comparación.")

st.markdown("---")

# 5) Si existe revision_df, mostrar comparación
if "revision_df" in st.session_state:
    dea_rev = st.session_state["revision_df"]
    model_rev = st.session_state["rev_model"]
    orientation_rev = st.session_state["rev_orientation"]

    st.subheader("Comparación de Eficiencias: Original vs Revisado")

    # Unir ambos DataFrames por DMU
    df_orig_cmp = dea_orig.set_index("DMU")[["efficiency"]].rename(columns={"efficiency": "orig_eff"})
    df_rev_cmp = dea_rev.set_index("DMU")[["efficiency"]].rename(columns={"efficiency": "rev_eff"})
    df_cmp = df_orig_cmp.join(df_rev_cmp, how="outer")
    df_cmp["delta"] = df_cmp["rev_eff"] - df_cmp["orig_eff"]
    df_cmp = df_cmp.reset_index()

    st.dataframe(df_cmp, use_container_width=True)

    st.markdown("#### Histograma de diferencia en eficiencia (rev – orig)")
    fig_delta = plot_efficiency_histogram(df_cmp.rename(columns={"delta": "delta"}), bins=20, column="delta")
    st.plotly_chart(fig_delta, use_container_width=True)

    # 5.1 Botón para descargar DEA revisado
    csv_rev = dea_rev.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Descargar DEA Revisado (CSV)",
        data=csv_rev,
        file_name="dea_revised.csv",
        mime="text/csv",
    )

