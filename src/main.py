# ===============================================================
#                INICIO DEL CÓDIGO DE DEPURACIÓN
# ===============================================================
st.warning("--- INFORMACIÓN DE DEPURACIÓN ---")
if 'df' in st.session_state:
    st.write("La clave 'df' SÍ EXISTE en st.session_state.")
    st.write(f"El tipo de st.session_state.df es: `{type(st.session_state.df)}`")
else:
    st.error("La clave 'df' NO EXISTE en st.session_state.")

st.write("Contenido completo de st.session_state:")
st.json(st.session_state.to_dict())
st.write("--- FIN DE LA INFORMACIÓN DE DEPURACIÓN ---")
st.divider()
# ===============================================================
#                 FIN DEL CÓDIGO DE DEPURACIÓN
# ===============================================================
