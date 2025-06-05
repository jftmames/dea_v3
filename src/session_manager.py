# src/session_manager.py
import streamlit as st
import json
import datetime
import pandas as pd

def get_db_connection():
    """Establece y devuelve una conexión a la base de datos externa."""
    try:
        return st.connection("sessions_db", type="sql")
    except Exception as e:
        st.error(f"Error al conectar con la base de datos. Asegúrate de que los 'Secrets' están bien configurados. Detalle: {e}")
        return None

def init_db():
    """Crea la tabla si no existe. Se ejecuta si la carga o guardado inicial falla."""
    try:
        conn = get_db_connection()
        if conn:
            with conn.session as s:
                s.execute("""
                    CREATE TABLE IF NOT EXISTS inquiry_sessions (
                        session_id TEXT PRIMARY KEY, user_id TEXT, timestamp TEXT,
                        inquiry_tree TEXT, eee_metrics TEXT, notes TEXT, dmu_col TEXT,
                        input_cols TEXT, output_cols TEXT, df_data TEXT,
                        dea_results TEXT, df_tree_data TEXT, df_eee_data TEXT
                    );
                """)
    except Exception as e:
        st.warning(f"No se pudo inicializar la base de datos: {e}")

def save_session(**kwargs):
    """Guarda una nueva sesión en la base de datos remota."""
    conn = get_db_connection()
    if not conn:
        st.error("No se pudo guardar la sesión: Conexión a la base de datos fallida.")
        return

    try:
        session_id = str(datetime.datetime.now().timestamp())
        timestamp = datetime.datetime.now().isoformat()
        
        # Prepara los datos para la inserción
        session_data = {"session_id": session_id, "timestamp": timestamp}
        session_data.update(kwargs)

        # Convierte dicts y lists a JSON strings para guardarlos
        for key, value in session_data.items():
            if isinstance(value, (dict, list)):
                session_data[key] = json.dumps(value)
        
        df_to_insert = pd.DataFrame([session_data])
        
        # Usar el método write de st.connection es más robusto
        conn.write(df_to_insert, "inquiry_sessions", if_exists="append")
        
        st.success("¡Sesión guardada en la base de datos remota!")
        st.balloons()
    except Exception as e:
        st.error(f"Ocurrió un error al guardar la sesión: {e}")
        # Intenta inicializar la BD por si la tabla no existía
        init_db()


def load_sessions(user_id: str) -> list[dict]:
    """Recupera las sesiones de un usuario desde la base de datos remota."""
    conn = get_db_connection()
    if not conn:
        return []

    try:
        query = f"SELECT * FROM inquiry_sessions WHERE user_id = '{user_id}' ORDER BY timestamp DESC"
        df_sessions = conn.query(query)
        
        if df_sessions.empty:
            return []

        # Convierte el DataFrame a una lista de diccionarios, decodificando los JSON
        sesiones = []
        for _, row in df_sessions.iterrows():
            session_dict = row.to_dict()
            for key, value in session_dict.items():
                if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                    try:
                        session_dict[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        pass
            sesiones.append(session_dict)
        return sesiones
    except Exception as e:
        st.warning(f"No se pudieron cargar las sesiones. Es posible que no existan o haya un problema de conexión. Error: {e}")
        # Intenta inicializar la BD por si la tabla no existía
        init_db()
        return []
