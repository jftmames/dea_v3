# src/session_manager.py
import streamlit as st
import json
import datetime
import pandas as pd

# Ya no necesitamos una ruta de archivo local. La conexión se gestiona a través de st.connection.

def get_db_connection():
    """
    Establece y devuelve una conexión a la base de datos externa
    utilizando la configuración de secrets de Streamlit.
    """
    # El nombre "sessions_db" debe coincidir con el que usaste en los secrets: [connections.sessions_db]
    return st.connection("sessions_db", type="sql")

def init_db():
    """
    Crea la tabla 'inquiry_sessions' en la base de datos remota si no existe.
    """
    conn = get_db_connection()
    with conn.session as s:
        # Usamos TEXT para los JSON, ya que SQLite no tiene un tipo JSON nativo.
        s.execute("""
            CREATE TABLE IF NOT EXISTS inquiry_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                timestamp TEXT,
                inquiry_tree TEXT,
                eee_metrics TEXT,
                notes TEXT,
                dmu_col TEXT,
                input_cols TEXT,
                output_cols TEXT,
                df_data TEXT,
                dea_results TEXT,
                df_tree_data TEXT,
                df_eee_data TEXT
            );
        """)

def save_session(
    user_id: str,
    inquiry_tree: dict,
    eee_metrics: dict,
    notes: str,
    dmu_col: str = None,
    input_cols: list[str] = None,
    output_cols: list[str] = None,
    df_data: dict = None,
    dea_results: dict = None,
    df_tree_data: dict = None,
    df_eee_data: dict = None
):
    """Guarda una nueva sesión en la base de datos remota de forma segura."""
    conn = get_db_connection()
    
    session_id = str(datetime.datetime.now().timestamp())
    timestamp = datetime.datetime.now().isoformat()
    
    # Prepara los datos para la inserción en un DataFrame
    session_data = {
        "session_id": [session_id],
        "user_id": [user_id],
        "timestamp": [timestamp],
        "inquiry_tree": [json.dumps(inquiry_tree) if inquiry_tree else None],
        "eee_metrics": [json.dumps(eee_metrics) if eee_metrics else None],
        "notes": [notes],
        "dmu_col": [dmu_col],
        "input_cols": [json.dumps(input_cols) if input_cols else None],
        "output_cols": [json.dumps(output_cols) if output_cols else None],
        "df_data": [json.dumps(df_data) if df_data else None],
        "dea_results": [json.dumps(dea_results) if dea_results else None],
        "df_tree_data": [json.dumps(df_tree_data) if df_tree_data else None],
        "df_eee_data": [json.dumps(df_eee_data) if df_eee_data else None]
    }
    df_to_insert = pd.DataFrame(session_data)
    
    # Utiliza conn.write para añadir los datos a la tabla. Es el método recomendado.
    with conn.session as s:
        s.execute(f"INSERT INTO inquiry_sessions ({', '.join(df_to_insert.columns)}) VALUES ({', '.join(['?']*len(df_to_insert.columns))})", tuple(df_to_insert.iloc[0]))
        s.commit()


def load_sessions(user_id: str) -> list[dict]:
    """Recupera las sesiones de un usuario desde la base de datos remota."""
    conn = get_db_connection()
    
    # Primero, asegúrate de que la tabla exista para evitar errores en la primera ejecución
    try:
        # Ejecuta una consulta para leer los datos
        query = f"SELECT * FROM inquiry_sessions WHERE user_id = '{user_id}' ORDER BY timestamp DESC"
        df_sessions = conn.query(query)
    except Exception:
        # Si la tabla no existe, inicialízala y devuelve una lista vacía
        init_db()
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
                    pass # Si no es un JSON válido, déjalo como está
        sesiones.append(session_dict)
        
    return sesiones
