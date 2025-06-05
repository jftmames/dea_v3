# src/session_manager.py
import sqlite3
import json
import datetime

DB_PATH = "sessions.db"

def init_db():
    """
    Crea la base SQLite con la tabla inquiry_sessions si no existe.
    Solo incluye las columnas que realmente usa la base actual:
    session_id, user_id, timestamp, inquiry_tree, eee_score, notes.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS inquiry_sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT,
            timestamp TEXT,
            inquiry_tree TEXT,   -- JSON string
            eee_score REAL,
            notes TEXT,
            -- Add more columns to store the full session state if needed
            -- For example, store selected inputs/outputs, results dataframes etc.
            -- This example uses a simplified schema as per the original file.
            dmu_col TEXT,
            input_cols TEXT, -- JSON string of list
            output_cols TEXT, -- JSON string of list
            df_data TEXT, -- JSON string of dataframe data
            dea_results TEXT, -- JSON string of DEA results (excluding figures)
            df_tree_data TEXT, -- JSON string of df_tree
            df_eee_data TEXT -- JSON string of df_eee
        );
    """)
    conn.commit()
    conn.close()

def save_session(
    user_id: str,
    inquiry_tree: dict,
    eee_score: float,
    notes: str,
    # New parameters to save full session state
    dmu_col: str = None,
    input_cols: list[str] = None,
    output_cols: list[str] = None,
    df_data: dict = None, # DataFrame.to_dict('records')
    dea_results: dict = None, # Results dict, with DFs as dicts
    df_tree_data: dict = None, # df_tree.to_dict('records')
    df_eee_data: dict = None # df_eee.to_dict('records')
):
    """
    Guarda una nueva sesión en la base de datos, incluyendo datos adicionales del estado.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    session_id = str(datetime.datetime.now().timestamp())
    timestamp = datetime.datetime.now().isoformat()
    inquiry_tree_json = json.dumps(inquiry_tree)
    
    # Convert lists/dicts to JSON strings for storage
    input_cols_json = json.dumps(input_cols) if input_cols is not None else None
    output_cols_json = json.dumps(output_cols) if output_cols is not None else None
    df_data_json = json.dumps(df_data) if df_data is not None else None
    dea_results_json = json.dumps(dea_results) if dea_results is not None else None
    df_tree_data_json = json.dumps(df_tree_data) if df_tree_data is not None else None
    df_eee_data_json = json.dumps(df_eee_data) if df_eee_data is not None else None


    cur.execute("""
        INSERT INTO inquiry_sessions (
            session_id,
            user_id,
            timestamp,
            inquiry_tree,
            eee_score,
            notes,
            dmu_col,
            input_cols,
            output_cols,
            df_data,
            dea_results,
            df_tree_data,
            df_eee_data
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        user_id,
        timestamp,
        inquiry_tree_json,
        eee_score,
        notes,
        dmu_col,
        input_cols_json,
        output_cols_json,
        df_data_json,
        dea_results_json,
        df_tree_data_json,
        df_eee_data_json
    ))
    conn.commit()
    conn.close()

def load_sessions(user_id: str) -> list[dict]:
    """
    Recupera sesiones de un usuario dado.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            session_id,
            timestamp,
            inquiry_tree,
            eee_score,
            notes,
            dmu_col,
            input_cols,
            output_cols,
            df_data,
            dea_results,
            df_tree_data,
            df_eee_data
        FROM inquiry_sessions
        WHERE user_id = ?
    """, (user_id,))
    rows = cur.fetchall()
    conn.close()

    sesiones = []
    for row in rows:
        # Map row indices to dict keys
        session_dict = {
            "session_id": row[0],
            "timestamp": row[1],
            "inquiry_tree": json.loads(row[2]) if row[2] else {},
            "eee_score": row[3],
            "notes": row[4],
            "dmu_col": row[5],
            "input_cols": json.loads(row[6]) if row[6] else [],
            "output_cols": json.loads(row[7]) if row[7] else [],
            "df_data": json.loads(row[8]) if row[8] else None, # Dataframe data
            "dea_results": json.loads(row[9]) if row[9] else None, # DEA results
            "df_tree": json.loads(row[10]) if row[10] else None, # df_tree data
            "df_eee": json.loads(row[11]) if row[11] else None # df_eee data
        }
        sesiones.append(session_dict)

    return sesiones
