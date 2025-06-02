# src/session_manager.py

import sqlite3
import json
import datetime
import os
import pandas as pd  # para to_json() y read_json()

DB_PATH = "sessions.db"

def init_db():
    """
    Crea la base SQLite con tabla inquiry_sessions si no existe.
    NOTA: aquí creamos únicamente las columnas mínimas que había antes,
    sin el campo dmu_column. Si al final vas a necesitar dmu_column
    permanentemente, tendrás que volver a añadirlo y ejecutar un ALTER TABLE
    o recrear la tabla con esa columna.
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
            input_cols TEXT,     -- JSON string
            output_cols TEXT,    -- JSON string
            df_ccr TEXT,         -- JSON string de df_ccr
            df_bcc TEXT          -- JSON string de df_bcc
        );
    """)
    conn.commit()
    conn.close()

def save_session(
    user_id: str,
    results: dict,
    inquiry_tree: dict,
    eee_score: float,
    notes: str
):
    """
    Guarda una nueva sesión en la base de datos.
    Ahora, en lugar de pasar dmu_column, input_cols, output_cols, df_ccr y df_bcc
    como parámetros separados, esperamos que vengan dentro de results:
      - results["dmu_column"]
      - results["input_cols"]
      - results["output_cols"]
      - results["df_ccr"]  (DataFrame)
      - results["df_bcc"]  (DataFrame)
    De esta forma evitamos el “unexpected keyword argument 'results'”.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    session_id = results.get("session_id", str(datetime.datetime.now().timestamp()))
    timestamp = datetime.datetime.now().isoformat()

    # Extraer dmu_column, input/output cols y DataFrames de results
    dmu_column   = results.get("dmu_column", "")
    input_cols   = results.get("input_cols", [])
    output_cols  = results.get("output_cols", [])
    df_ccr_df    = results.get("df_ccr", pd.DataFrame())
    df_bcc_df    = results.get("df_bcc", pd.DataFrame())

    # Serializar a JSON
    df_ccr_json  = df_ccr_df.to_json(orient='records')
    df_bcc_json  = df_bcc_df.to_json(orient='records')
    input_cols_j = json.dumps(input_cols)
    output_cols_j= json.dumps(output_cols)
    inquiry_tree_j = json.dumps(inquiry_tree)

    cur.execute("""
        INSERT INTO inquiry_sessions (
            session_id,
            user_id,
            timestamp,
            inquiry_tree,
            eee_score,
            notes,
            input_cols,
            output_cols,
            df_ccr,
            df_bcc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        user_id,
        timestamp,
        inquiry_tree_j,
        eee_score,
        notes,
        input_cols_j,
        output_cols_j,
        df_ccr_json,
        df_bcc_json
    ))
    conn.commit()
    conn.close()

def load_sessions(user_id: str) -> list[dict]:
    """
    Recupera sesiones de un usuario dado, sin tratar de leer dmu_column
    (porque la tabla original no la tiene). Retorna lista de dicts con keys:
    session_id, timestamp, inquiry_tree, eee_score, notes, input_cols, output_cols, df_ccr, df_bcc.
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
            input_cols,
            output_cols,
            df_ccr,
            df_bcc
        FROM inquiry_sessions
        WHERE user_id = ?
    """, (user_id,))
    rows = cur.fetchall()
    conn.close()

    sesiones = []
    for row in rows:
        # row índices:
        # 0 = session_id
        # 1 = timestamp
        # 2 = inquiry_tree (JSON)
        # 3 = eee_score
        # 4 = notes
        # 5 = input_cols (JSON)
        # 6 = output_cols (JSON)
        # 7 = df_ccr (JSON)
        # 8 = df_bcc (JSON)

        inquiry_tree_data = json.loads(row[2]) if row[2] else {}
        input_cols_list   = json.loads(row[5]) if row[5] else []
        output_cols_list  = json.loads(row[6]) if row[6] else []

        try:
            df_ccr = pd.read_json(row[7], orient='records')
        except Exception:
            df_ccr = pd.DataFrame()

        try:
            df_bcc = pd.read_json(row[8], orient='records')
        except Exception:
            df_bcc = pd.DataFrame()

        sesiones.append({
            "session_id": row[0],
            "timestamp": row[1],
            "inquiry_tree": inquiry_tree_data,
            "eee_score": row[3],
            "notes": row[4],
            "input_cols": input_cols_list,
            "output_cols": output_cols_list,
            "df_ccr": df_ccr,
            "df_bcc": df_bcc
        })

    return sesiones
