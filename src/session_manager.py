# src/session_manager.py

import sqlite3
import json
import datetime
import os
import pandas as pd  # Necesitamos para .to_json() y pd.read_json()

DB_PATH = "sessions.db"

def init_db():
    """
    Crea la base SQLite con tabla inquiry_sessions si no existe.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS inquiry_sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT,
            timestamp TEXT,
            inquiry_tree TEXT,        -- JSON string
            eee_score REAL,
            notes TEXT,
            dmu_column TEXT,         -- Columna DMU
            input_cols TEXT,         -- JSON string
            output_cols TEXT,        -- JSON string
            df_ccr TEXT,             -- JSON string de df_ccr
            df_bcc TEXT              -- JSON string de df_bcc
            -- Añadir más campos si se necesitan más resultados o contexto
        );
    """)
    conn.commit()
    conn.close()

def save_session(
    session_id: str,
    user_id: str,
    inquiry_tree: dict,
    eee_score: float,
    notes: str,
    dmu_column: str,        # Agregado: columna DMU
    input_cols: list[str],  # Agregado: inputs utilizados
    output_cols: list[str], # Agregado: outputs utilizados
    df_ccr: pd.DataFrame,   # Agregado: DataFrame de resultados CCR
    df_bcc: pd.DataFrame    # Agregado: DataFrame de resultados BCC
):
    """
    Guarda una nueva sesión en la base de datos, incluyendo los DataFrames
    de resultados de DEA convertidos a JSON.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    timestamp = datetime.datetime.now().isoformat()

    # Convertir DataFrames a JSON
    df_ccr_json = df_ccr.to_json(orient='records')  # 'records' para lista de dicts
    df_bcc_json = df_bcc.to_json(orient='records')

    cur.execute("""
        INSERT INTO inquiry_sessions (
            session_id,
            user_id,
            timestamp,
            inquiry_tree,
            eee_score,
            notes,
            dmu_column,
            input_cols,
            output_cols,
            df_ccr,
            df_bcc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        user_id,
        timestamp,
        json.dumps(inquiry_tree),
        eee_score,
        notes,
        dmu_column,
        json.dumps(input_cols),
        json.dumps(output_cols),
        df_ccr_json,
        df_bcc_json
    ))
    conn.commit()
    conn.close()

def load_sessions(user_id: str) -> list[dict]:
    """
    Recupera sesiones de un usuario dado con toda la información almacenada.
    Retorna lista de dicts con keys: session_id, timestamp, inquiry_tree, eee_score,
    notes, dmu_column, input_cols, output_cols, df_ccr, df_bcc.
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
            dmu_column,
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
        # row índices: 0=session_id,1=timestamp,2=inquiry_tree,3=eee_score,4=notes,
        #             5=dmu_column,6=input_cols,7=output_cols,8=df_ccr,9=df_bcc
        inquiry_tree_data = json.loads(row[2]) if row[2] else {}
        input_cols_list = json.loads(row[6]) if row[6] else []
        output_cols_list = json.loads(row[7]) if row[7] else []

        # Reconstruir DataFrames desde JSON
        try:
            df_ccr = pd.read_json(row[8], orient='records')
        except Exception:
            df_ccr = pd.DataFrame()
        try:
            df_bcc = pd.read_json(row[9], orient='records')
        except Exception:
            df_bcc = pd.DataFrame()

        sesiones.append({
            "session_id": row[0],
            "timestamp": row[1],
            "inquiry_tree": inquiry_tree_data,
            "eee_score": row[3],
            "notes": row[4],
            "dmu_column": row[5],
            "input_cols": input_cols_list,
            "output_cols": output_cols_list,
            "df_ccr": df_ccr,
            "df_bcc": df_bcc
        })
    return sesiones
