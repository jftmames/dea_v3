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
            notes TEXT
        );
    """)
    conn.commit()
    conn.close()

def save_session(
    user_id: str,
    inquiry_tree: dict,
    eee_score: float,
    notes: str
):
    """
    Guarda una nueva sesión en la base de datos, usando solo las columnas:
      session_id, user_id, timestamp, inquiry_tree, eee_score, notes.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    session_id = str(datetime.datetime.now().timestamp())
    timestamp = datetime.datetime.now().isoformat()
    inquiry_tree_json = json.dumps(inquiry_tree)

    cur.execute("""
        INSERT INTO inquiry_sessions (
            session_id,
            user_id,
            timestamp,
            inquiry_tree,
            eee_score,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        user_id,
        timestamp,
        inquiry_tree_json,
        eee_score,
        notes
    ))
    conn.commit()
    conn.close()

def load_sessions(user_id: str) -> list[dict]:
    """
    Recupera sesiones de un usuario dado, devolviendo lista de dicts con:
      session_id, timestamp, inquiry_tree, eee_score, notes.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            session_id,
            timestamp,
            inquiry_tree,
            eee_score,
            notes
        FROM inquiry_sessions
        WHERE user_id = ?
    """, (user_id,))
    rows = cur.fetchall()
    conn.close()

    sesiones = []
    for row in rows:
        # row índices: 0=session_id, 1=timestamp, 2=inquiry_tree, 3=eee_score, 4=notes
        inquiry_tree_data = json.loads(row[2]) if row[2] else {}
        sesiones.append({
            "session_id": row[0],
            "timestamp": row[1],
            "inquiry_tree": inquiry_tree_data,
            "eee_score": row[3],
            "notes": row[4]
        })

    return sesiones
