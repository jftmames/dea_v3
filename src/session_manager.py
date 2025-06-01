# src/session_manager.py

import sqlite3
import json
import datetime
import os

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
            inquiry_tree TEXT,       -- JSON string
            eee_score REAL,
            notes TEXT,
            dea_inputs TEXT,         -- JSON string
            dea_outputs TEXT,        -- JSON string
            dea_results TEXT         -- JSON string con resultados CCR/BCC
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
    dea_inputs: list[str],
    dea_outputs: list[str],
    dea_results: dict
):
    """
    Guarda una nueva sesión en la base.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    timestamp = datetime.datetime.now().isoformat()
    cur.execute("""
        INSERT INTO inquiry_sessions (session_id, user_id, timestamp, inquiry_tree, eee_score, notes, dea_inputs, dea_outputs, dea_results)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        user_id,
        timestamp,
        json.dumps(inquiry_tree),
        eee_score,
        notes,
        json.dumps(dea_inputs),
        json.dumps(dea_outputs),
        json.dumps(dea_results)
    ))
    conn.commit()
    conn.close()

def load_sessions(user_id: str) -> list[dict]:
    """
    Recupera todas las sesiones de un usuario dado.
    Retorna lista de dicts con keys: session_id, timestamp, inquiry_tree, eee_score, notes, dea_inputs, dea_outputs, dea_results.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT session_id, timestamp, inquiry_tree, eee_score, notes, dea_inputs, dea_outputs, dea_results
        FROM inquiry_sessions
        WHERE user_id = ?
    """, (user_id,))
    rows = cur.fetchall()
    conn.close()

    sesiones = []
    for row in rows:
        sesiones.append({
            "session_id": row[0],
            "timestamp": row[1],
            "inquiry_tree": json.loads(row[2]),
            "eee_score": row[3],
            "notes": row[4],
            "dea_inputs": json.loads(row[5]),
            "dea_outputs": json.loads(row[6]),
            "dea_results": json.loads(row[7])
        })
    return sesiones
