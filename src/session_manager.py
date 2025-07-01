# /src/session_manager.py
# --- VERSIÓN MODIFICADA PARA INCLUIR EL EPISTEMIC TRACKER ---

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional

# (Aquí iría el resto de tu clase SessionManager si tuviera más métodos)
# Por ahora, nos centramos en la funcionalidad de estado y tracker.

def initialize_session_state():
    """Inicializa el estado de la sesión si no existe."""
    # Estados generales de la aplicación
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'data_upload'
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'analysis_params' not in st.session_state:
        st.session_state.analysis_params = {}
    if 'inquiry_tree' not in st.session_state:
        st.session_state.inquiry_tree = None

    # --- INICIO DE LA INTEGRACIÓN DEL TRACKER EPISTÉMICO ---
    # Asegura que la lista para registrar eventos exista en la sesión.
    if 'epistemic_events' not in st.session_state:
        st.session_state.epistemic_events = []
    # --- FIN DE LA INTEGRACIÓN ---


def log_epistemic_event(event_type: str, data: Dict[str, Any]):
    """
    Registra un evento clave del proceso deliberativo del usuario.
    Esta función centraliza la lógica del "Epistemic Tracker".
    """
    import datetime
    event = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event_type": event_type,
        "data": data
    }
    st.session_state.epistemic_events.append(event)
    print(f"EVENT LOGGED: {event_type} - {data}")


def get_event_log() -> List[Dict[str, Any]]:
    """
    Devuelve el registro completo de eventos epistémicos.
    """
    return st.session_state.get('epistemic_events', [])


# Otras funciones de tu session_manager...
def get_data() -> Optional[pd.DataFrame]:
    return st.session_state.get('data')

def set_data(df: pd.DataFrame):
    st.session_state.data = df

# etc...
