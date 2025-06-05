# src/epistemic_metrics.py
import pandas as pd
import numpy as np

def compute_eee(
    inquiry_tree: dict,
    depth_limit: int,
    breadth_limit: int
) -> dict:
    """
    Calcula el Índice de Equilibrio Erotético (EEE) y devuelve un desglose.
    Retorna un diccionario con el score total y los componentes D1-D5.
    """
    if not inquiry_tree:
        return {"score": 0, "D1": 0, "D2": 0, "D3": 0, "D4": 0, "D5": 0}

    # (1) D1: Profundidad = número máximo de niveles en inquiry_tree / depth_limit
    max_depth = _max_tree_depth(inquiry_tree)
    D1 = min(max_depth / depth_limit, 1.0)

    # (2) D2: Pluralidad = número de nodos hijos únicos en nivel 1 / breadth_limit
    root = next(iter(inquiry_tree.values())) if inquiry_tree else {}
    num_children = len(root) if isinstance(root, dict) else 0
    D2 = min(num_children / breadth_limit, 1.0)

    # (3) D3: Trazabilidad = simplificamos a 1 si existe >1 rama, else 0.5
    D3 = 1.0 if num_children > 1 else 0.5

    # (4) D4: Reversibilidad = por implementarse, valor por defecto
    D4 = 0.5

    # (5) D5: Robustez ante disenso = si hay al menos dos subramas en raíz
    D5 = 1.0 if num_children >= 2 else 0.0

    # Pesos (pueden ajustarse más adelante)
    w1, w2, w3, w4, w5 = 1, 1, 1, 1, 1
    eee_score = (w1*D1 + w2*D2 + w3*D3 + w4*D4 + w5*D5) / (w1 + w2 + w3 + w4 + w5)
    
    return {
        "score": round(eee_score, 4),
        "D1": D1,
        "D2": D2,
        "D3": D3,
        "D4": D4,
        "D5": D5,
    }

def _max_tree_depth(tree: dict) -> int:
    """Devuelve la profundidad máxima de un diccionario anidado."""
    if not isinstance(tree, dict) or not tree:
        return 0
    # Ajuste para manejar correctamente el último nivel de anidación
    depths = [_max_tree_depth(v) for v in tree.values() if isinstance(v, dict)]
    return 1 + max(depths) if depths else 1
