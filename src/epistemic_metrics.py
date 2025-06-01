import pandas as pd
import numpy as np

def compute_eee(
    inquiry_tree: dict,
    depth_limit: int,
    breadth_limit: int
) -> float:
    """
    Calcula el Índice de Equilibrio Erotético (EEE) basado en:
      D1: Profundidad estructural
      D2: Pluralidad semántica
      D3: Trazabilidad razonadora
      D4: Reversibilidad efectiva
      D5: Robustez ante disenso
    Retorna un valor entre 0 y 1.
    """
    # (1) D1: Profundidad = número máximo de niveles en inquiry_tree / depth_limit
    max_depth = _max_tree_depth(inquiry_tree)
    D1 = min(max_depth / depth_limit, 1.0)

    # (2) D2: Pluralidad = número de nodos hijos únicos en nivel 1 / breadth_limit
    root = next(iter(inquiry_tree.values()))
    num_children = len(root) if isinstance(root, dict) else 0
    D2 = min(num_children / breadth_limit, 1.0)

    # (3) D3: Trazabilidad = asumimos que cada nodo lleva justificación implícita 
    #                => puntuar según # de ramas navegables; 
    #                para MVP, simplificamos a 1 si existe >1 rama, else 0.5
    D3 = 1.0 if num_children > 1 else 0.5

    # (4) D4: Reversibilidad = no tenemos histórico de reformulaciones en MVP,
    #                así que regresamos 0.5 por defecto (por implementarse en próxima iteración)
    D4 = 0.5

    # (5) D5: Robustez ante disenso = si hay al menos dos subramas en raíz
    D5 = 1.0 if num_children >= 2 else 0.0

    # Pesos (pueden ajustarse más adelante)
    w1, w2, w3, w4, w5 = 1, 1, 1, 1, 1
    eee = (w1*D1 + w2*D2 + w3*D3 + w4*D4 + w5*D5) / (w1 + w2 + w3 + w4 + w5)
    return round(eee, 4)

def _max_tree_depth(tree: dict) -> int:
    """Devuelve la profundidad máxima de un diccionario anidado."""
    if not isinstance(tree, dict) or not tree:
        return 0
    return 1 + max(_max_tree_depth(v) for v in tree.values() if isinstance(v, dict))
