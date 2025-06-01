# dea_models/constants.py

"""
Parámetros globales y constantes para todos los modelos DEA.
"""

# Tolerancia numérica mínima (por ejemplo, para detectar valores ≈0)
DEFAULT_TOLERANCE = 1e-6

# Máximo número de iteraciones para solvers (por si luego quieres exponerlo como parámetro)
DEFAULT_MAX_ITER = 10000

# Otros parámetros globales que empleen varios modelos 
EPS = 1e-9
BIG_M = 1e6
