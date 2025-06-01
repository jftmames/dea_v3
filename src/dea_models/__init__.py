# src/dea_models/__init__.py

"""
Paquete dea_models: lógica de modelos DEA.
Por ahora solo exportamos radial y utils; los demás módulos se implementarán luego.
"""

from .utils import validate_positive_dataframe, check_positive_data, check_zero_negative_data
from .radial import run_ccr, run_bcc

