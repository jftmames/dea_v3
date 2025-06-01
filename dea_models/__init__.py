# dea_models/__init__.py

"""
Paquete dea_models: aquí se concentra toda la lógica básica de los modelos DEA.
Al importar dea_models, podrás acceder a los siguientes módulos:

- radial: CCR/BCC radial
- nonradial: SBM y Directional Distance
- network: DEA en red / multietapa
- mpi: Malmquist Productivity Index
- cross_efficiency: Cross‐efficiency / peer appraisal
- window_analysis: Window‐analysis para series de tiempo
- stochastic: DEA con Bootstrapping / Stochastic DEA
- utils: funciones auxiliares (validación, transformaciones)
- directions: generación de vectores direccionales
- constants: parámetros globales (tolerancias, etc.)
"""

from .radial import run_ccr, run_bcc
from .nonradial import run_sbm, run_sbm_output, run_radial_distance
from .network import run_network_dea, run_multi_stage_network
from .mpi import compute_malmquist_phi
from .cross_efficiency import compute_cross_efficiency
from .window_analysis import run_window_dea
from .stochastic import run_stochastic_dea, bootstrap_efficiencies
from .utils import validate_dataframe, check_positive_data, check_zero_negative_data
from .directions import get_direction_vector, get_custom_direction_vector
from .constants import DEFAULT_TOLERANCE
