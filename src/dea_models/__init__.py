# jftmames/-dea-deliberativo-mvp/-dea-deliberativo-mvp-b44b8238c978ae0314af30717b9399634d28f8f9/src/dea_models/__init__.py
# Se exponen únicamente los modelos y utilidades principales desde el paquete.
# Las funciones de visualización se importarán directamente desde el módulo
# dea_models.visualizations para evitar errores de importación circular.

from .utils import validate_positive_dataframe, check_positive_data, check_zero_negative_data
from .radial import run_ccr, run_bcc
from .nonradial import run_sbm, run_radial_distance
from .mpi import compute_malmquist_phi
from .cross_efficiency import compute_cross_efficiency
from .window_analysis import run_window_dea
from .stochastic import run_stochastic_dea, bootstrap_efficiencies
from .auto_tuner import generate_candidates, evaluate_candidates
# Las líneas que importaban desde .visualizations han sido eliminadas.
