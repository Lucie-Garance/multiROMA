# pyroma/__init__.py
"""
MultiROMA: Python implementation of Representation Of Module Activity (ROMA) for multiomic data.
"""

from .multiroma import ROMA_MFA, ROMA_GSVD, GeneSetResult_MFA, GeneSetResult_GSVD, color
from .utils import integrate_projection_results
from .MIIC_input_generator import MIIC_input_generator

# submodules 
#from . import datasets
#from . import genesets
from . import plotting
from . import utils
from . import sparse_methods
from . import preprocessing

# submodule functions 
#from .datasets import pbmc3k, pbmc_ifnb
#from .genesets import use_hallmarks, use_reactome, use_progeny

__version__ = '0.0.1'

__all__ = [
    # Core classes
    'ROMA_MFA',
    'ROMA_GSVD',
    'GeneSetResult_MFA',
    'GeneSetResult_GSVD',
    'color',
    
    # Submodules
    #'datasets',
    #'genesets',
    'plotting',
    'utils',
    'sparse_methods',
    
    # Functions
    'integrate_projection_results',
    'MIIC_input_generator',
    #'pbmc3k',
    #'pbmc_ifnb',
    #'use_hallmarks',
    #'use_reactome',
    #'use_progeny',
]