"""Public-facing objects."""

from . import estimation, utilities

from .blocks.simple_block import simple
from .blocks.het_block import het
from .blocks.solved_block import solved
from .blocks.combined_block import combine, create_model
from .blocks.support.simple_displacement import apply_function
from .classes.steady_state_dict import SteadyStateDict
from .classes.impulse_dict import ImpulseDict
from .classes.jacobian_dict import JacobianDict

# Useful utilities for setting up HetBlocks
from .utilities.discretize import agrid, markov_rouwenhorst, markov_tauchen
from .utilities.interpolate import interpolate_y
from .utilities.optimized_routines import setmin

# Ensure warning uniformity across package
import warnings

# Force warnings.warn() to omit the source code line in the message
formatwarning_orig = warnings.formatwarning
warnings.formatwarning = lambda message, category, filename, lineno, line=None: \
    formatwarning_orig(message, category, filename, lineno, line='')
