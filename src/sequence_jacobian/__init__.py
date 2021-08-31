"""Public-facing objects."""

from . import estimation, jacobian, utilities, devtools

from .blocks.simple_block import simple
from .blocks.het_block import het, hetoutput
from .blocks.solved_block import solved
from .blocks.combined_block import combine, create_model
from .blocks.support.simple_displacement import apply_function
from .steady_state.classes import SteadyStateDict
from .jacobian.classes import JacobianDict
from .blocks.support.impulse import ImpulseDict

from .visualization.draw_dag import draw_dag, draw_solved, inspect_solved

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
